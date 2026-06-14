# Code Context: CUDA Kernel Analysis

## Files Retrieved
1. `include/blackwell/kernels.h` (lines 1-1433) — Full public API, 189+ kernel declarations
2. `src/kernels/gemv_int8.cu` (lines 1-1790) — Production INT8/INT4/FP4 GEMV kernels
3. `src/kernels/fused_int4_ops.cu` (lines 1-391) — Fused RMSNorm+INT4 quant, SwiGLU+INT4 quant
4. `src/kernels/decode.cu` (lines 1-898) — Attention decode, KV cache, softmax
5. `src/kernels/norm.cu` (lines 1-219) — RMSNorm, SwiGLU, vector_add
6. `src/kernels/rope.cu` (lines 1-242) — RoPE kernels (in-place, decode, out-of-place)
7. `src/kernels/fused_residual_norm.cu` (lines 1-128) — Residual+RMSNorm+INT8 quant fused
8. `src/kernels/gemv_fp8.cu` (lines 1-301) — FP8 E4M3 GEMV (abandoned path)
9. `src/kernels/gemm_int8_wmma.cu` (lines 1-186) — WMMA INT8 tensor core GEMM
10. `src/kernels/gemm_int8_wmma_fast.cu` (lines 1-135) — Optimized WMMA GEMM
11. `src/kernels/gemm_int8_mma.cu` (lines 1-28) — MMA stub (returns NotSupported)
12. `src/kernels/sample_gpu.cu` (lines 1-333) — GPU sampling (argmax, temperature, top-k)
13. `src/kernels/fused_quant_attn_wo.cu` (lines 1-161) — Fused quant+Wo GEMV (unused)
14. `include/blackwell/config.h` — Hardware constants (SM_120, 36 SMs, 128KB smem)
15. `server/inference_server_int4.cu` (lines 255-297) — INT4 decode hot path per layer
16. `server/inference_server_int4_batched.cu` (lines 420-530) — Batched INT4 decode hot path

---

# Kernel Enhancement Opportunities (Ranked by Impact)

## TIER 1: HIGH IMPACT — Throughput

### 1. INT4 GEMV: Scalar FP32 unpack → __dp4a INT8 packed dot product
**File:** `src/kernels/gemv_int8.cu:998-1067` (`gemv_int4_batched_kernel`)
**Problem:** INT4 batched kernel uses `int4_byte_to_floats()` → 16 scalar FP32 FMAs per K-block. INT8 warp kernel uses `__dp4a` (4× faster per 4 elements). INT4 could pack 2 nibbles → 1 int8, use __dp4a on 8-byte pairs.
**Expected:** 2-3× compute speedup on INT4 GEMV inner loop. GEMV is 92% of decode time.
**Current:**
```c
// lines 1037-1043 — scalar FP32 multiply per element pair
for (int j = 0; j < PB; ++j) {
    int4_byte_to_floats(wb[j], w0, w1);
    int4_byte_to_floats(xb[j], x0, x1);
    sum_f += w0 * x0 + w1 * x1;  // FP32 FMAs
}
```
**Fix:** Sign-extend nibbles to int8, pack pairs, use `__dp4a(w_int, x_int, sumi)`. Two int8 pairs per __dp4a call. 4 __dp4a calls process 16 INT4 values (8 bytes) vs 8 scalar FP32 FMAs.
**Risk:** INT4 offset-binary (nib-8) ≠ int8 signed range. Need nib→int8 sign extension: `(int8_t)((nib << 4) | (nib & 8 ? 0xF0 : 0))` or similar. Must verify numerical equivalence.
**Same issue in:** `gemv_int4_warp_kernel` (line 482-542), `gemv_fp32_int4_warp_kernel` (line 547-601)

### 2. Fused RMSNorm + INT4 quant NOT used in server (saves 7-9 launches/token)
**File:** `src/kernels/fused_int4_ops.cu:96-167` (`fused_rmsnorm_quant_int4_v2_kernel`)
**Problem:** Server calls `fused_rmsnorm` then `quantize_int4_batched` separately, 2 launches each × 3 calls/layer (input norm, post-attn norm, final norm) = 6 unnecessary launches/layer → 216/36 layers.
**Server hot path** (`server/inference_server_int4.cu:274-275`):
```c
fused_rmsnorm(d_xi_f, d_x32, W[l].rn_in, H, eps, st);       // launch 1
quantize_int4_batched(d_x_i4, d_x_i4_sc, d_xi_f, H, 1, st); // launch 2
// Could be: fused_rmsnorm_quant_int4(d_x_i4, d_x_i4_sc, d_x32, W[l].rn_in, H, eps, st);
```
**Same pattern at:** lines 291-292, 273 (post-attn norm + quant), 283-284 (final norm + quant)
**Expected:** ~6 fewer launches/layer × 36 layers = 216 fewer launches/token. Also eliminates `d_xi_f` intermediate buffer (H×4 = 16KB write+read).
**Also unused:** `fused_swiglu_quant_int4` (line 233-291) — server does `apply_swiglu` then `quantize_int4_batched` at lines 276-277.

### 3. QKV GEMV: 3 separate launches → 1 fused launch
**File:** `src/kernels/gemv_int8.cu:998-1067` — current 3 separate `gemv_int4_batched` calls
**Server** (`inference_server_int4.cu:276-278`):
```c
gemv_int4_batched(d_Q, ..., H, Q, 1, st);  // launch
gemv_int4_batched(d_K, ..., H, KV, 1, st); // launch
gemv_int4_batched(d_V, ..., H, KV, 1, st); // launch
```
**Existing unused kernel:** `fused_qkv_int4` (kernels.h, declared ~line 1100). Weight loaded once per K-block, reused for Q/K/V. Saves 2 launches + 2× activation reads/layer.
**Expected:** 2 fewer launches + 2× less activation read bandwidth per layer × 36 = 72 fewer launches/token.

### 4. Embedding H2D copy per token → GPU-side embedding lookup
**File:** `server/inference_server_int4.cu:268-270`:
```c
std::vector<float> h_embed(H);
dequant_embed_row(h_embed.data(), token_id, host_embed_d, host_embed_sc, H);
die(cudaMemcpyAsync(d_x32, h_embed.data(), H*4, cudaMemcpyHostToDevice, st), "embed");
```
**Problem:** CPU dequantizes INT4 embedding → FP32 on host, then H2D copies 16KB every token. This is a blocking sync point (vector alloc + dequant + H2D).
**Fix:** Pre-load INT4 embedding table to GPU (like server_int4_batched already does at line 213). Use a simple kernel: `d_x32[d] = (float)(embed_i4[token_id*K/2 + d/2] nibble - 8) * embed_sc[token_id*K/16 + d/16]`.
**Note:** AGENTS.md says "Embedding Pre-load Optimization (Session 71)" already done for server_int4_batched. But `inference_server_int4.cu` (non-batched) still uses CPU path.

### 5. Gate+Up GEMV: 2 separate launches → 1 fused launch
**Server** (`inference_server_int4.cu:274-275`):
```c
gemv_int4_batched(d_gate, ..., H, I, 1, st);
gemv_int4_batched(d_up, ..., H, I, 1, st);
```
**Existing unused kernels:** `fused_qkv_int4` (INT4 version declared in header). No `fused_gate_up_int4` exists yet, but pattern is clear. Would save 1 launch + 1× activation read/layer.

---

## TIER 2: MEDIUM IMPACT — Throughput / Quality

### 6. Attention softmax is single-thread serial (O(seq_len²) per thread)
**File:** `src/kernels/decode.cu:148-162`:
```c
// Serial loop over ALL positions by EACH thread for softmax
for (int t = 0; t < npos; ++t)
    if (scores[t] > maxv) maxv = scores[t];
// ...
for (int t = 0; t < npos; ++t) {
    float e = __expf(scores[t] - maxv);
    scores[t] = e;
    sumexp += e;
}
```
**Problem:** Each of 128 threads serially scans all `npos` scores. At seq_len=512, that's 512 serial iterations × 3 (max, exp, V-weighted-sum). Should use warp-cooperative reduction.
**Expected:** At seq_len=512, attention is 0.9% of time but this is O(n²) — will grow linearly with context.
**Fix:** Use warp shuffle reduction for max/sum, parallel exp.

### 7. RoPE + head_norm: 4 separate launches → 1 fused kernel
**Server** (`inference_server_int4.cu:261-264`):
```c
head_norm_kernel<<<nqh,128>>> (d_Q, ...);  // launch
head_norm_kernel<<<nkv,128>>> (d_K, ...);  // launch
apply_rope_kernel<<<nqh,hd/2>>> (d_Q, ...); // launch
apply_rope_kernel<<<nkv,hd/2>>> (d_K, ...); // launch
```
**Fix:** Fuse head_norm + RoPE into single kernel per Q/K (2 launches instead of 4). Both are element-wise over [heads × head_dim] with same grid.
**Expected:** 2 fewer launches/layer × 36 = 72 fewer launches/token.

### 8. INT4 RMSNorm quant kernel: H=4096 needs multi-block but smem layout assumed
**File:** `src/kernels/fused_int4_ops.cu:149-167`
**Problem:** `fused_rmsnorm_quant_int4_v2_kernel` computes RMSNorm per-block, but when N > 4096 (THREADS×EPT), multiple blocks are launched. Each block only sees its own elements → **incorrect RMSNorm** (normalizes over subset, not full vector).
**Check:** For H=4096, THREADS×EPT=256×16=4096, so exactly 1 block. For I=12288 (MLP dim), 3 blocks → **BUG**. But this kernel is unused in server, so no runtime impact.
**Risk:** If anyone enables this kernel for MLP dims, quality breaks silently.

### 9. INT4 batched GEMV grid: N blocks for N=151936 (lm_head) → wave quantization
**File:** `src/kernels/gemv_int8.cu:1071-1096`
**Problem:** Grid is `dim3(N, 1)` with 32 threads/block (1 warp). For lm_head N=151936 → 151936 blocks on 36 SMs. Wave efficiency depends on K/32 blocks per thread. Each block computes only 1 output row.
**Potential:** Could use multi-row blocks (e.g., 4 rows/block) to amortize block launch overhead. But GEMV is memory-bound at M=1, so launch overhead may be negligible vs memory latency.

### 10. Attention decode stores scores in shared memory — O(seq_len) smem
**File:** `src/kernels/decode.cu:130`
**Problem:** `scores` array in shared memory = `npos` floats. At seq_len=512, that's 2KB/head. For 32 heads × 2KB = 64KB if all heads in 1 block (they're not — 1 block/head). Current smem usage is fine for head-per-block.
**Future risk:** At seq_len=4096, 16KB/head still fits in 99KB smem. Not a blocker.

---

## TIER 3: MAINTAINABILITY / CODE QUALITY

### 11. 22+ dead kernel functions exported but never called
**Confirmed dead** (zero references in server/bench):
- `gemv_int8_pdl`, `gemv_int8_fp16sc`, `gemv_int8_unrolled`, `gemv_int8_warp_unrolled`, `gemv_int8_fp16cached`
- `gemv_fp8_fp32act`, `gemv_fp8_int8act`, `quantize_fp8_row` (FP8 path abandoned per AGENTS.md)
- `gemv_int4_asym_batched`, `gemv_fp32_int4_asym`, `gemv_fp32_int5_asym`, `quantize_int4_asym`, `fused_swiglu_quant_int4_asym`, `fused_residual_norm_int4_asym` (asymmetric paths dead)
- `fused_qkv_int4`, `fused_qkv_gemv`, `fused_gate_up_gemv`, `gemv_int8_gate_up` (fused QKV/gate-up unused)
- `gemv_bf16`, `clear_fp16_scale_caches`, `convert_scales_fp32_to_fp16`
- `fused_quant_attn_wo`, `fused_o_norm_pack`, `fused_pack_gemv_o`, `fused_swiglu_gemv`, `persistent_qkv_gemv`
- `fused_rmsnorm_quant_int4`, `fused_residual_norm_int4`, `fused_residual_norm_int4_fp32out`, `fused_swiglu_quant_int4`
- `unpack_int4_fp32`, `transpose_int4_weights`, `fused_decode`
- `gemm_int8_mma` (stub: `return cudaErrorNotSupported`)

**Action:** Don't delete (reference value), but add `__attribute__((unused))` or move to separate archive section.

### 12. fused_rmsnorm_quant_int4 v1 kernel has incomplete code (TODO comments)
**File:** `src/kernels/fused_int4_ops.cu:40-135`
**Problem:** First kernel `fused_rmsnorm_quant_int4_kernel` has dead branches:
```c
// lines 116-119:
if (lane_in_blk == 0) {
    // This thread handles the low nib of byte[byte_idx]
    // But we need all 16 lanes to cooperate for the byte
    // Alternative: each thread writes its own nib to a shared byte
}
```
Function body ends without writing output. Only `v2_kernel` is actually launched.
**Action:** Delete v1 kernel (dead code, confusing).

### 13. FP8 conversion uses powf() — extremely slow
**File:** `src/kernels/gemv_fp8.cu:21-31`
```c
value = (1.0f + (float)mant / 8.0f) * powf(2.0f, (float)(exp - 7));
```
**Problem:** `powf` is ~20 cycles. FP8→FP32 should use bit manipulation (~2 cycles). Should use `__nv_cvt_fp8_to_fp32` intrinsic or manual bit-shift.
**Impact:** Low — FP8 path abandoned. But if anyone revives it, performance will be terrible.

### 14. Error checking inconsistency: head_norm/apply_rope use bare kernel launches
**File:** `server/inference_server_int4.cu:261-264`
```c
head_norm_kernel<<<nqh,128,0,st>>>(d_Q, ...);  // no error check
apply_rope_kernel<<<nqh,hd/2,0,st>>>(d_Q, ...); // no error check
```
All `blackwell::kernels::` calls wrapped in `die()`. Raw `<<<>>>` launches are not. `cudaGetLastError()` not checked after kernel launch.

### 15. FP4 scale format mismatch in int4_byte_to_floats
**File:** `src/kernels/gemv_int8.cu:472-478` vs `src/kernels/fused_quant_attn_wo.cu:21-25`
```c
// gemv_int8.cu:472 — offset-binary (nib - 8)
int lo = (b & 0x0F) - 8;
int hi = ((b >> 4) & 0x0F) - 8;

// fused_quant_attn_wo.cu:21 — two's complement (nib > 7 ? nib - 16 : nib)
int lo = b & 0x0F; if (lo > 7) lo -= 16;
int hi = (b >> 4) & 0x0F; if (hi > 7) hi -= 16;
```
**Problem:** These are numerically equivalent for [-8..7] range BUT differ for edge case nib=0: offset-binary gives -8, two's complement gives 0. Inconsistent implementations across files.
**Note:** AGENTS.md Bug History says offset-binary (nib-8) is correct. The fused_quant_attn_wo version is wrong but it's dead code.

### 16. WMMA INT8 GEMM: K-loop re-fills c_frag each iteration
**File:** `src/kernels/gemm_int8_wmma_fast.cu:78-82`
```c
wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
wmma::fill_fragment(c_frag, 0);  // RESET every K block!
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```
**Problem:** Should accumulate across K blocks in `c_frag`, not reset. Current code computes A[wm,kb]×B[wn,kb] for each kb independently, then adds scaled result to `acc[]`. Correct but misses opportunity to use hardware accumulator (S32) across K, falling back to FP32 `acc[]` adds.
**Impact:** Medium for prefill. Decode doesn't use GEMM (M=1 → GEMV).

### 17. WMMA GEMM only 1 K-iteration per tile (no K-loop pipelining)
**File:** `src/kernels/gemm_int8_wmma_fast.cu:70-95`
**Problem:** Each K-block iteration does: load scales → sync → load fragments → mma_sync → dequant to acc. No double buffering, no K-tile pipelining. Could use `__pipeline_memcpy_async` or manual double-buffer to overlap loads with compute.
**Impact:** Prefill-only, medium.

---

## TIER 4: NUMERICAL PRECISION

### 18. INT4 quantization uses absmax/7 (symmetric, 4-bit)
**File:** `src/kernels/fused_int4_ops.cu:139` — `scale = fmaxf(abs_val / 7.0f, 1e-9f)`
**Problem:** INT4 range is [-8..7] but scale uses /7 (not /8). This wastes the -8 codepoint for symmetric quant. Should use /8 for full range, or /7 with clamp to [-7,7] (which it does).
**Quality impact:** Negligible — -8 is rarely the max value, and clamping to [-7,7] is standard.

### 19. All intermediate buffers are FP32 (could use FP16/BF16)
**Problem:** Hidden states, attention output, Q/K/V all stored as FP32 in decode loop. Each H=4096 → 16KB per buffer. Multiple buffers per layer.
**Potential:** Use FP16 or BF16 for residual stream → 2× less memory bandwidth for element-wise ops (vector_add, RMSNorm). GEMV output would still be FP32 accumulate, but cast to FP16 on write.
**Risk:** FP16 has limited dynamic range (±65504). BF16 better for activations but not native on all paths.
**Note:** GEMV is memory-bound on weight reads, not activation reads. Activation FP16 would only help element-wise ops (3.7% of time). Low ROI.

### 20. Attention scores stored as FP32 in shared memory
**File:** `src/kernels/decode.cu:136` — `float* scores = smem + head_dim`
**Problem:** Could use FP16 for attention scores (2× less smem, faster exp). Standard in flash attention.
**Impact:** Low at current context lengths. Attention is 0.9% of decode time.

---

## Architecture Summary

```
INT4 Server Decode (per layer, per token):
┌─────────────────────────────────────────────────────────┐
│ 1. Embedding H2D (CPU dequant + 16KB copy)              │ ← Remove (GPU lookup)
│ 2. cudaMemcpyAsync (D2D save residual)                  │ ← Fuse into norm
│ 3. fused_rmsnorm + quantize_int4 (2 launches)           │ ← Fuse: fused_rmsnorm_quant_int4
│ 4. 3× gemv_int4_batched (Q, K, V)                       │ ← Fuse: fused_qkv_int4
│ 5. 2× head_norm_kernel (Q, K)                           │ ← Fuse with RoPE
│ 6. 2× apply_rope_kernel (Q, K)                           │ ← Fuse with head_norm
│ 7. update_kv_cache                                      │
│ 8. attention_decode (serial softmax)                    │ ← Parallel softmax
│ 9. quantize_int4 + gemv_int4_batched (O proj)           │ ← Fuse: fused_quant_attn_wo
│ 10. vector_add (attn residual)                          │ ← Fuse: fused_residual_norm_int4
│ 11. cudaMemcpyAsync (D2D save residual)                 │ ← Eliminate (fuse)
│ 12. fused_rmsnorm + quantize_int4 (2 launches)          │ ← Fuse: fused_rmsnorm_quant_int4
│ 13. 2× gemv_int4_batched (gate, up)                     │ ← Fuse: gate+up in 1 launch
│ 14. apply_swiglu + quantize_int4 (2 launches)           │ ← Fuse: fused_swiglu_quant_int4
│ 15. gemv_int4_batched (down)                            │
│ 16. vector_add (mlp residual)                           │
└─────────────────────────────────────────────────────────┘
Current: ~24 launches/layer
Optimized: ~12-14 launches/layer (using existing fused kernels)
```

## Start Here

**`src/kernels/gemv_int8.cu:998-1067`** — `gemv_int4_batched_kernel`. This is the #1 bottleneck (92% of decode time). Convert scalar FP32 unpack to `__dp4a` INT8 packed dot product for 2-3× compute speedup.

Second: **`server/inference_server_int4.cu:255-279`** — Replace separate rmsnorm+quantize and swiglu+quantize calls with existing fused kernels (already implemented, just not wired up).
