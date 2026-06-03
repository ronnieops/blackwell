# Scout Report — INT8 vs INT4 Decode Pipeline

## Files Retrieved

1. `bench/text_generate.cu` (lines 1-298) — Full INT8 text generation: tokenizer, 28L decode, GPU sampling
2. `bench/decode_int4_batched_attn.cu` (lines 1-254) — INT4 benchmark with batched GEMV + batched attention, 1.7B
3. `bench/decode_int4_batched_attn_qwen3_8b.cu` (lines 1-260) — INT4 benchmark, 8B variant (H=4096, I=12288)
4. `include/blackwell/kernels.h` (lines 1-650+) — All kernel function signatures
5. `CMakeLists.txt` (lines 1-150) — Build system: library target, no bench targets in CMake
6. `src/kernels/fused_residual_norm_int4.cu` (lines 1-190) — Fused kernel, kFusedThreads=512 fix
7. `include/blackwell/bpe_tokenizer.h` (lines 1-250) — BPE tokenizer API

---

## Weight File Format

All weight files use same binary header format:

```c
int h[5];  // 20 bytes header
// .int8_t / .int4_t / .packed_fp4 file:
h[0] = K    // input features
h[1] = N    // output features
h[2] = 16   // block size (always 16)
h[3] = K/16 // num K-blocks
h[4] = N    // num N-blocks (per-row layout)
// Followed by packed data bytes

// .scale_t file: same header structure
// Followed by FP32 scales [h[3]*h[4]] floats
```

**INT4 specifics** (from `bench/decode_int4_batched_attn.cu:34-48`):
- `.int4_t`: packed nibbles, 2 vals/byte → `ds = K * N / 2`
- `.scale_t`: scales `[h[3]*h[4]]` = `[N × K/16]` (per-row per-block scales)
- `upload_w4()` allocates GPU memory via `cudaMalloc` + `cudaMemcpy`

**INT8 specifics** (from `bench/text_generate.cu:85-100`):
- `.int8_t`: one byte per element → `ds = K * N`
- `.scale_t`: scales `[h[3]*h[4]]` = `[N × K/16]` (per-row per-block)
- `dw()` helper: host load → `cudaMalloc` + `cudaMemcpy`

---

## INT8 Pipeline (text_generate.cu) — Per-Layer Kernel Call Sequence

End-to-end text gen. Tokenizer + embedding + 28-layer loop + lm_head + GPU sampling.

### Per-layer sequence (text_generate.cu:127-203):

| Step | Kernel | Inputs → Outputs | Lines |
|------|--------|-----------------|-------|
| 1 | `fused_rmsnorm(d_xi_f, input, d_rn_in[l], H, eps, st)` | FP32 → FP32 normed | 139-143 |
| 2 | `quantize_int8(d_ai, d_as, d_xi_f, H, st)` | FP32 → INT8 + scales | 148 |
| 3 | `gemv_int8_warp(d_Q, d_ai, d_as, W[l].q.d, W[l].q.sc, H, QD, st)` | INT8 → FP32 Q | 149-150 |
| 4 | `gemv_int8_warp(d_K, ...)` | INT8 → FP32 K | 151-152 |
| 5 | `gemv_int8_warp(d_V, ...)` | INT8 → FP32 V | 153-154 |
| 6 | `head_norm_kernel<<<nqh,128>>>(d_Q, W[l].qn, ...)` | FP32 Q → per-head normed Q | 157-158 |
| 7 | `head_norm_kernel<<<nkv,128>>>(d_K, W[l].kn, ...)` | FP32 K → per-head normed K | 159-160 |
| 8 | `apply_rope_kernel<<<nqh,hd/2>>>(d_Q, nqh, hd, step)` | RoPE on Q | 163-164 |
| 9 | `apply_rope_kernel<<<nkv,hd/2>>>(d_K, nkv, hd, step)` | RoPE on K | 165-166 |
| 10 | `update_kv_cache(d_kc+kb, d_vc+kb, d_K, d_V, 0, step, nkv, hd, MAXSEQ, st)` | Write K,V | 172 |
| 11 | `attention_decode_gqa(d_attn, d_Q, d_kc+kb, d_vc+kb, step, nqh, nkv, hd, MAXSEQ, st)` | Attn output | 173-174 |
| 12 | `quantize_int8(d_ai, d_as, d_attn, QD, st)` | FP32 attn → INT8 | 179 |
| 13 | `gemv_int8_warp(d_proj, d_ai, d_as, W[l].o.d, W[l].o.sc, QD, H, st)` | Wo projection | 180-181 |
| 14 | `vector_add_fp32(d_proj, d_proj, d_res_save, H, st)` | Residual add | 182-183 |
| 15 | `fused_rmsnorm(d_xi_f, d_proj, d_rn_post[l], H, eps, st)` | Post-attn norm | 190 |
| 16 | `quantize_int8(d_ai, d_as, d_xi_f, H, st)` | FP32 → INT8 | 195 |
| 17 | `gemv_int8_warp(d_gate, ...)` | Gate proj | 196-197 |
| 18 | `gemv_int8_warp(d_up, ...)` | Up proj | 198-199 |
| 19 | `apply_swiglu(d_mlp, d_gate, d_up, ID, st)` | SiLU(gate)*up | 200-201 |
| 20 | `quantize_int8(d_mi, d_ms, d_mlp, ID, st)` | FP32 → INT8 | 206 |
| 21 | `gemv_int8_warp(d_proj, d_mi, d_ms, W[l].d.d, W[l].d.sc, ID, H, st)` | Down proj | 207-208 |
| 22 | `vector_add_fp32(d_proj, d_proj, d_res_save2, H, st)` | MLP residual | 209-210 |

**Total: 22 kernel launches per layer** (INT8, unfused).

### After layer loop (text_generate.cu:215-230):
- `fused_rmsnorm(d_xi_f, d_proj, d_fn, H, eps, st)` — final norm
- `quantize_int8(d_ai, d_as, d_xi_f, H, st)` → `gemv_int8_warp(d_logits, ..., d_emb_d, d_emb_sc, H, V, st)` — lm_head
- `sample_gpu(d_logits, V, temperature, top_k, d_next_id, seed, step, st)` — GPU sampling
- `cudaMemcpy(&next_id, d_next_id, 4, cudaMemcpyDeviceToHost)` — transfer 1 int

### Tokenizer (text_generate.cu:66-73):
```cpp
blackwell::BpeTokenizer tokenizer;
tokenizer.load("tokenizer_data.bin");
auto input_ids = tokenizer.encode(prompt);  // string → vector<uint32_t>
tokenizer.decode(next_id);                   // token_id → string
```
Chat mode wraps prompt in `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n` template.

### Embedding lookup (text_generate.cu:116-120):
Host-side dequant: `h_embed[d] = (float)emb.d[tid*H+d] * emb.sc[tid*(H/16)+d/16]`
Then `cudaMemcpy` H2D.

### Weight loading (text_generate.cu:102-114):
```
weights_int8_bf16/{L}_self_attn.q_proj.int8_t + .scale_t  (7 layers per progress line)
weights_int8_bf16/{L}_input_layernorm.f32 + _post_attention_layernorm.f32 (RMSNorm weights)
weights_int8_bf16/qk_norms.f32  (28×2×128 = 7168 floats, per-head Q/K norms)
weights_int8_bf16/final_norm.f32 (H floats)
weights_int8_bf16/embed_tokens.int8_t + .scale_t (V×H matrix)
```

---

## INT4 Pipeline (decode_int4_batched_attn.cu) — Per-Layer Kernel Call Sequence

Benchmark only (no tokenizer/sampling). M sequences in parallel.

### Per-layer sequence for M sequences (benchmark loop, lines 145-212):

| Step | Kernel | Inputs → Outputs | Batched? |
|------|--------|-----------------|----------|
| 1 | `quantize_int4(d_x_i4_b+m, d_x_i4_sc_b+m, d_x32+m, H, 0)` | FP32 → INT4 input | Per-seq × M |
| 2 | `gemv_int4_batched(d_Q_b, d_x_i4_b, d_x_i4_sc_b, lw[l].q.d, lw[l].q.sc, H, Q, M, 0)` | Q projection | ✅ Batched |
| 3 | `gemv_int4_batched(d_K_b, ...)` | K projection | ✅ Batched |
| 4 | `gemv_int4_batched(d_V_b, ...)` | V projection | ✅ Batched |
| 5 | `update_kv_cache(kc_seq+off, vc_seq+off, d_K_b+m*KV, d_V_b+m*KV, 0, sq+1, nkv, hd, ms, 0)` | KV write | Per-seq × M |
| 6 | `attention_decode_batched_gqa(d_attn_b, d_Q_b, d_kc, d_vc, sq+1, nqh, nkv, hd, ms, M, kv_seq_stride, kv_layer_off, 0)` | Attn output | ✅ Batched (M) |
| 7 | `quantize_int4(d_attn_i4_b+m, d_attn_i4_sc_b+m, d_attn_b+m*Q, Q, 0)` | Attn → INT4 | Per-seq × M |
| 8 | `gemv_int4_batched(d_proj_b, d_attn_i4_b, d_attn_i4_sc_b, lw[l].o.d, lw[l].o.sc, Q, H, M, 0)` | Wo projection | ✅ Batched |
| 9 | `fused_residual_norm_int4(d_x_i4_b+m, d_x_i4_sc_b+m, d_proj_b+m*H, d_x32+m, d_rn, H, 1e-6f, 0)` | Residual+norm+INT4 quant | Per-seq × M |
| 10 | `gemv_int4_batched(d_gate_b, d_x_i4_b, d_x_i4_sc_b, lw[l].g.d, lw[l].g.sc, H, I, M, 0)` | Gate proj | ✅ Batched |
| 11 | `gemv_int4_batched(d_up_b, ...)` | Up proj | ✅ Batched |
| 12 | `fused_swiglu_quant_int4(d_mlp_i4_b+m, d_mlp_i4_sc_b+m, d_gate_b+m*I, d_up_b+m*I, I, 0)` | SwiGLU + INT4 quant | Per-seq × M |
| 13 | `gemv_int4_batched(d_proj_b, d_mlp_i4_b, d_mlp_i4_sc_b, lw[l].d.d, lw[l].d.sc, I, H, M, 0)` | Down proj | ✅ Batched |
| 14 | `fused_residual_norm_int4_fp32out(d_x_i4_b+m, d_x_i4_sc_b+m, d_x32+m, d_proj_b+m*H, d_x32+m, d_rn, H, 1e-6f, 0)` | Residual+norm+INT4+FP32 | Per-seq × M |

**Total: 14 kernel launches per layer** (vs 22 INT8). Fused kernels eliminate head_norm, RoPE, vector_add, separate quantize calls.

### Key differences from INT8:
1. **No head_norm or RoPE** — these are handled inside `attention_decode_batched_gqa`
2. **No separate SwiGLU** — fused with quantize (`fused_swiglu_quant_int4`)
3. **No separate residual add** — fused into `fused_residual_norm_int4` and `fused_residual_norm_int4_fp32out`
4. **`gemv_int4_batched`** replaces `gemv_int8_warp` — shares weight loads across M tokens
5. **`d_rn` is dummy** (all 1.0) — RMSNorm weight not loaded; identity norm used

### Buffer Layout (decode_int4_batched_attn.cu:67-87):

| Buffer | Size | Purpose |
|--------|------|---------|
| `d_x32` | `M × H × 4` | FP32 hidden state (per-seq) |
| `d_x_i4_b` | `M × H/2` | INT4 packed input (per-seq) |
| `d_x_i4_sc_b` | `M × H/16 × 4` | Input scales (per-seq) |
| `d_Q_b` | `M × Q × 4` | Q projection output |
| `d_K_b` | `M × KV × 4` | K projection output |
| `d_V_b` | `M × KV × 4` | V projection output |
| `d_attn_b` | `M × Q × 4` | Attention output |
| `d_proj_b` | `M × H × 4` | Shared temp for Wo/down output |
| `d_attn_i4_b` | `M × Q/2` | INT4 packed attention |
| `d_attn_i4_sc_b` | `M × Q/16 × 4` | Attention scales |
| `d_gate_b` | `M × I × 4` | Gate projection |
| `d_up_b` | `M × I × 4` | Up projection |
| `d_mlp_i4_b` | `M × I/2` | INT4 packed SwiGLU output |
| `d_mlp_i4_sc_b` | `M × I/16 × 4` | MLP scales |
| `d_mlp_sc_b` | `(H/16) × 4` | Unused (shared, single) |
| `d_rn` | `H × 4` | RMSNorm weight (all 1.0) |
| `d_kc` | `M × num_layers × nkv × ms × hd × 4` | K cache |
| `d_vc` | `M × num_layers × nkv × ms × hd × 4` | V cache |

### Weight loading (lines 48-56):
```
weights_int4_qwen3_1.7b/{L}_self_attn.q_proj.int4_t + .scale_t
weights_int4_qwen3_1.7b/{L}_self_attn.k_proj.int4_t + .scale_t
weights_int4_qwen3_1.7b/{L}_self_attn.v_proj.int4_t + .scale_t
weights_int4_qwen3_1.7b/{L}_self_attn.o_proj.int4_t + .scale_t
weights_int4_qwen3_1.7b/{L}_mlp.gate_proj.int4_t + .scale_t
weights_int4_qwen3_1.7b/{L}_mlp.up_proj.int4_t + .scale_t
weights_int4_qwen3_1.7b/{L}_mlp.down_proj.int4_t + .scale_t
```

---

## 8B Variant (decode_int4_batched_attn_qwen3_8b.cu)

Identical structure. Dimensions change:
```
H=4096, Q=4096, KV=1024, I=12288, nqh=32, nkv=8, hd=128
```
Weight path: `weights_int4_qwen3_8b/`

---

## CMake Build Targets

**No CMake bench targets.** CMakeLists.txt only builds:
- `add_library(blackwell_kernels STATIC ${KERNEL_SOURCES})` — library (177 symbols)
- `add_executable(blackwell_tests ...)` — GTest unit tests (optional)
- `pybind11_add_module(blackwell_pybind ...)` — Python bindings (optional)

All benches built via manual `nvcc` invocation (see bench file headers):
```bash
# INT4 batched attn
/usr/local/cuda-13.3/bin/nvcc -O3 -std=c++17 \
  -gencode=arch=compute_120a,code=sm_120a \
  -I include bench/decode_int4_batched_attn.cu build/libblackwell_kernels.a \
  -o bench/decode_int4_batched_attn

# text_generate
nvcc -O3 -std=c++17 \
  -gencode=arch=compute_120a,code=sm_120a \
  -I include bench/text_generate.cu build/libblackwell_kernels.a \
  -o bench/text_generate
```

---

## Fused Residual Norm INT4 (fused_residual_norm_int4.cu)

### kernel parameters:
- `kFusedThreads = 512` (was 256 — fix for H=4096)
- `kFusedREPT = 8` (elements per thread: 512×8=4096)
- `kBlockSize = 16` (quantization block)

### kernel phases:
1. **Phase 1**: Load `proj[idx] + residual[idx]`, store sum back, compute `sum_sq`
2. **Warp-reduce** `sum_sq` → smem warp_sums[16] → block_sum → `rsqrtf(block_sum/N + eps)`
3. **Phase 2**: Normalize `vals[r] * weight[idx] * rstd`, track per-block `absmax`
4. **Write scales**: lane 0 of each block-16 writes `absmax/7` to `x_sc`
5. **Phase 3**: Quantize → `roundf(v/sc)`, clamp [-8,7], pack nibble into byte

### Two API variants:

```cpp
// Mutates proj in-place (proj = proj + residual, then norm)
cudaError_t fused_residual_norm_int4(
    void* x_out, float* x_out_sc, float* proj,
    const float* residual, const float* norm_w,
    int N, float eps, cudaStream_t stream);

// Read-only proj_in, writes FP32 normalized to proj_out_fp32
cudaError_t fused_residual_norm_int4_fp32out(
    void* x_out, float* x_out_sc, float* proj_out_fp32,
    const float* proj_in, const float* residual,
    const float* norm_w, int N, float eps, cudaStream_t stream);
```

---

## Tokenizer API (bpe_tokenizer.h)

```cpp
blackwell::BpeTokenizer tokenizer;
int load(const char* path);  // binary file from scripts/prepare_tokenizer.py
std::vector<uint32_t> encode(const std::string& text);  // BPE encode
std::string decode(uint32_t token_id);  // single token → UTF-8
std::string decode(const std::vector<uint32_t>& ids);  // multi-token → UTF-8
```
Binary format: num_vocab | num_merges | num_added | 256×byte_enc(codepoint) | vocab_entries(id,len,str) | added_tokens(id,len,is_special,str) | merges(left_len,left_str,right_len,right_str)

---

## Architecture Summary

```
text_generate.cu (INT8)                         decode_int4_batched_attn.cu (INT4)
────────────────────────                        ────────────────────────────────
Tokenizer (host)                                ─ (no tokenizer)
│                                               │
Embedding lookup (host H2D)                     Init d_x32 to all-1.0
│                                               │
┌── 28-layer loop ──────────────┐               ┌── 28-layer loop ──────────────┐
│ fused_rmsnorm                 │               │ quantize_int4 (per-seq)       │
│ quantize_int8                 │               │ gemv_int4_batched (M) Q,K,V   │
│ gemv_int8_warp Q,K,V          │               │ update_kv_cache (per-seq)     │
│ head_norm (handwritten)       │               │ attention_decode_batched_gqa  │
│ apply_rope (handwritten)      │               │ quantize_int4 (per-seq) attn  │
│ update_kv_cache               │               │ gemv_int4_batched (M) Wo      │
│ attention_decode_gqa          │               │ fused_residual_norm_int4      │
│ quantize_int8 → gemv_warp Wo │               │ gemv_int4_batched (M) gate,up │
│ vector_add (residual)         │               │ fused_swiglu_quant_int4       │
│ fused_rmsnorm                 │               │ gemv_int4_batched (M) down    │
│ quantize_int8 → gemv_warp g,u │               │ fused_residual_norm_int4_fp32 │
│ apply_swiglu                  │               └──────────────────────────────┘
│ quantize_int8 → gemv_warp dn  │               │
│ vector_add (residual)         │               ─ (no lm_head, no sampling)
└──────────────────────────────┘               │
│                                               │
fused_rmsnorm (final)                           ─ (benchmark only)
quantize_int8 → gemv_warp lm_head
sample_gpu → cudaMemcpy 1 int
tokenizer.decode → print
```

---

## Start Here

`bench/decode_int4_batched_attn.cu` — reference INT4 pipeline. Shows exact kernel call sequence, buffer sizes, weight format, and batched pattern. Copy+modify for new models.

---

## Constraints & Notes

1. **No CMake bench targets** — all bench files compiled manually. Each bench is standalone CUDA source + link against `libblackwell_kernels.a`
2. **`d_rn` is dummy** (all 1.0) — RMSNorm weight never loaded in INT4 benches. Kernel uses identity weight. Real model would need per-layer norms loaded
3. **Scale init**: INT4 benches initialize all scale buffers to `1/7` before use (fallback if quantize never runs)
4. **KV cache fill**: warmup loop fills 128 positions per-seq before benchmark
5. **Per-seq vs batched**: `quantize_int4`, `update_kv_cache`, `fused_residual_norm_int4`, `fused_swiglu_quant_int4` run per-seq (M times per layer). `gemv_int4_batched` and `attention_decode_batched_gqa` run once (M tokens)
6. **No lm_head in INT4 benches** — benchmark stops at hidden state. For text generation, need lm_head GEMV + GPU sampling
7. **No final norm in INT4 benches** — benchmark skips final RMSNorm. Real pipeline needs it before lm_head