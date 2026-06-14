# Final Handoff Plan: Close INT4 M=1 Throughput Gap vs llama.cpp

**Date**: 2026-06-14  
**Target**: 56 t/s → 84+ t/s (1.5× speedup) on RTX 5060 Ti, Qwen3-8B INT4

---

## 1. Goal & Success Metric

**Goal**: Replace scalar FP32 unpack in `gemv_int4_batched_kernel` with dp4a INT8 dot-product path to eliminate the compute-bound bottleneck at M=1.

**Success metric**:
- `bench/text_generate_int4_qwen3_8b` reports **≥84 t/s** (M=1, 8B)
- `bench/bench_ppl_int4_8b` reports **PPL ≤23.52** (baseline) / **≤21.82** (AWQ)
- `curl /v1/completions` produces coherent output matching current server
- Batched path (M=8+) still works, no regression

**Hard target justification**: 131 t/s is the memory-bound ceiling for M=1 (3.8 GB weight traffic / 500 GB/s). Current 56 t/s = 42% of ceiling. dp4a should push to 80-100% of ceiling → **100-131 t/s realistic**.

---

## 2. What llama.cpp Q4_K_M Teaches (Transferable Techniques)

**Source**: `ggml/src/ggml-cuda/mmvq.cu` (MMVQ kernel), `vecdotq.cuh` (dot primitive).

### Key technique: dp4a INT8 SIMD, NOT tensor cores

llama.cpp Q4_K_M at M=1 uses **MMVQ** (`mul_mat_vec_q`) — a pure CUDA-core dp4a kernel. Tensor cores (MMQ) only for M>8. The GEMV is memory-bound; what matters is minimizing instruction overhead per byte loaded.

### Concrete lessons for our offset-binary INT4 block-16:

| Technique | llama.cpp Q4_K | Our current kernel | Transferable? |
|-----------|----------------|---------------------|---------------|
| Dot product primitive | `__dp4a` (4 INT8 FMA/cycle) | Scalar FP32 FMA (line 1031-1033) | **YES — primary fix** |
| Activations | Pre-quantized INT8 (Q8_1) | Quantized INT4 packed | Keep INT4 input, upcast in-kernel |
| Weight unpack | 4-bit→INT8 in registers, feed dp4a | 4-bit→FP32 scalar | **Replace with 4-bit→INT8→dp4a** |
| Warp strategy | 1 warp/row, stride-32 K loop | 1 warp/row, stride-32 K loop | Already matched ✅ |
| Scale fetch | 1 per 32 elements (sub-block) | 1 per 16 elements (block-16) | Leave as-is (scale overhead is NOT the bottleneck; compute is) |
| Load width | `uint4` 16-byte loads | `uint2` 8-byte loads (correct for 16 nibbles) | Already correct ✅ |
| Reduction | `__shfl_xor_sync` 5-step | `__shfl_xor_sync` 5-step | Already matched ✅ |

**Non-transferable**: Q4_K super-block format (6-bit sub-scales, FP16 super-scale). Would require re-quantization + more complex unpack → *increases* compute. Our problem is compute, not scale bandwidth.

### What explains llama.cpp's 89% BW utilization vs our 42%:

1. **dp4a processes 16 values in 4 instructions** (lines 47-50 of INT8 kernel). Our scalar path processes 16 values in **~41 instructions** (8 iterations × `int4_byte_to_floats` + FMA).
2. **10× instruction overhead** → compute-bound → memory pipeline starved → 42% BW utilization.
3. After dp4a fix: instruction count drops ~3.4× → kernel transitions from compute-bound to memory-bound → BW utilization should approach 80-90%.

---

## 3. What the Local Codebase Implies

### Current bottleneck evidence (verified by reading source):

**File**: `src/kernels/gemv_int8.cu`

**Kernel**: `gemv_int4_batched_kernel<M>` (lines 1000-1060)  
**Helper**: `int4_byte_to_floats()` (lines 472-478)  
**Launch**: `gemv_int4_batched()` (lines 1071-1104), grid=N, block=32

**The compute-bound code** (lines 1019-1034):
```cpp
for (int j = 0; j < PB; ++j) {       // PB = 8
    float w0, w1, x0, x1;
    int4_byte_to_floats(wb[j], w0, w1);  // 2 int ops + 2 int→float
    int4_byte_to_floats(xb[j], x0, x1);  // 2 int ops + 2 int→float
    sum_f += w0 * x0 + w1 * x1;          // 2 FMA
}
```
**~41 instructions per 16 elements.** Comment at line 466: "Strategy: scalar FP32 multiply-accumulate (no dp4a)."

**The INT8 reference** (`gemv_int8_warp_kernel`, lines 194-250):
```cpp
sumi = __dp4a(w32[0], x32[0], sumi);  // 4 INT8 FMA in 1 instruction
sumi = __dp4a(w32[1], x32[1], sumi);
sumi = __dp4a(w32[2], x32[2], sumi);
sumi = __dp4a(w32[3], x32[3], sumi);
acc += static_cast<float>(sumi) * prod_scale;
```
**4 dp4a instructions per 16 elements.** Same warp structure, same scale layout (`W_scale[N][K/16]`).

### Scale layout compatibility (verified):

- INT4 weights: `W_packed[N][K/2]` + `W_scale[N][K/16]` FP32 ← block-16, offset-binary nibbles
- INT8 weights: `W_int8[N][K]` + `W_scale[N][K/16]` FP32 ← block-16, signed INT8
- **Scale arrays are identical layout** (`[N][K/16]` FP32). Only weight packing differs (2/byte vs 1/byte).

### What must change:

The inner loop of `gemv_int4_batched_kernel` and `gemv_int4_warp_kernel` must replace scalar float unpack+FMA with nibble→int8 upcast + dp4a. The structure is:

```
Current:  load uint2 (8 bytes = 16 nibbles)
          → int4_byte_to_floats × 8 (→ 16 floats)
          → 8× scalar FMA
          → accumulate float sum

New:      load uint2 (8 bytes = 16 nibbles)
          → unpack 8 bytes → 16 int8_t (shift+mask, no int→float conversion)
          → 4× __dp4a (INT32 accumulate)
          → acc += (float)sumi * prod_scale
```

### Weight format: NO CHANGE needed

INT4 offset-binary nibbles (`nib-8` = `[-8,7]`) are already valid signed INT8 values. Upcast: `(b & 0xF) - 8` for lo nibble, `((b >> 4) & 0xF) - 8` for hi nibble. Result is `int8_t` in range `[-8,7]`. dp4a multiplies INT8×INT8 and accumulates in INT32 — identical math to current scalar path, just SIMD-parallel.

### Activation format: NO CHANGE needed

`quantize_int4` produces packed INT4 + FP32 block-16 scales. The kernel upcasts activation nibbles to INT8 on-the-fly (same as weights). `quantize_int4` kernel stays unchanged.

---

## 4. Recommended Approach

### Primary: dp4a INT4 inner loop (Option A)

**Rank**: #1 (highest impact / lowest risk)

**What**: Replace `int4_byte_to_floats` + scalar FMA with nibble→int8 upcast + `__dp4a` in both `gemv_int4_warp_kernel` (lines 482-537) and `gemv_int4_batched_kernel<M>` (lines 1000-1060).

**Impact**: ~3.4× fewer instructions per 16 elements → kernel transitions from compute-bound to memory-bound → expected **100-131 t/s** (memory ceiling).

**Risk**: Low. Math is identical (INT4×INT4 values, just accumulated via dp4a INT8 instead of scalar FP32). No weight format change, no re-quantization, no activation format change.

**Justification**: The gap is 100% explained by instruction count (41 vs 4 per 16 elements). Scale overhead (block-16 vs super-block) is NOT the bottleneck — if it were, we'd see ~70 t/s format-bound, not 56 t/s compute-bound.

### Alternative: Pre-upcast weights to INT8 at load time (Option B)

**Rank**: #2 (fallback if dp4a has register pressure issues)

**What**: At `upload_w4()`, expand INT4 nibbles to INT8 bytes. Call existing `gemv_int8_warp_kernel` / `gemv_int8_batched` directly.

**Impact**: Same dp4a speedup but 2× weight memory (5.3 GB → 10.6 GB, fits 16 GB) → memory-bound ceiling drops to ~65 t/s. **Worse than Option A** because it doubles weight traffic.

**Risk**: Low (reuses proven INT8 kernel), but **defeats the purpose of INT4** — the whole point is 2× less memory traffic.

### Alternative: WMMA tensor-core GEMV (Option C)

**Rank**: #3 (highest ceiling but highest effort)

**What**: Upcast INT4→INT8, use WMMA m16n16k16 with M-padded activations. Existing WMMA kernels (`gemm_int8_wmma.cu`, `gemm_int8_wmma_fast.cu`) are designed for prefill (M≥16), not decode (M=1). For M=1, 15/16 of M dimension is wasted.

**Impact**: Potentially 1.5-2× over dp4a for large N (lm_head N=151936), but uncertain. Requires weight layout change + fragment loading complexity.

**Risk**: High. **Not recommended for initial implementation.** Revisit only if dp4a path achieves <84 t/s and profiling shows remaining compute overhead.

---

## 5. Likely Files to Change

### Primary changes:

| File | Lines | Change |
|------|-------|--------|
| `src/kernels/gemv_int8.cu` | 472-478 | Add `int4_byte_to_int8s()` helper (nibble→int8, no float conversion) |
| `src/kernels/gemv_int8.cu` | 1019-1034 | Replace scalar float loop in `gemv_int4_batched_kernel<M>` with dp4a path |
| `src/kernels/gemv_int8.cu` | 515-535 | Same change in `gemv_int4_warp_kernel` (keep both in sync) |

### No changes needed (verified):

- `include/blackwell/kernels.h` — kernel signature unchanged (`gemv_int4_batched`, `gemv_int4_warp`)
- `src/kernels/gemv_int8.cu:1751` — `quantize_int4` unchanged (produces INT4 packed, scales stay FP32)
- `server/inference_server_int4.cu` — all callers use same API (lines 276-297, 331-392, 432-486, 520)
- `bench/text_generate_int4_qwen3_8b.cu` — uses same API
- `bench/text_generate_int4_batched.cu` — uses same API
- `bench/decode_int4_cgraph_8b.cu` — uses same API
- `bench/bench_gemv_int4.cu` — microbenchmark, uses same API
- `src/kernels/decode.cu` — `_pos` prefill variants unaffected (call attention, not GEMV)

### Build config (no change):

- `CMakeLists.txt` — no new source files
- `-arch=sm_120a` already set
- `__dp4a` confirmed supported on SM_120 (Blackwell Compatibility Guide)

---

## 6. Constraints & Non-Goals

### Constraints (invariants — must not violate):

1. **No PPL regression**: dp4a path must produce bit-identical results to scalar path (same INT4 values, same FP32 scale multiplication). Verify with `bench_ppl_int4_8b` ≤ 23.52 (baseline) / ≤ 21.82 (AWQ).
2. **No re-quantization**: Weight files in `weights_int4_qwen3_8b/` stay as-is. The kernel upcasts nibbles in registers.
3. **Server must work**: `inference_server_int4.cu` decode path (lines 275-520) + batched path (lines 429-486) + prefill path (lines 328-392, using `_pos` variants) must all produce correct output.
4. **`_pos` prefill variants intact**: `attention_decode_batched_gqa_pos`, `update_kv_cache_pos` in `src/kernels/decode.cu` are unrelated to GEMV — do not touch.
5. **Batched path (M=2-16) must not regress**: `gemv_int4_batched_kernel<M>` template handles all M values. dp4a must work for all M, not just M=1.
6. **Build**: `CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build && cmake --build build --parallel` must produce 189 symbols in `build/libblackwell_kernels.a`.
7. **CUDA Graph compatibility**: New kernel must be graph-safe (no H2D memcpy, no host interaction). dp4a path is naturally graph-safe.
8. **`killall hashcat`** before any measurement (60s respawn window).

### Non-goals:

- Do NOT switch to Q4_K super-block format (requires re-quantization + increases unpack complexity)
- Do NOT implement WMMA/tensor-core GEMV (Option C, deferred)
- Do NOT change weight file format or weight loading (`upload_w4`)
- Do NOT change activation quantization (`quantize_int4`)
- Do NOT change kernel API signatures
- Do NOT modify attention, RMSNorm, RoPE, head_norm, or KV cache kernels
- Do NOT touch INT8 production kernels (they're already using dp4a correctly)

---

## 7. Validation Plan

### Step 1: Build

```bash
killall hashcat 2>/dev/null
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build && cmake --build build --parallel
nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l  # expect 189
```

### Step 2: Microbenchmark (isolated GEMV timing)

```bash
./bench/bench_gemv_int4 4096 4096 1 100    # q_proj M=1
./bench/bench_gemv_int4 4096 12288 1 100   # gate_proj M=1
./bench/bench_gemv_int4 4096 151936 1 100  # lm_head M=1
./bench/bench_gemv_int4 4096 4096 8 100    # batched M=8 (regression check)
```
**Pass criteria**: µs/call drops ~2-3× vs pre-change baseline. BW utilization approaches 80%+ of 500 GB/s.

### Step 3: Correctness (text generation output)

```bash
./bench/text_generate_int4_qwen3_8b "The capital of France is" 30
# Expect: coherent text about Paris, ~84+ t/s, matches current output pattern
```
**Pass criteria**: Output coherent, throughput ≥84 t/s.

### Step 4: PPL (quality regression check)

```bash
./bench/bench_ppl_int4_8b
# Expect: PPL ≤23.52 (baseline) or ≤21.82 (AWQ)
```
**Pass criteria**: PPL within ±0.5 of pre-change value.

### Step 5: CUDA Graph decode

```bash
./bench/decode_int4_cgraph_8b 30
```
**Pass criteria**: Graph captures, instantiates, replays. No errors.

### Step 6: HTTP server end-to-end

```bash
# Rebuild server
CUDACXX=/usr/local/cuda-13.3/bin/nvcc nvcc -O3 -std=c++17 -arch=sm_120a \
  server/inference_server_int4.cu build/libblackwell_kernels.a \
  -I include -I /usr/local/cuda-13.3/include \
  -L /usr/local/cuda-13.3/targets/x86_64-linux/lib \
  -o server/inference_server_int4 -lcudart -lpthread -lz

# Start and test
./server/http_subprocess int4_8b &
sleep 5
curl http://localhost:8123/health
curl -X POST http://localhost:8123/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","max_tokens":10}'
```
**Pass criteria**: Coherent JSON response with Paris-related text, ~84+ t/s in health endpoint metrics.

### Step 7: nsys profile (before/after comparison)

```bash
nsys profile --trace=cuda ./bench/text_generate_int4_qwen3_8b "test" 20
# Before: gemv_int4_batched 92.2% of time, 54.8 µs avg
# After: expect gemv_int4_batched time drops to ~20-30 µs avg, BW % rises
```

---

## 8. Risks & Unresolved Questions

### Risks:

1. **Register pressure**: dp4a path may use more registers than scalar float (INT8 buffers + INT32 accumulator + float scale). Current `__launch_bounds__(32, 8)` targets 8 warps/SM. If registers spike, occupancy drops. **Mitigation**: Check register count with `--ptxas-options=-v`. If >64 regs/thread, reduce `__launch_bounds__` occupancy target from 8 to 4.

2. **INT32 overflow in dp4a accumulation**: Max product per element = 7×7=49. Per block-16: 16×49=784. Per thread (stride-32 over K=4096): 8 blocks × 784 = 6272. Well within INT32 max (2.1B). **No overflow risk.** (dp4a accumulates in INT32 across the 4-element SIMD; the outer accumulation per K-block is also INT32.)

3. **dp4a correctness on SM_120**: Confirmed supported via Blackwell Compatibility Guide. INT intrinsics (`__dp4a`, `__dp2a`) are available. No issue expected.

4. **Nibble→int8 unpack correctness**: `(b & 0xF) - 8` gives `[-8,7]` for lo nibble. `((b >> 4) & 0xF) - 8` gives `[-8,7]` for hi nibble. These are the same values as `int4_byte_to_floats` produces, just as int8_t instead of float. **Bit-identical math.**

5. **Sign extension in dp4a**: `__dp4a` treats input `int32_t` as 4 packed `int8_t`. Values `[-8,7]` fit in int8_t range `[-128,127]`. No sign extension issue — dp4a handles signed INT8 natively.

### Unresolved questions:

1. **Exact register count of dp4a path**: Must verify with `--ptxas-options=-v`. If register pressure forces occupancy below 4 warps/SM, BW utilization may drop. Current scalar path uses ~25 regs (per AGENTS.md comment at line 469 area).

2. **Whether batched M>1 benefits from dp4a or regresses**: At M=8, the kernel is already near memory-bound (weight loaded once, 8 activations dot-producted). dp4a may not help M>1 as much as M=1. Must benchmark both paths. If M>1 regresses, consider keeping scalar path for M≥4 via runtime dispatch. **Likely fine**: dp4a reduces per-element compute regardless of M.

3. **Should both `gemv_int4_warp_kernel` and `gemv_int4_batched_kernel` be updated?** Both have the same scalar loop. `gemv_int4_batched` is used by the server (production). `gemv_int4_warp` is used by some bench files. Update both for consistency, or update only batched and verify bench files still link correctly. **Recommendation**: update both, keep them algorithmically identical.

---

## 9. Implementation-Ready Meta-Prompt (Next Worker)

```
ROLE: CUDA kernel optimization engineer

TASK: Replace scalar FP32 unpack with dp4a INT8 dot-product in the INT4 GEMV
kernel to close the M=1 throughput gap (56 → 84+ t/s).

FILES:
- src/kernels/gemv_int8.cu — gemv_int4_batched_kernel<M> (lines 1000-1060)
  and gemv_int4_warp_kernel (lines 482-537). Helper int4_byte_to_floats
  (lines 472-478). Reference dp4a pattern: gemv_int8_warp_kernel (lines 194-250).

CHANGE: In both INT4 kernels' inner loop, replace int4_byte_to_floats + scalar
FMA with nibble→int8 upcast + __dp4a. Load uint2 (8 bytes = 16 nibbles), unpack
to 16 int8_t values in registers, use 4× __dp4a to compute INT32 partial dot,
then acc += (float)sumi * prod_scale. Scale layout ([N][K/16] FP32) and kernel
signatures are UNCHANGED. AWQ scales transparent (folded into W_scale already).

ACCEPTANCE:
- bench/text_generate_int4_qwen3_8b ≥84 t/s with coherent output
- bench/bench_ppl_int4_8b PPL ≤23.52 (no regression)
- bench/bench_gemv_int4 shows ≥2× speedup per call vs baseline
- Batched M=8 path still works (no regression)
- 189 symbols in libblackwell_kernels.a

VALIDATION:
  killall hashcat; cmake --build build --parallel
  ./bench/bench_gemv_int4 4096 4096 1 100
  ./bench/text_generate_int4_qwen3_8b "The capital of France is" 30
  ./bench/bench_ppl_int4_8b
  ./bench/decode_int4_cgraph_8b 30
  ./server/http_subprocess int4_8b &  # then curl test

DO NOT:
- Change kernel signatures, weight format, or quantize_int4
- Touch attention/RMSNorm/RoPE/KV cache/_pos variants
- Switch to Q4_K super-blocks or re-quantize weights
- Modify server callers (same API)
- Implement WMMA/tensor-core path (deferred)
```
