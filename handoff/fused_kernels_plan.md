# Fused Kernel Implementation Plan

## Overview

Three kernel fusions to reduce GEMV calls per layer from 8 to 5 (saving 37.5%).

| Fusion | GEMV calls saved | Current bottleneck | Expected gain |
|--------|-----------------|-------------------|---------------|
| Fused QKV | 2 (Q, K, V → 1) | 3× activation read + quantize | +25% M=1 |
| Fused gate+up | 1 (gate, up → 1) | 2× activation read | +12.5% M=1 |
| INT4 GEMM (prefill) | N/A | Token-by-token prefill | +19% long prompts |

---

## 1. Fused QKV for INT4 (FP16 scales)

### Goal
Replace 3 separate `gemv_int4_batched_f16wsc` calls (Q, K, V) with one fused kernel that reads activation once.

### Design
```cuda
// Current: 3 kernel launches per layer
gemv_int4_batched_f16wsc(d_Q, x, x_sc, W_q, W_q_sc, H, Q, M, st);
gemv_int4_batched_f16wsc(d_K, x, x_sc, W_k, W_k_sc, H, KV, M, st);
gemv_int4_batched_f16wsc(d_V, x, x_sc, W_v, W_v_sc, H, KV, M, st);

// Fused: 1 kernel launch
fused_qkv_int4_f16wsc(d_Q, d_K, d_V, x, x_sc, W_q, W_q_sc, W_k, W_k_sc, W_v, W_v_sc, H, Q, KV, M, st);
```

### Kernel signature
```cuda
cudaError_t fused_qkv_int4_f16wsc(
    float*          Q_out,       // [M][Q]
    float*          K_out,       // [M][KV]
    float*          V_out,       // [M][KV]
    const uint8_t*  x_packed,    // [M][K/2] INT4 activations
    const float*    x_scale,     // [M][K/16] FP32 scales
    const uint8_t*  W_q_packed,  // [N_q][K/2]
    const void*     W_q_scale,   // __half* [N_q][K/16]
    const uint8_t*  W_k_packed,  // [N_kv][K/2]
    const void*     W_k_scale,   // __half* [N_kv][K/16]
    const uint8_t*  W_v_packed,  // [N_kv][K/2]
    const void*     W_v_scale,   // __half* [N_kv][K/16]
    int             K,
    int             N_q,
    int             N_kv,
    int             M,
    cudaStream_t    stream = 0);
```

### Implementation strategy
- Grid: `max(N_q, N_kv) × M` blocks, 32 threads/block
- Block at `(n, m)`: if `n < N_q`, compute Q row n for token m; if `n < N_kv`, compute K+V row n
- Weight data loaded once per K-block per output row
- Activation data shared across Q/K/V via register reuse (same K-block loaded once, used 3×)
- Uses dp4a inner loop (same as existing gemv_int4_batched_kernel)
- Adds template parameter for WScaleT (float/__half) to support FP16 scales

### Files to create
- `src/kernels/fused_qkv_int4.cu` (new)
- Update `include/blackwell/kernels.h` (add declaration)
- Update `CMakeLists.txt` (add to KERNEL_SOURCES)

### Files to change
- `bench/text_generate_int4_batched.cu` (replace QKV calls)
- `server/inference_server_int4_batched.cu` (replace QKV calls)
- `server/inference_server_int4.cu` (replace QKV calls)

### Validation
```bash
./bench/text_generate_int4_batched "The capital of France is" 1 30 weights_int4_qwen3_8b_fp16sc
# Expected: 62→~75 t/s M=1
./bench/text_generate_int4_batched "The capital of France is" 8 20 weights_int4_qwen3_8b_fp16sc
# Expected: 171→~190 t/s M=8
```

### Risks
- Register pressure: 3× output accumulators per thread may spill to local memory
- smem needed for activation reuse across Q/K/V — 4KB for K=4096, 12KB for I=12288
- But we can avoid smem: each thread loads K-block from global memory once, computes all 3 dot products in registers

---

## 2. Fused gate+up for INT4 (FP16 scales)

### Goal
Replace 2 separate `gemv_int4_batched_f16wsc` calls (gate, up) with one fused kernel that reads activation once.

### Design
```cuda
// Current: 2 kernel launches per layer
gemv_int4_batched_f16wsc(d_gate, x, x_sc, W_g, W_g_sc, H, I, M, st);
gemv_int4_batched_f16wsc(d_up, x, x_sc, W_u, W_u_sc, H, I, M, st);

// Fused: 1 kernel launch
fused_gate_up_int4_f16wsc(d_gate, d_up, x, x_sc, W_g, W_g_sc, W_u, W_u_sc, H, I, M, st);
```

### Kernel signature
```cuda
cudaError_t fused_gate_up_int4_f16wsc(
    float*          gate_out,    // [M][I]
    float*          up_out,      // [M][I]
    const uint8_t*  x_packed,    // [M][K/2] INT4 activations
    const float*    x_scale,     // [M][K/16] FP32 scales
    const uint8_t*  W_g_packed,  // [I][K/2]
    const void*     W_g_scale,   // __half* [I][K/16]
    const uint8_t*  W_u_packed,  // [I][K/2]
    const void*     W_u_scale,   // __half* [I][K/16]
    int             K,
    int             I,
    int             M,
    cudaStream_t    stream = 0);
```

### Implementation strategy
- Grid: I × M blocks, 32 threads/block
- Each block processes one output row of both gate and up
- Weight data loaded once per K-block, used for both gate and up
- Actication data loaded once per K-block
- Two separate fp32 accumulators (gate_acc, up_acc)
- dp4a inner loop
- Template on WScaleT

### Files to create
- `src/kernels/fused_gate_up_int4.cu` (new)
- Update `include/blackwell/kernels.h`
- Update `CMakeLists.txt`

### Files to change
- `bench/text_generate_int4_batched.cu`
- Server files

### Validation
Same as fused QKV.

---

## 3. INT4 GEMM for Prefill

### Goal
Replace token-by-token decode-style prefill with a batched GEMM that processes all prompt tokens in parallel through each layer.

### Design
```cuda
// Current: per-token decode (M tokens, M iterations over layers)
for (int m = 0; m < M; ++m) {
    for (int l = 0; l < NL; ++l) {
        gemv_int4_batched_f16wsc(d_Q_m, x_m, ..., W[l].q, ..., 1, st);
        // ... head_norm, RoPE, attention (reads K/V cache from all prior tokens)
    }
}

// Fused GEMM: process all M tokens as a matrix multiply
// Q = X @ W_q^T  where X is [M, H], W_q is [Q, H]
// K = X @ W_k^T  where X is [M, H], W_k is [KV, H]
// V = X @ W_v^T  where X is [M, H], W_v is [KV, H]
//
// Then per-token: head_norm, RoPE, attention (still per-token due to causal mask)
```

### Key insight
The per-token part (head_norm, RoPE, attention) is cheap (~5 μs/token). The expensive part is the QKV GEMVs (~46 μs each). By replacing 3× GEMV per token with 3× GEMM, we get:

- Q GEMM: `[M, H] × [H, Q]` = M × Q × H operations
- vs Q GEMV: M × Q × H operations (same FLOPs but much better GPU utilization)

The GEMM can use tensor cores (WMMA) for 10-100× better throughput than GEMV.

### Required kernels
```cuda
// INT4 × INT8 GEMM using tensor cores
// A: [M][K/2] packed INT4, A_scale: [M][K/16] FP32
// B: [N][K/2] packed INT4, B_scale: [N][K/16] __half
// C: [M][N] FP32 output
cudaError_t gemm_int4_int8(
    float*          C,
    const uint8_t*  A_packed,   // [M][K/2]
    const float*    A_scale,    // [M][K/16]
    const uint8_t*  B_packed,   // [N][K/2]
    const void*     B_scale,    // __half* [N][K/16]
    int             M,
    int             N,
    int             K,
    cudaStream_t    stream = 0);
```

### Implementation strategy
- Use WMMA (Warp Matrix Multiply-Accumulate) with INT4/INT8 tensor cores on SM120a
- Tile: 64×64×64 or 128×128×64
- Dequantize INT4 blocks on the fly: load packed nibbles, unpack to INT8, apply scales
- fp16 weight scales → convert to fp32 in registers
- fp32 activation scales → multiply into accumulator
- Result: fp32 accumulator → write C

### Alternative (simpler)
If tensor core GEMM is too complex, use batched GEMV:
```cuda
// Process all M tokens through one layer via batched GEMV
gemv_int4_batched_f16wsc(d_Q_all, x_packed, x_sc, W_q, W_q_sc, H, Q, M, st);
```
This already works (the batched GEMV kernel supports M up to 16). The prefilling function already uses this pattern (`prefill_tokens_batched` in the server). The gain comes from reducing per-token overhead (kernel launches, memory copies).

### Files to create
- `src/kernels/gemm_int4.cu` (new, tensor core GEMM)

### Files to change
- `include/blackwell/kernels.h`
- `CMakeLists.txt`
- `server/inference_server_int4.cu` (prefill_tokens_batched — already uses batched GEMV)

### Validation
- Compare prefill time: `bench/prefill_decode_benchmark.cu` (INT8) vs INT4 equivalent
- Server throughput with long prompts

---

## Implementation Order

1. **Fused gate+up** (simplest, same N=I for both matrices, no smem needed)
2. **Fused QKV** (different N for Q vs K/V, slightly more complex)
3. **INT4 GEMM** (most complex, requires tensor core programming)

## Expected Impact Summary

| Change | M=1 (t/s) | M=8 (t/s) | Effort |
|--------|-----------|-----------|--------|
| Baseline | 62 | 171 | — |
| +fused gate+up | 70 | 185 | 1 day |
| +fused QKV | 78 | 200 | 1-2 days |
| +GEMM prefill | 78 | 200+ | 3-5 days |