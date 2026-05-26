# Code Context — Blackwell FP4 Kernel Repo

## Files Retrieved

1. `include/blackwell/kernels.h` (lines 1-150) — All 19 public API signatures
2. `include/blackwell/config.h` (lines 1-70) — Hardware constants, tile sizes
3. `src/kernels/gemm.cu` (lines 1-290) — GEMM 128×128×64 + GEMV dynamic K
4. `src/kernels/decode.cu` (lines 1-245) — attention_decode v2 + update_kv_cache + load_kv_cache_qkgv (stub)
5. `src/kernels/fused_decode.cu` (lines 1-100) — fused_qkv_gemv
6. `src/kernels/norm.cu` (lines 110-140) — fused_rmsnorm, apply_swiglu wrappers
7. `src/kernels/rope.cu` (lines 155-180) — fused_rope wrapper
8. `src/kernels/quantize.cu` (lines 85-135) — pack_fp4, unpack_fp4, coalesced_copy wrappers
9. `src/kernels/attention.cu` (lines 1-30) — attention_fp4 stub
10. `src/kernels/prefill.cu` (lines 1-35) — run_prefill_layer stub (not in public API header)
11. `src/kernels/cuda_graphs.cu` (lines 1-30) — capture/launch/destroy_decode_graph stubs
12. `src/kernels/memory.cu` (lines 1-35) — shared_copy_async, async_pipeline_stage stubs
13. `bench/phase_a.cu` (lines 1-210) — Correctness + benchmark harness
14. `bench/decode_bench.cu` (lines 1-270) — End-to-end decode benchmark with CUDA Graphs

---

## 1. Public API — All 19 Signatures (`include/blackwell/kernels.h`)

### Quant/Memory (3 implemented)
| Function | Signature | Status |
|---|---|---|
| `pack_fp4` | `(void* out_fp4, const float* in_fp32, const float* scale_out, int num_elements, cudaStream_t stream=0)` | Done: `src/kernels/quantize.cu` |
| `unpack_fp4` | `(float* out_fp32, const void* in_fp4, const float* scale_in, int num_elements, cudaStream_t stream=0)` | Done: `src/kernels/quantize.cu` |
| `coalesced_copy` | `(float* dst, const float* src, int num_elements, cudaStream_t stream=0)` | Done: `src/kernels/quantize.cu` |

### MatMul (2 implemented + 1 dispatch wrapper)
| Function | Signature | Status |
|---|---|---|
| `gemm_fp4_block_scaled` | `(float* C, const void* A_fp4, const float* A_scale, const void* B_fp4, const float* B_scale, int M, int N, int K, cudaStream_t stream=0)` | Done: `src/kernels/gemm.cu`. Validates M%128=N%128=K%64=0. Dynamic smem=80KB. 2-stage cp.async pipeline. |
| `gemv_fp4` | `(float* y, const void* x_fp4, const float* x_scale, const void* W_fp4, const float* W_scale, int in_features, int out_features, cudaStream_t stream=0)` | Done: `src/kernels/gemm.cu`. Dynamic K (any multiple of 16). 256 threads/block, grid=ceil(N/256). No smem. |
| `dispatch_matmul` | `(float* C, const void* A, const void* B, const float* A_scale, const float* B_scale, int M, int N, int K, KernelMode mode, cudaStream_t stream=0)` | Done: routes to GEMM (Prefill) or GEMV (Decode) |

### Fused Epilogues (3 implemented)
| Function | Signature | Status |
|---|---|---|
| `fused_rmsnorm` | `(float* out, const float* inp, const float* weight, int num_elements, float eps, cudaStream_t stream=0)` | Done: `src/kernels/norm.cu`. Single block (128 threads, max 4096 elements). |
| `fused_rope` | `(float* out_inplace, const float* cos_cache, const float* sin_cache, int heads, int seq_len, int head_dim, cudaStream_t stream=0)` | Done: `src/kernels/rope.cu`. Grid=heads×seq_len, batch=1 only. |
| `apply_swiglu` | `(float* out, const float* gate, const float* up, int num_elements, cudaStream_t stream=0)` | Done: `src/kernels/norm.cu`. Grid=ceil(num/256). |

### Decode Attention (2 implemented, 1 stub)
| Function | Signature | Status |
|---|---|---|
| `attention_decode` | `(float* output, const float* Q, const float* K_cache, const float* V_cache, int seq_pos, int num_heads, int head_dim, int max_seq_len, cudaStream_t stream=0)` | Done: `src/kernels/decode.cu`. 256 threads/head, smem scores[4096]=16KB. One grid block per head. Uses __shfl_xor_sync for warp reduction. Sequential softmax + V-weighted sum. |
| `update_kv_cache` | `(float* k_cache, float* v_cache, const float* k_new, const float* v_new, int batch_idx, int seq_pos, int num_heads, int head_dim, int max_seq_len, cudaStream_t stream=0)` | Done: `src/kernels/decode.cu`. Only batch_idx=0. 256 thread blocks, writes to position seq_pos. |
| `load_kv_cache_qkgv` | `(float* Q, float* K_val, float* V_val, const float* k_cache, const float* v_cache, int batch_idx, int seq_pos, int num_heads, int head_dim, int max_seq_len, cudaStream_t stream=0)` | **STUB**: returns `cudaErrorNotReady` |

### Prefill Attention (1 stub)
| Function | Signature | Status |
|---|---|---|
| `attention_fp4` | `(float* output, const void* Q_fp4, const void* K_fp4, const void* V_fp4, const float* Q_scale, const float* K_scale, const float* V_scale, int batch_size, int seq_len, int num_heads, int head_dim, float scale, cudaStream_t stream=0)` | **STUB**: `src/kernels/attention.cu`, returns `cudaErrorNotReady` |

### Fused Decode (1 implemented)
| Function | Signature | Status |
|---|---|---|
| `fused_qkv_gemv` | `(float* Q_out, float* K_out, float* V_out, const void* x_fp4, const float* x_scale, const void* W_q_fp4, const float* W_q_scale, const void* W_k_fp4, const float* W_k_scale, const void* W_v_fp4, const float* W_v_scale, int hidden, int q_dim, int kv_dim, cudaStream_t stream=0)` | Done: `src/kernels/fused_decode.cu`. Grid=(3,) — one block per Q/K/V. 256 threads each. Each thread handles 1 output dim. Serial K loop over hidden. No smem. |

### CUDA Graphs (3 stubs)
| Function | Signature | Status |
|---|---|---|
| `capture_decode_graph` | `(void** graph_out, void** node_out, void* graph_exec_out, float* d_temp_storage, size_t temp_storage_bytes, cudaStream_t stream=0)` | **STUB**: `cudaErrorNotReady` |
| `launch_decode_graph` | `(void* graph_exec, cudaStream_t stream=0)` | **STUB**: `cudaErrorNotReady` |
| `destroy_decode_graph` | `(void* graph_exec, void* graph)` | **STUB**: `cudaErrorNotReady` |

**Note**: `run_prefill_layer` exists in `src/kernels/prefill.cu` (stub) but is **NOT** declared in `kernels.h`. Dead code or future feature.

---

## 2. Key Implementation Details

### GEMM (`gemm.cu`) — 128×128×64 CTA
- **8 warps** (256 threads), mapped as 4 along M × 2 along N.
- Each warp holds **8** 16×16 WMMA output fragments (2 M-frags × 4 N-frags).
- **2-stage cp.async pipeline**: raw FP4 loaded via `__pipeline_memcpy_async` (16-byte chunks), then dequantized in smem to FP16.
- **Dynamic shared memory**: 80 KB total — 8 KB raw A + 8 KB raw B + 16 KB FP16 A ping + 16 KB FP16 A pong + 16 KB FP16 B ping + 16 KB FP16 B pong.
- **K-tiling**: iterates over `num_tiles = K/64`, each tile does 4× WMMA `mma_sync` of 16×16×16.
- **Fallback**: non-SM_120 arch writes zeros.
- **Constraint**: M%128, N%128, K%64 must be zero, else `cudaErrorInvalidValue`.

### GEMV (`gemm.cu`) — Dynamic K Decode
- **256 threads/block**, grid = `ceil(N/256)`.
- No shared memory. Each thread handles 1 output element `y[n_out]`.
- Inner loop over `k in 0..K`: reads `x_fp4[k]` (same addr for all threads → L1 broadcast), `W_fp4[k*N+n_out]` (coalesced within warp).
- Dequant: `fp4_val * scale[block]`.
- **K must be multiple of 16** (FP4 block size), otherwise error. No upper bound check.
- **Key risk**: each thread reads K values independently — K=2048 means 2048 serial iterations × N/256 blocks. No cache tiling.

### Fused QKV GEMV (`fused_decode.cu`)
- **3 blocks in grid** (block 0=Q, 1=K, 2=V), each 256 threads.
- Each thread handles 1 output element of Q (2048) or K/V (512).
- 3-way branch via `blockIdx.x` to select weight pointer.
- Inner loop over `k in 0..hidden` (2048) — same serial pattern as GEMV.
- **No smem**, no warp-level collaboration.
- **Risk**: 3 blocks × 256 threads = 768 threads total. With 36 SMs, only ~10 SMs actually active. Underutilized for small kv_dim (512). But eliminates 2 kernel launches per layer.

### Attention Decode v2 (`decode.cu`)
- **1 block per head** (256 threads, 8 warps).
- **smem layout**: Q[128] (512 B) + scores[4096] (16,384 B) = 16,896 B total. `cudaFuncSetAttribute` sets 16 KB max dynamic smem.
- **QK dot**: each warp handles disjoint position range (`t_start = warp_id; t < npos; t += 8`). Each lane loads 4 K-values, dots with 4 Q-registers, then `__shfl_xor_sync` reduction across warp. Lane 0 writes `scores[t]`.
- **Softmax**: sequential loop over npos (all threads do same work). Numerically stable (max subtract, exp, sum, normalize).
- **V-weighted sum**: each lane handles 1 dimension `[lane_id]` (head_dim=128), accumulates `sum_t w[t] * V_cache[t][lane_id]`.
- **Assumption**: head_dim=128 hardcoded in register layout (4× Q_reg, each reading 32 elements via lane_id). head_dim=128 is `kHeadDim` in config but not enforced — will silently misbehave for other values.
- **Risk**: single-block per head limits max_seq_len to 4096 positions (scores array size). For longer contexts, need multi-block or larger smem.

### RMSNorm (`norm.cu`)
- Single block, 128 threads, max 4096 elements.
- Warp-reduce sum of squares, then broadcast normalization factor.
- **Risk**: 128 threads for 4096 elements = 32 elements per thread. Single block means 1 SM active — fine for one layer, but 28 layers × 1 block each = serialized across SMs.

### Pack/Unpack/CoalescedCopy (`quantize.cu`)
- Straightforward element-wise kernels. Pack: quantize FP32→FP4 E2M1 using per-block scale. Unpack: reverse. CoalescedCopy: memcpy via global loads/stores.

---

## 3. Configuration Constants (`include/blackwell/config.h`)

| Constant | Value | Notes |
|---|---|---|
| `kSMArchitecture` | 120 | sm_120 / compute_120 |
| `kMaxWarpsPerSM` | 48 | |
| `kMaxRegistersPerSM` | 65536 | |
| `kMaxSharedMemBytesPerSM` | 131072 | 128 KB |
| `kMaxSharedMemBytesPerBlock` | 101376 | 99 KB |
| `kMaxThreadsPerBlock` | 1024 | |
| `kFP4BlockSize` | 16 | Elements per shared scale |
| `kGEMMTileM/N/K` | 128/128/64 | CTA tile |
| `kGEMMWarps` | 8 | 8 warps = 256 threads |
| `kGEMMWarpsM/N` | 4 / 2 | Warp grid mapping |
| `kWMMAFragM/N/K` | 16/16/16 | WMMA fragment shape |
| `kFragsPerWarpM/N` | 2 / 4 | 8 output fragments per warp |
| `kGEMVTileM/N/K` | 8/64/64 | Defined but not used in GEMV kernel (uses 256-thread block instead) |

**Note**: `kGEMVTileM/N/K` appear unused. GEMV kernel uses block-level threading, not tiling.

---

## 4. Stubs and Dead Code

| File | Function | Returns | Risk |
|---|---|---|---|
| `attention.cu` | `attention_fp4` | `cudaErrorNotReady` | Prefill attention needed for end-to-end |
| `prefill.cu` | `run_prefill_layer` | `cudaErrorNotReady` | Not in public header — dead or pending |
| `cuda_graphs.cu` | `capture_decode_graph` | `cudaErrorNotReady` | Graph capture moved to bench (inline) |
| `cuda_graphs.cu` | `launch_decode_graph` | `cudaErrorNotReady` | Not called |
| `cuda_graphs.cu` | `destroy_decode_graph` | `cudaErrorNotReady` | Not called |
| `memory.cu` | `shared_copy_async` | `cudaErrorNotReady` | Pending async copy utility |
| `memory.cu` | `async_pipeline_stage` | `cudaErrorNotReady` | Pending |
| `decode.cu` | `load_kv_cache_qkgv` | `cudaErrorNotReady` | Needed if QKV come pre-packed |

---

## 5. Correctness Verification (`bench/phase_a.cu`)

### Test flow
1. **pack/unpack**: Generate random FP32 data, compute uniform absmax/3 scale, pack to FP4, unpack back, compute `max_rel_err` against original.
2. **GEMV** (small K=64 + dynamic K): Reference `cpu_gemv_ref` computes FP32 dot product. GPU output compared via `max_rel_err`. Benchmarks measure latency (ms) and bandwidth (GB/s).
3. **GEMM**: All-ones input (A=1, B=1), uniform scale=1/3. Measures latency + bandwidth. No correctness comparison (no CPU GEMM ref) — just checks non-crash, non-NaN via successful runs.
4. **RMSNorm + SwiGLU**: All-ones input. Latency + bandwidth only.

### What's NOT verified
- **GEMM output correctness**: no reference computation, no relative error check. Only benchmark timing.
- **Attention decode**: not tested in phase_a at all — only in `decode_bench.cu`.
- **Fused QKV GEMV**: not tested independently.
- **dispatch_matmul**: not tested.
- **Fused RoPE**: not tested in phase_a (only in `decode_bench.cu` as part of full pipeline? No — decode_bench doesn't call RoPE either).

### Risk: GEMM correctness unknown
GEMM uses all-ones input with uniform scale. Since `1.0 * scale(1/3) = 0.333` for both A and B, and WMMA accumulates 16×16×16 = 4096 multiplies per fragment, output = `4096 * 0.333 * 0.333 = ~455`. No reference comparison done — kernel could silently produce wrong results and pass timing.

---

## 6. Decode Benchmark (`bench/decode_bench.cu`)

### Simulated layer pipeline
```
x_fp4 → fused_qkv_gemv(Q,K,V) → update_kv_cache → attention_decode → 
pack_fp4(attn_out) → gemv_fp4(W_o) → fused_rmsnorm → pack_fp4(x) → next layer
```

### Model parameters
- hidden_dim=2048, Q_heads=16, KV_heads=4, head_dim=128
- 4 layers (not 28 — reduced for faster testing)
- max_seq_len=2048
- All weights uniform 1.0, scale=1/3

### Warmup/Setup
- 3 warmup tokens
- KV cache pre-filled to seq_pos=128 so attention has non-trivial workload

### CUDA Graph capture
Captures all 4 layers × 6 kernel launches = 24 ops as a single CUDA Graph. Benchmarks graph vs direct launch. Reports `t/s` and `scaled to 28 layers` extrapolation.

### What it tests
- Correctness: none (no output validation, no ref comparison)
- Performance: per-token ms, t/s, graph vs direct overhead comparison

### Risk
No correctness check. If attention_decode or fused_qkv produce wrong values, benchmark still "passes" with meaningless numbers.

---

## 7. Architecture Summary

```
Data flow per decode step:
  x_fp4 (hidden=2048)
    → fused_qkv_gemv           [1 kernel, 3 blocks]
        → Q (2048), K (512), V (512)  [FP32]
    → update_kv_cache          [2 kernels, N/256 blocks each]
        → k_cache, v_cache     [FP32, per-layer]
    → attention_decode         [num_heads=16 kernels, 1 block/head]
        → attn_out (2048)      [FP32]
    → pack_fp4                 [1 kernel]
        → attn_fp4             [FP4]
    → gemv_fp4(W_o)           [1 kernel]
        → proj_out (2048)      [FP32]
    → fused_rmsnorm           [1 kernel]
        → x_fp32               [FP32]
    → pack_fp4                 [1 kernel]
        → x_fp4 (next layer)   [FP4]

Total: ~7 kernel launches per layer = ~28 per 4-layer step → ~196 for 28 layers
```

CUDA Graph captures these into 1 graph object, eliminating launch overhead.

---

## 8. Start Here

**`src/kernels/decode.cu`** — attention_decode v2 is the most complex working kernel. Read it first to understand the warp-parallel QK dot product pattern, smem score buffering, and sequential softmax/V-sum. Then `src/kernels/gemm.cu` for the cp.async pipeline and WMMA usage.

For adding a new kernel: `src/kernels/fused_decode.cu` is simplest working pattern (flat loop, no smem, no warp sync).

---

## 9. Open Questions / Risks

1. **GEMM correctness unverified** — no reference comparison in phase_a. Could produce wrong results.
2. **attention_decode hardcodes head_dim=128** — will break for other head_dims. No guard.
3. **attention_decode max 4096 seq_len** — smem scores array limits context. Need >= 8192 for real models.
4. **GEMV serial K loop** — 2048 iterations per thread, no tiling, no L1 reuse for W (stride=N between K rows). Memory-bound but sequential access pattern may underutilize HBM bandwidth.
5. **GEMM double dequant** — cp.async loads raw FP4, then threads dequant to FP16. This uses 2× smem bandwidth (write raw, read raw, write fp16). Could fuse dequant into the cp.async wait stage.
6. **fused_qkv_gemv replicates work** — 3 blocks each iterate hidden=2048 independently. x_fp4 loaded 3× by each block. For KV with kv_dim=512, could fuse into single block with different output ranges.
7. **fused_rmsnorm capped at 4096 elements** — larger layer norms need multi-block reduction.
8. **No prefill attention** — `attention_fp4` is stub. End-to-end only works for decode.
