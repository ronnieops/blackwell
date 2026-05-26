# Code Context — Blackwell FP4 Kernels

## Build State

| Artifact | Path | Size | Last Modified |
|----------|------|------|---------------|
| Static lib | `build/libblackwell_kernels.a` | 256 KB | 2026-05-26 15:15 |
| Benchmark binary | `bench/phase_a` | 1.2 MB ELF x86-64 | 2026-05-26 15:24 |
| GTest tests | `build/` (no test binary) | — | GTest not found — tests disabled |
| pybind11 module | `build/` (no .so) | — | pybind11 not found — bindings disabled |

**Build command**: `CUDACXX=/usr/local/cuda-12.8/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel`

**Target**: SM_120 native (RTX 5060 Ti, CC 12.0), CUDA 12.8, host compiler g++-12.

**Library symbol count**: 36 text symbols total (27 namespace-qualified functions + 9 `__device_stub__` wrappers). 18 public API functions in `blackwell::kernels::`, 9 `(anonymous namespace)` device kernels.

---

## Files Retrieved

1. **`include/blackwell/kernels.h`** (174 lines) — Public API signatures. 18 functions + `KernelMode` enum.
2. **`include/blackwell/config.h`** (64 lines) — Tile sizes, block limits, SM_120 constants.
3. **`CMakeLists.txt`** (135 lines) — Build config, SM_120 flags, compiler workarounds.
4. **`HANDOFF.md`** — Operational state, known bugs, pending tasks. Last validated 2026-05-26 15:24 UTC.
5. **`bench/phase_a.cu`** (290 lines) — Integrated benchmark. Synthetic uniform data (all 1.0s).
6. **`bench/PHASE_A_RESULTS.md`** — Phase A results: GEMM 3.6–5.3 GB/s, GEMV up to 168 GB/s at 6K elements.
7. **`src/kernels/gemm.cu`** (251 lines) — GEMM (working, slow) + GEMV (K=64 hardcoded).
8. **`src/kernels/quantize.cu`** (136 lines) — pack_fp4 + unpack_fp4 (verified correct).
9. **`src/kernels/norm.cu`** (140 lines) — RMSNorm + SwiGLU (working).
10. **`src/kernels/rope.cu`** (180 lines) — fused_rope (working).
11. **`src/kernels/attention.cu`** (32 lines) — Stub, returns `cudaErrorNotReady`.
12. **`src/kernels/decode.cu`** (41 lines) — Stub KV-cache ops, returns `cudaErrorNotReady`.
13. **`src/kernels/prefill.cu`** (36 lines) — Stub, returns `cudaErrorNotReady`.
14. **`src/kernels/cuda_graphs.cu`** (38 lines) — Stub, returns `cudaErrorNotReady`.
15. **`src/kernels/memory.cu`** (36 lines) — Stub async copy ops, returns `cudaErrorNotReady`.

---

## Key Code

### Public API (`include/blackwell/kernels.h`)
```cpp
namespace blackwell { namespace kernels {
enum class KernelMode { Prefill, Decode };

// Working (verified)
cudaError_t pack_fp4(void* out_fp4, const float* in_fp32, const float* scale_out, int num_elements, cudaStream_t stream = 0);
cudaError_t unpack_fp4(float* out_fp32, const void* in_fp4, const float* scale_in, int num_elements, cudaStream_t stream = 0);
cudaError_t gemm_fp4_block_scaled(float* C, const void* A_fp4, const float* A_scale, const void* B_fp4, const float* B_scale, int M, int N, int K, cudaStream_t stream = 0);
cudaError_t gemv_fp4(float* y, const void* x_fp4, const float* x_scale, const void* W_fp4, const float* W_scale, int in_features, int out_features, cudaStream_t stream = 0);
cudaError_t fused_rmsnorm(float* out, const float* inp, const float* weight, int num_elements, float eps, cudaStream_t stream = 0);
cudaError_t fused_rope(float* out_inplace, const float* cos_cache, const float* sin_cache, int heads, int seq_len, int head_dim, cudaStream_t stream = 0);
cudaError_t apply_swiglu(float* out, const float* gate, const float* up, int num_elements, cudaStream_t stream = 0);
cudaError_t dispatch_matmul(float* C, const void* A, const void* B, const float* A_scale, const float* B_scale, int M, int N, int K, KernelMode mode, cudaStream_t stream = 0);
cudaError_t coalesced_copy(float* dst, const float* src, int num_elements, cudaStream_t stream = 0);

// Stubs (return cudaErrorNotReady)
cudaError_t attention_fp4(...);
cudaError_t update_kv_cache(...);
cudaError_t load_kv_cache_qkgv(...);
cudaError_t capture_decode_graph(...);
cudaError_t launch_decode_graph(...);
cudaError_t destroy_decode_graph(...);
} }
```

### Tile Sizes (`include/blackwell/config.h`)
```cpp
constexpr int kGEMMTileM = 16;   // WMMA fragment size
constexpr int kGEMMTileN = 16;
constexpr int kGEMMTileK = 64;
constexpr int kGEMVTileM = 8;
constexpr int kGEMVTileN = 64;
constexpr int kGEMVTileK = 64;
constexpr int kFP4BlockSize = 16;   // block-scaled FP4 grouping
constexpr int kMaxSharedMemBytesPerBlock = 101376;  // 99 KB
```

### Source files by role

| File | Lines | Role | Status |
|------|-------|------|--------|
| `src/kernels/gemm.cu` | 251 | GEMM + GEMV kernel implementations | Working; GEMM 3–5 GB/s (needs optimization); GEMV K=64 hardcoded |
| `src/kernels/quantize.cu` | 136 | FP4 pack/unpack + coalesced_copy | Verified correct |
| `src/kernels/norm.cu` | 140 | RMSNorm + SwiGLU | Working |
| `src/kernels/rope.cu` | 180 | RoPE (in-place + out-of-place) | Working |
| `src/kernels/attention.cu` | 32 | Fused attention placeholder | Stub |
| `src/kernels/decode.cu` | 41 | KV-cache update/load | Stub |
| `src/kernels/prefill.cu` | 36 | Prefill layer orchestrator | Stub |
| `src/kernels/cuda_graphs.cu` | 38 | CUDA Graph capture/launch/destroy | Stub |
| `src/kernels/memory.cu` | 36 | Async copy / pipeline helpers | Stub |

### Source file timestamps vs build
```
src/kernels/gemm.cu      2026-05-26 15:14  — most recent edit (GEMM/GEMV)
include/blackwell/config.h 2026-05-26 15:07  — edited
bench/phase_a.cu         2026-05-26 15:21
build/libblackwell_kernels.a 2026-05-26 15:15
bench/phase_a            2026-05-26 15:24
```
Bench binary built after library (separate nvcc link step). Consistent.

---

## Architecture

```
bench/phase_a.cu  (driver: calls public API, measures latency/BW)
       │
       ▼
 blackwell::kernels::*  (public API in include/blackwell/kernels.h)
       │
       ▼
 Device kernels (anonymous namespace in each .cu)
       │
       ▼
 nvcuda::wmma::mma_sync  (GEMM, guarded #if __CUDA_ARCH__ >= 800)
```

**Data flow** (GEMM prefill):
1. Caller provides FP4-quantized A, B matrices + per-block scale arrays
2. `dispatch_matmul` routes to `gemm_fp4_block_scaled` (Prefill) or `gemv_fp4` (Decode)
3. Kernel loops K in 64-element tiles, loads FP4 → on-the-fly FP16 conversion → WMMA multiply-accumulate
4. Result accumulates in FP32 fragment → stores to C

**Performance**: GEMM 3–5 GB/s (0.6–1% of 500 GB/s peak). GEMV 1.6–168 GB/s (launch-overhead-bound below ~2K outputs).

**Stub components**: attention, KV-cache, prefill runner, CUDA Graphs — all return `cudaErrorNotReady`.

---

## Start Here

Open **`src/kernels/gemm.cu`** (line 1) — GEMM is the primary performance bottleneck (3–5 GB/s) and the next optimization target per HANDOFF.md. It contains both `gemm_fp4_kernel` (WMMA-based) and `gemv_fp4_kernel` (K=64 hardcoded). The tile sizes in `include/blackwell/config.h` (kGEMMTile* = 16/16/64) are the starting point for optimization.

---

## Discrepancies vs HANDOFF.md

| Item | HANDOFF Claims | Actual | Match? |
|------|---------------|--------|--------|
| Git repo | ❌ Not initialized | No `.git` dir | ✅ |
| libblackwell_kernels.a | ✅ Builds clean | 256 KB .a exists | ✅ |
| bench/phase_a | ✅ Runs clean | 1.2 MB ELF exists | ✅ |
| Public API symbols | 18 in blackwell::kernels:: (not anonymous) | 18 public + 9 anonymous kernels + 9 device_stubs = 36 total T symbols | ✅ |
| GEMM throughput | 3–5 GB/s | PHASE_A_RESULTS.md reports 3.6–5.3 GB/s | ✅ |
| GEMV K=64 hardcoded | ✅ Confirmed | gemv_fp4 kernel uses single K=64 tile | ✅ |
| Stub fns (cudaErrorNotReady) | attention, KV-cache, prefill, CUDA Graphs | Same 9 stubs found | ✅ |
| Timestamps | Phase A complete ~15:24 UTC | bench/phase_a built 15:24, .a built 15:15 | ✅ |
| GTest | "if not found → disabled" | Not found, no test binary | ✅ |
| pybind11 | "if not found → disabled" | Not found, no .so | ✅ |

**No discrepancies found.** Project state matches HANDOFF.md description exactly.

---

## Constraints & Risks

1. **GEMM 3–5 GB/s** (red block) — must hit 100+ GB/s for viable LLM inference. Small 16×16 tiles + no vectorized loads + no async copy.
2. **GEMV K=64 hardcoded** (yellow block) — Qwen3-1.7B needs hidden_dim=2048 → 32 GEMV invocations per layer → launch overhead dominates.
3. **sizeof(__nv_fp4_e2m1) = 1 byte** — allocations must be N*1, not N/2.
4. **No git history** — no rollback safety for major edits.
5. **All stubs** — attention, KV-cache, prefill orchestrator, CUDA Graphs are non-functional.
6. **WMMA code guarded by `__CUDA_ARCH__ >= 800`** — works on SM_120 (≥800), but non-SM_120 fallback doesn't exist.
7. **Benchmark uses uniform 1.0 data** — real model weight distribution will differ (optimization may behave differently).
