# AGENTS.md - blackwell

Custom CUDA kernels for INT8 + FP4 LLM inference on RTX 5060 Ti (Blackwell, SM_120a).

---

## 1. Mission

Benchmark INT8 forward pass throughput vs llama.cpp (Q4_K_M) baseline (114 t/s).
Current: **93.9 t/s** (28L INT8 pipeline), **128 t/s** with CUDA Graph, **60 t/s** Mode D prefill+decode. Text output correct.

---

## 2. Active State

**Stack**: CUDA 13.3, SM_120a, CMake, C++17
**Target**: RTX 5060 Ti 16 GB, compute 12.0, 36 SMs, ~500 GB/s GDDR7
**Nvcc path**: `/usr/local/cuda-13.3/bin/nvcc`
**Library**: 78 symbols in `build/libblackwell_kernels.a`

**Production kernels (INT8 path)**:
- `gemv_int8` — INT8 GEMV, `__dp4a` SIMD, 775 GB/s (kernel), 260 GB/s (effective)
- `gemv_int8_batched` — batched M=1-8, sweet spot M=3-4 (1.4× speedup)
- `gemv_int8_splitk` — split-K (K_splits=4), 779 GB/s
- `pack_int8` — FP32 → INT8 quant
- `transpose_int8_weights` — W (K×N) → W_t (N×K) + scales
- `fused_rmsnorm_quant_int8` — RMSNorm + INT8 quant (1 kernel)
- `fused_gate_up_gemv` — fused gate+up MLP projection
- `fused_rmsnorm` — single-block warp-reduced RMSNorm
- `apply_swiglu` — silu(gate) × up, elementwise
- `fused_rope` — in-place rotation, smem cos/sin cache
- `attention_decode_gqa` — GQA-aware decode attention
- `update_kv_cache` — KV cache write with per-layer offset

**Research kernels (FP4 path)**:
- `gemv_fp4_nv` — NVF4 scalar GEMV, UE4M3 scales, 98 GB/s (correct, not competitive)
- `pack_fp4` / `unpack_fp4` — FP4 E2M1 quant/dequant
- `gemm_fp4_block_scaled` — FP4 GEMM prefill
- `gemm_int8` — INT8 GEMM prefill (M>1, per-block scales, 4×4 tiling)

**Deprecated / DO NOT USE**:
- `gemv_int8_persistent` — 23× slower than baseline
- `gemv_int8_from_fp4` — 2.8× slower than baseline
- `phase_a.cu` — depends on unimplemented symbols (`gemv_fp4_splitk`, `gemv_fp4_v3`, `gemv_fp4_batched`)
- NVF4 tensor core MMA — scale factor layout mismatch for GEMV

---

## 3. Build & Run

### Build
```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### Benchmark
```bash
./bench/decode_full_int8 4                    # INT8 pipeline, 93.9 t/s (scaled 28L)
./bench/text_generate "The capital of France is" 30  # Text gen, "Paris" ✓
./bench/inference_server 28 4 20 8            # CUDA Graph 128 t/s, batched 40 req/s
```

### Model
`/mnt/data/ai/hf/qwen3-1.7b-base/` — Qwen3-1.7B, 28 layers, safetensors, 3.3 GB

---

## 4. Key Findings

| Finding | Value | Notes |
|---------|-------|-------|
| INT8 GEMV kernel BW | 775 GB/s | `__dp4a` SIMD |
| INT8 effective BW | 260 GB/s | Weight-bound (L2 cache miss) |
| NVF4 scalar BW | 98 GB/s | FP4→float conversion ceiling |
| INT8 vs NVF4 | 2.65× | FP4 can't match INT8 for decode |
| GEMM prefill | 78 GB/s | 3× faster than llama.cpp |
| CUDA Graph speedup | ~10% | 117→128 t/s |
| Batched GEMV M=8 | 17344 batch t/s | 18.86× vs per-kernel |

---

## 5. Constraints

- `CUDACXX` env var must be set before `project()` in CMakeLists.txt
- `compute_120a` required (not `compute_120`) — FP4 block-scale MMA needs 12Xa
- `namespace wmma = nvcuda::wmma` (alias, NOT `using wmma =`)
- `sizeof(__nv_fp4_e2m1)` = 1 byte (not 0.5)
- All weight matrices exceed L2 cache (32 MB) — architectural limit for single-token decode
- System ptxas may be old — ensure CUDA 13.3 in PATH

---

## 6. Bug History

### vector_add_fp32_kernel (2026-05-28) — FIXED
`src/kernels/norm.cu`: reversed `=` in float4 path wrote uninitialized data TO input buffer.
Fix: `float4 va = ((float4*)a)[idx];` (load, not store).

### RoPE frequency (2026-05-29) — FIXED
All 5 bench files: `idxf = i2/hd` doubled exponent → 2× rotation speed.
Fix: `theta = pos * powf(rope_theta, -2.0f * d / head_dim);`

### head_norm cross-warp (2026-05-29) — FIXED
All 5 bench files: `__shfl_xor_sync` with off=64/32 no-ops on 32-lane warps → 1/4 sums.
Fix: smem[4] warp partials → shuffle-reduce across 4 warps.

### inference_server syntax (2026-05-29) — FIXED
Stray `}` after head_norm_kernel closing brace. Deleted.

---

## 7. Known Issues

1. **Mode D prefill** — FIXED (6e775eb). GEMM B buffer OOB in synthetic prefill. Now runs: 68 t/s pipeline.
2. **FP32 text_generate broken** — `text_generate_fp32.cu` produces worse output than INT8. Separate issue (BF16 weight file convention or cuBLAS transpose).
3. **GEMM prefill correctness unverified** — no reference comparison. Timing-only validation.
4. **7 stub functions unimplemented** — `attention_fp4`, `load_kv_cache_qkgv`, `capture_decode_graph`, `launch_decode_graph`, `destroy_decode_graph`, `shared_copy_async`, `async_pipeline_stage`.

---

## 8. Anti-Hallucination Rules

- **Do not invent APIs, files, commands, env vars, or requirements.** Read the actual header/source before calling a function.
- **Prefer repo evidence over assumptions.** If you need a function signature, read `include/blackwell/kernels.h`.
- **Mark unknowns explicitly.** "Not checked" or "unknown behavior" in comments.
- **Never overwrite higher-priority instructions.**
- **Preserve user intent and existing project conventions.**

---

## 9. Development Loop

```
observe → plan → edit → build → test → reflect → update AGENTS.md only if useful
```

Build: `CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake --build build --parallel`
Test: `./bench/decode_full_int8 4` (pipeline), `./bench/text_generate ...` (correctness)
Verify: `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l` (expect 78)

---

## 10. Seed Principles

1. Smallest correct change. One kernel, one fix, one test.
2. Verify before broad edits.
3. Prefer repo evidence. Read code before assuming.
4. No churn.
5. Kernels first, framework later.
