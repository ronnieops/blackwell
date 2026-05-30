# AGENTS.md - blackwell

Custom CUDA kernels for INT8 + FP4 LLM inference on RTX 5060 Ti (Blackwell, SM_120a).

---

## 1. Mission

Benchmark INT8 forward pass throughput vs llama.cpp (Q4_K_M) baseline (**253.6 t/s**, re-measured 2026-05-30).
Current: **173.8 t/s** CUDA Graph (warp-cooperative GEMV, 69% of baseline), **155.5 t/s** per-kernel. Text output correct. **92 library symbols**.

---

## 2. Active State

**Stack**: CUDA 13.3, SM_120a, CMake, C++17
**Target**: RTX 5060 Ti 16 GB, compute 12.0, 36 SMs, ~500 GB/s GDDR7
**Nvcc path**: `/usr/local/cuda-13.3/bin/nvcc`
**Library**: 92 symbols in `build/libblackwell_kernels.a`

**Production kernels (INT8 path)**:
- `gemv_int8_warp` — Warp-cooperative INT8 GEMV (1 warp/row, shuffle reduce, **173.7 t/s**)
- `gemv_fp32_int8_per_row_warp` — Warp-cooperative FP32×INT8 GEMV (1 warp/row, shuffle reduce)
- `gemv_int8` — Legacy INT8 GEMV, `__dp4a` SIMD, 775 GB/s (kernel), 260 GB/s (effective). **Superseded by gemv_int8_warp**
- `gemv_int8_batched` — batched M=1-8, sweet spot M=3-4 (1.4× speedup)
- `gemv_int8_splitk` — split-K (K_splits=4), 779 GB/s
- `pack_int8` / `quantize_int8` — FP32 → INT8 quant with block scales
- `transpose_int8_weights` — W (K×N) → W_t (N×K) + scales
- `fused_rmsnorm_quant_int8` — RMSNorm + INT8 quant (1 kernel)
- `fused_gate_up_gemv` — fused gate+up MLP projection
- `fused_rmsnorm` — single-block warp-reduced RMSNorm
- `apply_swiglu` — silu(gate) × up, elementwise
- `fused_rope` / `fused_rope_decode` — in-place rotation, smem cos/sin cache
- `attention_decode_gqa` — GQA-aware decode attention
- `update_kv_cache` — KV cache write with per-layer offset

**Research kernels (FP4 path)**:
- `gemv_fp4_nv` / `gemv_fp4_nv_opt` — NVF4 scalar GEMV, UE4M3 scales, 98 GB/s (correct, not competitive)
- `pack_fp4` / `unpack_fp4` — FP4 E2M1 quant/dequant
- `gemm_fp4_block_scaled` — FP4 GEMM prefill
- `gemm_int8` / `gemm_int8_dp4a` — INT8 GEMM prefill (M>1, per-block scales, 4×4 tiling)

**Deprecated / DO NOT USE**:
- `gemv_int8_from_fp4` — 2.8× slower than baseline
- `phase_a.cu` — depends on unimplemented symbols (`gemv_fp4_splitk`, `gemv_fp4_v3`, `gemv_fp4_batched`)
- NVF4 tensor core MMA — scale factor layout mismatch for GEMV

---

## 3. Build & Run

### Build
```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### Benchmark
```bash
./bench/decode_int8_cgraph 28              # CUDA Graph 173 t/s (production)
./bench/decode_full_int8 28                # Per-kernel 155 t/s
./bench/bench_warp_gemv                    # Isolated warp vs old GEMV comparison
./bench/text_generate "The capital of France is" 30  # Text gen, "Paris" ✓
```

### Model
`/mnt/data/ai/hf/qwen3-1.7b-base/` — Qwen3-1.7B, 28 layers, safetensors, 3.3 GB

---

## 4. Key Findings

| Finding | Value | Notes |
|---------|-------|-------|
| Warp GEMV speedup | **2.5×** (Q proj), **2×** (MLP) | Coalesced loads (1 warp/row) |
| INT8 CUDA Graph (warp) | **173.7 t/s** | 152% of 114 target |
| INT8 per-kernel (warp) | **155.5 t/s** | 136% of 114 target |
| INT8 effective BW | 260 GB/s | Weight-bound (L2 cache miss) |
| NVF4 scalar BW | 98 GB/s | FP4→float conversion ceiling |
| GEMM prefill | 78 GB/s | 3× faster than llama.cpp |
| CUDA Graph speedup | ~10% | 155→173 t/s (warp path) |
| L2 cache hints | +0.3% | Marginal (no weight reuse) |
| Attention decode | 13.5% of pipeline | New non-GEMV bottleneck |
| MLP GEMV | 57.7% of pipeline | Still dominant cost |

---

## 5. Constraints

- `CUDACXX` env var must be set before `project()` in CMakeLists.txt
- `compute_120a` required (not `compute_120`) — FP4 block-scale MMA needs 12Xa
- `namespace wmma = nvcuda::wmma` (alias, NOT `using wmma =`)
- `sizeof(__nv_fp4_e2m1)` = 1 byte (not 0.5)
- All weight matrices exceed L2 cache (32 MB) — architectural limit for single-token decode
- System ptxas may be old — ensure CUDA 13.3 in PATH
- Warp kernel requires K%16==0 and N%16==0 (inherited from block-16 quantization)
- Warp stride-32 loop: K/16 must divide evenly for balanced work (true for K=2048, 6144)
- `gemv_int8_warp` is the production path — other bench files may still use old `gemv_int8`

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

1. **FP32 text_generate broken** — `text_generate_fp32.cu` produces worse output than INT8. Separate issue (BF16 weight file convention or cuBLAS transpose).
2. **GEMM prefill correctness unverified** — no reference comparison. Timing-only validation.
3. **Inconsistent bench files** — Only `decode_int8_cgraph.cu` and `decode_full_int8.cu` use `gemv_int8_warp`. ~20 other bench files still use old `gemv_int8` or `gemv_fp32_int8_per_row`.
4. **text_generate head_norm bug** — Pre-existing. "FAIL head_norm l=0". In `text_generate.cu` path (uses `gemv_fp32_int8_per_row`, not `gemv_int8`).
5. **CUDA Graph correctness drift** — Per-kernel vs graph outputs diverge after 25 iterations (max diff ~4.0). Caused by FP4 quantization sensitivity. Not a correctness bug — synthetic all-ones input.

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
Test: `./bench/decode_int8_cgraph 28` (CUDA Graph production path), `./bench/text_generate ...` (correctness)
Verify: `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l` (expect 92)

---

## 10. Seed Principles

1. Smallest correct change. One kernel, one fix, one test.
2. Verify before broad edits.
3. Prefer repo evidence. Read code before assuming.
4. No churn.
5. Kernels first, framework later.
