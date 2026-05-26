# Research: FP4 Block-Scaled GEMM Optimization for Blackwell SM_120 (RTX 5060 Ti)

## Summary

Current blackwell FP4 GEMM achieves only 3.6–5.3 GB/s (1% of 500 GB/s peak). Root causes: 16×16 CTA tile with 4 warps insufficient, no pipelined shared memory, no vectorized FP4→FP16 dequant, and no async copy (cp.async). Fix requires: 128×128 CTA tile with 8 warps, 2-stage cp.async pipeline for global→shared FP4 loads, vectorized uint4 loads with block-scale dequant in registers, and K-tiling with hidden_dim=2048→32×64-tile iterations. SM_120 lacks TMA/tcgen05, so warp-level `mma.sync` (16×16×64 fragments) + manual smem management is the only path.

---

## Findings

### 1. Tile Size Recommendations for WMMA on SM_120

SM_120 (RTX 5060 Ti) does **not** support tcgen05 (UMMA) or TMA — those require sm_120a / sm_100a. Only warp-level `nvcuda::wmma::mma_sync` is available, using `16×16×64` fragments with `__nv_fp4_e2m1`. Each warp computes one 16×16 output tile from 16×64 of A (FP4 dequant→FP16) × 64×16 of B (FP16).  Multiple fragments per warp and multiple warps per CTA scale up.

**Recommended CTA tile**: `128×128×64`. Decomposed as: 8 warps × 2 fragments/warp (M-dim) × 4 fragments/warp (N-dim) = 8×16=128 M, 4×16×2=128 N. Each warp: `mma_sync(aligned, 16, 16, 64)` called in loop.  CUTLASS SM120 block-scaled GEMM collective uses `(128,128,64)` or `(128,256,64)` tiles with warp specialization   [Source](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp)  [Source](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79b_blackwell_geforce_nvfp4_nvfp4_gemm.cu).

**Do not use 32×32 or 64×64 single-fragment tiles** — those waste tensor core utilization on SM_120.  The WMMA API only supports `16×16×64` as the largest fragment.  Scale CTAs to 128+ on M/N dims for good occupancy   [Source](https://nvidia-cutlass-22.mintlify.app/architectures/blackwell).

### 2. Shared Memory Tiling with Async Copy (cp.async)

SM_120 has ~228 KB shared memory per SM.  With `128×128×64` CTA tile and FP16 operands for WMMA:

- A smem (FP16): `128×64×2B = 16 KB`
- B smem (FP16): `64×128×2B = 16 KB`
- Total smem per tile: 32 KB (before pipelining)
- With 2-stage pipeline: `2 × 32 KB = 64 KB`
- With 3-stage: `96 KB` (possible but tight with registers)

**No TMA on SM_120**.  Use `cp.async` (PTX `cp.async.ca.shared.global.L4B` or `L8B`) for global→smem copies.  Pipeline pattern: prefetch next A/B tiles into smem pipeline stage while computing current stage.  Use `cp.async.commit_group` + `cp.async.wait_group` for synchronization   [Source](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)  [Source](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/collective/sm100_mma_cpasync_warpspecialized.hpp).

**Critical**: Use XOR swizzle (swizzle mode `\pkg` / `Swizzle<3,4,3>` in CUTLASS) on shared memory layout to avoid bank conflicts during WMMA loads.  Without swizzle, 16×64 tile loads from smem have 16-way bank conflicts, halving effective bandwidth   [Source](https://www.wingedge777.com/en/article/ba9e9d9171004edc).

### 3. Vectorized FP4→FP16 Dequant Loads

NVFP4 E2M1 has 16 values (4 bits each), packed 2 per byte.  Block scaling: 1 scale per group of 16 FP4 values along K-dim.

**Load pattern** — global memory:

1. Load 16-byte chunk from `A_fp4` global array → `uint4` vector (32 FP4 values or 2 groups of 16)
2. Load corresponding scale values from `A_scale` (2 × float)

**Dequant in registers** — not in shared memory:

```cu
// NVFP4 LUT: 16 possible values
__device__ constexpr float fp4_lut[16] = {
    0.25f, 0.5f, 1.0f, 2.0f,      // positive: 0b00xx
    -0.25f, -0.5f, -1.0f, -2.0f,  // negative: 0b01xx
    4.0f, -4.0f, 6.0f, -6.0f,     // extended: 0b10xx
    0.0f, 0.0f, 0.0f, 0.0f        // NaN/zero: 0b11xx
};

// Load 2 FP4 values per byte, unpack LUT indices
uint4 data = *reinterpret_cast<const uint4*>(&A_fp4[idx]);
// For each byte: low nibble = element[j], high nibble = element[j+1]
// Lookup fp4_lut[scale * nibble], store as half2
```

Write dequantized `half` values to smem for WMMA consumption   [Source](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/gemm/fp4_gemm_template_sm120.h)  [Source](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79b_blackwell_geforce_nvfp4_nvfp4_gemm.cu).

**Key insight**: Dequantize in registers, write FP16 to shared memory.  Do NOT store FP4 in smem and dequant on each WMMA read — that burns ALU on every fragment load.  Dequant once per global load   [Source](https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/).

**Vec4 bound**: Use `uint4` (16 bytes = 128 bits) vector loads for coalesced global reads.  This gives 32 FP4 values per load instruction, maximizing L2→register bandwidth.

### 4. K-loop Tiling for GEMV with hidden_dim=2048

Current GEMV is hardcoded to K=64.  Hidden_dim=2048 → need 32× K-accumulations per output element.

**Approach for batched GEMV** (multiple tokens or output rows):

1. CTA tile M-dim: 1 or 4 (batch tokens).  CTA processes `(M_out, N_out)` tile for each K-tile.
2. For each K-tile (64 K-elements): load `A[0:4, k:k+64]` FP4, dequant→FP16 in regs.  Load `B[k:k+64, 0:N]` FP16 from shared.
3. Accumulate in `float` registers: `C[m][n] += dot(A_tile[m][:], B_tile[:][n])`.
4. After all 32 K-tiles: write `C[4][N]` output.

**Registers**: 4 output rows × 128 N = 512 float accumulators.  Too many.  Instead: tile N-dim too.  Load B tile per iteration into smem, reuse for all M rows.  Process N in 16-element chunks (one warp's output).   [Source](https://veitner.bearblog.dev/nvfp4-gemv-improved/).

**Key optimization for GEMV**: Prefetch next K-tile via cp.async while computing current tile.  Overlap global load latency with dequant+dot product.  Two-stage pipeline: smem buf0 holds current B tile (FP16), cp.async loads next into buf1   [Source](https://github.com/gpu-mode/reference-kernels/blob/main/problems/nvidia/nvfp4_gemv/template_cute.py).

**Register budget**: 64 registers × 4 warps × 8 warps/CTA = plenty.  Each thread holds ~4 partial dot products.  Use `__syncwarp()` for warp-level reduction per output element.

### 5. Blackwell SM_120 Specific Optimizations (CUDA 12.8)

**Architecture constraints**:
- 36 SMs, ~500 GB/s GDDR7 bandwidth (RTX 5060 Ti)
- 228 KB shared memory per SM (unified L1/smem)
- 4 warp schedulers per SM, 128 CUDA cores per SM
- Max threads per SM: 2048 (same as Ada/Hopper)
- Max thread blocks per SM: limited by 64 registers/thread budget

**What's available**:
- `nvcuda::wmma::mma_sync` with `__nv_fp4_e2m1`: native FP4→FP16 tensor core path.  Hardware does FP4×FP16→FP32 accumulation   [Source](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).
- `cp.async`: available.  `cp.async.bulk`: likely missing on RTX 5060 Ti (GB206 die may skip TMA hardware).  Verify with `cuobjdump` on compiled binary.
- `__nv_fp4_e2m1`: sizeof=1 byte, 2 values per byte
- CUDA 12.8 provides `__nv_fp4_e2m1` type and `nvcuda::wmma` support for sm_120 (confirmed working)   [Source](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu).

**What's NOT available on SM_120 vs SM_100/SM_120a**:
- No `tcgen05.mma` (UMMA) — only warp-level `mma.sync`
- No TMA (Tensor Memory Accelerator) — no `cp.async.bulk.tensor`, no `tcgen05.ld/st`
- No thread block clusters (or limited) — likely no cluster launch control   [Source](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html).
- No `stmatrix` for epilogue stores — use normal `stg` with vectorized stores

**Occupancy targets**:
- Current kernel: 1 CTA/SM, 4 warps (128 threads), ~5% occupancy
- Target: 2–4 CTAs/SM, 8 warps/CTA (256 threads), 64+ regs/thread, 64–96 KB smem/CTA
- Expected occupancy: 50–75% (16–24 of 32 warps/SM active)

**Warp specialization** (CUTLASS pattern):  Separate producer warps (load + dequant) from consumer warps (WMMA + accumulate).  On SM_120 with no TMA, warp specialization reduces PCIe/minimum-issue overheads   [Source](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm100_gemm_cpasync_warpspecialized.hpp).

---

## Sources

### Kept
- **NVIDIA CUTLASS examples/79_blackwell_geforce_gemm/79b_blackwell_geforce_nvfp4_nvfp4_gemm.cu** — Official NVFP4×NVFP4 reference on SM_120.  Shows tile structure, block-scaling API, cp.async pipeline.  [Source](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79b_blackwell_geforce_nvfp4_nvfp4_gemm.cu)
- **CUTLASS sm120_blockscaled_mma_tma.hpp** — Collective-level TMA+block-scaled MMA for SM_120.  Tile sizes (128×128), pipeline stages, swizzle layout.  [Source](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp)
- **FlashInfer fp4_gemm_template_sm120.h** — Production FP4 GEMM template for SM_120.  Group scaling layout, dequant pattern, K=64 block scaling.  [Source](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/gemm/fp4_gemm_template_sm120.h)
- **"Twelve Attempts at an FP4 Kernel" (Amandeep Singh)** — Detailed walkthrough of GEMV/GEMM optimization tries.  Covers smem dequant, scaling, cp.async, K-tiling, warp specialization.  [Source](https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/)
- **NVIDIA Blackwell Tuning Guide** — SM occupancy, shared memory sizes (228 KB), GDDR7 tuning, cp.async guidance.  [Source](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
- **NVIDIA RTX Blackwell GPU Architecture PDF** — Official FP4 tensor core support, SM block diagram, 5th-gen tensor core specs.  [Source](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
- **CUTLASS sm100_mma_cpasync_warpspecialized.hpp** — Producer/consumer warp specialization pattern usable on SM_120.  [Source](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm100_gemm_cpasync_warpspecialized.hpp)
- **"NVFP4 GEMV improved" (Simon Veitner)** — Batched GEMV with K-split, parallel reduction over K-dim, prefetching.  [Source](https://veitner.bearblog.dev/nvfp4-gemv-improved/)
- **CUTLASS CuTeDSL dense_blockscaled_gemm_persistent.py** — Persistent kernel pattern for small-M (GEMV-like) block-scaled GEMM on SM_120.  [Source](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py)

### Dropped
- CUTLASS SM100-only examples (sm_100a TMA/UMMA patterns) — TMA not available on SM_120 RTX 5060 Ti
- Generic Hopper WGMMA references — WGMMA is sm_90, not applicable to sm_120
- CICC reverse engineering refs — Interesting but secondary to official NVIDIA sources
- PrimitiveContext/blackwell — No source code, configuration-only

---

## Gaps

1. **TMA support on GB206 (RTX 5060 Ti)**: Not confirmed whether GB206 die includes TMA unit.  RTX 5090 (GB202) has TMA.  RTX 5060 Ti may not.  Verify with `deviceQuery` / `cuobjdump -ptx` on compiled kernel.  If TMA available, switch to `cp.async.bulk.tensor` for ~20% better global→smem throughput.

2. **WMMA→UMMA fallback**: SM_120 may support a subset of tcgen05 beyond warp-level mma.sync.  Not documented in CUDA 12.8 headers.  Check `cuda_fp4.h` for `__umma` instructions.  If available, 64×64 tiles become possible, doubling arithmetic intensity.

3. **Best register/shared memory split**: 228 KB unified.  Optimal split between dequant-LUT-smem, pipeline smem, and register file not profiled.  Use occupancy API (`cudaOccupancyMaxPotentialBlockSize`) to find sweet spot.

4. **Warp count×tile size Pareto frontier**: RTX 5060 Ti has 36 SMs.  At 8 warps/CTA with 128×128 tile, only 4 CTAs fit per SM → 144 total tiles in flight.  For M=512, only 4× parallelism hiding latency poorly.  Tile down or use persistent kernel for better occupancy.

5. **Batched GEMV synergy**: Qwen3-1.7B has hidden_dim=2048.  Decode phase is GEMV-bound.  If batch size > 1, this becomes batched GEMV (small M GEMM).  Optimization immediately shifts toward memory-bound→compute-bound transition analysis at M=4,16,32.

---

## Next Steps

1. **Rewrite GEMM kernel with 128×128×64 CTA, 8 warps, cp.async 2-stage pipeline, vectorized uint4 FP4 loads, register dequant→FP16 smem store**
2. **Rewrite GEMV with K=64 tiling loop, hidden_dim=2048 (32 iterations), prefetch next K-tile via cp.async while accumulating**
3. **Benchmark each change incrementally — measure GB/s after each step**
4. **If TMA verified on RTX 5060 Ti, adopt `cp.async.bulk` for smem loads → ~15-25% bandwidth improvement**
5. **Consider CUTLASS integration: replace hand-rolled GEMM with CUTLASS 3.8 sm120_blockscaled kernel template (less code, more tested)**
