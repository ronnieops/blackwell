# Research: Blackwell SM_120 (RTX 5060 Ti) FP4 GEMV Optimization

## Summary

FP4 GEMV on Blackwell is fundamentally memory-bound: 4× smaller weight reads dominate over compute. For down_proj (N=6144, K=2048), L2 cache residency and memory coalescing are critical. Current `gemv_fp4` with K=64 hardcoded achieves ~3–5 GB/s vs 448 GB/s peak—the gap is software, not hardware.

---

## Findings

### 1. GEMV Best Practices (Small M, Large N) Without TMA

- **Memory bandwidth dominates.** FP4 GEMV for LLM inference is memory-bound everywhere—the 4× reduction in weight reads is the win. [1](https://forums.developer.nvidia.com/t/custom-fp4-cuda-kernel-129-tflops-on-dgx-spark-with-pre-quantized-weight-cache/361600), [2](https://www.spheron.network/blog/fp4-quantization-blackwell-gpu-cost/)
- **GEMV ≠ GEMM.** GEMV patterns require different tiling: N-dimension (columns) is large and must tile for cache; M-dimension is small (batch=1 or few). Block-level GEMM tiling (16×16 WMMA tiles) is wrong for GEMV. [3](https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/)
- **Row-wise quantization is key.** Per-row scale factors must be loaded and applied. For down_proj (N=6144, K=2048), each row has K elements sharing one scale. Cache row-wise scales in registers. [3](https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/)
- **Register pressure vs ILP.** Amandeep's twelve-attempt journey found that aggressive register usage for K-dimension tiling failed; ILP with simpler loads worked better. [3](https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/)
- **Async loads critical for hiding latency.** cp.async (non-blocking) keeps memory pipeline full. Without it, loads stall compute. [4](https://sandyresearch.github.io/chipmunk-part-III/)

### 2. L2 Cache Optimization for N=6144 Down_proj

- **L2 persistence API available on Blackwell.** `cudaMallocAsync` + `cudaMemAdviseSetPreferredLocation` + `cudaMemAdviseSetAccessedBy` control L2 residency similar to Ampere. [5](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
- **N=6144 exceeds L2 working set.** RTX 5060 Ti L2 cache size not publicly confirmed; B200 has 126 MB partitioned L2. With K=2048 and FP4 weights (2 bytes per 2 values = 1 byte), one 6144×2048 matrix = ~12 MB. Multiple layers exceed L2. [6](https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut)
- **L1/Shared Memory split matters.** Blackwell has unified L1/shared. For GEMV, favor shared memory for weight tiling. Use `cudaFuncSetAttribute` with `cudaFuncAttributePreferredSharedMemoryCarveout`. [5](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
- **Prefetch weights into L2.** Load weight matrix once; reuse across batch. For inference with repeated token generation, weight L2 residency = huge speedup.
- **Reduced L1/shared memory concern on Blackwell.** Some reports indicate smaller effective L1; test different carveout values. [7](https://www.emergentmind.com/topics/nvidia-blackwell-gpus)

### 3. Published FP4 GEMV vs GEMM Benchmarks on Consumer Blackwell

- **DGX Spark (B200):** 129 TFLOPS achieved with pre-quantized weight cache—memory-bound regime. [1](https://forums.developer.nvidia.com/t/custom-fp4-cuda-kernel-129-tflops-on-dgx-spark-with-pre-quantized-weight-cache/361600)
- **RTX 5060 Ti 16 GB:** 448 GB/s GDDR7 memory bandwidth, 189.63 AI TOPS (INT8 dense). [8](https://gpupoet.com/gpu/learn/card/nvidia-geforce-rtx-5060-ti)
- **Memory-bound = compute inefficiency is fine.** When memory-bound, actual TFLOPS are irrelevant; bytes/second is the metric. FP4 wins because 4 elements fit in 1 byte vs 16 bytes for FP32.
- **GEMM severely underperforms GEMV in current impl.** Current `gemm_fp4_block_scaled` uses 16×16 WMMA tiles—wrong for prefill where M=seq_len. GEMV path exists but K=64 hardcoded. [AGENTS.md]
- **cuTe DSL now supports NVFP4 GEMV.** CUTLASS 4.4+ exposes NVFP4 grouped GEMM and GEMV via CuTe abstractions. [9](https://docs.nvidia.com/cutlass/4.4.2/overview.html)

### 4. cp.async vs Regular Loads for Memory-Bound Kernels

- **cp.async bypasses register file.** Direct GMEM→SMEM path, non-blocking. Critical for hiding memory latency. [10](https://research.meekolab.com/messing-around-with-gpus-again)
- **Retains 85–90% performance in sparse kernels.** Even with sparse patterns, async copy maintains high efficiency. [4](https://sandyresearch.github.io/chipmunk-part-III/)
- **Four-byte alignment required.** `cp.async.cg.shared.global` needs 4-byte aligned pointers. For FP4 (1 byte), must pack into 4-byte loads.
- **Latency hiding is the win.** While single load latency similar, async copy allows full pipeline utilization across loop iterations. For GEMV with K=2048, this is the difference between 3 GB/s and 300+ GB/s.
- **No TMA needed.** TMA (Hopper+) is more efficient but requires specific hardware. cp.async works on all modern NVIDIA GPUs including SM_120.

### 5. Warp-Level vs Block-Level GEMV Patterns

- **Blackwell MMA = tcgen05.mma, single-thread instruction.** Replaces Hopper warp-synchronous MMA. Each thread independently issues MMA ops. [11](https://arxiv.org/html/2512.02189v3)
- **WGMMA (Hopper) ≠ Blackwell MMA.** Blackwell dropped WGMMA warpgroup abstraction. Thread-level MMA is now the primitive. [11](https://arxiv.org/html/2512.02189v3)
- **For GEMV: warp-level reduction over N.** Each warp handles portion of N dimension. Warp-shuffle reductions for final accumulation. Register-based accumulation preferred over shared memory.
- **Block-level for large N tiling.** N=6144 exceeds warp capacity. Block-level tiling splits N into chunks; each block processes one output element. Shared memory for weight tiles.
- **Current impl uses wmma::mma_sync (Ampere-era).** `namespace wmma = nvcuda::wmma` with `fill_fragment`/`load_fragment`/`mma_sync`—works but not optimal for SM_120. WMMA supports FP16 accumulation only; FP4 requires FP16 accumulation + dequantization. [AGENTS.md]
- **Register tiling beats shared memory for small K.** For GEMV, K dimension (hidden_dim) is small enough to tile in registers. Shared memory bank conflicts + latency hurt more than help.

---

## Contradictions with Current Implementation

| Current Impl | Research Finding | Impact |
|---|---|---|
| `gemv_fp4` with K=64 hardcoded | Real layers need K-tiling over 2048 | Broken for production models |
| 16×16 WMMA tiles for GEMM | Wrong for prefill (large M) | GEMM severely underperforms |
| No cp.async in GEMV | cp.async critical for memory-bound | Memory pipeline not hidden |
| No L2 cache hints | L2 persistence = huge speedup | Weight reloaded each token |
| wmma::mma_sync (Ampere) | Blackwell has tcgen05.mma | Not using native SM_120 MMA |
| FP4 E2M1 with scale=absmax/3 | Standard NVFP4 uses MX format | May not match CUDA toolkit support |
| Single block for GEMV | Block-level N tiling needed for 6144 | Underutilizes GPU |

---

## Recommended Optimization Approaches

1. **GEMV: K-tiling over 2048.** Current K=64 → extend to 2048 with loop tiling. Each thread processes K/WARP_SIZE elements.
2. **cp.async pipeline.** Load next tile while computing current tile. Hide GMEM latency completely.
3. **L2 persistence for weights.** Use `cudaMallocAsync` + `cudaMemAdviseSetPreferredLocation` for weight matrix. Avoid reloading.
4. **Row-wise scale caching.** Scales are small (1 per K elements). Cache in registers; re-use across M dimension.
5. **Warp-level reduction.** After K-tiling, warp-shuffle reduction over N chunk. Avoid shared memory sync.
6. **SM_120 native MMA.** Replace `wmma::mma_sync` with inline PTX `mma.sync` targeting FP16 accumulation with FP4 A/B operands if available, or stick with WMMA FP16 for correctness.
7. **Shared memory carveout.** Set `cudaFuncAttributePreferredSharedMemoryCarveout` to maximum for weight caching.

---

## Sources

- Kept: [NVIDIA Forums - 129 TFLOPS DGX Spark FP4](https://forums.developer.nvidia.com/t/custom-fp4-cuda-kernel-129-tflops-on-dgx-spark-with-pre-quantized-weight-cache/361600) — memory-bound regime evidence
- Kept: [Amandeep Singh - Twelve Attempts at FP4 GEMV](https://amandeepsp.github.io/blog/nvfp4-blackwell-gemv/) — detailed GEMV optimization journey
- Kept: [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) — L2 persistence, shared memory split
- Kept: [ArXiv Microbenchmarking Blackwell](https://arxiv.org/html/2512.02189v3) — tcgen05.mma vs WGMMA, architecture details
- Kept: [ChipAndCheese - Blackwell Analysis](https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut) — L2 partition details
- Kept: [Modular - Matrix Multiplication Blackwell](https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-1-introduction) — GEMM patterns
- Kept: [TechPowerUp - RTX 5060 Ti Specs](https://www.techpowerup.com/gpu-specs/geforce-rtx-5060-ti-16-gb.c4292) — 448 GB/s bandwidth confirmation
- Dropped: YouTube reviews — no technical kernel details
- Dropped: Reddit benchmarks — anecdotal, no methodology

---

## Gaps

- RTX 5060 Ti specific L2 cache size not publicly documented (B200 = 126 MB, consumer may differ)
- SM_120 FP4 tensor core instruction set not fully documented in public PTX
- Current FP4 E2M1 implementation may not match NVIDIA's NVFP4 MX format specification
- No published RTX 5060 Ti FP4 GEMV benchmarks found

## Suggested Next Steps

1. Read `src/gemv_fp4.cu` and `src/gemm_fp4_block_scaled.cu` for current implementation details
2. Benchmark current `gemv_fp4` with K=2048 (real hidden_dim) vs K=64
3. Add cp.async pipeline to GEMV kernel
4. Test L2 persistence with `cudaMallocAsync` for weight matrix
5. Compare `wmma::mma_sync` vs raw PTX `mma.sync` on SM_120
