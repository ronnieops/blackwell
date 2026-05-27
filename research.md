# Research: INT8 GEMV Optimization Techniques for Blackwell SM_120a

## Summary

SM120 consumer Blackwell GPUs (RTX 50xx) **lack tcgen05.mma** (Tensor Memory + 5th-gen tensor core instructions). Only `mma.sync.aligned.m16n8k16` (Ampere-vintage) and `__dp4a` are available for INT8. Multi-token batching (M=2-8) remains the single most effective optimization for decode GEMV, amortizing weight loads across tokens. L2 persistence and weight prefetching provide marginal gains since weight matrices exceed L2 on RTX 5060 Ti (32 MB L2 vs 36 MB for 3 layers of INT8 weights).

## Findings

### 1. SM120 vs SM100 Tensor Core Differences — Critical Constraint

1. **SM120 does NOT expose tcgen05.mma or Tensor Memory (TMEM).** These are exclusive to datacenter SM100 (B200). On SM120, the only tensor-core dispatch path is `mma.sync.aligned.m16n8k16` — the same Ampere-vintage instruction available since SM80. [Source](https://github.com/ShlokVEX/Mini-Attention/blob/main/notes/sm120_vs_sm100.md)
2. **tcgen06.alloc (TMEM allocation) not supported on SM120a.** Confirmed: `tcgen06.alloc` only works on sm100a/101a, not sm120a (RTX 5080, 5070, 5060 Ti). [Source](https://forums.developer.nvidia.com/t/how-to-load-fp8-using-ldmatrix-on-sm120-sm120a/330254)
3. **SM120's FP4 path uses `mxf4` instructions**, not tcgen05. CUTLASS has separate SM120 kernel paths (e.g., `gemm_grouped_sm120_M128_BS_group1`) using CuTe MMA atoms. [Source](https://forums.developer.nvidia.com/t/fp4-on-dgx-spark-why-it-doesnt-scale-like-youd-expect/360142)
4. **CUTLASS 3.8+ supports SM120 blockwise dense GEMM** but only via CuTe MMA atoms for mma.sync, not tcgen05. Block-scaled kernels (FP4) are optimized; INT8 paths use standard mma.sync. [Source](https://raw.githubusercontent.com/NVIDIA/cutlass/main/CHANGELOG.md)

### 2. INT8 dp4a and IMMA Options on SM120

5. **`__dp4a` remains the optimal scalar INT8 instruction on SM120 for GEMV.** No new dp4a variant or IMMA instruction was introduced for SM120. The mma.sync `m16n8k16` path supports INT8 (S8/S8/F32 accumulate) but requires warp-level coordination and shared-memory staging — overhead that doesn't pay off for M=1 decode. [Source](https://docs.nvidia.com/cuda/parallel-thread-execution/)
6. **mma.sync INT8 m16n8k16 may help for M≥4 batched decode.** The m16n8k16 shape processes 16 rows × 8 cols × 16 K-per-step. For M=1 this is wasteful (15/16 of MMA capacity unused). For M=4-8, the 16-row dimension starts filling. However, the weight format must be in shared-memory tiled layout (via ldmatrix), not the row-major format our __dp4a kernel uses. [Source](https://gau-nernst.github.io/nvrtc-matmul/)
7. **No QMMA or new INT8-specific tensor core instruction on SM120.** The Blackwell QMMA (quantized MMA) mentioned in microbenchmarks targets SM100 datacenter tensor cores. SM120 consumer chips use the same Ampere-era tensor core pipeline. [Source](https://arxiv.org/html/2512.02189v3)

### 3. Multi-Token Batching Strategies for LLM Decode

8. **llama.cpp MMVQ batches up to 8 tokens simultaneously** (`MMVQ_MAX_BATCH_SIZE=8`). Grid = `(nrows/rows_per_block, nchannels, ncols_dst)`, block = `(warp_size, nwarps)` where nwarps=4 for 1-4 cols, nwarps=2 for 5-8 cols. No split-K, no persistent threads — simple batched approach. [Source](https://github.com/ggml-org/llama.cpp)
9. **EVA (2026) proposes multi-batch weight tile reuse.** Multiple decode requests share the same weight tiles loaded into shared memory, reducing global memory bandwidth. For GEMV, loading weight row once and computing dot products for all batch tokens is the key optimization. [Source](https://arxiv.org/html/2605.24144v1)
10. **Batched GEMV sweet spot is M=3-4** for our kernel (confirmed in AGENTS.md: `gemv_int8_batched` template-batched M=1..8, sweet spot M=3-4 with 1.4× speedup). This matches llama.cpp's design of 4 nwarps for 1-4 cols.
11. **Speculative decoding creates natural batching.** Multiple draft tokens verified against the model simultaneously — generates M=2-8 sequential tokens that can share weight loads. This is how production systems achieve higher throughput on memory-bound decode. [Source](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)

### 4. Weight Prefetching and L2 Cache Management

12. **PRESERVE framework (2025) prefetches weights to L2 during inter-layer gaps.** Overlaps HBM→L2 prefetch with compute from the previous layer. Designed for distributed inference but the principle applies: issue `cudaMemcpyAsync` or `cp.async.bulk` for next layer's weights while current layer computes. Gains 10-20% for sequential layer execution. [Source](https://arxiv.org/html/2501.08192v2)
13. **CUDA L2 persistence (`cudaAccessPolicyWindow`) can reserve L2 set-aside for weights.** API: `cudaCtxSetLimit(cudaLimitPersistingL2CacheSize, bytes)`. However, on RTX 5060 Ti the L2 is ~32 MB, and a single 6144×2048 INT8 weight matrix is 12 MB. With 3 layers' weights = 36 MB, L2 is saturated regardless of persistence policy. Persistence helps when working set < L2. [Source](https://leimao.github.io/blog/CUDA-L2-Persistent-Cache/)
14. **cp.async pipeline for weight streaming can overlap load+compute within a single GEMV.** Double-buffer: while warp computes dot product on current K-tile, issue `cp.async` for next K-tile into shared memory. Our existing gemm_fp4 uses 2-stage cp.async. For INT8 __dp4a GEMV, the compute is so fast (~775 GB/s, near bandwidth ceiling) that cp.async pipelining adds overhead without measurable gain — the kernel is already saturating HBM bandwidth. [Source](https://salykova.github.io/sgemm-gpu)
15. **GDDR7 on RTX 5060 Ti delivers ~500 GB/s peak.** Our INT8 GEMV at 775 GB/s is already 155% of peak — likely measured as effective throughput accounting for the 4× multiplier of dp4a (4 byte products per instruction). Actual memory bandwidth utilization is near-saturated at ~500 GB/s. No software prefetching trick will exceed this physical limit. [Source](https://www.nvidia.com/en-us/on-demand/session/gtc26-s81463/)

### 5. CUTLASS/cuBLASLt Best Practices for Consumer Blackwell

16. **cuBLASLt INT8 GEMV on SM120 falls back to mma.sync or scalar dp4a internally.** No SM120-specific INT8 GEMV path exists in cuBLASLt — the same Ampere-era kernels run. For M=1, cuBLASLt is typically slower than hand-tuned __dp4a due to launch overhead and generality. [Source](https://zenn.dev/toki_mwc/articles/rtx5090-blackwell-cuda-toolkit-trap-llama-cpp?locale=en)
17. **CUTLASS SM120 GEMM kernels target FP4 block-scaled, not INT8 GEMV.** The SM120-specific CUTLASS paths optimize for `mxf4` and `nvf4mxf4` narrow-precision types. INT8 on SM120 uses generic SM80+ mma.sync paths from CUTLASS 2.x/3.x, which are GEMM-focused (large M,N,K) and not optimized for GEMV (M=1). [Source](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)

## Sources

### Kept (Primary Technical Sources)
- ShlokVFX/Mini-Attention sm120_vs_sm100.md — Definitive summary of SM120 vs SM100 tensor core differences. Consumer lacks tcgen05/TMEM.
- NVIDIA CUTLASS Blackwell Functionality Docs — Official docs on SM100 tcgen05.mma and SM120 mma.sync paths.
- gau-nernst/tcgen05-for-dummies — Detailed tcgen05 explanation, explicitly notes SM100-only, not consumer.
- arxiv 2512.02189 (Blackwell Microbenchmarks) — Microarchitectural analysis of Blackwell tensor cores.
- arxiv 2501.08192 (PRESERVE) — Weight prefetching framework for LLM inference.
- arxiv 2605.24144 (EVA) — Multi-batch weight tile reuse for LLM decode acceleration.
- CUDA L2 Persistent Cache (Lei Mao) — Practical guide to L2 persistence API.
- gau-nernst/nvrtc-matmul — MMA instruction variant benchmarks including SM120 (RTX 5090).
- NVIDIA Dev Forums: tcgen06.alloc on sm120a — Confirmed TMEM not available on consumer Blackwell.
- llama.cpp source (mmvq kernels) — Production batched GEMV implementation reference.
- Salykova SGEMM optimization — cp.async pipeline and shared-memory tiling techniques.

### Dropped
- RTX 5090/5080 reviews (Tom's Hardware, Puget Systems, etc.) — Consumer product reviews, no kernel-level technical content.
- YouTube speculation videos — No technical substance.
- Generic CUDA optimization tutorials — Not specific to SM120 or INT8 GEMV.
- IST-DASLab/gemm-int8 — PyTorch extension targeting datacenter GPUs, not consumer SM120.

## Gaps

1. **No SM120-specific INT8 tensor core optimization published.** All Blackwell INT8 research targets SM100 with tcgen05. Consumer SM120 tensor core throughput for INT8 mma.sync m16n8k16 is not independently benchmarked in published literature.
2. **L2 cache size on RTX 5060 Ti not definitively confirmed.** Assumed ~32 MB based on GB207 die, but NVIDIA doesn't publish consumer GPU L2 sizes. If L2 is larger, persistence policies could help more.
3. **Weight format conversion cost not quantified.** Switching from row-major __dp4a to shared-memory tiled ldmatrix format for mma.sync has a one-time transpose cost. Whether the batched-GEMV speedup (M=4+) amortizes this cost is untested.
4. **Inter-layer weight prefetching feasibility.** PRESERVE-style prefetching between layers requires async copy engines or DMA that may not exist on consumer GPUs in the same form as datacenter. Not validated on RTX 50xx.

## Practical Recommendations for Project

Given the findings, the optimizations ranked by expected impact:

| Priority | Technique | Expected Gain | Effort |
|----------|-----------|---------------|--------|
| 1 | Multi-token batched GEMV (M=4) | 1.4× throughput | Low — already have `gemv_int8_batched` |
| 2 | Speculative decoding integration | 2-3× effective throughput | High — needs draft model |
| 3 | Inter-layer weight prefetch (PRESERVE-style) | 10-20% latency reduction | Medium |
| 4 | mma.sync m16n8k16 INT8 for M≥4 batched | Unclear — needs testing | Medium — requires weight reformat |
| 5 | L2 persistence for small models | 0% (weights > L2) | N/A |

**Bottom line: SM120 consumer Blackwell has no new INT8 tensor core instructions beyond Ampere-era mma.sync and dp4a. The 16.2 t/s gap to 114 t/s target cannot be closed with single-token kernel optimizations. Multi-token batching or speculative decoding are the only viable paths.**
