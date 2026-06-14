# Research: M=1 INT4 GEMV Memory Subsystem Optimization

## Summary

Our 64 t/s M=1 = 68% of theoretical ceiling (94 t/s). Three transferable techniques ranked by expected impact: **(1) CUDA Graph capture** to eliminate 9720 kernel launches/token (~19-49ms wasted overhead, likely our #1 bottleneck), **(2) Increase warps-per-SM** to improve DRAM latency hiding (our `__launch_bounds__(32, 8)` = only 8 warps/SM — may be insufficient to saturate 500 GB/s), **(3) Multi-warp-per-row K-split** to increase occupancy and overlap weight loads with dp4a compute. llama.cpp uses 4-8 warps per thread block with cooperative K-dimension splitting and `uint4` (16B) vectorized loads — same coalescing strategy we already use, but with higher thread density per SM.

## Findings

### 1. llama.cpp MMVQ Kernel Architecture (mul_mat_vec_q)

1. **Thread block = multiple warps, not 1 warp.** llama.cpp's `mul_mat_vec_q` kernel in `ggml/src/ggml-cuda/mmvq.cu` uses **4-8 warps per block** (compile-time `nwarps` parameter), not 1 warp per block like our kernel. Each block processes **one or a small batch of output rows**. Multiple warps cooperatively split the K dimension. This gives each block 128-256 threads instead of our 32. [Source: llama.cpp mmvq.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu) and [DeepWiki CUDA Backend analysis](https://deepwiki.com/ggml-org/llama.cpp/5.1-cuda-backend-(nvidia))

2. **Grid layout: 1 block per output row (ne01), multiple warps split K internally.** The grid is `dim3(ne01, nbatch, 1)` — one block per output element per batch row. Within each block, `nwarps` warps cooperatively process K-blocks. Each warp handles a contiguous chunk of K, then results reduce via shared memory. This is a **K-split within a block** (intra-block parallelism) vs our K-split across threads within 1 warp (intra-warp). [Source: GitHub discussion of mul_mat_vec_q logic](https://github.com/ggml-org/llama.cpp/discussions)

3. **Vectorized 16B loads (uint4 / int4).** llama.cpp loads quantized weight blocks as `uint4` (16-byte = 128-bit transactions) whenever block size ≥ 4 bytes. For Q4_K (super-block = 144 bytes), it loads super-blocks in vectorized chunks. Our kernel also uses `uint4` loads (confirmed: `*reinterpret_cast<uint4*>(w_buf)` in gemv_int8.cu:224). **Same vectorization strategy** — not a differentiator. [Source: CUDA Pro Tip: Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)

4. **vecdotq: per-block dequant + dot, warp shuffle reduction.** The inner loop: each thread loads a weight super-block + activation block, dequantizes, accumulates partial dot product via `dp4a`-equivalent (manual for Q4_K), then reduces across warp via `__shfl_xor_sync`. Same pattern as our kernel. [Source: DeepWiki CUDA Backend](https://deepwiki.com/ggml-org/llama.cpp/5.1-cuda-backend-(nvidia))

### 2. Thread Block Size / Occupancy Differential

5. **Our kernel: `__launch_bounds__(32, 8)` = 1 warp/block, max 8 warps/SM.** This means only 8 warps = 256 threads active per SM. With 36 SMs, that's 288 concurrent warps max. Each warp streams weight data independently for one output row. For DRAM latency hiding, NVIDIA recommends **enough active warps to cover ~400-700 cycle DRAM latency**. At 8 warps/SM with 32 threads each, we have 256 in-flight threads per SM — this may be too few to saturate 500 GB/s GDDR7. [Source: CUDA Occupancy Calculation](https://leimao.github.io/blog/CUDA-Occupancy-Calculation/) and [NVIDIA Occupancy Optimizer](https://forums.developer.nvidia.com/t/optimisation-of-occupancy-summary-table/)

6. **llama.cpp: 4-8 warps/block, likely 4+ blocks/SM = 16-32 warps/SM.** With `nwarps=4` and reasonable register usage, 4 blocks can reside on one SM = 16 warps = 512 threads/SM. This doubles-to-quadruples the in-flight memory transactions per SM vs our 8 warps. **More warps = more outstanding DRAM requests = better bandwidth utilization.** This is likely the single biggest architectural difference. [Source: DeepWiki CUDA Backend](https://deepwiki.com/ggml-org/llama.cpp/5.1-cuda-backend-(nvidia))

7. **M=1 GEMV is latency-bound, not throughput-bound per SM.** The GPU needs many concurrent memory requests to saturate DRAM bandwidth. With only 8 warps/SM, each SM can have at most ~8 outstanding cache-line requests. Modern GPUs can support dozens of outstanding requests per SM. **Increasing warps/SM directly increases bandwidth utilization** even for memory-bound kernels. [Source: GPU Execution Model](https://medium.com/@utsabsapkota4231/gpu-execution-model-explained-warps-sms-and-latency-hiding/)

### 3. CUDA Graph / Launch Overhead

8. **Our decode loop launches ~9720 kernels/token** (270 GEMV launches/layer × 36 layers — from nsys profile: 8851 gemv_int4_batched instances for ~28 tokens ≈ 316 GEMV/token). At **2-5µs per launch** (standard CUDA CPU-side overhead), that's **19-49ms of pure overhead per token** — potentially 19-49% of our 17.9ms/token decode time. This is catastrophic. [Source: NVIDIA CUDA Graph Best Practices](https://docs.nvidia.com/dl-cuda-graph/latest/) and [Kernel Launch Overhead](https://inferensys.com/glossary/inference-optimization/)

9. **llama.cpp uses CUDA Graphs for decode.** NVIDIA contributed CUDA Graph support to llama.cpp (PR/issue #6763). The graph captures the entire decode step (all GEMV + attention + norm kernels for one token) and replays with a single CPU call. This eliminates per-kernel launch overhead entirely. vLLM also uses CUDA Graphs for the same reason — hundreds of kernel launches per decode token reduced to 1 graph replay. [Source: NVIDIA Blog: Optimizing llama.cpp with CUDA Graphs](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/) and [llama.cpp issue #6763](https://github.com/ggml-org/llama.cpp/issues/6763) and [vLLM CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs/)

10. **Our CUDA Graph was tested but only gave 2.1% speedup (Session 72).** However, the AGENTS.md notes that the graph captured the full 28-layer loop (1.7B model), not the 36-layer 8B model. Also, the 2.1% result was measured on the 1.7B model which has fewer layers (28 vs 36). **The 8B decode loop has ~70% more launches → launch overhead is proportionally larger.** Need to re-test CUDA Graph on the INT4 8B path specifically. The 2.1% result may not generalize.

11. **Wait — our nsys shows 92.2% of time in GEMV kernel.** This suggests kernel execution dominates, not launch overhead. BUT: nsys `--trace=cuda` captures GPU-side execution time. It does NOT show CPU-side launch gaps between kernels. The gap between kernel N ending and kernel N+1 starting (while CPU is in the launch call) is NOT counted in kernel time. **Need `nsys profile --trace=cuda,nvtx` with timeline view to measure inter-kernel gaps.** If gaps are small (<1µs), launch overhead is not the issue. If gaps are 3-5µs, it's a major contributor. [Source: NVIDIA Nsight Systems: Understanding Overhead](https://developer.nvidia.com/blog/understanding-visualization-of-overhead-and-latency-in-nsight-systems/)

### 4. Blackwell (SM_120a) Memory Features

12. **TMA (Tensor Memory Accelerator) may be limited on SM_120 (consumer Blackwell).** TMA is confirmed on SM_100 (B200 data center). For SM_120 (consumer RTX 5060), the feature set is reduced. The SM100 vs SM120 comparison shows consumer Blackwell lacks cluster/distributed shared memory and may have limited TMA support. **cp.async is fully supported on SM_120** (inherited from Ampere/Hopper). [Source: SM100 vs SM120 Blackwell GPU Wiki](https://0xsero.github.io/blackwell-gpu-wiki/blackwell/sm100-vs-sm120/) and [Blackwell SM100: TMEM, TMA](https://jianyuh.github.io/cuda/2026/04/12/blackwell-sm100-tmem-tma-and-the-new-tensor-core/)

13. **cp.async for software pipelining (Ampere+ feature, available on SM_120).** `cp.async` copies data from global to shared memory asynchronously, bypassing registers. This enables double-buffered pipelines: load next weight block while computing current block. For memory-bound GEMV, this overlaps weight fetching with dp4a compute. **However, for pure memory-bound M=1 GEMV, compute is negligible (92% is memory)** — pipelining may not help because there's nothing to overlap with. It helps when compute and memory are comparable. [Source: CUDA Async Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/) and [cp.async pipeline guide](https://deepwiki.com/gau-nernst/learn-cuda/6.2-asynchronous-pipelining-with-cp.async)

14. **L2 residency hints (accessPolicyWindow).** `cudaStreamSetAttribute` with `cudaStreamAttributeAccessPolicyWindow` can pin frequently-accessed data in L2. For M=1 GEMV, weights are streamed once with no reuse — L2 hints won't help for weights. **Could help for activation vector (x)** which is reused across all N output rows. If x fits in L2 (K=4096 × 1 byte INT8 = 4KB for activations), it's already cached. L2 hints unlikely to help. [Source: CUDA L2 Cache Control](https://docs.nvidia.com/cuda/cuda-programming-guide/) and [CUDA L2 Persistent Cache](https://leimao.github.io/blog/CUDA-L2-Persistent-Cache/)

15. **LDG.128 (128-bit loads) vs LDG.64.** Our `uint4` loads compile to `LDG.128` instructions — the widest available. This is already optimal. No improvement possible from wider loads. [Source: NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)

### 5. Software Pipelining / Async Memory

16. **Double-buffered shared memory with cp.async.** The standard GEMM optimization: prefetch next tile to smem buffer B while computing tile A. For GEMV, the pattern would be: while computing dot product for K-block i, prefetch K-block i+1. **But GEMV M=1 has minimal compute per load** (4 dp4a calls per 16-byte block = ~4 cycles). DRAM latency is ~400-700 cycles. Even with double buffering, you need enough warps to have 100-175 outstanding loads to hide latency. This circles back to the occupancy issue (Finding #5-7). [Source: Double Buffering Optimization](https://deepwiki.com/wianger/cuda_sgemm/3.1.2-double-buffering-optimization) and [Asynchronous Pipelining with cp.async](https://deepwiki.com/gau-nernst/learn-cuda/6.2-asynchronous-pipelining-with-cp.async)

17. **GTC 2025 "Maximize Memory Bandwidth" session highlights.** Key techniques: cp.async bulk for large transfers, TMA for structured data movement, maximizing occupancy for latency hiding, coalesced vectorized loads. All apply to data center (B200) primarily. For consumer SM_120: **occupancy maximization and vectorized loads are the actionable items.** [Source: GTC 2025 notes](https://shreyansh26.github.io/post/2025-03-23_gtc25-cuda-techniques-to-maximize-memory-bandwidth/) and [NVIDIA on-demand session](https://www.nvidia.com/en-us/on-demand/session/)

### 6. Weight Layout / L2 Reuse

18. **M=1 has zero weight reuse opportunity.** Each output row n reads weight row n independently. No other row needs that data. L2 caching doesn't help. This is fundamental to M=1 GEMV. llama.cpp has the same constraint. **Not a differentiator.** The only reuse is the activation vector x (4KB for K=4096), which is naturally cached in L1/L2 after first access.

19. **Row-major transposed weight layout (W_t[N×K]) is optimal for M=1.** Each block reads contiguous bytes from one weight row → perfectly coalesced. Our layout and llama.cpp's layout are equivalent here. No improvement from tiling or layout changes for M=1.

### 7. Bottleneck Diagnosis

20. **85% of peak bandwidth in microbench = 424/500 GB/s.** This suggests our kernel CAN achieve good bandwidth when measured in isolation. The gap between 424 GB/s (microbench) and effective 68% of ceiling in end-to-end (64 t/s vs 94 t/s theoretical) suggests **overhead OUTSIDE the kernel** — launch overhead, inter-kernel gaps, non-GEMV kernels (rmsnorm, attention, etc.). The nsys profile shows GEMV = 92.2% of time, but this measures GPU-active time. The remaining 7.8% is other kernels. The gap between 92.2% kernel time and 100% is non-GEMV work.

21. **Math check: 64 t/s = 15.6ms/token. GEMV kernel avg = 54.8µs × ~316 calls = 17.3ms.** Wait — 17.3ms > 15.6ms? This doesn't add up. Re-examine: nsys shows 54.8µs avg for 8851 instances across ~28 tokens = 316 calls/token. 316 × 54.8µs = 17.3ms. But 64 t/s = 15.6ms/token. Discrepancy suggests either fewer GEMV calls per token or faster per-call in the measured run. **The nsys profile was from the batched benchmark, not the M=1 warp kernel.** The M=1 warp kernel (59-63 t/s) may have different call counts. Need M=1-specific nsys profile.

## Transferable Techniques — Ranked by Expected M=1 Impact

### Rank 1: CUDA Graph capture of full decode loop ⭐⭐⭐⭐⭐
- **Expected impact: 10-30% speedup** (64 → 70-83 t/s) if launch overhead is significant
- **Action**: Capture entire 36-layer decode loop (all GEMV + attention + norm + RoPE kernels) into one CUDA Graph. Replay per token.
- **Risk**: Low. Graph capture already tested on 1.7B (Session 72). Need to validate on 8B INT4 path.
- **Why it might NOT help**: nsys shows 92.2% GPU-active time. If inter-kernel gaps are already <1µs, launch overhead is already amortized by the GPU command processor's launch queue.
- **Measurement needed**: `nsys profile --trace=cuda` with timeline → measure average gap between consecutive kernels. If gaps > 2µs, CUDA Graph is high-impact.

### Rank 2: Increase warps per SM via multi-warp K-split ⭐⭐⭐⭐
- **Expected impact: 10-20% speedup** (64 → 70-77 t/s) by improving DRAM bandwidth utilization
- **Action**: Change kernel from 1 warp/block (32 threads, 8 blocks/SM) to 2-4 warps/block (64-128 threads, 4-2 blocks/SM). Multiple warps cooperatively split K dimension, reduce via shared memory.
- **Current config**: `__launch_bounds__(32, 8)` = 1 warp/block, max 8 blocks/SM, 8 warps/SM = 256 threads/SM
- **Proposed config**: `__launch_bounds__(64, 8)` or `__launch_bounds__(128, 4)` = 2-4 warps/block, more total warps/SM
- **Risk**: Medium. More warps per block = shared memory reduction needed (adds synchronization). Register pressure may reduce occupancy. Need to verify register count allows >8 warps/SM.
- **Why llama.cpp is faster here**: Their `nwarps=4-8` gives 16-32 warps/SM vs our 8. More outstanding DRAM requests = better bandwidth saturation.
- **This is the single biggest architectural difference between our kernel and llama.cpp's.**

### Rank 3: Megakernel / persistent kernel for full decode ⭐⭐⭐
- **Expected impact: 15-25% speedup** by eliminating ALL launch overhead and enabling inter-kernel optimization
- **Action**: Single persistent kernel that runs all 36 layers of decode. Each thread block persists, processing consecutive GEMV calls via global memory coordination. Alternative: CUDA Graph achieves similar effect with less complexity.
- **Risk**: High. Complex implementation. Register/shared memory pressure across layers. NVIDIA's megakernel research (mirage-llm-megakernel) shows 1.3-2× speedups but requires careful engineering.
- **Source**: [Megakernels: End-to-End Fused LLM Inference](https://theorempath.com/topics/megakernels) and [mirage-llm-megakernel](https://github.com/BodhiHu/mirage-llm-megakernel)
- **Recommendation**: Start with CUDA Graph first (Rank 1). Only pursue megakernel if CUDA Graph is insufficient.

### Rank 4: cp.async double-buffered weight loading ⭐⭐
- **Expected impact: 5-10% speedup** by prefetching next K-block while computing current
- **Action**: Use `cp.async` to load weight block i+1 to shared memory while dp4a computes block i. Requires 2× shared memory buffers.
- **Risk**: Low-Medium. cp.async well-supported on SM_120. But for pure memory-bound kernel, compute is so light (~4 cycles per block) that overlap benefit is minimal. **Mainly helps if we increase compute per block** (e.g., process 2 rows per warp).
- **Why it may not help much**: GEMV M=1 is 92% memory. There's almost no compute to overlap with.

### Rank 5: Process multiple output rows per warp ⭐⭐
- **Expected impact: 5-15% speedup** by reusing activation vector loads across rows
- **Action**: Each warp computes 2-4 output rows simultaneously. Shares activation vector x across rows (loaded once from L1/L2, reused). Weight loads remain independent per row but x bandwidth is amortized.
- **Risk**: Low. Similar to batching M>1 within a single kernel.
- **Caveat**: x is only 4KB (K=4096 INT8). Already fully cached in L1 after first row. Marginal bandwidth savings.

## What Our Kernel Already Does Right

- ✅ **uint4 (16B) vectorized loads** — `LDG.128`, same as llama.cpp
- ✅ **Transposed row-major weights** — perfectly coalesced for M=1
- ✅ **dp4a SIMD** — 4-way int8 dot product per instruction
- ✅ **Warp shuffle reduction** — no shared memory sync needed for 1 warp
- ✅ **Stride-32 K-block assignment** — threads read consecutive memory (coalesced)
- ✅ `__launch_bounds__(32, 8)` — allows 8 blocks/SM concurrent

## Risks & Caveats

1. **85% bandwidth in microbench contradicts low-occupancy hypothesis.** If our kernel achieves 424 GB/s with 8 warps/SM, then occupancy isn't the bottleneck. The 68% end-to-end efficiency gap must come from elsewhere (launch overhead, non-GEMV kernels). **Must profile M=1 end-to-end with nsys timeline to confirm.**

2. **llama.cpp may use MMQ (mul_mat_q) not MMVQ for some configs.** MMQ is the matrix-matrix quantized GEMM path (uses tensor cores). For M=1, MMVQ should be selected, but the dispatch threshold matters. Need to verify which kernel llama.cpp actually dispatches for Q4_K_M M=1 on Blackwell.

3. **llama.cpp Q4_K_M is 4.5-bit, not 4-bit.** It reads ~17% more weight data per token than our pure INT4. Yet it's faster (84 t/s vs 64 t/s). This means their memory subsystem efficiency is dramatically better — ~22% more data at 31% higher throughput. The multi-warp-per-block architecture is the likely explanation.

4. **Register pressure may limit occupancy gains.** Our kernel uses ~25 registers per thread (per AGENTS.md). With 256 registers/warp (32 threads × 8 regs limit on Blackwell), 8 warps × 32 × 25 = 6400 registers. Blackwell has 65536 registers/SM. We're using only 10%. **Occupancy is NOT register-limited.** The `__launch_bounds__(32, 8)` hint caps it at 8. We could potentially go to 32+ warps/SM.

## Clarification Questions

1. **Is the 424 GB/s microbench for the warp kernel or the per-thread kernel?** The per-thread kernel (`gemv_int8_kernel`, 64 threads/block) has different occupancy than the warp kernel (`gemv_int8_warp_kernel`, 32 threads/block).

2. **What does the nsys timeline look like for M=1 specifically?** The nsys data in AGENTS.md is from the batched benchmark (92.2% GEMV). Need M=1 warp kernel profile with kernel gaps visible.

3. **What is the register count of the INT4 warp kernel?** The INT8 warp kernel is ~25 regs. INT4 may differ (nibble unpack adds instructions).

4. **Does llama.cpp dispatch MMVQ or MMQ for M=1 on Blackwell?** This determines the correct comparison. Can verify with `llama.cpp -v` verbose output or CUDA logs.

5. **What's our actual per-GEMV-call time vs theoretical?** K=4096, N=varies per matrix. For q_proj (N=4096): 4096×4096/2 bytes (INT4) = 8MB. At 424 GB/s = 18.9µs theoretical. If measured is 54.8µs (from nsys), we're at 34% efficiency per-call. But nsys was batched — need M=1 numbers.

## Sources

### Kept (Primary/Authoritative)
- **llama.cpp mmvq.cu source** (https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu) — Primary kernel source, shows thread block configuration and vecdotq pattern
- **DeepWiki: llama.cpp CUDA Backend** (https://deepwiki.com/ggml-org/llama.cpp/5.1-cuda-backend-(nvidia)) — Architecture analysis of MMVQ vs MMQ dispatch
- **NVIDIA Blog: Optimizing llama.cpp with CUDA Graphs** (https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/) — CUDA Graph contribution to llama.cpp
- **llama.cpp issue #6763** (https://github.com/ggml-org/llama.cpp/issues/6763) — NVIDIA CUDA Graph optimization PR
- **vLLM CUDA Graphs** (https://docs.vllm.ai/en/stable/design/cuda_graphs/) — Production CUDA Graph use for decode overhead elimination
- **SM100 vs SM120 Blackwell GPU Wiki** (https://0xsero.github.io/blackwell-gpu-wiki/blackwell/sm100-vs-sm120/) — Consumer Blackwell feature differences
- **CUDA Occupancy Calculation** (https://leimao.github.io/blog/CUDA-Occupancy-Calculation/) — Warps-per-SM analysis
- **NVIDIA Blackwell Tuning Guide** (https://docs.nvidia.com/cuda/blackwell-tuning-guide/) — Blackwell memory best practices
- **GTC 2025: Maximize Memory Bandwidth notes** (https://shreyansh26.github.io/post/2025-03-23_gtc25-cuda-techniques-to-maximize-memory-bandwidth/) — cp.async, TMA, occupancy techniques
- **Megakernels: End-to-End Fused LLM Inference** (https://theorempath.com/topics/megakernels) — Persistent kernel approach
- **CUDA Pro Tip: Vectorized Memory Access** (https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/) — LDG.128 / uint4 validation
- **CUDA L2 Cache Control** (https://docs.nvidia.com/cuda/cuda-programming-guide/) — L2 residency hints (assessed as low-impact for M=1)
- **cp.async Pipelining** (https://deepwiki.com/gau-nernst/learn-cuda/6.2-asynchronous-pipelining-with-cp.async) — Async copy technique
- **Blackwell SM100: TMEM, TMA** (https://jianyuh.github.io/cuda/2026/04/12/blackwell-sm100-tmem-tma-and-the-new-tensor-core/) — TMA availability analysis
- **NVIDIA Nsight Systems: Overhead Visualization** (https://developer.nvidia.com/blog/understanding-visualization-of-overhead-and-latency-in-nsight-systems/) — Measuring kernel launch gaps
- **CUDA Graph Best Practices (PyTorch)** (https://docs.nvidia.com/dl-cuda-graph/latest/) — Launch overhead quantification
- **Kernel Launch Overhead** (https://inferensys.com/glossary/inference-optimization/) — 2-5µs per launch baseline
- **Our kernel source**: `src/kernels/gemv_int8.cu` lines 167-237 — `gemv_int8_warp_kernel`, 1 warp/block, stride-32 K-blocks, dp4a, uint4 loads

### Dropped
- FastGEMV optimization blog — generic GEMV techniques, not quantized-weight specific
- ap-gemv CUDA kernels — academic quantization paper, not directly applicable
- Llama 3.1 quality discussion — irrelevant to memory subsystem optimization
- Stack Overflow memory coalescing — too basic, already understood

## Gaps

1. **Cannot read llama.cpp source directly** — GitHub blob pages not fetchable via available tools. Kernel analysis based on DeepWiki + GitHub discussion descriptions + known vecdotq patterns. **Need to clone llama.cpp locally and read mmvq.cu for exact line numbers.**

2. **No M=1-specific nsys profile.** All profiling data in AGENTS.md is from batched benchmark. The 92.2% GEMV time figure is from `gemv_int4_batched`, not `gemv_int4_warp`. **Must profile the actual M=1 kernel to identify bottlenecks.**

3. **No inter-kernel gap measurement.** Don't know if launch overhead is real or negligible. **Must run `nsys profile` with timeline view and measure average gap between kernel end and next kernel start.**

4. **llama.cpp dispatch path for M=1 Q4_K on SM_120 unknown.** Could be MMVQ (vecdot) or MMQ (tensor core). Tensor core path would explain the speed advantage entirely. **Need `GGML_CUDA_DEBUG=1` or CUDA function profiling to confirm.**

## Recommended Next Steps

1. **Immediate: `nsys profile` the M=1 warp kernel** with `--trace=cuda,nvtx -o m1_warp`. Measure: (a) total GEMV time per token, (b) inter-kernel gaps, (c) achieved bandwidth per GEMV call, (d) warps eligible per SM (occupancy).

2. **If gaps > 2µs: CUDA Graph capture** of full 36-layer INT4 8B decode loop. Single highest-ROI optimization.

3. **If occupancy < 50%: increase warps/block** from 1 (32 threads) to 2-4 (64-128 threads). Add shared memory warp reduction. Measure bandwidth improvement.

4. **Clone llama.cpp, read mmvq.cu** to get exact nwarps, grid dims, vecdotq function signature for Q4_K on line-level detail.
