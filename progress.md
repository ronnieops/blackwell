# Progress Log

## Session: M=1 GEMV External Research (2026-06-14)

**Task**: Research llama.cpp MMVQ kernel memory patterns + Blackwell memory best practices for M=1 INT4 GEMV optimization.

**Status**: IN PROGRESS — compiling findings into handoff/m1-external.md

### Research angles covered:
1. ✅ llama.cpp mmvq.cu kernel architecture (thread/warp config, memory access patterns)
2. ✅ CUDA Graph usage in llama.cpp / vLLM for launch overhead elimination
3. ✅ Blackwell SM_120 memory best practices (TMA, cp.async, LDG, L2 hints)
4. ✅ Occupancy / warp utilization analysis for 1-warp-per-row GEMV
5. ✅ Software pipelining / double buffering for memory-bound kernels
6. ✅ Megakernel / persistent kernel approaches
7. ✅ Our kernel architecture review (gemv_int8_warp_kernel: 1 warp/row, 32 threads, stride-32 K-blocks, dp4a, __launch_bounds__(32,8))

### Key finding:
Our kernel: `__launch_bounds__(32, 8)` = max 8 warps/SM × 36 SMs = 288 concurrent warps. Grid = N blocks (N=output dim). Each block = 1 warp = 1 output row. Weight loads coalesced (same row, stride-32 across threads). The bottleneck is NOT coalescing — it's likely **launch overhead** (9720 launches/token × ~2-5µs = 19-49ms overhead) and **occupancy** (only 8 warps/SM may not fully hide DRAM latency).
