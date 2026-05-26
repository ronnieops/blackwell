# Phase B Results — GEMM & GEMV Optimization on RTX 5060 Ti

**Date**: 2026-05-26  
**Device**: NVIDIA GeForce RTX 5060 Ti (CC 12.0, 15858 MiB VRAM, 36 SMs)  
**CUDA**: 12.8, SM_120 native cubins

---

## GEMM Optimization (Prefill Path)

Rewrote from 16×16×64 tile → 128×128×64 CTA with cp.async 2-stage pipeline.

| Shape | Before | After | Speedup | % of Peak |
|-------|--------|-------|---------|-----------|
| 512×2048×2048 | 5.3 GB/s | 37.7 GB/s | **7.1×** | 7.5% |
| 512×6144×2048 | 5.0 GB/s | 35.3 GB/s | **7.1×** | 7.1% |
| 512×2048×6144 | 3.8 GB/s | 27.0 GB/s | **7.1×** | 5.4% |
| 2048×2048×2048 | 3.6 GB/s | 25.5 GB/s | **7.1×** | 5.1% |

**Optimizations applied**:
1. **Tile size**: 16×16 → 128×128 (8 warps, 4 M×2 N fragment grid)
2. **Vectorized loads**: uint4 (128-bit) FP4 global loads
3. **Register dequant**: FP4→FP16 in registers before smem store
4. **cp.async pipeline**: 2-stage double-buffered, 80 KB dynamic smem
5. **Register pressure**: 136 regs, 0 spill (was 156)

**Limitation**: Compute-bound — WMMA throughput is the bottleneck, not memory bandwidth. Next step: warp specialization (abandoned due to register thrashing, 255 regs + 132 B spill).

---

## GEMV Rewrite (Decode Path)

Removed K=64 hardcoded restriction. Dynamic K-tiling for any K multiple of 16.

| Shape | Latency | GB/s | Rel Err |
|-------|---------|------|---------|
| 64×64 | 0.008 ms | 0.5 | 0.0 |
| 128×64 | 0.008 ms | 1.1 | 0.0 |
| 2048×64 | 0.008 ms | 16.8 | 0.0 |
| 6144×64 | 0.008 ms | 51.5 | 0.0 |
| 64×2048 | 0.170 ms | 0.8 | 0.0 |
| 128×2048 | 0.172 ms | 1.6 | 0.0 |
| 2048×2048 | 0.193 ms | 22.1 | 0.0 |
| 6144×2048 | 0.193 ms | 66.5 | 0.0 |
| 2048×4096 | 0.385 ms | 22.2 | 0.0 |

**Key changes**:
- 256 threads/block (was 64)
- Proper 2D block-scale layout for W_scale
- Coalesced global reads within warp
- L1 broadcast for input x

---

## Decode Attention v2

Warp-parallel QK dot products with smem Q broadcast.

| seq_pos | Latency | t/s | GFLOPS |
|---------|---------|-----|--------|
| 8 | 4.2 µs | 239K | 8.8 |
| 32 | 5.1 µs | 197K | 26.6 |
| 64 | 6.8 µs | 147K | 39.2 |
| 128 | 11.1 µs | 90K | 47.5 |
| 256 | 19.0 µs | 53K | 55.4 |
| 512 | 34.8 µs | 29K | 60.3 |

QK dot uses all 8 warps in parallel (8× vs sequential). V weighted sum sequential over `t` with 32-way output parallelism.

---

## Fused QKV GEMV

Single kernel: Q = x@W_q, K = x@W_k, V = x@W_v.

| Method | Time | Speedup |
|--------|------|---------|
| 3 separate GEMVs | 0.576 ms | 1× |
| 1 fused QKV | 0.192 ms | **3×** |

Saves 2 kernel launches + reuses x from L1 across all 3 projections.

---

## Key Findings

1. **GEMM 25-38 GB/s**: 7× improvement over Phase A baseline. WMMA compute-bound.
2. **GEMV dynamic K**: Any K multiple of 16. Perfect correctness (rel_err=0).
3. **Fused QKV**: 3× faster than separate GEMVs. Critical for decode.
4. **TMA not available**: `cp.async.cg` is optimal for sm_120. No `cp.async.bulk`.
5. **Decode attention v2**: 11 µs at seq_pos=128 (8× faster QK scores).
