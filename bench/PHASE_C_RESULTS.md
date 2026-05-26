# Phase C Results — End-to-End Decode Benchmark

**Date**: 2026-05-26  
**Device**: NVIDIA GeForce RTX 5060 Ti (CC 12.0, 15858 MiB VRAM, 36 SMs)

---

## Decode Pipeline

Per-token decode: Q, K, V projections → attention → output projection → RMSNorm.

Config: hidden=2048, Qheads=16, KVheads=4, head_dim=128, layers=4, max_seq=2048.

| Metric                    | Value               |
| ------------------------- | ------------------- |
| 4-layer per-token         | **1.67 ms**         |
| 4-layer throughput        | **601 t/s**         |
| 28-layer estimate         | **86 t/s**          |
| Target (llama.cpp Q4_K_M) | **114 t/s** (tg128) |
| Gap to target             | **25%**             |

---

## Performance Breakdown (per token, 4 layers)

| Component                       | Time    | % of total |
| ------------------------------- | ------- | ---------- |
| Fused QKV (Q, K, V projections) | 0.77 ms | 46%        |
| attention_decode (seq_pos=128)  | 0.18 ms | 11%        |
| update_kv_cache                 | 0.02 ms | 1%         |
| pack_fp4 (attn_out)             | 0.01 ms | <1%        |
| gemv_fp4 (output projection)    | 0.77 ms | 46%        |
| RMSNorm + pack_fp4 (x)          | 0.03 ms | 2%         |

---

## Component Latency (microbenchmarks)

| Kernel                                  | Latency  |
| --------------------------------------- | -------- |
| fused_qkv_gemv (2048×2048 + 2×2048×512) | 0.192 ms |
| gemv_fp4 (2048×2048)                    | 0.193 ms |
| attention_decode (16 heads, seq=128)    | 0.011 ms |
| update_kv_cache (4 heads)               | 0.002 ms |
| pack_fp4 (2048 elements)                | 0.002 ms |
| fused_rmsnorm (2048)                    | 0.006 ms |

---

## CUDA Graphs

CUDA Graph capture reduces launch overhead from ~240 µs (80 kernel launches) to ~5 µs (1 graph launch). However, this only improved throughput by **1%** — launch overhead is not the bottleneck. The bottleneck is GEMV compute time at 77% of decode.

---

## Remaining Bottlenecks

1. **GEMV compute time** (77% of decode): Each 2048×2048 GEMV takes 0.193 ms. 37 registers, memory-bound. Tensor core not usable for GEMV (M=1).
2. **FP4 packing overhead**: Need pack_fp4 between attention (FP32 output) and output projection (FP4 input). Adds 0.01 ms.
3. **Scaling to 28 layers**: Linear scaling from 4 to 28 layers → 86 t/s estimate.

---

## Comparison vs llama.cpp

| Metric          | llama.cpp (Q4_K_M) | Blackwell FP4              | Ratio     |
| --------------- | ------------------ | -------------------------- | --------- |
| Prefill (pp512) | 4,560 t/s          | ~4,700 t/s (est from GEMM) | ~1.0×     |
| Decode (tg128)  | 114 t/s            | 86 t/s (est 28 layers)     | **0.75×** |

FP4 decode is within 25% of highly-optimized Q4_K_M on same hardware. The gap is primarily GEMV compute efficiency — llama.cpp uses tensor-core-accelerated small-M matmul (cuBLAS/cuBLASLt) while our FP4 GEMV is a simple loop-kernel with 37 registers.
