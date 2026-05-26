# Blackwell Phase A Benchmark

Device: NVIDIA GeForce RTX 5060 Ti (CC 12.0, 15858 MiB VRAM, 36 SMs)

## 1. FP4 Pack/Unpack Correctness

| Operation   | Elements | Max Rel Error |
| ----------- | -------- | ------------- |
| pack+unpack | 512      | 1.0000e+00    |

## 2. FP4 GEMV (Decode Path)

| Op       | Shape (out x in) | Lat (ms) | GB/s  | Rel Err    |
| -------- | ---------------- | -------- | ----- | ---------- |
| gemv_fp4 | 64 x 64          | 0.003    | 1.6   | 1.0000e+00 |
| gemv_fp4 | 128 x 64         | 0.003    | 3.5   | 1.0000e+00 |
| gemv_fp4 | 2,048 x 64       | 0.003    | 47.1  | 1.0000e+00 |
| gemv_fp4 | 6,144 x 64       | 0.004    | 102.6 | 1.0000e+00 |

## 3. FP4 GEMM (Prefill Path)

| Op       | Shape                 | Lat (ms) | GB/s |
| -------- | --------------------- | -------- | ---- |
| gemm_fp4 | 512 x 2,048 x 2,048   | 1.766    | 5.3  |
| gemm_fp4 | 512 x 6,144 x 2,048   | 5.219    | 5.0  |
| gemm_fp4 | 512 x 2,048 x 6,144   | 5.304    | 3.8  |
| gemm_fp4 | 2,048 x 2,048 x 2,048 | 6.958    | 3.6  |

## 4. Fused Epilogues

| Op            | Elements  | Lat (ms) | GB/s   |
| ------------- | --------- | -------- | ------ |
| fused_rmsnorm | 4,096     | 0.008    | 6.0    |
| fused_rmsnorm | 2,048     | 0.006    | 3.9    |
| apply_swiglu  | 524,288   | 0.006    | 1022.2 |
| apply_swiglu  | 1,572,864 | 0.012    | 1527.3 |

## 5. Summary

Phase A establishes baseline FP4 kernel throughput on RTX 5060 Ti.
Compare with llama-bench: Qwen3.5-4B Q4_K_M on same hardware:
prefill: ~4560 t/s (pp512) decode: ~114 t/s (tg128)
