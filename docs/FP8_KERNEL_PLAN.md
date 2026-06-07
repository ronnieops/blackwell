# FP8 Kernel Path — OBSOLETE (as of Session 56)

**FP8 path abandoned.** INT8 block-16 wins on both quality AND speed.

## Why it's obsolete

| Metric | INT8 block-16 | FP8 per-row |
|--------|---------------|-------------|
| PPL (1.7B) | **18.65** (1.5× BF16) | 41.75 (3.4× BF16) |
| GEMV speed | **0.003 ms** (dp4a) | 0.015 ms (4.5× slower) |
| Weight PSNR | **72.7 dB** | 57.9 dB |

INT8 is 4.5× faster AND 2.3× better quality than FP8 for weight-only quantization.

## What we built

| Artifact | Status |
|----------|--------|
| scripts/quantize_fp8.py | Reference (FP8 E4M3 per-row quantizer) |
| bench/bench_ppl_fp8.cu | Reference (FP8 PPL benchmark) |
| src/kernels/gemv_fp8.cu | Reference (FP8 GPU GEMV kernel) |
| docs/FP8_KERNEL_PLAN.md | OBSOLETE — this file |
| weights_fp8_bf16/ | Deleted (1.9 GB) |

## The real problem was never quantization format

The "INT8 quality wall" (PPL=7.3M) was caused by **wrong model dimensions**:
nqh=32, nkv=4, hd=64, KV=512 — actual Qwen3-1.7B config is nqh=16, nkv=8, hd=128, KV=1024.

With correct dims, INT8 block-16 gives PPL=18.65 — only 1.5× worse than BF16.
