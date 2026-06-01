# Code Context

## Files Retrieved
1. `src/kernels/` — 26 `.cu` source files (listed below)
2. `include/blackwell/kernels.h` (full file, ~600 lines) — all exported function signatures
3. `CMakeLists.txt` (full file) — build config, kernel sources list, targets
4. `bench/` — ~75 executables, ~45 `.cu` sources, 2 `.py`, 3 `.md`
5. `scripts/` — 10 Python scripts
6. `weights_int8_qwen35_9b/` — 677 files, 11 GB

## Source Files in src/kernels/

| File | Purpose |
|------|---------|
| `attention.cu` | Decode attention kernels (single, GQA, batched GQA) |
| `cuda_graphs.cu` | CUDA Graph capture/launch/destroy helpers |
| `decode.cu` | Decode orchestration, seq_pos management, KV cache |
| `fp16_scale_cache.cu` | FP16 scale caching for INT8 GEMV |
| `fused_decode.cu` | Fused gate+up MLP GEMV |
| `fused_mlp.cu` | Fused MLP projections |
| `fused_o_norm.cu` | Fused O-projection + RMSNorm + FP4 pack |
| `gated_delta_net.cu` | GatedDeltaNet kernels (conv1d, recurrent step, RMSNormGated) for Qwen3.5-9B |
| `gemm_int8_mma.cu` | MMA.sync INT8 GEMM (stub, returns notSupported) |
| `gemm_int8_wmma.cu` | WMMA INT8 GEMM (4.8× over dp4a) |
| `gemm_int8_wmma_fast.cu` | Optimized WMMA 32×32 tiles (4.3-5.0K GFLOPS) |
| `gemm.cu` | FP4 block-scaled GEMM (large+small CTA), FP4 GEMV, FP4 splitk/v3/batched |
| `gemv_bf16.cu` | BF16 GEMV |
| `gemv_fp4_nv.cu` | NVF4 scalar GEMV (UE4M3 scales) |
| `gemv_int8_fp16sc.cu` | INT8 GEMV with FP16 scales |
| `gemv_int8_pdl.cu` | INT8 GEMV with PDL |
| `gemv_int8_unrolled.cu` | INT8 GEMV with 4× loop unrolling |
| `gemv_int8_warp_unrolled.cu` | Warp INT8 GEMV with 4× unrolling |
| `gemv_int8.cu` | Core INT8 GEMV variants: baseline, per-row, warp, batched, splitk, FP32×INT8, FP4 warp, INT4 warp, quantize, transpose |
| `gemv_v2.cu` | FP4 GEMV v2/v3/splitk/batched |
| `memory.cu` | Pack/unpack FP4, coalesced copy, unpack_fp4_pack_int8 |
| `norm.cu` | fused_rmsnorm, vector_add_fp32 |
| `prefill.cu` | Prefill layer orchestration, prefill attention |
| `quantize.cu` | pack_int8, quantize_int8, transpose_int8_weights |
| `rope.cu` | fused_rope, fused_rope_decode |
| `sample_gpu.cu` | GPU softmax + top-k + cuRAND sampling, argmax |

## Bench Executables (75 compiled)

Key production benchmarks:
- `decode_int8_cgraph` — CUDA Graph INT8 decode (M=1, 118.8 t/s)
- `decode_int8_batched_cgraph_attn` — Batched attn + CUDA Graph (M=8, **326.8 t/s**, production best)
- `decode_int8_generic` — Configurable INT8 decode for any model size
- `decode_int8_batched_cgraph` — Batched CUDA Graph (M=4/8)
- `text_generate` — End-to-end text generation (greedy/sampled)
- `decode_qwen35_9b` — Qwen3.5-9B decode (GatedDeltaNet)
- `decode_prefill` — Prefill path
- `speculative_decode_cgraph` — Speculative decoding

Utility/conversion:
- `convert_weights_int8` / `convert_weights_packed_fp4` / `convert_weights_int4`
- `verify_gemm` / `verify_gemm_dp4a` / `verify_int8_pipeline`

## Git Status — Uncommitted Changes

Modified:
- `AGENTS.md`, `CMakeLists.txt`, `HANDOFF.md`
- `bench/text_generate.cu`
- `include/blackwell/kernels.h`
- `src/kernels/decode.cu`, `src/kernels/sample_gpu.cu`

Untracked:
- `.pi/agents/` (session files)
- `bench/decode_qwen35_9b.cu`
- `scripts/quantize_qwen35.py`
- `src/kernels/gated_delta_net.cu`
- `context.md`, `research.md`

## Build Status

**Library is UP TO DATE** — no `.cu` source file is newer than `build/libblackwell_kernels.a`.
Library has **141 exported symbols** (confirmed via `nm`).
Last modified: timestamp 1780323642 (~2026-06-01).

## Scripts (10 files)

| Script | Purpose |
|--------|---------|
| `check_quality.py` | Quality verification |
| `export_bf16.py` | BF16 weight export |
| `extract_norms.py` | Extract normalization weights |
| `nvfp4_quantize.py` | NVF4 quantization |
| `prepare_tokenizer.py` | Tokenizer setup |
| `quantize_generic.py` | Generic INT8 quantization |
| `quantize_per_row_06b.py` | Per-row INT8 for 0.6B model |
| `quantize_per_row.py` | Per-row INT8 quantization |
| `quantize_qwen35.py` | Qwen3.5 quantization |
| `validate_full_pipeline.py` | Full pipeline validation |

## Weights — weights_int8_qwen35_9b/

- **677 files**, **11 GB** total
- Per-layer structure: `N_input_layernorm.f32`, `N_linear_attn.*.int8_t/.scale_t`, `N_mlp.*.int8_t/.scale_t`
- Special: `embed_tokens.int8_t/.scale_t`, `final_norm.f32`, `lm_head.int8_t/.scale_t`
- Qwen3.5-9B uses GatedDeltaNet: has `A_log.f32`, `conv1d.weight.f16`, `dt_bias.f32`, `in_proj_a/b/z/qkv`

## Key Exported Functions (kernels.h)

### INT8 Production Path
- `gemv_fp32_int8_per_row_warp` — **Production GEMV** (warp-cooperative, 1 warp/row)
- `gemv_int8_warp` — Warp INT8 GEMV (quantized activations)
- `gemv_int8_batched` — Batched M=1-8
- `attention_decode_batched_gqa` — Batched GQA decode attention
- `fused_rmsnorm_quant_int8` — RMSNorm + INT8 quant (fused)
- `fused_rmsnorm` — RMSNorm
- `apply_swiglu` — SiLU(gate) × up
- `fused_rope` / `fused_rope_decode` — RoPE
- `update_kv_cache` — KV cache write
- `sample_gpu` / `sample_argmax_gpu` — GPU sampling

### GEMM (Prefill)
- `gemm_int8_wmma` / `gemm_int8_wmma_fast` — WMMA tensor core GEMM
- `gemm_int8_dp4a` — dp4a GEMM
- `gemm_int8` — FP32×INT8 GEMM
- `gemm_fp4_block_scaled` / `gemm_fp4_block_scaled_small` — FP4 GEMM

### GatedDeltaNet (Qwen3.5-9B)
- `gated_delta_conv1d_update` — Conv1d + SiLU
- `gated_delta_recurrent_step` — SSM recurrent step
- `gated_delta_rmsnorm_gated` — Fused RMSNormGated

### Utility
- `quantize_int8` / `pack_int8` / `transpose_int8_weights`
- `pack_fp4` / `unpack_fp4`
- `vector_add_fp32`
- `update_decode_seq_pos` / `get_seq_pos_device_ptr` / `get_seq_pos_host_ptr`
- `clear_fp16_scale_caches` / `convert_scales_fp32_to_fp16`

## Architecture

```
CMakeLists.txt
├── libblackwell_kernels.a (26 .cu → 141 symbols)
│   ├── Decode path: norm → gemv_warp → rope → attention → gemv_warp → swiglu → gemv_warp → norm
│   ├── Prefill path: gemm_wmma → attention_prefill → gemm_wmma
│   └── GatedDeltaNet: conv1d → recurrent_step → rmsnorm_gated
├── bench/ (75 executables, each .cu links libblackwell_kernels.a)
├── scripts/ (Python quantization/validation)
└── include/blackwell/kernels.h (public API)
```

Data flow for INT8 decode:
1. `fused_rmsnorm_quant_int8` → INT8 activations + scales
2. `gemv_fp32_int8_per_row_warp` × 3 (Q/K/V projections)
3. `fused_rope_decode` (Q/K rotation)
4. `update_kv_cache` (write K/V)
5. `attention_decode_batched_gqa` (batched M sequences)
6. `gemv_fp32_int8_per_row_warp` (O-projection)
7. `vector_add_fp32` (residual)
8. Repeat for MLP: norm → gate/up gemv → swiglu → down gemv → residual
9. `sample_gpu` / `sample_argmax_gpu` (logit sampling)

## Start Here

Open `include/blackwell/kernels.h` — single file defining the entire public API. All 141 symbols declared here with full docstrings. Then `src/kernels/gemv_int8.cu` for the production GEMV implementations (contains warp, per-row, batched variants).
