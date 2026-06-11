# NVIDIA/Model-Optimizer Scout Report

## Files Retrieved

1. `README.md` (root) — project overview, techniques, install, support matrix
2. `pyproject.toml` (root) — dependencies, extras, build config, Python 3.10–3.13, torch>=2.8
3. `CLAUDE.md` (root) — agent instructions
4. `modelopt/torch/quantization/config.py` (lines 1–1460) — quantization formats, cfg schema, presets
5. `modelopt/torch/quantization/tensor_quant.py` (full) — FP8/INT/floating-point quantization ops
6. `modelopt/torch/quantization/model_quant.py` (full) — public API: `quantize()`, `calibrate()`, `auto_quantize()`
7. `modelopt/torch/quantization/__init__.py` (full) — mtq namespace exports
8. `modelopt/torch/quantization/calib/max.py` (lines 1–111) — MaxCalibrator
9. `modelopt/torch/export/unified_export_hf.py` (lines 1–1153 of 1402) — HF checkpoint export pipeline
10. `modelopt/torch/export/__init__.py` (full) — export package init
11. `modelopt/torch/export/model_config.py` (full) — ModelConfig/LinearConfig/QKVConfig dataclasses
12. `modelopt/torch/quantization/qtensor/nvfp4_tensor.py` (full) — NVFP4 E2M1 quantization format
13. `modelopt/torch/quantization/qtensor/` (ls) — base_qtensor, fp8_tensor, int4_tensor, int8_tensor, mxfp4_tensor, mxfp8_tensor, nf4_tensor, nvfp4_tensor
14. `modelopt_recipes/ptq.md` (full) — recipe catalog, body/KV/cache scheme docs
15. `modelopt/torch/kernels/quantization/` (ls) — attention/, conv/, gemm/ triton kernels
16. `examples/llm_ptq/README.md` (full) — PTQ pipeline, AutoQuantize, export targets
17. `modelopt/deploy/llm/generate.py` — LLM deployment generate

---

## Supported Formats

| Format | Code name | Precision | Block size | Hardware |
|--------|-----------|-----------|------------|----------|
| FP8 | `FP8_DEFAULT_CFG` | E4M3 / E5M2 | per-tensor or per-channel | Hopper+ (sm_90+) |
| INT8 SmoothQuant | `INT8_SMOOTHQUANT_CFG` | W8A8 | per-tensor | All CUDA |
| INT4 AWQ | `INT4_AWQ_CFG` | W4A16 | block-128 | All CUDA |
| W4A8 AWQ | `W4A8_AWQ_BETA_CFG` | W4A8 | block-128 | All CUDA |
| NVFP4 | `NVFP4_DEFAULT_CFG` | E2M1 (FP4) + FP8 scales | block-16 | Blackwell+ (sm_120+) |
| NVFP4 AWQ | `NVFP4_AWQ_LITE_CFG` | E2M1 + FP8 scales | block-16 | Blackwell+ |
| NVFP4 SVDQuant | `NVFP4_SVDQUANT` | E2M1 + SVD low-rank | block-16 | Blackwell+ |
| MXFP4 | `MXFP4_DEFAULT_CFG` | E4M3 + E8M0 scales | dynamic | Blackwell+ |
| MXFP8 | `MXFP8_DEFAULT_CFG` | E4M3 + E8M0 scales | dynamic | Blackwell+ |
| W4A16 NVFP4 | `W4A16_NVFP4` | E2M1, BF16 act | block-16 | Blackwell+ |
| W4A8 NVFP4 FP8 | `W4A8_NVFP4_FP8` | INT4 + FP8 act | block-128 | Blackwell+ |
| FP8 PB (real) | `FP8_PB_REAL` | W8A8, real quant | per-tensor | Hopper+ |
| FP8 PC PT | `FP8_PC_PT` | W8A8, per-channel | per-channel | Hopper+ |
| INT8 WO | `INT8_WO` | W8A16 | block-128 | All CUDA |

### NVFP4 Format Detail (`nvfp4_tensor.py`)

- **Storage**: packed `uint8` (2 nibbles/byte), 16 values/block
- **Encoding**: E2M1 — 8 values: `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` positive, negated for negative
- **Per-block scale**: `FP8 E4M3` (2^-9 to 448), derived from `amax / 6.0`
- **Per-tensor scale (scale_2)**: `FP8 E8M0` — `global_amax / (6.0 * 448.0)`
- **Weight dequant**: `_cast_fp4()` unpacks nibbles → ordinal → e2m1 lookup table → multiply by `scale * scale_2`
- **Block size**: 16 (fixed for Blackwell GEMM)
- **Calibration**: max (fast), MSE (sweep), local Hessian (FP8 scale sweep), GPTQ (layerwise weight update)
- **Double quantization**: per-block FP8 scales quantized from FP32 → E8M0 (MX-style)
- **TRT-LLM path**: `torch.ops.trtllm.fp4_quantize()` for block-16 on sm_90+ via cutlass

---

## Conversion Pipeline

### High-Level Flow

```
HF/Megatron/ONNX model
        ↓
1. apply_mode(model, "quantize", config) — replace Linear with QuantLinear, insert TensorQuantizers
        ↓
2. calibrate(model, algorithm, forward_loop) — run calibration data, collect amax per quantizer
        ↓
3. requantize_resmooth_fused_llm_layers(model) — fuse QKV, fuse layernorms with pre_quant_scale
        ↓
4. _process_quantized_modules(model) — export quantized weights: pack → register scale buffers
        ↓
5. export_hf_checkpoint(model, export_dir) → safetensors + config.json
        ↓
Deploy on TRT-LLM / vLLM / SGLang
```

### Quantization Config Schema

```python
config = {
    "quant_cfg": [
        # Deny all by default
        {"quantizer_name": "*", "enable": False},
        # Enable weight+input quantizers per layer type
        {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": (4,3), "axis": 0, "block_sizes": {-1: 16}}},
        {"quantizer_name": "*input_quantizer",  "cfg": {"num_bits": (4,3), "axis": None}},
        # Exclude sensitive layers
        {"quantizer_name": "*lm_head*", "enable": False},
    ],
    "algorithm": {"method": "max"}  # or "mse", "awq_lite", "awq_full", "local_hessian", etc.
}
```

### Key Config Presets (`config.py`)

```python
FP8_DEFAULT_CFG       = _load_quantize_config_dict("configs/ptq/presets/model/fp8")
INT4_AWQ_CFG          = _load_quantize_config_dict("configs/ptq/presets/model/int4_awq")
NVFP4_DEFAULT_CFG     = _load_quantize_config_dict("configs/ptq/presets/model/nvfp4")
MXFP8_DEFAULT_CFG      = _load_quantize_config_dict("configs/ptq/presets/model/mxfp8")
```

### Calibrators (`modelopt/torch/quantization/calib/`)

| File | Class | Algorithm |
|------|-------|-----------|
| `max.py` | `MaxCalibrator` | absmax |
| `mse.py` | `MseCalibrator` | MSE search over amax multipliers; optionally FP8 scale sweep |
| `histogram.py` | `HistogramCalibrator` | KL divergence |
| `bias.py` | `BiasCalibrator` | per-axis bias calibration |
| calibrator.py | `_Calibrator` | base abstract class |

### Recipe System (`modelopt_recipes/`)

18 shipped YAML recipes under `modelopt_recipes/general/ptq/`:
- `fp8_default-kv_fp8_cast`
- `nvfp4_default-kv_fp8_cast`
- `nvfp4_mlp_only-kv_fp8_cast` ← recommended for dense models on Blackwell
- `nvfp4_experts_only-kv_fp8_cast` ← recommended for MoE on Blackwell
- `nvfp4_omlp_only-kv_fp8_cast`
- `nvfp4_weight_only-kv_fp8_cast`
- `int4_blockwise_weight_only`
- ...plus model-specific recipes for Qwen3.5, Gemma, Nemotron VL, etc.

Usage: `--recipe general/ptq/nvfp4_mlp_only-kv_fp8_cast` or `--recipe huggingface/llama/ptq/...`

---

## Key Types and Interfaces

### `TensorQuantizer` (`nn/modules/tensor_quantizer.py`)
- Wraps a quantizer (weight/input/output)
- Key attrs: `_amax`, `_pre_quant_scale`, `block_sizes`, `num_bits`
- SequentialQuantizer for chained formats (W4A8: INT4 then FP8)

### `QuantLinear` (`nn/modules/quant_linear.py`)
- Replaces `nn.Linear` after `apply_mode`
- Has `weight_quantizer`, `input_quantizer`, `output_quantizer`
- `fold_weight()` for fast eval

### `modelopt.torch.quantization.qtensor.*`
- `BaseQuantizedTensor` — base class
- `NVFP4QTensor` — E2M1 pack/unpack, dequant via `_unpack_tensor` + e2m1 lookup table
- `FP8Tensor`, `INT4Tensor`, `INT8Tensor`, `MXFP4Tensor`, `MXFP8Tensor`, `NF4Tensor`

### `modelopt.torch.export.model_config`
- `ModelConfig` — top-level model metadata
- `LinearConfig` — per-layer: quantization, weight, scale, prequant_scaling, block_size
- `QKVConfig` — merged QKV: concat weights, max of scales
- `AttentionConfig`, `MLPConfig`, `MOEConfig`, `DecoderLayerConfig`

### Public API (`model_quant.py`)

```python
# PTQ: quantize + calibrate in one call
model = mtq.quantize(model, config, forward_loop)

# Separate steps
model = apply_mode(model, "quantize", config)   # insert quantizers
model = mtq.calibrate(model, "max", forward_loop)  # collect amax

# AutoQuantize: search per-layer format
model, state = mtq.auto_quantize(
    model,
    constraints={"effective_bits": 4.8},
    quantization_formats=["NVFP4_DEFAULT_CFG", "FP8_DEFAULT_CFG"],
    data_loader=...,
    forward_step=...,
    loss_func=...,
)

# Postprocess
model = mtq.fold_weight(model)  # fold scales into weights
mtq.print_quant_summary(model)  # debug
```

### Export API (`export/unified_export_hf.py`)

```python
from modelopt.torch.export import export_hf_checkpoint

export_hf_checkpoint(
    model,
    export_dir,          # output directory
    dtype=torch.bfloat16,
    # For vLLM fakequant export:
    # vllm_fakequant_export=True
)
```

Outputs:
- `model.safetensors` — quantized weights packed as uint8 + scale buffers
- `config.json` — quantization_config, architecture, dtype, vocab_size
- `tokenizer.json`, `tokenizer_config.json` — copied from source

---

## Dependencies

### Core (`pyproject.toml`)
- `torch>=2.8` (requires CUDA)
- `ninja`, `numpy`, `packaging`, `setuptools>=80`
- `PyYAML>=6.0`, `omegaconf>=2.3.0`, `pydantic>=2.0`
- `safetensors`, `scipy`, `regex`, `rich`, `tqdm`
- `nvidia-ml-py>=12`

### Extras
- `hf`: `transformers>=4.56,<5.10`, `accelerate>=1.0.0`, `diffusers>=0.32.2`, `peft>=0.17.0`, `huggingface_hub>=0.24.0`
- `onnx`: `onnx~=1.21.0`, `onnx-graphsurgeon>=0.6.1`, `polygraphy>=0.49.22`
- `dev`: `pytest`, `coverage`, `mypy==1.17.1`, `ruff==0.12.11`, `sphinx`
- Python 3.10–3.13, SM support: sm_90 (Hopper), sm_120 (Blackwell)

---

## Architecture

```
modelopt/torch/
├── quantization/
│   ├── config.py          — QuantizerAttributeConfig, QuantizeConfig, preset CFGs
│   ├── tensor_quant.py    — FakeTensorQuantFunction, DynamicBlockQuantization, FP8 ops
│   ├── model_quant.py     — quantize(), calibrate(), auto_quantize() public API
│   ├── calib/             — MaxCalibrator, MseCalibrator, HistogramCalibrator, BiasCalibrator
│   ├── nn/modules/        — QuantLinear, QuantEmbedding, QuantConv, TensorQuantizer
│   ├── qtensor/           — NVFP4QTensor, FP8Tensor, INT4Tensor, INT8Tensor, MXFP4/8Tensor
│   ├── backends/         — nvfp4_gemm, fp8_per_tensor_gemm (Triton CUDA kernels)
│   ├── algorithms.py      — AutoQuantizeGradientSearcher, AutoQuantizeKLDivSearcher
│   ├── conversion.py      — apply_mode dispatcher, set_quantizer_by_cfg
│   ├── export_onnx.py     — ONNX export with DQ/Q nodes
│   └── plugins/           — HuggingFace integration, accelerate, transformers trainer
├── export/
│   ├── unified_export_hf.py   — export_hf_checkpoint (safetensors + config)
│   ├── model_config.py        — ModelConfig/LinearConfig dataclasses
│   ├── model_utils.py         — get_language_model_from_vl, is_multimodal_model
│   ├── layer_utils.py         — is_quantlinear, sync_moe_gate_up_amax, get_expert_linear_names
│   ├── quant_utils.py         — get_quant_config, to_quantized_weight, get_weight_scaling_factor
│   └── plugins/              — SpeculativeDecodingExporter
├── kernels/quantization/
│   ├── attention/         — quantized attention kernels
│   ├── conv/              — quantized conv kernels
│   └── gemm/              — Triton GEMM kernels (FP8, INT8, NVFP4, MX)
├── deploy/llm/generate.py  — deployment generation
└── recipes/ (modelopt_recipes/)
    ├── general/ptq/       — 18 shipped YAML recipe files
    ├── huggingface/       — model-specific recipe overrides
    └── configs/           — shared quant_cfg snippets
```

---

## Integration Points

### HF Model → ModelOpt → TRT-LLM/vLLM/SGLang

```python
# 1. Load HF model
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")

# 2. Quantize
import modelopt.torch.quantization as mtq
model = mtq.quantize(model, mtq.NVFP4_MLP_ONLY_CFG, forward_loop)

# 3. Export
from modelopt.torch.export import export_hf_checkpoint
export_hf_checkpoint(model, "./quantized_qwen3_8b")

# 4. Deploy
# TRT-LLM:
from tensorrt_llm import LLM
llm = LLM("./quantized_qwen3_8b")

# vLLM:
from vllm import LLM
llm = LLM("./quantized_qwen3_8b", quantization="modelopt")

# SGLang:
import sglang as sgl
llm = sgl.Engine(model_path="./quantized_qwen3_8b", quantization="modelopt")
```

### Recipe-based Quantization

```bash
python hf_ptq.py \
  --pyt_ckpt_path Qwen/Qwen3-8B \
  --recipe general/ptq/nvfp4_mlp_only-kv_fp8_cast \
  --export_path ./quantized_qwen3_8b
```

### Custom Calibrator

```python
from modelopt.torch.quantization.calib import _Calibrator
class MyCalibrator(_Calibrator):
    def collect(self, x): ...
    def compute_amax(self): return ...

config = {
    "quant_cfg": [...],
    "algorithm": {"method": "max", "calibrator": MyCalibrator}
}
```

### Custom Quantization Backend

```python
# Register via TensorQuantizer backend registry
from modelopt.torch.nn.modules.tensor_quantizer import register_quant_backend
register_quant_backend("my_backend", my_forward_func, my_backward_func)
# Then in config: {"backend": "my_backend", "backend_extra_args": {...}}
```

### AutoQuantize Integration

```python
# Mixed-precision search
model, state = mtq.auto_quantize(
    model,
    constraints={"effective_bits": 4.8},
    quantization_formats=["NVFP4_DEFAULT_CFG", "FP8_DEFAULT_CFG"],
    data_loader=calib_loader,
    forward_step=lambda m, b: m(b),
    loss_func=lambda out, b: loss(out, b),
)
config = mtq.get_auto_quantize_config(state, {"effective_bits": 5.0})
```

---

## Start Here

**`examples/llm_ptq/scripts/huggingface_example.sh`** — canonical end-to-end script. Shows quant → eval → export flow for all formats.

**`modelopt/torch/quantization/config.py`** (lines 1–200) — understand quantization config schema before touching anything else.

**`modelopt/torch/export/unified_export_hf.py`** — read `_export_transformers_checkpoint()` (lines 580–800) for the weight export flow: fuse → requantize → process quantized modules → save safetensors.

**`modelopt/torch/quantization/qtensor/nvfp4_tensor.py`** — NVFP4 E2M1 encoding, `_cast_fp4()`, `quantize()`, `dequantize()`. Relevant if blackwell needs custom INT4→NVFP4 conversion or scale layout.

---

## Relevance to Blackwell Project

- **NVFP4 E2M1 format**: ModelOpt uses E2M1 with block-16 and FP8 E4M3 per-block scales — NOT offset-binary INT4. Our `gemv_int4_batched` uses INT4 block-16 with FP32 scales. Different encoding, different scale layout. Cannot directly reuse ModelOpt weights without conversion.
- **Calibration**: ModelOpt provides max/MSE/Hessian/AWQ calibration. Our AWQ script uses random normal proxy (128 samples). ModelOpt has proper calibration with `forward_loop` over calibration dataset.
- **Export format**: ModelOpt exports safetensors with `quantization_config` metadata for TRT-LLM/vLLM/SGLang. We use flat binary weight files with separate scale tensors.
- **Scope**: ModelOpt is full-stack (PTQ → export → deploy). Our stack is inference-kernel-only (weights → decode). ModelOpt calibration could feed our kernel path if we convert formats.
- **Recipe YAML**: `modelopt_recipes/` declarative config files could serve as reference for blackwell's quantization config system.
