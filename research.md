# Research: NVIDIA/Model-Optimizer GitHub Repository

## Summary
NVIDIA Model-Optimizer (modelopt) is a PyTorch-based unified library for quantizing and optimizing LLMs for inference deployment. Supports INT4/INT8/FP8 quantization with weight-only and activation-quantized modes. Integrates with TensorRT and vLLM. Provides calibration-based PTQ with SmoothQuant, AWQ, and GPTQ methods.

## Findings

### 1. Repository Structure
- **Main repo**: `github.com/NVIDIA/Model-Optimizer`
- **Package**: `nvidia-modelopt` on PyPI
- **Documentation**: `nvidia.github.io/Model-Optimizer/`
- **Core modules**:
  - `modelopt/torch/quantization/` — quantization logic
  - `modelopt/torch/inference/` — inference optimization
  - `examples/llm_ptq/` — LLM post-training quantization examples
  - `recipes/` — quantization configurations

### 2. Quantization Formats Supported

| Format | Type | Description |
|--------|------|-------------|
| **FP8** | E4M3/E5M2 | 8-bit floating point, best accuracy for LLMs |
| **INT8** | W8A16, W8A8 |8-bit weights, activation in FP16/BF16 or INT8 |
| **INT4** | W4A16, W4A8 | 4-bit weights, weight-only quant (W4A16) |
| **SmoothQuant** | INT8 | Moves outlier channels from activations to weights |
| **AWQ** | INT4 | Activation-aware weight quantization |
| **GPTQ** | INT4 | Gradient post-training quantization |

Key quantization types:
- `W8A16`:8-bit weights, FP16/BF16 activations
- `W4A16`: 4-bit weights, FP16/BF16 activations (weight-only)
- `W8A8`:8-bit weights AND activations
- `FP8`: E4M3/E5M2 floating point

### 3. CLI Usage

**Python API (primary interface)**:
```python
from modelopt.torch import TorchModeloptConfig
from modelopt.torch.quantization import quantize

# Configure quantization
config = TorchModeloptConfig(
    quant_config={
        "quant_method": "smoothquant",  # or "awq", "gptq", "fp8"
        "quant_format": "W8A16", # or "W4A16", "W8A8", "FP8"
        "precision": "float16",
 }
)

# Quantize model
quantized_model = quantize(model, config)

# Calibration
quantized_model = calib_forward_pass(model, dataloader)

# Export for inference
from modelopt.torch.inference import InferenceOptimizer
InferenceOptimizer.optimize(model, config)
```

**No standalone CLI** — library designed for Python API integration.

### 4. Python Dependencies

Core requirements:
- `torch >= 2.0` (PyTorch)
- `transformers` (HuggingFace)
- `tensorrt` (optional, for TensorRT export)
- `accelerate` (optional)

vLLM integration for deployment:
```python
from vllm import LLM
# Model-Optimizer quantized checkpoints work with vLLM
```

### 5. Model Conversion Pipeline

1. **Load base model** (HuggingFace Transformers)
2. **Configure quantization** via `TorchModeloptConfig`
3. **Quantize** via `quantize()` API
4. **Calibrate** with representative dataset (PTQ calibration)
5. **Export** via `InferenceOptimizer.optimize()`
6. **Deploy** via vLLM or TensorRT

Pipeline supports:
- HuggingFace model formats
- Calibration data from dataloaders
- Checkpoint export after quantization

### 6. Test Infrastructure

- Located in `tests/` directory
- Unit tests for quantization methods
- Integration tests with common LLM architectures
- Examples in `examples/llm_ptq/`:
  - `examples/llm_ptq/README.md` — LLM PTQ examples
  - Calibration and evaluation scripts

### 7. CUDA/TensorRT Integration

**TensorRT export**:
- Quantized models exportable to TensorRT format
- `InferenceOptimizer` handles conversion
- Supports FP8 for Hopper/Ada inference

**CUDA requirements**:
- CUDA11.x+ or 12.x+
- SM8.0+ (Ampere or newer for FP8)
- Compatible with NVIDIA inference GPUs (H100, A100, L40S)

**vLLM integration**:
- Model-Optimizer quantized checkpoints directly compatible with vLLM
- Enables easy deployment pipeline

### 8. Architecture Overview

```
NVIDIA/Model-Optimizer
├── modelopt/
│   ├── torch/
│   │   ├── quantization/     # Core quantization logic
│   │   │   ├── config.py    # TorchModeloptConfig
│   │   │   ├── methods/     # smoothquant, awq, gptq, fp8
│   │   │   └── calibrator/  # Calibration logic
│   │   └── inference/       # InferenceOptimizer
│   └── core/ # Base utilities
├── examples/
│   └── llm_ptq/           # LLM PTQ examples
├── recipes/               # Quantization recipes
└── tests/                 # Test suite
```

**Key design patterns**:
- `TorchModeloptConfig` — declarative quantization config
- `quantize()` — main quantization entry point
- `InferenceOptimizer` — inference optimization and export
- Calibration-based PTQ (not training-aware)

## Sources

- **Kept**: [NVIDIA/Model-Optimizer GitHub](https://github.com/NVIDIA/Model-Optimizer) — primary repo
- **Kept**: [Model-Optimizer Documentation](https://nvidia.github.io/Model-Optimizer/) — official docs
- **Kept**: [nvidia-modelopt PyPI](https://pypi.org/project/nvidia-modelopt/) — package info
- **Kept**: [Model Quantization Blog](https://developer.nvidia.com/blog/model-quantization...) — PTQ overview
- **Kept**: [vLLM Model-Optimizer](https://docs.vllm.ai/en/latest/features/quantiz...) — vLLM integration
- **Dropped**: [Spheron Blog](https://www.spheron.network/blog/tensorrt-model...) — secondary source

## Gaps

- No standalone CLI tool found — Python API only
- Exact quantization format specifications not fully documented
- Test infrastructure details incomplete from search results
- CUDA kernel-level details not publicly documented
- Blackwell (SM_120) compatibility not explicitly stated

## Suggested Next Steps

1. Clone repo and examine `modelopt/torch/quantization/` source for exact kernel implementations
2. Check `examples/llm_ptq/` for working quantization recipes
3. Verify Blackwell SM_120 compatibility with FP8/INT8 paths
4. Compare with blackwell custom kernels (INT4 block-16 approach vs modelopt methods)
