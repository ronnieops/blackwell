# Blackwell Inference Engine

High-performance INT4 LLM inference on RTX 5060 Ti (Blackwell GB206).

**Model**: Qwen3-8B INT4 (AWQ, α=0.6)
**Throughput**: ~56-63 tokens/sec
**Quality**: PPL 21.82 (1.76× BF16 baseline)
**Weight size**: 5.3 GB (INT4 symmetric)

## Quick Start

```bash
# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Start server
./server/http_subprocess batched &

# Generate text
curl -X POST http://localhost:8123/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","max_tokens":20}'
```

## Performance

| Config | t/s | PPL | Notes |
|--------|-----|-----|-------|
| BF16 (llama.cpp Q8_0) | 74 | 12.4 | Baseline |
| **INT4 batched** | **56-63** | **21.82** | Production |
| INT4 warp | 56 | 21.82 | Alternative |

### Profiling Results (nsys)

| Kernel | % Time | Avg (μs) | Notes |
|--------|--------|----------|-------|
| GEMV (INT4 batched) | 92.2% | 54.8 | Bottleneck |
| RMSNorm | 3.7% | 7.6 | Light |
| Quantize | 1.3% | 1.3 | Light |
| Attention | 0.9% | 3.9 | Light |

**Key insight**: Weight loading (H2D memory) is the bottleneck. GEMV compute is efficient.

## API

### Health Check
```bash
curl http://localhost:8123/health
```
```json
{"status":"ok","model":"blackwell-8B","gpu_used_mb":7287,"uptime_sec":120,"requests":42,"errors":0,"avg_latency_ms":235.0}
```

### Text Completion
```bash
curl -X POST http://localhost:8123/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The capital of France is",
    "max_tokens": 20,
    "temperature": 0.0,
    "repetition_penalty": 1.5
  }'
```

### Batch Completion
```bash
curl -X POST http://localhost:8123/v1/batch \
  -H "Content-Type: application/json" \
  -d '{"prompts":["Hello","World"],"max_tokens":10}'
```

### Chat Completion
```bash
curl -X POST http://localhost:8123/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":10}'
```

## Server Modes

| Mode | Binary | t/s | Description |
|------|--------|-----|-------------|
| `batched` | `inference_server_int4_batched` | ~63 | Batched GEMV, M=1 |
| `int4_8b` | `inference_server_int4` | ~56 | Warp GEMV |

## Benchmark Suite

```bash
# Quick test (~30s)
python3 scripts/benchmark_suite.py --quick

# Full test (~5 min)
python3 scripts/benchmark_suite.py
```

Tests: health, models, correctness, throughput, memory stability, repetition penalty.

## Architecture

```
HTTP Server (http_subprocess)
    └── Inference Engine
        ├── Tokenizer (BPE)
        ├── Embedding (INT4 dequantize)
        ├── 36 Layers
        │   ├── RMSNorm
        │   ├── QKV GEMV (INT4 batched)
        │   ├── Attention (GQA, 32Q/8KV)
        │   ├── KV Cache Update
        │   ├── Output Projection (INT4)
        │   ├── Residual Add
        │   ├── SwiGLU (INT4 GEMV)
        │   └── MLP Down (INT4)
        └── LM Head → softmax → sample
```

### Key Kernels

- **gemv_int4_batched**: Warp-cooperative GEMV, M sequences, dp4a SIMD
- **fused_rmsnorm**: Warp-reduced RMSNorm, single pass
- **attention_decode_batched_gqa**: FlashAttention-style GQA decode

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Hidden dim | 4096 |
| Intermediate | 10944 |
| Layers | 36 |
| Q heads | 32 |
| KV heads | 8 |
| Head dim | 128 |
| Vocab | 151643 |
| Quantization | INT4 symmetric |
| Block size | 16 |
| AWQ alpha | 0.6 |

## Files

```
├── docs/
│   ├── API.md           # API documentation
│   ├── DEPLOYMENT.md    # Deployment guide
│   └── ARCHITECTURE.md  # Kernel architecture
├── server/
│   ├── inference_server_int4_batched.cu  # INT4 batched server
│   ├── http_subprocess.cpp              # HTTP wrapper
│   └── http_subprocess                  # Compiled binary
├── bench/
│   ├── text_generate_int4_batched.cu    # Benchmark
│   └── profile_decode.cu                # Profiler
├── scripts/
│   ├── benchmark_suite.py              # Test suite
│   └── quantize_awq_int4_8b.py          # AWQ quantization
├── include/blackwell/
│   └── kernels.h                        # Kernel API
└── build/libblackwell_kernels.a         # Kernel library
```

## Docker

```bash
# Build
docker build -f Dockerfile.int4 -t blackwell-server:int4 .

# Run
docker run --gpus all -p 8080:8080 \
  -v /path/to/weights_int4_qwen3_8b:/app/weights_int4_qwen3_8b \
  blackwell-server:int4 8080 int4_8b
```

## Development

```bash
# Build
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build && cmake --build build

# Run benchmark
./bench/text_generate_int4_batched "test" 1 50

# Profile
nsys profile --trace=cuda ./bench/text_generate_int4_batched "test" 1 30
```

## Known Issues

1. **Non-deterministic output**: GPU FP operations may produce slightly different outputs across runs. Model quality is consistent.

2. **9B quality blocked**: Qwen3.5-9B produces garbled output due to SSM instability (A_log > 0 for 68.8% of layer-4 channels).

## License

See AGENTS.md for project context and history.