# Blackwell Inference Server — API Documentation

## Overview

High-performance INT4 LLM inference server for Qwen3-8B on RTX 5060 Ti (Blackwell GB206).

**Performance**: ~56-63 tokens/sec, PPL 21.82
**Weight size**: 5.3 GB (INT4 symmetric)
**Quantization**: AWQ with α=0.6

## Quick Start

```bash
# Start server
./server/http_subprocess batched &

# Health check
curl http://localhost:8123/health

# Generate text
curl -X POST http://localhost:8123/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","max_tokens":20}'
```

## Endpoints

### GET /health

Health check with metrics.

**Response:**
```json
{
  "status": "ok",
  "model": "blackwell-8B",
  "gpu_used_mb": 14440,
  "gpu_total_mb": 15849,
  "uptime_sec": 120,
  "requests": 42,
  "errors": 0,
  "avg_latency_ms": 235.0
}
```

| Field | Type | Description |
|-------|------|-------------|
| status | string | "ok" if server running |
| model | string | Model identifier |
| gpu_used_mb | int | GPU memory used (MB) |
| gpu_total_mb | int | Total GPU memory (MB) |
| uptime_sec | int | Server uptime (seconds) |
| requests | int | Total requests served |
| errors | int | Total errors |
| avg_latency_ms | float | Average latency (ms) |

### GET /v1/models

List available models.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "blackwell-8B",
      "object": "model",
      "created": 0,
      "owned_by": "blackwell",
      "root": "blackwell-8B"
    }
  ]
}
```

### POST /v1/completions

Generate text completion.

**Request:**
```json
{
  "prompt": "The capital of France is",
  "max_tokens": 20,
  "temperature": 0.0,
  "top_k": 0,
  "repetition_penalty": 1.5
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| prompt | string | required | Input text |
| max_tokens | int | 20 | Max tokens to generate |
| temperature | float | 0.0 | Sampling temperature (0 = greedy) |
| top_k | int | 0 | Top-k sampling (0 = disabled) |
| repetition_penalty | float | 1.5 | Repetition penalty (1.0 = off) |

**Response:**
```json
{
  "id": "cmpl-0",
  "object": "text_completion",
  "created": 0,
  "model": "blackwell-8B",
  "choices": [
    {
      "text": " Paris, a city in the north of France.",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 21,
    "total_tokens": 21
  }
}
```

### POST /v1/chat/completions

Generate chat completion (OpenAI-compatible).

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "max_tokens": 20,
  "temperature": 0.0
}
```

**Response:**
```json
{
  "id": "cmpl-0",
  "object": "chat.completion",
  "created": 0,
  "model": "blackwell-8B",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": " Paris, a city in the north of France."
      },
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 21,
    "total_tokens": 21
  }
}
```

### POST /v1/batch

Batch completion for multiple prompts.

**Request:**
```json
{
  "prompts": ["Prompt 1", "Prompt 2", "Prompt 3"],
  "max_tokens": 20,
  "temperature": 0.0,
  "repetition_penalty": 1.5
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| prompts | array | required | Array of 1-8 prompts |
| max_tokens | int | 20 | Max tokens per prompt |
| temperature | float | 0.0 | Sampling temperature |
| repetition_penalty | float | 1.5 | Repetition penalty |

**Response:**
```json
{
  "text": [
    "{\"tokens\":[[12095,11,264,892,374]],\"text\":\" Paris, a city\"}"
  ]
}
```

Note: Response is JSON-escaped. Parse `text[0]` as JSON to get tokens and text.

## Server Modes

| Mode | Binary | Throughput | Description |
|------|--------|------------|-------------|
| batched | `inference_server_int4_batched` | ~63 t/s | Batched GEMV kernel, M=1 |
| int4_8b | `inference_server_int4` | ~56 t/s | Warp GEMV kernel |

Start with:
```bash
./server/http_subprocess batched  # faster
# or
./server/http_subprocess int4_8b  # alternative
```

## Performance Notes

- **Throughput**: ~56-63 tokens/sec depending on mode
- **Latency**: ~16-18 ms per token
- **Memory**: ~14.4 GB GPU memory
- **Quality**: PPL 21.82 (1.76× BF16 baseline)

### Factors Affecting Performance

1. **Prompt length**: Longer prompts = more KV cache entries
2. **Batch size**: Currently M=1 supported
3. **Temperature**: >0 adds stochastic sampling overhead
4. **Repetition penalty**: Enabled by default (1.5)

## Error Handling

| Error Code | Description |
|------------|-------------|
| 400 | Invalid request (bad JSON) |
| 504 | Generation timeout (>30s) |
| 500 | Server error |

## Benchmark Suite

Run automated tests:
```bash
python3 scripts/benchmark_suite.py --quick  # ~30s
python3 scripts/benchmark_suite.py          # ~5 min (full)
```

Tests:
- Health endpoint
- Model listing
- Output correctness
- Throughput measurement
- Memory stability
- Repetition penalty