# Blackwell Server — Deployment Guide

## Requirements

- **GPU**: NVIDIA RTX 5060 Ti (16 GB), compute 12.0 (Blackwell)
- **CUDA**: 13.3+
- **Driver**: Latest NVIDIA driver
- **Disk**: 10 GB free space

## Installation

### 1. Clone/Copy Project

```bash
cd /mnt/data/dev/projects/blackwell
```

### 2. Build

```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### 3. Prepare Weights

INT4 8B weights: `weights_int4_qwen3_8b/` (5.3 GB)

```bash
ls -la weights_int4_qwen3_8b/ | head -20
```

Expected files:
- `layer_0.qkv.int4_t` through `layer_35.down.int4_t`
- `layer_*.layernorm.f32`
- `embed.int4_t`, `embed.scale.f32`
- `lm_head.int4_t`, `lm_head.scale.f32`

## Running the Server

### Single Instance

```bash
# Kill any conflicting processes
killall hashcat 2>/dev/null

# Start server
./server/http_subprocess batched &

# Verify it's running
curl http://localhost:8123/health
```

### Docker

```bash
# Build image
docker build -f Dockerfile.int4 -t blackwell-server:int4 .

# Run
docker run --gpus all -p 8080:8080 \
  -v /path/to/weights_int4_qwen3_8b:/app/weights_int4_qwen3_8b \
  blackwell-server:int4 8080 int4_8b
```

## Testing

### Health Check

```bash
curl http://localhost:8123/health
```

Expected response:
```json
{"status":"ok","model":"blackwell-8B","gpu_used_mb":14440,"uptime_sec":10,"requests":0,"errors":0,"avg_latency_ms":0.0}
```

### Generate Text

```bash
curl -X POST http://localhost:8123/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","max_tokens":20}'
```

### Batch Completion

```bash
curl -X POST http://localhost:8123/v1/batch \
  -H "Content-Type: application/json" \
  -d '{"prompts":["Hello","World"],"max_tokens":10}'
```

## Performance Tuning

### Repetition Penalty

Default: 1.5 (reduces token looping)

```bash
# Disable repetition penalty
curl -X POST http://localhost:8123/v1/completions \
  -d '{"prompt":"test","max_tokens":20,"repetition_penalty":1.0}'
```

### Temperature

Default: 0.0 (greedy decoding)

```bash
# Stochastic sampling
curl -X POST http://localhost:8123/v1/completions \
  -d '{"prompt":"test","max_tokens":20,"temperature":0.7}'
```

## Monitoring

### Benchmark Suite

```bash
python3 scripts/benchmark_suite.py --quick
```

Tests throughput, correctness, memory stability.

### Metrics

Health endpoint exposes:
- Request count
- Error count
- Average latency
- GPU memory usage

## Troubleshooting

### Server Won't Start

```bash
# Check GPU is available
nvidia-smi

# Check port is free
lsof -i :8123

# Check weights exist
ls weights_int4_qwen3_8b/
```

### Out of Memory

```bash
# Kill hashcat and other GPU processes
killall hashcat
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
```

### Slow Performance

Check:
1. No other GPU processes running
2. Using `batched` mode for higher throughput
3. Temperature = 0 for fastest decoding

## Production Deployment

### systemd Service

Create `/etc/systemd/system/blackwell.service`:

```ini
[Unit]
Description=Blackwell Inference Server
After=cuda.service

[Service]
Type=simple
User=youruser
WorkingDirectory=/mnt/data/dev/projects/blackwell
ExecStart=/mnt/data/dev/projects/blackwell/server/http_subprocess batched
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable blackwell
sudo systemctl start blackwell
sudo systemctl status blackwell
```

### Reverse Proxy (nginx)

```nginx
server {
    listen 8080;
    location / {
        proxy_pass http://127.0.0.1:8123;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
}
```

## Architecture

```
HTTP Server (http_subprocess)
    │
    └── Inference Engine (inference_server_int4_batched)
            │
            ├── Tokenizer (BpeTokenizer)
            ├── Embedding (INT4 dequantize)
            ├── 36 Layers
            │   ├── RMSNorm
            │   ├── QKV GEMV (INT4)
            │   ├── Attention (GQA)
            │   ├── KV Cache Update
            │   ├── Output Projection (INT4)
            │   ├── Residual Add
            │   ├── SwiGLU (INT4 GEMV)
            │   └── MLP Down (INT4 GEMV)
            └── LM Head (INT4 GEMV) → softmax → sample
```

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
| AWQ alpha | 0.6 |