# Blackwell Inference — Custom CUDA Kernels for RTX 5060 Ti

Target: **RTX 5060 Ti 16 GB** (compute capability 12.0 / sm_120)

## Performance

| Method | t/s | vs llama.cpp Q4_K_M |
|--------|-----|---------------------|
| llama.cpp Q4_K_M (b9500, 1.7B) | 293.4 | 100% |
| Blackwell INT8 server (correct model, M=1) | ~106 | 36% |
| Blackwell INT8 benchmark CUDA Graph M=8 | 575 | 196% ⚠️ |
| Blackwell INT8 benchmark per-kernel M=8 | 324 | 111% |

⚠️ Benchmark 575 t/s omits head_norm and RoPE — not achievable with correct model. Server uses correct Qwen3-1.7B architecture with per-layer RMSNorm, Q/K head norms, and RoPE.

### 9B GatedDeltaNet (Qwen3.5-9B)

| Method | t/s | vs llama.cpp Q3_K_M |
|--------|-----|----------------------|
| llama.cpp Q3_K_M (b9500, 9B) | 71.4 | 100% |
| 9B M=1 (per-seq) | 45.7 | 64% |
| **9B M=8 (batched GEMV + RMSNorm)** | **52.1** | **73%** |

**Profile breakdown (M=8)**: MLP 74.8%, LinAttn GEMV 18%, FullAttn GEMV 3.2%, Conv+Rec 2.1%.
9B is bandwidth-bound — weight matrices (200 MB/layer) exceed L2 cache. Further optimization has negligible room.

## Key optimizations

- **INT8 warp-cooperative GEMV**: 1 warp per row, dp4a SIMD, shuffle reduce — 181 t/s M=1 (no head_norm/RoPE)
- **Batched GEMV**: all M sequences in one kernel call (attention + linear attn projections)
- **Batched RMSNorm**: `fused_rmsnorm_batched` — M blocks × 128 threads, saves M-1 launches/layer
- **Batched attention**: single kernel for all M sequences vs loop of single-seq calls
- **CUDA Graph**: 3.55× speedup in benchmark (no head_norm/RoPE)
- **Fused kernels**: RMSNorm+quant, RMSNorm+quant+pack, SwiGLU+quant combined
- **Device-side seq_pos**: graph-safe RoPE via `update_decode_seq_pos` + `d_seq_pos` device pointer
- **GatedDeltaNet SSM**: depthwise conv1d + recurrent step for linear attention layers

## Architecture (1.7B server)

Qwen3-1.7B decode flow per layer:
```
input layernorm → quantize → QKV GEMV → head_norm (Q,K) → RoPE → attention → Wo → residual1
post-attention layernorm → quantize → SwiGLU → down → residual2
```
Per-layer weights: `{L}_input_layernorm.f32`, `{L}_post_attention_layernorm.f32`, `qk_norms.f32` (Q/K head norms). RoPE: `rope_theta=1000000`.

## Project structure

```
blackwell/
├── CMakeLists.txt
├── AGENTS.md                 # detailed technical reference
├── HANDOFF.md                # session continuity doc
├── src/kernels/             # .cu implementations (191 symbols in lib)
├── include/blackwell/        # public headers
├── bench/                    # benchmarks and correctness tests
├── server/
│   ├── inference_server_nofp4.cu   # C++ inference daemon (stdin/stdout JSON)
│   ├── http_subprocess.cpp          # C++ HTTP server (httplib, OpenAI API)
│   └── http_server.py              # Python HTTP fallback
└── Dockerfile                 # production container (v0.4.1)
```

## Build

```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Build server binaries
nvcc -O3 -std=c++17 -arch=sm_120a \
  server/inference_server_nofp4.cu build/libblackwell_kernels.a \
  -I include -o server/inference_server -lcudart -lpthread -lz

# Build HTTP server
g++ -O2 server/http_subprocess.cpp -I include -o server/http_subprocess \
  -lpthread -lz -lssl -lcrypto
```

## Run

```bash
killall hashcat 2>/dev/null  # MUST before any measurement

# HTTP server (production, v0.4.1)
./server/http_subprocess weights_int8_bf16 2>&1 &
# Endpoints:
curl http://localhost:8123/health
curl -X POST http://localhost:8123/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","max_tokens":5}'
curl -X POST http://localhost:8123/v1/completions/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","max_tokens":5}'  # SSE streaming

# Direct pipe (debugging)
echo '{"prompt":"hi","max_tokens":3}' | ./server/inference_server weights_int8_bf16

# 1.7B benchmarks
./bench/decode_int8_cgraph 28                        # M=1: 163 t/s (no head_norm/RoPE)
./bench/decode_int8_nofp4 28 8                       # M=8: 575 t/s CUDA Graph (no head_norm/RoPE)
./bench/text_generate "The capital of France is" 30   # Correctness check

# 8B benchmark
./bench/decode_int8_cgraph_qwen3_8b 36               # M=1: 46 t/s

# 9B GatedDeltaNet benchmark
./bench/decode_qwen35_9b weights_int8_qwen35_9b 20   # M=1: 45.7 t/s
./bench/decode_qwen35_9b_batched_v2 8 20              # M=8: 52.1 t/s (batched GEMV + RMSNorm)

# Docker
docker build -t blackwell-server .
docker run --gpus all -p 8080:8080 blackwell-server
```

## HTTP API (OpenAI-compatible)

```bash
# Completions
curl -X POST http://localhost:8123/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","max_tokens":5,"temperature":0}'

# Streaming (SSE, token-by-token)
curl -X POST http://localhost:8123/v1/completions/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","max_tokens":5}'

# Chat completions (uses <|im_start|>/<|im_end|> tokens)
curl -X POST http://localhost:8123/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hi"}],"max_tokens":10}'

# Models
curl http://localhost:8123/v1/models

# Health
curl http://localhost:8123/health
```

## Correctness

"The capital of France is" → tokens `[12095, 11, 264, 892, 374]` = " Paris, a which is"
Identical to `text_generate.cu` greedy output. ✅

## Known limitations

- **Sub-8-bit quality dead**: INT4/INT5 produce garbled text after 28 layers (23-29 dB PSNR, attention softmax amplifies noise)
- **llama.cpp GGUF not supported**: uses separate weight files
- **M>8 not viable**: register pressure in batched GEMV
- **9B bandwidth-bound**: MLP weights (200 MB/layer) exceed L2 cache. Further optimization has negligible room.

## INT4/INT5 quality failure modes

All sub-8-bit paths tested and failed:
- Symmetric INT4: 23 dB PSNR → garbled
- Asymmetric INT4: 23 dB PSNR → garbled
- FP32×INT4 (weight-only): 23 dB PSNR → garbled
- FP32×INT5 (weight-only): 29 dB PSNR → garbled
- Per-channel INT4: 16 dB → worse than block-16
- Mixed INT4 attn + INT8 MLP → garbled

Root cause: attention softmax amplifies quantization noise — 23 dB/GEMV compounds to ~5 dB at lm_head.