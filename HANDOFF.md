# Handoff — Session 82+ (2026-06-13)

## Project Status: PRODUCTION-READY

Custom CUDA kernels for INT8/INT4 LLM inference on RTX 5060 Ti (Blackwell GB206).
All major optimization paths explored. Production path solid.

---

## Production Server

```bash
killall hashcat 2>/dev/null  # MUST DO BEFORE ANY MEASUREMENT
./server/http_subprocess qwen3-8b &
curl http://localhost:8123/health
curl -X POST http://localhost:8123/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","max_tokens":30}'
```

**Endpoints**: `/health`, `/metrics` (Prometheus), `/v1/completions`, `/v1/chat/completions`,
`/v1/batch`, `/v1/completions/stream`

**Server hardening**: Rate limiting (5 req/s, burst 10), subprocess auto-restart on crash,
payload size limit (1MB), max_tokens clamp [1,2048], snprintf overflow protection,
escape_json control-char handling, mkstemp temp files.

## Benchmarks (Qwen3-8B INT4, RTX 5060 Ti)

| Config | t/s | ms/tok | vs llama.cpp |
|--------|-----|--------|-------------|
| llama.cpp Q4_K_M | 84 | 11.9 | 1.0× baseline |
| Blackwell M=1 | 56 | 17.9 | 67% |
| Blackwell M=8 | 119 | 8.4 | 142% |
| Blackwell M=48 | 154 | 6.5 | 183% |

**PPL**: 23.52 (1.9× BF16). AWQ α=0.6: 21.82.

## Multi-chunk Prefill — FIXED (Session 82)

Multi-chunk prefill now works for prompts of any length.

**Root cause**: `update_kv_cache` and `attention_decode_batched_gqa` used a shared pinned
host buffer for async H2D copy of `seq_pos`. In tight prefill loops, CPU overwrote the
buffer before the async memcpy read it.

**Fix**: Added `_pos` kernel variants with direct seq_pos argument:
- `update_kv_cache_pos()` — KV cache write, no H2D copy
- `attention_decode_batched_gqa_pos()` — attention, no H2D copy

## Kernel Library

198 exported symbols in `build/libblackwell_kernels.a`.

**Build**:
```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

**Build server**:
```bash
nvcc -O3 -std=c++17 -arch=sm_120a server/inference_server_int4.cu \
  build/libblackwell_kernels.a -I include -lcudart -lpthread -lz \
  -o server/inference_server_int4
```

**Build HTTP wrapper**:
```bash
g++ -O2 server/http_subprocess.cpp -I include /tmp/httplib.o \
  -lpthread -lz -lssl -lcrypto -o server/http_subprocess
```

## Abandoned Paths (Do Not Retry)

| Path | Reason |
|------|--------|
| INT2 8B | PPL 47B (3.8M× worse than INT4). Activation quant accumulation. |
| NVFP4 E2M1 | PPL 24,850. Double quantization shifts weights. Format mismatch. |
| FP8 per-row | PPL 41.75, 4.5× slower than INT8. No dp4a. |
| INT4/INT5 1.7B | All sub-8-bit dead after 28+ layers. |
| Speculative decoding | 1.3% acceptance rate. INT4 draft quality insufficient. |
| Llama 3.1 8B INT4 | PPL 273K. Model-specific precision requirement. |
| 9B GatedDeltaNet | SSM instability (A_log > 0 → exp growth). Blocked. |
| TMA weight streaming | Confirmed possible, 2-3 day effort, uncertain 0-15% gain. |
| WGMMA | 10-20% at M≥4 only. Batched already 148 t/s. |
| CUDA Graph | 2.1% gain, ceiling understood. GEMV dominates (92%). |

## Remaining Work (Low Priority)

1. **New model port** — GGUF bridge ready (DeepSeek, Phi-4, Mistral)
2. **TMA prototype** — Confirmed available on GB206. 2-3 days, uncertain payoff.
3. **Write up / paper** — All data collected and verified.

## Key Files

```
server/
  http_subprocess.cpp        — HTTP server (httplib, rate limit, restart, Prometheus)
  http_subprocess            — compiled binary
  inference_server_int4.cu   — INT4 8B inference daemon (multi-chunk prefill)
  inference_server_int4      — compiled binary
  inference_server_gemma.cu  — Gemma 4 12B INT4 server

src/kernels/
  gemv_int8.cu               — INT8/INT4 GEMV (warp, batched, splitk, fused)
  decode.cu                  — Attention (GQA, batched, KV cache, _pos variants)
  attention.cu               — attention_prefill_v3 kernel
  norm.cu                    — RMSNorm + quant fusions

bench/
  text_generate_int4_qwen3_8b.cu — 8B end-to-end (59 t/s)
  text_generate_int4_batched.cu  — Batched (M=1:61, M=48:154 t/s)

include/blackwell/kernels.h  — 198 kernel declarations
AGENTS.md                    — Full project reference
DOCS.md                      — Public-facing documentation
research.md                  — Optimization research + recommendations
```
