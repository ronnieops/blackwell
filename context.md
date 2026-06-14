# Code Context — Blackwell Server Architecture

## Server Architecture

```
http_subprocess (port 8123, model_name)
  │
  ├─ fork() → inference_server_int4 subprocess
  │   IPC: JSON via temp file → pipe write; JSON response via pipe read
  │
  ├─ httplib::Server (thread pool)
  │   GET  /health          — GPU memory, uptime, request stats
  │   GET  /metrics         — Prometheus format (requests, errors, latency, uptime)
  │   GET  /v1/models       — model list
  │   POST /v1/completions   — text completion (rate limited, restart checked)
  │   POST /v1/chat/completions — chat with template (rate limited)
  │   POST /v1/batch         — batch completion (up to 8 prompts)
  │   POST /v1/completions/stream — SSE streaming
  │
  └─ SubprocessEngine
      ├─ start(model) — fork/exec, pipe setup, LocalTokenizer init
      ├─ generate(prompt, ...) — send JSON, read response, decode tokens
      ├─ generate_batch(prompts, ...) — batched decode
      ├─ ensure_subprocess_alive() — waitpid check, auto-restart on crash
      └─ RateLimiter — token bucket (5 req/s, burst 10), mutex-protected
```

## INT4 Decode Pipeline (per layer)

```
input → RMSNorm → INT4 quant → QKV GEMV (batched) → head_norm(Q,K) → RoPE
  → KV cache write → attention (GQA) → Wo GEMV → residual add
  → RMSNorm → INT4 quant → gate/up GEMV → SwiGLU → down GEMV → residual add
```

## Multi-chunk Prefill Pipeline

```
prefill_tokens_batched(token_ids, offset=0, M)
  while M > 0:
    chunk = min(M, MAX_BATCH=16)
    embed chunk tokens → batched QKV → head_norm + RoPE (per-token)
    KV cache write (update_kv_cache_pos) → attention (per-token, _pos variant)
    Wo → residual → RMSNorm → gate/up → SwiGLU → down → residual
    offset += chunk; M -= chunk
  copy last token hidden state → d_x32 → lm_head
```

## KV Cache Layout

```
d_kc: [NL][nkv][MAXSEQ][hd]  — per-layer, per-head, per-position
d_vc: [NL][nkv][MAXSEQ][hd]
Index: l * nkv * MAXSEQ * hd + h * MAXSEQ * hd + pos * hd + d
```

## Key Dimensions (Qwen3-8B)

| Param | Value |
|-------|-------|
| H (hidden) | 4096 |
| I (intermediate) | 12288 |
| NL (layers) | 36 |
| nqh (query heads) | 32 |
| nkv (KV heads) | 8 |
| hd (head dim) | 128 |
| V (vocab) | 151936 |
| MAXSEQ | 2048 |
| MAX_BATCH | 16 |

## Kernel Library (198 symbols)

Production INT4 path:
- `gemv_int4_batched` — batched INT4 GEMV (M=1-16, template switch)
- `fused_rmsnorm_batched` — RMSNorm (1 block/sequence)
- `quantize_int4_batched` — FP32→INT4 with block-16 scales
- `update_kv_cache_pos` — KV cache write (direct seq_pos, no H2D race)
- `attention_decode_batched_gqa_pos` — GQA attention (direct seq_pos)
- `apply_swiglu` — silu(gate)×up
- `head_norm_kernel` — Q/K head normalization
- `apply_rope_kernel` — rotary position embedding
- `sample_gpu` — softmax + sampling
- `apply_repetition_penalty` — token logit penalty

## Files

```
server/http_subprocess.cpp     — HTTP server + SubprocessEngine
server/inference_server_int4.cu — INT4 inference daemon
src/kernels/gemv_int8.cu       — INT8/INT4 GEMV kernels
src/kernels/decode.cu          — Attention + KV cache kernels
src/kernels/attention.cu       — attention_prefill_v3
src/kernels/norm.cu            — RMSNorm kernels
include/blackwell/kernels.h    — API declarations
```
