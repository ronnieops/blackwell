# Code Context: Blackwell Server Analysis

## Files Retrieved
1. `server/http_subprocess.cpp` (lines 1-817) — HTTP wrapper, forks inference subprocess, IPC via pipe+temp-file, all endpoint handlers, rate limiter, metrics
2. `server/inference_server_int4.cu` (lines 1-655) — INT4 8B Qwen3 inference daemon, JSON stdio protocol, prefill+decode, streaming SSE
3. `server/inference_server_int4_batched.cu` (lines 1-386) — INT4 8B "batched" variant (sequential per-sequence, pre-loaded FP32 embedding)
4. `server/inference_server_nofp4.cu` (lines 1-919) — INT8 1.7B/8B server, CUDA Graph, batched prefill+decode, FP32 residual
5. `server/http_server.py` (lines 1-106) — Python fallback HTTP wrapper, minimal
6. `server/inference_server_gemma.cu` (lines 1-60) — Gemma 4 12B INT4 daemon (partial read)
7. `Dockerfile` (lines 1-50) — INT8 multi-model image, v0.8.1
8. `Dockerfile.int4` (lines 1-49) — INT4 multi-model image, v0.10.0
9. `docker-compose.yml` (lines 1-43) — Multi-model compose, v0.7.0 images

---

## 1. Server Architecture

### Subprocess Model
- `http_subprocess.cpp:160-170`: `fork()` + `pipe()` + `dup2()` for stdin/stdout
- Single global `SubprocessEngine g_engine` (line 375) — **one inference process for all requests**
- IPC: JSON written to temp file (`/tmp/inf_req_PID`), piped to subprocess stdin; response read from stdout via `fgets()`
- Protocol: newline-delimited JSON. Input: `{"prompts":[...],"max_tokens":N,...}`. Output: `{"tokens":[[...]],"text":[...]}` or SSE `data: {...}\n\n`

### Request Lifecycle
1. HTTP request → httplib thread pool dispatches handler
2. Rate limiter check (`g_rate_limiter.allow()`, 5 req/s burst 10)
3. `ensure_subprocess_alive()` — `waitpid(WNOHANG)` check, restart if dead
4. Acquire `g_engine.lock` mutex (`generate()` line 183, `generate_batch()` line 304)
5. Write JSON request to temp file → pipe to subprocess stdin
6. `select()` with 30s timeout (non-streaming) / 120s (streaming) to read response
7. Parse tokens from JSON, decode locally via `LocalTokenizer` (BPE), return OpenAI-format JSON

### Concurrency
- httplib uses **thread pool** (`CPPHTTPLIB_THREAD_POOL_COUNT = max(8, hw_concurrency-1)` per `httplib.h:162`)
- BUT `SubprocessEngine::lock` mutex serializes ALL requests through single subprocess (line 183, 304)
- **Net effect: only 1 request in-flight at a time** regardless of thread pool
- Streaming endpoint (`/v1/completions/stream`, line 721) bypasses mutex entirely — writes directly to `g_engine.get_write_fd()` → **race condition with concurrent requests**

---

## 2. Protocol Gaps (OpenAI API Compatibility)

### Endpoints Present
| Endpoint | Status | File:Line |
|----------|--------|-----------|
| `GET /health` | ✅ Basic (GPU mem, uptime, req count) | http_subprocess.cpp:537 |
| `GET /metrics` | ✅ Prometheus (4 metrics) | http_subprocess.cpp:567 |
| `GET /v1/models` | ✅ Model list | http_subprocess.cpp:599 |
| `POST /v1/completions` | ✅ Basic | http_subprocess.cpp:645 |
| `POST /v1/chat/completions` | ✅ Basic (Qwen3/Llama templates) | http_subprocess.cpp:606 |
| `POST /v1/batch` | ✅ Custom (not OpenAI) | http_subprocess.cpp:690 |
| `POST /v1/completions/stream` | ✅ Custom (not OpenAI path) | http_subprocess.cpp:719 |

### Missing / Broken
| Feature | Status | Details |
|---------|--------|---------|
| **Streaming for /v1/chat/completions** | ❌ Missing | Only `/v1/completions/stream` has SSE. Chat endpoint ignores `stream` param (line 621 checks but chat doesn't) |
| **Streaming for /v1/completions** | ⚠️ Non-standard | Must use `/v1/completions/stream` instead of `stream:true` in body. Wrong endpoint path for OpenAI compat |
| **Tool calls / function calling** | ❌ Missing | No support anywhere |
| **Logprobs** | ❌ Missing | No support |
| **n > 1** (multiple completions) | ❌ Missing | Hardcoded single completion |
| **Stop sequences** | ❌ Missing | Only EOS token (151643) stops generation |
| **Structured output / JSON mode** | ❌ Missing | |
| **`echo` (return prompt)** | ❌ Missing | |
| **`best_of`** | ❌ Missing | |
| **`suffix`** | ❌ Missing | |
| **Token usage accuracy** | ❌ Broken | `prompt_tokens` hardcoded to 0 or 1 (line 637: `"prompt_tokens":1`, line 679: `"prompt_tokens":0`). Never computed from actual tokenization |
| **`top_p` (nucleus sampling)** | ❌ Missing | Only `top_k` supported |
| **`presence_penalty` / `frequency_penalty`** | ❌ Missing | Only `repetition_penalty` (custom param) |
| **`seed`** | ❌ Missing | Hardcoded `0xdeadbeef` seed (inference_server_int4.cu:506) |
| **`max_tokens` enforcement** | ⚠️ Partial | Clamped to [1,2048]. No context length validation — if prompt + max_tokens > MAXSEQ, silent overflow |
| **Multiple messages in chat** | ❌ Broken | `extract_chat_content()` (line 414) returns FIRST `"content"` found — ignores system messages, multi-turn, assistant context |
| **Chat templates** | ⚠️ Limited | Only system+user single-turn. No multi-turn, no system prompt customization from request |

---

## 3. Performance

### Connection Handling
- **Thread-per-request** via httplib ThreadPool (default 8+ threads)
- httplib uses blocking sockets, not epoll/kqueue — fine for low concurrency
- Keep-alive: default httplib settings (not configured)

### Batching Across Requests
- ❌ **No continuous batching.** Single global mutex serializes all requests. Only `/v1/batch` endpoint does batched inference (multiple prompts in one subprocess call)
- The `/v1/batch` endpoint sends M prompts together, but inference_server processes them **sequentially** (`for (size_t pi = 0; pi < str_prompts.size(); pi++)` in inference_server_int4.cu:558)
- `inference_server_int4_batched.cu` also sequential: `for (size_t i = 0; i < prompts.size(); ++i) { generate_one(...) }` (line 358)
- The "batched" name is misleading — it's sequential generation with a single response

### KV Cache
- ❌ **No KV cache reuse across requests.** `generate()` clears cache at start (`cudaMemset(d_kc,0,...)` inference_server_int4.cu:482)
- No prefix caching for shared system prompts
- MAXSEQ=2048 (inference_server_int4.cu) / 4096 (batched variant) — mismatch

### Embedding
- inference_server_int4.cu: **CPU-side dequantization** per token (`dequant_embed_row()` line 87, H2D copy each step) — avoidable overhead
- inference_server_int4_batched.cu: Pre-loads FP32 embedding table to GPU (line 193), D2D copy instead — **better but not used by primary server**

---

## 4. Robustness

### Error Handling
- `die()` macro: any CUDA error → `exit(1)` (inference_server_int4.cu:26). **Subprocess crashes on any CUDA error**
- `ensure_subprocess_alive()` (http_subprocess.cpp:372): detects dead subprocess via `waitpid(WNOHANG)`, restarts. Good.
- But: **restart loses all state** (no warmup, no weight reload confirmation timing — just `sleep(2)` after start, line 530)
- `fread()` return values: checked in inference_server_int4.cu (truncation detection), **NOT checked** in inference_server_int4_batched.cu (void cast, line 152-153)

### Under Load / Backpressure
- Rate limiter: 5 req/s, burst 10 (line 35). Returns 429 when exceeded
- **No queue.** Concurrent requests serialize on mutex. If one request takes 18s (30 tokens at 8B), all others wait
- No backpressure mechanism — requests pile up in httplib's thread pool, each blocked on mutex

### OOM Handling
- ❌ **No GPU OOM handling.** `cudaMalloc` failures go through `die()` → `exit(1)` → subprocess crash → restart attempt
- If model too large for GPU, restart loop (crash → restart → crash)
- No pre-flight memory check

### Input Validation
- `max_tokens` clamped [1,2048] (line 662)
- No prompt length validation — prompt + max_tokens can exceed MAXSEQ silently
- JSON parsing: hand-rolled `json_string_at()` / `json_int_at()` — **no proper JSON parser**. Fragile: nested objects, arrays, escaped chars may break
- `/v1/batch` prompt parsing (line 700): hand-rolled string parser, may fail on complex JSON
- Payload limit: 1MB (line 534)

---

## 5. Deployment

### Docker Quality
| Issue | Severity | Details |
|-------|----------|---------|
| **Outdated images** | HIGH | Dockerfile pins v0.8.1, Dockerfile.int4 pins v0.10.0, docker-compose uses `v0.7.0`. Current code is v0.11.x |
| **No HEALTHCHECK** | MEDIUM | No `HEALTHCHECK` directive in either Dockerfile. Rely on external monitoring |
| **CUDA libs hack** | MEDIUM | `COPY cuda-libs/libcudart.so.13*` from host — fragile, version mismatch risk |
| **No multi-stage build** | LOW | Runtime image includes build artifacts, though binaries are pre-compiled |
| **Insecure apt flags** | MEDIUM | `--allow-unauthenticated`, `AllowInsecureRepositories` — suppresses package verification |
| **docker-compose missing INT4** | MEDIUM | No `blackwell-int4` service. Uses old v0.7.0 image for all services |
| **No resource limits** | LOW | No `deploy.resources.limits` (memory, CPU). GPU sharing via `CUDA_VISIBLE_DEVICES=0` — all containers share one GPU |

### Health Checks
- `GET /health` returns JSON (not for Docker HEALTHCHECK format)
- No readiness vs liveness distinction
- No model-warmup indicator — server accepts requests before subprocess ready (`sleep(2)` heuristic)

### Metrics Coverage
Present:
- `blackwell_requests_total` (counter)
- `blackwell_errors_total` (counter)
- `blackwell_latency_ms` (gauge — avg only, no histogram)
- `blackwell_uptime_seconds` (gauge)

Missing:
- ❌ GPU memory gauge (`blackwell_gpu_memory_used_bytes`, `blackwell_gpu_memory_total_bytes`)
- ❌ GPU utilization
- ❌ Request latency histogram/percentiles (p50, p95, p99)
- ❌ Tokens generated counter (`blackwell_tokens_generated_total`)
- ❌ Time-to-first-token (TTFT) for streaming
- ❌ Active requests gauge
- ❌ Rate limit rejections counter
- ❌ Subprocess restart counter
- ❌ Prefill vs decode latency breakdown
- ❌ KV cache utilization

### Graceful Shutdown
- ❌ **No SIGINT/SIGTERM handler.** Process killed → orphaned subprocess. `g_engine.stop()` only called from destructor (line 143), which may not run on SIGTERM
- `svr.listen()` blocks forever (line 815). No `svr.stop()` path
- Inference subprocess: infinite `while(true)` loop (inference_server_int4.cu:530). Killed by parent's `SIGKILL` (http_subprocess.cpp:357)

---

## 6. Safety

### Rate Limiting
- Token bucket: 5 req/s, burst 10 (line 35). Applied to all POST endpoints
- ⚠️ **Global limiter, not per-IP.** Single abusive client exhausts budget for all clients
- Not configurable at runtime (hardcoded)
- `GET /health` and `GET /metrics` not rate limited (fine for monitoring)

### Input Validation
- Prompt content: no sanitization beyond JSON escaping. Prompt injection surface = full (LLM output is the only defense)
- `max_tokens` clamped but no total context length check
- No API key / authentication on any endpoint (open server, binds `0.0.0.0`)
- Temperature not bounded (negative values accepted)
- `top_k` not bounded (negative values accepted, `0` = disabled)

### Max Context Enforcement
- ❌ **Not enforced.** If prompt tokenizes to 2000 tokens + max_tokens=2048, decode loop runs past MAXSEQ boundary → buffer overflow in KV cache (silent memory corruption)
- MAXSEQ hardcoded per server variant (2048 vs 4096 mismatch between int4.cu and int4_batched.cu)

---

## 7. Observability

### Tracing
- ❌ No request tracing (no request IDs logged to subprocess, no distributed tracing)
- `fprintf(stderr, "[REQ] %s\n", ...)` in subprocess logs full request (inference_server_int4.cu:535) — useful for debug, noisy for production

### Per-Request Latency Breakdown
- ❌ Only end-to-end latency tracked (`g_total_latency_ms`)
- No prefill vs decode timing
- No per-kernel timing (though nsys profiles exist for benchmarks)
- No TTFT for streaming

### Token Counting
- ❌ **Wrong.** `prompt_tokens` hardcoded to 0 or 1 (http_subprocess.cpp:637, 679)
- `completion_tokens` = `tokens.size()` — correct for generated tokens
- `total_tokens` = prompt + completion — wrong because prompt is wrong
- No actual tokenizer call for prompt token counting (would need pre-tokenization)

---

## Ranked Gaps + Enhancement Opportunities

### P0 — Critical (Correctness / Security)

1. **Streaming endpoint race condition** — `http_subprocess.cpp:760-770`: writes to subprocess stdin WITHOUT acquiring `g_engine.lock`. Concurrent streaming request corrupts subprocess stdin for any in-flight non-streaming request. Fix: acquire lock in ContentProvider lambda before writing.

2. **No context length validation** — `inference_server_int4.cu:480-508`: prompt + max_tokens can exceed MAXSEQ. Decode loop writes past KV cache bounds → silent GPU memory corruption. Fix: `assert(input_ids.size() + max_new <= MAXSEQ)`, return 400 if exceeded.

3. **No graceful shutdown** — `http_subprocess.cpp:815`: `svr.listen()` blocks, no signal handler. SIGTERM → orphaned subprocess holding GPU memory. Fix: `signal(SIGTERM, handler)` → `svr.stop()` + `g_engine.stop()`.

4. **MAXSEQ mismatch** — `inference_server_int4.cu:24` MAXSEQ=2048 vs `inference_server_int4_batched.cu:22` MAXSEQ=4096. Same model, different limits. KV cache allocation differs. Pick one.

### P1 — High (API Compatibility / Performance)

5. **No continuous batching** — Single mutex serializes all requests. For a production server, need request queue → batched decode step → interleaved output. Architecture requires fundamental rework: subprocess must handle concurrent sequences sharing the decode loop.

6. **`prompt_tokens` hardcoded wrong** — `http_subprocess.cpp:637,679`. Should tokenize prompt and count. Breaks any usage-based billing or cost tracking.

7. **Multi-turn chat broken** — `http_subprocess.cpp:414`: `extract_chat_content()` returns first `"content"` value. Ignores system prompt, assistant history, multi-turn context. Clients sending conversation arrays get wrong results.

8. **Streaming not OpenAI-compatible** — `POST /v1/completions/stream` is custom path. OpenAI uses `POST /v1/completions` with `"stream":true`. Chat completions have no streaming at all. Fix: add `stream` param handling in both endpoints.

9. **Missing `top_p`, `stop` sequences, `seed`** — Core sampling params missing. `stop` sequences especially important for structured workflows.

10. **Hand-rolled JSON parser** — `json_string_at()`, `json_int_at()` etc (http_subprocess.cpp:424-460). Fragile, no nested object support, no array-of-objects for messages. Use rapidjson or nlohmann/json.

### P2 — Medium (Reliability / Observability)

11. **`fread()` unchecked in batched server** — `inference_server_int4_batched.cu:152-153`: `(void)fread(...)`. Truncated weight files → silent garbage → wrong output. Fix: check return values like inference_server_int4.cu does.

12. **No GPU OOM pre-flight check** — `cudaMemGetInfo()` before model load. Fail fast with clear error instead of crash loop.

13. **Metrics missing key gauges** — No GPU memory, no latency histogram, no token throughput, no active requests. Add Prometheus histogram for latency.

14. **Rate limiter not per-client** — Single global bucket. Add per-IP tracking (or at least per-API-key if auth added).

15. **No API authentication** — Server binds `0.0.0.0:8123` with no auth. Add Bearer token check.

16. **Docker images outdated** — Dockerfile (v0.8.1), Dockerfile.int4 (v0.10.0), docker-compose (v0.7.0). Current = v0.11.x. All three need rebuild.

17. **No Docker HEALTHCHECK** — Add `HEALTHCHECK CMD curl -f http://localhost:8080/health || exit 1`.

### P3 — Low (Polish)

18. **CPU-side embedding dequant** — `inference_server_int4.cu:87,488`: `dequant_embed_row()` on CPU + H2D copy per token. Batched server already pre-loads FP32 embedding to GPU (D2D copy). Port optimization to primary server.

19. **No KV cache reuse** — Common system prompts re-computed every request. Prefix caching would cut TTFT significantly.

20. **`finish_reason` always `"stop"`** — Should be `"length"` when max_tokens reached, `"stop"` when EOS hit. Currently always `"stop"` regardless.

21. **Repetition penalty default mismatch** — `http_subprocess.cpp:621` defaults to `1.5f`, but inference server defaults to `1.5f` too. AGENTS.md says default should be `1.0` (off). Non-default penalty changes output character.

22. **Temperature/top_k not validated** — Negative temperature, negative top_k accepted without error.

---

## Start Here
`server/http_subprocess.cpp` — the HTTP layer. All API gaps, concurrency issues, and observability gaps live here. The streaming race (line 760) and missing graceful shutdown (line 815) are the most impactful fixes.

## Supervisor coordination
No blockers. Analysis complete. All findings in scout_servers.md.
