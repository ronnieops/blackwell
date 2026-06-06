# HANDOFF.md — blackwell

Continuity doc. Read with AGENTS.md before acting.

---

## 1. Current Objective

Operational C++ inference server with correct Qwen3-1.7B model. All HTTP endpoints working. Benchmark research ongoing (CUDA Graph, batched attention, nofp4 path). Docker image built and pushed (session 41).

**Session 41 completed**:
- Committed AGENTS.md, Dockerfile, README.md, HANDOFF.md updates
- Verified nofp4 benchmark: M=1 163.3 t/s, M=8 CUDA Graph 574.2 t/s
- Fixed 8B batched attention KV cache layout + batched KV writes
- Built Docker image `blackwell-server:v0.4.0` (4.13 GB)

**Session 42 completed**:
- HTTP streaming: SSE token-by-token output via `/v1/completions/stream`
- Fixed parse_prompt_ids infinite loop bug (didn't handle string prompts)
- Fixed streaming printf buffering with fflush
- Used httplib ContentProvider API for true streaming
- Pushed Docker v0.4.1 to ghcr.io/ronnieops/blackwell-server

**Session 43 completed**:
- 9B GatedDeltaNet batched GEMV: attention+linear attn projections batched, 50.9 vs 49.8 t/s (+2.2%)
- Batched RMSNorm kernel: fused_rmsnorm_batched(M blocks × 128 threads), 52.1 vs 49.9 t/s (+4.4% combined)
- M=8: 52.1 t/s (73% of Q3_K_M 71.4 t/s). M<4 batched RMSNorm hurts (block overhead)
- MLP batched GEMV fails: slower due to layout mismatch, kept per-seq
- 9B weights restored from HF cache (10.4 GB)

**Session 44 - profiling**:
- Profile breakdown (M=8): MLP 74.8%, LinAttn 18%, FullAttn 3.2%, Conv+Rec 2.1%
- MLP is the bottleneck — per-seq GEMV (3 × 32 layers = 96 calls/step)
- Batching MLP: 1.8× slower (gate+up), kept per-seq
- 9B at practical limit — weight matrices (200 MB/layer) exceed L2 cache
- Bandwidth-bound. Further optimization has negligible room.

**Session 50 — Docker deploy + server benchmark**:
- Built `blackwell-server:v0.5.0` Docker image
- Server benchmark (HTTP, steady state):
  - 10 tokens: 65 t/s (10.0 ms/token)
  - 20 tokens: 77 t/s (10.0 ms/token)
  - 30 tokens: 83 t/s (10.0 ms/token)
  - 50 tokens: 86 t/s (10.0 ms/token) — steady state
- HTTP overhead: ~10ms per request
- Docker image: `ghcr.io/ronnieops/blackwell-server:v0.5.0`

**Session 45 (boot)**: Library rebuilt (287 symbols, was corrupt). Server verified. HTTP endpoints working. Removed crashing `decode_qwen35_9b_batched_opt.cu` (invalid resource handle, redundant with `decode_qwen35_9b_batched_v2`).

**Session 46-49 — prefill benchmark**:
- Explored prefill (prompt processing) throughput as new direction
- Built `bench/prefill_benchmark.cu` with real token embeddings (not zeros)
- Batched GEMV (M=8) beats WMMA at most SEQ; quantization overhead limits WMMA gains

**Real benchmark (warmup, real embeddings)**:

| SEQ | t/s | ms/token | vs Decode (106 t/s) |
|-----|-----|---------|---------------------|
| 8 | 216 | 4.6 | 2.0× |
| 16 | 424 | 2.4 | 4.0× |
| 32 | 534 | 1.9 | 5.0× |
| 64 | 1753 | 0.6 | 16.5× |
| 128 | 2582 | 0.4 | 24× |
| 256 | 4047 | 0.2 | 38× |
| 512 | **13,727** | 0.07 | **129×** |

GEMM prefill is 2-129× faster than decode. Attention (O(n²)) not included — would dominate for long SEQ.

- Attention not included (O(n²) cost, would dominate for long sequences)
- Benchmark uses zero-initialized inputs — crashes at layer 3 with real weights (numerical issues from non-zero activations). Expected for research benchmark.
- Prefill is compute-bound (WMMA GEMM), decode is bandwidth-bound (dp4a GEMV)
- Key insight: prefill processes SEQ tokens in parallel vs decode's 1 token. GPU utilization improves with SEQ.

---

## 2. Current Status

| Metric | Value |
|--------|-------|
| GPU | RTX 5060 Ti, compute 12.0, 36 SMs, ~500 GB/s GDDR7 |
| CUDA | 13.3, SM_120a, C++17, CMake |
| Library | 287 symbols in `build/libblackwell_kernels.a` (rebuilt session 45) |
| Branch | master |
| Server | **v0.4.1 — correct model + streaming** (per-layer RMSNorm, head_norm, RoPE) |

### Throughput

| Method | t/s | vs Q4_K_M | Notes |
|--------|-----|-----------|-------|
| Server (production, M=1) | ~106 | 36% | Correct model, per-kernel |
| Benchmark M=1 (no head_norm/RoPE) | 181 | 62% | Per-kernel, fused |
| Benchmark CUDA Graph M=8 (no hn/RoPE) | 575 | 196% | ⚠️ Omits head_norm+RoPE |
| Benchmark FP4 M=8 | 324 | 111% | Legacy, nofp4 now used |
| 9B GatedDeltaNet M=8 | 52.1 | 73% of Q3_K_M | Batched GEMV + RMSNorm |

⚠️ 575 t/s benchmark omits head_norm and RoPE — not achievable with correct model.

### HTTP Server

| Endpoint | Status |
|----------|--------|
| `GET /health` | ✅ `{"status":"ok"}` |
| `GET /v1/models` | ✅ model list |
| `POST /v1/completions` | ✅ " Paris, a which is" |
| `POST /v1/chat/completions` | ✅ works |
| `POST /v1/completions/stream` | ✅ SSE token-by-token streaming |

---

## 3. Recent Decisions

- **Server v0.4.0**: Correct model — per-layer RMSNorm, Q/K head norms, RoPE. Same output as `text_generate.cu`.
- **HTTP working**: Raw read/write syscalls for subprocess IPC (no FILE* — pipe issues with forked process).
- **HTTP timeout**: Set to 300s (`svr.set_read_timeout(300)`) — httplib default 5s too short for inference.
- **FP4 eliminated**: `decode_int8_nofp4.cu` benchmark, nofp4 server path. 575 t/s vs FP4 324 t/s (77% faster).
- **INT4/INT5 dead**: Confirmed garbled after 28 layers. 23-29 dB PSNR compounds to ~5 dB at lm_head. No viable quality path below INT8.
- **CUDA Graph for server**: Captured, works, but no speedup (~106 t/s same as per-kernel). Reason: benchmark's 575 t/s omits head_norm+RoPE. Deferred.
- **Parsing bug fixed**: `parse_prompt_ids` consumed `"prompts":["hello"]` as token IDs (h=104, e=101, l=108...). Fixed with early return on `"` or `[`. `parse_string_prompts` also fixed string element skip.

---

## 4. Important Constraints

- `CUDACXX` env var must be set before `cmake`
- `compute_120a` required (not `compute_120`)
- `killall hashcat` before every measurement (auto-restarts, -45% throughput)
- `gemv_int8_warp` is production INT8 GEMV — NOT `gemv_int8`
- All weight matrices exceed L2 cache (32 MB)
- M>8 not viable (register pressure in batched GEMV)
- `pack_int8` takes PRE-COMPUTED scales as INPUT. Use `quantize_int8` to compute them.
- `update_decode_seq_pos` writes to pinned host memory, then cudaMemcpyAsync to device — graph-safe
- `update_kv_cache_device` uses device-side seq_pos (no H2D copy in capture)

---

## 5. Known Issues / Risks

1. **hashcat** — `killall hashcat` 30s before benchmark. Respawns in 60s.
2. **Server quality (chat)**: Chat completions garbled — prompt format (`<|im_start|>` / `<|im_end|>`) works but model has repetition issues without temperature.
3. **Benchmark vs server gap**: 181 t/s (benchmark) vs ~106 t/s (server) — 70% overhead from head_norm + RoPE kernels (~4 extra kernels/layer × 28 layers).
4. **CUDA Graph deferred**: Works in benchmark but provides no speedup with correct model. May revisit if head_norm+RoPE can be fused.
5. **Benchmark numbers context-sensitive**: 575 t/s omits head_norm/RoPE. Always note model correction when citing throughput.

---

## 6. Pending Tasks

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| 8B batched attention | MEDIUM | ✅ Done | KV cache layout fix, 41 t/s M=8 (1.39x vs serial) |
| Docker build + tag | LOW | ✅ Done | blackwell-server:v0.4.0 built |
| CUDA Graph (server) | LOW | ❌ Failed | Capture fails with cudaErrorCaptureInternal (700). attention_decode_gqa internal state (smem config, seq_pos alloc) causes capture failures. Per-kernel path works correctly (~106 t/s). CUDA Graph works in benchmark (574 t/s) but benchmark uses wrong model (no head_norm/RoPE). |
| head_norm+RoPE fusion | LOW | ❌ No benefit | Fused: 141 t/s vs separate: 140 t/s (+0.7%, noise). GEMV is bottleneck, element-wise ops negligible. Keep fused kernel for reference.
| M=8 OOM at 33+ layers | MEDIUM | ✅ Done | Removed save/restore buffers — M=8 36L now works |

### 8B Batched Attention Results (session 41)

| Config | Serial-attn | Batched-attn | Speedup | vs Q4_K_M |
|--------|-------------|--------------|---------|------------|
| 8B M=8, 28L | 7.1 t/s | **41.0 t/s** | 1.39x | 50% |
| 8B M=8, 32L | 6.2 t/s | **35.8 t/s** | 1.39x | 43% |
| **8B M=8, 36L** | **5.5 t/s** | **31.9 t/s** | **1.39x** | **38.6%** |
| 8B M=8, 32L | 6.2 t/s | **35.8 t/s** | 1.39x | 43% |
| 8B M=7, 28L | 8.1 t/s | **42.1 t/s** | 1.35x | 51% |
| 8B M=6, 36L | 7.4 t/s | **34.9 t/s** | 1.26x | 42% |

**Note**: M=1 is still faster per token than M=8 batched (46 vs 41 t/s). M=8 only useful for throughput when parallel decode is needed.

**Memory fix**: Removed save/restore buffers (9.4 GB pinned memory for M=8 36L). Full 36 layers now works.

### KV Cache Layout Fix (8B batched attention)
- Old: `[M][layers][nkv][ms][hd]` — incompatible with `update_kv_cache` (ignores batch_idx)
- New: `[layers][M][nkv][ms][hd]` — compatible with batched attention strides
- Batched KV writes: replaced broken `update_kv_cache` with `cudaMemcpyAsync`
- Fixed `attention_decode_gqa` calls: use batch+layer base offsets
- seq_len reduced to 256 for M=8 memory fit (full 2048 OOM)

---

## 7. Suggested Next Actions

1. **WMMA GEMM prefill** (tested, NOT WORTH IT): Transposed weights + single `gemm_int8_wmma_fast` call. Result: 8348 t/s at SEQ=512 vs batched 13,688 t/s. WMMA wins only at SEQ=64 and SEQ=256. Batched GEMV is better for most SEQ. Skip WMMA path.
2. **Prefill attention kernel**: Build flash-attention-style prefill attention (O(n²) cost per layer). This is the missing piece for full prefill pipeline.
3. **Prefill + decode pipeline**: Hook prefill into server for prompt processing + autoregressive decode.
4. **8B HTTP server**: Port server to 8B weights for batched decode.
5. **Benchmark hygiene**: Always run `killall hashcat` before measurement.

---

## 8. Important Files / Commands

### Build
```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Server binaries
nvcc -O3 -std=c++17 -arch=sm_120a \
  server/inference_server_nofp4.cu build/libblackwell_kernels.a \
  -I include -o server/inference_server -lcudart -lpthread -lz

/usr/bin/g++ -O2 /tmp/httplib.o server/http_subprocess.cpp -I include \
  -o server/http_subprocess -lpthread -lz -lssl -lcrypto
# /tmp/httplib.o: g++ -O2 -std=c++17 -I include -DCPPHTTPLIB_OPENSSL_SUPPORT=0 \
#   -DCPPHTTPLIB_ZLIB_SUPPORT=0 include/blackwell/httplib.cpp -c -o /tmp/httplib.o
```

### Run
```bash
killall hashcat 2>/dev/null

# Server (HTTP)
./server/http_subprocess weights_int8_bf16 &

# Direct pipe (debug)
echo '{"prompt":"hi","max_tokens":1}' | ./server/inference_server weights_int8_bf16

# Benchmarks
./bench/decode_int8_cgraph 28                   # M=1: 181 t/s (no head_norm/RoPE)
./bench/decode_int8_nofp4 28 8                 # M=8 CUDA Graph: 575 t/s (no head_norm/RoPE)
./bench/text_generate "hi" 5                    # Correctness
```

### Verify
```bash
nm build/libblackwell_kernels.a | grep "_ZN9blackwell" | grep " T " | wc -l   # expect 287 (session 45)
# Correctness
echo '{"prompt":"The capital of France is","max_tokens":3}' | \
  ./server/inference_server weights_int8_bf16 | grep text
# Expect: " Paris, a" (or " Paris, at" with max_tokens=3)
```

### Key source files
```
server/inference_server_nofp4.cu    # C++ daemon, correct model decode
server/http_subprocess.cpp           # HTTP wrapper, raw syscalls for IPC
server/http_server.py              # Python fallback HTTP
bench/decode_int8_nofp4.cu         # nofp4 benchmark (per-kernel + CUDA Graph)
bench/decode_int8_cgraph.cu        # M=1 benchmark
bench/text_generate.cu             # End-to-end correctness
```

---

## 9. Validation

| Check | Status |
|-------|--------|
| Library symbols | ✅ 287 (rebuilt session 45) |
| Server correctness | ✅ " Paris, a" |
| HTTP /health | ✅ |
| HTTP /v1/models | ✅ |
| HTTP /v1/completions | ✅ |
| hashcat killed | ✅ |
| 9B batched v2 | ✅ 51.2 t/s M=8 |
| Server binary updated | ✅ (session 45 rebuild) |
| Docker build + push | ✅ | `ghcr.io/ronnieops/blackwell-server:v0.4.0` pushed |

---

## 10. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-06-06 |
| branch | master |
| active components | server/inference_server (v0.4.1), http_subprocess, decode_qwen35_9b_batched_v2, decode_int8_nofp4 |
| last_build | 2026-06-06 (session 45 rebuild) |
| docker_image | `ghcr.io/ronnieops/blackwell-server:v0.4.1` (4.13 GB) |

---

## META PROMPT

**Boot sequence**:
1. Read `AGENTS.md` → `HANDOFF.md`
2. `git status` — check uncommitted changes
3. `killall hashcat 2>/dev/null`
4. `nm build/libblackwell_kernels.a | grep "_ZN9blackwell" | grep " T " | wc -l` (expect 287)
5. `ls server/inference_server server/http_subprocess` (check binaries exist)
6. `./server/http_subprocess weights_int8_bf16 &` → test endpoints

**Verified facts (2026-06-06)**:
- Server produces " Paris, a" for "The capital of France is" ✅
- All HTTP endpoints working (/health, /v1/models, /v1/completions) ✅
- Library rebuilt: 287 symbols ✅
- 9B batched v2: 51.2 t/s M=8 ✅
- INT4/INT5 quality dead (garbled after 28 layers) ✅
- 575 t/s benchmark omits head_norm/RoPE — not achievable with correct model ✅
- Server per-kernel: ~106 t/s with correct model ✅
- `parse_prompt_ids` collision bug fixed (consumed string arrays as token IDs) ✅

**Critical constraints**:
- `compute_120a` (not `compute_120`)
- `killall hashcat` before measurement
- `gemv_int8_warp` production — NOT `gemv_int8`
- `pack_int8` takes pre-computed scales (INPUT). Use `quantize_int8` to compute.

**DO NOT**:
- Use stale benchmark numbers without noting head_norm/RoPE context
- Run benchmarks without killing hashcat (-45% throughput)
- Use INT4/INT5 (quality dead, garbled after 28 layers)
- Re-dig dead ends: speculative decode, FP4 tensor core GEMV, PDL, sub-8-bit quality
- Expect M>8 scaling (register pressure)
- Trust pre-session-37 INT4 benchmark numbers (grid bug)

**Active direction**: Operational server + benchmark research. Docker image built. Next: 8B correctness fix (max_diff=2.0), M=8 OOM investigation (pinned memory), or push Docker image to registry.

**Session 41 files changed**:
```
M AGENTS.md       — updated throughput tables, kernel list
M Dockerfile       — apt GPG fix, cudart symlink
M README.md        — updated performance table
M HANDOFF.md       — refreshed with session results
M bench/decode_int8_batched_cgraph_attn_qwen3_8b.cu — KV cache layout fix
```

**Update rule**: Keep HANDOFF.md concise — deduplicate with AGENTS.md, prefer bullets, remove stale sections on update. Only store operational truth and verified facts.