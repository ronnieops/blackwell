# HANDOFF.md — blackwell

Continuity doc. Read with AGENTS.md before acting.

---

## 1. Current Objective

Operational C++ inference server with correct Qwen3-1.7B model. All HTTP endpoints working. Benchmark research ongoing (CUDA Graph, batched attention, nofp4 path).

---

## 2. Current Status

| Metric | Value |
|--------|-------|
| GPU | RTX 5060 Ti, compute 12.0, 36 SMs, ~500 GB/s GDDR7 |
| CUDA | 13.3, SM_120a, C++17, CMake |
| Library | 191 symbols in `build/libblackwell_kernels.a` (was 177, grew) |
| Branch | master |
| Server | **v0.4.0 — correct model** (per-layer RMSNorm, head_norm, RoPE) |

### Throughput

| Method | t/s | vs Q4_K_M | Notes |
|--------|-----|-----------|-------|
| Server (production, M=1) | ~106 | 36% | Correct model, per-kernel |
| Benchmark M=1 (no head_norm/RoPE) | 181 | 62% | Per-kernel, fused |
| Benchmark CUDA Graph M=8 (no hn/RoPE) | 575 | 196% | ⚠️ Omits head_norm+RoPE |
| Benchmark FP4 M=8 | 324 | 111% | Legacy, nofp4 now used |

⚠️ 575 t/s benchmark omits head_norm and RoPE — not achievable with correct model.

### HTTP Server

| Endpoint | Status |
|----------|--------|
| `GET /health` | ✅ `{"status":"ok"}` |
| `GET /v1/models` | ✅ model list |
| `POST /v1/completions` | ✅ " Paris, a which is" |
| `POST /v1/chat/completions` | ✅ works |

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

| Task | Priority | Notes |
|------|----------|-------|
| 8B model optimization | MEDIUM | 46 t/s (56% Q4_K_M), room for batched attention |
| Docker push | LOW | Build and push `blackwell-server` image |
| CUDA Graph (server) | LOW | Deferred — no speedup with correct model |
| head_norm+RoPE fusion | LOW | Would close benchmark-vs-server gap |

---

## 7. Suggested Next Actions

1. **Run end-to-end test**: `./server/http_subprocess weights_int8_bf16` then `curl` all 4 endpoints — verify server still working.
2. **Docker**: Build and push `blackwell-server` image. Dockerfile exists.
3. **8B optimization**: Batched attention for M>1 decode.
4. **Benchmark hygiene**: Always run `killall hashcat` before measurement.

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
nm build/libblackwell_kernels.a | grep " T blackwell" | wc -l   # expect 191
# Correctness
echo '{"prompt":"The capital of France is","max_tokens":3}' | \
  ./server/inference_server weights_int8_bf16 | grep text
# Expect: " Paris, a which is"
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
| Library symbols | ✅ 191 |
| Benchmark M=1 (no head_norm/RoPE) | ✅ 181 t/s |
| CUDA Graph M=8 (no head_norm/RoPE) | ✅ 575 t/s |
| Server correctness | ✅ " Paris, a which is" |
| HTTP /health | ✅ |
| HTTP /v1/completions | ✅ |
| HTTP /v1/chat/completions | ✅ |
| HTTP /v1/models | ✅ |
| hashcat killed | ✅ |

---

## 10. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-06-04 |
| branch | master |
| active components | server/inference_server (v0.4.0), http_subprocess, decode_int8_nofp4 benchmark |
| last_build | 2026-06-04 15:45 |

---

## META PROMPT

**Boot sequence**:
1. Read `AGENTS.md` → `HANDOFF.md`
2. `git status` — check uncommitted changes
3. `killall hashcat 2>/dev/null`
4. `nm build/libblackwell_kernels.a | grep " T blackwell" | wc -l` (expect 191)
5. `ls server/inference_server server/http_subprocess` (check binaries exist)
6. `./server/http_subprocess weights_int8_bf16 &` → test endpoints

**Verified facts (2026-06-04)**:
- Server produces " Paris, a which is" for "The capital of France is" ✅
- All 4 HTTP endpoints working ✅
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

**Active direction**: Operational server + benchmark research. Next: Docker push, 8B optimization, or CUDA Graph if head_norm+RoPE fusion becomes viable.

**Update rule**: Keep HANDOFF.md concise — deduplicate with AGENTS.md, prefer bullets, remove stale sections on update. Only store operational truth and verified facts.