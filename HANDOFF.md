# HANDOFF.md — blackwell

Continuity doc. Read `AGENTS.md` AND this file before acting.

---

## 1. Current Objective

Operational INT8 inference server (decode-only). Prefill integration abandoned. Next: prefill refactor per `docs/PREFILL_REFACTOR_PLAN.md`.

---

## 2. Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Server (decode-only) | ✅ Working | Produces correct output |
| HTTP endpoints | ✅ Working | /health, /v1/completions, /v1/chat/completions |
| Library | 195 symbols | `build/libblackwell_kernels.a` |
| Prefill | ❌ Abandoned | Cache layout incompatible, M=1 produces wrong hidden states |

### Throughput

| Method | t/s | Notes |
|--------|-----|-------|
| Server (M=1, correct model) | ~106 | head_norm + RoPE included |
| Benchmark M=1 (no head_norm/RoPE) | 181 | ⚠️ Omits head_norm+RoPE |
| Benchmark M=8 CUDA Graph | 575 | ⚠️ Omits head_norm+RoPE |
| 9B GatedDeltaNet M=8 | 52.1 | Batched GEMV + RMSNorm |

---

## 3. Recent Decisions

- **Server decode-only**: Reverted to session 50 state (commit 0069b35). Correct output verified.
- **Prefill abandoned**: Server prefill integration failed due to cache layout incompatibility. See `docs/PREFILL_REFACTOR_PLAN.md` for refactor plan.
- **INT4/INT5 dead**: Quality garbage after 28 layers. No viable path below INT8.
- **Docker image**: `ghcr.io/ronnieops/blackwell-server:v0.5.1` (decode-only)

---

## 4. Important Constraints

- `compute_120a` required (not `compute_120`)
- `killall hashcat` before every measurement (auto-restarts in 60s, -45% throughput)
- `gemv_int8_warp` is production GEMV — NOT `gemv_int8`
- `pack_int8` takes pre-computed scales as INPUT. Use `quantize_int8` to compute.
- M>8 not viable (register pressure)
- Weight matrices exceed L2 cache (32 MB)

---

## 5. Known Issues / Risks

| Issue | Status | Notes |
|-------|--------|-------|
| hashcat interference | ⚠️ | Always `killall hashcat` before measurement |
| Server vs benchmark gap | ~40% | head_norm + RoPE adds ~70% overhead |
| Prefill integration | ❌ Blocked | Cache layout `[NL][ms][nkv][hd]` incompatible with batched attention |

---

## 6. Pending Tasks

| Task | Priority | Status |
|------|----------|--------|
| Prefill refactor | HIGH | See `docs/PREFILL_REFACTOR_PLAN.md` |
| 8B HTTP server | MEDIUM | Port server to 8B weights |
| CUDA Graph (server) | LOW | Deferred — no speedup with correct model |

---

## 7. Suggested Next Actions

1. **Read `docs/PREFILL_REFACTOR_PLAN.md`** — full plan for prefill integration
2. **Follow the plan** — add `d_x` buffer, rewrite `batched_prefill`, verify M=1 match
3. **If blocked**: Keep server decode-only. Pre-fill requests fall back to per-token decode.

---

## 8. Important Files / Commands

### Build
```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

nvcc -O3 -std=c++17 -arch=sm_120a server/inference_server_nofp4.cu \
  build/libblackwell_kernels.a -I include -o server/inference_server \
  -lcudart -lpthread -lz

g++ -O2 server/http_subprocess.cpp /tmp/httplib.o -I include \
  -o server/http_subprocess -lpthread -lz -lssl -lcrypto
```

### Run
```bash
killall hashcat 2>/dev/null

# Server
./server/http_subprocess &
curl -X POST http://localhost:8123/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","max_tokens":5,"temperature":0}'

# Benchmarks
./bench/decode_int8_cgraph 28          # M=1: 181 t/s (no head_norm/RoPE)
./bench/text_generate "hi" 5           # Correctness
```

### Key files
```
server/inference_server_nofp4.cu    # Decode-only server
docs/PREFILL_REFACTOR_PLAN.md        # Prefill integration plan
bench/prefill_decode_benchmark.cu   # Standalone prefill benchmark
```

---

## 9. Validation

| Check | Value |
|-------|-------|
| Server output (greedy) | " Paris, a which is" ✅ |
| HTTP /v1/completions | ✅ |
| Library symbols | 195 ✅ |
| Server binary | 3159104 bytes ✅ |
| http_subprocess | 1195800 bytes ✅ |

---

## 10. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-06-06 |
| branch | master |
| repo_state | Clean (1 uncommitted: HANDOFF.md) |
| active components | server (decode-only), bench/*, lib |
| last_session | 52 |
| docker_image | `ghcr.io/ronnieops/blackwell-server:v0.5.1` |

---

## META PROMPT

**Boot sequence**:
1. Read `AGENTS.md` → `HANDOFF.md`
2. `git status` — check uncommitted changes
3. `killall hashcat 2>/dev/null`
4. `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l` (expect 195)
5. `./server/http_subprocess &` → test `curl -s -X POST http://localhost:8123/v1/completions -H "Content-Type: application/json" -d '{"prompt":"hi","max_tokens":3,"temperature":0}'`

**Current priorities**:
1. Prefill refactor (see `docs/PREFILL_REFACTOR_PLAN.md`)
2. Verify server correctness before any changes
3. Test incrementally — don't change multiple things between tests

**Verified facts**:
- Server produces " Paris, a which is" (greedy, temp=0) ✅
- Library 195 symbols, all kernels present ✅
- Prefill integration abandoned — cache layout incompatible ✅
- Decode-only path is correct and working ✅

**DO NOT**:
- Use benchmark numbers without noting head_norm/RoPE context
- Run measurements without `killall hashcat`
- Use INT4/INT5 (quality dead)
- Trust pre-session-37 INT4 benchmark numbers (grid bug)
- Expect M>8 scaling (register pressure)
- Re-dig: FP4 GEMM, speculative decode, PDL, sub-8-bit quality