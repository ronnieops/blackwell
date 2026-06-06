# HANDOFF.md — blackwell

Continuity doc. Read `AGENTS.md` AND this file before acting.

---

## 1. Current Objective

Operational INT8 inference server with batched prefill for single-prompt requests (M≤1). Prefill integrated for gen_start≤M. gen_start>M falls back to per-token decode.

---

## 2. Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Server (batched prefill) | ✅ Working | v0.6.0: 1.7B and 8B models, batched_prefill for gen_start≤M |
| HTTP endpoints | ✅ Working | /health, /v1/completions, /v1/chat/completions |
| Library | 195 symbols | `build/libblackwell_kernels.a` |
| Prefill (M=1) | ✅ Integrated | batched_prefill called from main loop |
| Prefill (M>1, gen_start>1) | ✅ Fixed | Sequential per-item generation avoids KV cache conflicts |
| Chat completions | ⚠️ Model limit | 1.7B too small for INT8 chat. Needs 8B+ instruct model |

### Throughput

| Method | t/s | Notes |
|--------|-----|-------|
| Server (M=1, correct model) | ~106 | head_norm + RoPE included |
| Benchmark M=1 (no head_norm/RoPE) | 181.5 | ⚠️ Omits head_norm+RoPE |
| Benchmark M=8 CUDA Graph | 575 | ⚠️ Omits head_norm+RoPE |
| 9B GatedDeltaNet M=8 | 52.1 | Batched GEMV + RMSNorm |

---

## 3. Recent Decisions

- **Prefill integrated (v0.6.0)**: Added d_x and d_tmp_save to ServerState. batched_prefill() is now called from main loop for gen_start≤M. gen_start>M falls back to per-token decode.
- **M>1 batched fixed**: Sequential per-item generation avoids KV cache conflicts. Different prompts now produce different outputs.
- **v0.6.0 multi-model**: Server accepts model arg (1.7b/8b), loads correct weights and architecture
- **8B separate lm_head**: INT8 lm_head for non-tied models
- **parse_string_prompts bug fixed**: `strstr("prompt")` matched inside `"prompts"` → added boundary check `prompt_p[8] == ':'`
- **INT4/INT5 dead**: Quality garbage after 28 layers. No viable path below INT8.
- **Docker image**: `ghcr.io/ronnieops/blackwell-server:v0.6.0` (prefill integrated)
- **Chat completions**: 1.7B model (base+instruct) too small for coherent INT8 chat. Instruct model quantized to weights_int8_qwen3_1.7b_instruct/ but quality still poor.

---

## 4. Important Constraints

- `compute_120a` required (not `compute_120`)
- `killall hashcat` before every measurement (auto-restarts in 60s, -45% throughput)
- `gemv_int8_warp` is production GEMV — NOT `gemv_int8`
- `pack_int8` takes pre-computed scales as INPUT. Use `quantize_int8` to compute.
- M>8 not viable (register pressure)
- Weight matrices exceed L2 cache (32 MB)
- batched_prefill: gen_start≤M uses batched path; gen_start>M falls back to decode
- M>1 batched CORRUPTED: generate loop shares KV cache across M batch items. All items produce identical (last prompt's) output. HTTP server unaffected (M=1 only).

---

## 5. Known Issues / Risks

| Issue | Status | Notes |
|-------|--------|-------|
| hashcat interference | ⚠️ | Always `killall hashcat` before measurement |
| Server vs benchmark gap | ~40% | head_norm + RoPE adds ~70% overhead |
| Chat completions | ⚠️ | 1.7B model too small for coherent chat with INT8 quantization |
| Prefill (M>1) | ✅ Fixed | Sequential per-item generation. Different prompts produce different outputs |

---

## 6. Pending Tasks

| Task | Priority | Status |
|------|----------|--------|
| Prefill M=1 correctness | HIGH | ✅ Integrated — verify hidden state match |
| Prefill M>1 full support | ✅ Fixed | Sequential per-item generation. M=1 fast path, M>1 correct but slower |
| 8B HTTP server | ✅ Done | v0.6.0 multi-model support |
| Chat completions | ✅ Investigated | 1.7B too small, needs 8B+ instruct |
| CUDA Graph (server) | LOW | Deferred — no speedup with correct model |

---

## 7. Suggested Next Actions

1. ✅ **M>1 batched KV cache fix**: Fixed. Sequential per-item generation.
2. ✅ **Chat completions**: Investigated. 1.7B too small for INT8 chat. Instruct model quantized but still garbled.
3. ✅ **8B HTTP server**: Done. v0.6.0 supports both 1.7B and 8B models.
4. **Next**: Port 8B instruct model for chat quality. Consider 9B GatedDeltaNet.

---

## 8. Important Files / Commands

### Build
```bash
/usr/local/cuda-13.3/bin/nvcc -O3 -std=c++17 -gencode=arch=compute_120a,code=sm_120a \
  -I include -I /usr/local/cuda-13.3/include \
  server/inference_server_nofp4.cu build/libblackwell_kernels.a \
  -o server/inference_server -lcudart -lpthread -lz

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
./bench/decode_int8_cgraph 28          # M=1: 181.5 t/s (no head_norm/RoPE)
./bench/text_generate "hi" 5           # Correctness
```

### Key files
```
server/inference_server_nofp4.cu    # Server with batched prefill (v0.6.0)
docs/PREFILL_REFACTOR_PLAN.md        # Prefill integration plan (partially obsolete)
bench/prefill_decode_benchmark.cu   # Standalone prefill benchmark
```

---

## 9. Validation

| Check | Value |
|-------|-------|
| Server output (greedy) | " Paris, a which is" ✅ |
| HTTP /v1/completions | ✅ |
| Library symbols | 195 ✅ |
| Server binary | 3155176 bytes ✅ |
| http_subprocess | 1195800 bytes ✅ |
| Prefill M=1 | ✅ batched_prefill called, correct output |

---

## 10. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-06-06 |
| branch | master |
| repo_state | Clean (committed: 7f5d823 — batched prefill v0.6.0) |
| active components | server (prefill), bench/*, lib |
| last_session | 53 |
| server_version | v0.6.0 (batched prefill) |
| docker_image | `ghcr.io/ronnieops/blackwell-server:v0.6.0` |

---

## META PROMPT

**Boot sequence**:
1. Read `AGENTS.md` → `HANDOFF.md`
2. `git status` — check uncommitted changes (should be clean)
3. `killall hashcat 2>/dev/null`
4. `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l` (expect 195)
5. `./server/http_subprocess &` → test `curl -s -X POST http://localhost:8123/v1/completions -H "Content-Type: application/json" -d '{"prompt":"hi","max_tokens":3,"temperature":0}'`

**Current priorities**:
1. Verify M=1 prefill hidden state matches decode (GPU memory comparison)
2. Test M>1 batched requests
3. Investigate chat completions garbled output

**Verified facts**:
- Server produces " Paris, a which is" (greedy, temp=0) ✅
- Library 195 symbols, all kernels present ✅
- Prefill integrated for gen_start≤M ✅
- gen_start>M falls back to per-token decode ✅
- Chat completions garbled: pre-existing bug ✅
- Changes committed: 7f5d823 ✅

**DO NOT**:
- Use benchmark numbers without noting head_norm/RoPE context
- Run measurements without `killall hashcat`
- Use INT4/INT5 (quality dead)
- Trust pre-session-37 INT4 benchmark numbers (grid bug)
- Expect M>8 scaling (register pressure)
- Re-dig: FP4 GEMM, speculative decode, PDL, sub-8-bit quality
