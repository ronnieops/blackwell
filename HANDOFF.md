# HANDOFF.md — blackwell

Read `AGENTS.md` AND this file before acting.

---

## 1. Current Objective

INT4 8B production path stabilized. INT4 8B: 56 t/s, PPL 23.52 (1.9× BF16).
14× faster than INT8 server (3.9 t/s). Weight size 5.3 GB vs 9.6 GB INT8.

**Session 63 fixes**:
1. http_subprocess defaulted to temperature=0.7f → garbled output. Fixed to
   temperature=0.0f (greedy) to match benchmark behavior.
2. Added repetition_penalty support: all 4 request handlers now send
   rep_pen=1.5 by default. Server apply_repetition_penalty kernel uses it.
   Clients can override via JSON body. Eliminates token looping.
3. Docker image built: blackwell-server:int4 (148 MB, tested ✅)

---

## 2. Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| INT4 8B HTTP server | ✅ Production | `http_subprocess int4_8b`, 56 t/s |
| INT4 batched server | ✅ Works | M prompts processed sequentially with own KV cache |
| INT4 benchmark | ✅ 57 t/s | `bench/text_generate_int4_8b` |
| INT4 PPL benchmark | ✅ PPL 23.52 | `bench/bench_ppl_int4_8b` |
| Repetition penalty | ✅ Works | Reduces token looping |
| CUDA Graph | ⚠️ 1% gain | `cudaMemcpyAsync` in attention blocks capture |
| Docker | ✅ Built | `blackwell-server:int4` (148 MB) |
| Build | ✅ 177 kernels | `libblackwell_kernels.a` |
| INT8 server | ✅ Works | 8B: 3.9 t/s, 1.7B: 23 t/s |

### PPL Quality (8B)

| Config | PPL | vs BF16 |
|--------|-----|---------|
| BF16 (llama.cpp Q8_0) | **12.4** | 1.0× |
| INT8 block-16 | **18.65** | 1.5× |
| INT4 symmetric | **23.52** | 1.9× |

---

## 3. Recent Decisions

- **INT4 8B is production viable**: 56 t/s (14× faster than INT8), coherent output, PPL 23.52
- **INT4 batched**: M prompts processed sequentially with own KV cache. M=1 works perfectly. M>1 crashes in original design — fixed by sequential processing.
- **CUDA Graph limited**: 1% speedup. Root cause: `update_kv_cache` and `attention_decode_batched_gqa` use `cudaMemcpyAsync` internally — not CUDA Graph compatible. Full speedup requires custom kernels.
- **Repetition penalty added**: Works at 56 t/s, reduces token looping
- **Docker image built**: `blackwell-server:int4` (148 MB)
- **Dead code removed**: 8 stale benchmark files (FP8, INT5, FP4)
- **upload_w4 scale buffer bug FIXED**: Root cause of all INT4 crashes. `ss = h[3]*h[4]` from int4_t header (256) instead of scale_t header (38.9M for lm_head)

---

## 4. Important Constraints

- **CORRECT 8B DIMS: nqh=32, nkv=8, hd=128, KV=1024**
- **CORRECT 1.7B DIMS: nqh=16, nkv=8, hd=128, KV=1024**
- `compute_120a` required (NOT `compute_120`)
- `killall hashcat` before every measurement
- INT4 is symmetric (no zero point): `nib - 8` for signed values
- INT4 activation quantization: `sc = max(1e-10, absmax)/7`, `q = clamp(round(v/sc), -7, 7)`

---

## 5. Known Issues / Risks

| Issue | Severity | Notes |
|-------|----------|-------|
| CUDA Graph limited | MEDIUM | Attention kernels use cudaMemcpyAsync — blocks full capture |
| INT4 PPL 23.52 vs BF16 12.4 | MEDIUM | Symmetric quantization, no calibration. AWQ could improve. |
| 9B quality BLOCKED | HIGH | SSM instability: A_log > 0 for 68.8% of layer-4 channels |
| Token looping | MEDIUM | Reduced with repetition_penalty=1.3 |

---

## 6. Pending Tasks

| Task | Priority | Notes |
|------|----------|-------|
| True batched GEMV (M>1) | MEDIUM | Would need separate Q/K/V buffers per sequence |
| INT4 calibration (AWQ) | HIGH | PPL 23→18? potential improvement |
| 9B quality fix | HIGH | SSM instability root cause unknown |

---

## 7. Suggested Next Actions

1. **True batched GEMV**: Separate Q/K/V buffers per sequence + batched kernel
2. **INT4 calibration**: AWQ-style per-channel scales for PPL improvement
3. **9B quality**: Investigate SSM A_log instability
4. **Docker push**: Push `blackwell-server:int4` to ghcr.io

---

## 8. Important Files / Commands

### Key binaries
```
server/inference_server_int4           — INT4 server (2.7 MB)
server/inference_server_int4_batched  — Batched INT4 server (2.7 MB)
server/http_subprocess                 — HTTP wrapper
bench/text_generate_int4_8b           — INT4 benchmark (57 t/s)
bench/bench_ppl_int4_8b               — PPL benchmark (23.52)
bench/decode_int4_cgraph_8b           — CUDA Graph benchmark
```

### Docker
```
docker build -f Dockerfile.int4 -t blackwell-server:int4 .
docker run --gpus all -p 8080:8080 \
  -v /path/to/weights_int4_qwen3_8b:/app/weights_int4_qwen3_8b \
  -v /path/to/weights_int8_qwen3_8b:/app/weights_int8_qwen3_8b \
  blackwell-server:int4 8080 int4_8b
```

### Build
```bash
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### Run
```bash
killall hashcat 2>/dev/null
curl -X POST http://localhost:8124/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","max_tokens":30,"repetition_penalty":1.3}'
```

---

## 9. Validation Status

| Check | Value |
|-------|-------|
| INT4 8B throughput | **56 t/s** ✅ |
| INT4 8B PPL | **23.52** ✅ (1.9× BF16) |
| INT4 batched server | ✅ Works (M sequential) |
| INT4 docker image | ✅ Built (148 MB) |
| Build kernel count | **177** ✅ |
| Repetition penalty | ✅ Works |
| CUDA Graph speedup | 1% ⚠️ |
| 8B dims verified | nqh=32, nkv=8, hd=128, KV=1024 ✅ |
| 1.7B dims verified | nqh=16, nkv=8, hd=128, KV=1024 ✅ |

---

## 10. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-06-08 |
| branch | master |
| repo_state | Modified (uncommitted INT4 files, AGENTS.md updated) |
| session | 63 (temp fix, rep_pen, Docker, PPL 23.52 confirmed) |
| key_finding | INT4 8B production viable at 56 t/s, PPL 23.52. upload_w4 scale bug fixed. CUDA Graph limited by cudaMemcpyAsync. |
| next_priority | True batched GEMV, INT4 calibration, or 9B quality |

**BUG FIX (Session 63)**: http_subprocess default temp=0.7f caused garbled output.
Root cause: all 4 request handlers defaulted to temp=0.7f, top_k=40 instead of
temp=0.0f, top_k=0. Server inference_server_int4 defaulted to temp=0.0f but
http_subprocess overrode it. Benchmark works (57 t/s, greedy) but HTTP server
was broken. Fixed in server/http_subprocess.cpp. Output now matches benchmark.

---

## META PROMPT

**Boot sequence**:
1. Read `AGENTS.md` → `HANDOFF.md`
2. `git status` — check state
3. `killall hashcat 2>/dev/null`
4. `nvidia-smi --query-compute-apps` — ensure no stale GPU processes

**Verified facts**:
- INT4 8B: **56 t/s**, PPL **23.52** (1.9× BF16)
- INT8 8B: **3.9 t/s**, PPL **18.65** (1.5× BF16)
- 8B dims: nqh=**32**, nkv=**8**, hd=**128**, KV=**1024**
- INT4 is symmetric: `nib - 8` for signed values, scale = absmax/7
- upload_w4 bug: read scale_t header for scale count, NOT int4_t header

**DO NOT**:
- Use old dimension values (nqh=32, nkv=4, hd=64, KV=512 for 1.7B)
- Trust pre-session-56 quality numbers (wrong dims)
- Expect CUDA Graph speedup > 1% (cudaMemcpyAsync blocks capture)
- Re-dig dead ends: FP8, INT5, FP4, asymmetric INT4

**Build verification**: `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l` → expect 177