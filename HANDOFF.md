# HANDOFF.md — blackwell

Read `AGENTS.md` AND this file before acting.

---

## 1. Current Objective

INT4 8B production path stabilized. INT4 batched benchmark: M=1:63, M=2:115,
M=4:148, M=8:168 t/s. Server uses batched kernels (gemv_int4_batched).

**Session 65 fixes**:
1. Session 64 smem fix `((128+4096)*4)` broke INT4 output (garbage).
   Reverted `src/kernels/decode.cu` to session 63 version (smem=4096*4).
2. Batched benchmark OOM at M=3+. Fixed by reducing MAXSEQ 4096→512.
3. Server updated to use batched kernels (gemv_int4_batched M=1).
   Output now matches benchmark exactly.
4. INT8 8B server has CUDA Graph issues (garbage output). Not investigated.

**Session 64 finding** (reverted):
- smem_bytes = `(128+4096)*4` caused output divergence
- Root cause unknown — attention kernel smem layout doesn't match expected
- smem_bytes = `4096*4` (16 KB) works correctly

---

## 2. Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| INT4 8B HTTP server | ✅ Production | `inference_server_int4`, 56 t/s |
| INT4 batched benchmark | ✅ M=1-8 | M=1:63, M=2:115, M=4:148, M=8:168 t/s |
| INT4 batched M=9+ | ❌ Broken | Garbage output (unknown root cause) |
| INT4 PPL benchmark | ✅ PPL 23.52 | `bench/bench_ppl_int4_8b` |
| Repetition penalty | ✅ Works | Reduces token looping |
| CUDA Graph | ⚠️ Issues | Breaks INT8 8B server output |
| Docker | ✅ Built | `blackwell-server:int4` (148 MB) |
| Build | ✅ 179 kernels | `libblackwell_kernels.a` |
| INT8 8B server | ❌ Broken | CUDA Graph causes garbage output |

### Batched INT4 Throughput

| M | t/s | ms/tok | Notes |
|---|-----|--------|-------|
| 1 | 63 | 15.8 | Batched GEMV (40% faster than single-seq gemv_int4_warp) |
| 2 | 116 | 8.6 | |
| 4 | 148 | 6.8 | |
| 8 | 168 | 5.9 | |
| 9+ | — | — | Garbage output |

---

## 3. Recent Decisions

- **Batched GEMV kernels faster**: `gemv_int4_batched` 40% faster than `gemv_int4_warp` even at M=1
- **MAXSEQ=512 for batched**: M=3+ OOM with MAXSEQ=4096. Reduced to 512 allows M=8.
- **M=9+ broken**: Garbage output. Unknown root cause.
- **smem_bytes change breaks output**: Session 64 smem fix `((128+4096)*4)` caused divergence. Reverted to `4096*4`.
- **Server uses batched kernels**: All GEMV ops now use `gemv_int4_batched` (M=1). Output matches benchmark.
- **INT8 8B server broken**: CUDA Graph causes garbage output. Not investigated.

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
| M=9+ batched broken | MEDIUM | Garbage output. Unknown root cause. |
| INT8 8B server broken | MEDIUM | CUDA Graph causes garbage output. |
| CUDA Graph limited | MEDIUM | Attention kernels use cudaMemcpyAsync — blocks full capture |
| INT4 PPL 23.52 vs BF16 12.4 | MEDIUM | Symmetric quantization, no calibration. |
| 9B quality BLOCKED | HIGH | SSM instability: A_log > 0 for 68.8% of layer-4 channels |

---

## 6. Pending Tasks

| Task | Priority | Notes |
|------|----------|-------|
| Fix M=9+ garbage | MEDIUM | Unknown root cause (possibly memory alignment) |
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
bench/text_generate_int4_qwen3_8b     — Single-seq INT4 benchmark (44 t/s)
bench/text_generate_int4_batched       — Batched INT4 benchmark (M=1-8)
bench/bench_ppl_int4_8b               — PPL benchmark (23.52)
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
| Batched benchmark M=1-8 | ✅ Works |
| Batched server output | ✅ Matches benchmark |
| Build kernel count | **179** ✅ |
| Repetition penalty | ✅ Works |
| smem regression fixed | ✅ Reverted decode.cu |
| Server batched kernels | ✅ gemv_int4_batched (M=1) |
| Server=benchmark match | ✅ Output identical |
| INT8 8B server | ❌ Broken (CUDA Graph) |
| 8B dims verified | nqh=32, nkv=8, hd=128, KV=1024 ✅ |

---

## 10. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-06-09 |
| branch | master |
| repo_state | Clean (pushed to origin) |
| session | 65 (smem regression fixed, batched benchmark restored) |
| key_finding | Batched INT4: M=1:63, M=2:115, M=4:148, M=8:168 t/s. Server uses batched kernels. Session 64 smem fix `((128+4096)*4)` broke output — reverted. |
| next_priority | Fix M=9+ garbage, INT4 calibration, or investigate smem divergence |

**BUG FIX (Session 65)**: Session 64 smem fix `((128+4096)*4` caused INT4 output
divergence ("is is is is is is"). Reverted `src/kernels/decode.cu` to session 63
version (smem=4096*4). Output restored: "The capital of France is a city in
the state of".

**BUG FIX (Session 65)**: Server used `gemv_int4_warp` but benchmark used
`gemv_int4_batched`. Updated server to use batched kernels (M=1). Output now
matches benchmark exactly.

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

**Build verification**: `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l` → expect 179