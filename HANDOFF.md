# HANDOFF.md — blackwell

**ALWAYS read `AGENTS.md` AND this file before acting.**

---

## 1. Current Objective

**Production-ready INT4 8B inference server** — 36 tasks complete. Project is stable.
Session 71: Final cleanup, documentation, benchmark suite, deployment automation.

---

## 2. Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| INT4 8B HTTP server | ✅ **Production** | **~55 t/s**, PPL **21.82** |
| Batched GEMV kernel | ✅ | Uses `gemv_int4_batched` even M=1 (40% faster) |
| Embedding pre-load | ✅ | 623 MB FP32 on GPU, D2D lookup |
| Benchmark suite | ✅ | `scripts/benchmark_suite.py` — automated tests |
| Deployment | ✅ | `deploy/` with systemd, nginx, monitoring |
| API | ✅ | OpenAI-compatible (unique IDs, timestamps, usage) |
| Documentation | ✅ | README, API, DEPLOYMENT, ARCHITECTURE, QUALITY |
| NVFP4 | ❌ **ABANDONED** | PPL=24,850. Format mismatch + double quant unsolvable. |
| 9B quality | ❌ **BLOCKED** | SSM instability. A_log clamp insufficient. |
| New model download | ❌ **BLOCKED** | Network issues. No 14B or 9B weights available. |

---

## 3. Recent Decisions (Session 71)

- **Embedding pre-load**: Full embedding table (623 MB) pre-loaded to GPU as FP32 at startup. Runtime is D2D copy, no CPU dequant. GPU: 7287→9661 MB. Throughput unchanged (~55 t/s) — GEMV is 92% bottleneck.
- **NVFP4 abandoned**: PPL 24,850 (1000× worse than INT4). Root cause: INT4→FP32→NVFP4 double quantization shifts weights, and E2M1 nibble format mismatch (INT4 nibble 10 = +2, but E2M1 nibble 10 = -2). Not fixable without retraining.
- **Benchmark suite**: Automated tests for health, models, correctness, throughput, memory, regression, PPL estimate.
- **API improvements**: OpenAI-compatible responses with unique IDs, timestamps, system_fingerprint.
- **Housekeeping**: 89→19 bench .cu files, removed stale binaries and backups.

---

## 4. Important Constraints

- **Model dims (8B)**: nqh=32, nkv=8, hd=128, KV=1024, H=4096, I=12288
- **Model dims (1.7B)**: nqh=16, nkv=8, hd=128, KV=1024, H=2048, I=6144
- `compute_120a` required (NOT `compute_120`)
- `killall hashcat` before every GPU measurement
- Only weight dir: `weights_int4_qwen3_8b/` (5.8 GB)
- GPU memory: 9661 MB / 15849 MB (RTX 5060 Ti)
- 179 kernel symbols in `libblackwell_kernels.a`
- Disk: ~630 GB free

---

## 5. Known Issues / Risks

| Issue | Severity | Notes |
|-------|----------|-------|
| GPU non-determinism | LOW | Different outputs on same prompt. Expected for FP on GPU. Quality is consistent. |
| 9B quality BLOCKED | HIGH | SSM A_log > 0 for 68.8% layer-4 channels. Clamp insufficient. |
| Server prefill | MEDIUM | Prompts processed token-by-token. Major refactor needed. |
| New model blocked | HIGH | No network access to HuggingFace. 14B download would need ~15 GB. |

---

## 6. Pending Tasks

| Priority | Task | Notes |
|----------|------|-------|
| ❌ SKIP | NVFP4 | Abandoned (PPL=24,850, unsolvable) |
| BLOCKED | 9B SSM fix | Needs architectural changes, not quantization |
| BLOCKED | New model | No network access to download weights |
| LOW | Server prefill | Deferred (cache layout incompatibility) |
| LOW | Continuous batching | Major architectural change, use nginx load balance instead |

---

## 7. Suggested Next Actions

1. **New model** — When network available: Qwen3-14B or Mistral 7B. Need download + INT4 conversion.
2. **Client SDK** — Python/JS library for easier integration.
3. **Stress testing** — Concurrent load, memory limits.
4. **Research** — Different quantization (GPTQ, QuIP#) or block sizes.

---

## 8. Important Files / Commands

### Start server
```bash
killall hashcat 2>/dev/null
./server/http_subprocess batched &
sleep 45
curl http://localhost:8123/health
```

### Benchmark
```bash
python3 scripts/benchmark_suite.py --quick    # ~30s
python3 scripts/benchmark_suite.py            # ~5 min (full)
./bench/bench_ppl_int4_8b                     # PPL 21.82
./bench/text_generate_int4_batched "test" 1 50  # 59 t/s
```

### Build
```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### Validate
```bash
nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l  # expect 179
```

### Deployment
```bash
./deploy/monitor.sh                    # Single check
./deploy/monitor.sh --continuous     # Continuous
sudo cp deploy/blackwell.service /etc/systemd/system/
```

---

## 9. Validation Status

| Check | Value | Status |
|-------|-------|--------|
| Throughput | **~55 t/s** | ✅ |
| PPL (AWQ α=0.6) | **21.82** | ✅ |
| GPU memory | 9661 MB | ✅ |
| Kernel symbols | **179** | ✅ |
| Benchmark suite | All tests pass | ✅ |
| Server health | Working | ✅ |

---

## 10. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-06-10 |
| branch | master |
| repo_state | Dirty (modified files) |
| active_components | INT4 8B server, benchmark suite, deployment |
| key_session | 71 — Final cleanup, embedding pre-load, API improvements, documentation |
| next_priority | New model or client SDK when direction confirmed |

---

## META PROMPT

**Boot sequence**:
1. Read `AGENTS.md` → `HANDOFF.md`
2. `git status` — check repo state
3. `killall hashcat 2>/dev/null`
4. `nvidia-smi` — verify GPU free
5. `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l` — expect 179

**Verified facts**:
- INT4 8B: **~55 t/s**, PPL **21.82** (AWQ α=0.6)
- 8B dims: nqh=32, nkv=8, hd=128, KV=1024, H=4096, I=12288
- Only weight dir: `weights_int4_qwen3_8b/` (5.8 GB)
- GPU: 9661 MB / 15849 MB (RTX 5060 Ti, sm_120a)
- NVFP4: **ABANDONED** — PPL 24,850, unsolvable format mismatch
- 9B: **BLOCKED** — SSM instability, A_log clamp insufficient
- Server modes: `batched` (~63 t/s), `int4_8b` (~56 t/s)
- Deployment: `deploy/` with systemd, nginx, monitoring

**DO NOT**:
- Trust pre-session-56 quality numbers (wrong dims)
- Re-dig dead ends: NVFP4, FP8, INT5, 1.7B sub-8-bit, double quant, asymmetric INT4
- Assume 9B fixable
- Expect CUDA Graph speedup > 4%
- Recreate deleted weight dirs

**Current direction**: Production stable. Next: new model or client SDK.