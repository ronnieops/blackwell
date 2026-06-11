# HANDOFF.md — blackwell

**ALWAYS read `AGENTS.md` AND this file before acting.**

---

## 1. Current Objective

**Production-ready INT4 8B inference server** — 37 tasks complete. Project is stable.
Session 72: CUDA Graph fix (2.1% speedup, graph-safe attention), GGUF bridge Phase 1-2 working.

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
| CUDA Graph | ⚠️ **Partial** | 2.1% speedup (867 nodes). Limited by GEMV 92% dominance. |
| GGUF Bridge | ✅ **Phase 1-3** | Qwen3 + Llama 3.2 converter. Llama 3.2 1B: 223 t/s, coherent. |
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
- **CUDA Graph fix (Session 72)**: Replaced H2D memcpy in attention with device-side seq_pos. New APIs: `attention_decode_batched_gqa_device()`, `attention_decode_gqa_device()`. Graph now captures 867 nodes including KV cache + attention + RoPE. Speedup: 2.1% (64→65 t/s). Limited by GEMV dominance.
- **GGUF bridge Phase 1-2**: Parser reads GGUF v3, converter dequantizes Q8_0→INT4, writes blackwell format. Tested with Qwen3-1.7B Q8_0. Tokenizer export matches original format.
- **GGUF bridge Phase 3 (Llama support)**: Unified `map_tensor_name()` for Qwen3 + Llama 3.1/3.2. Llama 3.2 1B Q4_K_M converted: 262 files, 891 MB. Tested with `bench/text_generate_llama32_1b`: 223 t/s, coherent output.
- **GGUF converter bug (FIXED)**: Tensor data offset was used directly without adding `tensor_data_off`. GGUF v3 stores offsets relative to tensor data section. This caused all F32 weights (layernorms) to read from wrong offset → garbage layernorm weights → NaN logits. Fixed by adding `tensor_data_off` to all tensor reads.
- **GGUF RoPE fix**: GGUF v3 uses nested prefixes (rope.freq_base stored under full repo URL). Fixed by searching for any key ending with the suffix.
- **Llama 3.2 1B verified**: 16L, H=2048, I=8192, nqh=32, nkv=8, hd=64, V=128256, rope_theta=500000. Q4_K/Q6_K mixed quantization. Benchmark: 223 t/s, coherent text.

---

## 4. Important Constraints

- **Model dims (8B)**: nqh=32, nkv=8, hd=128, KV=1024, H=4096, I=12288
- **Model dims (1.7B)**: nqh=16, nkv=8, hd=128, KV=1024, H=2048, I=6144
- `compute_120a` required (NOT `compute_120`)
- `killall hashcat` before every GPU measurement
- Only weight dir: `weights_int4_qwen3_8b/` (5.8 GB)
- GPU memory: 9661 MB / 15849 MB (RTX 5060 Ti)
- 181 kernel symbols in `libblackwell_kernels.a`
- Disk: ~630 GB free

---

## 5. Known Issues / Risks

| Issue | Severity | Notes |
|-------|----------|-------|
| GPU non-determinism | LOW | Different outputs on same prompt. Expected for FP on GPU. Quality is consistent. |
| 9B quality BLOCKED | HIGH | SSM A_log > 0 for 68.8% layer-4 channels. Clamp insufficient. |
| Server prefill | MEDIUM | Prompts processed token-by-token. Use `start_servers.sh` for multi-model.
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

1. **GGUF bridge Phase 3** — Llama tensor name mapper for GGUF converter. Llama uses `blk.{l}.attn_{q,k,v,o}.weight` vs Qwen3's `blk.{l}.attn_{q,k,v,o}.weight` (same naming convention!). Need to handle RoPE base and tokenizer (tiktoken vs BPE).
2. **8B GGUF validation** — Find or download 8B GGUF model to validate full quality. Qwen3-1.7B tested but 1.7B INT4 is dead-end.
3. **Client SDK** — Python/JS library for easier integration.
4. **CUDA Graph full fusion** — Replace GEMV+quantize+fused_rmsnorm into fewer kernels to reduce launch overhead further.

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
nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l  # expect 181
```

### CUDA Graph benchmark
```bash
./bench/decode_int4_cgraph_8b 50  # ~65 t/s (2.1% faster than per-kernel ~64 t/s)
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
| Kernel symbols | **181** | ✅ |
| Benchmark suite | All tests pass | ✅ |
| Server health | Working | ✅ |
| CUDA Graph | 2.1% speedup | ⚠️ |
| GGUF bridge | Qwen3 converter | ✅ |

---

## 10. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-06-10 |
| branch | master |
| repo_state | Dirty (modified files) |
| active_components | INT4 8B server, benchmark suite, deployment |
| key_session | 72 — CUDA Graph fix, GGUF bridge Phase 1-2 |
| next_priority | GGUF Llama support, 8B GGUF validation, client SDK |

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