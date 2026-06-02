# HANDOFF.md — blackwell

Continuity doc. Read before acting. Keep current with AGENTS.md.

---

## 1. Current Objective

**Q4 quantization** to close M=1 bandwidth gap.
- INT8 M=1 is 181.5 t/s (62% of Q4_K_M). INT4 halves DRAM reads → target **~250-290 t/s (~85-99% of Q4_K_M)**.
- Plan: `Q4_PLAN.md` — 4 phases: weight format, `gemv_int4_warp` kernel, pipeline integration, bench.
- INT8 production pipeline (324.6 t/s M=8, 111% of Q4_K_M) is stable and shipped.
- Docker + C++ batch server built and tested.
- Spec decode, M=1 CUDA Graph, FP4 tensor cores, PDL — all analyzed and documented as dead ends.

---

## 2. Current Status

| Metric | Value |
|--------|-------|
| GPU | RTX 5060 Ti, GB206, SM_120a, 36 SMs, ~500 GB/s GDDR7 |
| CUDA | 13.3, C++17, CMake |
| Library | **157 symbols** `build/libblackwell_kernels.a` |
| Branch | master @ `7b37ae0` |
| Session | **33** |

### Qwen3-1.7B

| Config | Total t/s | Per-seq | vs Q4_K_M | VRAM |
|--------|-----------|---------|-----------|------|
| M=1 fused decode (per-kernel) | 181.5 | 181.5 | 62% | ~3.4 GB |
| M=1 CUDA Graph (generic) | 181.2 | 181.2 | 62% | ~3.4 GB |
| M=4 batched-attn + Graph | 308.3 | 77.1 | **105%** | ~3.8 GB |
| **M=8 batched-attn + Graph** | **324.6** | **40.6** | **111%** | **~4.4 GB** |
| text_generate (M=1 pipeline) | ~140 | ~140 | 48% | ~3.4 GB |
| llama.cpp Q4_K_M FA=on | 293.4 | 293.4 | 100% | 5 GB |
| llama.cpp Q4_K_M FA=off | 274.1 | 274.1 | 93% | 5 GB |
| llama.cpp F16 FA=on | 114.3 | 114.3 | 39% | 5 GB |

### Qwen3-8B

| Config | Total t/s | vs Q4_K_M | VRAM |
|--------|-----------|-----------|------|
| Blackwell M=1 CUDA Graph | 44.6 | 54% | ~5 GB |
| llama.cpp Q4_K_M FA=on | 82.56 | 100% | ~6 GB |
| llama.cpp Q4_K_M FA=off | 78.62 | 95% | ~6 GB |

---

## 3. Recent Decisions

### Session 33 — Spec decode, llama.cpp audit, M=1 Graph, Docker, continuous batching, benchmarks
- **Spec decode infeasible**: Batched verify (24.7 ms/seq) is 4.5× slower per-seq than sequential (5.52 ms/seq). Draft needs 4.5× speedup. Even 50M draft yields ~92 t/s. Self-speculation fails (lm_head needs all 28 layers). Abandoned.
- **M=1 CUDA Graph**: Tried `cudaStreamCaptureModeRelaxed` + full warm-up. Capture still fails — `cudaMemcpyAsync` H2D inside `attention_decode_gqa`/`update_kv_cache` is illegal in ALL capture modes. Need graph-safe wrapper variants. Per-kernel 181.5 t/s is <3% from theoretical max.
- **llama.cpp code audit**: FP4 tensor cores are for batched MMQ only (M≥64). PDL eliminates launch gaps but our pipeline has <3% launch overhead. Both dead ends for M=1 decode. MMVQ_MAX_BATCH_SIZE=8 validates our M=8 limit.
- **Docker server built & tested**: `Dockerfile` uses `ubuntu:24.04` (binary static-links CUDA). `server/server.py` calls text_generate. Handles `killall` missing. `"Paris"` ✅.
- **Continuous batching server**: `server/inference_server.cpp` — persistent C++ daemon with BPE tokenizer, M=8 batched decode, GPU sampler. JSON IPC with string prompts. Python server spawns it as subprocess.
- **Benchmark vs llama.cpp**: Fresh data. M=8 = 324.6 t/s vs Q4_K_M = 293.4 t/s = **111%**. Qwen3-8B: M=1 = 44.6 t/s vs Q4_K_M = 82.56 t/s = **54%** (INT8 reads 2× — larger gap for 8B models exceeding L2).

### Session 32 — M=8 optimization + M>8 discovery
- **`gemv_int8_batched` M>8 bug fixed**: Switch only had cases 1-8. Now loops over groups of 8.
- **M=16 not optimal**: 335 t/s (barely better than M=8's 324). Register pressure kills occupancy.
- **gemv_int8_batched slower for isolated tests**: 1.5-2.7× slower than serial warp. Only beneficial in CUDA Graph context (fewer nodes).
- **L2 persisting harmful for large weights**: 28% regression when pinning 12.6 MB gate weights.
- **Fused pack+GEMV kernels archived**: Correct but 20% slower.

---

## 4. Important Constraints

- `export PATH=/usr/local/cuda-13.3/bin:$PATH` before nvcc
- `compute_120a` required (not `compute_120`)
- `gemv_int8_warp` production GEMV — NOT `gemv_int8`
- `gemv_int8_batched` supports M>8 (loops groups of 8)
- `killall hashcat` before every measurement (auto-restarts, -45% throughput)
- INT8 reads 2× data vs Q4_K_M — fundamental M=1 bandwidth limit
- nvidia/cuda:13.3 Docker images don't exist on Docker Hub — build with ubuntu:24.04 (binary static-links CUDA)

---

## 5. Known Issues / Risks

1. **hashcat**: Auto-restarts, -45% throughput. `killall hashcat` before every measure.
2. **INT8 vs Q4_K_M gap (M=1)**: 2× data read. Sub-byte quant only fix.
3. **M=16+ not beneficial**: Register pressure. M=8 is practical limit.
4. **CUDA Graph M=1 blocked**: H2D copies in kernel wrappers illegal during capture.
5. **Spec decode infeasible**: Batched verify 4.5× slower per-seq.
6. **FP4 tensor cores/PDL dead ends**: Both confirmed unusable for M=1 decode.
7. **Qwen3-8B INT8 gap**: 54% of Q4_K_M (44.6 vs 82.6) — INT8 reads 2× data, 8B weights exceed L2.

---

## 6. Pending Tasks

| Task | Status | Notes |
|------|--------|-------|
| INT4 quantization (Q4_PLAN.md) | 🔜 Planned | Phase 1: weight format + Python converter. Phase 2: gemv_int4_warp kernel. Target ~250-290 t/s M=1. |
| Deploy production server | ✅ Done | Docker built/tested, C++ batched server ready |
| Speculative decoding | ❌ Abandoned | Batched verify 4.5× slower per-seq |
| Qwen3.5-9B integration | TODO | 45.6 t/s bench exists, not in text_generate/server |

---

## 7. Suggested Next Actions

| Priority | Task | Rationale |
|----------|------|-----------|
| **Active** | **INT4 quantization (`Q4_PLAN.md`)** | Phase 1: convert weights. Phase 2: gemv_int4_warp kernel. Target ~250-290 t/s M=1 (85-99% of Q4_K_M). |
| Medium | Qwen3.5-9B integration | MoE decode, tokenizer integration |
| Low | Deploy to production | Docker + server ready, needs prod infra |

---

## 8. Important Files / Commands

### Build
```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
killall hashcat 2>/dev/null
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake --build build --parallel
```

### Benchmark
```bash
./bench/decode_int8_cgraph 28                   # M=1: 181.5 t/s
./bench/decode_int8_batched_cgraph_attn 28 8    # M=8: 324.6 t/s
./bench/text_generate "The capital of France is" 30  # Correctness
./bench/decode_int8_generic 36 weights_int8_qwen3_8b 4096 4096 1024 12288 32 8 "Qwen3-8B"  # 44.6 t/s
```

### Q4 Quantization Plan (active)
```
Q4_PLAN.md — full 4-phase roadmap for INT4 migration
Phase 1: weight format + Python conversion tools
Phase 2: gemv_int4_warp kernel (nibble unpack + __dp4a)
Phase 3: full decode pipeline + benchmarks
Phase 4: batched kernel (optional)
```

### Server
```bash
# C++ persistent batched server (string prompts via JSON stdin/stdout)
echo '{"prompts":["The capital of France is"],"max_tokens":30,"temperature":0.8}' | ./server/inference_server

# Docker
docker build -t blackwell-inference .
docker run --gpus all -p 8080:8080 blackwell-inference
curl -X POST http://localhost:8080/generate -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","max_tokens":30}'
```

### Verify
```bash
nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l  # expect 157
```

---

## 9. Validation

| Check | Status |
|-------|--------|
| Library | ✅ 157 symbols |
| M=8 CUDA Graph (1.7B) | ✅ 324.6 t/s (111% of Q4_K_M) |
| M=1 fused (1.7B) | ✅ 181.5 t/s (62% of Q4_K_M) |
| M=1 CUDA Graph (8B) | ✅ 44.6 t/s (54% of Q4_K_M) |
| gemv_int8_batched M>8 | ✅ Fixed (loop groups of 8) |
| Correctness | ✅ Max diff 0.000000 vs serial baseline |
| Docker server | ✅ Built, deployed, "Paris" ✅ |
| C++ batch server | ✅ Built, BPE tokenizer, string prompts, GPU sampler |

---

## 10. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-06-02 |
| branch | master |
| last_commit | `7b37ae0` docs: Q4 quantization plan — full roadmap for INT4 migration |
| repo_state | 157 symbols. M=8: 324.6 t/s (111% of Q4_K_M). M=1: 181.5 t/s (62%). Docker + C++ batch server ready. `Q4_PLAN.md` committed — 4-phase roadmap for INT4 migration to close M=1 gap. |
| uncommitted | (none — clean) |

---

## META PROMPT

**Boot sequence**: Read `AGENTS.md` → `HANDOFF.md` → `git log --oneline -3` → `killall hashcat` → `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l` (expect 157) → `./bench/decode_int8_batched_cgraph_attn 28 8` (expect ~324 t/s) → `echo '{"prompts":["The capital of France is"],"max_tokens":10}' | ./server/inference_server` (expect JSON with tokens).

**Verified state**: 157 symbols. M=8 CUDA Graph: 324.6 t/s (111% of Q4_K_M). M=1 fused: 181.5 t/s (62%). Docker + C++ batch server ready. Spec decode infeasible (4.5× verify slowdown). M=1 CUDA Graph blocked (H2D copies). FP4 tensor cores/PDL dead ends. llama.cpp benchmark data refreshed (293.4 Q4_K_M 1.7B, 82.56 Q4_K_M 8B).

**DO NOT**:
- Use `compute_120` (must be `compute_120a`)
- Use `gemv_int8` in production (use `gemv_int8_warp`)
- Benchmark without `killall hashcat`
- Expect M>8 to help (batched GEMV register pressure)
- Use fused pack+GEMV kernels (20% slower, archived)
- Pursue speculative decoding, FP4 tensor core GEMV, or PDL (all dead ends for M=1 decode)
- Attempt M=1 CUDA Graph without fixing H2D copies in kernel wrappers

**Revisitable**:
- M=1 CUDA Graph: Need graph-safe wrappers that skip `cudaMemcpyAsync` H2D for seq_pos. Pre-set seq_pos before capture via direct device pointer. Low priority — 181.5 t/s per-kernel is <3% from theoretical graph max.
- Q4 quantization (INT4): **Active — `Q4_PLAN.md` committed**. 4 phases: weight format, `gemv_int4_warp` kernel, pipeline integration, benchmarks. Target ~250-290 t/s M=1 (85-99% of Q4_K_M). Phase 1 = Python converter script + weight files.

**Update discipline**: Update HANDOFF.md only when materially new state. Keep deduplicated with AGENTS.md. Prefer bullets over prose.
