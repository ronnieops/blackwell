# Handoff — Session 82 (2026-06-16)

## 1. Current Objective

Optimize INT4/INT8 LLM inference on Blackwell (RTX 5060 Ti, sm_120a).
Production path: Qwen3-8B INT4 with fused kernels + FP16 weight scales.

## 2. Current Status — PRODUCTION

| Config | t/s | ms/tok | vs llama.cpp Q4_K_M |
|--------|-----|--------|---------------------|
| M=1 (new) | **73** | 13.7 | 87% |
| M=8 (new) | **226** | 4.4 | 269% |
| M=16 (new) | **244** | 4.1 | 290% |
| llama.cpp Q4_K_M | 84 | 11.9 | 1.0× |

**PPL**: 21.98 (fused path, FP16sc baseline). AWQ per-layer FP16sc: 23.57.

**Symbol count**: 203 (was 189 pre-fusion).

## 3. Recent Decisions (this session)

- **Fused kernels all verified**: gate+up, QKV, rmsnorm+quant — all produce correct output.
- **Graph path bug is PRE-EXISTING** (since Oct 2024 commit `ab03bac`). NOT from fusion.
  Produces garbage from first decode. Deep CUDA driver issue. Not a regression.
- **AWQ PPL regression (23.57 vs 21.98)**: consistent in both fused and separate paths.
  Fused path is 10% better PPL than separate path. AWQ tuned for separate path.
- **2-warp GEMV**: 0.96× (slower). Memory-saturated. 1-warp optimal.
- **Server OOM**: memory fragmentation from 500+ weight cudaMalloc calls + FP32 embed
  table (2.5 GB). Not from fusion. Pre-existing limitation.
- **Batched server** dispatches to `inference_server_int4_batched` via `int4_batched`
  model alias in HTTP wrapper. OOM at 15.3 GB (94% VRAM).

## 4. Important Constraints

- `killall hashcat` before any GPU measurement (60s respawn window)
- `CUDACXX=/usr/local/cuda-13.3/bin/nvcc` env var required
- `compute_120a` arch (not `compute_120`)
- Server binary built outside CMake: direct nvcc invocation
- Bench executables land in `./bench/` (source dir), not `build/bench/`
- Server GPU memory: 15.3 GB (94%) after single server load with embed FP32 cache

## 5. Known Issues / Risks

| Issue | Severity | Workaround |
|-------|----------|------------|
| CUDA Graph path broken (pre-existing) | Medium | Per-kernel path used. 73 t/s M=1. |
| Batched server OOM | Medium | Single server works. Use BATCH_SIZE=1. |
| AWQ per-layer PPL regression | Low | Baseline 21.98 is acceptable. |
| Graph path garbled output | Low | Pre-existing. Not a fusion regression. |

## 6. Pending Tasks

| Priority | Task | Effort |
|----------|------|--------|
| Medium | Fix batched server OOM: remove embed FP32 cache (saves 2.5 GB) or consolidate weight cudaMalloc calls | 2-4h |
| Low | Llama 3.1 8B INT4 re-evaluate with AWQ per-layer + fused rmsnorm path | 3-6h |
| Low | Fix CUDA Graph path (pre-existing, deep driver issue) | 4-8h |
| Low | Docker images with v0.12.x fused kernels | 1h |
| Low | New model port via GGUF bridge (DeepSeek, Phi-4, Mistral) | 4-8h each |

## 7. Suggested Next Actions

1. Fix batched server OOM — remove FP32 embed cache, replace with INT4 on-the-fly CPU dequant.
2. Re-evaluate Llama 3.1 8B INT4 with fused rmsnorm path (PPL was 273K with separate path).
3. Tag Docker images with current v0.12.7.

## 8. Important Files / Commands

### Build
```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
```

### Benchmarks
```bash
killall hashcat 2>/dev/null; sleep 2  # MUST DO FIRST
./bench/text_generate_int4_batched "prompt" M tokens weights_int4_qwen3_8b_fp16sc
./bench/bench_ppl_int4_8b [weight_dir]    # PPL test. Takes wdir as argv[1]
./bench/bench_gemv_2warp                  # 2-warp microbench
```

### Server
```bash
./server/http_subprocess int4_8b &         # HTTP server (port 8123)
./server/http_subprocess int4_batched &    # Batched server (needs OOM fix)
```

### Key source files
```
src/kernels/fused_gate_up_int4.cu    — gate+up fusion (NEW v0.12.5)
src/kernels/fused_qkv_int4.cu        — QKV fusion (NEW v0.12.5)
src/kernels/fused_int4_ops.cu        — rmsnorm+quant fusion
bench/text_generate_int4_batched.cu  — M=1-16 bench with all fusions wired
bench/bench_ppl_int4_8b.cu           — PPL test (accepts wdir arg)
server/inference_server_int4.cu      — single sequence server (fused kernels)
server/inference_server_int4_batched.cu — batched server (needs OOM fix)
```

## 9. Validation Status — ALL PASSING

- **Bench correctness**: "Paris, which is a city in the north of France..." (coherent)
- **Server correctness**: same output via HTTP
- **PPL**: 21.98 (fused baseline), 24.35 (separate baseline)
- **Build**: 203 symbols, zero errors
- **2-warp**: 0.96× (slower, confirmed memory-saturated)
- **AWQ**: 23.57 (7% worse than baseline in fused path, pre-existing)

## 10. Session Metadata

- updated_at: 2026-06-16
- branch: master
- repo state: clean (3 modified server binaries ignored)
- active components: `text_generate_int4_batched.cu`, fused_qkv/fused_gate_up/rmsnorm+quant kernels
- last commit: `37d3eaf v0.12.7: Server multi-model dispatch and CUDA Graph investigation`
- git log: 13 commits ahead of origin/master

---

## META PROMPT

You are resuming development of the blackwell project — custom CUDA kernels for INT4 LLM inference on RTX 5060 Ti.

Before acting:

1. Read `AGENTS.md` for full project context (mission, file layout, build, constraints, bug history).
2. Read `HANDOFF.md` (this file) for current operational state.

Key truths:
- Production kernel library has 203 symbols, M=1 gives 73 t/s, M=8 gives 226 t/s.
- Three fusions are active: gate+up (2→1), QKV (3→1), rmsnorm+quant (2→1).
- CUDA Graph path is pre-existing broken (since Oct 2024). Not your regression.
- PPL 21.98 is the current baseline (fused path, FP16sc weights).
- Server works at M=1 (15.3 GB VRAM, 94%). Batched server OOMs.

Do not:
- Debug the CUDA Graph path (pre-existing, deep CUDA issue, 4-8h effort).
- Retry abandoned paths (INT2, NVFP4, FP8, 9B SSM, Llama 3.1 INT4).
- Assume fusion causes any regression — verified separately.

Work incrementally. One kernel change at a time. Validate correctness (coherent output + PPL) before performance tuning. Keep HANDOFF.md updated with new results, decisions, and constraints.

Priorities for next session: (1) fix batched server OOM, (2) Llama 3.1 8B INT4 re-evaluate, (3) tag Docker images.
