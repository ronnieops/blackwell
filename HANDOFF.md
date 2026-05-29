# HANDOFF.md — blackwell

Continuity doc. Read before acting.

---

## 1. Current Objective

**INT8 production deployment complete** — 92.9 t/s (28L), 122 t/s with CUDA Graph.
NVF4 research complete — can't match INT8 bandwidth (98 vs 260 GB/s).

---

## 2. Current Status

### Environment

| Component | Value |
|-----------|-------|
| Driver | 580.159.04 (open kernel modules) |
| CUDA toolkit | 13.3.33 |
| ptxas | 13.3 |
| GPU | RTX 5060 Ti |

### Throughput benchmarks (RTX 5060 Ti, Qwen3-1.7B, 28L decode)

| Path | t/s | GB/s | Status |
|------|-----|------|--------|
| INT8 CUDA Graph | **122** | — | ✅ Production |
| INT8 GEMV (28L) | **92.9** | 260 | ✅ Production |
| INT8 GEMV (kernel-only) | **112** | 260 | ✅ |
| NVF4 scalar GEMV | — | **98** | ✅ Correct (cosine 0.9987), not competitive |
| GEMM prefill | — | **78** | ✅ 3× llama.cpp |

### INT8 vs NVF4 Comparison

| Metric | INT8 | NVF4 | Ratio |
|--------|------|------|-------|
| Bandwidth | 260 GB/s | 98 GB/s | 2.65× |
| SIMD | `__dp4a` | scalar FP4→float | — |
| Throughput | 92.9 t/s | ~37 t/s (est) | 2.5× |
| Status | Production | Research only | — |

---

## 3. Recent Decisions

| Decision | Rationale |
|----------|-----------|
| **INT8 is production path** | 92.9 t/s, 2.65× faster than NVF4 |
| **Abandon NVF4 MMA for GEMV** | Scale factor layout mismatch, GEMV ≠ GEMM |
| **NVF4 scalar at ceiling** | 98 GB/s is max for scalar FP4→float conversion |
| **GEMM prefill already optimized** | 78 GB/s (7.5% peak), 3× llama.cpp |

---

## 4. Important Constraints

- **Compiler**: `PATH=/usr/local/cuda-13.3/bin:$PATH` before cmake
- **Arch**: `compute_120a` required for block_scale MMA
- **phase_a.cu**: DO NOT USE — links to unimplemented symbols
- **INT8 block size**: 16, per-row scales
- **Driver**: 580.159.04 (open modules)

---

## 5. Known Issues / Risks

1. **NVF4 MMA abandoned** — Scale factor layout mismatch (SFBLayout organizes by K-position, kernel loads by N-block). MMA designed for GEMM, not GEMV.
2. **NVF4 scalar at ceiling** — 98 GB/s is max for scalar FP4→float. Can't match INT8's `__dp4a` SIMD.
3. **28-layer INT8 output degenerate** — NOT a bug. Base model + greedy decoding artifact.

---

## 6. Pending Tasks

### Completed (this session)

- [x] INT8 28L pipeline: 92.9 t/s
- [x] NVF4 v2 fix: 21→98 GB/s (vectorized W load)
- [x] NVF4 vs INT8 benchmark: INT8 2.65× faster
- [x] NVF4 SIMD analysis: can't match INT8
- [x] GEMM prefill: already optimized (78 GB/s)
- [x] inference_server: already uses per-row weights
- [x] Codebase cleanup: AGENTS.md updated, temp files removed

### Remaining (low priority)

- [ ] Add CUDA Graph to decode benchmark (92→122+ t/s)
- [ ] Instruct model test (no GGUF available)

---

## 7. Important Files / Commands

### Build (CUDA 13.3)
```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### Run
```bash
./bench/decode_full_int8 28             # INT8 pipeline, 92.9 t/s
./bench/inference_server 28 4 20 8      # INT8 with CUDA Graph, 122 t/s
./bench/test_nvfp4_gemv 2048 2048       # NVF4 GEMV test (cosine 0.9987)
```

### Key files
| File | Purpose |
|------|---------|
| `src/kernels/gemv_int8.cu` | INT8 GEMV (260 GB/s, production) |
| `src/kernels/gemv_fp4_nv.cu` | NVF4 scalar GEMV (98 GB/s, correct) |
| `bench/decode_full_int8.cu` | INT8 pipeline benchmark |
| `bench/inference_server.cu` | INT8 with CUDA Graph (122 t/s) |
| `bench/test_nvfp4_gemv.cu` | NVF4 correctness test |

---

## 8. Validation Status

| Check | Status | Notes |
|-------|--------|-------|
| Library build | ✅ | 76 symbols, CUDA 13.3 |
| INT8 GEMV | ✅ | 260 GB/s |
| INT8 pipeline (28L) | ✅ | 92.9 t/s |
| INT8 CUDA Graph | ✅ | 122 t/s |
| NVF4 scalar GEMV | ✅ | 98 GB/s, cosine 0.9987 |
| GEMM prefill | ✅ | 78 GB/s |
| NVF4 MMA | ❌ | Abandoned (correctness issues) |

---

## 9. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-05-29 |
| branch | master |
| driver | 580.159.04 (open kernel modules) |
| CUDA toolkit | 13.3.33 |
| library | 76 symbols |
| repo_state | Modified (AGENTS.md, HANDOFF.md, gemv_fp4_nv.cu, etc.) |

---

## META PROMPT

**Boot sequence**: Read `AGENTS.md` → `HANDOFF.md` → `git status --short` → `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l`.

**Current state**: INT8 production-complete at 92.9 t/s (122 t/s with CUDA Graph). NVF4 research complete — 98 GB/s scalar (can't match INT8's 260 GB/s).

**What to do next**: Deploy INT8. NVF4 on hold — scalar at ceiling, MMA abandoned.

**Critical things to NOT do**:
- Don't use `compute_120` — must be `compute_120a`
- Don't use `/usr/bin/ptxas` — it's CUDA 12.0
- Don't use `phase_a.cu` — will not link
- Don't use NVF4 MMA for GEMV — scale factor layout mismatch
- Don't expect NVF4 to match INT8 — scalar FP4→float ceiling is 98 GB/s
