# HANDOFF.md — blackwell

Continuity doc. Read before acting.

---

## 1. Current Objective

**INT8 production-complete.** Deploy or optimize further. NVF4 on hold.

---

## 2. Current Status

### Environment

| Component | Value |
|-----------|-------|
| Driver | 580.159.04 (open kernel modules) |
| CUDA toolkit | 13.3.33 |
| GPU | RTX 5060 Ti (SM_120a, 36 SMs) |
| Library | 76 symbols in libblackwell_kernels.a |

### Throughput (28L, Qwen3-1.7B)

| Path | t/s | Notes |
|------|-----|-------|
| INT8 CUDA Graph | **128** | ✅ Best single-user |
| INT8 per-kernel | **117** | ✅ |
| INT8 pipeline (4L scaled) | **93.9** | ✅ decode_full_int8 |
| text_generate | ~30 | Cold start + tokenizer overhead |
| Batched GEMV M=4 | 40 req/s | ✅ Multi-user |
| Batched GEMV M=8 | 17344 batch t/s | ✅ Peak |
| NVF4 scalar GEMV | 98 GB/s | ✅ Correct (cosine 0.999), not competitive |
| GEMM prefill | 78 GB/s | ✅ 3× llama.cpp |

### vs llama.cpp baseline (114 t/s Q4_K_M)

INT8 CUDA Graph: **112%** of baseline ✅

---

## 3. Recent Decisions

| Decision | Rationale |
|----------|-----------|
| INT8 is production path | 2.65× faster than NVF4 |
| NVF4 MMA abandoned | Scale factor layout mismatch for GEMV |
| CUDA Graph for single-user | 10% speedup, clean |
| Batched GEMV for multi-user | 18.86× throughput gain |

---

## 4. Constraints

- **Compiler**: `PATH=/usr/local/cuda-13.3/bin:$PATH` before cmake
- **Arch**: `compute_120a` required (not `compute_120`)
- **phase_a.cu**: DO NOT USE — unimplemented symbols
- **INT8 block size**: 16, per-row scales

---

## 5. Known Issues

1. **Mode D prefill** — FIXED (6e775eb). GEMM B buffer OOB in synthetic prefill. Now runs: 68 t/s full pipeline, 106 t/s decode.
2. **FP32 text_generate broken** — cuBLAS path worse than INT8. Separate issue.
3. **GEMM prefill correctness** — no reference comparison. Timing-only.
4. **7 stub functions** — unimplemented (see AGENTS.md §7).

---

## 6. Pending Tasks

### Low priority
- [ ] Fix FP32 text_generate (cuBLAS path)
- [ ] Verify GEMM prefill correctness against reference

---

## 7. Files & Commands

### Build
```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
```

### Run
```bash
./bench/decode_full_int8 4                    # 93.9 t/s
./bench/text_generate "The capital of France is" 30  # "Paris" ✓
./bench/inference_server 28 4 20 8            # 128 t/s CUDA Graph
```

### Key files
| File | Purpose |
|------|---------|
| `src/kernels/gemv_int8.cu` | INT8 GEMV (production) |
| `src/kernels/gemv_fp4_nv.cu` | NVF4 scalar GEMV (research) |
| `bench/decode_full_int8.cu` | INT8 pipeline benchmark |
| `bench/text_generate.cu` | Text generation (correct output) |
| `bench/inference_server.cu` | CUDA Graph + batched serving |
| `include/blackwell/kernels.h` | Public API (76 symbols) |

---

## 8. Validation

| Check | Status |
|-------|--------|
| Library build | ✅ 76 symbols |
| INT8 GEMV | ✅ 260 GB/s |
| INT8 pipeline 28L | ✅ 93.9 t/s |
| INT8 CUDA Graph | ✅ 128 t/s |
| text_generate output | ✅ "Paris" |
| inference_server Modes A-C | ✅ |
| inference_server Mode D | ✅ 68 t/s pipeline |
| NVF4 scalar GEMV | ✅ cosine 0.999 |
| NVF4 MMA | ❌ abandoned |

---

## 9. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-05-29 |
| branch | master |
| last_commit | `6e775eb` Mode D prefill OOB fix |
| repo_state | Clean (binaries untracked) |
| library | 76 symbols |

---

## META PROMPT

**Boot sequence**: Read `AGENTS.md` → `HANDOFF.md` → `git status --short` → `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l`.

**Current state**: INT8 production-complete at 93.9 t/s (128 t/s CUDA Graph). NVF4 research complete — 98 GB/s scalar (can't match INT8's 260 GB/s).

**What to do next**: Proceed to next optimization or deploy. All modes pass. All Phase G bugs fixed.

**Critical things to NOT do**:
- Don't use `compute_120` — must be `compute_120a`
- Don't use `/usr/bin/ptxas` — it's CUDA 12.0
- Don't use `phase_a.cu` — will not link
- Don't use NVF4 MMA for GEMV — scale factor layout mismatch
- Don't expect NVF4 to match INT8 — scalar FP4→float ceiling is 98 GB/s
