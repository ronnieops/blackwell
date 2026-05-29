# HANDOFF.md — blackwell

Continuity doc. Read before acting.

---

## 1. Current Objective

**INT8 production-complete.** All modes pass, all stubs implemented, GEMM verified, RoPE in CUDA Graph, thread-safety fixed. Ready for deployment or new feature work.

---

## 2. Current Status

### Environment

| Component | Value |
|-----------|-------|
| Driver | 580.159.04 (open kernel modules) |
| CUDA toolkit | 13.3.33 |
| GPU | RTX 5060 Ti (SM_120a, 36 SMs) |
| Library | **82 symbols** in libblackwell_kernels.a |
| Thread-safety | Atomic spin-locks for init (decode.cu) |

### Throughput (28L, Qwen3-1.7B)

| Path | t/s | Notes |
|------|-----|-------|
| INT8 CUDA Graph | **99** | ✅ Best single-user (with RoPE) |
| INT8 per-kernel | **92-94** | ✅ |
| INT8 pipeline (4L scaled) | **92.5** | ✅ decode_full_int8 |
| Mode D prefill+decode | **87** | ✅ INT8 GEMM, 165ms prefill |
| Mode D decode only | **87** | ✅ |
| Batched GEMV M=4 | **61** req/s | ✅ Multi-user (3.4× after fix) |
| Batched GEMV M=8 | 17344 batch t/s | ✅ Peak |
| Speculative (M=4) | 227 batch t/s | ✅ 2.18× vs autoregressive |

### vs llama.cpp baseline (114 t/s Q4_K_M)

INT8 CUDA Graph: **87%** of baseline. Re-quantized weights correct but 15% slower than pre-fix measurements.

text_generate: **Correct output** ("Paris", "Versailles" for France prompt).

---

## 3. Recent Decisions

| Decision | Rationale |
|----------|-----------|
| INT8 is production path | 2.65× faster than NVF4 |
| Per-row scales [N, K/16] | GEMV/GEMM kernels use `n_out * num_K_blks` indexing |
| CUDA Graph with RoPE | `fused_rope_decode` reads seq_pos from device memory |
| Persistent kernel removed | 23× slower, 0 callers, dead code |
| GEMM __dp4a abandoned | On-the-fly quant overhead 4× slower than scalar |
| GEMM 4×4 tile kept | Larger tiles reduce occupancy, no benefit |
| Atomic spin-locks | CUDA 13.3 lacks `<mutex>`/`<atomic>` in .cu files |

---

## 4. Constraints

- **Compiler**: `PATH=/usr/local/cuda-13.3/bin:$PATH` before cmake
- **Arch**: `compute_120a` required (not `compute_120`)
- **phase_a.cu**: DO NOT USE — will not link
- **INT8 block size**: 16, per-row scales
- **INT8 GEMM**: K must be multiple of 16
- **CUDA 13.3**: No `<mutex>`/`<atomic>` headers in .cu files — use GCC `__sync` builtins

---

## 5. Known Issues

1. **Throughput regression** — 15% drop from re-quantization (correct scale layout). Cost of correctness. Old weights had misaligned scales that happened to be faster.
2. **FP32 text_generate** — Precision accumulation over 28 layers. Not a code bug — inherent to BF16 format.
3. **GEMM __dp4a** — On-the-fly FP32→INT8 quant overhead exceeds benefit. Would need pre-quantized activations.
4. **Docker/API packaging** — Not yet done.

---

## 6. Pending Tasks

- [ ] Package inference server (Docker, API wrapper)
- [ ] Consider pre-quantized activations for GEMM __dp4a path

---

## 7. Files & Commands

### Build
```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
```

### Run
```bash
./bench/text_generate "The capital of France is" 30       # Correct output
./bench/inference_server 28 4 20 8                         # 99 t/s CUDA Graph, 87 t/s Mode D
./bench/verify_gemm 128                                    # 7/7 GEMM correctness PASS
./bench/decode_full_int8 4                                 # 92.5 t/s scaled
```

### Key files
| File | Purpose |
|------|---------|
| `src/kernels/gemv_int8.cu` | INT8 GEMV + GEMM (production) |
| `src/kernels/rope.cu` | RoPE + fused_rope_decode |
| `src/kernels/decode.cu` | Attention decode + seq_pos (thread-safe) |
| `src/kernels/fused_o_norm.cu` | RMSNorm + INT8 quant |
| `src/kernels/cuda_graphs.cu` | CUDA Graph lifecycle API |
| `src/kernels/memory.cu` | Shared-memory tiled copy + async pipeline |
| `src/kernels/prefill.cu` | Prefill layer orchestration |
| `bench/text_generate.cu` | Text generation (correct output) |
| `bench/inference_server.cu` | CUDA Graph + batched serving + Mode D |
| `bench/verify_gemm.cu` | GEMM correctness verification |
| `scripts/quantize_per_row.py` | INT8 quantization (per-row scales) |
| `include/blackwell/kernels.h` | Public API (82 symbols) |

---

## 8. Validation

| Check | Status |
|-------|--------|
| Library build | ✅ 82 symbols |
| INT8 GEMV | ✅ 775 GB/s kernel |
| INT8 GEMM (7 projections) | ✅ cosine=1.00000000, max_err<0.00002 |
| INT8 pipeline 28L | ✅ 92.5 t/s |
| INT8 CUDA Graph + RoPE | ✅ 99 t/s |
| text_generate output | ✅ Correct |
| Mode A (per-kernel) | ✅ 92-94 t/s |
| Mode A' (CUDA Graph) | ✅ 99 t/s |
| Mode B (batched per-kernel) | ✅ 23 req/s |
| Mode C (batched GEMV) | ✅ 61 req/s |
| Mode D (INT8 GEMM prefill) | ✅ 87 t/s |
| Speculative decode | ✅ 2.18× batch throughput |
| Thread-safety | ✅ Atomic spin-locks |
| Stub functions | ✅ 8/8 implemented |

---

## 9. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-05-29 |
| branch | master |
| last_commit | `f364667` fix: multi-stream thread-safety |
| repo_state | 82 symbols, clean working tree (untracked binaries) |
| sessions_completed | 2 (scale fix + feature implementation) |

---

## META PROMPT

**Boot sequence**: Read `AGENTS.md` → `HANDOFF.md` → `git status --short` → `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l`.

**Current state**: INT8 production-complete at 99 t/s CUDA Graph with RoPE. All 8 stubs implemented. GEMM verified 7/7. Thread-safety fixed. 82 library symbols.

**What to do next**: Deploy (Docker/API) or new feature work. Core kernel development is done.

**Critical things to NOT do**:
- Don't use `compute_120` — must be `compute_120a`
- Don't use `/usr/bin/ptxas` — it's CUDA 12.0
- Don't use `phase_a.cu` — will not link
- Don't use NVF4 MMA for GEMV — scale factor layout mismatch
- Don't use `<mutex>`/`<atomic>` in .cu files — CUDA 13.3 doesn't support them
- Don't expect FP32 text_generate to match INT8 — precision accumulation over 28 layers
- Don't use `n_blk * num_K_blks` for scale access — must use `n_out * num_K_blks` (per-row layout)
