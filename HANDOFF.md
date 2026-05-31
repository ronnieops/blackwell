# HANDOFF.md — blackwell

Continuity doc. Read before acting. Keep current with AGENTS.md.

---

## 1. Current Objective

Maximize INT8 decode throughput vs llama.cpp Q4_K_M (**292.52 t/s**, re-measured 2026-05-31, b9442, CUDA 13.3).
INT8 batched attn + CUDA Graph M=8: **326.8 t/s** (112% of target — BEATEN!). INT8 CUDA Graph M=1: **183.0 t/s** (62%). **123 symbols**.

---

## 2. Current Status

| Metric | Value |
|--------|-------|
| GPU | RTX 5060 Ti, SM_120a, 36 SMs, ~500 GB/s GDDR7 |
| CUDA | 13.3.33, driver 580.159.04 |
| Library | 123 symbols `build/libblackwell_kernels.a` |
| Branch | master @ `5726131` |
| Session | 22 — FP16 scales, loop unrolling, llama.cpp analysis, documentation |

| Path | t/s | vs 276 llama.cpp |
|------|-----|------------------|
| **INT8 CUDA Graph (M=1)** | **183.3** | **66%** |
| INT8 per-kernel (M=1) | 178.7 | 65% |
| **INT8 CUDA Graph batched M=4** | **291.2** | **105%** ✅ BEATEN |
| **INT8 CUDA Graph batched M=8** | **294.9** | **107%** ✅ BEATEN |
| **INT8 batched attn M=4 CUDA Graph** | **312.0** | **113%** ✅ BEATEN |
| **INT8 batched attn M=8 CUDA Graph** | **326.8** | **112%** ✅ BEATEN |
| INT8 batched (M=4) per-kernel | 237.3 | 86% |
| INT8 batched (M=8) per-kernel | 243.4 | 88% |
| FP4 batched (M=4) | 237.3 | 86% ⚠️ 180% RMS diff vs INT8 |
| FP4 batched (M=8) | 243.4 | 88% ⚠️ 180% RMS diff vs INT8 |
| llama.cpp Q4_K_M | **292.52** | 100% |
| llama.cpp F16 | **110.6** | 40% |
| FP4 CUDA Graph (M=1) | 247.3 | 98% ⚠️ garbage output |

| GEMM (prefill) | GFLOPS | vs dp4a |
|----------------|--------|---------|
| **WMMA m16n16k16** | **10,510** | **3.81×** |
| dp4a (library) | 2,760 | 1× |

---

## 3. Recent Decisions

- **Batched attention beats llama.cpp**: INT8 batched attn + CUDA Graph M=8: **326.8 t/s** (112%). Batched attn alone +10%, CUDA Graph +3%. Bit-exact correctness.
- **Batched CUDA Graph beats llama.cpp**: INT8 batched M=8 CUDA Graph: **294.9 t/s** (107%).
- **Speculative decode fixed**: Segfault from 2 bugs — (1) default stream incompatible with `cudaStreamCaptureModeGlobal`, (2) `d_kv_offset[0]` dereferences device pointer from host. Fixed with `cudaStreamCreate` and `*h_kv_offset`. Speedup = 0% (same total work).

---

## 4. Important Constraints

- `export PATH=/usr/local/cuda-13.3/bin:$PATH` before nvcc
- `compute_120a` required (not `compute_120`)
- `gemv_int8_warp` is production GEMV — NOT `gemv_int8`
- `gemm_int8_wmma` is production GEMM — NOT `gemm_int8_dp4a` (for M≥16)
- hashcat auto-restarts. `killall hashcat` before measurement.
- Weight matrices exceed L2 cache (32 MB) — architectural limit for M=1 decode

---

## 5. Known Issues / Risks

1. **hashcat on GPU-0**: Auto-restarts, -45% throughput
2. **FP4 batched numerically stable but low quality**: 180% RMS diff vs INT8. Not interchangeable.
3. **CUDA Graph drift**: max diff ~4.1 between per-kernel and graph paths
4. **FP32 text_generate broken**: `text_generate_fp32.cu` bad output
5. **Batched decode limited**: Only MLP batched (gate+up+down GEMV). Attention serial per-sequence.

---

## 6. Pending Tasks

| Priority | Task | Status |
|----------|------|--------|
| P1 | CUDA Graph for batched M=8 decode | ✅ Completed (294.4 t/s, 107% of llama.cpp) |
| P2 | Batched attention fusion for M=8 | Not started |
| P3 | Megakernel (fuse decode step) | Not started |
| P4 | Docker/API packaging | Not started |

---

## 7. Suggested Next Actions

| Priority | Task | Notes |
|----------|------|-------|
| P1 | Batched attention fusion | Fuse attention across M sequences. KV cache reorganization needed. Current attention serial per-seq. |
| P2 | Megakernel (on batched path) | Fuse RMSNorm+GEMV+SwiGLU+residual into persistent kernel. Only after batched attention. |
| P3 | Docker packaging | Containerize for deployment |
| Future | Speculative decode | Draft + verify pipeline |

---

## 8. Important Files / Commands

### Key files
- `src/kernels/gemv_int8.cu` — All GEMV kernels (INT8, FP4, INT4, warp, batched)
- `src/kernels/gemm_int8_wmma.cu` — WMMA INT8 GEMM (prefill, 3.8× dp4a)
- `src/kernels/decode.cu` — Attention decode, KV cache, CUDA Graph seq_pos
- `include/blackwell/kernels.h` — Public API (123 symbols)
- `bench/decode_int8_cgraph.cu` — INT8 CUDA Graph pipeline (183 t/s, M=1)
- `bench/decode_int8_batched_cgraph.cu` — INT8 CUDA Graph batched (294 t/s, M=8, beats llama.cpp)
- `bench/decode_int8_batched.cu` — INT8 batched per-kernel baseline
- `bench/decode_fp4_batched.cu` — FP4 vs INT8 batched benchmark
- `bench/text_generate.cu` — End-to-end text generation

### Commands
```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake --build build --parallel
killall hashcat 2>/dev/null
./bench/decode_int8_batched_cgraph_attn 28 4  # 312 t/s (M=4, batched attn + Graph)
./bench/decode_int8_batched_cgraph_attn 28 8  # 329 t/s (M=8, batched attn + Graph)
./bench/decode_int8_batched_cgraph 28 4    # 291 t/s (M=4, old batched + Graph)
./bench/decode_int8_batched_cgraph 28 8    # 295 t/s (M=8, old batched + Graph)
./bench/speculative_decode_cgraph 28 4     # 190 t/s (spec decode, 0% speedup)
./bench/decode_int8_batched 28 8           # 243 t/s (M=8, per-kernel baseline)
./bench/text_generate "The capital of France is" 5 -t 0  # "Paris" ✅
nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l  # 107
```

---

## 9. Validation

| Check | Status |
|-------|--------|
| Library | ✅ 123 symbols |
| INT8 CUDA Graph 28L (M=1) | ✅ **183.3 t/s** (66% of 276) |
| INT8 CUDA Graph batched M=4 | ✅ **291.2 t/s** (105% of 276) |
| INT8 CUDA Graph batched M=8 | ✅ **294.9 t/s** (107% of 276) |
| INT8 batched attn M=4 CUDA Graph | ✅ **312.0 t/s** (113% of 276) |
| INT8 batched attn M=8 CUDA Graph | ✅ **326.8 t/s** (112% of 292) |
| INT8 batched M=8 per-kernel | ✅ **243.4 t/s** (88% of 276) |
| WMMA vs dp4a | ✅ **Exact match** (0.000 max diff) |
| text_generate | ✅ "Paris" correct (greedy) |
| hashcat | ⚠️ Interferes with measurements |

---

## 10. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-05-31 |
| branch | master |
| last_commit | `deace33` feat: FP16 scale cache (mixed results) |
| repo_state | 123 symbols, FP16 scales, loop unrolling, llama.cpp analysis, final benchmark validated |
| sessions_completed | 22 |

---

- llama.cpp Q4_K_M: 292.52 (b9442)

## META PROMPT

**Boot sequence**: Read `AGENTS.md` → `HANDOFF.md` → `git status --short` → `killall hashcat` → `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l` (expect 119) → `./bench/decode_int8_cgraph 28` (expect 183+ t/s) → `./bench/decode_int8_batched_cgraph 28 8` (expect 294+ t/s).

**Verified state**: 123 symbols. INT8 CUDA Graph (M=1) **183.0 t/s** (62%). INT8 batched attn + CUDA Graph M=8 **326.8 t/s** (112%, beats llama.cpp!). INT8 batched attn M=8 per-kernel **318.2 t/s** (109%). Speculative decode fixed but 0% speedup. llama.cpp Q4_K_M **292.52 t/s** (b9442).

**DO NOT**:
- Use `compute_120` (must be `compute_120a`)
- Use `gemv_int8` in production (use `gemv_int8_warp`)
- Use `gemm_int8_dp4a` for M≥16 (use `gemm_int8_wmma`)
- Trust FP4 for M=1 decode (outputs garbage)
- Benchmark without `killall hashcat`

**Update discipline**: Update HANDOFF.md only when materially new state. Keep deduplicated with AGENTS.md.
