# HANDOFF.md — blackwell

Continuity doc. Read before acting. Keep current with AGENTS.md.

---

## 1. Current Objective

Maximize INT8 decode throughput vs llama.cpp Q4_K_M (**276.0 t/s**, re-measured 2026-05-30, b9389).
INT8: **183.5 t/s** CUDA Graph (66% of target). **103 symbols**. Batched M=8 reaches 88% of target.

---

## 2. Current Status

| Metric | Value |
|--------|-------|
| GPU | RTX 5060 Ti, SM_120a, 36 SMs, ~500 GB/s GDDR7 |
| CUDA | 13.3.33, driver 580.159.04 |
| Library | 103 symbols `build/libblackwell_kernels.a` |
| Branch | master @ `f55a705` |

| Path | t/s | vs 276 llama.cpp |
|------|-----|------------------|
| **INT8 CUDA Graph (M=1)** | **183.5** | **66%** |
| INT8 per-kernel (M=1) | 162.9 | 59% |
| INT8 batched (M=4) | **237.3** | **86%** |
| INT8 batched (M=8) | **243.4** | **88%** |
| FP4 batched (M=4) | 237.3 | 86% ⚠️ 180% RMS diff vs INT8 |
| FP4 batched (M=8) | 243.4 | 88% ⚠️ 180% RMS diff vs INT8 |
| llama.cpp Q4_K_M | **276.0** | 100% |
| llama.cpp F16 | **110.6** | 40% |
| FP4 CUDA Graph (M=1) | 247.3 | 98% ⚠️ garbage output |

| GEMM (prefill) | GFLOPS | vs dp4a |
|----------------|--------|---------|
| **WMMA m16n16k16** | **10,510** | **3.81×** |
| dp4a (library) | 2,760 | 1× |

---

## 3. Recent Decisions

- **Batched decode (M≥4)**: INT8 batched M=8: 243.4 t/s (88% of 276 llama.cpp). Attention still serial per-sequence (KV cache). Batched MLP only.
- **FP4 abandoned for M=1**: Error amplifies 16.8×/layer through RMSNorm. E2M1 min=0.5 can't represent small FP32 values.
- **FP4 batched numerically stable but low quality**: 180% relative RMS diff vs INT8. Different quantization schemes not interchangeable.
- **WMMA dequant fixed**: Per-block scale now matches dp4a exactly (0.000 max diff). Removed cudaMalloc per call.
- **WMMA tensor cores**: 3.81× speedup for INT8 GEMM (prefill path). `gemm_int8_wmma` in library.
- **text_generate migrated**: Uses `quantize_int8` + `gemv_int8_warp`. Outputs "Paris" correctly.
- **llama.cpp baseline updated**: Q4_K_M now 276.0 t/s (b9389, was 270.1 b9159). F16: 110.6 t/s.

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
| P1 | Megakernel (fuse decode step) | Not started |
| P2 | Docker/API packaging | Not started |
| P3 | Per-block WMMA dequant | ✅ Completed (exact match vs dp4a) |

---

## 7. Suggested Next Actions

| Priority | Task | Notes |
|----------|------|-------|
| P1 | Megakernel | Fuse RMSNorm+GEMV+SwiGLU+residual into persistent kernel |
| P2 | Docker packaging | Containerize for deployment |
| Future | Speculative decode | Draft + verify pipeline |
| Future | Batched attention | Fuse attention with KV cache for true batched decode |

---

## 8. Important Files / Commands

### Key files
- `src/kernels/gemv_int8.cu` — All GEMV kernels (INT8, FP4, INT4, warp, batched)
- `src/kernels/gemm_int8_wmma.cu` — WMMA INT8 GEMM (prefill, 3.8× dp4a)
- `src/kernels/decode.cu` — Attention decode, KV cache, CUDA Graph seq_pos
- `include/blackwell/kernels.h` — Public API (103 symbols)
- `bench/decode_int8_cgraph.cu` — INT8 CUDA Graph pipeline (183 t/s)
- `bench/decode_int8_batched.cu` — INT8 batched decode (243 t/s M=8)
- `bench/decode_fp4_batched.cu` — FP4 vs INT8 batched benchmark
- `bench/text_generate.cu` — End-to-end text generation

### Commands
```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake --build build --parallel
killall hashcat 2>/dev/null
./bench/decode_int8_cgraph 28              # 183 t/s (M=1)
./bench/decode_int8_batched 28 8           # 243 t/s (M=8)
./bench/text_generate "The capital of France is" 5 -t 0  # "Paris" ✅
nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l  # 103
```

---

## 9. Validation

| Check | Status |
|-------|--------|
| Library | ✅ 103 symbols |
| INT8 CUDA Graph 28L | ✅ **183.5 t/s** (66% of 276) |
| INT8 batched M=8 | ✅ **243.4 t/s** (88% of 276) |
| WMMA vs dp4a | ✅ **Exact match** (0.000 max diff) |
| text_generate | ✅ "Paris" correct (greedy) |
| hashcat | ⚠️ Interferes with measurements |

---

## 10. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-05-30 |
| branch | master |
| last_commit | `f55a705` fix: L2 cache hint targets graph_stream |
| repo_state | 103 symbols, WMMA per-block dequant fixed, batched corrected |
| sessions_completed | 18 |

---

- llama.cpp Q4_K_M: 276.0 (b9389)

## META PROMPT

**Boot sequence**: Read `AGENTS.md` → `HANDOFF.md` → `git status --short` → `killall hashcat` → `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l` (expect 103) → `./bench/decode_int8_cgraph 28` (expect 183+ t/s).

**Verified state**: 103 symbols. INT8 CUDA Graph **183.5 t/s** (66% of 276). INT8 batched M=8 **243.4 t/s** (88%). WMMA **exact match** vs dp4a. llama.cpp Q4_K_M **276.0 t/s** (b9389).

**DO NOT**:
- Use `compute_120` (must be `compute_120a`)
- Use `gemv_int8` in production (use `gemv_int8_warp`)
- Use `gemm_int8_dp4a` for M≥16 (use `gemm_int8_wmma`)
- Trust FP4 for M=1 decode (outputs garbage)
- Benchmark without `killall hashcat`

**Update discipline**: Update HANDOFF.md only when materially new state. Keep deduplicated with AGENTS.md.
