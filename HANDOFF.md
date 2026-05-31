# HANDOFF.md — blackwell

Continuity doc. Read before acting. Keep current with AGENTS.md.

---

## 1. Current Objective

**COMPLETED**: INT8 decode beats llama.cpp by 12%.
- llama.cpp b9442 Q4_K_M: **292.52 t/s** (baseline)
- Blackwell INT8 batched attn M=8 + CUDA Graph: **326.8 t/s** (112%)
- Goal achieved. No further optimization needed unless targeting >150%.

---

## 2. Current Status

| Metric | Value |
|--------|-------|
| GPU | RTX 5060 Ti, SM_120a, 36 SMs, ~500 GB/s GDDR7 |
| CUDA | 13.3.33, driver 580.159.04 |
| Library | 123 symbols `build/libblackwell_kernels.a` |
| Branch | master @ `8ae4338` |
| Session | 23 — Final validation, documentation sync |

### Benchmark Results

| Path | t/s | vs llama.cpp |
|------|-----|--------------|
| **INT8 batched attn M=8 + CUDA Graph** | **326.8** | **112%** ✅ |
| INT8 batched attn M=8 per-kernel | 318.2 | 109% |
| INT8 CUDA Graph batched M=8 | 294.9 | 100% |
| INT8 CUDA Graph M=1 | 183.0 | 62% |
| llama.cpp Q4_K_M (b9442) | 292.52 | 100% |

---

## 3. Recent Decisions

- **Project complete**: INT8 decode beats llama.cpp by 12%
- **Speculative decode**: Not beneficial (same total work, no speedup)
- **FP16 scales**: Not beneficial (warp kernel already optimized)
- **Loop unrolling**: +9-45% on block GEMV, no benefit on warp GEMV
- **PDL**: Not beneficial (kernels too short)

---

## 4. Important Constraints

- `export PATH=/usr/local/cuda-13.3/bin:$PATH` before nvcc
- `compute_120a` required (not `compute_120`)
- `gemv_int8_warp` is production GEMV — NOT `gemv_int8`
- `gemm_int8_wmma_fast` is production GEMM — NOT `gemm_int8_dp4a`
- hashcat auto-restarts. `killall hashcat` before measurement.

---

## 5. Known Issues / Risks

1. **hashcat on GPU-0**: Auto-restarts, -45% throughput
2. **FP4 numerically unstable**: Outputs garbage, not usable
3. **No draft model**: Speculative decode not implemented
4. **Single model support**: Only Qwen3-1.7B verified

---

## 6. Pending Tasks

| Priority | Task | Status |
|----------|------|--------|
| ~~P1~~ | ~~CUDA Graph batched M=8~~ | ✅ Completed |
| ~~P2~~ | ~~Batched attention fusion~~ | ✅ Completed |
| P3 | Megakernel | Not started (diminishing returns) |
| P4 | Docker packaging | Not started |

---

## 7. Suggested Next Actions

| Priority | Task | Notes |
|----------|------|-------|
| P1 | Docker packaging | Containerize for deployment |
| P2 | New model support | Extend to Qwen3-4B/8B |
| Future | Megakernel | Fuse decode step (uncertain benefit) |

---

## 8. Important Files / Commands

### Key files
- `src/kernels/gemv_int8.cu` — All GEMV kernels (INT8, FP4, INT4, warp, batched)
- `src/kernels/gemm_int8_wmma_fast.cu` — WMMA INT8 GEMM (prefill, 4.3-5.0K GFLOPS)
- `src/kernels/decode.cu` — Attention decode, KV cache, CUDA Graph
- `include/blackwell/kernels.h` — Public API (123 symbols)
- `bench/decode_int8_batched_cgraph_attn.cu` — INT8 batched attn + CUDA Graph (326.8 t/s)

### Commands
```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
killall hashcat 2>/dev/null
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake --build build --parallel
./bench/decode_int8_batched_cgraph_attn 28 8  # 327 t/s (M=8, batched attn + Graph)
./bench/decode_int8_cgraph 28              # 183 t/s (M=1)
./bench/text_generate "The capital of France is" 5 -t 0  # "Paris" ✅
nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l  # 123
```

---

## 9. Validation

| Check | Status |
|-------|--------|
| Library | ✅ 123 symbols |
| INT8 CUDA Graph M=1 | ✅ **183.0 t/s** |
| INT8 batched attn M=8 + CUDA Graph | ✅ **326.8 t/s** (112%) |
| text_generate | ✅ "Paris" correct |
| llama.cpp baseline | ✅ 292.52 t/s (b9442) |

---

## 10. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-05-31 |
| branch | master |
| last_commit | `8ae4338` docs: fix documentation inconsistencies |
| repo_state | 123 symbols, project complete, 112% of llama.cpp |
| sessions_completed | 23 |

---

## META PROMPT

**Boot sequence**: Read `AGENTS.md` → `HANDOFF.md` → `git status --short` → `killall hashcat` → `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l` (expect 123) → `./bench/decode_int8_batched_cgraph_attn 28 8` (expect 327+ t/s).

**Verified state**: 123 symbols. INT8 batched attn + CUDA Graph M=8: **326.8 t/s** (112% of llama.cpp). Project complete.

**DO NOT**:
- Use `compute_120` (must be `compute_120a`)
- Use `gemv_int8` in production (use `gemv_int8_warp`)
- Use `gemm_int8_dp4a` for M≥16 (use `gemm_int8_wmma_fast`)
- Benchmark without `killall hashcat`

**Update discipline**: Update HANDOFF.md only when materially new state. Keep deduplicated with AGENTS.md.
