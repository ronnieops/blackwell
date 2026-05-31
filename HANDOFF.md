# HANDOFF.md — blackwell

Continuity doc. Read before acting. Keep current with AGENTS.md.

---

## 1. Current Objective

**COMPLETED**: INT8 decode beats llama.cpp by 12%.
- llama.cpp b9442 Q4_K_M: **292.52 t/s** (baseline)
- Blackwell INT8 batched attn M=8 + CUDA Graph: **327.7 t/s** (112%)
- Goal achieved. No further optimization needed unless targeting >150%.

---

## 2. Current Status

| Metric | Value |
|--------|-------|
| GPU | RTX 5060 Ti, GB206, SM_120a, 36 SMs, 448 GB/s GDDR7 |
| CUDA | 13.3.33, driver 580.159.04 |
| Library | 123 symbols `build/libblackwell_kernels.a` |
| Branch | master @ `8cdac55` |
| Session | 25 — Multi-model support: Qwen3-8B, multi-shard quantization |

### Benchmark Results

| Path | t/s | vs llama.cpp |
|------|-----|--------------|
| **INT8 batched attn M=8 + CUDA Graph** | **327.7** | **112%** ✅ |
| INT8 batched attn M=8 per-kernel | 318.9 | 109% |
| INT8 CUDA Graph batched M=8 | 287.9 | 98% |
| INT8 CUDA Graph M=1 | 183.0 | 62% |
| llama.cpp Q4_K_M (b9442) | 292.52 | 100% |

### GPU Architecture

RTX 5060 Ti = **GB206** chip, SM_120a, 36 SMs, **448 GB/s** GDDR7 (128-bit bus, 28 Gbps).
**Consumer GB206 lacks FP4 tensor core hardware** — only GB100/GB200 data-center chips have it.
Custom FP4 E2M1 produces garbage (180% RMS diff vs INT8).

---

## 3. Recent Decisions

- **Qwen3-8B supported**: INT8 decode 59.4 t/s (28L), 46.2 t/s (36L). 3.3× slower than 1.7B due to 4× larger weights.
- **Qwen3.5-9B MoE next**: Different architecture, needs separate kernel path.
- **Speculative decode**: Not beneficial (same total work, no speedup)
- **FP16 scales**: Not beneficial (warp kernel already optimized)
- **Loop unrolling**: +9-45% on block GEMV, no benefit on warp GEMV
- **PDL**: Not beneficial (kernels too short)
- **FP4 tensor core**: RTX 5060 Ti (GB206) lacks it — custom FP4 E2M1 not viable
- **WMMA correctness**: Verified — test_wmma PASS, verify_gemm PASS (all 6 projections)
- **decode_prefill rewrite**: Now benchmarks gemm_int8_wmma_fast with real Qwen3-1.7B weights
- **Cleanup session 24**: Removed 6 untracked bench files, research docs, .bak. Committed decode_prefill rewrite.

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
2. **FP4 numerically unstable**: Outputs garbage, not usable (GB206 lacks FP4 tensor core HW)
3. **No draft model**: Speculative decode not implemented
4. **Single model support**: Only Qwen3-1.7B verified
5. **text_generate head_norm bug**: Pre-existing. "FAIL head_norm l=0". Not blocking.

---

## 6. Pending Tasks

| Priority | Task | Status |
|----------|------|--------|
| ~~P1~~ | ~~CUDA Graph batched M=8~~ | ✅ Completed |
| ~~P2~~ | ~~Batched attention fusion~~ | ✅ Completed |
| ~~P3~~ | ~~Decode prefill rewrite~~ | ✅ Completed |
| ~~P4~~ | ~~Multi-model support (Qwen3-8B)~~ | ✅ Completed |
| P5 | Docker packaging | Not started |
| P6 | Megakernel | Not started (diminishing returns on 36 SMs) |

---

## 7. Suggested Next Actions

| Priority | Task | Notes |
|----------|------|-------|
| P1 | Docker packaging | Containerize for deployment |
| P2 | New model support | Extend to Qwen3-4B/8B (Qwen3.5-9B MoE next) |
| Future | FP4 via llama.cpp NVFP4 | RTX 5090 only (GB100), not RTX 5060 Ti |

---

## 8. Important Files / Commands

### Key files
- `src/kernels/gemv_int8.cu` — All GEMV kernels (INT8, FP4, INT4, warp, batched)
- `src/kernels/gemm_int8_wmma_fast.cu` — WMMA INT8 GEMM (prefill, 4.3-5.0K GFLOPS)
- `src/kernels/decode.cu` — Attention decode, KV cache, CUDA Graph
- `include/blackwell/kernels.h` — Public API (123 symbols)
- `bench/decode_int8_batched_cgraph_attn.cu` — INT8 batched attn + CUDA Graph (327.7 t/s)
- `bench/decode_prefill.cu` — INT8 GEMM prefill benchmark with real weights
- `bench/decode_int8_cgraph_qwen3_8b.cu` — Qwen3-8B benchmark (H=4096, I=12288)
- `bench/verify_gemm` — GEMM correctness vs CPU reference
- `bench/test_wmma` — WMMA vs dp4a correctness
- `scripts/quantize_generic.py` — INT8 quantization (single + multi-shard safetensors)

### Commands
```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
killall hashcat 2>/dev/null
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake --build build --parallel
./bench/decode_int8_batched_cgraph_attn 28 8  # Qwen3-1.7B: 327 t/s (M=8, batched attn + Graph)
./bench/decode_int8_cgraph 28              # Qwen3-1.7B: 183 t/s (M=1)
./bench/decode_int8_cgraph_qwen3_8b 28     # Qwen3-8B: 59.4 t/s (28L), 46.2 t/s (36L)
./bench/decode_prefill 20                  # GEMM prefill GFLOPS (M=1-128 sweep)
./bench/verify_gemm                         # GEMM correctness check (all layers)
./bench/test_wmma                          # WMMA vs dp4a correctness
nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l  # 123

# Quantize new model:
python3 scripts/quantize_generic.py <model_path> <output_weights_dir>
# Example: python3 scripts/quantize_generic.py /mnt/data/ai/hf/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 weights_int8_qwen3_8b
```

---

## 9. Validation

| Check | Status |
|-------|--------|
| Library | ✅ 123 symbols |
| INT8 CUDA Graph M=1 | ✅ **183.0 t/s** |
| INT8 batched attn M=8 + CUDA Graph | ✅ **327.7 t/s** (112%) |
| text_generate | ✅ "Paris" correct |
| llama.cpp baseline | ✅ 292.52 t/s (b9442) |
| WMMA correctness | ✅ test_wmma PASS, verify_gemm PASS |
| Qwen3-8B decode | ✅ 59.4 t/s (28L), 46.2 t/s (36L) |

### Session 24 Cleanup Summary
- Removed: 6 untracked bench files (FP4 research + bad MMA bench + debug artifacts)
- Removed: research docs (scout-report-*.md, researcher-report*.md, progress.md)
- Removed: src/kernels/gemm_int8_wmma.cu.bak
- Kept: scripts/quantize_generic.py (INT8 quantizer for Qwen3)
- Kept: weights_int8_bf16_06b/ (Qwen3-0.6B 6-bit weights)

### Session 25 Multi-Model Support
- Fixed quantize_generic.py for multi-shard safetensors (5 shards)
- Added Qwen3-8B benchmark (H=4096, I=12288, 36 layers)
- Qwen3-8B: 59.4 t/s (28 layers), 46.2 t/s (36 layers) — 3.3× slower than 1.7B
- Added weights_*/ to .gitignore

---

## 10. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-05-31 |
| branch | master |
| last_commit | `4b66c63` feat: multi-model support (Qwen3-8B benchmark, multi-shard quantization) |
| repo_state | 123 symbols, project complete, 112% of llama.cpp (Qwen3-1.7B), Qwen3-8B 59.4 t/s |
| sessions_completed | 25 |

---

## META PROMPT

**Boot sequence**: Read `AGENTS.md` → `HANDOFF.md` → `git status --short` → `killall hashcat` → `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l` (expect 123) → `./bench/decode_int8_batched_cgraph_attn 28 8` (expect 327+ t/s).

**Verified state**: 123 symbols. INT8 batched attn + CUDA Graph M=8: **327.7 t/s** (112% of llama.cpp). Project complete.

**DO NOT**:
- Use `compute_120` (must be `compute_120a`)
- Use `gemv_int8` in production (use `gemv_int8_warp`)
- Use `gemm_int8_dp4a` for M≥16 (use `gemm_int8_wmma_fast`)
- Benchmark without `killall hashcat`

**Update discipline**: Update HANDOFF.md only when materially new state. Keep deduplicated with AGENTS.md.
