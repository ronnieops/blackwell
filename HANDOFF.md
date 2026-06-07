# HANDOFF.md — blackwell

Read `AGENTS.md` AND this file before acting.

---

## 1. Current Objective

Stabilize INT8 production path. Model dimensions were wrong in ALL pre-session-56 code (nqh=32 vs 16, nkv=4 vs 8, hd=64 vs 128, KV=512 vs 1024). With correct dims, INT8 block-16 gives PPL=18.65 (1.5× BF16).

FP8 path ABANDONED — INT8 wins on both quality (PPL 18.65 vs 41.75) and speed (4.5× faster).

---

## 2. Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| 1.7B HTTP server | ✅ Correct dims | Already used nqh=16/nkv=8/hd=128/KV=1024 |
| 1.7B bench_ppl | ✅ Fixed | Dims corrected this session, PPL=18.65 |
| 8B HTTP server | ⚠️ Unknown | Dims not verified — may be wrong |
| 9B GDN HTTP server | ⚠️ Dim suspicion | q_proj N=8192=32×256 vs server NQ=16. May use half of Q weights. |
| FP8 path | ⚠️ Abandoned | Reference code kept in src/ |
| Library | 171 symbols | +6 from FP8 kernel (kept as reference) |
| Disk | ~61% | 516G free |

### PPL Quality (1.7B, WikiText-2, 512 ctx)

| Config | PPL | vs BF16 |
|--------|-----|---------|
| BF16 (llama.cpp Q8_0) | **12.4** | 1.0× |
| INT8 block-16 (correct dims) | **18.65** | 1.5× |
| FP8 per-row (abandoned) | 41.75 | 3.4× |
| INT8 (old, wrong dims) | 7,351,868 | INVALID |

---

## 3. Recent Decisions

- **CRITICAL: Wrong model dimensions found** (Session 56). All code used nqh=32, nkv=4, hd=64, KV=512. Actual Qwen3-1.7B: nqh=16, nkv=8, hd=128, KV=1024.
- **No INT8 quality wall**. The PPL=7.3M was entirely a config bug. INT8 block-16 gives PPL=18.65 with correct dims.
- **FP8 path ABANDONED**. FP8 per-row is 4.5× slower (no dp4a) AND 2.3× worse quality than INT8. FP8 advantage only exists for tensor-core mixed-precision (FP8×FP8→FP32 accumulate), not weight-only dequant.
- **Activation quantization is fine** with correct dims. INT8 weight+act gives PPL=18.65 (same as weight-only).
- **FP8 GEMV kernel written** as reference (src/kernels/gemv_fp8.cu) but not used in production.
- **AGENTS.md updated** to reflect corrected dims and strategy.

---

## 4. Important Constraints

- **CORRECT 1.7B DIMS: nqh=16, nkv=8, hd=128, KV=1024** (NOT 32/4/64/512)
- `compute_120a` required (NOT `compute_120`)
- `killall hashcat` before every measurement
- `gemv_int8_warp` is production INT8 GEMV
- 1.7B intermediate_size=6144
- All pre-session-56 quality numbers are INVALID (wrong dims)
- head_dim=128 means hn_kernel needs 128 threads, RoPE needs 64 threads

---

## 5. Known Issues / Risks

| Issue | Severity | Notes |
|-------|----------|-------|
| 9B q_proj dim suspicion | HIGH | q_proj weight N=8192=32 heads×256 dim. Server NQ=16. May use half of Q weights. Needs config.json verification. |
| Pre-session-56 benchmarks invalid | HIGH | Any quality number from before this session used wrong dims |
| hashcat interference | ⚠️ | Always kill before measurement |

---

## 6. Pending Tasks

| Task | Priority | Notes |
|------|----------|-------|
| Verify 9B full_attention NQ value | HIGH | q_proj N=8192 suggests 32 heads vs server NQ=16. Find config.json or re-download model config. |
| Re-run server benchmarks with correct dims | MEDIUM | 1.7B already correct. 8B bench files also correct. Just need fresh measurements. |
| Clean up stale FP8 weights | LOW | ~2GB, can delete |
| Remove FP8 bench files from repo | LOW | Or mark as reference |

---

## 7. Suggested Next Actions

1. **Verify 8B and 9B dims**: Check weight file headers against config.json for both models
2. **Re-run server benchmarks**: Get corrected throughput numbers for 1.7B with proper dims
3. **Consider improving INT8 quality further**: PPL 18.65 vs BF16 12.4. Options: per-channel scaling, absmax-per-row activations
4. **Tag v0.8.0**: First release with correct model dimensions

---

## 8. Important Files / Commands

### Files created/modified this session
```
scripts/quantize_fp8.py            — FP8 quantizer (reference)
bench/bench_ppl_fp8.cu             — FP8 PPL benchmark (reference)
bench/bench_ppl_int8_fp32act.cu    — INT8 weight-only PPL benchmark
src/kernels/gemv_fp8.cu            — FP8 GEMV kernel (reference, not production)
include/blackwell/kernels.h        — Added FP8 kernel declarations
CMakeLists.txt                     — Added gemv_fp8.cu
weights_fp8_bf16/                  — FP8 weights (reference, 1.9 GB)
AGENTS.md                          — Updated with correct dims and strategy
docs/FP8_KERNEL_PLAN.md            — Marked OBSOLETE
```

### Build
```bash
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### Run
```bash
killall hashcat 2>/dev/null
./bench/bench_ppl 1.7b 20          # Corrected: PPL~18.65 (was 7.3M)
./server/http_subprocess 1.7b &    # Quality: semi-coherent text
```

---

## 9. Validation Status

| Check | Value |
|-------|-------|
| INT8 PPL (correct dims) | **18.65** ✅ (1.5× BF16) |
| BF16 reference PPL | **12.4** ✅ (from llama.cpp Q8_0) |
| 1.7B dims verified | nqh=16, nkv=8, hd=128, KV=1024 ✅ (config.json + weights match) |
| 8B dims verified | nqh=32, nkv=8, hd=128, KV=1024 ✅ (config.json + weights match) |
| 9B dims | Suspicious: q_proj N=8192 (32 heads) vs server NQ=16 ❓ |
| Server dims (1.7B) | Correct per-layer qk_norms loading ✅ |
| Library symbols | 171 ✅ |
| All active bench files | Correct dims verified ✅ |
| FP8 GEMV speed | 4.5× slower than INT8 (verified) ❌ |

---

## 10. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-06-07 |
| branch | master |
| repo_state | Modified (new files + edits, uncommitted) |
| session | 56 (FP8 Phase 1 + dimension bug discovery + strategy pivot) |
| key_finding | 9B q_proj dim suspicion (N=8192 vs server NQ=16). 1.7B and 8B dims all correct. All bench files verified. FP8 path abandoned. |
| next_priority | Verify 9B full_attention NQ (find config.json or re-download), re-run server benchmarks |

---

## META PROMPT

**Boot sequence**:
1. Read `AGENTS.md` → `HANDOFF.md`
2. `git status` — check state
3. `killall hashcat 2>/dev/null`

**Verified facts**:
- Qwen3-1.7B: nqh=**16**, nkv=**8**, hd=**128**, KV=**1024** (NOT 32/4/64/512)
- INT8 block-16 PPL = **18.65** (1.5× BF16 baseline 12.4)
- No INT8 quality wall — the 7.3M PPL was a dims bug
- FP8 path ABANDONED — INT8 wins quality AND speed
- Server already had correct dims for 1.7B; bench_ppl.cu was wrong

**DO NOT**:
- Use old dimension values (nqh=32, nkv=4, hd=64, KV=512)
- Trust any pre-session-56 PPL/quality number
- Invest more in FP8 weight-only kernels (no benefit over INT8)
- Re-dig dead ends: FP4 GEMM, speculative decode, sub-8-bit quantization
- Assume head_dim=64 — it's 128 for Qwen3-1.7B
- Delete FP8 reference code — keep for future reference
