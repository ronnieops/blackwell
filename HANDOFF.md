# HANDOFF.md — blackwell

Continuity doc. Read with AGENTS.md before acting.

---

## 1. Current Objective

**text_generate INT4 inference path created** (`bench/text_generate_int4.cu`). Pipeline structurally correct (17 kernels/layer, correct residuals + head_norms + RoPE) but 4-bit symmetric quantization noise compounds across 28 layers — first token diverges from INT8 greedy. Needs asymmetric quantization or fine-tuning for quality. M=1 throughput: 262 t/s (89% Q4_K_M). M=8: 3586 t/s (1222% Q4_K_M).

---

## 2. Current Status

| Metric | Value |
|--------|-------|
| GPU | RTX 5060 Ti, compute 12.0, 36 SMs, ~500 GB/s GDDR7 |
| CUDA | 13.3, SM_120a, C++17, CMake |
| Library | **177 symbols** in `build/libblackwell_kernels.a` |
| Branch | master |
| Session | **37 (end)** |

### INT4 batched attention — CORRECTED (post-grid-bug-fix)

| Model | M | Per-seq t/s | vs Q4_K_M | Notes |
|-------|---|-------------|-----------|-------|
| Qwen3-1.7B | 1 | **261.7** | **89%** | Post-fix. Was 610 (208%) |
| Qwen3-1.7B | 8 | **3586.4** | **1222%** | Post-fix. Was 11293 (3848%) |
| Qwen3-8B | 1 | 342.9 (UNVERIFIED) | — | Pre-dates grid fix |
| Qwen3-8B | 8 | 5640.3 (UNVERIFIED) | — | Pre-dates grid fix |

### Bugs found (session 37)

| Bug | Severity | Fix |
|-----|----------|-----|
| `gemv_int4_batched` grid `N/32` → `N` | **CRITICAL** — only 1/32 rows computed | `dim3 grid(N, M)` |
| INT4 nibble sign-extend `if(lo>7)lo-=16` → `nib-8` | HIGH — wrong scale levels | Both `gemv_int8.cu` + `gemv_int4_batched.cu` |
| INT4 weight scales ~1e-23 | HIGH — re-quantize required | `read_tensor` bug, re-run script |
| Missing head_norm + RoPE in INT4 pipeline | MEDIUM — copied from bench | Added handwritten kernels |

---

## 3. Recent Decisions

- **gemv_int4_batched grid bug** — `dim3 grid(N/32, M)` was wrong. Now `grid(N, M)`. **ALL pre-session-37 INT4 benchmark numbers invalid**
- **Nibble offset-binary fix** — INT4 packs as `nib = q+8` (offset-binary). GEMV unpack must use `nib - 8`, not two's complement sign-extend
- **text_generate_int4 pipeline** — uses 17 kernels/layer (not 14 fused). Has correct residual buffers (`d_res`), head norms, RoPE
- **INT4 quality insufficient** — 4-bit symmetric ~14% per-value error compounds across 28 layers
- **INT4 weights must be re-quantized after batch runs** — `read_tensor` has `f.seek(0)` bug

---

## 4. Important Constraints

- `PATH` must include `/usr/local/cuda-13.3/bin` before nvcc
- `compute_120a` required (not `compute_120`)
- `killall hashcat` before every measurement (auto-restarts, -45% throughput)
- `gemv_int8_warp` is production INT8 GEMV — NOT `gemv_int8`
- CUDA Graph harmful on Blackwell (~10× slower)
- L2 persisting cache harmful for weights >8 MB — only d_rn (8 KB) safe
- M>8 not viable (batched GEMV register pressure)
- Speculative decode, FP4 tensor core GEMV, PDL — all dead ends
- INT4 weights corrupt after batch model loading; re-quantize before use

---

## 5. Known Issues / Risks

1. **hashcat** — `killall hashcat` 30s before benchmark. 60s respawn.
2. **INT4 text_generate quality** — garbled after 28 layers. Symmetric 4-bit ~14% error compounds. Needs asymmetric Q or fine-tuning.
3. **8B INT4 benchmarks UNVERIFIED** — pre-date grid fix. Need re-run.
4. **weights_int4_* corruption** — `read_tensor()` in `quantize_generic.py` has `f.seek(0)` offset bug. Re-run script if weights loaded fresh.
5. **text_generate repetition** — greedy decode repeats. Use `-t 0.8` or `-k 40`.

---

## 6. Pending Tasks

| Task | Status | Notes |
|------|--------|-------|
| text_generate INT4 inference | ✅ | `bench/text_generate_int4.cu` created, builds, runs. Quality needs improvement |
| 8B INT4 re-benchmark (post-grid-fix) | 🔜 | Current 342.9/5640.3 numbers invalid |
| INT4 quality fix (asymmetric Q) | 🔜 | 4-bit symmetric compounds error across 28 layers |
| Docker server INT4 support | ⏸ | Blocked on INT4 quality |
| Qwen3.6-27B INT4 decode | ⏸ | Depends on quality fix |
| Pipeline SNR on 8B (36 layers) | ⏸ | Low priority |

---

## 7. Suggested Next Actions

1. Verify INT4 vs INT8 at single-layer level (measure per-layer SNR)
2. Implement asymmetric per-block INT4 quantization (separate +ve/-ve scale per block)
3. Or use per-channel INT4 (one scale per output channel instead of per-block)
4. If quality fixed, re-bench 8B INT4 with corrected kernel
5. Extend text_generate_int4 to 8B

---

## 8. Important Files / Commands

### Build
```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
killall hashcat 2>/dev/null
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
```

### Key source
- `bench/text_generate_int4.cu` — **NEW** INT4 end-to-end generation (session 37)
- `bench/text_generate.cu` — INT8 reference (correct residuals, head norms, RoPE)
- `bench/decode_int4_batched_attn.cu` — INT4 benchmark (identity norms, no head_norm/RoPE)
- `src/kernels/gemv_int4_batched.cu` — **FIXED** grid bug + nibble sign bug
- `src/kernels/gemv_int8.cu` — **FIXED** INT4 nibble sign bug (lines 473-476)
- `scripts/quantize_generic.py` — INT4 weight generation (has f.seek(0) bug)

### Benchmark
```bash
./bench/decode_int4_batched_attn 28 1              # 1.7B: ~262 t/s (89% Q4_K_M)
./bench/decode_int4_batched_attn 28 8              # 1.7B M=8: ~3586 t/s (1222%)
./bench/text_generate_int4 "The capital of France is" 30  # INT4 generation (garbled)
./bench/text_generate "The capital of France is" 30       # INT8 reference
```

### INT4 weights (re-quantize if corrupt)
```bash
python3 scripts/quantize_generic.py /mnt/data/ai/hf/qwen3-1.7b-base weights_int4_qwen3_1.7b int4
```

### Verify
```bash
nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l  # expect 177
```

---

## 9. Validation

| Check | Status |
|-------|--------|
| Library symbols | ✅ 177 |
| INT4 1.7B M=1 (post-grid-fix) | ✅ 261.7 t/s (89% Q4_K_M) |
| INT4 1.7B M=8 (post-grid-fix) | ✅ 3586.4 t/s (1222% Q4_K_M) |
| INT4 single-layer GEMV accuracy | ✅ Avg err 0.11 vs INT8 (expected for 4-bit) |
| INT4 embedding quality | ✅ RMS diff 0.002 vs INT8 |
| LB: text_generate_int4 builds | ✅ |
| LB: hashcat killed | ✅ |

---

## 10. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-06-02 |
| branch | master |
| last_commit | `09b8217` + uncommitted changes |
| uncommitted | AGENTS.md, HANDOFF.md, `bench/text_generate_int4.cu`, `src/kernels/gemv_int4_batched.cu`, `src/kernels/gemv_int8.cu` |
| active components | `bench/text_generate_int4.cu` (created), `gemv_int4_batched.cu` (fixed), `gemv_int8.cu` (fixed) |

---

## META PROMPT

**Boot**: Read `AGENTS.md` → `HANDOFF.md` → `git log --oneline -3` → `killall hashcat` → `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l` (expect 177) → `./bench/decode_int4_batched_attn 28 1` (expect ~262 t/s) → `./bench/text_generate_int4 "hello" 5 -t 0` (expect garbled output — quality known issue).

**Verified**: 177 symbols. INT4 1.7B M=1: 261.7 t/s (89% Q4_K_M). M=8: 3586.4 t/s (1222%).

**CRITICAL BUGS (session 37)**:
1. `gemv_int4_batched` grid was `N/32` — only computed 1/32 output rows. Fixed to `dim3 grid(N, M)`. **ALL pre-fix INT4 benchmarks invalid**.
2. INT4 nibble unpack used wrong sign-extend (`if(lo>7)lo-=16`) instead of offset-binary decode (`nib - 8`). Fixed in both `gemv_int8.cu` and `gemv_int4_batched.cu`.
3. INT4 weight scales were ~1e-23 (corrupt). Re-quantize output fixed it.

**text_generate_int4 pipeline**: Structurally correct (17 kernels/layer, correct residuals, head_norms, RoPE). But INT4 symmetric quant noise ~14% per-value (~1.5% for INT8). Compounds across 28 layers. First token diverges from INT8 ground truth.

**DO NOT**:
- Use `compute_120` (must be `compute_120a`)
- Use `gemv_int8` (use `gemv_int8_warp`)
- Benchmark without `killall hashcat` (-45%)
- Use `decode_int4_batched.cu` or `decode_int4_cgraph.cu` (stale data / 2.4× slower)
- Expect M>8 scaling
- Rely on pre-session-37 INT4 benchmark numbers
- Re-dig dead ends: speculative decode, FP4 tensor core GEMV, PDL, CUDA Graph

**Active direction**: Fix INT4 quality (asymmetric quantization) or accept that 4-bit symmetric is insufficient for 28-layer deep model. Re-bench 8B with corrected kernel if quality fix works.

**Verify repo state before edits**: `git status`. Prefer incremental edits.

**Keep HANDOFF.md concise** — deduplicate with AGENTS.md, prefer bullets, remove stale sections on update.