# M=1 INT4 Gap Closure Plan — Final Handoff

**Goal:** 64 t/s → 84+ t/s (target), ceiling 95 t/s. Need −3.72ms/token.

---

## 1. Bottleneck Breakdown (nsys-verified, M=1 warp kernel, 15 gen tokens)

Source: `/tmp/nsys_int4_8b_warp.nsys-rep` — profiled this session.

### Time budget: 15.60ms/token (64 t/s)

| Component | ms/token | % | Recoverable ms | Source |
|-----------|----------|---|----------------|--------|
| **GEMV kernels** (253 calls) | **13.651** | **87.5%** | **2.28** (FP16 scales) | nsys: 204.8ms / 15 tokens |
| └ lm_head (1 call, N=151936) | 1.098 | 7.0% | 0.18 (FP16 scales) | nsys: max=1098µs |
| └ q/o_proj (72 calls, N=4096) | ~2.4 | 15.4% | 0.40 (FP16 scales) | Computed from median 31µs |
| └ gate/up/down (108 calls, N=12288) | ~10.1 | 64.7% | 1.70 (FP16 scales) | Computed from median 31-54µs |
| └ k/v_proj (72 calls, N=1024) | ~0.7 | 4.5% | 0.12 (FP16 scales) | Computed from min 9.7µs |
| **Non-GEMV kernels** | **1.130** | **7.2%** | **~0.15** (fusion) | nsys sum |
| └ rmsnorm (1091 calls) | 0.555 | 3.6% | 0.05 | nsys |
| └ quantize_int4 (2171 calls, 1-thread!) | 0.194 | 1.2% | 0.05 | nsys |
| └ attn_batched (540) | 0.101 | 0.6% | — | nsys |
| └ head_norm (1080) | 0.081 | 0.5% | — | nsys |
| └ vector_add (1080) | 0.061 | 0.4% | — | nsys |
| └ apply_rope (1080) | 0.055 | 0.4% | — | nsys |
| └ swiglu (540) | 0.039 | 0.3% | — | nsys |
| └ update_kv (540) | 0.028 | 0.2% | — | nsys |
| **CPU/launch gap** | **0.819** | **5.3%** | **0.35** (CUDA Graph) | Wall − GPU total |
| **TOTAL** | **15.60** | **100%** | | |

### Recoverable gap ranked by t/s impact

| Rank | Fix | Recoverable ms | → t/s | Confidence | Evidence |
|------|-----|---------------|-------|------------|----------|
| **1** | **FP16 weight scales** | **2.28** | **64→80** | HIGH | Scale=33.3% of traffic (verified: gate_proj `0_mlp.gate_proj.int4_t`=25.17MB, `.scale_t`=12.58MB). Save 0.946GB/token. At 416 GB/s achieved = 2.28ms |
| **2** | **Increase GEMV occupancy** | **0.5-1.0** | **80→84-88** | MEDIUM | Currently 8 warps/SM (17% occupancy), 33 regs/thread confirmed via `cuobjdump`. Register-limited max=62 warps/SM. `__launch_bounds__(32,8)` is artificial cap |
| **3** | **CUDA Graph + kernel fusion** | **0.35-0.50** | **88→90-92** | MEDIUM | Graph already tested: 2.2% (0.35ms). Fused kernels exist but unused. `quantize_int4` launched with **1 thread/block** (gemv_int8.cu:1797,1817) |
| — | uint4 (16B) loads | ~0 | — | — | Already LDG.128 via uint2 per-thread. Memory-bound at 93% BW. No gain |
| — | cp.async double-buffer | ~0 | — | — | Compute too light (4 dp4a/block ≈ 4 cycles). Nothing to overlap |
| — | L2 residency hints | ~0 | — | — | M=1 has zero weight reuse. Activation x=4KB already L1-cached |

### Why "30 t/s gap" is really "15-20 t/s recoverable"

- Task states ceiling=94 t/s (500 GB/s), but **actual peak BW=448 GB/s** (GDDR7 14001MHz × 128-bit × 2 PAM4, `nvidia-smi` confirmed)
- At 448 peak + FP32 scales: ceiling = **79 t/s**, not 94
- At 448 peak + FP16 scales: ceiling = **95 t/s**
- **FP16 scales is the single largest optimization** — it expands the ceiling from 79 to 95 t/s by removing 33% of traffic

---

## 2. What llama.cpp Does Differently

Source: `handoff/m1-external.md` (researcher findings)

| Technique | llama.cpp | Blackwell | Transferable? |
|-----------|-----------|-----------|---------------|
| **Multi-warp per block** | 4-8 warps/block (mmvq.cu), 16-32 warps/SM | 1 warp/block, 8 warps/SM | **YES — biggest architectural gap** |
| **K-split within block** | Warps cooperatively split K, reduce via smem | K-split across threads in 1 warp | **YES — but needs smem reduction** |
| **CUDA Graph** | Full decode loop captured (NVIDIA PR #6763) | Tested 2.2% on 8B | Already in place, limited gain |
| **4.5-bit quantization** | Q4_K_M = 4.5 bits/elem | Pure INT4 = 4 bits/elem | N/A — we read LESS data per token |
| **uint4 (16B) loads** | Yes | Yes (uint2 per thread = LDG.128) | **Already equivalent** |
| **dp4a/vecdotq** | Manual for Q4_K, dp4a-equivalent | dp4a | **Already equivalent** |

**Key insight from researcher:** llama.cpp reads **17% more weight data** (4.5-bit vs 4-bit) yet achieves **31% higher throughput** (84 vs 64 t/s). The multi-warp-per-block architecture is the differentiator — more outstanding DRAM requests per SM.

---

## 3. Recommended Approach

### **PRIMARY: FP16 weight scales** (2.28ms savings, 64→80 t/s)

**Why:** Scales are 33.3% of total GEMV memory traffic (verified from weight file sizes). Converting FP32→FP16 cuts 0.946 GB/token — equivalent to eliminating all k/v/q/o_proj GEMV time. At achieved 416 GB/s, this is 2.28ms/token = +16 t/s.

**Implementation:**
1. Re-quantize weights with FP16 scales: `scripts/quantize_awq_int4_8b.py` — modify scale dtype from `np.float32` to `np.float16`, write `.scale_t` files as FP16
2. Kernel change: `src/kernels/gemv_int8.cu:535` — replace `float w_sc = W_scale[...]` with `__half w_sc = ((__half*)W_scale)[...]`; cast to float before multiply
3. Activation scales (`x_scale`) stay FP32 (tiny, K/16 elements, reused across all N blocks)
4. Only weight scales (`W_scale`) convert to FP16

**Precision analysis:** Block-16 absmax scales have range ~0.001-0.5. FP16 has 11-bit mantissa = ~0.05% relative error per scale. Over 36 layers × 252 GEMV calls, accumulated error should be <0.5%. **Must re-verify PPL.**

### **SECONDARY: Increase GEMV occupancy** (0.5-1.0ms, 80→84-88 t/s)

**Why:** 8 warps/SM (17% occupancy) is artificially capped by `__launch_bounds__(32, 8)`. With 33 regs/thread (confirmed), register-limited max is 62 warps/SM. Going to 32 warps/SM (67% occupancy) increases outstanding DRAM requests 4×.

**But:** Naive `__launch_bounds__(32, 32)` hurts small matrices. k/v_proj (N=1024) at 32 blocks/SM = 0.89 waves — under-subscribed, SMs idle.

**Correct approach — multi-row-per-warp (2 rows/warp):**
- Each warp computes 2 output rows. Grid = N/2 blocks.
- Activation vector x loaded once, reused across 2 rows (doubles x reuse).
- Weight rows are independent (no reuse possible for M=1).
- Effective: same 32 warps/SM but N/2 blocks → more waves for small matrices.
- Requires shared memory reduction across 2 row-accumulators or separate warp-shuffle per row.

**Alternative — per-matrix-size launch config:**
- Large N (gate/up/down/lm_head): `__launch_bounds__(32, 32)` — many waves, benefit from high occupancy
- Small N (k/v_proj): `__launch_bounds__(32, 8)` — keep current, avoid under-subscription
- Dispatch based on N at call site.

### **TERTIARY: CUDA Graph + kernel fusion** (0.35-0.50ms, 88→90-92 t/s)

**Why:** Already tested at 2.2% (0.35ms). Fused kernels (`fused_rmsnorm_quant_int4`, `fused_swiglu_quant_int4`) exist in `src/kernels/fused_int4_ops.cu` but are **not used in any decode loop**. `quantize_int4_kernel` launched with **1 thread/block** (gemv_int8.cu:1797) — terrible occupancy, 145 launches/token.

**Implementation:**
1. Replace `fused_rmsnorm` + `quantize_int4` with `fused_rmsnorm_quant_int4` (2→1 kernel)
2. Replace `apply_swiglu` + `quantize_int4` with `fused_swiglu_quant_int4` (2→1 kernel)
3. Capture full 36-layer decode in CUDA Graph (graph-safe APIs already exist: `update_kv_cache_device`, `attention_decode_batched_gqa_device`, `fused_rope_decode`)

**Note:** `fused_residual_norm_int4` (header `kernels.h:1231`) is **declared but not implemented** — only stub in header, no `.cu` body. Would need implementation if used.

---

## 4. Files to Change

| File | Lines | Change | Phase |
|------|-------|--------|-------|
| `scripts/quantize_awq_int4_8b.py` | scale write | FP32→FP16 scale dtype | Phase 1 |
| `src/kernels/gemv_int8.cu` | 535 (warp), 1058 (batched) | FP16 scale load: `__half w_sc = ((__half*)W_scale)[idx]` | Phase 1 |
| `src/kernels/gemv_int8.cu` | 509, 1028 | Template on scale dtype or new kernel variant | Phase 1 |
| `include/blackwell/kernels.h` | gemv_int4_warp/batched signatures | Add FP16-scale variants or template parameter | Phase 1 |
| `bench/text_generate_int4_qwen3_8b.cu` | 290-330 (decode loop) | FP16 scale weights, fused kernels, remove debug D2H syncs (lines 318-320) | Phase 1+3 |
| `server/inference_server_int4.cu` | 271-300, 519-521 (decode loop) | Same changes as bench | Phase 1+3 |
| `src/kernels/gemv_int8.cu` | 509-556 (warp kernel) | Multi-row-per-warp or occupancy tuning | Phase 2 |
| `src/kernels/gemv_int8.cu` | 1028-1110 (batched kernel) | Same | Phase 2 |
| `bench/decode_int4_cgraph_8b.cu` | full file | Re-capture graph with fused kernels | Phase 3 |
| Weight files | `weights_int4_qwen3_8b/*.scale_t` | Regenerate with FP16 scales | Phase 1 |

### Weight regeneration

```bash
# Current: gate_proj packed=25.17MB, scale=12.58MB (FP32)
# After:   gate_proj packed=25.17MB, scale=6.29MB (FP16)
# Total weight dir: 5.8GB → 4.9GB
python3 scripts/quantize_awq_int4_8b.py --output weights_int4_qwen3_8b_fp16sc/ --scale-dtype fp16
```

---

## 5. Constraints

1. **PPL must hold at ≤23.52** (current INT4 baseline). FP16 scales will change PPL — must re-run `bench/bench_ppl_int4_8b.cu` after regeneration. If PPL degrades >5%, revert to FP32 scales.
2. **No re-quantization of packed weights** — only scale arrays change format. Packed INT4 nibbles stay identical. This is a scale-only format change.
3. **dp4a change already in place** — kernel uses `__dp4a` (gemv_int8.cu:541-544). Bit-identical to scalar path. Do NOT touch.
4. **Both paths must keep working:** `bench/text_generate_int4_qwen3_8b.cu` (warp), `server/inference_server_int4.cu` (batched M=1). Server uses `gemv_int4_batched` M=1.
5. **CUDA Graph already tested at 2.2%** — do NOT expect more than 0.5ms from it alone. Only worthwhile combined with kernel fusion.
6. **M>1 batched path must not regress** — `gemv_int4_batched_kernel<M>` shares the same occupancy characteristics. Any occupancy change affects M=1-16.
7. **`killall hashcat` before all measurements.**

---

## 6. Validation

### Phase 1 (FP16 scales) — MUST pass before proceeding

```bash
killall hashcat 2>/dev/null

# 1. PPL check — must be ≤24.7 (5% degradation from 23.52)
./bench/bench_ppl_int4_8b
# Expected: PPL ≤ 24.7. If >25, FP16 scales precision insufficient.

# 2. Correctness — text must be coherent
./bench/text_generate_int4_qwen3_8b "The capital of France is" 30
# Expected: "Paris..." (coherent). Compare token IDs with FP32-scale baseline.

# 3. Throughput
nsys profile --trace=cuda --stats=true \
  -o /tmp/m1_fp16sc --force-overwrite=true \
  ./bench/text_generate_int4_qwen3_8b "The capital of France is" 20
# Expected: 75-80 t/s (from nsys t/s or wall clock)

# 4. BW verification
# gemv_int4_warp max (lm_head) should drop from 1098µs to ~920µs (16.7% less data)
```

### Phase 2 (occupancy) — microbench + end-to-end

```bash
killall hashcat 2>/dev/null

# 1. Register count check (must stay ≤64 to allow 32 warps/SM)
/usr/local/cuda-13.3/bin/cuobjdump --dump-resource-usage build/libblackwell_kernels.a | grep "gemv_int4"
# Expected: REG ≤ 48 (currently 33)

# 2. BW improvement for large matrices
nsys profile --trace=cuda --stats=true ./bench/text_generate_int4_qwen3_8b "test" 20
# Compare gemv_int4 median (currently 31µs for N=4096) — expect 5-15% reduction for gate/up/down

# 3. Small matrix check — k/v_proj must NOT regress
# Min gemv_int4 time (currently 9.7µs for N=1024) — must stay ≤10µs
```

### Phase 3 (graph + fusion)

```bash
killall hashcat 2>/dev/null

# Graph benchmark
./bench/decode_int4_cgraph_8b 20
# Compare per-kernel vs graph — expect 3-5% speedup (up from 2.2%)

# ncu memory throughput (if ncu works — prior sessions had version mismatch)
/usr/local/cuda-13.3/bin/ncu --target-processes all \
  --kernel-name "gemv_int4" --launch-count 3 \
  --section MemoryWorkloadAnalysis \
  ./bench/text_generate_int4_qwen3_8b "test" 3
```

### Server validation

```bash
killall hashcat 2>/dev/null
./server/http_subprocess int4_8b &
curl -s -X POST http://localhost:8123/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","max_tokens":10}'
# Expected: coherent text, ~80 t/s
```

---

## 7. Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **FP16 scale precision loss** | MEDIUM | PPL +0.5-2.0 | Re-verify PPL. If PPL>25, use BF16 scales (same size, better range) or keep FP32 for small-magnitude scales |
| **Occupancy increase hurts small matrices** | HIGH (k/v_proj) | −0.2ms regression | Use per-matrix-size dispatch or multi-row-per-warp (preserves wave count) |
| **Register spill from multi-row-per-warp** | MEDIUM | Occupancy drops back | 33 regs × 2 rows = ~50 regs/thread. Still fits 32 warps/SM (50×32×32=51200 < 65536) |
| **CUDA Graph capture complexity** | LOW | Already tested | Graph-safe APIs exist. Main risk: fusing kernels changes graph topology |
| **Weight regeneration time** | LOW | 5-10 min | One-time cost. Script exists. |
| **Server/bench divergence** | MEDIUM | Different paths | Both use same `gemv_int4` kernel. Scale format change is kernel-level, affects both equally |

### FP16 scale precision — detailed analysis

Block-16 absmax scales for INT4 have typical magnitude 0.01-0.5. FP16 range: ±65504, precision: 11 mantissa bits (~3 decimal digits). For scale=0.01: FP16 represents it as 0.00999451 — relative error 0.05%. For scale=0.5: exact. Over 36 layers, accumulated error ≈ √36 × 0.05% ≈ 0.3%. **Expected PPL change: +0.1-0.5.** Should be safe.

**Fallback if FP16 fails:** BF16 scales (2 bytes, 8-bit mantissa, better for large dynamic range). Same traffic savings, slightly better precision for extreme values.

---

## 8. Unresolved Questions (need user decision)

1. **Is weight regeneration in scope?** FP16 scales require re-running quantization script on original FP32/BF16 weights. Output: new `weights_int4_qwen3_8b_fp16sc/` directory (~4.9 GB). This is the highest-impact fix (+16 t/s) but requires disk space and ~10 min compute.

2. **Target ceiling: 84 or 95 t/s?** At 448 GB/s peak + FP16 scales, theoretical max is 95 t/s. Reaching 84 requires FP16 scales (80 t/s) + partial occupancy gain. Reaching 95 requires FP16 scales + full occupancy + graph/fusion + hitting 448 GB/s peak (currently 416 = 93%).

3. **Should Phase 2 (occupancy) use multi-row-per-warp or per-matrix-size dispatch?** Multi-row-per-warp is simpler (one kernel change) but doubles register pressure. Per-matrix-size dispatch is more surgical but requires two kernel variants + call-site logic. **Recommend multi-row-per-warp first** (simpler, validate, then optimize if needed).

4. **BF16 scales as alternative to FP16?** Same traffic savings (2 bytes vs 4), better precision for large dynamic range. Slightly more complex kernel (BF16→FP32 conversion via `__bfloat162float`). Worth trying if FP16 PPL degrades.

5. **Should `fused_residual_norm_int4` be implemented?** Header declares it (`kernels.h:1231`) but no implementation exists. It would fuse residual_add + rmsnorm + quantize (3→1 kernel), saving ~0.15ms/token. Medium implementation effort.

---

## 9. Implementation-Ready Meta-Prompt (≤200 words)

### Goal
Close M=1 INT4 gap: 64→84+ t/s. Three phases, strict order.

### Phase 1 — FP16 weight scales (PRIMARY, +16 t/s)
1. Modify `scripts/quantize_awq_int4_8b.py`: write `.scale_t` as `np.float16` instead of `np.float32`. Output to `weights_int4_qwen3_8b_fp16sc/`.
2. In `src/kernels/gemv_int8.cu` lines 535 and 1058: change `float w_sc = W_scale[...]` to `float w_sc = __half2float(((__half*)W_scale)[...])`. Add scale-dtype template or new kernel variant.
3. Update `bench/text_generate_int4_qwen3_8b.cu` and `server/inference_server_int4.cu` to load FP16 scale files.
4. Remove debug D2H syncs at bench lines 318-320.
5. **VALIDATE:** PPL ≤24.7 (`bench/bench_ppl_int4_8b`), coherent text, nsys shows 75-80 t/s.

### Phase 2 — Occupancy (+4-8 t/s)
Change `__launch_bounds__(32, 8)` to multi-row-per-warp (2 rows/warp, 64 threads/block). Validate register count ≤48. Verify k/v_proj (N=1024) doesn't regress.

### Phase 3 — Graph + fusion (+2-4 t/s)
Replace `fused_rmsnorm`+`quantize_int4` with `fused_rmsnorm_quant_int4`. Replace `apply_swiglu`+`quantize_int4` with `fused_swiglu_quant_int4`. Re-capture CUDA Graph.

### Hard constraints
- PPL ≤24.7 (was 23.52). dp4a unchanged. Both bench+server paths work. `killall hashcat` before all measurements.
- Do NOT implement `fused_residual_norm_int4` (no body exists — out of scope).
- Verify with `cuobjdump --dump-resource-usage` after any kernel change.

### Stop conditions
- Phase 1 fails PPL → try BF16 scales → if still fails, STOP and report.
- Phase 2 regresses k/v_proj → revert, report, skip to Phase 3.
- Total ≥84 t/s achieved → STOP. Report results.
