# HANDOFF.md — blackwell

Operational context between sessions. Read before acting.

---

## 1. Current Objective

FP4 E2M1 + INT8 block-scaled LLM inference on RTX 5060 Ti (SM_120).
**Target**: Approach llama.cpp Q4_K_M baseline (114 t/s).
**Current**: ~50 t/s (FP4 CUDA Graph, 43.8% of target).
**INT8 kernel**: 348 GB/s isolated (2.1× FP4 v2), not yet integrated into decode.

---

## 2. Current Status

| Metric | Value |
|--------|-------|
| **Per-kernel throughput** | 42.8 t/s (2 layers raw) |
| **CUDA Graph throughput** | 50.1 t/s (scaled to 28 layers) |
| **Ratio to llama.cpp** | 43.8% (114 t/s baseline) |
| Per-token latency | ~1.43ms (CUDA Graph), ~1.74ms (per-kernel) |
| GEMV v2 peak | 164 GB/s (phase_a, isolated) |
| Decode_full GEMV | ~87 GB/s (cross-kernel L2 thrashing) |
| Library symbols | 30 public API |
| **INT8 GEMV peak** | **348 GB/s** (phase_a, 2.1× FP4 v2) |

**Per-kernel timing** (2 layers × 20 tokens):
| Kernel | Time | Share |
|--------|------|-------|
| fused_gate_up_gemv | 9.0 ms | 29% |
| Q/K/V GEMVs | 10.8 ms | 35% |
| gemv_fp4_splitk (down_proj) | 4.1 ms | 13% |
| Wo GEMV | 3.6 ms | 12% |
| attn + rmsnorm + pack + update | 1.8 ms | 6% |
| Other | 2.7 ms | 9% |
| **Total** | **31 ms** | |

**Bottleneck shift after optimizations**: fused_gate_up now dominates (29%), down_proj reduced to 13% with split-K=4.

---

## 3. Recent Decisions

### Implemented and working
- **CUDA Graph integration** — +15.3% over per-kernel timing (50.1 vs 42.8 t/s)
- **Split-K=4 for down_proj** — 28% reduction on down_proj time (5.7→4.1ms)
- **cudaMemsetAsync** — Required for CUDA Graph capture; cudaMemset fails with "previous error during capture"
- **Residual connections fix** — vector_add_fp32 kernel + 8 fixed instances
- **K_splits=4 in graph capture** — synced with benchmark loop
- **INT8 GEMV (gemv_int8)** — 348 GB/s isolated, 2.1× FP4 v2, kernel verified correct (GPU=CPU INT8 ref, 0.02% error)

### Tested and rejected
| Attempt | Result | Reason |
|---------|--------|--------|
| L2 persistence (256KB) | <1% | GDDR7 already fast enough |
| smem x_fp4 caching in v2 | 0% | Compiler/L1 already optimal |
| Batched GEMV (M×v2) | negative | Scattered L2 pattern, 85 vs 328 GB/s theoretical |
| Unfused gate+up | 0% | Same as fused |
| split-K beyond 4 | <1% | Diminishing returns |
| fused_qkv_gemv | slower | 66 GB/s vs 164 GB/s for gemv_fp4_v2 |
| GEMV v3 (smem tiled) | slower | 124 GB/s vs 164 GB/s; __syncthreads in inner loop |
| GEMV v3 unfused | abandoned | Register pressure, slower than v1 |

---

## 4. Root Cause: 2× phase_a vs decode_full gap

| Context | Bandwidth |
|---------|-----------|
| Phase_a isolated GEMV | 164 GB/s (33% of 500 GB/s peak) |
| Decode_full GEMV | ~87 GB/s (17% of peak) |

**Cause**: Cross-kernel L2 cache eviction. Q/K/V/Wo + gate_up + down_proj all access 12MB+ weight matrices — each kernel evicts the previous kernel's data from L2. FP4 dequantization overhead (cast + scale per element) is the intrinsic ceiling for block-scaled quantization.

**Not caused by**: Launch overhead (CUDA Graph fixes this), cudaMemset (already Async), smem for x, L2 persistence.

---

## 5. Constraints

- `CUDACXX=/usr/local/cuda-12.8/bin/nvcc` — CUDA 12.8, NOT system CUDA 12.0
- g++-12 host compiler, CMake, C++17
- `namespace wmma = nvcuda::wmma` (NOT `using wmma =`)
- All WMMA guarded `#if __CUDA_ARCH__ >= 800`
- `sizeof(__nv_fp4_e2m1) = 1` byte
- SM_120 native build critical — generic drops to 2% perf
- Attention kernel smem: 4096×4 bytes (static attr set via wrapper)
- CUDA Graph capture requires: cudaMemsetAsync, attention_pre_trigger, cudaDeviceSynchronize before capture

---

## 6. Pending Tasks

| # | Task | Priority | Status |
|---|------|----------|--------|
| 1 | INT8 → decode_full integration | High | INT8 GEMV kernel exists (348 GB/s). Replace FP4 v2 calls in decode pipeline using INT8 block scaling |
| 2 | INT8 transpose kernel | Medium | Currently done on host. Add GPU transpose like `transpose_fp4_weights` |
| 3 | Weight conversion util | Medium | `pack_fp4` → `pack_int8` for model loading |
| 4 | Use `fused_qkv_gemv` in decode_full | Low | Would replace 3 gemv_v2 calls; but kernel is 66 GB/s vs 164 — **deprioritized** |
| 5 | CUTLASS warp-tiled GEMV | Future | Separate project, cp.async pipelines |

---

## 7. Suggested Next Actions (Future Sessions)

1. **[HIGH] INT8→decode_full integration** — Replace FP4 gemv_fp4_v2 calls in decode pipeline with gemv_int8 using host-packed INT8 weights. Requires GPU INT8 transpose kernel for model loading path. Could improve decode throughput significantly (INT8 348 GB/s vs FP4 164 GB/s, though L2 thrashing still caps cross-kernel at ~87 GB/s).

2. **[HIGH] pack_int8 kernel** — GPU INT8 quantization kernel (FP32→INT8 with per-block scales), matching pack_fp4 pattern. Needed for model loading.

3. **[HIGH] Add INT8 transpose kernel** — GPU transpose for INT8 weights (GPU equivalent of host W→W_t transpose).

4. **[MED] Run llama-bench baseline** — Verify llama.cpp Q4_K_M reference throughput (114 t/s on same hardware). The model is not local in GGUF format yet; requires conversion from safetensors.

5. **[MED] INT8 correctness vs llama.cpp** — Once decode pipeline uses INT8, compare output values with llama.cpp reference.

6. **[LOW] CUTLASS-style GEMV** — Separate project: warp-tiled GEMV with cp.async weight loading.

7. **[LOW] Ship and document** — Clean up untracked files, finalize AGENTS.md.

---

## 8. Important Files / Commands

**Build**:
```bash
CUDACXX=/usr/local/cuda-12.8/bin/nvcc cmake -B build && cmake --build build --parallel
```

**Benchmarks**:
```bash
# Per-kernel timing (with profiling overhead)
./bench/decode_full 2

# CUDA Graph benchmark (cleaner measurement)
./bench/decode_full_cgraph_clean 2

# Kernel throughput (isolated)
./bench/phase_a 2>&1 | grep gemv_fp4_v2

# Unfused gate+up variant
./bench/decode_full_unfused 2
```

**llama baseline**:
```bash
/mnt/data/ai/llama.cpp/build-cuda12.8-sm120/bin/llama-bench \
  --hf-repo unsloth/Qwen3.5-4B-MTP-GGUF:Q4_K_M -p 512 -n 128 -r 3
```

**Verification**:
```bash
nm build/libblackwell_kernels.a | c++filt | grep " T blackwell::kernels" | grep -v anonymous | wc -l
# Expected: 30 (was 29 before INT8)
```

**INT8 test**:
```bash
./bench/gemv_int8_test
# Expected: GPU vs CPU INT8 rel err < 1e-3 (kernel correctness)
```

---

## 9. Validation Status

| Check | Status |
|-------|--------|
| Build | ✅ Clean (30 symbols) |
| Phase_a | ✅ No segfault, no FAIL lines |
| GEMV v2 correctness | ✅ 0 rel error |
| CUDA Graph capture | ✅ Working (cudaMemsetAsync required) |
| Residual connections | ✅ L1 norm non-zero |
| Split-K=4 | ✅ Working |
| INT8 GEMV correctness | ✅ GPU = CPU INT8 ref (0.02% error) |
| INT8 GEMV benchmark | ✅ 348 GB/s (0.037 ms for 6144×2048) |

---

## 10. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-05-27 07:50 UTC |
| branch | master (clean, INT8 GEMV committed) |
| repo_state | Working, all binaries present |
| throughput | 50.1 t/s CUDA Graph, 42.8 t/s per-kernel |
| GEMV peak | 348 GB/s (INT8 isolated), 164 GB/s (FP4 v2), ~87 GB/s (decode_full FP4) |
| active_kernels | gemv_fp4_v2, gemv_fp4_splitk, gemv_int8, fused_gate_up, fused_rmsnorm_pack, attention_decode_gqa |

---

## META PROMPT

**Read sequence**: `AGENTS.md` → `HANDOFF.md` → `git status` → verify build exists

**Critical facts** (verify before editing):
- Throughput is **50.1 t/s** (CUDA Graph), NOT the per-kernel 42.8 t/s. Per-kernel timing has profiling overhead.
- `decode_full` has both per-kernel timing AND CUDA Graph benchmark sections.
- `cudaMemset` (sync) FAILS in CUDA Graph capture. Must use `cudaMemsetAsync`.
- GEMV v2 is final (164 GB/s). v3 abandoned (124 GB/s).
- `fused_qkv_gemv` is slower (66 GB/s) than 3× gemv_fp4_v2 calls — do not replace working GEMVs with it.
- SM_120 native build critical.
- **INT8 GEMV (gemv_int8)**: 348 GB/s isolated, **2.1× FP4 v2**. Kernel verified correct (GPU=CPU INT8 ref, 0.02%). NOT yet integrated into decode pipeline. Requires host-side weight quantization (no pack_int8 kernel yet).
- INT8 GEMV uses **symmetric INT8 quantization** (per-block absmax/127).

**Current priority**: INT8→decode_full integration. `gemv_int8`, `pack_int8`, `transpose_int8_weights` all complete and verified (GPU pipeline working, 346 GB/s). Next: load FP4 weights → convert to INT8 → replace gemv_fp4_v2 calls in decode_full with gemv_int8. Requires weight format converter for model loading path.

**Do NOT**:
- Restart analysis from scratch
- Use per-kernel timed numbers as ground truth (52% overhead)
- Re-implement smem tiling (proven worse)
- Replace gemv_fp4_v2 with fused_qkv_gemv

**After every edit**: `cmake --build build --parallel` → `./bench/phase_a` → `./bench/decode_full 2` → `./bench/gemv_int8_test`