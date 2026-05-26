# HANDOFF.md — blackwell

Preserves operational context between sessions. Read before acting.

---

## 1. Current Objective

FP4 E2M1 block-scaled LLM inference kernels on RTX 5060 Ti (SM_120).  
Full decode pipeline (attention + MLP) working with real Qwen3-1.7B weights.  
Next: close decode gap — MLP GEMVs are the bottleneck (87% of per-layer time).

---

## 2. Current Status

| Domain | Status | Note |
|--------|--------|------|
| GEMM (prefill) | ✅ 25–38 GB/s | 128×128×64 CTA, cp.async pipeline, 136 regs |
| GEMV (decode) | ✅ Dynamic K | Any K multiple of 16. 2048×2048 = 0.193ms |
| Fused QKV | ⚠️ Broken for >256 outputs | Grid(3,) kernel only has 256 threads → only writes first 256 of N outputs. Broken for kv_dim=1024. Use separate GEMVs instead. |
| Decode attention | ✅ 11µs at seq=128 | Warp-parallel QK dot, smem broadcast |
| update_kv_cache | ✅ | Verified correct |
| fused_rmsnorm/SwiGLU/RoPE | ✅ | All working |
| fused_rmsnorm_pack | ✅ NEW | Single-block kernel: RMSNorm + FP4 pack. Saves 1 kernel launch per layer. 256 threads × 8 elements. |
| pack_fp4/unpack_fp4 | ✅ | E2M1 verified. sizeof=1 byte |
| Attention-only decode | ✅ 86 t/s (28-layer est) | Real weights, 4 layers, attention only |
| Full decode (attn + MLP) | ✅ 10.9 t/s (28-layer est) | Real weights, 4 layers. MLP dominates (87% of time). |
| Decode full benchmark | ✅ NEW | `bench/decode_full` — attention + MLP with real weights |
| Stubs (attention_fp4, prefill, CUDA Graphs) | 🟡 | Return cudaErrorNotReady |
| llama.cpp baseline | ✅ | 4560 t/s pp512, 114 t/s tg128 |

**22 public API symbols** (was 20, added `fused_rmsnorm_pack` + `fused_o_norm_pack`). All outside anonymous namespace.

---

## 3. Recent Decisions

- **fused_rmsnorm_pack**: Single-block 256-thread kernel (8 elements/thread). Fuses RMSNorm + FP4 pack. 1.01x speedup (GEMV dominates). Correctness verified (0 error).
- **fused_o_norm_pack**: Convenience wrapper — calls gemv_fp4 then fused_rmsnorm_pack. 2 kernels instead of 3. Allocates internal temp buffer.
- **fused_qkv_gemv broken for real weights**: Grid(3,) × 256 threads only writes 256 of N outputs. Works for kv_dim≤256 (synthetic). Real Qwen3-1.7B has kv_dim=1024. **Must use separate GEMVs for now.**
- **Qwen3-1.7B real dimensions**: hidden=2048, q_dim=2048 (16 heads), kv_dim=1024 (8 KV heads), intermediate=6144, head_dim=128, 28 layers.
- **MLP is the real bottleneck**: Attention-only 86 t/s → Full (attn+MLP) 10.9 t/s. MLP GEMVs are 2.3× more work than attention GEMVs per layer.
- **GEMM tile 128×128×64**: 8 warps, cp.async 2-stage pipeline. 80 KB dynamic smem.
- **GEMV dynamic K**: 256 threads/block, walks K sequentially (coalesced reads). 37 regs, 0 spill.
- **CUDA Graphs**: 1% improvement. Launch overhead not the bottleneck.
- **SM_120 does NOT expose tcgen05.mma**: Only datacenter SM100. SM120 uses wmma/mma.sync only.

---

## 4. Important Constraints

- `CUDACXX=/usr/local/cuda-12.8/bin/nvcc` before `project()` in CMakeLists.txt
- g++-12 host compiler (CUDA 12.8 rejects GCC 13+)
- `namespace wmma = nvcuda::wmma` (NOT `using wmma =` — type alias fails)
- All WMMA code guarded `#if __CUDA_ARCH__ >= 800`
- Shared mem: 99 KB/block max. GEMM uses 80 KB dynamic.
- SM_120 native **critical**: generic PTX = 47× slower.
- CMake uses `CUDA::cudart` not `CUDA::CUDA` (CMake 3.28 compat).
- `fused_qkv_gemv` broken for output dims > 256 — don't use with real weights.

---

## 5. Known Issues / Risks

| Issue | Severity | Root Cause | Fix |
|-------|----------|------------|-----|
| Full decode 10.9 vs 114 t/s | 🔴 Major | MLP GEMVs dominate (87% of time) | Fuse gate+up GEMVs, optimize GEMV throughput, tensor-core GEMV |
| fused_qkv_gemv broken for kv_dim>256 | 🔴 Bug | Grid(3,1) × 256 threads → only 256 outputs | Use separate GEMVs or fix to multi-block |
| Attention-only 86 vs 114 t/s | 🟡 Under target | GEMV compute-bound | Same as MLP fix |
| GEMM correctness unverified | 🟡 Risk | No reference comparison | Add CPU GEMM reference to phase_a |
| attention_decode hardcodes head_dim=128 | 🟢 OK for Qwen3 | Register layout assumes 128 | Guard if supporting other models |
| decode_bench scaling math wrong | 🟢 Display bug | `tps * (28/N)` instead of `1000 / (per_token * 28/N)` | Fix display |

---

## 6. Pending Tasks

| Task | Priority |
|------|----------|
| Fix fused_qkv_gemv for multi-block (support >256 outputs) | High |
| Fuse gate + up GEMVs (same input x, 2 projections) | High |
| Nsight Compute profile on GEMV bottleneck | Medium |
| Verify GEMM output correctness | Medium |
| Investigate tensor-core GEMV feasibility | Low |

---

## 7. Suggested Next Actions

1. **Fix fused_qkv_gemv** — use grid(ceil(q_dim/256), ceil(kv_dim/256)) instead of grid(3,1). Or use cooperative approach.
2. **Fuse gate + up MLP GEMVs** — same input x, two projections → one kernel. Same pattern as fused_qkv but with correct multi-block design.
3. **Nsight Compute profile** — identify why GEMV is slow. Memory throughput? Compute? Register pressure?
4. **Optimize GEMV for large K** — gate/up are 2048×6144. Consider vectorized loads, K-tiling with smem.

---

## 8. Important Files / Commands

**Build**: `CUDACXX=/usr/local/cuda-12.8/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel`

**Benchmarks**:
- `bench/phase_a` — basic correctness + throughput
- `bench/decode_bench` — attention-only decode (synthetic weights, kv_dim=512)
- `bench/decode_real N` — attention-only decode with real Qwen3-1.7B weights (N layers)
- `bench/decode_full N` — **NEW** full decode (attention + MLP) with real weights
- `bench/test_fused_o_norm` — verify fused_rmsnorm_pack correctness + benchmark
- `tools/extract_weights --layers N` — extract FP4 weights from safetensors

**llama-baseline**: `/mnt/data/ai/llama.cpp/build-cuda12.8-sm120/bin/llama-bench --hf-repo unsloth/Qwen3.5-4B-MTP-GGUF:Q4_K_M -p 512 -n 128 -r 3`

**Key source**:
- `include/blackwell/kernels.h` — 22 public API signatures
- `include/blackwell/config.h` — 128×128 tile, 8 warps, constants
- `src/kernels/gemm.cu` — GEMM (cp.async pipeline) + GEMV (dynamic K)
- `src/kernels/decode.cu` — update_kv_cache + attention_decode v2
- `src/kernels/fused_decode.cu` — fused_qkv_gemv (⚠️ broken for >256 outputs)
- `src/kernels/fused_o_norm.cu` — **NEW** fused_rmsnorm_pack + fused_o_norm_pack
- `src/kernels/quantize.cu` — pack_fp4/unpack_fp4
- `src/kernels/norm.cu` — fused_rmsnorm, apply_swiglu

**Verify symbols**: `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell::kernels" | grep -v anonymous | wc -l` → 22

---

## 9. Validation Status

**Last validated**: 2026-05-26 (this session)  
**Build**: ✅ Clean  
**Run**: ✅ phase_a passes, decode_bench runs, decode_full runs with real weights  
**Public symbols**: 22, none in anonymous namespace  
**Real weights**: 4 layers extracted (attention + MLP), full decode verified  
**fused_rmsnorm_pack**: ✅ Correctness verified (0 error vs separate rmsnorm+pack)  

---

## 10. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-05-26 (this session) |
| branch | master |
| repo state | 11 commits, uncommitted changes (fused_o_norm, decode_full) |
| active components | GEMM, GEMV, decode (attention+MLP), fused rmsnorm+pack |
| nvcc | /usr/local/cuda-12.8/bin/nvcc |
| host compiler | g++-12 |
| target | SM_120 native (compute_120) |

---

## META PROMPT

**Read sequence:** `AGENTS.md` → `HANDOFF.md` (this file) → `git status` → verify build artifacts exist.

**Current priorities:**
- MLP GEMVs are the bottleneck (87% of decode time). gate/up projections (2048→6144) are 3× larger than attention projections.
- fused_qkv_gemv is broken for real weights (kv_dim=1024). Use separate GEMVs until fixed.
- Full decode benchmark at `bench/decode_full`. Attention-only at `bench/decode_real`.
- Do NOT restart analysis — continue incrementally.
- After every edit: build → symbols → bench run → no regressions.
- DO NOT re-derive decisions from HANDOFF.md — they're recorded for continuity.
- Update HANDOFF.md only for materially important state changes. Keep concise. No narrative history.
