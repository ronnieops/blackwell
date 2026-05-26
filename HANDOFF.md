# HANDOFF.md — blackwell

Preserves operational context between sessions. Read before acting.

---

## 1. Current Objective

FP4 E2M1 block-scaled LLM inference kernels on RTX 5060 Ti (SM_120).  
Baseline kernels working. Next: close end-to-end decode gap vs llama.cpp (86 vs 114 t/s).

---

## 2. Current Status

| Domain | Status | Note |
|--------|--------|------|
| GEMM (prefill) | ✅ 25–38 GB/s | 128×128×64 CTA, cp.async pipeline, 136 regs |
| GEMV (decode) | ✅ Dynamic K | Any K multiple of 16. 2048×2048 = 0.193ms |
| Fused QKV | ✅ 3× speedup | Single kernel: Q, K, V projections |
| Decode attention | ✅ 11µs at seq=128 | Warp-parallel QK dot, smem broadcast |
| update_kv_cache | ✅ | Verified correct |
| fused_rmsnorm/SwiGLU/RoPE | ✅ | All working |
| pack_fp4/unpack_fp4 | ✅ | E2M1 verified. sizeof=1 byte |
| Decode benchmark | ✅ 86 t/s (28-layer est) | Real Qwen3-1.7B weights (4 layers) |
| Stubs (attention_fp4, prefill, CUDA Graphs) | 🟡 | Return cudaErrorNotReady |
| llama.cpp baseline | ✅ | 4560 t/s pp512, 114 t/s tg128 |

**20 public API symbols**. All outside anonymous namespace.

---

## 3. Recent Decisions

- **GEMM tile 128×128×64**: 8 warps, cp.async 2-stage pipeline. 80 KB dynamic smem. Verified: cp.async.cg is optimal (no `cp.async.bulk` on RTX 5060 Ti).
- **GEMV dynamic K**: 256 threads/block, walks K sequentially (coalesced reads). 37 regs, 0 spill.
- **Fused QKV**: Grid(3,1) kernel. 3× speedup over separate GEMVs. Same x reused from L1.
- **Decode attention v2**: All 8 warps do parallel QK dot. V weight still sequential over seq (not bottleneck).
- **Warp specialization abandoned**: 255 reg + 132 B spill. Pipelining already milked.
- **CUDA Graphs**: 1% improvement. Launch overhead not the bottleneck.
- **Real weight quantizer**: BF16→float→FP4 with 16×16 block scaling. Output: binary files with header.
- **sizeof(__nv_fp4_e2m1)=1 byte** (N elements = N bytes).

---

## 4. Important Constraints

- `CUDACXX=/usr/local/cuda-12.8/bin/nvcc` before `project()` in CMakeLists.txt
- g++-12 host compiler (CUDA 12.8 rejects GCC 13+)
- `namespace wmma = nvcuda::wmma` (NOT `using wmma =` — type alias fails)
- All WMMA code guarded `#if __CUDA_ARCH__ >= 800`
- Shared mem: 99 KB/block max. GEMM uses 80 KB dynamic.
- SM_120 native **critical**: generic PTX = 47× slower.
- CMake uses `CUDA::cudart` not `CUDA::CUDA` (CMake 3.28 compat).

---

## 5. Known Issues / Risks

| Issue | Severity | Root Cause | Fix |
|-------|----------|------------|-----|
| Decode 86 vs 114 t/s | 🟡 Under target | GEMV compute-bound (77% of decode) | Fused O proj + rmsnorm + pack, or tensor-core GEMV |
| SwiGLU GB/s > peak | 🟡 Artifact | Timer too coarse (~3us) | Use larger batches |
| FP4 rel error=1 for small vals | 🟢 Expected | E2M1 rounds <0.5×scale to 0 | OK for LLM weights >0.1 |
| No real lm_head, embeddings | 🟢 Not blocking | Only attention layer weights extracted | Add if full inference needed |

---

## 6. Pending Tasks

| Task | Priority |
|------|----------|
| All Phase A/B/C tasks complete | ✅ |
| Next: fuse O GEMV + rmsnorm + pack into single kernel | Medium |
| Next: add SwiGLU + MLP to decode pipeline | Medium |
| Next: Nsight Compute profiling for GEMV | Low |

---

## 7. Suggested Next Actions

1. **Fuse O projection + RMSNorm + pack** — single kernel. Cuts from 3 launches to 1 per layer.
2. **Add SwiGLU + MLP** — gate_proj, up_proj, down_proj for full layer decode.
3. **Nsight Compute profile** — identify GEMV micro-bottleneck.

---

## 8. Important Files / Commands

**Build**: `CUDACXX=/usr/local/cuda-12.8/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel`

**Benchmarks**:
- `bench/phase_a` — basic correctness + throughput
- `bench/decode_bench` — end-to-end decode (synthetic weights)
- `bench/decode_real N` — decode with real Qwen3-1.7B weights (N layers)
- `bench/test_decode` — verify decode kernels
- `tools/extract_weights --layers N` — extract FP4 weights from safetensors

**llama-baseline**: `/mnt/data/ai/llama.cpp/build-cuda12.8-sm120/bin/llama-bench --hf-repo unsloth/Qwen3.5-4B-MTP-GGUF:Q4_K_M -p 512 -n 128 -r 3`

**Key source**:
- `include/blackwell/kernels.h` — 20 public API signatures
- `include/blackwell/config.h` — 128×128 tile, 8 warps, constants
- `src/kernels/gemm.cu` — GEMM (cp.async pipeline) + GEMV (dynamic K)
- `src/kernels/decode.cu` — update_kv_cache + attention_decode v2
- `src/kernels/fused_decode.cu` — fused_qkv_gemv (3× speedup)
- `src/kernels/quantize.cu` — pack_fp4/unpack_fp4

**Verify symbols**: `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell::kernels" | grep -v anonymous | wc -l` → 20

---

## 9. Validation Status

**Last validated**: 2026-05-26 16:45 UTC  
**Build**: ✅ Clean  
**Run**: ✅ phase_a passes, test_decode PASS, decode_bench runs clean  
**Public symbols**: 20, none in anonymous namespace  
**Real weights**: 4 layers extracted, decode matches synthetic perf  

---

## 10. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-05-26 16:45 UTC |
| branch | master |
| repo state | 11 commits, clean working tree |
| active components | GEMM, GEMV, decode (attention+KV), fused QKV, weight extraction |
| nvcc | /usr/local/cuda-12.8/bin/nvcc |
| host compiler | g++-12 |
| target | SM_120 native (compute_120) |

---

## META PROMPT

**Read sequence:** `AGENTS.md` → `HANDOFF.md` (this file) → `git status` → verify build artifacts exist.

**Current priorities:**
- GEMV is the bottleneck (0.193ms per 2048×2048, 77% of decode). Fuse O proj + norm + pack next.
- All Phase A/B/C benchmark results in `bench/PHASE_*_RESULTS.md`.
- Real weights at `weights/*.fp4` (4 layers of 28). `tools/extract_weights` generates them.
- Do NOT restart analysis — continue incrementally.
- After every edit: build → symbols → bench run → no regressions.
- DO NOT re-derive decisions from HANDOFF.md — they're recorded for continuity.
- Update HANDOFF.md only for materially important state changes. Keep concise. No narrative history.
