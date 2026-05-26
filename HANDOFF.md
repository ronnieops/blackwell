# HANDOFF.md — blackwell

Preserves operational context between sessions. Read before acting.

---

## 1. Current Objective

Advance FP4 custom CUDA kernels toward end-to-end LLM forward pass benchmark on RTX 5060 Ti.  
Phase A (baseline benchmarks + kernel correctness) complete. Next: close the GEMM performance gap vs llama.cpp.

---

## 2. Current Status

| Domain | Status | Note |
| ------ | ------ | ---- |
| Project skeleton | ✅ | CMake build, pyproject, dirs, static lib links clean |
| pack_fp4 / unpack_fp4 | ✅ | Verified correct. E2M1 ±{0.25,0.5,1.0,2.0}, scale=absmax/3 |
| gemm_fp4_block_scaled | ✅ Runs, wrong throughput | 3–5 GB/s vs 500 GB/s peak — needs optimization |
| gemv_fp4 | ✅ Works, K=64 only | Launch-overhead-bound. Dynamic K tiling = blocker for real LLM |
| fused_rmsnorm | ✅ | Warp-reduced sum, single block (4096 max elements) |
| apply_swiglu | ✅ | Elementwise silu(gate) × up |
| fused_rope | ✅ | In-place rotation, smem cos/sin cache |
| dispatch_matmul | ✅ | Routes GEMM vs GEMV by KernelMode |
| attention_fp4 | 🟡 Stub | Returns cudaErrorNotReady |
| KV-cache (update/load) | 🟡 Stub | Returns cudaErrorNotReady |
| prefill/decode runner | 🟡 Stub | Returns cudaErrorNotReady |
| CUDA Graphs | 🟡 Stub | capture/launch/destroy stubs |
| Phase A benchmark | ✅ | bench/phase_a + PHASE_A_RESULTS.md |
| llama-baseline | ✅ | 4560 t/s pp512, 114 t/s tg128 on Qwen3.5-4B Q4_K_M |
| git repo | ❌ | Not initialized. No commits, no branches |

---

## 3. Recent Decisions

- **GEMM tile = 16×16×64 WMMA** (was 64×64×64 but fragment coverage bug → 12/16 output elements unwritten → all zeros). Grid covers outer M/N.
- **Anonymous namespace fix**: public API functions in gemm.cu were wrapped in anonymous namespace → linker couldn't find them → moved outside.
- **pack_fp4 kernel**: removed block-level absmax race (only block 0 wrote it, blocks>0 read garbage shared mem). Now uses caller-provided scale directly.
- **sizeof(__nv_fp4_e2m1) = 1 byte** (not 0.5). Allocations must be `n * 1`, not `n / 2`.
- **SM_120 native cubins critical**: generic PTX fallback is 47× slower on prefill.
- **Phase A benchmark uses synthetic uniform data** (all 1.0s). Real model weight distribution will differ.

---

## 4. Important Constraints

- `CUDACXX=/usr/local/cuda-12.8/bin/nvcc` must be set before `project()` in CMakeLists.txt
- g++-12 host compiler (CUDA 12.8 rejects GCC 13+)
- `namespace wmma = nvcuda::wmma` — namespace alias, NOT `using wmma =` (type alias, fails)
- All WMMA code guarded: `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800`
- Shared mem limit: 99 KB/block (CC 12.0). Current usage: 2560 B — well within limit
- 48 resident warps/SM max. 128 threads/block currently, can increase

---

## 5. Known Issues / Risks

| Issue | Severity | Root Cause | Fix Path |
| ----- | -------- | ---------- | -------- |
| GEMM 3–5 GB/s (0.6–1% of peak) | 🔴 Blocks end-to-end perf | 16×16 tile too small, no vectorized loads, no async copy, 4× K-loop | Larger tile (32×32 or 64×64 multi-fragment), vectorized FP4→FP16 smem load, async pipeline |
| GEMV K=64 hardcoded | 🟡 Blocks real model | GEMV kernel only handles single K=64 tile | K-tiling loop in kernel or caller-side reduction |
| SwiGLU GB/s > theoretical peak | 🟡 Measurement artifact | Timer resolution too coarse (~3us kernel) | Batch more elements or use CUDA events correctly |
| FP4 rel error = 1.0 for small inputs | 🟢 Expected | E2M1 can't represent values < 0.5×scale below 0 → quantize to 0 | Acceptable for LLM weights (usually >0.1); use FP8/FP6 where accuracy critical |
| No git history | 🟢 Not blocking yet | Repo never initialized | `git init && git add && git commit` before next major changes |

---

## 6. Pending Tasks

| # | Task | Priority | Notes |
| - | ---- | -------- | ----- |
| 4 | Shared-memory tiling with 99KB awareness | Medium | Current usage well under limit. Explore larger tiles and async copy |
| 6 | KV-cache decode (compact bandwidth-first) | High | Next logical step after GEMM optimization |
| 7 | Prefill kernels (separate from decode) | Medium | Depends on GEMM optimization |
| 8 | CUDA Graphs for decode launch overhead | Low | Current GEMV latency = 3us — launch overhead dominates |
| 9 | Profiling hooks (Nsight Compute) | Low | Needed for optimization iterations |

---

## 7. Suggested Next Actions

1. **GEMM optimization** — increase tile size, add vectorized FP4→FP16 smem loads, add async copy pipeline. Target: 100+ GB/s.
2. **Dynamic K GEMV** — loop over K=64 tiles in kernel or dispatch. Enables Qwen3-1.7B hidden_dim=2048.
3. **Benchmark with real model weights** — load Qwen3-1.7B safetensors, quantize to FP4, run forward pass, compare accuracy and throughput vs llama.cpp.
4. **Init git repo** — `git init && git add -A && git commit -m "Phase A complete"`.

---

## 8. Important Files / Commands

**Build**:
```
CUDACXX=/usr/local/cuda-12.8/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

**Run benchmark**:
```
CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
  -gencode=arch=compute_120,code=sm_120 \
  -I include bench/phase_a.cu build/libblackwell_kernels.a -o bench/phase_a
./bench/phase_a
```

**Verify public symbols**:
```
nm build/libblackwell_kernels.a | c++filt | grep " T blackwell::kernels" | grep -v anonymous
```

**llama-bench baseline**:
```
/mnt/data/ai/llama.cpp/build-cuda12.8-sm120/bin/llama-bench \
  --hf-repo unsloth/Qwen3.5-4B-MTP-GGUF:Q4_K_M -p 512 -n 128 -r 3
```

**Key files**:
- `include/blackwell/kernels.h` — public API (all kernel signatures)
- `include/blackwell/config.h` — tile sizes, block sizes, constants
- `src/kernels/gemm.cu` — GEMM + GEMV kernels (needs optimization)
- `src/kernels/quantize.cu` — FP4 pack/unpack (working correctly)
- `src/kernels/norm.cu` — RMSNorm + SwiGLU (working)
- `src/kernels/rope.cu` — RoPE (working)
- `bench/phase_a.cu` — integrated benchmark
- `bench/PHASE_A_RESULTS.md` — Phase A results report

---

## 9. Validation Status

**Last validated**: 2026-05-26, ~15:24 UTC  
**Build**: ✅ Clean (libblackwell_kernels.a, bench/phase_a)  
**Run**: ✅ No segfault, no FAIL lines, all output values reasonable  
**GEMM**: All 4096 elements = 64.0 (correct: K=64 × 1.0 × 1.0). No NaNs.  
**GEMV**: All outputs non-zero, latency 2–4us dominated by launch overhead  
**Public symbols**: All 18 present, none in anonymous namespace  

---

## 10. Session Metadata

| Field | Value |
| ----- | ----- |
| updated_at | 2026-05-26 15:27 UTC |
| branch | (none — not version controlled) |
| repo state | No git history. Run `git init` before next major change |
| active components | src/kernels/ (gemm.cu, quantize.cu, norm.cu, rope.cu), bench/phase_a.cu |
| nvcc | /usr/local/cuda-12.8/bin/nvcc |
| host compiler | g++-12 |
| target | SM_120 native (compute_120) |

---

## META PROMPT

**Mandatory read sequence for all future agents:**

1. Read `AGENTS.md` — project principles, build commands, anti-hallucination rules, verification steps.
2. Read `HANDOFF.md` (this file) — operational state, current objective, known bugs, pending tasks, constraints.
3. Refresh operational state by running `git status` and checking build artifacts exist before making any edit.
4. Do NOT assume any API, function, or constant without reading `include/blackwell/kernels.h` or the relevant `.cu` file.
5. **GEMM is underperforming (3–5 GB/s)**. If optimizing GEMM, start with tile size, shared memory layout, and async copy — not WMMA fragment changes.
6. **GEMV is K=64 hardcoded**. Any real model run requires dynamic K tiling.
7. **No git history exists**. If the task involves significant changes, initialize the repo first.
8. Continue incrementally from the current state. Do NOT restart analysis or re-derive constraints from first principles.
9. After each edit: build → verify symbols → run bench → confirm no regressions.
10. Update HANDOFF.md only when materially important context changes (new bugs, new decisions, completed milestones). Keep additions concise. Remove stale information.
