# HANDOFF.md — blackwell

Preserves operational context between sessions. Read before acting.

---

## 1. Current Objective

FP4 E2M1 block-scaled LLM inference kernels on RTX 5060 Ti (SM_120).  
Decode pipeline with all optimizations working. Current: 40.6 t/s (28-layer est), 35.6% of llama.cpp (114 t/s target).

---

## 2. Current Status

| Domain | Status | Note |
|--------|--------|------|
| GEMM (prefill) | ✅ 25–38 GB/s | 128×128×64 CTA, cp.async pipeline, 136 regs |
| GEMV v1 (strided) | ✅ | Original, K iterations, 22 GB/s (4.4% peak) |
| GEMV v2 (vectorized) | ✅ NEW | uint4 block loads, K/16 iterations, 55-164 GB/s (11-33% peak), **2.5× speedup** |
| Transposed weights | ✅ NEW | W_t [N×K] layout enables sequential K access |
| GEMV benchmark | ✅ | `bench/gemv_v2_bench` — compares v1 vs v2 |
| fused_gate_up_gemv | ✅ NEW | Fused gate + up in one kernel |
| fused_rmsnorm_pack | ✅ | Single-block RMSNorm + FP4 pack |
| attention_decode_gqa | ✅ NEW | GQA-aware (16 Q heads, 8 KV heads) |
| fused_qkv_gemv | ✅ FIXED | Multi-block Grid(3, tiles) for q_dim/kv_dim > 256 |
| Decode full benchmark | ✅ | `bench/decode_full` — all optimizations integrated |
| SwiGLU, update_kv_cache | ✅ | Working |
| llama.cpp baseline | ✅ | 4560 t/s pp512, 114 t/s tg128 |

**26 public API symbols** (was 20). All outside anonymous namespace.

---

## 3. Performance Summary

| Config | 4-layer t/s | 28-layer est | % of target |
|--------|-------------|--------------|-------------|
| Attention-only (v1 GEMV) | 601 | 86 | 75% |
| Full + MLP (v1 GEMV) | 76 | 10.9 | 9.6% |
| Full + MLP (v2 GEMV, transposed) | 284 | 40.6 | **35.6%** |

**3.7× decode speedup** from original to fully optimized.

---

## 4. GEMV v2 Optimization

**Root cause of bottleneck**: Original GEMV reads `W[k*N + n]` — stride N bytes between K iterations. Each cache line (128 bytes) only uses 1 byte (0.8% utilization). Peak BW 500 GB/s, achieved 22 GB/s.

**Fix**: Transpose weights to `W_t[N×K]`. Then reads become sequential: `W_t[n*K + kb*16 + j]`. uint4 load (16 bytes) per iteration instead of byte. K/16 iterations instead of K.

**Results**:
| Test | v1 ms | v2 ms | speedup | v2 GB/s | %peak |
|------|-------|-------|---------|---------|-------|
| O-proj (2048×2048) | 0.193 | 0.077 | **2.5×** | 55.3 | 11.1% |
| K/V-proj (2048×1024) | 0.193 | 0.077 | **2.5×** | 27.8 | 5.6% |
| gate/up (2048×6144) | 0.193 | 0.078 | **2.5×** | 164.4 | **32.9%** |
| down (6144×2048) | 0.576 | 0.229 | **2.5×** | 55.8 | 11.2% |

**Remaining gap**: v2 still only 11-33% of peak BW. Further optimizations needed.

---

## 5. Known Issues / Risks

| Issue | Severity | Root Cause | Fix |
|-------|----------|------------|-----|
| 40.6 vs 114 t/s | 🟡 Under target | down_proj GEMV dominates (2.1ms of 3.5ms) | Optimize down_proj or batch it |
| GEMV v2 only 33% peak | 🟡 Bottleneck | L2 cache thrashing for large N | Paged weights, smem tiling for x |
| GQA attention | ✅ FIXED | Was reading wrong KV head range | Added kv_head = q_head * kv/q |

---

## 6. Important Constraints

- `CUDACXX=/usr/local/cuda-12.8/bin/nvcc` before `project()` in CMakeLists.txt
- g++-12 host compiler, CUDA 12.8
- `namespace wmma = nvcuda::wmma` (NOT `using wmma =`)
- All WMMA code guarded `#if __CUDA_ARCH__ >= 800`
- Shared mem: 99 KB/block max
- SM_120: no tcgen05.mma (only wmma/mma.sync)
- Transposed weights required for GEMV v2

---

## 7. Pending Tasks

| Task | Priority |
|------|----------|
| Optimize down_proj (6144→2048) GEMV | High |
| Profile down_proj vs gate/up (same data size, different dims) | Medium |
| Investigate L2 cache behavior for N=6144 | Medium |
| Consider batching token decode (multiple x vectors simultaneously) | Low |

---

## 8. Key Files / Commands

**Build**: `CUDACXX=/usr/local/cuda-12.8/bin/nvcc cmake -B build && cmake --build build --parallel`

**Benchmarks**:
- `bench/phase_a` — kernel throughput
- `bench/gemv_v2_bench` — v1 vs v2 comparison
- `bench/gemv_char` — GEMV characterization
- `bench/decode_full N` — full decode with real weights
- `bench/decode_bench` — synthetic weights (attention only)

**Verify symbols**: `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell::kernels" | grep -v anonymous | wc -l` → 26

---

## 9. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-05-26 (this session) |
| branch | master |
| nvcc | /usr/local/cuda-12.8/bin/nvcc |
| target | SM_120 native |

---

## META PROMPT

**Read sequence:** `AGENTS.md` → `HANDOFF.md` → `git status` → verify build

**Current priorities:**
- GEMV v2 is 2.5× faster than v1, still 11-33% of peak. down_proj dominates decode time.
- Transposed weights required for GEMV v2. Pre-transpose at startup.
- Do NOT restart analysis — continue incrementally.
- After every edit: build → symbols → bench run → no regressions.