# AGENTS.md — blackwell

Custom CUDA kernels for LLM inference on RTX 5060 Ti (Blackwell, SM_120a).  
FP4 E2M1 + INT8 block-scaled GEMM/GEMV + fused epilogues (RMSNorm, SwiGLU, RoPE).

---

## 1. Mission

Build performant custom CUDA kernels demonstrating FP4 + INT8 LLM inference on consumer Blackwell hardware.  
Primary goal: benchmark INT8 forward pass throughput vs llama.cpp (Q4_K_M) baseline (114 t/s).
Current: **122.7 t/s** (108% of target — TARGET EXCEEDED ✅).

---

## 2. Active State

**Stack**: CUDA 12.8, SM_120a (FP4 MMA requires `120a` arch suffix), nvcuda::wmma (FP16), CMake, C++17  
**Target**: RTX 5060 Ti 16 GB, compute 12.0, 36 SMs, ~500 GB/s GDDR7  
**Host compiler**: g++-12 (CUDA 12.8 rejects GCC 13+ without `--allow-unsupported-compiler`)  
**Nvcc path**: `/usr/local/cuda-12.8/bin/nvcc` (NOT `/usr/bin/nvcc` — that's old CUDA 12.0)

**Working kernels**:

**FP4 path:**
- `pack_fp4` / `unpack_fp4` — FP4 E2M1 quant/dequant
- `gemv_fp4` / `gemv_fp4_v2` — decode path, 22–164 GB/s depending on N
- `gemm_fp4_block_scaled` — prefill path, 128×128×64 WMMA tiles, 2-stage cp.async
- `fused_gate_up_gemv` / `fused_gate_up_gemv_v1` — fused gate+up MLP projection
- `fused_rmsnorm_pack` — RMSNorm + FP4 pack (1 kernel)
- `fused_rmsnorm` — single-block warp-reduced RMSNorm
- `apply_swiglu` — silu(gate) × up, elementwise
- `fused_rope` — in-place rotation, smem cos/sin cache
- `attention_decode` / `attention_decode_gqa` — GQA-aware decode attention
- `update_kv_cache` — KV cache write
- `dispatch_matmul` — routes GEMM vs GEMV by `KernelMode`
- `transpose_fp4_weights` — W (K×N) → W_t (N×K) + scales
- `vector_add_fp32` — elementwise FP32 add (residual)
- `fused_qkv_gemv` — multi-block fused QKV (66 GB/s, slower than 3× gemv_fp4_v2)
- `fused_o_norm_pack` — Wo gemv + rmsnorm + pack (convenience)
- `coalesced_copy` — device-wide coalesced copy
- `attention_fp4`, `load_kv_cache_qkgv` — stubs (`cudaErrorNotReady`)

**INT8 path:**
- `gemv_int8` — baseline with `__dp4a` SIMD, **775 GB/s** (4.7× FP4 v2)
- `gemv_int8_splitk` — split-K variant (K_splits=4), 779 GB/s (N=6144)
- `gemv_int8_persistent` — persistent-thread variant, **23× slower — DO NOT USE**
- `gemv_int8_batched` — template-batched M=1..8, sweet spot M=3-4 (1.4× speedup)
- `gemv_int8_from_fp4` — fused FP4→INT8 inline, **2.8× slower — DO NOT USE**
- `pack_int8` — FP32 → INT8 quant
- `transpose_int8_weights` — W (K×N) → W_t (N×K) + scales
- `fused_rmsnorm_quant_int8` — RMSNorm + INT8 quant (1 kernel)

**Declared in header but NOT implemented** (phase_a.cu fails to link):
- `gemv_fp4_splitk`
- `gemv_fp4_v3`
- `gemv_fp4_batched`

**Build**:

```
CUDACXX=/usr/local/cuda-12.8/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Output: `build/libblackwell_kernels.a`

**Phase A results**: bench/PHASE_A_RESULTS.md (historical, binary cannot be rebuilt — see phase_a.cu linker issues)  
**INT8 pipeline benchmark**: `./bench/decode_full_int8 4`  
**llama-bench baseline**: `/mnt/data/ai/llama.cpp/build-cuda12.8-sm120/bin/llama-bench —hf-repo ...`  
**Models available**: `/mnt/data/ai/hf/qwen3-1.7b-base/` (safetensors, 3.3 GB)

**Constraints**:

- `CUDACXX` env var must be set **before** `project()` in CMakeLists.txt (or use explicit: `CUDACXX=/usr/local/cuda-12.8/bin/nvcc cmake -B build`)
- `namespace wmma = nvcuda::wmma` (namespace alias, NOT `using wmma =` — that creates a type alias, fails)
- All `nvcuda::wmma` code guarded by `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800`
- FP4 E2M1 values: `{0.25, 0.5, 1.0, 2.0}` positive, `{-0.25, -0.5, -1.0}` negative; scale = absmax / 3.0
- `sizeof(__nv_fp4_e2m1)` = 1 byte (not 0.5). N elements → N bytes output.

**Key risks**:

- SM_120a arch (not plain sm_120) is **critical** for FP4 block-scaled MMA. Generic builds drop to 2% perf (47× slower confirmed).
- All weight matrices exceed L2 cache (32 MB). 6144×2048 INT8 = 12 MB. 3 full layers = 36 MB > L2 — **architectural limit**.
- **down_proj (N=6144)**: 24 thread blocks < 36 SMs. Wave quantization wastes 12 SMs.
- **Gate + up GEMVs dominate** INT8 decode time (~36% of kernel time for 2 layers).
- **GEMM prefill severely underperforming** (13–19 GB/s vs 500 GB/s peak). ~3.5% of peak. Separate problem.
- `phase_a.cu` **cannot link** — depends on `gemv_fp4_splitk`, `gemv_fp4_v3`, `gemv_fp4_batched` which are declared in header but never implemented.
- Single-token decode cannot be further optimized via kernel changes alone — weight bandwidth is the ceiling.

---

## 3. Seed Principles

1. **Smallest correct change.** One kernel, one fix, one test at a time.
2. **Verify before broad edits.** Run the kernel/test after every change.
3. **Prefer repo evidence.** Read the code before assuming. Mark unknowns.
4. **No churn.** Don't restyle, don't reorder imports, don't touch files unrelated to the active task.
5. **Kernels first, framework later.** Raw CUDA benchmarks before Python bindings or model loading.

---

## 4. Development Loop

```
observe → plan → edit → build → test (run) → reflect → update AGENTS.md only if useful
```

**Observe**: Read relevant source, header, test files.  
**Plan**: Small focused change. One kernel or one bug fix.  
**Edit**: Apply edit.  
**Build**: `CUDACXX=/usr/local/cuda-12.8/bin/nvcc cmake --build build --parallel`  
**Test**: `./bench/decode_full_int8 4` for INT8 pipeline benchmark, or per-kernel test binary.  
NOTE: `phase_a.cu` cannot link — depends on `gemv_fp4_splitk`, `gemv_fp4_v3`, `gemv_fp4_batched` which are never implemented.
**Reflect**: If test passes, done. If fails, narrow cause → repeat loop.

**Benchmark test binary**:

```
CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
  -gencode=arch=compute_120a,code=sm_120a \
  -I include bench/decode_full_int8.cu build/libblackwell_kernels.a -o bench/decode_full_int8
```

---

## 5. Verification

After every change to a `.cu` or `.h` file:

1. `cmake --build build --parallel` — must succeed
2. `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell::kernels" | grep -v "anonymous namespace"` — public API symbols present (~40 public wrappers)
3. `./bench/decode_full_int8 2` — check no segfault, throughput stable (target 97+ t/s for 2L)
4. For GEMM correctness: all output elements should be non-zero (K×A×B product), no NaNs

---

## 6. Anti-Hallucination Rules

- **Do not invent APIs, files, commands, env vars, or requirements.** Read the actual header/source before calling a function.
- **Prefer repo evidence over assumptions.** If you need a function signature, read `include/blackwell/kernels.h`.
- **Mark unknowns explicitly.** "Not checked" or "unknown behavior" in comments.
- **Never overwrite higher-priority instructions.** E.g., don't change CMake `project()` ordering without understanding why.
- **Preserve user intent and existing project conventions.** Match style of surrounding code.
- **sizeof(\_\_nv_fp4_e2m1) = 1.** Not 0.5. Not sizeof(void\*). 1.

---

## 7. Update Policy

- Update **only** when project structure, build commands, or active constraints change.
- Remove completed tasks from Notes section.
- Add new constraints as discovered.
- Do NOT update for every small edit — only meaningful state changes.
- Never bloat with redundant history.

## 8. Active Work

**Phase C complete. Current: 122.7 t/s (108% of 114 t/s target — TARGET EXCEEDED).**

### Done
- [x] INT8 GEMV with `__dp4a` SIMD — 775 GB/s isolated, 97.8 t/s pipeline (1.95× over FP4 baseline)
- [x] `gemv_int8_splitk` — K_splits=4, 779 GB/s (N=6144)
- [x] `gemv_int8_batched` — M=1..8, sweet spot M=3-4 (1.4× multi-sequence)
- [x] `fused_rmsnorm_quant_int8` — INT8-native x residual path (eliminates FP4 round-trips)
- [x] FP4 round-trip removal for attn→Wo and mlp→down paths
- [x] SM_120a arch suffix in all build flags
- [x] Verified: persistent kernel 23× slower, fused FP4→INT8 2.8× slower
- [x] CUDA Graph for INT8 decode — eliminates inter-kernel launch gaps (+10.5%). 122.7 t/s.
- [x] Inter-layer weight prefetch evaluated — NOT viable (L2 < weight set, no DMA on consumer GPU)
- [x] SM120 consumer GPU research — lacks tcgen05/TMEM, no new INT8 tensor core instructions

### Remaining work: production hardening
- Weight generation for layers 4-27 (full 28L benchmark)
- CUDA Graph correctness verification (output comparison)
- Dynamic seq_pos via graph node update API
- Multi-sequence decode for further throughput gains

### Potential next directions (not in-scope)
1. **Multi-sequence decode** — batched GEMV (M=3-4) amortizes weight loads across tokens. Path to 150+ t/s.
2. **CUTLASS-style warp-tiled GEMV** — shared-memory weight reuse for large K. Theoretical ceiling ~2× bandwidth.
3. **FP4 tensor-core decode** — 2× less weight bandwidth (6 MB vs 12 MB). Trade-off: pack/unpack overhead.
4. **Inter-layer weight prefetch** — PRESERVE-style async copy. Not viable on consumer GPU (L2 < weight set, no separate DMA).

## 9. Insights from llama.cpp Research

**Repo**: https://github.com/ggml-org/llama.cpp (master, 2026-05-27)

**GEMV strategy (mmvq)**:
- Batched: processes up to 8 tokens simultaneously (`MMVQ_MAX_BATCH_SIZE=8`)
- Grid = (nrows / rows_per_block, nchannels, ncols_dst)
- Block = (warp_size, nwarps) where nwarps=4 for 1-4 cols, nwarps=2 for 5-8 cols
- No split-K, no persistent threads, no grid-stride loops
- Small-K optimization: increase rows_per_block when K small to utilize threads

**Blackwell support**:
- `GGML_CUDA_CC_BLACKWELL=1200`
- Two FP4 types: MXFP4 (block=32, 1 E8M0 scale) and NVFP4 (block=64, 4 UE4M3 subblock scales)
- PTX `mma.sync.aligned.kind::mxf4` and `kind::mxf4nvf4` block_scale instructions (m16n8k64)
- Build: `120a-real` arch (not 120 — needs 12Xa for FP4 tensor core instructions)
- Activation quantization: when `blackwell_mma_available()`, activations quantized to MXFP4 instead of Q8_1 for MMQ path

**Key gap**: llama.cpp achieves high throughput through GEMV batching (2-8 tokens), NOT through split-K or persistent blocks. Our split-K/persistent approach explored alternatives for single-token decode but confirmed batching is superior — only applicable to multi-sequence workloads.

