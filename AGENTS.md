# AGENTS.md — blackwell

Custom CUDA kernels for LLM inference on RTX 5060 Ti (Blackwell, SM_120).  
FP4 E2M1 block-scaled GEMM/GEMV + fused epilogues (RMSNorm, SwiGLU, RoPE).

---

## 1. Mission

Build performant custom CUDA kernels demonstrating FP4 LLM inference on consumer Blackwell hardware.  
Primary goal: benchmark FP4 forward pass throughput vs llama.cpp (Q4_K_M) baseline.

---

## 2. Active State

**Stack**: CUDA 12.8, SM_120 native, nvcuda::wmma (FP16), CMake, C++17  
**Target**: RTX 5060 Ti 16 GB, compute 12.0, 36 SMs, ~500 GB/s GDDR7  
**Host compiler**: g++-12 (CUDA 12.8 rejects GCC 13+ without `--allow-unsupported-compiler`)  
**Nvcc path**: `/usr/local/cuda-12.8/bin/nvcc` (NOT `/usr/bin/nvcc` — that's old CUDA 12.0)

**Working kernels**:

- `pack_fp4` / `unpack_fp4` — FP4 E2M1 quant/dequant (verified)
- `gemv_fp4` — decode path, K=64 hardcoded (TODO #2: dynamic K)
- `gemm_fp4_block_scaled` — prefill path, 16×16×64 WMMA tiles
- `fused_rmsnorm` — single-block warp-reduced RMSNorm
- `apply_swiglu` — silu(gate) × up, elementwise
- `fused_rope` — in-place rotation, smem cos/sin cache
- `dispatch_matmul` — routes GEMM vs GEMV by `KernelMode`
- `attention_fp4`, `update_kv_cache`, `load_kv_cache_qkgv` — stubs (return `cudaErrorNotReady`)

**Build**:

```
CUDACXX=/usr/local/cuda-12.8/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Output: `build/libblackwell_kernels.a`

**Phase A complete**: bench/phase_a executable + bench/PHASE_A_RESULTS.md  
**llama-bench baseline**: `/mnt/data/ai/llama.cpp/build-cuda12.8-sm120/bin/llama-bench —hf-repo ...`  
**Models available**: `/mnt/data/ai/hf/qwen3-1.7b-base/` (safetensors, 3.3 GB)

**Constraints**:

- `CUDACXX` env var must be set **before** `project()` in CMakeLists.txt (or use explicit: `CUDACXX=/usr/local/cuda-12.8/bin/nvcc cmake -B build`)
- `namespace wmma = nvcuda::wmma` (namespace alias, NOT `using wmma =` — that creates a type alias, fails)
- All `nvcuda::wmma` code guarded by `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800`
- FP4 E2M1 values: `{0.25, 0.5, 1.0, 2.0}` positive, `{-0.25, -0.5, -1.0}` negative; scale = absmax / 3.0
- `sizeof(__nv_fp4_e2m1)` = 1 byte (not 0.5). N elements → N bytes output.

**Key risks**:

- SM_120 native is **critical**. Generic builds drop to 2% perf (47× slower confirmed).
- GEMM severely underperforming (3–5 GB/s vs 500 GB/s peak). Tile size too small (16×16), no async copy, no vectorized FP4 loads.
- GEMV K=64 hardcoded. Real LLM layers need K-tiling over hidden_dim=2048.

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
**Test**: `./bench/phase_a` for integrated test, or `make`-specific test binary.  
**Reflect**: If test passes, done. If fails, narrow cause → repeat loop.

**Benchmark test binary**:

```
CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
  -gencode=arch=compute_120,code=sm_120 \
  -I include bench/phase_a.cu build/libblackwell_kernels.a -o bench/phase_a
```

---

## 5. Verification

After every change to a `.cu` or `.h` file:

1. `cmake --build build --parallel` — must succeed
2. `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell::kernels"` — public API symbols present, NOT in anonymous namespace
3. `./bench/phase_a` — check no segfault, no "FAIL" lines, output values reasonable (not all zeros, not inf)
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
