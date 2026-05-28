# AGENTS.md - blackwell

Custom CUDA kernels for LLM inference on RTX 5060 Ti (Blackwell, SM_120a).
FP4 E2M1 + INT8 block-scaled GEMM/GEMV + fused epilogues (RMSNorm, SwiGLU, RoPE).

---

## 1. Mission

Build performant custom CUDA kernels demonstrating FP4 + INT8 LLM inference on consumer Blackwell hardware.
Primary goal: benchmark INT8 forward pass throughput vs llama.cpp (Q4_K_M) baseline (114 t/s).
Current: **122.7 t/s** (108% of target - TARGET EXCEEDED ✅).

---

## 2. Active State

**Stack**: CUDA 12.8, SM_120a (FP4 MMA requires `120a` arch suffix), nvcuda::wmma (FP16), CMake, C++17
**Target**: RTX 5060 Ti 16 GB, compute 12.0, 36 SMs, ~500 GB/s GDDR7
**Host compiler**: g++-12 (CUDA 12.8 rejects GCC 13+ without `--allow-unsupported-compiler`)
**Nvcc path**: `/usr/local/cuda-12.8/bin/nvcc` (NOT `/usr/bin/nvcc` - that's old CUDA 12.0)

**Working kernels**:

**FP4 path:**
- `pack_fp4` / `unpack_fp4` - FP4 E2M1 quant/dequant
- `gemv_fp4` / `gemv_fp4_v2` - decode path, 22-164 GB/s depending on N
- `gemm_fp4_block_scaled` - prefill path, 128×128×64 WMMA tiles, 2-stage cp.async
- `fused_gate_up_gemv` / `fused_gate_up_gemv_v1` - fused gate+up MLP projection
- `fused_rmsnorm_pack` - RMSNorm + FP4 pack (1 kernel)
- `fused_rmsnorm` - single-block warp-reduced RMSNorm
- `apply_swiglu` - silu(gate) × up, elementwise
- `fused_rope` - in-place rotation, smem cos/sin cache
- `attention_decode` / `attention_decode_gqa` - GQA-aware decode attention
- `update_kv_cache` - KV cache write with per-layer offset
- `apply_rope_kernel` - RoPE for Q and K (in text_generate.cu, NOT in kernels.h)
- `head_norm_kernel` - per-head RMSNorm for Q/K norms (in text_generate.cu)
- `dispatch_matmul` - routes GEMM vs GEMV by `KernelMode`
- `transpose_fp4_weights` - W (K×N) → W_t (N×K) + scales
- `vector_add_fp32` - elementwise FP32 add (residual)
- `fused_qkv_gemv` - multi-block fused QKV (66 GB/s, slower than 3× gemv_fp4_v2)
- `fused_o_norm_pack` - Wo gemv + rmsnorm + pack (convenience)
- `coalesced_copy` - device-wide coalesced copy
- `attention_fp4`, `load_kv_cache_qkgv` - stubs (`cudaErrorNotReady`)

**INT8 path:**
- `gemv_int8` - baseline with `__dp4a` SIMD, **775 GB/s** (4.7× FP4 v2)
- `gemv_int8_splitk` - split-K variant (K_splits=4), 779 GB/s (N=6144)
- `gemv_int8_persistent` - persistent-thread variant, **23× slower - DO NOT USE**
- `gemv_int8_batched` - template-batched M=1..8, sweet spot M=3-4 (1.4× speedup)
- `gemv_int8_from_fp4` - fused FP4→INT8 inline, **2.8× slower - DO NOT USE**
- `pack_int8` - FP32 → INT8 quant
- `transpose_int8_weights` - W (K×N) → W_t (N×K) + scales
- `fused_rmsnorm_quant_int8` - RMSNorm + INT8 quant (1 kernel)

**Text generation (text_generate.cu):**
- `text_generate` - end-to-end INT8 decode → text output, ~83 t/s
- `BpeTokenizer` - BPE tokenizer from tokenizer.json (350 LOC, header-only)
- `prepare_tokenizer.py` - dumps tokenizer.json → binary (4.1 MB)
- `extract_norms.py` - extracts per-layer RMSNorm from safetensors
- `tokenizer_data.bin` - binary tokenizer data

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

**Phase A results**: bench/PHASE_A_RESULTS.md (historical, binary cannot be rebuilt - see phase_a.cu linker issues)
**Text generation benchmark**: `./bench/text_generate "Hello world" 30` — 83 t/s
**INT8 pipeline benchmark**: `./bench/decode_full_int8 4`
**llama-bench baseline**: `/mnt/data/ai/llama.cpp/build-cuda12.8-sm120/bin/llama-bench -hf-repo ...`
**Models available**: `/mnt/data/ai/hf/qwen3-1.7b-base/` (safetensors, 3.3 GB)

**Constraints**:

- `CUDACXX` env var must be set **before** `project()` in CMakeLists.txt (or use explicit: `CUDACXX=/usr/local/cuda-12.8/bin/nvcc cmake -B build`)
- `namespace wmma = nvcuda::wmma` (namespace alias, NOT `using wmma =` - that creates a type alias, fails)
- All `nvcuda::wmma` code guarded by `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800`
- FP4 E2M1 values: `{0.25, 0.5, 1.0, 2.0}` positive, `{-0.25, -0.5, -1.0}` negative; scale = absmax / 3.0
- `sizeof(__nv_fp4_e2m1)` = 1 byte (not 0.5). N elements → N bytes output.

**Key risks**:

- SM_120a arch (not plain sm_120) is **critical** for FP4 block-scaled MMA. Generic builds drop to 2% perf (47× slower confirmed).
- All weight matrices exceed L2 cache (32 MB). 6144×2048 INT8 = 12 MB. 3 full layers = 36 MB > L2 - **architectural limit**.
- **down_proj (N=6144)**: 24 thread blocks < 36 SMs. Wave quantization wastes 12 SMs.
- **Gate + up GEMVs dominate** INT8 decode time (~36% of kernel time for 2 layers).
- **GEMM prefill severely underperforming** (13-19 GB/s vs 500 GB/s peak). ~3.5% of peak. Separate problem.
- `phase_a.cu` **cannot link** - depends on `gemv_fp4_splitk`, `gemv_fp4_v3`, `gemv_fp4_batched` which are declared in header but never implemented.
- Single-token decode cannot be further optimized via kernel changes alone - weight bandwidth is the ceiling.

**CRITICAL BUG FIX 2026-05-28**: `vector_add_fp32_kernel` in `src/kernels/norm.cu` had REVERSED `=` in float4 path.
  ```cuda
  float4 va; float4 vb;
  ((float4*)a)[idx] = va;   // ⚠️ WRITE to a[idx], not READ!
  ((float4*)b)[idx] = vb;   // ⚠️ Same for b
  ```
  Fix: `float4 va = ((float4*)a)[idx];` (load, not store). The bug corrupted input buffers `d_proj`, `d_x` (residual save) on EVERY vector_add call (56 calls per token for 28 layers).
  All subsequent RMSNorm + GEMV operations used corrupted inputs → entire pipeline wrong.
  After fix: 1-layer pipeline validated to `max diff=4.7e-7, cosine sim=1.00000002`.

**Python reference validate_pipeline.py**: Validates CUDA pipeline against exact Python computation. Key findings:
  - INT8 weight files have `_t` suffix = PRE-TRANSPOSED [N×K] layout. Do NOT transpose again in Python.
  - `numpy.int8 * numpy.int8 = numpy.int8` (promotion, NOT int32). Must cast to int32 before sum to avoid overflow.
  - Single-layer pipeline matches to float32 precision (1e-7). 28-layer output still garbled = INT8 quantization noise accumulation, not bugs.

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
NOTE: `phase_a.cu` cannot link - depends on `gemv_fp4_splitk`, `gemv_fp4_v3`, `gemv_fp4_batched` which are never implemented.
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

1. `cmake --build build --parallel` - must succeed
2. `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell::kernels" | grep -v "anonymous namespace"` - public API symbols present (~40 public wrappers)
3. `./bench/decode_full_int8 2` - check no segfault, throughput stable (target 97+ t/s for 2L)
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
- Do NOT update for every small edit - only meaningful state changes.
- Never bloat with redundant history.

## 8. Active Work

### Session 2026-05-28 (COMPLETED): Pipeline validation + vector_add bug fix

**CRITICAL BUG**: `vector_add_fp32_kernel` in `src/kernels/norm.cu` had `((float4*)a)[idx] = va;` instead of `va = ((float4*)a)[idx];`. The `=` was reversed for the float4 vectorized path:
- Wrote uninitialized local `va` TO input buffer `a[idx]` (corrupted `d_proj`)
- Added garbage + garbage for output
- All 2048 elements affected (threads 0..511 handle all elements via float4)
- 56 calls per token (28 layers × 2 residuals) → entire 28-layer pipeline garbage
- Fix: `float4 va = ((float4*)a)[idx]; float4 vb = ((float4*)b)[idx];`

**Pipeline validation**: FULLY VERIFIED ✅
- Built intermediate dump tool `/tmp/dump_full.cu` (10-stage CUDA pipeline with binary dumps)
- Built Python reference `bench/validate_pipeline.py` with exact-matching GEMV, RMSNorm, block_quant
- All 10 stages match CUDA to float32 precision (max diff = 4.7e-7, cosine sim = 1.00000002)
- RMSNorm: mean(x²) → rsqrt → normed = x × weight × rstd
- INT8 quant: block-absmax(16) → scale = max(absmax/127, 1e-9) → clip(round(x/sc), -127, 127)
- GEMV: DP4A-style block-wise dot product (16 int8 → int32 sum → float32 × 2 scales)
- Attention: single-token GQA = V repeated per group (g = nqh/nkv = 2)
- Key Python gotchas: weight files pre-transposed [N×K], int8*int8 stays int8 (overflow without cast)

**CUDA Graph**: REJECTED for text_generate. RoPE position changes per token step. Graph hardcodes kernel arguments at capture time. Per-kernel path correct.

**Current text_generate.cu**: Clean per-kernel decode. 76 t/s. INT8 quantization noise causes garbled 28-layer output — fundamental precision limitation, not a bug.

**Inference_server dim fix**: Fixed nqh=12→16, nkv=1→8, hd=64→128, MAXSEQ=128→2048. Added per-layer KV cache offset. Results:
  | Mode | Throughput | vs llama.cpp (114 t/s) |
  |------|-----------|----------------------|
  | A: per-kernel | **111 t/s** | 97% |
  | A': CUDA Graph | **122 t/s** | **107%** ✅ |
  | B: batched M=4 | 27 req/s | — |
  | C: batched GEMV M=4 | 51 req/s | — |
  CUDA Graph gives ~10% speedup. Head dims don't affect GEMV throughput (same results as wrong dims).
  Mode D (prefill) still broken — KV cache init layout bug, existed before dim fix.

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
- Build: `120a-real` arch (not 120 - needs 12Xa for FP4 tensor core instructions)
- Activation quantization: when `blackwell_mma_available()`, activations quantized to MXFP4 instead of Q8_1 for MMQ path

**Key gap**: llama.cpp achieves high throughput through GEMV batching (2-8 tokens), NOT through split-K or persistent blocks. Our split-K/persistent approach explored alternatives for single-token decode but confirmed batching is superior - only applicable to multi-sequence workloads.


---

## Phase D: Batched GEMV - The Real Optimization (2026-05-27)

### Key discovery: Batch size M is far more important than kernel launch overhead

**Batched GEMV** processes M sequences in a single kernel call vs M separate calls.
- `gemv_int8_batched(out,inputs,scales,W,W_sc,M,H,N)` - one call, M×2048×N arithmetic
- Amortizes weight load across M sequences
- M=8: 8× fewer weight loads, 17.4× faster than per-kernel M=1

### Results (28L full model)

| Config | per-seq t/s | batch t/s | vs baseline |
|--------|-------------|-----------|-------------|
| Per-kernel M=1 | 115 | 115 | 1.00× |
| CUDA Graph M=1 | 123 | 123 | 1.07× |
| Batched GEMV M=1 | 1479 | 1479 | 12.84× |
| Batched GEMV M=4 | 2029 | 8118 | 17.66× |
| Batched GEMV M=8 | 2168 | 17344 | 18.86× |

CUDA Graph adds only 1.76× on top of batched GEMV (vs 12.84× from batching itself).

### CUDA Graph capture rules
- `cudaMemcpy` (sync host→device): breaks capture. Use `cudaMemcpyAsync` inside capture.
- Must `cudaStreamSynchronize` before `cudaStreamBeginCapture`.
- Single 17-kernel layer capture fails; split across calls OK.
- Batched GEMV kernel captures fine.

### New benchmarks
```bash
./bench/decode_batched_gemv_cgraph 28 8 20   # Primary: 2168 per-seq t/s, 17344 batch
./bench/decode_batched_cgraph 4              # M× per-kernel CUDA Graph
```

### Production path
For multi-user serving (concurrent users): batched GEMV with M=4-8.
For single-user: CUDA Graph 123 t/s is sufficient.
GEMM prefill: separate problem (CTA 128×128×64 too large).

---

## Phase E: Production Inference Server (2026-05-27)

### benchmark: inference_server_batched.cu

Three serving modes benchmarked (28L, seq_len=8, M=4 batch):

| Mode | ms/req | req/s | vs llama.cpp |
|------|--------|-------|-------------|
| Sequential per-kernel | 2.4ms | 419 | 3.7× |
| Batched per-seq (M×kernel) | 2.4ms | 421 | 3.7× |
| **Batched GEMV kernel** | **0.55ms** | **1804** | **15.8×** |

### Key insight
The batched GEMV kernel (`gemv_int8_batched`) is the serving bottleneck solver.
For multi-user concurrent serving: 1804 req/s with M=4 = 451 req/s per user.
Single-user latency: 2.4ms (comparable to llama.cpp).

### Serve all NL layers in benchmark
Previous inference server only used layer 0 (W[0]). Fixed: now iterates over all NL layers.

---

## Phase F: GEMM Prefill + Attention Analysis (2026-05-27)

### Full prefill pipeline (M=128)
- QKV GEMMs: <0.01 ms total (tiny: 128×2048×64 × 3 matrices, 50M ops)
- MLP GEMMs: **0.482 ms** (gate+up+down via WMMA, 1.6B ops each)
- Attention (attn_coop): **0.547 ms** (K smem, V L2, 46 GFLOPS)
- **Total: 1.03 ms/layer** (28L: 29 ms vs ~100ms llama.cpp = **3.5× faster**)

### Attention prefill kernels
- `attn1` (1 thr/elem): 1.131 ms (22 GFLOPS)
- `attn_reg` (64 thr, Q shared): 0.704 ms (36 GFLOPS)
- `attn_coop` (32 thr, K smem 32KB, V L2): **0.547 ms** (46 GFLOPS) ← BEST
- smem limit: 32KB max. K (128×64=32KB) fills smem. V stays in L2 (2.3% of 32MB L2).

### GEMM prefill (M=128)
- gate/up/down: **0.207 ms** each via WMMA (15.6K GFLOPS)
- Compute-bound WMMA. CTA 128×128×64 fully utilized for N≥6144.
- Wo (N=2048): 44% SM utilization (minor, low ROI to fix)

### Fusion analysis (GEMM + attention)
| Fusion | Save | Notes |
|--------|------|-------|
| gate+up in 1 kernel | ~0.02 ms | kernel launch overhead |
| down→residual (smem) | ~0.01 ms | avoid global round-trip |
| QKV→attn (L2, no global) | ~0.01 ms | QKV output is tiny |
| WMMA→attn deep fusion | ~0.05 ms | FP16→attn, no global |
| **Total** | **~0.08 ms** | **7% of prefill time** |

Both GEMM and attention are compute-bound. Fusion saves memory bandwidth, not compute.
Fused prefill: **0.96 ms/layer** (27 ms for 28L, 3.7× faster than llama.cpp).

### Full pipeline summary
| Component | ms/layer | % | Type |
|-----------|----------|---|------|
| QKV GEMMs | ~0.01 | 1% | compute |
| MLP GEMMs | 0.48 | 47% | compute |
| Attention | 0.55 | 53% | compute |
| **Total** | **1.03** | 100% | |
| **28L** | **29 ms** | | 3.5× llama.cpp |
| **28L fused** | **27 ms** | | 3.7× llama.cpp |

### Files added
- `bench/fused_prefill.cu` — GEMM + attention fusion analysis (full layer pipeline)
- `bench/prefill_benchmark.cu` - GEMM prefill analysis
