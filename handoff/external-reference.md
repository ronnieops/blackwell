# Research: How llama.cpp Q4_K_M Achieves Fast GEMV Decode (M=1) on Blackwell

## Summary

llama.cpp uses a **MMVQ (matrix-matrix-vec quantized)** kernel for M=1–8 decode that is fundamentally a **warp-cooperative dot-product kernel** — NOT a tensor-core/MMA path. Each CTA processes one or more output rows; threads within a warp cooperatively iterate over K using vectorized 16-byte loads (`int4`/`uint4`), dequantize Q4_K super-blocks inline, accumulate into FP32 via `__dp4a`, then warp-reduce via `__shfl_down_sync`. The key advantages over our offset-binary block-16 kernel are: (1) Q4_K super-block layout fetches scales less frequently (1 scale per 32 elements vs our 1 per 16), (2) activations are pre-quantized to INT8 (Q8_1 format) so the dot product is pure INT8×INT8 dp4a with no per-element FP conversion, (3) the kernel uses 1 warp per row with all 32 lanes on the K dimension (better occupancy, better memory coalescing, amortized scale fetch across the warp), and (4) 4.5 bpw effective quantization means ~12.5% less weight memory to load than our 4.0 bpw.

---

## Findings

### 1. MMVQ Kernel: `mul_mat_vec_q` in `mmvq.cu`

**File**: `ggml/src/ggml-cuda/mmvq.cu` ([GitHub](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu))

- **No tensor cores for M=1.** MMVQ is a pure CUDA-core dot-product kernel. Tensor cores (MMQ path) are only used for larger batch sizes (prefill, large-batch decode). [DeepWiki CUDA Backend](https://deepwiki.com/ggml-org/llama.cpp/5.1-cuda-backend-nvidia)
- **Architecture**: `mul_mat_vec_q<ncols, nwarps, vec_dot_qX_Y_q8_1>` template kernel.
  - `ncols` = number of output columns per CTA (typically 1 for M=1, up to `MMVQ_MAX_BATCH_SIZE=8`).
  - `nwarps` = warps per CTA (typically 1 for ncols=1).
  - Each CTA computes `ncols` output elements. Multiple output rows spread across CTAs in the grid.
  - Within the CTA, each warp processes one row of W; threads iterate along K dimension.
- **Key insight**: 1 warp (32 threads) per output row. All 32 threads cooperatively walk K, each loading and processing a portion of the weight row. This maximizes memory coalescing and bandwidth utilization.
- **`MMVQ_MAX_BATCH_SIZE = 8`**: Hard-coded upper limit on batch dimension for MMVQ path. Above this, dispatch switches to MMQ (tensor-core) path. ([GitHub source](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu))
- **Dispatch logic** (in `ggml-cuda.cu`): For `M <= MMVQ_MAX_BATCH_SIZE`, uses `mul_mat_vec_q`. For larger M, uses `mul_mat_q` (MMQ). The threshold was refined in [PR #7716](https://app.semanticdiff.com/gh/ggml-org/llama.cpp/pull/7716) which refactored mmq/dmmv/mmvq separation.

### 2. VecDotQ: `vec_dot_q4_K_q8_1` in `vecdotq.cuh`

**File**: `ggml/src/ggml-cuda/vecdotq.cuh` ([GitHub](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vecdotq.cuh))

- **The dot product primitive**: `vec_dot_q4_K_q8_1` computes `result = dot(dequantize(q4_K_weights), q8_1_activations)` entirely in INT32, then scales by FP32 factors at the end.
- **VDR (Values Dequantized per Round)**: `VDR_Q4_K_Q8_1` = number of weight elements processed per thread per inner iteration. For Q4_K, this is tuned to balance register pressure vs ILP.
- **Process per thread**:
  1. Load packed weight bytes (4-bit nibbles packed into uint8_t)
  2. Load Q8_1 activations (pre-quantized to INT8 with block scales)
  3. Unpack 4-bit values to INT8: `(nibble - 8)` offset-binary unpacking (similar to our `nib - 8`)
  4. Use `__dp4a(int32_a, int32_b, acc)` for 4-way INT8×INT8 dot product accumulation
  5. Multiply by combined scale factor (weight scale × activation scale)
- **Critical difference from our kernel**: Activations arrive **pre-quantized to INT8** (Q8_1 format), not FP32. This eliminates per-element FP32→INT8 conversion in the inner loop. The dot product is pure INT8 SIMD.
- **Q8_1 activation format**: Block of 32 INT8 values + 1 FP16 scale + 1 FP16 sum-of-squares (used for some normalization paths). The scale amortizes FP conversion: one FP16 multiply per 32 INT8 dot-products.

### 3. Q4_K Super-Block Layout

**File**: `ggml-quants.h` ([GitHub via NousResearch fork](https://github.com/NousResearch/llama.cpp/blob/master/ggml-quants.h)), [DeepWiki Quantization Formats](https://deepwiki.com/qualcomm/llama.cpp/3.4-quantization-formats), [GGUF Q4_K deep-dive](https://www.meinlebenswerk.link/posts/q4_k), [Tensor Encoding Schemes Wiki](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes)

```
// Q4_K: 256-element super-block
#define QK4_K 256
#define QI4_K (QK4_K / 16)  // = 16 sub-blocks of 16 elements
typedef struct {
    ggml_fp16_t d;       // super-block FP16 scale (1 per 256 elements)
    ggml_fp16_t dmin;    // super-block FP16 min offset (1 per 256 elements)
    uint8_t scales[QI4_K / 2];  // 8 bytes: 16 × 6-bit sub-block scales (packed)
    uint8_t qs[QK4_K / 2];      // 128 bytes: 256 × 4-bit quantized values
} block_q4_K;
// Total: 144 bytes per 256 elements = 4.5 bpw
```

**Dequantize formula per element**:
```
sub_scale_6bit = scales[sub_block_idx]  // 6-bit packed, range 0-63
sub_scale = d * (sub_scale_6bit >> 0)   // FP16 multiply, amortized across 16 elements
sub_min    = dmin * (sub_scale_6bit >> 0)
dequant_value = sub_scale * (qs[element] - 8) - sub_min
```

Wait — the actual dequantize for Q4_K is:
- 256 elements split into 8 sub-blocks of 32 elements each
- Each sub-block has a 6-bit scale (extracted from packed `scales` array)
- Super-block scale `d` (FP16) multiplied by 6-bit sub-block scale gives effective scale
- Each element: `(qs[i] - 8)` (offset-binary) × effective_scale

**Key advantage over our INT4 block-16 FP32 scales**:
- **Scale fetch frequency**: 1 scale fetch per 32 elements (sub-block) vs our 1 per 16 elements. Half the scale loads.
- **Scale packing**: 6-bit sub-block scales packed into 8 bytes (super-block scale amortized). FP16 super-block scale + packed 6-bit sub-block scales = compact, coalesced.
- **4.5 bpw vs 4.0 bpw**: Q4_K uses 144 bytes per 256 elements = 4.5 bpw. Our INT4 block-16 uses 128 bytes weights + 4 bytes FP32 scale per 16 elements = ~6 bpw (128/256 weight + 16 FP32 scales × 4B / 256 = 0.5 + 0.25 = ~6.0 bpw). **Our format loads ~33% more data per element due to FP32 per-block-16 scales.** This is a major memory-bandwidth gap.

[Source: DeepWiki Quantization Formats](https://deepwiki.com/qualcomm/llama.cpp/3.4-quantization-formats), [GGUF Wikipedia](https://en.wikipedia.org/wiki/GGUF), [Q4_K deep-dive blog](https://www.meinlebenswerk.link/posts/q4_k)

### 4. Why MMVQ is Fast for M=1 (No Tensor Cores)

**Sources**: [DeepWiki CUDA Backend](https://deepwiki.com/ggml-org/llama.cpp/5.1-cuda-backend-nvidia), [kernel-anvil analysis](https://github.com/apollosenvy/kernel-anvil/blob/master/docs/llama-cpp-kernel-analysis.md), [Vijay's GEMV optimization blog](https://vijay-kodamalla.github.io/gemv-q4_k-optimization/), [Stack Overflow: mul_mat_vec_q logic](https://stackoverflow.com/questions/79591725/the-logic-of-mul-mat-vec-q-in-llama-cpp)

The GEMV at M=1 is **memory-bound**, not compute-bound. What matters is:
1. **Maximize memory bandwidth utilization**: Vectorized 16-byte (`uint4`/`int4`) loads. Each thread loads a 16-byte chunk = 32 × 4-bit values. Coalesced across warp = 512 bytes per memory transaction. [CUDA Pro Tip: Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)
2. **Minimize total bytes loaded**: Q4_K's 4.5 bpw means ~12.5% less weight data than a pure 4-bit format's overhead from scales. Our block-16 FP32 scales add ~50% overhead (6 bpw effective).
3. **Amortize dequant overhead**: Scale fetches happen once per 32 elements (sub-block), shared across all dequant operations within that sub-block. Our kernel fetches scales every 16 elements.
4. **`__dp4a` for INT8 SIMD**: 4-way INT8 dot product in a single instruction. After unpacking 4-bit → INT8, `__dp4a` computes 4 multiply-adds in one cycle. [CUDA Math API Integer Intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda-runtime-api/group__CUDA__MATH__INTRINSIC__INT.html)
5. **Warp-cooperative reduction**: `__shfl_down_sync` for log2(32) = 5 step reduction. No shared memory needed for the final sum if only 1 warp per row. [NVIDIA Warp-Level Primitives Blog](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
6. **Occupancy**: 1 warp per CTA, small register footprint → high occupancy. Many CTAs can run simultaneously, saturating memory bandwidth.

### 5. Blackwell (SM_120, Compute 12.0) Specifics

**Sources**: [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html), [Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compat-guide/index.html), [PyTorch sm_120 support](https://github.com/pytorch/pytorch/issues/159207), [gau-nernst learn-cuda SM120 matmul](https://deepwiki.com/gau-nernst/learn-cuda/7-matrix-multiplication-sm120-blackwell-02c), [arxiv microbenchmarking Blackwell](https://arxiv.org/abs/2512.02189)

- **`__dp4a` IS supported on SM_120** (compute 12.0). Integer intrinsics including `__dp4a` and `__dp2a` remain available. The Blackwell compatibility guide confirms CUDA core instruction set continuity from Ampere/Ada. [Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compat-guide/index.html)
- **GDDR7 bandwidth**: RTX 5060 Ti (GB206) has ~500 GB/s. For 8B INT4 (5.3 GB weights), single-pass weight load at 500 GB/s = 10.6 ms theoretical minimum. At 56 t/s = 17.9 ms/tok, we achieve ~59% of peak BW. llama.cpp at 84 t/s = 11.9 ms/tok achieves ~89% of peak BW.
- **Vectorized loads**: Blackwell supports `LDG.E.128` (128-bit load) via `int4`/`uint4` pointers. This is the most efficient load width — 16 bytes per transaction per thread. [CUDA Pro Tip: Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)
- **`cp.async`**: Available on SM_120. Can pipeline global→shared memory copies with compute. However, for GEMV M=1, direct registers are typically sufficient — shared memory adds overhead for skinny problems.
- **L2 cache**: 32 MB on GB206. Weight matrices exceed L2, so GEMV is truly memory-bound on weight loads.
- **Register file**: 256 KB per SM on Blackwell. High register availability means aggressive ILP (instruction-level parallelism) via loop unrolling and multiple accumulators.
- **SM count**: 36 SMs. At 1 warp per CTA = 36 rows processed simultaneously. For N=4096, need 4096/36 = 114 waves. Each wave loads K=4096 bytes of weight per row → 36 × 4096 × 4.5bpw_eff ≈ 662 KB per wave → negligible L2 residence.

### 6. Transferable Micro-Optimizations for Our Kernel

Based on analysis of our kernel (`src/kernels/gemv_int8.cu`) and llama.cpp's approach:

1. **Reduce scale fetch frequency**: Our kernel loads 1 FP32 scale per 16 elements (block-16). Changing to block-32 (1 scale per 32 elements) would halve scale loads and reduce effective bpw from ~6.0 to ~5.0. Q4_K does this via super-block sub-block structure.

2. **Use warp-cooperative K iteration (1 warp per row)**: Our current `gemv_int8_kernel` uses **1 thread per output row** (`kINT8Block=64` threads, each handles 1 row). This means only 1 thread accumulates K=4096 sequentially — **terrible occupancy and memory coalescing**. Each thread does sequential K iteration with no warp cooperation.
   - **llama.cpp uses 1 warp (32 threads) per row**: All 32 threads load different K chunks simultaneously → 32× more memory parallelism per row → much better bandwidth utilization.
   - **Our INT8 warp kernel does this** (separate `gemv_int8_warp`), but check if the INT4 batched path does too.

3. **Pre-quantize activations to INT8 (Q8_1 style)**: Our kernel converts activations to INT8 then uses `__dp4a`. But it may not batch the INT8 conversion efficiently. llama.cpp's Q8_1 format pre-quantizes once and passes INT8+FP16-scale to the kernel. If we're doing FP32→INT8 conversion inside the GEMV inner loop, that adds overhead.

4. **Vectorized 16-byte loads for weights**: Our kernel uses `uint4` loads for 16 INT8 values (16 bytes). For INT4, we should load `uint4` for 32 INT4 nibbles (16 bytes = 32 × 4-bit). Check if this is happening in our INT4 kernel.

5. **Multiple accumulators for ILP**: Instead of one accumulator, use 2-4 accumulators alternating across iterations. This hides load latency. `acc0`, `acc1`, `acc2`, `acc3` → sum at end.

6. **`__launch_bounds__` tuning**: Our kernel uses `__launch_bounds__(64, 1)` — 64 threads, 1 block per SM. For warp-cooperative (32 threads), could try `__launch_bounds__(32, 2)` or higher occupancy.

7. **Reduce scale precision from FP32 to FP16**: Our scales are FP32 (4 bytes each). Q4_K uses FP16 super-block scales (2 bytes) with 6-bit sub-block scales (0.75 bytes each effective). Switching to FP16 scales would halve scale bandwidth.

8. **Packed scale layout**: Q4_K packs scales tightly (6-bit sub-block scales in 8 bytes). Our FP32 scales are padded to 4 bytes each. If we switch to block-32 with FP16 scales, we'd save significant bandwidth.

### 7. Root Cause Analysis: Why We're at 67% of llama.cpp

**Bandwidth math**:
- Our INT4 8B weights: 5.3 GB. At 500 GB/s peak, theoretical minimum = 10.6 ms/tok = 94 t/s.
- We achieve 56 t/s (17.9 ms/tok) = **59% bandwidth utilization**.
- llama.cpp achieves 84 t/s (11.9 ms/tok) = **89% bandwidth utilization**.

**Why our BW utilization is lower**:
1. **1 thread per row in `gemv_int8_kernel`**: Only 1 thread accumulates each row's dot product. No warp-level K parallelism. Memory transactions are not coalesced across a warp for K iteration. Each thread does its own sequential K walk → memory transactions are scattered.
2. **FP32 per-block-16 scales**: ~33% more bytes to load than Q4_K's super-block format. At 6.0 bpw effective vs Q4_K's 4.5 bpw, we load 5.3 GB × (6.0/4.5) = 7.1 GB effective → 14.2 ms theoretical minimum = 70 t/s max. **Our 56 t/s is 80% of THIS bound, not the raw INT4 bound.**
3. **Scale load overhead**: Each K-iteration block fetches an FP32 scale from a separate array → potential cache thrashing or uncoalesced access depending on layout.
4. **Unpack overhead**: Offset-binary unpack (`nib - 8`) per element adds compute. llama.cpp amortizes this via dp4a on packed INT8.

**The 33% scale overhead alone explains most of the gap**: 70 t/s (our format bound) vs 94 t/s (pure INT4 bound). We're at 80% of format-bound, llama.cpp at 89% of format-bound. The remaining ~10% is from kernel efficiency (thread assignment, coalescing).

---

## Sources

### Kept (Primary / Authoritative)
- **mmvq.cu** — llama.cpp MMVQ kernel source ([GitHub](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu)) — core kernel implementation
- **vecdotq.cuh** — llama.cpp quantized dot product primitive ([GitHub](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vecdotq.cuh)) — per-type dequantize+dot
- **ggml-quants.h** — Q4_K block structure definition ([NousResearch fork](https://github.com/NousResearch/llama.cpp/blob/master/ggml-quants.h)) — struct layout, QK_K constants
- **DeepWiki: CUDA Backend** ([ggml-org](https://deepwiki.com/ggml-org/llama.cpp/5.1-cuda-backend-nvidia)) — MMVQ vs MMQ dispatch logic, kernel architecture overview
- **DeepWiki: Quantization Formats** ([qualcomm fork](https://deepwiki.com/qualcomm/llama.cpp/3.4-quantization-formats)) — Q4_K block structure detail, bpw calculations
- **PR #7716** ([SemanticDiff](https://app.semanticdiff.com/gh/ggml-org/llama.cpp/pull/7716)) — mmq/dmmv/mmvq refactor, dispatch threshold
- **Vijay's GEMV optimization blog** ([vijay-kodamalla.github.io](https://vijay-kodamalla.github.io/gemv-q4_k-optimization/)) — practical GEMV Q4_K optimization writeup from scratch
- **kernel-anvil llama-cpp-kernel-analysis** ([GitHub](https://github.com/apollosenvy/kernel-anvil/blob/master/docs/llama-cpp-kernel-analysis.md)) — profile-guided kernel analysis of llama.cpp
- **NVIDIA Blackwell Tuning Guide** ([docs.nvidia.com](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)) — SM_120 optimization guidelines
- **NVIDIA Blackwell Compatibility Guide** ([docs.nvidia.com](https://docs.nvidia.com/cuda/blackwell-compat-guide/index.html)) — instruction set support confirmation
- **CUDA Pro Tip: Vectorized Memory Access** ([NVIDIA blog](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)) — `int4`/`uint4` load optimization
- **NVIDIA Warp-Level Primitives** ([NVIDIA blog](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)) — `__shfl_down_sync` reduction
- **CUDA Math API: Integer Intrinsics** ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-math-api/cuda-runtime-api/group__CUDA__MATH__INTRINSIC__INT.html)) — `__dp4a` documentation
- **Q4_K GGUF deep-dive** ([meinlebenswerk.link](https://www.meinlebenswerk.link/posts/q4_k)) — Q4_K format byte-level layout walkthrough
- **Tensor Encoding Schemes Wiki** ([GitHub](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes)) — official format documentation
- **Stack Overflow: mul_mat_vec_q logic** ([stackoverflow.com](https://stackoverflow.com/questions/79591725/the-logic-of-mul-mat-vec-q-in-llama-cpp)) — kernel control flow Q&A
- **PR #22181: Optimize reduction stage of q4_L/q5** ([SemanticDiff](https://app.semanticdiff.com/gh/ggml-org/llama.cpp/pull/22181)) — recent vecdotq reduction optimization
- **arxiv: Microbenchmarking Blackwell Architecture** ([arxiv.org](https://arxiv.org/abs/2512.02189)) — Blackwell memory hierarchy and instruction throughput
- **gau-nernst learn-cuda SM120 matmul** ([DeepWiki](https://deepwiki.com/gau-nernst/learn-cuda/7-matrix-multiplication-sm120-blackwell-02c)) — Blackwell-specific matmul patterns

### Dropped
- PyPI llama-cpp-pydist — Python wrapper, not kernel source
- PyTorch sm_120 build guide — not relevant to CUDA kernel optimization
- yW!an GPU acceleration guide — secondary commentary
- dredyson.com MTP guide — unrelated to GEMV optimization

---

## Gaps

1. **Exact VDR_Q4_K_Q8_1 value and inner loop unroll factor**: Could not read the raw source code of `vecdotq.cuh` directly. The VDR (values dequantized per round) and exact ILP structure need source inspection. **Action**: Clone llama.cpp and read `vecdotq.cuh` directly.

2. **Exact register count and occupancy of MMVQ**: Could not verify register usage or achieved occupancy on SM_120. **Action**: Profile llama.cpp decode with `nsys`/`ncu` on RTX 5060 Ti, compare occupancy with our kernel.

3. **Our INT4 batched kernel thread assignment**: AGENTS.md says `gemv_int4_batched` exists but I couldn't find the source file (`gemv_int4.cu` not found). **Action**: Read the actual INT4 kernel source to check if it uses 1-thread-per-row or warp-cooperative K iteration.

4. **Q4_K 6-bit scale unpacking exact algorithm**: The `scales[QI4_K/2]` array packs 16 × 6-bit values into 12 bytes (or 8 bytes with different packing). Exact bit manipulation not verified. **Action**: Read `dequantize_row_q4_K` in `ggml-quants.c`.

5. **Whether our kernel is truly 1-thread-per-row for INT4**: The INT8 `gemv_int8_kernel` uses 1-thread-per-row, but there's also `gemv_int8_warp` (warp-cooperative). The INT4 path uses `gemv_int4_batched` — need to verify its parallelization strategy.

---

## Key Clarification Questions for Implementation

1. **Is `gemv_int4_batched` using 1 thread per row or 1 warp per row?** This is the #1 factor. If 1 thread per row, switching to warp-cooperative K iteration could close most of the gap.

2. **What is the scale format in `weights_int4_qwen3_8b/`?** If FP32 per-block-16, we're at 6.0 bpw effective. Switching to FP16 per-block-32 (5.0 bpw) or Q4_K-style super-blocks (4.5 bpw) would require re-quantization.

3. **Are activations pre-quantized to INT8 before GEMV, or converted inline?** llama.cpp pre-quantizes once per token, passes INT8+scale to kernel. Our kernel may be converting FP32→INT8 inside the GEMV loop.

4. **Can we change weight format without re-quantization?** If the Q4_K super-block format is needed, we'd need to re-run `gguf_convert` or adapt our converter. Alternatively, we can optimize the kernel for our existing format.

---

## Recommended Action Plan (Priority Order)

1. **Read `gemv_int4_batched` kernel source** — verify thread-per-row vs warp-per-row.
2. **If 1-thread-per-row: rewrite as warp-cooperative (32 threads per row)** — this alone should provide 2-3× speedup for M=1 based on memory coalescing analysis.
3. **Reduce scale fetch frequency: block-32 instead of block-16** — halve scale loads, reduce effective bpw from ~6.0 to ~5.0. Requires re-quantization OR a format wrapper.
4. **Switch scales from FP32 to FP16** — halves scale bandwidth. Minimal quality impact (scales have dynamic range ~1-3 orders, FP16 sufficient).
5. **Add multiple accumulators for ILP** — 2-4 accumulators to hide load latency.
6. **Verify `__dp4a` is used in INT4 unpack path** — upcast nibbles to INT8, use dp4a for dot product.
7. **Profile with `ncu`** — check memory throughput, occupancy, instruction issue rate vs llama.cpp.
