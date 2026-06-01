# Research: Blackwell CUDA Inference Project — Current State

## Summary

Blackwell custom CUDA INT8 inference on RTX 5060 Ti (GB206, SM_120a) achieves **326.8 t/s** with batched attention M=8 + CUDA Graph — 119% of llama.cpp Q4_K_M baseline (274.4 t/s) for Qwen3-1.7B. Library has **141 symbols**. Qwen3.5-9B hybrid Mamba decode achieves 45.7 t/s (64% of llama.cpp). FP4/INT4 paths are confirmed dead ends on GB206. All major tasks from prior sessions completed.

## Findings

### 1. Optimization Paths Explored

1. **Warp-cooperative GEMV** — `gemv_int8_warp` (production): 1 warp/row, shuffle-reduce, perfectly coalesced loads. **2.5–4.6× faster** than legacy `gemv_int8`. [`src/kernels/gemv_int8.cu:286-325`](src/kernels/gemv_int8.cu)
2. **INT8 batched attention M=8 + CUDA Graph** — `attention_decode_batched_gqa` + graph capture. **326.8 t/s** (best result). +9.8% from batched attn, +2.7% from graph. [`bench/decode_int8_batched_cgraph_attn.cu`](bench/decode_int8_batched_cgraph_attn.cu)
3. **WMMA INT8 GEMM (prefill)** — `gemm_int8_wmma_fast`: 32×32 tiles, 4 warps, direct FP32 accumulation. **13.0 TFLOPS** (3× vs old dp4a). [`src/kernels/gemm_int8_wmma_fast.cu`](src/kernels/gemm_int8_wmma_fast.cu)
4. **Block GEMV 4× unrolling** — `gemv_int8_unrolled`: +9-45% speedup, K-dependent. [`include/blackwell/kernels.h`](include/blackwell/kernels.h)
5. **FP16 scales** — `gemv_int8_fp16sc`: +2-13% from reduced scale bandwidth. [`include/blackwell/kernels.h`](include/blackwell/kernels.h)
6. **FP4 packed GEMV** — `gemv_fp4_warp`: 2 vals/byte E2M1, numerically unstable (~10^8 values). 247 t/s but outputs garbage. **Dead end on GB206**. [`src/kernels/gemv_int8.cu:354-420`](src/kernels/gemv_int8.cu)
7. **INT4 packed GEMV** — `gemv_int4_warp`: scalar unpack, 0.36× slower than INT8. **Dead end**. [`src/kernels/gemv_int8.cu:423-475`](src/kernels/gemv_int8.cu)
8. **Split-K GEMV** — `gemv_int8_splitk`: atomic partial sums, 779 GB/s kernel bandwidth. Marginal gain over warp GEMV for decode. [`src/kernels/gemv_int8.cu:915-970`](src/kernels/gemv_int8.cu)
9. **PDL (Programmatic Dependent Launch)** — `gemv_int8_pdl`: no benefit, kernels too short. [`include/blackwell/kernels.h`](include/blackwell/kernels.h)
10. **Speculative decode** — `speculative_decode_cgraph`: 190 t/s, 0% speedup (same total work). [`AGENTS.md §2`](AGENTS.md)
11. **GatedDeltaNet (Qwen3.5-9B)** — `gated_delta_net.cu`: conv1d update, recurrent SSM step, RMSNormGated. 45.7 t/s decode for 9B model. [`src/kernels/gated_delta_net.cu`](src/kernels/gated_delta_net.cu)
12. **GPU sampling** — `sample_gpu.cu`: softmax + top-k + cuRAND, replaces 607 KB host memcpy. [`src/kernels/sample_gpu.cu`](src/kernels/sample_gpu.cu)

### 2. Benchmark Results (Latest)

| Model | Config | Blackwell INT8 | llama.cpp | Ratio |
|-------|--------|----------------|-----------|-------|
| Qwen3-1.7B | Batched attn M=8 + CUDA Graph | **326.8 t/s** | 274.4 t/s (Q4_K_M) | **119%** ✅ |
| Qwen3-1.7B | CUDA Graph M=1 | 182.8 t/s | 274.4 t/s (Q4_K_M) | 67% |
| Qwen3-1.7B | text_generate (no graph) | ~127 t/s | 274.4 t/s (Q4_K_M) | 46% |
| Qwen3-1.7B | — | 183.6 t/s | 111.5 t/s (F16) | **+65%** ✅ |
| Qwen3-0.6B | CUDA Graph | 447.4 t/s | — | — |
| Qwen3-8B | CUDA Graph 28L | 57.4 t/s | 78.7 t/s (Q4_K_M) | 73% |
| Qwen3-8B | CUDA Graph 36L | 44.5 t/s | — | — |
| Qwen3.5-9B | Decode (hybrid Mamba) | 45.7 t/s | 71.4 t/s (Q4_K_M) | 64% |
| GEMM prefill | M=128 | **13.0 TFLOPS** | 4.3 TFLOPS (old) | 3× ✅ |

Source: [`benchmark-results.md`](benchmark-results.md), [`HANDOFF.md §2`](HANDOFF.md)

### 3. TODO/FIXME/HACK Comments Found

All TODOs are in **test stub files** only. No TODO/FIXME/HACK in production source or bench files.

| File | Line | Comment |
|------|------|---------|
| [`tests/test_gemm.cu`](tests/test_gemm.cu) | 19 | `// TODO(#2): replace with real FP4 GEMM correctness test once implemented.` |
| [`tests/test_gemm.cu`](tests/test_gemm.cu) | 33 | `// TODO(#6): real GEMV test once implemented — A=M=1, large N, known weights.` |
| [`tests/test_norm.cu`](tests/test_norm.cu) | 12 | `// TODO(#5): real RMSNorm test once implemented` |
| [`tests/test_norm.cu`](tests/test_norm.cu) | 17 | `// TODO(#5): SwiGLU test: out = silu(gate) * up` |
| [`tests/test_attention.cu`](tests/test_attention.cu) | 12 | `// TODO(#7): attention_fp4 test — Q/K/V known values, verify softmax math.` |
| [`tests/test_attention.cu`](tests/test_attention.cu) | 15 | `// TODO(#6): load_kv_cache_qkgv + update_kv_cache roundtrip test.` |
| [`tests/test_attention.cu`](tests/test_attention.cu) | 29 | `// TODO(#5): fused_rope test — cos/sin precomputed, apply rotation.` |
| [`tests/test_memory.cu`](tests/test_memory.cu) | 15 | `// TODO(#3): pack_fp4 test — roundtrip FP32 → FP4 → FP32.` |
| [`tests/test_memory.cu`](tests/test_memory.cu) | 24 | `// TODO(#3): coalesced_copy — verify alignment and throughput.` |

**All tests are stubs** — they `GTEST_SKIP()` with "not yet implemented". No actual test failures possible since all real tests are disabled. Production correctness verified via bench harnesses (`verify_gemm`, `validate_pipeline`, `text_generate`).

### 4. Qwen3.5-9B Decode Path

[`bench/decode_qwen35_9b.cu`](bench/decode_qwen35_9b.cu) — 32 layers, hybrid architecture:

- **24 linear_attention layers** (indices 0,1,2,4,5,6,8,9,10,...): GatedDeltaNet SSM
  - `in_proj_qkv` [8192×4096] → conv1d (kernel=4) → split Q/K/V
  - `in_proj_a` [32×4096] → alpha (decay), `in_proj_b` [32×4096] → beta (gate)
  - `gated_delta_recurrent_step`: broadcast NK→NV, SSM update [NV×HD×HD state]
  - `gated_delta_rmsnorm_gated`: norm(output) × silu(z)
  - `out_proj` [4096×4096]

- **8 full_attention layers** (indices 3,7,11,15,19,23,27,31): Standard GQA
  - H=4096, head_dim=256, NQ=16, NKV=4
  - `head_norm_k` (custom inline kernel) for Q/K norms
  - `rope_k` (custom inline kernel) for rotary embeddings
  - `attention_decode_gqa` for decode attention
  - `update_kv_cache` for KV cache writes

- **MLP** (all 32 layers): gate [4096→12288] + up [4096→12288] → SwiGLU → down [12288→4096]

- Per layer: ~12 kernel calls (linear) or ~14 kernel calls (full attn)
- **45.7 t/s** decode throughput, weight-bound (INT8 reads 7.9 GB/token)

### 5. text_generate — Qwen3-1.7B Only

[`bench/text_generate.cu`](bench/text_generate.cu) works for **Qwen3-1.7B only** (28 layers, H=2048):
- End-to-end: BPE tokenize → embedding lookup → 28L decode → GPU sample → decode token
- ~127 t/s greedy, ~119 t/s with sampling
- No CUDA Graph (sequential per-kernel launch)
- GPU sampler: `sample_gpu` handles argmax/temperature/top-k entirely on device
- Verified: `"The capital of France is"` → `"Paris"` ✅

**text_generate does NOT work with Qwen3.5-9B** — hardcoded constants for Qwen3-1.7B (H=2048, nqh=16, nkv=8, NL=28, V=151936). Qwen3.5-9B needs H=4096, 32 layers, hybrid Mamba/GQA, different weight directory, different tokenizer (vocab=248320). HANDOFF.md notes "text_generate for Qwen3.5-9B" as a future task.

### 6. Architecture Decisions

1. **INT8 warp GEMV over FP4/INT4** — GB206 has no usable sub-byte tensor cores for GEMV. INT8 + `__dp4a` SIMD is optimal.
2. **Per-block-16 quantization** — weights and activations use 16-element blocks with absmax scales. K must be %16==0.
3. **Transposed weight layout** — W_t [N×K] for coalesced warp reads. All weights pre-transposed during quantization.
4. **CUDA Graph for batched decode** — captures all kernels per step, eliminates launch overhead. Not beneficial for Qwen3.5-9B (480+ kernels, graph overhead exceeds savings).
5. **Fused kernels where profitable** — `fused_rmsnorm_quant_int8` (2→1), `fused_gate_up_gemv` (2→1). Not aggressively fused — kernel launch overhead is small relative to compute.
6. **GPU sampling** — eliminates 607 KB host memcpy per token. cuRAND-based stochastic sampling on device.

### 7. Known Issues

| Issue | Severity | Status |
|-------|----------|--------|
| hashcat auto-restarts on GPU-0 | ⚠️ -45% throughput | Must `killall hashcat` before measurement |
| INT8 vs Q4_K_M gap (M=1) | ℹ️ Hardware limit | INT8 reads 4× more data than Q4_K_M |
| FP4/INT4 numerically unstable | ❌ Dead end | E2M1 can't use dp4a, scalar path too slow |
| text_generate no CUDA Graph | ℹ️ Low priority | ~127 t/s, graph would add ~1-6% |
| Tests are all stubs | ℹ️ No coverage | All GTEST_SKIP, correctness via bench harnesses |
| Qwen3.5-9B text_generate missing | ℹ️ Future work | Needs tokenizer + hybrid model integration |
| decode_int8_cgraph warmup mismatch | ℹ️ Pre-existing | warmup stream 0 vs graph_stream, 182 t/s works fine |

## Sources

- **Kept**: [`AGENTS.md`](AGENTS.md) — canonical kernel inventory, build commands, findings table, constraints
- **Kept**: [`HANDOFF.md`](HANDOFF.md) — session 28 state, 141 symbols, Qwen3.5-9B 45.7 t/s, all tasks done
- **Kept**: [`benchmark-results.md`](benchmark-results.md) — detailed throughput numbers vs llama.cpp baselines
- **Kept**: [`context.md`](context.md) — symbol inventory, architecture, key code patterns
- **Kept**: [`README.md`](README.md) — project overview, build instructions, constraints
- **Kept**: [`bench/decode_qwen35_9b.cu`](bench/decode_qwen35_9b.cu) — Qwen3.5-9B hybrid Mamba decode path
- **Kept**: [`bench/text_generate.cu`](bench/text_generate.cu) — end-to-end Qwen3-1.7B text generation
- **Kept**: [`src/kernels/gemv_int8.cu`](src/kernels/gemv_int8.cu) — production GEMV kernels (warp, batched, splitk, FP4, INT4)
- **Kept**: [`src/kernels/gated_delta_net.cu`](src/kernels/gated_delta_net.cu) — GatedDeltaNet SSM kernels
- **Kept**: [`src/kernels/decode.cu`](src/kernels/decode.cu) — KV cache + decode attention (v4 for head_dim>128)
- **Kept**: [`src/kernels/sample_gpu.cu`](src/kernels/sample_gpu.cu) — GPU logit sampling
- **Kept**: [`include/blackwell/kernels.h`](include/blackwell/kernels.h) — full public API, 141 symbols
- **Kept**: [`CMakeLists.txt`](CMakeLists.txt) — build config, 26 kernel source files
- **Kept**: [`tests/test_*.cu`](tests/) — all stub tests with TODO comments

## Gaps

1. **No real unit tests** — All 9 test cases are `GTEST_SKIP()` stubs. Correctness verified only through bench harnesses. Risk: regressions in kernel math won't be caught automatically.
2. **text_generate for Qwen3.5-9B** — Not implemented. Needs: tokenizer for 248320 vocab, model constants (H=4096, 32L, hybrid), weight loading from `weights_int8_qwen35_9b/`, integration of both linear_attn and full_attn paths.
3. **text_generate CUDA Graph** — Not implemented for Qwen3-1.7B either. Potential +1-6% gain.
4. **Memory profiling** — No Nsight Compute profiling data in repo. Bandwidth utilization (260 GB/s effective vs 500 GB/s peak) not fully explained.
5. **Quantization quality** — No perplexity/accuracy benchmarks. INT8 block-16 quantization quality vs FP16 baseline not measured.
6. **Prefill path** — `gemm_int8_wmma_fast` achieves 13.0 TFLOPS but only 26% utilization. Room for improvement not explored.
