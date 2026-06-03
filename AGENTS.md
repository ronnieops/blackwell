# AGENTS.md - blackwell

Custom CUDA kernels for INT8 + INT4 LLM inference on RTX 5060 Ti (Blackwell, GB206).

---

## 1. Mission

INT4 decode throughput vs llama.cpp Q4_K_M baseline.

**INT4 1.7B batched attention (M=1): 261.7 t/s (89% of Q4_K_M 293.4).** Fixed gemv_int4_batched grid bug (`dim3 grid(N/32,M)` → `N`). 177 library symbols.

**INT4 1.7B batched attention (M=8): 3586.4 t/s (1222% of Q4_K_M).** Super-linear scaling via batched GEMV + batched attention.

**INT4 8B batched attention (M=1): 342.9 t/s (415% of Q4_K_M 82.56).** 8B numbers pre-date grid bug fix — UNVERIFIED with corrected kernel.

**INT4 8B batched attention (M=8): 5640.3 t/s (6831% of Q4_K_M).** UNVERIFIED with corrected kernel.

### llama.cpp comparison (build 9442, CUDA 13.3, RTX 5060 Ti)

| Model | Quant | GPU | tg128 | vs Our INT4 |
|-------|-------|-----|-------|-------------|
| Qwen3-8B | Q4_K_M | 100% ✅ | 82.66 | **4.15×** |
| Qwen3.6-27B | Q3_K_M | 100% ✅ | 26.73 | — |
| Qwen3.6-27B-MTP | Q3_K_M | 100% ✅ | 26.75 | — |
| Qwen3.6-35B-A3B | Q2_K_XL | 100% ✅ | 118.70 | — |
| Qwen3.6-35B-A3B-MTP | Q2_K_XL | 100% ✅ | 118.47 | — |

27B Q4_K_M (16-17 GiB) exceeds 16 GB VRAM. Full-GPU numbers use Q3_K_M (fits 12.6 GiB).
35B-A3B MoE (~3B active params) beats 27B dense by 4.4× at decode despite larger total size.
NVFP4 models (safetensors) downloaded but can't benchmark with llama-bench.

**Docker production server ready**: `Dockerfile` + `server/server.py`. 324 t/s beats Q4_K_M by 10%.

---

## 2. Active State

**Stack**: CUDA 13.3, SM_120a, CMake, C++17
**Target**: RTX 5060 Ti 16 GB, compute 12.0, 36 SMs, ~500 GB/s GDDR7
**Nvcc path**: `/usr/local/cuda-13.3/bin/nvcc`
**Library**: 177 symbols in `build/libblackwell_kernels.a`

**WARNING**: `hashcat` runs persistently on this GPU (PID changes, auto-restarts). Uses 3740MiB VRAM at 95%+ util. Kills benchmark throughput ~45%. `kill all hashcat` before any measurement.

**Production kernels (INT8 path)**:
- `gemv_fp32_int8_per_row_warp` — Warp-cooperative FP32×INT8 GEMV (1 warp/row, shuffle reduce)
- `gemv_int8` — Legacy INT8 GEMV, `__dp4a` SIMD, 775 GB/s (kernel), 260 GB/s (effective). **Superseded by gemv_int8_warp**
- `gemv_int8_batched` — batched M=1-8, sweet spot M=3-4 (1.4× speedup)
- `gemv_int8_splitk` — split-K (K_splits=4), 779 GB/s
- `pack_int8` / `quantize_int8` — FP32 → INT8 quant with block scales
- `transpose_int8_weights` — W (K×N) → W_t (N×K) + scales
- `fused_rmsnorm_quant_int8` — RMSNorm + INT8 quant (1 kernel)
- `gemv_int8_gate_up` — fused gate+up MLP projection
- `gemv_int8_qkv` — fused Q/K/V projection (3 kernels → 1)
- `fused_unpack_fp4_quant` — FP4 unpack + INT8 quant (fused pipeline)
- `fused_residual_norm` — residual add + RMSNorm + INT8 quant (fused pipeline)
- `fused_swiglu_quant` — SwiGLU + INT8 quant (fused pipeline)
- `fused_rmsnorm_pack` — RMSNorm + FP4 pack (fused pipeline)
- `fused_rmsnorm` — single-block warp-reduced RMSNorm
- `apply_swiglu` — silu(gate) × up, elementwise
- `fused_rope` / `fused_rope_decode` — in-place rotation, smem cos/sin cache
- `attention_decode_gqa` — GQA decode attention (M=1 path)
- `attention_decode_batched_gqa` — Batched GQA decode attention (M sequences in parallel)
- `update_kv_cache` — KV cache write with per-layer offset
- `gemm_int8_wmma` — WMMA INT8 GEMM (prefill, 3.8× dp4a)
- `gemm_int8_wmma_fast` — Optimized WMMA (32×32 tiles, 4 warps, 4.3-5.0K GFLOPS)
- `gemm_int8_mma` — Stub (returns cudaErrorNotSupported)
- `sample_gpu` — GPU softmax + top-k + cuRAND weighted sampling (replaces 607KB memcpy)
- `sample_argmax_gpu` — GPU argmax (4-byte output, greedy decode)
- `gated_delta_conv1d_update` — 1D depthwise conv + SiLU for GatedDeltaNet (Qwen3.5-9B)
- `gated_delta_recurrent_step` — SSM recurrent step with QK broadcast (NK→NV heads)
- `gated_delta_rmsnorm_gated` — Fused RMSNormGated (norm × silu gate)
- `attention_decode_kernel_v4` — Decode attention for head_dim>128 (Qwen3.5-9B full attn)

**Optimized GEMV kernels**:
- `gemv_int8_unrolled` — Block-cooperative with 4× unrolling (+9-45%)
- `gemv_int8_warp_unrolled` — Warp-cooperative with 4× unrolling (no benefit)
- `gemv_int8_fp16sc` — FP16 scales (+2-13%)
- `gemv_int8_pdl` — PDL kernel launch (no benefit, kernels too short)

**Research kernels (FP4 path)**:
- `gemv_fp4_nv` / `gemv_fp4_nv_opt` — NVF4 scalar GEMV, UE4M3 scales, 98 GB/s (correct, not competitive)
- `pack_fp4` / `unpack_fp4` — FP4 E2M1 quant/dequant
- `gemm_fp4_block_scaled` — FP4 GEMM prefill
- `gemm_int8` / `gemm_int8_dp4a` — INT8 GEMM prefill (M>1, per-block scales, 4×4 tiling)
- `gemv_fp4_warp` — Packed FP4 warp GEMV (2 vals/byte, E2M1, 29 regs)
- `gemv_fp32_fp4_warp` — FP32×packed FP4 warp GEMV (47 regs)
- `decode_fp4_cgraph.cu` — Full 28L FP4 pipeline benchmark (CUDA Graph, 247 t/s, numerically unstable)
- `gemv_int4_warp` — INT4 warp GEMV, scalar unpack
- `gemv_int4_batched` — INT4 batched GEMV (M=1-8, N per bandwidth)
- `transpose_int4_weights` — W (K×N/2) → W_t (N×K/2), scales transposed
- `quantize_int4` — FP32 → packed INT4 (block-16, absmax/7, nibble-pack)
- `unpack_int4_fp32` — packed INT4 → FP32
- `fused_residual_norm_int4` — fused residual add + RMSNorm + INT4 quant (3→1 kernel)
- `fused_residual_norm_int4_fp32out` — same + FP32 normalized output (for next layer)
- `fused_swiglu_quant_int4` — fused SwiGLU + INT4 quant (2→1)

**FP4 packed: numerically unstable** (247 vs 181 t/s INT8). Throughput competitive but outputs garbage. E2M1 nibble→float per-element conversion can't use __dp4a SIMD.

**Deprecated / DO NOT USE**:
- `gemv_int8_from_fp4` — 2.8× slower than baseline
- `phase_a.cu` — depends on unimplemented symbols (`gemv_fp4_splitk`, `gemv_fp4_v3`, `gemv_fp4_batched`)
- NVF4 tensor core MMA — scale factor layout mismatch for GEMV

---

## 3. Build & Run

### Build
```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### Benchmark
```bash
killall hashcat 2>/dev/null  # MUST DO BEFORE ANY MEASUREMENT
./bench/decode_int8_cgraph 28                       # INT8 1.7B: 181.5 t/s (14 kernels/layer)
./bench/decode_int8_batched_cgraph_attn 28 8        # INT8 1.7B M=8: 324.3 total t/s (111% of Q4_K_M)
./bench/decode_int4_batched_attn 28 1               # INT4 1.7B: ~262 t/s (89% of Q4_K_M)
./bench/decode_int4_batched_attn 28 8               # INT4 1.7B M=8: ~3586 t/s (1222% of Q4_K_M)
./bench/text_generate_int4 "The capital of France is" 30  # INT4 text generation (session 37)
./bench/text_generate "The capital of France is" 30 # Correctness
```

### Model
`/mnt/data/ai/hf/qwen3-1.7b-base/` — Qwen3-1.7B, 28 layers, safetensors, 3.3 GB

### INT4 weights
`weights_int4_qwen3_1.7b/` — 1.3 GB (62% of INT8 2.1 GB). 28 layers × 7 weights + embed_tokens.

## 4. Key Findings

| Finding | Value | Notes |
|---------|-------|-------|
| **INT4 batched-attn M=1 (1.7B)** | **261.7 t/s** | **89%** of Q4_K_M (293.4). Post-fix: grid bug corrected. 17 kernels/layer |
| **INT4 batched-attn M=8 (1.7B)** | **3586.4 t/s** | **1222%** of Q4_K_M. Super-linear scaling via batched GEMV |
| **INT4 batched-attn M=1 (8B)** | **342.9 t/s (UNVERIFIED)** | Pre-dates grid fix. 8B bench needs re-run |
| **INT4 batched-attn M=2 (8B)** | **1046.4 t/s (UNVERIFIED)** | Pre-dates grid fix |
| **INT4 batched-attn M=4 (8B)** | **2614.8 t/s (UNVERIFIED)** | Pre-dates grid fix |
| **INT4 batched-attn M=8 (8B)** | **5640.3 t/s (UNVERIFIED)** | Pre-dates grid fix |
| INT4 per-layer time (1.7B, M=1) | 0.136 ms | 28 layers = 3.82 ms total. 17 kernels/layer (was 0.058 from 1/32 work) |
| INT4 per-layer time (8B) | 0.081 ms (UNVERIFIED) | Also affected by grid bug |
| Warp GEMV speedup | **2.5–4.6×** vs old gemv_int8 | Coalesced loads (1 warp/row) |
| INT8 fused (M=1) | **181.5 t/s** | 14 kernels/layer (was 20), 30% launch reduction |
| INT8 batched-attn M=8 CUDA Graph | **324.3 total t/s** | **111%** of Q4_K_M |
| INT4 weight compression | **1.3 GB** vs INT8 **2.1 GB** | 62% of INT8 |
| llama.cpp Q4_K_M FA=on | **293.4 t/s** | Qwen3-1.7B, build b9442, CUDA 13.3 |
| llama.cpp Q4_K_M FA=off | **274.1 t/s** | Qwen3-1.7B |
| llama.cpp F16 FA=on | **114.3 t/s** | Qwen3-1.7B |
| llama.cpp Q4_K_M FA=on (8B) | **82.56 t/s** | Qwen3-8B, build b9442 |
| llama.cpp Q4_K_M (3.5-9B) | 71.4 t/s | Qwen3.5-9B MoE |
| **llama.cpp Q3_K_M 27B (full GPU)** | **26.73 t/s** | Qwen3.6-27B, fits 16 GB VRAM |
| **llama.cpp Q3_K_M 27B-MTP (full GPU)** | **26.75 t/s** | Qwen3.6-27B-MTP, 66 layers |
| **llama.cpp Q2_K_XL 35B-A3B (full GPU)** | **118.70 t/s** | Qwen3.6-35B-A3B MoE, ~3B active |
| **llama.cpp Q2_K_XL 35B-A3B-MTP (full GPU)** | **118.47 t/s** | Qwen3.6-35B-A3B-MTP MoE |
| INT8 effective BW | 260 GB/s | Weight-bound (L2 cache miss) |
| GEMM prefill (after c_frag fix) | **13.0 TFLOPS** | 26% utilization |
| Pipeline SNR | **13.9 dB** | Constant across 28 layers, no compounding |
| CUDA Graph speedup | ~1-6% | Model-size dependent |
| hashcat interference | -45% throughput | Kills GPU-0 ~every 60s |
| INT4/FP4 sub-byte GEMV | ❌ Not competitive | ~35 inst/byte unpack vs 0.31 inst/byte dp4a |
| **INT4 text_generate quality** | **Garbled after 28L** | 4-bit symmetric quant noise compounds across layers. Needs asymmetric/per-channel |
| **gemv_int4_batched grid bug** | **N/32 → N** | `dim3 grid(N/32,M)` only computed 1/32 rows. All pre-session-37 INT4 benchmarks invalid |
| **INT4 weight corruption** | **Scales ~1e-23** | `quantize_generic.py` produced corrupt scales. Re-quantized. Re-quantize after batch runs |
| **gemv_int4 sign-extension bug** | **nib-8 vs sign-extend** | Both gemv_int4_warp and gemv_int4_batched used wrong `if(lo>7)lo-=16`. Fixed: `nib-8` |
---

---

## 5. Constraints

- `CUDACXX` env var must be set before `project()` in CMakeLists.txt
- `compute_120a` required (not `compute_120`) — FP4 block-scale MMA needs 12Xa
- `namespace wmma = nvcuda::wmma` (alias, NOT `using wmma =`)
- `sizeof(__nv_fp4_e2m1)` = 1 byte (not 0.5)
- All weight matrices exceed L2 cache (32 MB) — architectural limit for single-token decode
- System ptxas may be old — ensure CUDA 13.3 in PATH
- Warp kernel requires K%16==0 and N%16==0 (inherited from block-16 quantization)
- Warp stride-32 loop: K/16 must divide evenly for balanced work (true for K=2048, 6144)
- `gemv_int8_warp` is the production path — 22 bench files migrated (164 call sites). Some legacy files remain.
- hashcat runs on GPU-0 — kills ~45% throughput. Must `killall hashcat` before measurement
- `gemm_int8_wmma` per-block dequant exact match vs dp4a (0.000 max diff)

---

## 6. Bug History

### vector_add_fp32_kernel (2026-05-28) — FIXED
`src/kernels/norm.cu`: reversed `=` in float4 path wrote uninitialized data TO input buffer.
Fix: `float4 va = ((float4*)a)[idx];` (load, not store).

### RoPE frequency (2026-05-29) — FIXED
All 5 bench files: `idxf = i2/hd` doubled exponent → 2× rotation speed.
Fix: `theta = pos * powf(rope_theta, -2.0f * d / head_dim);`

### head_norm cross-warp (2026-05-29) — FIXED
All 5 bench files: `__shfl_xor_sync` with off=64/32 no-ops on 32-lane warps → 1/4 sums.
Fix: smem[4] warp partials → shuffle-reduce across 4 warps.

### inference_server syntax (2026-05-29) — FIXED
Stray `}` after head_norm_kernel closing brace. Deleted.

### INT4 fused_residual_norm_int4_fp32out buffer aliasing (2026-06-02) — FIXED
`fused_residual_norm_int4_fp32out` wrote INT4 packed output + FP32 normalized output
to the same buffer (`d_x32`). INT4 output (bytes 0-1023 = 12.5% of buffer) corrupted the
first 256 FP32 elements used as next layer's input. Fix: INT4 output → `d_x_i4`, FP32 → `d_x32`.
Affected: `decode_int4_cgraph.cu`, `decode_int4_batched.cu`, `bench/decode_int4_batched_attn.cu`.

### Stale residual in INT4 decode benchmarks (2026-06-02) — FIXED
All INT4 benchmarks copied `d_res = d_x32` once at layer 0 but reused `d_res` for all
layers 0-27. Fix: pass current `d_x32` (updated per layer) as residual to fused kernels.

### fused_residual_norm only processes first 2048 elements (2026-06-02) — FIXED
`src/kernels/fused_residual_norm_int4.cu`: kernel launch used 256 threads × 8 REPT = 2048 elements,
but Qwen3-8B (H=4096) needs 4096. Second half of hidden state (indices 2048-4095) retained stale
initial value 1.0, propagating through all 36 layers — 50% of hidden state wrong per layer.
Only affected 8B (H=4096). 1.7B (H=2048) was correct.
Fix: `kFusedThreads=256→512`, `warp_sums[8]→[16]`. Both `fused_residual_norm_int4` and
`fused_residual_norm_int4_fp32out` variants fixed.
Discovered via `verify_int4_tail_check` correctness tool. Pre-fix 8B benchmarks reported
~2% inflated throughput (kernel did ½ work).
Warmup loop ran single layer 0, leaving layers 1-27 cold in the timing loop.
Fix: iterate over all num_layers in warmup.

### gemv_int4_batched grid bug (2026-06-02) — FIXED
`src/kernels/gemv_int4_batched.cu`: `dim3 grid(N / 32, M)` only launched N/32 output row blocks,
computing only 1/32 of output rows. Remaining 31/32 rows stayed zero. Undetected because INT4
benchmark measured throughput (half the work → double the speed). Fix: `dim3 grid(N, M)`.
**ALL INT4 benchmarks pre-session-37 invalid.** Corrected: 1.7B M=1 261.7 t/s (was 610.2).
Affected: `gemv_int4_batched_kernel`, all INT4 benchmarks relying on it.

### INT4 nibble sign-extension bug (2026-06-02) — FIXED
`src/kernels/gemv_int8.cu` and `src/kernels/gemv_int4_batched.cu`: `int4_byte_to_floats` used
3-bit two's complement sign-extend (`if(lo>7)lo-=16`). INT4 stores offset-binary nibbles (`q+8`
for [-8..7]) where nib=0→q=-8, nib=8→q=0, nib=15→q=7. Sign-extend produced inverted values:
nib=8 (q=0) → -8, nib=15 (q=7) → -1. Fix: `nib - 8` for both lo and hi.
Affected: `gemv_int4_warp_kernel`, `gemv_int4_batched_kernel`, `dequant_embed_row`.

### INT4 weight corruption (2026-06-02) — FIXED
`weights_int4_qwen3_1.7b/` scales all ~1e-23 (essentially zero). Root cause: `quantize_generic.py`
read BF16 safetensors but `read_tensor` has `f.seek(0)` call that resets file position after
reading header length, corrupting tensor data offset. Re-running quantization produced correct
scales (~0.01). Fix: re-quantize all INT4 weights. Pre-session-37 outputs from these weights
produced garbage text. Re-quantize after any batch run that reprocesses weights.

---

## 7. Known Issues

1. **hashcat runs persistently** on GPU-0 (PID changes, auto-restarts). Uses 3740MiB VRAM at 95%+ util. Kills benchmark throughput ~45%. `killall hashcat` before any measurement — 60s window before respawn
2. **22 bench files migrated to `gemv_int8_warp`** (164 call sites). Production path: decode_int8_cgraph and decode_full_int8.
3. **text_generate repetition** — Greedy decode repeats (normal for argmax). Use -t 0.8 or -k 40 for better output.
4. **GEMM prefill correctness verified** — test_wmma PASS, verify_gemm PASS, decode_prefill 3× speedup committed.
5. **text_generate head_norm bug** — ✅ **FIXED**. No FAIL head_norm. Uses gemv_int8_warp (not old per_row).
6. **FP4 packed numerically unstable** — 247 vs 181 t/s. Throughput competitive but outputs garbage (~10^8 values). E2M1 nibble→float overhead can't use __dp4a SIMD.
7. **L2 cache hint targets wrong stream** — FIXED (commit f55a705). Targets graph_stream.
8. **Speculative decode CUDA Graph crash** — static cudaMalloc in decode.cu needs warm-up first
9. **Docker/API packaging** — ✅ Done (session 26)
10. **WMMA dequant correct** — `gemm_int8_wmma_fast` per-block dequant confirmed correct (advisor analysis). Per-iteration SMEM load correctly indexes K-block. AGENTS.md §10 note was wrong.
11. **GPU sampling** — ✅ Done (session 28). `sample_gpu` handles argmax, temperature, and top-k on GPU. No host fallback needed.
13. **CUDA Graph capture (M=1) re-evaluated (session 33)** — Original session-30 diagnosis: `cudaFuncSetAttribute` in `attention_decode_gqa` + H2D pinned `cudaMemcpyAsync` (seq_pos) conflict with `cudaStreamCaptureModeGlobal`. Illegal memory access during capture.
    - **llama.cpp analysis**: Uses `cudaStreamCaptureModeRelaxed` (not `Global`). Their `CUDA_SET_SHARED_MEMORY_LIMIT` macro calls `cudaFuncSetAttribute` once before capture via static guard — works in Relaxed mode.
    - **Tried**: Full warm-up (14 kernels/layer × 1 layer) + `cudaStreamCaptureModeRelaxed` on `decode_int8_cgraph.cu`. Warm-up succeeded (static allocs + smem attr all triggered). Capture still failed.
    - **Root cause**: `attention_decode_gqa` and `update_kv_cache` wrappers in `src/kernels/decode.cu` call `cudaMemcpyAsync (H2D, pinned)` for `seq_pos` on the capturing stream. This is illegal in ANY capture mode — not a mode-selection issue.
    - **Fix path**: Need graph-safe wrapper variants that skip H2D copy (assume seq_pos pre-set via direct device pointer write before capture) or use `cudaGraphKernelNodeParams` with direct device memory. Per-kernel fused path (181.5 t/s) remains production target for M=1.
14. **CUDA Graph (M=8 batched) works** — `decode_int8_batched_cgraph_attn` captures 28 layers × 8 sequences = 224 kernel launches successfully. 326.8 t/s (111% of Q4_K_M FA=on, 119% of FA=off).
15. **Fused pack+GEMV kernels (session 31)** — `fused_pack_gemv_o` + `fused_swiglu_gemv` numerically correct but ~20% slower (144.6 t/s). Two-phase kernels (quant→sync→GEMV) add quantization overhead to GEMV critical path. Not used in production benchmark. Archives correct kernels (157 symbols).
16. **gemv_int8_batched is SLOWER than gemv_int8_warp** (session 32) — Isolated test: serial warp GEMV is 1.5-2.7× faster than batched GEMV for all GEMV sizes (N=1024-6144). Reason: serial has higher occupancy (M×N blocks vs N blocks). However, in CUDA Graph context, batched MLP is faster (fewer graph nodes = less overhead). Production benchmark uses batched MLP (gate/up/down) + serial Q/K/V + batched attention. M=8 CUDA Graph: 323 t/s.
17. **L2 persisting cache harmful for large weights** (session 32) — Pinning 12.6 MB gate weights in L2 persisting cache caused 28% regression. Evicts other cached data (up/down weights, attention data). d_rn (8 KB) persisting is neutral. Removed L2 persisting for MLP weights.
18. **Speculative decode infeasible** (session 33) — Batched verify (24.7 ms/seq) is **4.5× slower per-seq** than sequential (5.52 ms/seq). Draft must be 4.5× faster to break even. Even tiny 50M draft only yields ~92 t/s. Self-speculation (skip layers) won't work — lm_head needs all 28 layers. No early-exit head exists. Abandoned.
20. **llama.cpp code audit** (session 33) — Deep analysis of `ggml/src/ggml-cuda/` for opportunities:
    - **M=1 CUDA Graph blocked by H2D copies**: `attention_decode_gqa`/`update_kv_cache` call `cudaMemcpyAsync (H2D, pinned)` for `seq_pos`. Illegal on capturing stream in ALL modes. Fix requires graph-safe wrappers.
    - **FP4 tensor cores (BLACKWELL_MMA_AVAILABLE)**: llama.cpp `vec_dot_fp4_fp4_mma` uses `mma_block_scaled_fp4` with 16×8 tiles for MMQ (batched, M≥64). Useless for M=1 GEMV decode. Our `gemm_fp4_block_scaled` already implements FP4 tensor core GEMM for prefill.
    - **PDL (Programmatic Dependent Launch)**: Hopper+ device-side primitives (`ggml_cuda_pdl_sync`/`ggml_cuda_pdl_lc`). Blackwell supports it but PDL eliminates inter-kernel gaps for dense kernel chains — our M=1 pipeline has 14 kernels/layer (trivial launch overhead, ~3% of total time). Not worth the complexity.
    - **MMVQ_MAX_BATCH_SIZE=8**: llama.cpp caps quantized batch at 8. Validates our M=8 register pressure limit.
    - **All 34 kernel source files** in `ggml/src/ggml-cuda/` — well-organized, template-instance pattern for specialized kernels.
21. **INT4 text_generate quality insufficient** (session 37) — Pipeline structurally correct but 4-bit symmetric quantization noise (~14% per-value error) compounds across 28 layers. First token diverges from INT8 greedy. Needs asymmetric per-channel quantization or fine-tuning to be useful.
22. **INT4 weights must be re-quantified after batch runs** (session 37) — `quantize_generic.py` `read_tensor()` function has `f.seek(0)` call that corrupts offset when called after a prior `f.seek(8)`. Causes ~1e-23 scale values. Running the script from scratch produces correct output.

---

## 8. Anti-Hallucination Rules

- **Do not invent APIs, files, commands, env vars, or requirements.** Read the actual header/source before calling a function.
- **Prefer repo evidence over assumptions.** If you need a function signature, read `include/blackwell/kernels.h`.
- **Mark unknowns explicitly.** "Not checked" or "unknown behavior" in comments.
- **Never overwrite higher-priority instructions.**
- **Preserve user intent and existing project conventions.**

---

## 9. Development Loop

```
observe → plan → edit → build → test → reflect → update AGENTS.md only if useful
```

Build: `CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake --build build --parallel`
Test: `./bench/decode_int8_cgraph 28` (CUDA Graph production path), `./bench/text_generate ...` (correctness)
Verify: `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l` (expect 177)

---

## 10. Seed Principles

1. Smallest correct change. One kernel, one fix, one test.
2. Verify before broad edits.
3. Prefer repo evidence. Read code before assuming.
4. No churn.
5. Kernels first, framework later.

---

## 11. Session 34/35 — INT4 Batched Decode (Q4 Complete)

**Objective**: Close M=1 bandwidth gap. INT4 halves weight reads → target ~250-290 t/s (85-99% of Q4_K_M).

**Result**: **612.8 t/s (209% of Q4_K_M)** — exceeds target by 2×.

### What was built

| Component | Status | Result |
|-----------|--------|--------|
| `gemv_int4_warp` kernel | ✅ Built | 60 regs, scalar unpack (no __dp4a) |
| `gemv_int4_batched` | ✅ Built | M=1-8, N per bandwidth |
| `transpose_int4_weights` | ✅ Built | W (K×N/2) → W_t (N×K/2) |
| `quantize_int4` | ✅ Built | FP32 → packed INT4, block-16 |
| `unpack_int4_fp32` | ✅ Built | packed INT4 → FP32 |
| `fused_residual_norm_int4` | ✅ Built | residual+rmsnorm+quant (3→1 kernel) |
| `fused_residual_norm_int4_fp32out` | ✅ Built | same + FP32 normalized output |
| `fused_swiglu_quant_int4` | ✅ Built | SwiGLU + INT4 quant (2→1 kernel) |
| `weights_int4_qwen3_1.7b/` | ✅ Generated | 1.3 GB (62% of INT8 2.1 GB), 394 files |
| `scripts/quantize_generic.py` | ✅ Extended | INT4 support via `--format int4` |
| `bench/decode_int4_batched_attn` | ✅ Benchmark | **612.8 t/s** M=1 (209% Q4_K_M) |

### Benchmark results — decode_int4_batched_attn

| M | Per-seq t/s | vs Q4_K_M | Scaling |
|---|------------|-----------|---------|
| 1 | **612.8** | **209%** | 1.0× |
| 2 | **1881.5** | **641%** | 3.1× |
| 4 | **4922.6** | **1679%** | 8.0× |
| 8 | **11284.5** | **3845%** | 18.4× |

### Critical bugs fixed during development

1. **INT4-FP32 buffer aliasing**: `fused_residual_norm_int4_fp32out` wrote both INT4 packed output
and FP32 normalized output to the same buffer (`d_x32`). INT4 output (bytes 0-1023 = 2048 nibbles)
corrupted the first 256 float32 elements of the FP32 hidden state (12.5% of H). 
Fix: separate buffers — INT4 → `d_x_i4`, FP32 → `d_x32`.

2. **Stale residual**: All old benchmarks copied `d_res = d_x32` once at layer 0, then reused
`d_res` for all 28 layers. Fix: pass current `d_x32` (updated per layer) as residual.

3. **Per-layer quantization**: `process_seq` quantized once at function entry, not per layer.
Layers 2-28 received stale quantized data from layer 0. Fix: quantize `d_x32` at start of each
layer loop.

### Performance breakdown (M=1, per layer)

- Per-layer: 0.058 ms = 58μs
- 28 layers: 1.63 ms total
- 14 kernels/layer: 7× gemv_int4_batched (Q,K,V,O,gate,up,down), 1× quantize_int4,
  1× attention_decode_batched_gqa, 1× fused_residual_norm_int4,
  1× fused_swiglu_quant_int4, 1× fused_residual_norm_int4_fp32out,
  8 M× update_kv_cache (serial per-seq)
- Launch overhead ~2/3 of total time at M=1

### Why INT4 is 209% of Q4_K_M

- **2× less DRAM reads** vs INT8: 0.5 bytes/val packed + 0.25 bytes/val scales = 0.75 bytes/val total
  (Q4_K_M: 0.50 bytes/val with super-block 256 and FP16 dual scales = 0.515)
- **Batched GEMM** (`gemv_int4_batched`) amortizes weight loads across M sequences: 1× DRAM read
  used for all M tokens
- **Batched attention** (`attention_decode_batched_gqa`) processes M sequences in one kernel call
  instead of M serial calls — saves (M-1)×kernel launch overhead
- **Fused kernels** save 2-3 kernel launches each per layer (residual+norm+quant = 1 kernel vs 3)
