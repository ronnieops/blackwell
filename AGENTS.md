# AGENTS.md - blackwell

Custom CUDA kernels for INT8 + FP4 LLM inference on RTX 5060 Ti (Blackwell, GB206).

---

## 1. Mission

Benchmark INT8 forward pass throughput vs llama.cpp (Q4_K_M) baseline (**292.9 t/s** FA=on, CUDA 13.3).
INT8 batched attn + CUDA Graph (M=8): **323.5 t/s** (110% of Q4_K_M). **+183% vs llama.cpp F16** (114.3 t/s). **157 library symbols**.

**Speculative decode infeasible**: Batched verify path (24.7 ms/seq) is **4.5× slower per-seq** than sequential decode (5.52 ms/seq). Draft model would need 4.5× speedup to break even. Even tiny 50M draft (~18× speedup) yields only ~92 t/s due to low acceptance rates. Self-speculation (skip layers) won't work — lm_head needs all layers. Abandoned.

**Docker production server ready**: `Dockerfile` + `server/server.py`. 324 t/s beats Q4_K_M by 10%. Ship it.

**M=1 INT8 decode: 176.5 t/s (60% of Q4_K_M).** Bandwidth-limited: INT8 reads 1 byte/param vs Q4_K_M's 0.5 byte/param. Cannot close gap through fusion alone.

**M=8 INT8 decode: 324 t/s (110% of Q4_K_M).** Batched attention + CUDA Graph is the competitive path. M>8: batched GEMV slower due to register pressure (gemv_int8_batched only supported M≤8, now fixed to loop over groups of 8). M=16: 335 t/s (batched MLP) — barely better than M=8.

**Fused kernel launches (M=1)**: 14 kernels/layer (was 20). 30% reduction.
- `fused_unpack_fp4_quant` — unpack FP4 + quantize INT8 (replaces 2 kernels)
- `gemv_int8_qkv` — fused Q/K/V GEMV (replaces 3 kernels)
- `gemv_int8_gate_up` — fused gate+up GEMV (replaces 2 kernels)
- `fused_swiglu_quant` — SwiGLU + INT8 quant (replaces 2 kernels)
- `fused_residual_norm` — residual add + RMSNorm + quant (replaces 3 kernels)

---

## 2. Active State

**Stack**: CUDA 13.3, SM_120a, CMake, C++17
**Target**: RTX 5060 Ti 16 GB, compute 12.0, 36 SMs, ~500 GB/s GDDR7
**Nvcc path**: `/usr/local/cuda-13.3/bin/nvcc`
**Library**: 157 symbols in `build/libblackwell_kernels.a`

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
- `gemv_int4_warp` — INT4 warp GEMV (not competitive, 0.40× slower than INT8)

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
./bench/decode_int8_cgraph 28              # Fused per-kernel 176.5 t/s (M=1, 14 kernels/layer)
./bench/decode_int8_batched_cgraph_attn 28 8  # Batched attn + Graph M=8: 323.5 t/s (110% of Q4_K_M)
./bench/decode_int8_generic 28 weights_int8_bf16 2048 2048 1024 6144 16 8 "Qwen3-1.7B"  # 176.5 t/s
./bench/decode_int8_generic 28 weights_int8_qwen3_06b 1024 1024 512 3072 8 4 "Qwen3-0.6B"  # 447.4 t/s
./bench/decode_int8_generic 36 weights_int8_qwen3_8b 4096 4096 1024 12288 32 8 "Qwen3-8B"  # 44.5 t/s
./bench/speculative_decode_cgraph 28 4          # Spec decode: 190 t/s
./bench/bench_warp_gemv                    # Isolated warp vs old GEMV
./bench/text_generate "The capital of France is" 30  # Text gen, "Paris" ✓
```

### Model
`/mnt/data/ai/hf/qwen3-1.7b-base/` — Qwen3-1.7B, 28 layers, safetensors, 3.3 GB

---

## 4. Key Findings

| Finding | Value | Notes |
|---------|-------|-------|
| Warp GEMV speedup | **2.5–4.6×** vs old gemv_int8 | Coalesced loads (1 warp/row) |
| INT8 fused (M=1) | **176.5 t/s** | 14 kernels/layer (was 20), 30% launch reduction |
| INT8 batched-attn M=8 CUDA Graph | **323.5 t/s** | **110%** of llama.cpp Q4_K_M (292.9) |
| INT8 batched-attn M=8 vs llama.cpp F16 | **+183%** | 323.5 vs 114.3 t/s |
| INT8 batched-attn M=4 CUDA Graph | **307.5 t/s** | **105%** of Q4_K_M |
| INT8 batched-attn M=1 CUDA Graph | **118.5 t/s** | 40% of Q4_K_M |
| INT8 generic CUDA Graph (1.7B) | **181.7 t/s** | 62% of Q4_K_M |
| INT8 generic CUDA Graph (0.6B) | **444.1 t/s** | H=1024 |
| INT8 generic CUDA Graph (8B, 28L) | **57.3 t/s** | H=4096, 69% of Q4_K_M (82.5) |
| INT8 generic CUDA Graph (8B, 36L) | **44.6 t/s** | H=4096 |
| WMMA GEMM (INT8) | **10,510 GFLOPS** | 3.81× over dp4a |
| WMMA FAST GEMM (INT8) | **4.3-5.0K GFLOPS** | 1.2-1.4× over dp4a (real weights) |
| Block GEMV unrolling | **+9-45%** | 4× unroll, K-dependent |
| Speculative decode (M=4) | **190 t/s** | 0% speedup — same total work as autoregressive |
| FP4 batched (M=8) | 243.4 t/s | 83% ⚠️ 180% RMS diff vs INT8 |
| llama.cpp Q4_K_M FA=on | **292.9 t/s** | Qwen3-1.7B, build b9442, CUDA 13.3 |
| llama.cpp Q4_K_M FA=off | **274.3 t/s** | Qwen3-1.7B (old baseline path) |
| llama.cpp F16 FA=on | **114.3 t/s** | Qwen3-1.7B |
| llama.cpp F16 FA=off | **111.2 t/s** | Qwen3-1.7B (old baseline path) |
| llama.cpp Q4_K_M FA=on (8B) | **82.5 t/s** | Qwen3-8B, build b9442 |
| llama.cpp Q4_K_M (3.5-9B) | 71.4 t/s | Qwen3.5-9B MoE |
| Qwen3.5-9B INT8 decode | **45.6 t/s** | 64% of Q4_K_M, weight-bound (INT8 reads 2× Q4) |
| INT8 effective BW | 260 GB/s | Weight-bound (L2 cache miss) |
| GEMM prefill (before fix) | 4.3 TFLOPS | 8.7% utilization |
| GEMM prefill (after c_frag fix) | **13.0 TFLOPS** | **3× speedup**, 26% utilization |
| Pipeline SNR | **13.9 dB** | Constant across 28 layers, no compounding |
| CUDA Graph speedup | ~1-6% | Model-size dependent |
| Batched attention speedup | **+9.8%** | M=8 vs serial-attn |
| hashcat interference | -45% throughput | Kills GPU-0 ~every 60s |
| INT4/FP4 sub-byte GEMV | ❌ Not competitive | ~35 inst/byte unpack vs 0.31 inst/byte dp4a |

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

---

## 7. Known Issues

1. **hashcat runs persistently** on GPU-0 (PID 57393/64789, auto-restarts). Uses 3740MiB VRAM. Kills benchmark throughput ~45%. Must `killall hashcat` before any measurement — 60s window before respawn
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
    - **M=1 CUDA Graph may be salvageable**: llama.cpp uses `cudaStreamCaptureModeRelaxed` (not `Global`). Their `CUDA_SET_SHARED_MEMORY_LIMIT` macro calls `cudaFuncSetAttribute` during capture via static guard. Works in Relaxed mode. Our failure was Global-mode specific — Relaxed mode allows operations like `cudaMalloc`/`cudaFuncSetAttribute` during capture. Try switching to `cudaStreamCaptureModeRelaxed`.
    - **FP4 tensor cores viable**: llama.cpp has `BLACKWELL_MMA_AVAILABLE` for NVFP4/MXFP4 MMQ using tensor cores (not `__dp4a`). Our FP4 path tried scalar `__dp4a` and failed numerically. Tensor core MMQ may close the gap.
    - **PDL (Programmatic Dependent Launch)**: llama.cpp supports PDL for Hopper+ Blackwell. Eliminates kernel launch overhead without CUDA Graph. Check SM_120a support.
    - **MMVQ_MAX_BATCH_SIZE=8**: llama.cpp caps quantized batch at 8. Validates our M=8 register pressure limit. Same architecture.
    - **All 34 kernel source files** in `ggml/src/ggml-cuda/` — well-organized, template-instance pattern for specialized kernels.

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
Verify: `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l` (expect 157)

---

## 10. Seed Principles

1. Smallest correct change. One kernel, one fix, one test.
2. Verify before broad edits.
3. Prefer repo evidence. Read code before assuming.
4. No churn.
5. Kernels first, framework later.
