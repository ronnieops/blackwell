# AGENTS.md - blackwell

Custom CUDA kernels for INT8 LLM inference on RTX 5060 Ti (Blackwell, GB206).

---

## 1. Mission

INT8 decode throughput vs llama.cpp Q4_K_M.

**Production**: 1.7B INT8 M=8 at **324 t/s (111% of Q4_K_M 293 t/s)**. Beats Q4_K_M by 11%.
**M=1**: 181 t/s (62% of Q4_K_M). BW-saturated at 260 GB/s effective.
**8B INT8**: 31-46 t/s. Quality upgrade path, bandwidth-bound.

**INT4/INT5 quality dead**. All sub-8-bit paths produce garbled text after 28+ layers. Attention softmax amplifies quantization noise — 23 dB PSNR per GEMV compounds to ~5 dB at lm_head. 4/5-bit quantization fundamentally insufficient for 28-layer transformer quality.

### llama.cpp comparison (build 9442, CUDA 13.3, RTX 5060 Ti)

| Model | Quant | tg128 | vs Our INT8 |
|-------|-------|-------|-------------|
| Qwen3-1.7B | Q4_K_M | 293.4 | **1.7B M=8 324 t/s (111%)** |
| Qwen3-1.7B | Q4_K_M | 293.4 | 1.7B M=1 181 t/s (62%) |
| Qwen3-8B | Q4_K_M | 82.66 | 8B M=1 46 t/s (56%) |
| Qwen3.5-9B | Q3_K_M | 71.4 | 9B M=1 45 t/s (63%) |

---

## 2. Active State

**Stack**: CUDA 13.3, SM_120a, CMake, C++17
**Target**: RTX 5060 Ti 16 GB, compute 12.0, 36 SMs, ~500 GB/s GDDR7
**Nvcc path**: `/usr/local/cuda-13.3/bin/nvcc`
**Library**: 191 symbols in `build/libblackwell_kernels.a`

**Production kernels (INT8 path)**:
- `gemv_int8_warp` — Warp-cooperative INT8 GEMV (1 warp/row, dp4a SIMD, shuffle reduce)
- `gemv_int8_batched` — Batched INT8 GEMV M=1-8
- `gemv_int8_splitk` — Split-K INT8 GEMV (K_splits=4)
- `fused_rmsnorm_quant_int8` — RMSNorm + INT8 quant (1 kernel)
- `fused_swiglu_quant` — SwiGLU + INT8 quant (fused)
- `fused_rmsnorm` — Single-block warp-reduced RMSNorm
- `attention_decode_gqa` — GQA decode attention (M=1)
- `attention_decode_batched_gqa` — Batched GQA decode (M seq)
- `update_kv_cache` — KV cache write with per-layer offset
- `pack_int8` / `quantize_int8` — FP32 → INT8 quant with block scales
- `vector_add_fp32` — Elementwise FP32 addition
- `apply_swiglu` — silu(gate) × up
- `apply_rope` / `fused_rope_decode` — In-place RoPE
- `gemv_int8_gate_up` — Fused gate+up INT8 GEMV (0.91× slower than serial)
- `sample_gpu` / `sample_argmax_gpu` — GPU softmax + sampling
- `absmax_scales_kernel` — Block absmax scale computation

**GatedDeltaNet kernels (Qwen3.5-9B)**:
- `gated_delta_conv1d_update` — 1D depthwise conv + SiLU
- `gated_delta_recurrent_step` — SSM recurrent step (NK→NV heads)
- `gated_delta_rmsnorm_gated` — RMSNormGated with SiLU gate
- `attention_decode_kernel_v4` — Decode attention for head_dim=256

**GEMM kernels**:
- `gemm_int8_wmma` / `gemm_int8_wmma_fast` — WMMA INT8 GEMM (prefill)

**Research kernels (DO NOT USE)**:
- `gemv_int8_from_fp4` — 2.8× slower
- `gemv_fp4_warp` / `gemv_fp4_nv` — FP4 GEMV, not competitive
- `gemv_fp32_fp4_warp` — FP32×FP4 packed GEMV
- `gemv_int4_warp` / `gemv_int4_batched` — INT4 GEMV (quality dead)
- `gemv_fp32_int4_asym` — FP32×INT4 asymmetric (122 dB exact, useless quality: 23 dB PSNR)
- `gemv_fp32_int5_asym` — FP32×INT5 asymmetric (122 dB exact, useless quality: 29 dB PSNR)
- `gemv_int4_asym_batched` — INT4 asymmetric batch GEMV
- All `quantize_int4*` / `fused_*_int4*` / `fused_*_int4_asym*` — quality dead
- `gemm_int8` / `gemm_int8_dp4a` — Superseded by WMMA
- `gemm_fp4_block_scaled` — FP4 tensor core GEMM (prefill, unused)
- `decode_fp4_cgraph.cu` — FP4 pipeline (numerically unstable)

**Kept for reference**: `bench/bench_batched_gemv.cu` — M=8 vs serial GEMV comparison tool, `bench/decode_qwen35_9b_batched.cu` — GatedDeltaNet M=8 batched, `scripts/extract_8b_norms.py` — 8B support file extraction.

---

## 3. Build & Run

### Build
```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### 1.7B benchmarks (production)
```bash
killall hashcat 2>/dev/null  # MUST DO BEFORE ANY MEASUREMENT
./bench/decode_int8_cgraph 28                       # M=1: 181 t/s
./bench/decode_int8_batched_cgraph_attn 28 8        # M=8: 324 t/s ✅ beats Q4_K_M
./bench/text_generate "The capital of France is" 30 # Correctness
```

### 8B benchmarks
```bash
./bench/decode_int8_cgraph_qwen3_8b 36              # M=1: 46 t/s
./bench/decode_int8_batched_cgraph_attn_qwen3_8b 28 8 # M=8: 40 t/s
./bench/text_generate_qwen3_8b "The capital of France is" 30
```

### Qwen3.5-9B GatedDeltaNet
```bash
./bench/decode_qwen35_9b weights_int8_qwen35_9b 20        # M=1: 45 t/s
./bench/decode_qwen35_9b_batched 8 20                      # M=8: 50 t/s
```

### Diagnostics
```bash
nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l  # expect 191
./bench/verify_int4_asym_full 42    # INT4 28L SNR (for reference)
```

### Docker server
```bash
docker build -t blackwell-server .
docker run --gpus all -p 8080:8080 blackwell-server
# POST http://localhost:8080/generate with {"prompt": "...", "max_tokens": 50}
```

---

## 4. File Layout

### Weight directories
```
weights_int8_bf16/            # 1.7B INT8 weights (2.1 GB)
weights_int4_qwen3_1.7b/      # 1.7B INT4 symmetric (dead end)
weights_int4_qwen3_1.7b_asym/ # 1.7B INT4 asymmetric (dead end)
weights_int5_qwen3_1.7b_asym/ # 1.7B INT5 asymmetric (dead end)
weights_int8_qwen3_8b/        # 8B INT8 weights + norms (9.6 GB)
weights_int8_qwen35_9b/       # 9B GatedDeltaNet INT8 (11 GB)
```

### Key source files
```
src/kernels/
  gemv_int8.cu            — Production INT8 GEMV (warp, batched, splitk, pack, fused)
  decode.cu               — Attention (GQA, batched, KV cache, RoPE)
  fused_rmsnorm.cu        — RMSNorm + quant + pack fusions
  gemm_int8.cu            — WMMA INT8 GEMM (prefill)
  gated_delta_net.cu      — GatedDeltaNet SSM kernels
  gemv_fp32_int4_asym.cu  — INT4 research (122 dB exact, dead-end)
  gemv_fp32_int5_asym.cu  — INT5 research (122 dB exact, dead-end)
  gemv_int8_gate_up.cu    — Fused gate+up GEMV (0.91×)
bench/
  text_generate.cu              — 1.7B end-to-end text generation
  text_generate_qwen3_8b.cu     — 8B end-to-end text generation
  text_generate_int4.cu         — INT5 text generation (garbled, reference only)
  decode_int8_cgraph.cu         — 1.7B M=1 CUDA Graph benchmark
  decode_int8_batched_cgraph_attn.cu — 1.7B M=8 batched benchmark
```

---

## 5. Key Findings

| Finding | Value |
|---------|-------|
| **1.7B INT8 M=8 production** | **324 t/s (111% of Q4_K_M)** |
| 1.7B INT8 M=1 | 181 t/s (62% of Q4_K_M, BW-saturated) |
| 8B INT8 M=1 | 46 t/s (56% of Q4_K_M) |
| 9B GatedDeltaNet M=8 | 50 t/s (70% of Q3_K_M) |
| Effective BW (1.7B) | 260 GB/s (52% of 500 GB/s peak) |
| Effective BW (8B) | 319 GB/s (63% of peak) |
| Sub-8-bit quality | ❌ Dead. Attention softmax amplifies noise. |
| Batched GEMV vs serial | 2-2.7× slower per call |
| hashcat interference | -45% throughput, must kill before measurement |

### Quality paths (all tested, all dead)
| Path | PSNR/GEMV | Result |
|------|-----------|--------|
| Symmetric INT4 | 23 dB | Garbled |
| Asymmetric INT4 | 23 dB | Garbled |
| FP32×INT4 (weight-only) | 23 dB | Garbled |
| FP32×INT5 (weight-only) | 29 dB | Garbled |
| Mixed INT4 attn + INT8 MLP | — | Garbled |
| Per-channel INT4 | 16 dB | Worse than block-16 |

### Bottleneck analysis
- M=1: 95% of bandwidth floor. No optimization possible.
- M=8: Batched GEMV slower than serial. CUDA Graph saves 2.6%.
- 8B: FP4 state overhead creates 9× bandwidth floor gap.
- GatedDeltaNet: SSM ~5 us vs GEMV ~5000 us per layer.

---

## 6. Constraints

- `CUDACXX` env var must be set before `project()` in CMakeLists.txt
- `compute_120a` required (not `compute_120`)
- `killall hashcat` before any measurement — 60s respawn window
- `gemv_int8_warp` is production INT8 GEMV
- All weight matrices exceed L2 cache (32 MB)
- M>8 not viable (register pressure in batched GEMV)
- llama.cpp GGUF format not supported — uses separate weight files

---

## 7. Docker Production Server

`Dockerfile` + `server/server.py` supports:
- REST API at port 8080
- POST `/generate` with `prompt`, `max_tokens`, `temperature`, `top_k`
- Streaming response via chunked transfer
- 1.7B model at 181 t/s (M=1) or batched M=8 (324 t/s)
- GPU sampling (softmax + temperature + top-k via sample_gpu)
- Health check at GET `/health`

Build and run: `docker build -t blackwell-server . && docker run --gpus all -p 8080:8080 blackwell-server`

---

## 8. Development Loop

```
observe → plan → edit → build → test → reflect → update AGENTS.md only if useful
```

Build: `CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build && cmake --build build --parallel`
Test: `./bench/decode_int8_batched_cgraph_attn 28 8` (M=8 production)
Verify: `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l` (expect 191)

---

## 9. Anti-Hallucination Rules

- **Do not invent APIs, files, commands, env vars, or requirements.** Read the actual header/source before calling a function.
- **Prefer repo evidence over assumptions.** If you need a function signature, read `include/blackwell/kernels.h`.
- **Mark unknowns explicitly.** "Not checked" or "unknown behavior" in comments.
- **Never overwrite higher-priority instructions.**
- **Preserve user intent and existing project conventions.**

---

## 10. Seed Principles

1. Smallest correct change. One kernel, one fix, one test.
2. Verify before broad edits.
3. Prefer repo evidence. Read code before assuming.
4. No churn.
5. Kernels first, framework later.

---

## 11. Bug History

### vector_add_fp32_kernel (2026-05-28) — FIXED
`src/kernels/norm.cu`: reversed `=` in float4 path wrote uninitialized data TO input buffer.
Fix: `float4 va = ((float4*)a)[idx];` (load, not store).

### RoPE frequency (2026-05-29) — FIXED
All 5 bench files: `idxf = i2/hd` doubled exponent → 2× rotation speed.
Fix: `theta = pos * powf(rope_theta, -2.0f * d / head_dim);`

### head_norm cross-warp (2026-05-29) — FIXED
All 5 bench files: `__shfl_xor_sync` with off=64/32 no-ops on 32-lane warps → 1/4 sums.
Fix: smem[4] warp partials → shuffle-reduce across 4 warps.

### INT4 fused_residual_norm_int4_fp32out buffer aliasing (2026-06-02) — FIXED
INT4 output corrupted FP32 buffers used by next layer.
Fix: separate output buffers for INT4 and FP32.

### fused_residual_norm only processes first 2048 elements (2026-06-02) — FIXED
Only affected Qwen3-8B (H=4096). Thread count 256→512. Warmup loop bug.
Fix: kFusedThreads=256→512, iterate all layers in warmup.

### gemv_int4_batched grid bug (2026-06-02) — FIXED
`dim3 grid(N/32,M)` only computed 1/32 of output rows.
Fix: `dim3 grid(N, M)`. All pre-session-37 INT4 benchmarks invalidated.

### INT4 nibble sign-extension bug (2026-06-02) — FIXED
Used wrong 3-bit two's complement sign-extend instead of nib-8 offset-binary.
Fix: `nib - 8` for both lo and hi nibbles.

### INT4 weight corruption (2026-06-02) — FIXED
Scales ~1e-23 due to `f.seek(0)` bug in `read_tensor()`.
Fix: re-run quantization from scratch.