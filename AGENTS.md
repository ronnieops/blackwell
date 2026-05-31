# AGENTS.md - blackwell

Custom CUDA kernels for INT8 + FP4 LLM inference on RTX 5060 Ti (Blackwell, SM_120a).

---

## 1. Mission

Benchmark INT8 forward pass throughput vs llama.cpp (Q4_K_M) baseline (**276.0 t/s**, re-measured 2026-05-30, b9389).
INT8 batched attn + CUDA Graph (M=8): **328.7 t/s** (119% of baseline). INT8 CUDA Graph batched M=8: **294.9 t/s** (107%). INT8 CUDA Graph (M=1): **183.3 t/s** (66%). **107 library symbols**.

---

## 2. Active State

**Stack**: CUDA 13.3, SM_120a, CMake, C++17
**Target**: RTX 5060 Ti 16 GB, compute 12.0, 36 SMs, ~500 GB/s GDDR7
**Nvcc path**: `/usr/local/cuda-13.3/bin/nvcc`
**Library**: 107 symbols in `build/libblackwell_kernels.a`

**WARNING**: `hashcat` runs persistently on this GPU (PID changes, auto-restarts). Uses 3740MiB VRAM at 95%+ util. Kills benchmark throughput ~45%. `kill all hashcat` before any measurement.

**Production kernels (INT8 path)**:
- `gemv_fp32_int8_per_row_warp` — Warp-cooperative FP32×INT8 GEMV (1 warp/row, shuffle reduce)
- `gemv_int8` — Legacy INT8 GEMV, `__dp4a` SIMD, 775 GB/s (kernel), 260 GB/s (effective). **Superseded by gemv_int8_warp**
- `gemv_int8_batched` — batched M=1-8, sweet spot M=3-4 (1.4× speedup)
- `gemv_int8_splitk` — split-K (K_splits=4), 779 GB/s
- `pack_int8` / `quantize_int8` — FP32 → INT8 quant with block scales
- `transpose_int8_weights` — W (K×N) → W_t (N×K) + scales
- `fused_rmsnorm_quant_int8` — RMSNorm + INT8 quant (1 kernel)
- `fused_gate_up_gemv` — fused gate+up MLP projection
- `fused_rmsnorm` — single-block warp-reduced RMSNorm
- `apply_swiglu` — silu(gate) × up, elementwise
- `fused_rope` / `fused_rope_decode` — in-place rotation, smem cos/sin cache
- `attention_decode_batched_gqa` — Batched GQA decode attention (M sequences in parallel)
- `update_kv_cache` — KV cache write with per-layer offset
- `gemm_int8_wmma` — WMMA INT8 GEMM (prefill, 3.8× dp4a, simplified dequant)
- `gemm_int8_mma` — Stub (returns cudaErrorNotSupported)

**Research kernels (FP4 path)**:
- `gemv_fp4_nv` / `gemv_fp4_nv_opt` — NVF4 scalar GEMV, UE4M3 scales, 98 GB/s (correct, not competitive)
- `pack_fp4` / `unpack_fp4` — FP4 E2M1 quant/dequant
- `gemm_fp4_block_scaled` — FP4 GEMM prefill
- `gemm_int8` / `gemm_int8_dp4a` — INT8 GEMM prefill (M>1, per-block scales, 4×4 tiling)
- `gemv_fp4_warp` — Packed FP4 warp GEMV (2 vals/byte, E2M1, 29 regs)
- `gemv_fp32_fp4_warp` — FP32×packed FP4 warp GEMV (47 regs)
- `decode_fp4_cgraph.cu` — Full 28L FP4 pipeline benchmark (CUDA Graph, 247 t/s, numerically unstable)
- `gemv_int4_warp` — INT4 warp GEMV (not competitive, 0.40× slower than INT8)

**FP4 packed: numerically unstable** (247 vs 184 t/s INT8). Throughput competitive but outputs garbage. E2M1 nibble→float per-element conversion can't use __dp4a SIMD.

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
./bench/decode_int8_cgraph 28              # CUDA Graph 183 t/s (M=1)
./bench/decode_int8_batched_cgraph 28 4   # CUDA Graph batched M=4: 287 t/s
./bench/decode_int8_batched_cgraph 28 8   # CUDA Graph batched M=8: 294 t/s
./bench/decode_int8_batched_cgraph_attn 28 4  # Batched attn + Graph M=4: 307 t/s (111%)
./bench/decode_int8_batched_cgraph_attn 28 8  # Batched attn + Graph M=8: 329 t/s (119%)
./bench/speculative_decode_cgraph 28 4          # Spec decode: 190 t/s (0% speedup, fixed)
./bench/bench_warp_gemv                    # Isolated warp vs old GEMV
./bench/decode_fp4_cgraph 28               # FP4 packed CUDA Graph 247 t/s (unstable)
./bench/bench_packed_fp4                   # FP4 vs INT8 single-kernel
./bench/text_generate "The capital of France is" 30  # Text gen, "Paris" ✓
```

### Model
`/mnt/data/ai/hf/qwen3-1.7b-base/` — Qwen3-1.7B, 28 layers, safetensors, 3.3 GB

---

## 4. Key Findings

| Finding | Value | Notes |
|---------|-------|-------|
| Warp GEMV speedup | **2.5–4.6×** vs old gemv_int8 | Coalesced loads (1 warp/row) |
| INT8 CUDA Graph (warp) | **183.5 t/s** | 66% of 276 t/s llama.cpp baseline |
| INT8 per-kernel (warp) | **162.9 t/s** | |
| INT8 batched attn M=4 CUDA Graph | **312.0 t/s** | **113%** of 276 (BEATEN!) |
| INT8 batched attn M=8 CUDA Graph | **328.7 t/s** | **119%** of 276 (BEATEN!) |
| INT8 batched attn M=8 per-kernel | 262.4 t/s | 95% of 276 |
| INT8 CUDA Graph batched M=8 (old) | **294.9 t/s** | **107%** of 276 |
| INT8 CUDA Graph batched M=4 (old) | **291.2 t/s** | **105%** of 276 |
| Speculative decode (M=4) | **190 t/s** | 0% speedup — same total work as autoregressive |
| FP4 batched (M=4) | 237.3 t/s | 86% ⚠️ 180% RMS diff vs INT8 |
| FP4 batched (M=8) | 243.4 t/s | 88% ⚠️ 180% RMS diff vs INT8 |
| WMMA GEMM (INT8) | **10,510 GFLOPS** | 3.81× over dp4a |
| INT4 warp GEMV | **0.40× SLOWER** than INT8 | Nibble unpack overhead negates 2× BW savings |
| FP4 warp GEMV | **0.50× SLOWER** than INT8 | E2M1→float overhead, can't use dp4a |
| llama.cpp Q4_K_M | **276.0 t/s** | End-to-end, build 9389, CUDA 13.3 |
| llama.cpp F16 | **108.3 t/s** | End-to-end |
| INT8 effective BW | 260 GB/s | Weight-bound (L2 cache miss) |
| GEMM prefill | 78 GB/s | 3× faster than llama.cpp |
| CUDA Graph speedup | ~10% (M=1), ~17% (batched M=8), ~19% (batched attn M=8) | Eliminates kernel launch overhead |
| Batched attention speedup | **+8-9%** over serial per-seq attention | Fuses M×num_q_heads grid into 1 kernel |
| L2 cache hints | ✅ Fixed | Targets graph_stream (commit f55a705) |
| INT8 batched M=8 CUDA Graph | **294.4 t/s** | **107%** of 276 llama.cpp baseline |
| Attention decode | 13.5% of pipeline | Single largest non-GEMV kernel |
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
3. **FP32 text_generate broken** — `text_generate_fp32.cu` produces worse output than INT8. Separate issue (BF16 weight file convention or cuBLAS transpose)
4. **GEMM prefill correctness unverified** — no reference comparison. Timing-only validation
5. **text_generate head_norm bug** — Pre-existing. "FAIL head_norm l=0". In `text_generate.cu` (uses `gemv_fp32_int8_per_row`)
6. **FP4 packed numerically unstable** — 247 vs 184 t/s. Throughput competitive but outputs garbage (~10^8 values). E2M1 nibble→float overhead can't use __dp4a SIMD.
7. **L2 cache hint targets wrong stream** — FIXED (commit f55a705). Targets graph_stream.
8. **CUDA Graph drift** — INT8: max diff ~3.1 (known, numerical drift). FP4: outputs garbage (~10^8 values).
9. **Speculative decode CUDA Graph crash** — static cudaMalloc in decode.cu needs warm-up first
10. **Docker/API packaging** — Not done
11. **WMMA dequant simplified** — Uses first-block scale only (L1 diff=13.4 vs dp4a). Per-block scale pending.

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
Verify: `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l` (expect 103)

---

## 10. Seed Principles

1. Smallest correct change. One kernel, one fix, one test.
2. Verify before broad edits.
3. Prefer repo evidence. Read code before assuming.
4. No churn.
5. Kernels first, framework later.
