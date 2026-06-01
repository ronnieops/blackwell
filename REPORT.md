# Blackwell INT8 Decode Benchmark Report

## Executive Summary

Custom CUDA INT8 inference kernels for Qwen3-1.7B on RTX 5060 Ti (Blackwell, SM_120a).
Primary result: **M=8 batched decode at 324 t/s** (110% of llama.cpp Q4_K_M baseline of 293 t/s).

| Configuration | Total t/s | Per-seq t/s | vs Q4_K_M | VRAM |
|--------------|-----------|-------------|-----------|------|
| M=1 (decode) | 181 | 181 | 62% | 5 GB |
| **M=8 (CUDA Graph)** | **324** | **40.5** | **110%** | **5 GB** |
| M=16 (CUDA Graph) | 335 | 20.9 | 114% | 9 GB |
| llama.cpp Q4_K_M | 293 | 293 | 100% | 5 GB |
| llama.cpp F16 | 114 | 114 | 39% | 5 GB |

**Note**: Earlier measurement of 864 t/s at M=16 was incorrect — `gemv_int8_batched` silently returned zero for M>8 (switch statement only supported M=1-8). After fixing, real M=16 is 335 t/s. M=8 is the practical limit for batched GEMV.

---

## 1. Hardware & Software

- **GPU**: NVIDIA RTX 5060 Ti 16 GB, SM_120a, 36 SMs, ~500 GB/s GDDR7
- **CUDA**: 13.3, C++17, CMake
- **Model**: Qwen3-1.7B (28 layers, H=2048, Q=2048, KV=1024, I=6144, 16 Q heads, 8 KV heads)
- **Quantization**: INT8 per-block (16×16 blocks), weights transposed [N×K], scales [N/16 × K/16]
- **Weight files**: `weights_int8_bf16/` — pre-quantized INT8 weights from BF16 source

---

## 2. M=1 Decode Performance

### Baseline: llama.cpp
- Q4_K_M: **292.9 t/s** (Qwen3-1.7B, CUDA 13.2)
- F16: **114.3 t/s**

### INT8 Fused Pipeline (14 kernels/layer)
```
unpack_fp4_pack_int8 → gemv_int8_warp(Q) → gemv_int8_warp(K) → gemv_int8_warp(V)
→ update_kv_cache → attention_decode_gqa → pack_int8 → gemv_int8_warp(Wo)
→ fused_residual_norm → fused_rmsnorm_pack
→ unpack_fp4_pack_int8 → gemv_int8_warp(gate) → gemv_int8_warp(up)
→ fused_swiglu_quant → gemv_int8_warp(down) → fused_residual_norm → fused_rmsnorm_pack
```

**Result: 181 t/s (62% of Q4_K_M)**

Bandwidth-limited: INT8 reads 1 byte/param vs Q4_K_M's 0.5 bytes/param. Even perfect compute can't exceed ~50% of Q4_K_M throughput. The 62% comes from compute overhead (dequantization, SwiGLU, attention).

### Key M=1 Findings

| Finding | Detail |
|---------|--------|
| `gemv_int8_warp` | 1 warp/row, shuffle reduce. 260 GB/s effective. Core compute kernel. |
| `fused_pack_gemv_o` | Correct but 20% slower (quant overhead on GEMV critical path) |
| `fused_swiglu_gemv` | Correct but 20% slower (same reason) |
| CUDA Graph M=1 | **Failed** — `cudaFuncSetAttribute` in `attention_decode_gqa` incompatible with capture |
| Persistent kernel | Abandoned — correctness bugs in cross-warp shuffle, scale indexing |
| `gemv_int8_qkv` | Fused Q/K/V GEMV (3→1), ~1.46× speedup per kernel |

---

## 3. M=8 Decode Performance

### Architecture
- **Batched attention**: `attention_decode_batched_gqa` — processes M sequences in one kernel launch
- **CUDA Graph**: 28 layers × 119 kernels = 3332 kernel launches captured as single graph
- **Mixed serial/batched**: Serial Q/K/V GEMVs + batched MLP GEMVs + batched attention

### Pipeline Per Layer (119 kernels)
```
unpack_fp4_pack_int8 × M (8)     — FP4→INT8 per sequence
gemv_int8_warp(Q) × M (8)        — Q projection per sequence
gemv_int8_warp(K) × M (8)        — K projection per sequence
gemv_int8_warp(V) × M (8)        — V projection per sequence
update_kv_cache × M (8)          — KV cache write per sequence
attention_decode_batched_gqa × 1  — batched attention (replaces M serial calls)
pack_int8 × M (8)                — attn FP32→INT8 per sequence
gemv_int8_warp(Wo) × M (8)       — output projection per sequence
unpack_fp4 × M (8)               — residual FP4→FP32 per sequence
vector_add_fp32 × M (8)          — residual add per sequence
fused_rmsnorm_quant_int8 × M (8) — RMSNorm + INT8 quant per sequence
fused_rmsnorm_pack × M (8)       — RMSNorm + FP4 pack per sequence
gemv_int8_warp(gate) × M (8)     — gate projection per sequence
gemv_int8_warp(up) × M (8)       — up projection per sequence
apply_swiglu × M (8)             — SwiGLU activation per sequence
pack_int8 × M (8)                — MLP FP32→INT8 per sequence
gemv_int8_warp(down) × M (8)     — down projection per sequence
```

**Result: 323 t/s (110% of Q4_K_M)**

---

## 4. M=16 Decode Performance (Optimal)

### Scaling (corrected after fixing gemv_int8_batched M>8)
| M | Total t/s | Per-seq t/s | Per-step | VRAM |
|---|-----------|-------------|----------|------|
| 1 | 181 | 181 | 5.5 ms | 5 GB |
| **8** | **324** | **40.5** | **24.7 ms** | **5 GB** |
| 16 | 335 | 20.9 | 47.8 ms | 9 GB |

### Why M=16 Doesn't Help (Corrected)

Earlier measurement of 864 t/s at M=16 was **wrong** — `gemv_int8_batched` silently returned zero for M>8 (switch statement only had cases 1-8). The MLP GEMVs weren't running, so the benchmark only measured attention + other non-MLP kernels.

After fixing `gemv_int8_batched` to support M>8 (loop over groups of 8), the real M=16 throughput is **335 t/s** — barely better than M=8 (324 t/s). The batched GEMV kernel is slower for M>8 because:
- Each block processes M sequences → register pressure increases linearly with M
- For M=16: 16× activation registers per block → occupancy drops to 1-2 blocks/SM
- Serial `gemv_int8_warp` (1 warp/row) has better occupancy but more kernel launches

**M=8 is the practical limit for batched GEMV on this GPU.**
1. **Weight reuse**: 16 sequences share same 12.6 MB weight matrices. L2 cache (32 MB) holds gate+up+down weights (37.8 MB ≈ L2 size).
2. **KV cache**: 16 × 28 × 8 × 2048 × 128 × 4 = 4.48 GB (fits in DRAM with headroom)
3. **GPU occupancy**: 16 sequences × 8 GEMV kernels = 128 blocks → fully saturates 36 SMs
4. **Beyond M=16**: KV cache pressure (168 MB/seq) starts competing with weight data for DRAM bandwidth

### Bottleneck Profile (per layer, M=8)
```
gate_up GEMV (N=6144, K=2048):    0.381 ms  (39.7%)  ← DOMINANT
down GEMV (N=2048, K=6144):       0.169 ms  (17.6%)
residual+norm × 2 passes:         0.153 ms  (16.0%)
QKV GEMV (3 projections):         0.115 ms  (12.0%)
pack+Wo (attn output):            0.049 ms   (5.1%)
batched attention:                 0.031 ms   (3.2%)
update_kv × M:                    0.025 ms   (2.6%)
swiglu+pack:                      0.020 ms   (2.1%)
unpack_fp4_pack_int8 × M:         0.016 ms   (1.7%)
─────────────────────────────────────────────────
Total per layer:                  0.959 ms
Total 28 layers:                 26.9 ms (theoretical)
Actual per-step:                 18.5 ms (CUDA Graph, measured)
```

**Key insight**: gate_up + down GEMV = 57% of layer time. These are memory-bandwidth-bound (12.6 MB weights per GEMV). No kernel fusion can fix this — it's an architectural constraint of INT8 quantization.

---

## 5. Key Technical Findings

### What Works
1. **Batched attention** (`attention_decode_batched_gqa`): 9-22% speedup over serial attention for M≥8
2. **CUDA Graph**: 2-8% speedup by eliminating kernel launch overhead (significant when 119+ kernels/layer)
3. **Fused kernels**: `fused_residual_norm`, `fused_rmsnorm_pack`, `fused_swiglu_quant`, `fused_unpack_fp4_quant` — each saves 1 kernel launch per call
4. **Warp-cooperative GEMV** (`gemv_int8_warp`): 1 warp/row, shuffle reduce, coalesced weight loads

### What Doesn't Work
1. **`gemv_int8_batched`**: 1.5-2.7× SLOWER than serial `gemv_int8_warp` for all GEMV sizes. Higher register pressure per block reduces occupancy. Only beneficial in CUDA Graph context (fewer graph nodes).
2. **Fused pack+GEMV**: `fused_pack_gemv_o`, `fused_swiglu_gemv` — correct but 20% slower. Two-phase kernels (quant→sync→GEMV) add overhead exceeding kernel-launch savings.
3. **CUDA Graph for M=1**: `attention_decode_gqa` wrapper calls `cudaFuncSetAttribute` (smem config) which poisons capture on Blackwell.
4. **Persistent kernels**: Cross-warp attention shuffle, INT8→FP32 scale mismatch, KV cache layout issues. Abandoned.
5. **L2 persisting for large weights**: Pinning 12.6 MB gate weights in L2 → 28% regression. Evicts other cached data.
6. **FP4 quantization**: Numerically unstable (247 t/s but garbage outputs). E2M1 nibble→float can't use `__dp4a` SIMD.

### Architecture Decisions
- **INT8 per-block quantization**: 16×16 blocks, FP32 scales. Balances accuracy and speed.
- **KV cache**: FP32, per-layer offset `l × nkv × ms × hd`. `update_kv_cache` uses `cudaMemcpyAsync` H2D with pinned memory.
- **FP4 hidden state**: Activations stored as FP4 between layers to save VRAM. Unpacked to INT8 before GEMV.
- **GQA**: 16 Q heads, 8 KV heads (2:1 ratio). `attention_decode_batched_gqa` handles GQA correctly.

---

## 6. Library Inventory

**157 symbols** in `build/libblackwell_kernels.a`:

### Production Kernels
- `gemv_int8_warp` — Core INT8 GEMV (1 warp/row)
- `gemv_int8_batched` — Batched INT8 GEMV (slower, used in CUDA Graph)
- `attention_decode_gqa` — GQA decode attention
- `attention_decode_batched_gqa` — Batched GQA decode attention (M sequences)
- `update_kv_cache` — KV cache write with per-layer offset
- `pack_int8` / `quantize_int8` — FP32→INT8 quantization
- `unpack_fp4` / `pack_fp4` — FP4↔FP32 conversion
- `apply_swiglu` — SwiGLU activation (silu(gate) × up)
- `vector_add_fp32` — Elementwise add
- `fused_residual_norm` — Residual add + RMSNorm + INT8 quant
- `fused_rmsnorm_pack` — RMSNorm + FP4 pack
- `fused_unpack_fp4_quant` — FP4 unpack + INT8 quant
- `fused_swiglu_quant` — SwiGLU + INT8 quant
- `transpose_int8_weights` — Weight matrix transpose + scale transpose
- `sample_gpu` / `sample_argmax_gpu` — GPU sampling

### Research Kernels
- `gemv_fp4_warp` / `gemv_fp32_fp4_warp` — FP4 GEMV (not competitive)
- `gemm_int8_wmma` / `gemm_int8_wmma_fast` — WMMA INT8 GEMM (prefill)
- `fused_pack_gemv_o` — Fused pack+Wo GEMV (correct but slower)
- `fused_swiglu_gemv` — Fused SwiGLU+down GEMV (correct but slower)
- `persistent_qkv_gemv` — Persistent QKV stub (abandoned)

---

## 7. Build & Run

```bash
# Build
export PATH=/usr/local/cuda-13.3/bin:$PATH
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Benchmark
killall hashcat 2>/dev/null  # MUST DO — uses 3.7 GB VRAM, -45% throughput
./bench/decode_int8_cgraph 28                    # M=1: 181 t/s
./bench/decode_int8_batched_cgraph_attn 28 8     # M=8: 324 t/s (optimal)
./bench/decode_int8_generic 28 weights_int8_bf16 2048 2048 1024 6144 16 8 "Qwen3-1.7B"

# Text generation
./bench/text_generate "The capital of France is" 30
```

---

## 8. What's Next

### Maximize M=16 throughput
- Profile M=16 specific bottlenecks (may differ from M=8)
- Try split-K GEMV for gate_up (N=6144, largest GEMV)
- Experiment with L2 cache hints for weight prefetching
- Try increasing batch size beyond 16 with FP16 KV cache (saves 50% KV memory)

### Close M=1 gap to Q4_K_M
- Implement real Q4 quantization (GPTQ/AWQ from HuggingFace)
- INT4 weights would halve memory reads → ~350+ t/s M=1

### Production deployment
- Docker container with inference server
- Speculative decoding with draft model
- Continuous batching for variable-length sequences
