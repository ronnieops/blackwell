# Fusion Kernels + Experiments Results (Session ~82)

## Phase 1: Fused Gate+Up INT4 GEMV — DONE ✅

**Kernel**: `src/kernels/fused_gate_up_int4.cu`
- Loads activation x once per K-block, computes both gate and up projections
- 1 kernel launch instead of 2
- dp4a INT4×INT4 dot product, warp shuffle reduction

**Bug found & fixed**: `int4_8bytes_to_int4lanes` wrote single nibbles instead of packing 4 nibbles per INT32 lane. dp4a needs 4×INT8 packed per lane. Copied exact working impl from `gemv_int8.cu`.

**Throughput**:
| Config | Before | After | Gain |
|--------|--------|-------|------|
| M=1 | 63 t/s | 70 t/s | +11% |
| M=8 | 205 t/s | 220 t/s | +7% |

## Phase 2: Fused QKV INT4 GEMV — DONE ✅

**Kernel**: `src/kernels/fused_qkv_int4.cu`
- Grid: `max(Q_dim, KV_dim)` = 4096 blocks
- Blocks 0..1023 (KV_dim): compute Q+K+V from same activation (3× reuse)
- Blocks 1024..4095: compute Q only
- 1 kernel launch instead of 3

**Throughput** (combined with fused gate+up):
| Config | Before | After | Gain |
|--------|--------|-------|------|
| M=1 | 63 t/s | **71 t/s** | **+13%** |
| M=4 | 178 t/s | 193 t/s | +8% |
| M=8 | 205 t/s | **227 t/s** | **+11%** |
| M=16 | 220 t/s | **245 t/s** | **+11%** |

## Phase 3: AWQ + Fusion PPL — DONE ✅

bench_ppl_int4_8b now accepts weight dir as argv[1].

| Weight dir | PPL |
|-----------|-----|
| `weights_int4_qwen3_8b_fp16sc` (baseline) | **21.98** |
| `weights_int4_qwen3_8b_awq_perlayer_fp16sc` | **23.57** |

AWQ per-layer is 7% WORSE than baseline in this fused bench_ppl. AGENTS.md
recorded both at 21.98, but that was non-fused path. The fused_rmsnorm_quant_int4
path may interact differently with AWQ scales.

## Phase 4: 2-Warp GEMV Microbench — DONE ✅

**Bench**: `bench/bench_gemv_2warp.cu`
- 1-warp (32 threads) vs 2-warp (64 threads) per output row
- 2-warp uses cross-warp shared memory reduction

**Results**: 2-warp is SLOWER.
| Kernel | K=4096,N=12288 | K=4096,N=4096 |
|--------|----------------|---------------|
| 1-warp | 44.6 µs | 16.3 µs |
| 2-warp | 46.5 µs | 16.4 µs |
| Speedup | **0.96×** | **0.99×** |

Conclusion: doubling threads does NOT help. Kernel is memory-saturated.
Extra cross-warp reduction adds ~4% overhead. 1-warp remains optimal.

## Files Modified
- `src/kernels/fused_gate_up_int4.cu` — new fused gate+up kernel
- `src/kernels/fused_qkv_int4.cu` — new fused QKV kernel
- `bench/bench_gemv_2warp.cu` — new 2-warp microbench
- `bench/bench_ppl_int4_8b.cu` — added WDIR argv[1] parameter
- `bench/text_generate_int4_batched.cu` — wired fused QKV + gate+up (4 call sites)
- `server/inference_server_int4.cu` — wired fused QKV + gate+up (8 call sites)
- `server/inference_server_int4_batched.cu` — wired fused QKV + gate+up (2 call sites)
- `include/blackwell/kernels.h` — new declarations
- `CMakeLists.txt` — new kernel sources

## Symbol count: 189 → 203 (+14)
