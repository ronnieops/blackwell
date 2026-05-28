# Code Context — Blackwell Repo State (2026-05-28)

## 1. Python env
- `.venv/bin/python3` exists
- `torch`, `transformers`, `numpy` — all import OK

## 2. Weights
- `weights_int8_bf16/*.int8_t` — 197 INT8 weight files
- `weights_int8_bf16/*.f32` — 58 FP32 norm/scale files
- Covers 28-layer Qwen3-1.7B: 7 files per layer (q,k,v,o,gate,up,down) × 28 + embeddings + LM head
- All present, transposed [N×K] layout

## 3. Built binaries
| Binary | Size | Date |
|--------|------|------|
| `bench/inference_server` | 2.0 MB | May 28 11:41 |
| `bench/text_generate` | 1.9 MB | May 28 10:42 |
| `bench/validate_pipeline` | 1.8 MB | May 28 10:43 |

All three binaries exist and recent.

## 4. Library
- `build/libblackwell_kernels.a` — **1.3 MB**, built May 28 10:40
- ~40 public kernel wrappers (by `nm` count)

## 5. Build system (CMakeLists.txt:2,31)
- `project()` at line 2 (CXX + CUDA)
- `CUDACXX` env set at line 31 via `set(ENV{CUDACXX} "/usr/local/cuda-12.8/bin/nvcc")`
- ⚠️ Line 21 comment admits `set(ENV{})` happens AFTER `project()` — CMake ID step uses host `nvcc` (12.0) before override takes effect. Requires explicit `CUDACXX=/usr/local/cuda-12.8/bin/nvcc cmake -B build` on first configure.

## 6. Config constants (`include/blackwell/config.h`)
- **kGEMMTileM/N/K** = 128×128×64 (large CTA), 64×64×64 (small CTA)
- **kGEMMWarps** = 8 (4×2), kGEMMSmallWarps = 4 (2×2)
- **kWMMAFragM/N/K** = 16×16×16
- **kGEMVTileM/N/K** = 8×64×64
- **kINT8BlockSize** = 16, **kFP4BlockSize** = 16
- **kMaxBatchSize** = 8
- **No model-specific dims** (H=2048, nqh=16, nkv=8, hd=128, V=151936) — these live in `text_generate.cu:68-71`
- **SM**: `kSMArchitecture=120`, comments note `120a` arch required for FP4 MMA

## 7. Public API (`include/blackwell/kernels.h`)
41 public functions total, including:

| Group | Count | Key functions |
|-------|-------|---------------|
| GEMM/GEMV | 12 | `gemm_fp4_block_scaled`, `gemv_fp4/_v2`, `gemv_int8/_splitk/_batched`, `dispatch_matmul` |
| Quantize | 4 | `pack_fp4`, `unpack_fp4`, `pack_int8`, `transpose_fp4/int8_weights` |
| Fused epilogues | 6 | `fused_rmsnorm`, `fused_rope`, `apply_swiglu`, `vector_add_fp32`, `fused_rmsnorm_pack`, `fused_rmsnorm_quant_int8` |
| Fused proj GEMV | 4 | `fused_gate_up_gemv/_v1`, `fused_qkv_gemv`, `fused_o_norm_pack` |
| Attention | 6 | `attention_decode`, `attention_decode_gqa`, `attention_fp4`, `attention_prefill`, `update_kv_cache`, `load_kv_cache_qkgv` |
| CUDA Graph | 4 | `capture/launch/destroy_decode_graph`, `update_decode_seq_pos` |
| Other | 4 | `coalesced_copy`, `gemv_fp4_from_fp4`... |

⚠️ **Unimplemented:** `gemv_fp4_splitk`, `gemv_fp4_v3`, `gemv_fp4_batched` — declared but no definition. Cause `phase_a.cu` link failure.

## 8. Text generate (`bench/text_generate.cu`)
- **Model**: Qwen3-1.7B (28 layers)
- **Dims**: H=2048, QD=2048, KV=1024, ID=6144, nqh=16, nkv=8, hd=128, V=151936
- **MAXSEQ** = 4096
- **NL** = 28 (line 199), loads all 28 layers (line 221-222 loop)
- KV cache: `d_kc` and `d_vc` sized `NL*nkv*MAXSEQ*hd*4` (28×8×4096×128×4 bytes)
- Also loads `qk_h` from file (28×2×128 floats) — RoPE cos/sin cache

## 9. Attention stubs (`src/kernels/attention.cu`)
- `attention_fp4` and `load_kv_cache_qkgv` return `cudaErrorNotReady` (lines 144-151)
- `attention_prefill` — flash-style kernel in development, smem K+V = 32KB, Q in regs
- working: `attention_decode`, `attention_decode_gqa`, `update_kv_cache`

## Key files for changes

| File | Role |
|------|------|
| `include/blackwell/kernels.h` | Start here — public API signatures |
| `include/blackwell/config.h` | Tile sizes, architecture constants |
| `bench/text_generate.cu` | End-to-end INT8 decode pipeline |
| `src/kernels/attention.cu` | Attention prefill + stub implementations |
| `CMakeLists.txt` | Build system caveats (CUDACXX ordering) |
