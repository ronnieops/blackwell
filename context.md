# Code Context

## Repo State

**Last commit**: `09b8217 Handoff: session 35 complete. AGENTS.md audit + HANDOFF.md refresh`
**Branch**: main (implied by commit history)

### Uncommitted Changes (7 files)
1. `AGENTS.md` — 96 lines added (updated findings/docs)
2. `HANDOFF.md` — 211 lines changed (refresh)
3. `context.md` — deleted (159 lines, will be overwritten)
4. `research.md` — deleted (137 lines)
5. `src/kernels/fused_residual_norm_int4.cu` — 14 lines changed (likely bug fixes from session 35)
6. `src/kernels/gemv_int4_batched.cu` — 8 lines changed (grid bug or sign-extension fix)
7. `src/kernels/gemv_int8.cu` — 6 lines changed (sign-extension fix)

### Build Status
- **Build: PASS** — `cmake --build build --parallel` succeeds, `[100%] Built target blackwell_kernels`
- **Library symbols**: 177 (matches expected count from AGENTS.md)

### Key Binaries (all exist)
| Binary | Size | Date |
|--------|------|------|
| `bench/text_generate_int4` | 3.8 MB | Jun 2 20:53 |
| `bench/text_generate` | 3.6 MB | Jun 1 08:25 |
| `bench/decode_int4_batched_attn` | 2.1 MB | Jun 2 20:34 |

### INT4 Weight Directories
- `weights_int4_qwen3_1.7b/` — populated (`.int4_t` + `.scale_t` files per layer)
- `weights_int4_qwen3_8b/` — populated (same structure)

## Key Benchmark Numbers (from AGENTS.md §4)

| Metric | Value | Notes |
|--------|-------|-------|
| INT4 1.7B M=1 | 261.7 t/s | 89% of Q4_K_M (293.4). Post grid-bug fix |
| INT4 1.7B M=8 | 3586.4 t/s | 1222% of Q4_K_M |
| INT8 1.7B M=1 | 181.5 t/s | 14 kernels/layer |
| INT8 1.7B M=8 batched | 324.3 t/s | 111% of Q4_K_M |
| llama.cpp Q4_K_M 1.7B | 293.4 t/s | Baseline |
| llama.cpp Q4_K_M 8B | 82.56 t/s | Baseline |
| INT4 8B M=1 | 342.9 t/s | UNVERIFIED (pre grid-bug fix) |
| INT4 8B M=8 | 5640.3 t/s | UNVERIFIED |

**Note**: AGENTS.md has conflicting numbers. Section 11 (session 34/35) claims INT4 M=1 = **612.8 t/s (209% Q4_K_M)** and M=8 = **11284.5 t/s**. Section 4 claims 261.7 / 3586.4. The 612.8 number appears to be from the pre-grid-bug-fix era (session 34/35 header says "612.8 t/s" but grid bug was discovered in session 37). The corrected post-fix numbers in §4 are authoritative: **261.7 t/s M=1, 3586.4 t/s M=8**.

## Architecture

### Production INT4 Decode Pipeline (per layer, 17 kernels)
1. **gemv_int4_batched** ×7: Q, K, V, O, gate, up, down projections
2. **quantize_int4**: FP32→INT4 quant
3. **attention_decode_batched_gqa**: Batched GQA attention
4. **fused_residual_norm_int4**: residual + RMSNorm + INT4 quant (3→1)
5. **fused_swiglu_quant_int4**: SwiGLU + INT4 quant (2→1)
6. **fused_residual_norm_int4_fp32out**: residual + RMSNorm + INT4 + FP32 output
7. **update_kv_cache** ×M (serial per-seq)

### Weight Flow
`FP32 safetensors` → `quantize_generic.py` → `weights_int4_*/{layer}_{weight}.int4_t + .scale_t`
→ `transpose_int4_weights` at load time → row-major for GEMV

### Critical Kernel Files
- `src/kernels/gemv_int4_batched.cu` — Batched INT4 GEMV (production)
- `src/kernels/gemv_int8.cu` — INT8 GEMV + INT4 sign-extension helpers
- `src/kernels/fused_residual_norm_int4.cu` — Fused residual+norm+quant

## Constraints & Risks

1. **hashcat on GPU-0** — kills throughput 45%. Must `killall hashcat` before benchmarks.
2. **INT4 text quality garbage** — 4-bit symmetric quant noise compounds across 28 layers. Structural pipeline correct but output garbled.
3. **8B INT4 numbers UNVERIFIED** — pre-date grid bug fix. Need re-run.
4. **Uncommitted kernel changes** — 3 `.cu` files modified (bug fixes). Not committed yet.
5. **quantize_generic.py has seek(0) bug** — re-quantize weights after any batch run.

## Start Here

Open `bench/decode_int4_batched_attn.cu` — the production INT4 benchmark. Contains full 28-layer pipeline with all kernel calls. 19K lines, the authoritative performance reference.
