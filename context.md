# Code Context

## Files Retrieved
1. `build/libblackwell_kernels.a` — static kernel library, 179 symbols (was 177 per AGENTS.md)
2. `bench/text_generate_int4_qwen3_8b.cu` (lines 1-16445) — 8B INT4 end-to-end benchmark
3. `bench/text_generate_int4_batched.cu` (lines 1-21908) — NEW batched INT4 benchmark (untracked)
4. `server/inference_server_int4.cu` (lines 1-17977) — INT4 inference server
5. `weights_int4_qwen3_8b/` — 582 files, 5.8 GB INT4 weights
6. `AGENTS.md` vs `HANDOFF.md` — AGENTS.md is canonical, HANDOFF.md is a diff variant

## Key Code
- **Kernel count**: 179 (AGENTS.md says 177 — +2 new kernels added)
- **New kernel symbols** (not in AGENTS.md baseline):
  - `fused_swiglu_quant` — fused SwiGLU+quant wrapper
  - `fused_swiglu_quant_kernel` — inner kernel
- **Production INT4 kernels**: `gemv_int4_warp`, `gemv_int4_batched`, `quantize_int4`, `unpack_int4_fp32`
- **INT4 weight path**: `weights_int4_qwen3_8b/` — symmetric INT4, 5.8 GB

## Architecture
```
INT4 decode path (bench/text_generate_int4_qwen3_8b.cu):
  embed → quantize_int4 → gemv_int4_warp → RMSNorm → repeat×36

Server path (server/inference_server_int4.cu):
  embed → quantize_int4 → gemv_int4_warp → RMSNorm → repeat×36
  HTTP wrapper: http_subprocess forks inference_server_int4

Batched path (bench/text_generate_int4_batched.cu):
  M sequences → batched GEMV (gemv_int4_batched) → per-seq RMSNorm/attention
  Note: M>1 reported broken in AGENTS.md ("output divergence, root cause unknown")
```

## Start Here
`bench/text_generate_int4_qwen3_8b.cu` — canonical INT4 8B decode loop. Use as reference for correct single-sequence behavior. Compare against `bench/text_generate_int4_batched.cu` for batched divergence debugging.

## Discrepancies Found
1. **Kernel count mismatch**: 179 vs 177 documented — 2 new `fused_swiglu_quant*` symbols added
2. **Untracked file**: `bench/text_generate_int4_batched.cu` exists but not committed
3. **AGENTS.md stale**: Server version shows "Garbled output ❌" for 8B INT8 benchmark — this was FIXED (coherent output now)
4. **HANDOFF.md**: Variant doc with meta-prompt header, not canonical
5. **Dirty git state**: 8 modified files not staged — likely session work-in-progress

## Supervisor coordination
No blocking issues. Repo state consistent with recent session work.