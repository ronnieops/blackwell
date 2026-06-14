# Production Cleanup & Tune — ALL COMPLETE

## Done
- **Phase A**: Qwen3-8B INT4 server verified. http_subprocess rebuilt. 3/3 requests, 0 errors, 96ms avg, 6713 MB GPU.
- **Phase E**: Removed weights_int4_qwen3_8b/ (5.8 GB), gguf_convert_new/old, context-fp16-residual.md. No stale compute_120.
- **Phase C**: CUDA Graph not worth (2.1% speedup, GEMV 92% bottleneck). Skip.
- **Phase B**: Created scripts/quantize_llama31_8b_int8.py for INT8 fallback. Not run (9.6 GB, ~4 t/s).
- **191** kernel symbols (+2 for gemv_fp32_int4_warp)

## Current State
- Production path: Qwen3-8B INT4 (56 t/s, PPL 21.82, 6.7 GB GPU)
- Llama INT4: abandoned (weight quant fidelity, PPL 268K)
- Llama INT8 fallback: script ready but not executed
- Server: int4_8b on port 8123, multi-model via start_servers.sh
