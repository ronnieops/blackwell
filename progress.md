# Blackwell Project Progress

## 2026-06-09 — Research Complete

### Research Findings
- Read `include/blackwell/kernels.h` — 177 kernel symbols, full API documented
- Read `bench/text_generate_int4_batched.cu` (525 lines) — batched architecture
- Read `server/inference_server_int4.cu` (350 lines) — server implementation

### Key Kernel Signatures Documented
1. **INT4**: `gemv_int4_warp`, `gemv_int4_batched`, `quantize_int4`, `quantize_int4_batched`
2. **INT8**: `gemv_int8_warp` (production), `gemv_int8_batched`, `gemv_int8_gate_up`
3. **Attention**: `attention_decode_gqa`, `attention_decode_batched_gqa`
4. **Quantization**: `quantize_int8`, `fused_rmsnorm_quant_int8`, `fused_rmsnorm_batched`
5. **KV/RoPE**: `update_kv_cache`, `update_kv_cache_device`, `fused_rope_decode`

### Architecture Decisions
- Batched benchmark: M×buffer layout, separate KV caches per sequence
- Server: sequential prompt processing (no GPU batching)
- 36-layer decode loop with INT4 quantization at each projection

### Output
- Research report: `research.md` (8794 bytes)

## TODO
- [ ] No batched `apply_swiglu` — per-sequence loop required
- [ ] No batched `vector_add_fp32` — per-sequence loop required
- [ ] CUDA Graph integration deferred until head_norm/RoPE fusion