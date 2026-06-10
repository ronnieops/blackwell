# better-inference — Design

Bridge GGUF model format with Blackwell custom CUDA kernels.

## Goal

C++ library that loads GGUF models → converts to blackwell INT4 format → runs with blackwell's batched INT4 kernels. Brag: "Fastest INT4 inference on RTX 5060 Ti — 110 t/s batched 8B."

## Architecture

```
GGUF file (.gguf)
  │
  ▼
GGUF Parser (new)       ← lib only, no llama.cpp dep
  │  Reads tensors, metadata, tokenizer
  │  Tensor layout: name → data pointer + shape + type
  ▼
Tensor Converter (new)
  │  Q4_K_M → FP32 → INT4 block-16 (or passthrough for FP16)
  │  Config extraction: nqh, nkv, hd, rope_theta, hidden_size, layers
  ▼
Blackwell Weight Format  ← existing DevW4 / DevW8 structs
  │  .int4_t + .scale_t in GPU memory
  ▼
Blackwell Kernels        ← existing gemv_int4_batched, attention_decode_batched_gqa
  │  No kernel changes needed
  ▼
Inference Server         ← existing http_subprocess + inference_server
  │  Or new server binary
```

## Phase Plan

### Phase 1: GGUF Parser
- Read GGUF header, metadata key-value pairs
- Tensor info table: name, shape, type, offset
- Support: Q4_K_M, Q8_0, F16, F32
- Tokenizer extraction (BPE, tiktoken, sentencepiece models)

### Phase 2: Tensor Converter (Qwen3)
- Map GGUF tensor names → blackwell file names
- `model.layers.0.self_attn.q_proj.weight` → `0_self_attn.q_proj`
- Dequantize Q4_K_M → FP32 per tensor
- Re-quantize FP32 → INT4 block-16 (reuse `quantize_per_row_int4_sym` from Python)
- Load norms, rope config, head counts from GGUF metadata

### Phase 3: Llama 3.1 8B support
- Config mapping: Llama uses different tensor names, RoPE config, RMSNorm eps
- GQA: nqh=32, nkv=8, hd=128 (same as Qwen3-8B — kernels work as-is)
- Tokenizer: tiktoken (different from BPE)

### Phase 4: AWQ calibration service
- On first load of a GGUF → INT4 conversion, run AWQ calibration
- Cache calibrated .int4_t files alongside GGUF
- Future loads skip conversion

## No llama.cpp dependency

We implement our own GGUF parser. GGUF format is simple: header → metadata KV pairs → tensor info → aligned tensor data. ~500 lines.

## Existing code reuse

| Component | Status |
|-----------|--------|
| `gemv_int4_batched` M=1-16 | ✅ Done |
| `attention_decode_batched_gqa` | ✅ Done |
| `update_kv_cache` | ✅ Done |
| BPE tokenizer | ✅ Done |
| tiktoken tokenizer | ❌ Need — for Llama |
| HTTP server | ✅ Done (http_subprocess) |
| AWQ calibration | ✅ Done (Python script) |
