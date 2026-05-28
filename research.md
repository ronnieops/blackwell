# Research: INT8 Quality / HF Comparison Feasibility

## Summary

Full Python reference that reproduces the CUDA INT8 28-layer pipeline is straightforward — 1-layer pipeline already exists and is verified exact. Extending to 28 layers is mechanical repetition. Parallel HF BF16 baseline via `transformers` is also straightforward for measuring quantization noise. Main obstacle: Python must replicate BPE tokenizer logic exactly (or reuse the same token IDs from C++), and the 28-layer run will reproduce the same garbled output the CUDA pipeline already shows — that's expected quantization noise, not a bug.

## Findings

1. **1-layer Python reference exists and verified** — `bench/validate_pipeline.py` implements exact match to CUDA INT8 pipeline: block_quant (16-element absmax → scale = max(absmax/127, 1e-9), clip(round), saturate to [-127,127]), RMSNorm (mean(x²) → rsqrt), INT8 GEMV (DP4A block dot with dual-scale product), GQA attention (single-token = copy V per group). Validated: max diff=4.7e-7, cosine sim=1.00000002. [Source](file:///mnt/data/dev/projects/blackwell/bench/validate_pipeline.py)

2. **INT8 quantization error measurement exists** — `bench/verify_int8_accuracy.py` reads BF16 weights from `model.safetensors`, loads INT8 weights, dequantizes, computes per-weight and per-GEMV relative error. Covers all 7 weight matrices for layer 0. [Source](file:///mnt/data/dev/projects/blackwell/bench/verify_int8_accuracy.py)

3. **HF model ready for reference** — `/mnt/data/ai/hf/qwen3-1.7b-base/` has full `model.safetensors` (BF16), `config.json`, `tokenizer.json`. Config: Qwen3ForCausalLM, H=2048, ID=6144, 28 layers, nqh=16, nkv=8, hd=128, vocab=151936, tie_word_embeddings=True, rope_theta=1000000, rms_norm_eps=1e-6. `transformers` can load and run directly (BF16). [Source](file:///mnt/data/ai/hf/qwen3-1.7b-base/config.json)

4. **28-layer extension is mechanical** — Each layer uses identical operations. Weight files are per-layer: `{layer}_self_attn.q_proj.int8_t`, `{layer}_input_layernorm.f32`, etc. The CUDA code at `bench/text_generate.cu` loops `for(int l=0;l<NL;l++)` doing the same 10 steps per layer. Python just needs to iterate layers with per-layer file paths. Same embed tokens reused (tied). [Source](file:///mnt/data/dev/projects/blackwell/bench/text_generate.cu)

5. **Tokenization mismatch risk** — CUDA uses custom `BpeTokenizer` loaded from `tokenizer_data.bin` (prepared by `scripts/prepare_tokenizer.py`). Python via `transformers` would use HuggingFace tokenizer from `tokenizer.json`. If tokenization diverges (byte-level BPE quirks, added token handling), input IDs won't match and outputs can't be compared token-by-token. Solution: dump token IDs from C++ to a file, feed same IDs to Python. [Source](file:///mnt/data/dev/projects/blackwell/scripts/prepare_tokenizer.py)

6. **KV cache for multi-step generation** — Current `validate_pipeline.py` does single-token, seq_pos=0 only (GQA attention just copies V). Full autoregressive requires multi-position KV cache with `update_kv_cache` → `attention_decode_gqa` per step. Python must track K/V per layer per position per head, rebuild on each step. ~0.5 MB per layer (8 heads × 128 × 4096 × 4 bytes = 16 MB total). [Source](file:///mnt/data/dev/projects/blackwell/bench/text_generate.cu)

7. **Processing steps per layer (Python must replicate all)**:
   - Embedding: INT8 row lookup → dequant host-side (same as CUDA does with `emb.d[tid*H + d] * emb.sc[scale_idx]`)
   - RMSNorm + INT8 quant (fused: 1 operation in CUDA, 2 in Python)
   - 3× GEMV QKV (K=2048, N_proj depending on head count)
   - Q/K head norms (per-head RMSNorm — separate step, not fused)
   - RoPE on Q and K (cos/sin multiplication per pair)
   - KV cache write + GQA attention decode
   - Quantize attn output → Wo GEMV → residual
   - Post-attention RMSNorm + quant → gate+up GEMV ×2 → SwiGLU → quant → down GEMV → residual
   - Final RMSNorm + quant → lm_head GEMV (tied embed weights) → argmax

8. **Expected output: quantization noise, not bug** — AGENTS.md states "28-layer output still garbled = INT8 quantization noise accumulation, not bugs." Python reference will reproduce same garbled output. The value is: (a) confirming CUDA correctness end-to-end, (b) measuring per-layer noise accumulation vs BF16 reference, (c) identifying which layer or operation introduces most noise. [Source](file:///mnt/data/dev/projects/blackwell/AGENTS.md)

## Sources

- Kept: `bench/validate_pipeline.py` — exact 1-layer INT8 Python reference, verified against CUDA
- Kept: `bench/verify_int8_accuracy.py` — measures per-weight INT8 quantization error vs BF16
- Kept: `bench/text_generate.cu` — full 28-layer decode pipeline, weight loading, tokenizer, sampling
- Kept: `/mnt/data/ai/hf/qwen3-1.7b-base/config.json` — model architecture params
- Kept: `scripts/prepare_tokenizer.py` — tokenizer data dump format for BPE
- Kept: `scripts/extract_norms.py` — extracts per-layer RMSNorm weights to `.f32` files
- Kept: `include/blackwell/kernels.h` — INT8 kernel API (gemv_int8, fused_rmsnorm_quant_int8, pack_int8, etc.)
- Dropped: None — all sources directly relevant

## Gaps

1. **Multi-position KV cache in Python not implemented** — Current validate_pipeline.py only handles seq_pos=0. Need per-layer, per-head, per-position K/V storage for multi-step generation. Straightforward but ~200 lines Python.

2. **Token ID correspondence not verified** — Need to dump token IDs from C++ `text_generate` and feed same IDs to Python reference to ensure identical input. Tokenizer implementation mismatch is the #1 risk for meaningless comparison.

3. **No BF16 baseline output file exists** — Need to run HF model through `transformers` and dump per-layer intermediates. Can do with ~50 lines Python using `AutoModelForCausalLM`.

4. **Per-layer noise accumulation not measured** — verify_int8_accuracy.py only does layer 0. Need all 28 layers to see where noise compounds.

5. **lm_head weights not in INT8 directory** — embed_tokens used for both embedding and lm_head (tied). CUDA uses `d_emb_d` (INT8) for both. Need to verify lm_head path uses same weights correctly in Python.

6. **RoPE parameters**: Qwen3 uses rope_theta=1000000 (not 10000). `text_generate.cu` hardcodes `powf(10000.0f, ...)` — this is WRONG for Qwen3. Needs investigation: does this degrade output quality further? [Source](file:///mnt/data/dev/projects/blackwell/bench/text_generate.cu#L74)

## Approach

**Phase 1: Token ID bridge** (1-2 hours)
- Add `--dump-tokens` flag to `text_generate.cu` that writes input_ids and generated token IDs to file
- Read token IDs in Python, feed same IDs to both INT8 reference and HF BF16 model
- Verify tokenizer agreement on first 50 tokens

**Phase 2: Extend validate_pipeline.py to 28 layers** (2-3 hours)
- Load all INT8 weights per layer (loop 0..27)
- Loop same 10-step pipeline per layer
- Add multi-position KV cache (grows with each step)
- Add final norm + lm_head + argmax
- Compare CUDA output at each layer (dump CUDA intermediates via new kernel)
- Already have per-layer RMSNorm weights (`{layer}_input_layernorm.f32`)

**Phase 3: HF BF16 baseline** (1 hour)
- Use `transformers` with `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16)`
- Run same prompt, same token IDs
- Dump per-layer outputs (hidden states)
- Compute per-layer cosine similarity, max diff, SNR vs INT8 Python reference

**Phase 4: Accuracy analysis** (1 hour)
- Plot per-layer noise accumulation (layer 0 to 27)
- Identify which operation causes most degradation (QKV GEMV? Down GEMV? Quantization step?)
- Report SNR, perplexity impact estimate, qualitative output comparison
