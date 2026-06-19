# Model Port Plan — High-Value Candidates

RTX 5060 Ti 16 GB. INT4 block-16 symmetric quantization (offset-binary, absmax/7 scales).
Existing GGUF bridge (`better-inference/gguf_convert`) handles dense transformers with Q4_K/Q5_0/Q6_K/Q8_0 → INT4 block-16 conversion.

---

## Priority 1: Gemma 4 12B QAT Q4_0 — BLOCKED (GGUF norm corruption)

**Model**: `google/gemma-4-12B-it-qat-q4_0-unquantized` (and -assistant variant)
**Status**: BLOCKED — Google QAT GGUF has corrupted FP32 norm weights (values 4e17 instead of ~1.0).
See AGENTS.md "Gemma4 GGUF all corrupted" for details.

**To revive**: (a) re-export QAT from source with non-corrupt norms, (b) use non-QAT Gemma 4 GGUF
(unsloth variants), or (c) get raw FP32 weights from HuggingFace safetensors.

### Architecture (Gemma 4 12B, confirmed from HF safetensors)
| Parameter | Value | Notes |
|-----------|-------|-------|
| Hidden (H) | **3840** | NOT 4608 (plan was written before config confirmed) |
| Intermediate (I) | 15360 | |
| Layers (NL) | **48** | NOT 40 |
| Q heads (nqh) | 16 | |
| KV heads (nkv) | 8 | |
| Head dim (hd) | **256** | NOT 128 |
| Vocab (V) | 262144 | Same tokenizer as Gemma 3/4 |
| RoPE θ | 10000.0 | Same base as Gemma |
| Activation | GeGLU | **Differs from Qwen3 SwiGLU** |
| Norm | RMSNorm | Same |
| QK norm | Yes | Per-layer, native dims (256 SWA, 512 FA) |
| Sliding window | Every 6th layer | Layer 5/11/17/23/29/35/41/47 are FA (full attention) |
| Tied embeddings | Yes | embed_tokens == lm_head (no separate lm_head) |
| Final logit softcapping | cap=30.0 | |

### What needs changing (if unblocked)
1. **GeGLU kernel** — `apply_geglu.cu` exists. Uses `gelu_pytorch_tanh` (tanh approx, HF-compatible).
2. **GGUF parser** — Gemma 4 uses `gguf` metadata key `gemma4.*`. Need to handle FA layers (double Q heads, K=V, no v_proj).
3. **Config constants** — H=3840, I=15360, NL=48, nqh=16, nkv=8, hd=256, V=262144
4. **FA layer handling** — Layers 5/11/17/23/29/35/41/47: double Q heads (32×256=8192), K=V (2 heads=512), no v_proj, o_proj input 8192→3840
5. **QK norms** — Per-layer native dims (256 SWA, 512 FA). No replication needed.
6. **Tokenizer** — SentencePiece. Reuse `scripts/gemma_wrapper.py` pattern.

### Effort: 4-6 hours (if unblocked)
- GeGLU kernel: DONE (`src/kernels/apply_geglu.cu`)
- GGUF parser: 1-2 hours (FA layer handling)
- Bench: 1 hour
- Server: 30 min
- Validation (PPL + coherence): 1 hour
- GeGLU kernel: 30 min (simple element-wise, 1 warp block)
- GGUF parser: 1-2 hours (map tensor names, handle gemma4 keys)
- Bench: 1 hour
- Server: 30 min
- Validation (PPL + coherence): 1 hour

---

## Priority 2: Qwen3.6-27B / Qwen3.5-27B

**Model**: `Qwen/Qwen3.6-27B` or `Qwen/Qwen3.5-27B`
**Why**: Same architecture family as current Qwen3-8B. Minimal code changes.
**Size**: ~14.5 GB INT4 (FP16 scales). Tight fit — 1.5 GB margin. No M>1.

### Architecture (dense Qwen3.6-27B)
| Parameter | Value (est) | Notes |
|-----------|-------------|-------|
| Hidden (H) | 5120 | 4096 × 1.25 |
| Intermediate (I) | 14336 or 20480 | Check config.json |
| Layers (NL) | 64 | ~8B × 1.8 |
| Q heads (nqh) | 40 | GQA ratio 5:1 |
| KV heads (nkv) | 8 | Same ratio as 8B (32:8) |
| Head dim (hd) | 128 | Same |
| Vocab (V) | 152064 | Updated from 151936 |
| RoPE θ | 1000000.0 | Same as Qwen3 |
| Activation | SwiGLU | Same — no change needed |
| Norm | RMSNorm | Same |
| QK norms | Yes | Same as Qwen3 |
| MTP head | Yes | Multi-Token Prediction — skip for M=1 (predict only next token) |

### What needs changing
1. **Config block** — H=5120, I=?? (read from config.json), NL=64, nqh=40, nkv=8, V=152064
2. **Tokenizer** — Qwen3.6 uses tiktoken BPE with vocab 152064 (same format, larger). Need `prepare_tokenizer.py` update: download new tokenizer.json, write tokenizer_data.bin.
3. **GGUF parser** — Qwen3.6 may use same GGUF metadata keys as Qwen3 (`qwen3.*`). Verify `better-inference/gguf_test` against a GGUF.
4. **Weight size** — lm_head: 5120×152064 = 778M params × 2 bytes (INT4) = 389 MB. Embed: similar. **Total INT4 ≈ 14.5 GB** — will only fit with FP16 scales (no FP32 scales variant).
5. **MTP head** — Qwen3.6 has MTP (next-token-prediction) head. For decode, only standard lm_head is needed for next-token prediction. MTP head can be loaded but not used on M=1 decode.
6. **VRAM budget check**:
   - INT4 weights: ~14.5 GB
   - KV cache (M=1, 64 layers, 4096 ctx): 64 × 8 × 4096 × 128 × 4 = 1.1 GB
   - Buffers + overhead: ~0.4 GB
   - **Total: ~16.0 GB** — at limit. Need MAXSEQ=2048 or FP16 KV cache for margin.

### Files to create/modify
```
scripts/prepare_tokenizer_qwen36.py     — NEW (or update existing)
bench/text_generate_int4_qwen36_27b.cu — NEW bench
server/inference_server_int4_qwen36.cu  — NEW server
CMakeLists.txt                          — Add new bench target
better-inference/gguf.h                 — Handle qwen3.6 metadata (if different)
scripts/quantize_awq_int4_8b.py         — Fork for 27B dims
```

### Effort: 4-8 hours
- Config + tokenizer: 1 hour
- Bench adaptation: 2 hours (copy 8B bench, change dims)
- Server: 1 hour
- Quantization: 2 hours (download weights, run AWQ quant)
- Validation: 1 hour
- VRAM tuning: 0.5 hours (MAXSEQ reduction if needed)

**Risk**: VRAM may not fit. Mitigation: use FP16 KV cache (halve KV memory) or reduce MAXSEQ to 2048 (acceptable for typical chat).

---

## Priority 3: Gemma 4 E2B (5B) / E4B (8B)

**Model**: `google/gemma-4-E2B-it` or `google/gemma-4-E4B-it`
**Why**: Small dense models, fast throughput, fits comfortably.
**Size**: E2B ~3 GB INT4, E4B ~5 GB INT4. Plenty of room for M>1.

### Architecture (Gemma 4 E4B-8B dense)

Haven't confirmed exact config yet — need to inspect config.json.
E2B likely: H=2560, I=10240, NL=32, V=262144
E4B likely: H=3072, I=12288, NL=40, V=262144

Same Gemma 4 family characteristics as Priority 1 (GeGLU, sliding window, QK norms).

### What needs changing
Same as Priority 1 (GeGLU kernel, sliding window, SentencePiece) but smaller model.
GeGLU kernel is the common dependency — once written for Priority 1, applies here trivially.

### Files to create/modify
Same as Priority 1 + new config blocks.

### Effort: 2-4 hours (after GeGLU kernel exists)
- Config: 30 min
- Bench: 1 hour
- Server: 1 hour
- Validation: 1 hour

---

## Priority 4: Llama 3.2 3B

**Model**: `meta-llama/Llama-3.2-3B-Instruct`
**Why**: Already have GGUF converter working for 1B. 3B is trivial extension.
**Size**: ~1.8 GB INT4. Extremely fast (200+ t/s projected).

### Architecture
| Parameter | Value |
|-----------|-------|
| Hidden (H) | 3072 |
| Intermediate (I) | 8192 |
| Layers (NL) | 28 |
| Q heads (nqh) | 24 |
| KV heads (nkv) | 8 |
| Head dim (hd) | 128 |
| Vocab (V) | 128256 |
| RoPE θ | 500000 |
| Activation | SwiGLU |
| Norm | RMSNorm |
| QK norms | No |

### What needs changing
1. **Config block** — Simple config constant change
2. **Tokenizer** — Already have tiktoken for Llama 3.2 1B. Same format, same BPE type.

### Files to create/modify
```
bench/text_generate_llama32_3b.cu — NEW (copy 1B bench, change H/NL/NQ)
server/inference_server_llama32_3b.cu — NEW (or extend existing)
CMakeLists.txt — Add new target
```

### Effort: 1-2 hours
- Config + bench: 30 min
- Server: 30 min
- Validation: 30 min

---

## Table: Summary

| Priority | Model | INT4 Size | VRAM Margin | New Kernels | Effort | Risk |
|----------|-------|-----------|-------------|-------------|--------|------|
| 1 | Gemma 4 12B QAT | ~14.3 GB | 1.7 GB | GeGLU | 4-6h | Low (GeGLU straightforward) |
| 2 | Qwen3.6-27B | ~14.5 GB | ~0 GB | None | 4-8h | High (VRAM at limit) |
| 3 | Gemma 4 E2B/E4B | 3-5 GB | 11+ GB | GeGLU (shared) | 2-4h | Very low |
| 4 | Llama 3.2 3B | ~1.8 GB | 14+ GB | None | 1-2h | Trivial |

### Dependencies
- **GeGLU kernel** is prerequisite for all Gemma 4 ports. Once written, shareable.
- **VRAM stress for Qwen35.6-27B** may require reducing MAXSEQ or using FP16 KV cache.
- **Qwen3.6 tokenizer** update needed before any Qwen3.6/3.5 27B port.

### Recommended order of execution
1. **GeGLU kernel** (common dependency, pure CUDA element-wise, 30 min)
2. **Gemma 4 12B QAT** (highest quality/effort ratio)
3. **Llama 3.2 3B** (quick win while Gemma 4 ideas percolate)
4. **Gemma 4 E2B/E4B** (trivial after GeGLU exists)
5. **Qwen3.6-27B** (highest effort, VRAM risk — do last)
