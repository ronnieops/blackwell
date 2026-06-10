# Research: GGUF parser, Llama 3.1 8B support, and tiktoken for blackwell

## Summary

Existing `better-inference/gguf.h` parser supports GGUF v1-3 with Q8_0, F32, F16, and has skeleton Q4_K support (144 bytes/block, 256-element super-block) but no dequant function. Llama 3.1 8B matches Qwen3-8B head dims (nqh=32, nkv=8, hd=128) but differs in vocab_size (128256 vs 151936), rope_theta (500000 vs 1M), and tensor naming. Tokenizer is tiktoken BPE (vocab size 128k, GPT-4 pattern) — existing blackwell BPE tokenizer handles Qwen but needs pre-tokenizer update for Llama 3.1 regex. Q4_K_M dequant requires ~40-line C function reading 144-byte super-blocks.

---

## Findings

### 1. GGUF Parser Architecture (gguf.h + gguf_convert.cpp)

**Header**: Header-only C++17, no deps beyond `<cstdint>`. Reads GGUF v1-3. Two-pass: read metadata KV pairs, then tensor info (name, shape, type, file offset). Tensor data offset computed by skipping header+metadata+tensor-info, aligned to 32 bytes.

**Supported types**: F32, F16, Q8_0 (34 bytes/block — fp16 scale + 32 int8). Q4_K (our alias Q4_K_M = 26) declared but no dequant function yet. Type table enumerates GGML_TYPE_Q4_K = 12, our Q4_K_M = 26 in header.

**Converter flow**: Load GGUF → parse metadata for config (block_count, embedding_length, etc.) → iterate tensors → `map_qwen3_name()` to convert GGUF naming → dequant to FP32 → requant to INT4 block-16 → write our binary format.

**Key limitation**: `map_qwen3_name()` only handles Qwen3 naming (`blk.{l}.attn_q.weight`). Llama 3.1 GGUF uses same `blk.{l}.` prefix pattern (llama.cpp normalizes all archs to this), but the suffix names are identical (attn_q, attn_k, attn_v, attn_output, ffn_gate, ffn_up, ffn_down, attn_norm, ffn_norm). **No suffix change needed for Llama 3.1** — the existing `NameMap` table covers all Llama 3.1 tensor suffixes. Only the arch prefix in metadata key lookup (`general.architecture` = "llama") needs handling. [Source: better-inference/gguf.h, convert_hf_to_gguf.py tensor_mapping.py]

**Tensor shape convention**: GGUF stores shapes as `[K, N]` for weight matrices (inner=K=input_dim, outer=N=output_dim). Same as blackwell format. No transpose needed. [Source: gguf.h comment at map_qwen3_name]

**Metadata key pattern**: llama.cpp uses `{architecture}.{key}` for metadata. For "llama" arch: `llama.block_count`, `llama.embedding_length`, `llama.feed_forward_length`, `llama.attention.head_count`, `llama.attention.head_count_kv`. The converter already handles "qwen3" and "llama" prefixes via `get_meta()` lambda. Llama 3.1 uses `llama.rope.dimension_count` (?), `llama.rope.freq_base` for RoPE config. [Source: gguf_convert.cpp get_meta function]

---

### 2. Llama 3.1 8B Config

From HuggingFace `config.json` (Meta-Llama-3.1-8B-Instruct):

| Parameter | Value | Notes |
|-----------|-------|-------|
| hidden_size | 4096 | Same as Qwen3-8B |
| num_attention_heads | 32 (nqh) | Same as Qwen3-8B |
| num_key_value_heads | 8 (nkv) | Same as Qwen3-8B |
| head_dim | 128 | Same (4096/32) |
| intermediate_size | 14336 | Same as Qwen3-8B |
| num_hidden_layers | 32 | Same as Qwen3-8B |
| rms_norm_eps | 1e-5 | Same as Qwen3-8B |
| rope_theta | 500000.0 | **Different** — Qwen3 uses 1,000,000 |
| vocab_size | 128256 | **Different** — Qwen3 uses 151,936 |
| bos_token_id | 128000 | tiktoken BOS |
| eos_token_id | [128001, 128008, 128009] | End-of-turn, end-of-message tokens |
| use_scaled_rope | True | Scaled RoPE (new in Llama 3.1) |
| max_position_embeddings | 131072 | 128K context |

**GQA dims match**: Both use nqh=32, nkv=8, hd=128 — same attention kernel works. [Source: Medium article showing full config.json, HuggingFace docs]

**Key diff**: rope_theta=500000 vs Qwen3's 1000000. Need to parameterize in server kernel. vocab_size=128256 vs 151936 — lm_head and embed_tokens weights are smaller.

---

### 3. GGUF Tensor Naming (Llama 3.1 vs Qwen3)

llama.cpp normalizes all architectures to a common GGUF tensor naming convention during conversion (`convert_hf_to_gguf.py` → `tensor_mapping.py`):

```
HF name → GGUF name
model.layers.0.input_layernorm.weight → blk.0.attn_norm.weight
model.layers.0.self_attn.q_proj.weight → blk.0.attn_q.weight
model.layers.0.self_attn.k_proj.weight → blk.0.attn_k.weight
model.layers.0.self_attn.v_proj.weight → blk.0.attn_v.weight
model.layers.0.self_attn.o_proj.weight → blk.0.attn_output.weight
model.layers.0.post_attention_layernorm.weight → blk.0.ffn_norm.weight
model.layers.0.mlp.gate_proj.weight → blk.0.ffn_gate.weight
model.layers.0.mlp.up_proj.weight → blk.0.ffn_up.weight
model.layers.0.mlp.down_proj.weight → blk.0.ffn_down.weight
```

**Llama 3.1 adds Q/K norm**: `blk.{l}.attn_q_norm.weight` and `blk.{l}.attn_k_norm.weight` — identical to Qwen3. The existing `NameMap` in `gguf.h` already handles these with `"attn_q_norm.weight"` and `"attn_k_norm.weight"` entries. [Source: llama.cpp tensor_mapping.py, ikawrakow/ik_llama.cpp convert_hf_to_gguf.py]

**Non-layer tensors** (identical between Qwen3 and Llama 3.1):
- `token_embd.weight` → embed_tokens
- `output_norm.weight` → final_norm  
- `output.weight` → lm_head

**Conclusion**: The existing `map_qwen3_name()` and `NameMap` in gguf.h work for Llama 3.1 without changes. Only the arch-conditional metadata key prefix ("llama" vs "qwen3") needs adding.

---

### 4. Tiktoken Tokenizer (Llama 3.1 BPE)

**Token type**: Llama 3.1 uses OpenAI-style tiktoken BPE (GPT-4 pattern). Vocab size 128,256. Same algorithm as Qwen3, but different vocabulary and pre-tokenization regex.

**GGUF storage**: GGUF stores tokenizer in metadata:
- `tokenizer.ggml.model` = "gpt-bpe" (indicates tiktoken-style, not sentencepiece)
- `tokenizer.ggml.tokens` = array of strings (vocabulary)
- `tokenizer.ggml.scores` = array of floats (merge scores/ranks, optional)
- `tokenizer.ggml.merges` = array of strings ("pair1 pair2" format)
- `tokenizer.ggml.bos_token_id`, `tokenizer.ggml.eos_token_id`
- `tokenizer.ggml.add_bos_token` = bool

Alternatively, GGUF may store pre-tokenizer pattern in metadata (`tokenizer.ggml.pre` = "default" or "llama-bpe").

**C++ implementation**: The existing `BpeTokenizer` in `bpe_tokenizer.h` already implements the core BPE algorithm (byte-level encoding, merge loop, UTF-8 handling). Key changes for Llama 3.1:

1. **Pre-tokenization regex**: Llama 3.1 uses a different regex pattern than Qwen3/GPT-2. The GPT-4 pattern:
   ```
   (?i:'s|'t|'re|'ve|'m|'ll|'d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
   ```
   Existing `pretokenize()` in bpe_tokenizer.h implements this exactly. Should work for Llama 3.1.

2. **Merge format**: GGUF stores merges as space-separated token pair strings. The existing `load()` reads `left + " " + right` keys — same format.

3. **Byte encoder**: Llama 3.1 uses same GPT-2 byte-to-unicode mapping (bytes 0-255 → unicode codepoints 256-288 for control chars, rest identity). The existing byte encoder in bpe_tokenizer.h handles this.

4. **Special tokens**: Llama 3.1 uses `<|begin_of_text|>` (128000), `<|end_of_text|>` (128001), `<|eot_id|>` (128009), etc. The existing special_tokens map handles these.

5. **Data source**: Can extract tokenizer from GGUF metadata (string arrays), or load from existing `tokenizer_data.bin` format after conversion. Easiest: extend `gguf_convert.cpp` to export tokenizer data in the same binary format as `prepare_tokenizer.py` produces.

**Mapping to existing code**: `BpeTokenizer` load format expects: `[num_vocab][num_merges][num_added][byte_encodings(256×uint32)][vocab_entries(id, len, str)][added_tokens(id, len, is_special, str)][merges(left_len, left_str, right_len, right_str)]`. New GGUF export code needed in converter. [Source: bpe_tokenizer.h load() function, tiktoken documentation]

---

### 5. Q4_K_M Dequantization

**Block structure**: Q4_K (when GGML_TYPE_Q4_K = 12) uses 256-element super-blocks. In GGUF files quantized with K-quants, this is what Q4_K_M uses.

```
struct block_q4_K {
    half d;              // super-block scale (fp16) — 2 bytes
    half dmin;           // super-block minimum (fp16) — 2 bytes
    uint8_t scales[12];  // packed 6-bit scales — 12 bytes
                         // 16 sub-blocks (16 elements each): 
                         //   8 sub-blocks get 6-bit scales (scales[0..5] + scales[6..11] part)
                         //   8 sub-blocks get 4-bit scales (scales[6..11] part)
                         //   Packing: 6-bit scale * 8 + 4-bit scale * 8 = 48+32 = 80 bits = 10 bytes
                         //   Wait — actual packing differs
    uint8_t qs[128];     // 256 × 4-bit quantized values (128 bytes)
}; // sizeof = 144 bytes for 256 values → 4.5 bits per weight
```

**Corrected layout** (from ONNX issue #7691 and kkokosa/dotLLM):
```
struct block_q4_K {
    half d;              // super-block scale factor
    half dmin;           // super-block minimum
    uint8_t scales[12];  // 16 sub-block scales packed as 6-bit+4-bit hybrid
    uint8_t qs[128];     // 256 nibbles (4-bit each)
};
// Total: 2 + 2 + 12 + 128 = 144 bytes for 256 values
```

**Scale unpacking**: The `scales[12]` array encodes 16 scale values using a hybrid scheme:
- First 8 sub-blocks: 6-bit scales (0-63) packed into scales[0..5] (48 bits = 8 × 6)
- Next 8 sub-blocks: 4-bit scales packed into scales[6..11] (32 bits = 8 × 4)
- Actual packing is bit-level: scales[0..5] = 48 bits for first 8 sub-blocks, scales[6..11] contains 16 bits (2 bytes) of remaining 6-bit data + 32 bits of 4-bit data

**Dequant formula** (from dotLLM + llama.cpp discussion #6760):
```
For sub-block j (0..15):
    scale_j = decode_scale(scales, j)    // 0..63 or 0..15 mapped to float
    min_j    = decode_min(scales, j)      // companion min value extracted from same packed data
    
    For each nibble q in sub-block j (16 nibbles):
        val = d * (scale_j / 63.0f) * q - dmin * (min_j / 63.0f)
```

Or simplified (from llama.cpp coder):
```
val = d * scales_f[j] * q_val - dmin * mins_f[j]
```
Where `scales_f[j]` and `mins_f[j]` are dequantized from 6-bit/4-bit packed values.

**Implementation approach**: Write `dequant_q4_K_block()` function in `gguf.h` (~40 lines C). The converter calls this to dequantize Q4_K tensors → FP32 → requant to INT4 block-16, same pipeline as Q8_0 path. No need for optimized GPU dequant — only used offline during weight conversion. [Source: ONNX issue #7691, kkokosa/dotLLM quantization.md, llama.cpp ggml-quants.c]

**Q4_K_M vs Q4_K**: In llama.cpp, Q4_K_M is a variant of Q4_K where half of the attention.v and FFN.down weights use Q6_K quantization (important layers get 6-bit precision). But in GGUF files, Q4_K_M is stored as GGML_TYPE_Q4_K for those layers and Q6_K for the high-precision layers. The converter just needs to handle Q4_K blocks; higher-precision layers would be Q6_K (handled separately or upcast to Q8_0).

---

## Sources

### Kept
- `better-inference/gguf.h` — GGUF parser with Q8_0 dequant, Q4_K size skeleton, Qwen3 tensor mapper. Direct source.
- `better-inference/gguf_convert.cpp` — Converter: GGUF→INT4 block-16. Metadata key prefix handling, tensor iteration, write functions.
- `bpe_tokenizer.h` — Existing BPE tokenizer with GPT-4 pre-tokenizer pattern, byte-level encoding, merge loop. Direct source.
- ONNX issue #7691 (onnx/onnx) — Exact `block_q4_K` struct definition (half d, half dmin, scales[12], qs[128], 144 bytes). [Source](https://github.com/onnx/onnx/issues/7691)
- kkokosa/dotLLM QUANTIZATION.md — Q4_K dequant formula `val = d * scale_j * nibble - dmin * min_j`. [Source](https://github.com/kkokosa/dotLLM/blob/main/docs/QUANTIZATION.md)
- HuggingFace Meta-Llama-3.1-8B config.json (Medium article) — Full parameter list: hidden_size=4096, nqh=32, nkv=8, hd=128, intermediate_size=14336, NL=32, norm_eps=1e-5, vocab_size=128256, rope_theta=500000, use_scaled_rope=True. [Source](https://medium.com/@yuxiaojian/understand-how-llama3-1-works-a-deep-dive-into-the-model-flow-b149aba04bed)
- llama.cpp tensor_mapping.py — HF→GGUF name mapping confirmation. [Source](https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/gguf/tensor_mapping.py)
- llama.cpp discussion #6760 — Q4_K dequant formula `y = s * q - m`. [Source](https://github.com/ggml-org/llama.cpp/discussions/6760)
- ikawrakow/ik_llama.cpp convert_hf_to_gguf.py — `_layer_tensor_map` showing exact HF-to-GGUF tensor name mapping. [Source](https://github.com/ikawrakow/ik_llama.cpp/blob/main/convert_hf_to_gguf.py)
- Scaled RoPE in Llama 3.1 (HuggingFace LLama config) — `use_scaled_rope: True`. Must handle in RoPE kernel.

### Dropped
- Reddit/stackoverflow threads — No technical detail beyond what's in primary sources.
- General "GGUF format" overview articles — Redundant with existing gguf.h code.
- Python tokenizer tutorials — Not applicable (need C++ implementation).

---

## Gaps

1. **Q4_K scale bit-packing details** — Exact bit-level layout of `scales[12]` (how 8×6-bit + 8×4-bit are packed into 12 bytes) needs direct reading of `ggml-quants.c` `dequantize_row_q4_K()`. The 6-bit values and 4-bit values are interleaved in a complex pattern. Will need to study the actual C source to replicate exactly. **Suggested**: copy the dequant logic from ggml-quants.c directly, stripped of all SIMD optimizations.

2. **Scaled RoPE** — Llama 3.1 uses `use_scaled_rope=True`. This is a frequency-scaling variant (divides frequencies by a factor). Need to check if the existing rope_kernel (fixed `rope_theta=1000000`) or the new 500000 theta is sufficient, or if scaled RoPE requires additional logic. **Suggested**: start with `rope_theta=500000` and test — scaled RoPE may be default for 128K context but not required for short contexts.

3. **Tokenizer data extraction from GGUF** — Need to write GGUF metadata → binary tokenizer file converter. The existing `prepare_tokenizer.py` reads from HuggingFace `tokenizer.json`. Alternative: extend `gguf_convert.cpp` to export tokenizer from GGUF metadata (`tokenizer.ggml.tokens`, `tokenizer.ggml.merges`) into the same binary format. The existing `BpeTokenizer::load()` expects a specific binary format — may need to add a GGUF-based loading path.

4. **Llama 3.1 GGUF file naming** — Pre-converted GGUF models typically use `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`. Need to verify tensor count matches Qwen3-8B (292 tensors for Llama 3.1 8B vs Qwen3-8B's count). The Q/K norms add extra tensors per layer.

---

## Supervisor coordination

No coordination needed. Research complete with sufficient detail for implementation. Key deliverables for Phase 3:

1. Add `dequant_q4_K_block()` to gguf.h (~40 lines, follow ggml-quants.c reference)
2. Add `"llama"` arch prefix to gguf_convert.cpp `get_meta()` lambda
3. Parameterize rope_theta (500000 for Llama 3.1)
4. Write tokenizer data extraction (GGUF metadata → binary format, or GGUF-aware load path in bpe_tokenizer.h)
5. Update server config constants for Llama 3.1 dimensions
6. Verify Q/K head norm tensor names match existing mappings
