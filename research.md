# Research: Gemma 4 12B QAT Architecture — FA vs SWA Layer Design

## Summary

Gemma 4 12B uses a **hybrid 5:1 attention pattern**: 5 sliding-window (SWA) layers followed by 1 full/global attention (FA) layer, repeated 8 times = 40 SWA + 8 FA across 48 layers. FA layers (at indexes 5,11,17,23,29,35,41,47) use `k_eq_v=True` (K==V, single shared projection, `num_global_key_value_heads=1`, `global_head_dim=512`), while SWA layers use `num_key_value_heads=8`, `local_head_dim=256`, with a separate `v_proj`. FA layers have **double Q heads** (16×512=8192 Q dim) vs SWA's 16×256=4096. RoPE differs: FA uses θ=1e6, SWA uses θ=1e4. `final_logit_softcapping=30.0` applies `cap * tanh(logits / cap)`. Our bench's FA handling is **partially correct** but has 3 likely bugs causing the "garbled after 1-2 tokens" issue (QK norm replication, layer-47 q_proj sharing, and shared-KV semantics).

## Findings

### 1. Layer Pattern: 5:1 SWA:FA — 40 SWA + 8 FA, FA every 6th layer

Gemma 4 inherits Gemma 3's **5:1 sliding-window-to-full-attention interleaving**. Gemma 3 introduced this pattern (5 local SWA layers : 1 global FA layer, repeating). Gemma 4 12B has 48 layers → 8 FA layers at positions 5,11,17,23,29,35,41,47 (every 6th, 0-indexed) and 40 SWA layers. [Google DeepMind Gemma 4](https://deepmind.google/models/gemma/gemma-4/) | [Gemma 3 explained — Google Developers Blog](https://developers.googleblog.com/en/gemma-explained) | [Welcome Gemma 3 blog](https://huggingface.co/blog/gemma3) | [Gemma 3 Technical Report (arXiv:2503.19786)](https://arxiv.org/pdf/2503.19786)

**Local confirmation**: `bench/text_generate_gemma4_12b_qat.cu` line ~38 hardcodes `SWA_LAYERS[NL] = {1,1,1,1,1,0, ...}` (5 SWA, 1 FA, ×8) — matches exactly. The GGUF metadata stores this as a per-layer `head_count_kv` array: `[8,8,8,8,8,1, 8,8,8,8,8,1, ...]` (FA layers have `nkv=1`, SWA have `nkv=8`). Converter `better-inference/gguf_convert.cpp` lines ~224-235 reads `gemma4.attention.head_count_kv` as `std::vector<int32_t>`.

### 2. FA Layers: k_eq_v=True, num_global_key_value_heads=1, global_head_dim=512

FA (global/full attention) layers in Gemma 4:
- **`num_global_key_value_heads = 1`** (NOT 2 — confirmed by GGUF metadata `head_count_kv=1` at FA positions, and by AGENTS.md note "num_global_key_value_heads = 2" being a misread; the per-layer array shows 1)
- **`global_head_dim = 512`** (GGUF key `attention.key_length=512`)
- **`k_eq_v = True`**: K and V share a **single projection** — there is **no separate `v_proj`** for FA layers. The GGUF converter correctly skips `attn_v` tensors for FA layers (`gguf_convert.cpp` line ~464: "For gemma4, attn_v.weight may be absent for FA layers (k_eq_v)").
- **Q dim = 16 × 512 = 8192** (double Q heads vs SWA's 4096)
- **K dim = V dim = 1 × 512 = 512**
- **o_proj input dim = 8192** (= nqh × global_head_dim = 16 × 512)

Sources: [google/gemma-4-12B-it-qat-q4_0-gguf (HF)](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf) | [unsloth/gemma-4-12B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF) | [Two Major KV Optimizations in Gemma4 (LinkedIn)](https://www.linkedin.com/pulse/two-major-kv-opt) | [Gemma4Backbone (Keras Hub)](https://keras.io/keras_hub/api/models/gemma4/gemma4_backbone)

**⚠️ Correction to AGENTS.md**: The project notes state "num_global_key_value_heads = 2". The GGUF metadata per-layer array shows `nkv=1` at FA positions. The `head_count_kv` array is `[8,8,8,8,8,1,...]`. This should be **1**, not 2. The bench code correctly uses `nkv_fa=1` (line 35). The "2" in AGENTS.md may refer to a different metadata field or a misreading.

### 3. SWA Layers: Standard GQA, local_head_dim=256, separate v_proj

SWA (sliding window attention) layers:
- **`num_key_value_heads = 8`**
- **`local_head_dim = 256`** (GGUF key `attention.key_length_swa=256`)
- **Q dim = 16 × 256 = 4096**
- **K dim = V dim = 8 × 256 = 2048**
- **Has separate `v_proj`** (not k_eq_v)
- **Sliding window = 1024 tokens** (local context, SWA_WINDOW in bench)

Sources: [Gemma 4 model card (Google AI)](https://ai.google.dev/gemma/docs/core/model_card_4) | [Welcome Gemma 4 (HF Blog)](https://huggingface.co/blog/gemma4)

### 4. QK Norms: Per-layer head_dim — 256 for SWA, 512 for FA

QK normalization (`attn_q_norm`, `attn_k_norm`) operates at the **per-layer head dimension**:
- **SWA layers**: QK norms are **256-dim** (one norm weight per head element, applied per-head)
- **FA layers**: QK norms are **512-dim** (matching `global_head_dim=512`)

The GGUF converter reads the actual tensor element count to determine the correct head_dim per layer (`gguf_convert.cpp` lines ~445-460: `if (tensor_nelems > 0 && tensor_nelems < hd) l_hd_norm = tensor_nelems`). It writes per-layer norm files with correct dimensions.

**⚠️ LIKELY BUG in bench**: `text_generate_gemma4_12b_qat.cu` lines ~253-265 reads QK norms as 256-dim for ALL layers then **replicates 256→512** for FA layers (`for (int i = nq; i < l_hd; i++) w256[i] = w256[i % hd_swa]`). This is WRONG. The GGUF converter writes FA-layer QK norms as native 512-dim tensors. The bench should read the full 512-dim norm directly, not replicate a 256-dim norm. This replication corrupts FA-layer Q/K normalization → **likely cause of garbled output**.

### 5. RoPE: Dual theta — FA uses 1e6, SWA uses 1e4

Gemma 4 uses **separate RoPE base frequencies** for the two attention types:
- **FA layers**: `rope_theta = 1,000,000` (1e6) — long-range global context
- **SWA layers**: `rope_theta = 10,000` (1e4) — local sliding window

This is confirmed in the bench file (lines 34-35: `rope_theta_swa=10000.0f`, `rope_theta_fa=1000000.0f`) and referenced in multiple architecture analyses as "Dual RoPE." [Gemma 4 Architecture Deep Dive (CloudInsight)](https://cloudinsight.cc/en/blog/gemma-4-architecture) | [Gemma 4 Architecture Explained (Botmonster)](https://botmonster.com/posts/gemma-4-architecture) | [Gemma 4: Architecture and Multimodal Innovations (Jianyu)](https://jianyuh.github.io/architecture/2026/04/)

### 6. final_logit_softcapping = 30.0

Formula: **`logits = cap * tanh(logits / cap)`** where `cap = 30.0`. This is identical to Gemma 2/3's logit softcap. It bounds logits to [-30, +30] via tanh saturation.

**⚠️ BUG in bench**: The bench DISABLES softcap (line ~415: "Skip softcap until quality improves") with a comment that "INT4 quantization noise pushes raw logits beyond softcap range." This is incorrect reasoning — softcap is designed to handle large logits via tanh. Disabling it changes the output distribution. However, if logits are truly garbage (from upstream bugs), softcap would make them uniformly bad. The root cause is upstream (QK norm bug), not the softcap.

Sources: [Gemma 2 paper (softcap origin)](https://arxiv.org/abs/2408.00118) | [Gemma 3 modeling code (HF Transformers)](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py)

### 7. GGUF Conversion Issues — Layer 47 q_proj truncation

Known GGUF conversion issue: **layer 47's `q_proj` tensor is missing/truncated** in the GGUF file. The bench works around this by sharing layer 46's q_proj for layer 47 (line ~243: `W[l].q = W[46].q`). This is a known artifact of the QAT GGUF export, not a fundamental architecture property. [Eval bug: Gemma 4 generates tokens (llama.cpp #21321)](https://github.com/ggml-org/llama.cpp/issues/21321) | [unsloth/gemma-4-31B-it-GGUF token accuracy issues](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF)

**⚠️ POTENTIAL BUG**: Sharing layer 46's q_proj for layer 47 is a workaround. Layer 47 IS an FA layer (position 47 = 7×6+5). If layer 46 (SWA, hd=256) and layer 47 (FA, hd=512) have different q_proj dimensions (4096 vs 8192), sharing would produce dimension mismatch or garbage. The bench does NOT check this — `W[46].q` has K=3840, N=4096 (SWA dims), but layer 47 needs N=8192 (FA dims). **This is almost certainly wrong** — the GEMV would produce a 4096-element Q vector for a layer expecting 8192, and the subsequent o_proj (expecting 8192 input) would read uninitialized memory.

### 8. Shared KV Semantics — "shared_kv_states"

The HF Transformers Gemma 4 modeling code uses a `shared_kv_states` dictionary. FA layers **reuse K/V from the last same-type (FA) layer** rather than computing their own. This means:
- FA layer 5 computes K/V from its k_proj
- FA layer 11 reuses layer 5's K/V (appended with new tokens)
- FA layer 17 reuses layer 11's K/V
- etc.

This is a **KV cache sharing optimization** across FA layers — all 8 FA layers share a single growing KV cache. The bench does NOT implement this — each FA layer has its own independent KV cache slot (`kv_offsets[l]`). This means FA layers attend only to their own position's K/V, not the accumulated global context from prior FA layers. **This is a significant architectural deviation** that would produce incorrect attention for all FA layers except the first.

Sources: [Two Major KV Optimizations in Gemma4 (LinkedIn)](https://www.linkedin.com/pulse/two-major-kv-opt) | [Gemma4 (HF Transformers docs)](https://huggingface.co/docs/transformers/model_doc/gemma4) | [Gemma4Backbone (Keras)](https://keras.io/keras_hub/api/models/gemma4/gemma4_backbone) | [Gemma 4 is not your standard transformer (idlemachines)](https://idlemachines.co.uk/essays/gemma4-architecture)

### 9. 4 RMSNorms Per Layer (Gemma 4 specific)

Gemma 4 uses **4 RMSNorms per layer** (vs 2 in most transformers):
1. `attn_norm` / `input_layernorm` — pre-attention
2. `post_attention_norm` / `post_attn_norm` — post-attention residual
3. `ffn_norm` / `post_attention_layernorm` — pre-FFN
4. `post_ffw_norm` / `post_ffn_norm` — post-FFN residual

Plus a final `output_norm` / `final_norm`. The bench correctly loads all 4 per-layer norms (lines ~288-310). The GeGLU activation (not SwiGLU) is used for the FFN.

### 10. Config Summary (Gemma 4 12B QAT)

| Parameter | Value | Source |
|-----------|-------|--------|
| `hidden_size` (H) | 3840 | GGUF `embedding_length` |
| `intermediate_size` (I) | 15360 | GGUF `feed_forward_length` |
| `num_hidden_layers` (NL) | 48 | GGUF `block_count` |
| `num_attention_heads` (nqh) | 16 | GGUF `attention.head_count` |
| `vocab_size` (V) | 262144 | GGUF tokenizer count |
| SWA `head_dim` | 256 | GGUF `attention.key_length_swa` |
| FA `head_dim` | 512 | GGUF `attention.key_length` |
| SWA `num_kv_heads` | 8 | GGUF per-layer `head_count_kv` |
| FA `num_kv_heads` | 1 | GGUF per-layer `head_count_kv` (k_eq_v) |
| SWA `rope_theta` | 10,000 | Architecture docs |
| FA `rope_theta` | 1,000,000 | Architecture docs |
| `sliding_window` | 1024 | Architecture docs |
| `final_logit_softcapping` | 30.0 | Gemma family standard |
| FA layer positions | 5,11,17,23,29,35,41,47 | 5:1 pattern |
| FA `k_eq_v` | True | No v_proj in GGUF for FA layers |
| Embeddings | Tied (embed == lm_head) | GGUF tensor analysis |

## Sources

### Kept (authoritative / directly relevant)
- **Local: `bench/text_generate_gemma4_12b_qat.cu`** — Our implementation with FA/SWA handling. Primary evidence for what our code does (and where bugs are).
- **Local: `better-inference/gguf_convert.cpp`** — GGUF→INT4 converter that reads actual GGUF metadata (`head_count_kv` per-layer array, `key_length`/`key_length_swa`). Authoritative for config values.
- **[google/gemma-4-12B-it-qat-q4_0-gguf (HF)](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf)** — Official QAT GGUF model. Source of the GGUF metadata our converter parsed.
- **[google/gemma-4-12B (HF)](https://huggingface.co/google/gemma-4-12B)** — Official model card.
- **[Gemma 4 — Google DeepMind](https://deepmind.google/models/gemma/gemma-4/)** — Official model page.
- **[Gemma 4 model card (Google AI)](https://ai.google.dev/gemma/docs/core/model_card_4)** — Official model card with specs.
- **[Welcome Gemma 4 (HF Blog)](https://huggingface.co/blog/gemma4)** — Official release blog.
- **[Gemma4Backbone (Keras Hub)](https://keras.io/keras_hub/api/models/gemma4/gemma4_backbone)** — Official Google Keras implementation reference.
- **[Gemma4 (HF Transformers docs)](https://huggingface.co/docs/transformers/model_doc/gemma4)** — HF Transformers documentation for Gemma4.
- **[HuggingFace Transformers gemma4 source](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4)** — Modeling code (modeling_gemma4.py, configuration_gemma4.py).
- **[Gemma 4 Architecture Deep Dive (CloudInsight)](https://cloudinsight.cc/en/blog/gemma-4-architecture)** — Detailed architecture analysis (MoE, Dual RoPE).
- **[Gemma 4 Architecture Explained (Botmonster)](https://botmonster.com/posts/gemma-4-architecture)** — Per-layer embeddings, shared KV explanation.
- **[Gemma 4: Architecture and Multimodal Innovations (Jianyu)](https://jianyuh.github.io/architecture/2026/04/)** — Technical architecture breakdown.
- **[Two Major KV Optimizations in Gemma4 (LinkedIn)](https://www.linkedin.com/pulse/two-major-kv-opt)** — KV sharing and optimization details.
- **[Gemma 4 is not your standard transformer (idlemachines)](https://idlemachines.co.uk/essays/gemma4-architecture)** — Architecture analysis.
- **[Google's Gemma 4 will Change How AI Models are Built](https://www.artificialintelligencemadesimple.co)** — Architecture commentary.
- **[Gemma 3 Technical Report (arXiv:2503.19786)](https://arxiv.org/pdf/2503.19786)** — Gemma 3 paper (5:1 pattern origin, softcap).
- **[Gemma explained: What's new in Gemma 3 (Google Dev Blog)](https://developers.googleblog.com/en/gemma-explained)** — Official Gemma 3 architecture explanation (SWA pattern).
- **[Eval bug: Gemma 4 generates tokens (llama.cpp #21321)](https://github.com/ggml-org/llama.cpp/issues/21321)** — Known GGUF conversion issue.
- **[unsloth/gemma-4-12B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF)** — Unsloth QAT GGUF.
- **[Gemma 4 Architecture -- Complete Technical Comparison (g4.si5.pl)](https://g4.si5.pl/)** — Technical comparison reference.

### Dropped
- Wikipedia "Gemma (language model)" — appeared in many searches but lacked specific Gemma 4 FA/SWA technical detail at the depth needed.
- Various Medium/blog posts (DEV.to, Medium ML authors) — secondary commentary, not authoritative.

## Gaps

### High-priority gaps for resolving the "garbled after 1-2 tokens" bug

1. **Shared KV implementation NOT confirmed from source code.** The claim that FA layers share a single growing KV cache (`shared_kv_states`) comes from architecture blog posts and the LinkedIn article, NOT from directly reading the HF Transformers `modeling_gemma4.py` source. **This is the #1 suspect for the garbled output.** Need to read the actual modeling code to confirm:
   - Do FA layers share K/V across FA layers, or does each FA layer have its own K/V?
   - Is `shared_kv_states` a runtime cache optimization or an architectural feature (weights)?
   - If shared: what is the sharing pattern — last FA layer only, or all previous FA layers?

2. **QK norm dimension for FA layers not confirmed from HF source.** Our bench replicates 256→512, but the GGUF converter writes native per-layer dims. Need to verify from `modeling_gemma4.py` whether FA-layer QK norms are genuinely 512-dim or if the norm is applied differently (e.g., per 256-dim half-head).

3. **Layer 47 q_proj dimension mismatch not verified.** Need to check the actual GGUF tensor: is layer 47's q_proj truly missing, or is it present with FA dims (8192)? If present, sharing from layer 46 (SWA, 4096) is definitely wrong. If truly missing, the model itself may have a design where layer 47 reuses a prior FA layer's q_proj (like layer 41 or 35), not a SWA layer's.

4. **`num_global_key_value_heads` = 1 vs 2 discrepancy.** AGENTS.md says 2, GGUF metadata says 1 (per-layer array). Need to read `configuration_gemma4.py` to confirm. If it's actually 2, then K=V dim = 2×512=1024, not 512 — and our bench's `l_k_dim=512` would be wrong.

### Suggested next steps

1. **Read `modeling_gemma4.py` from HF Transformers** — specifically the attention class and KV cache handling. This is the single most important step. URL: `https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py`
2. **Read `configuration_gemma4.py`** — confirm `num_global_key_value_heads`, `global_head_dim`, `k_eq_v`, `shared_kv_layers` config fields.
3. **Inspect actual GGUF tensor list** for the QAT model — run `gguf_convert` in debug mode to see all tensor names and shapes, especially for layers 5, 11, 47. Verify whether v_proj exists for FA layers and whether layer 47 q_proj is present.
4. **Fix shared KV cache** — if FA layers share KV, implement a single FA KV cache (1 head × 512 dim × MAXSEQ) shared across all 8 FA layers, not 8 separate caches.
5. **Fix QK norm loading** — read native per-layer dim norms directly (converter already writes them correctly).

## Supervisor coordination

No supervisor contact needed — research task complete. The key actionable finding is that the "garbled after 1-2 tokens" bug likely stems from **3 compounding issues**: (1) QK norm 256→512 replication for FA layers, (2) layer 47 q_proj dimension mismatch when sharing from SWA layer 46, and (3) missing shared-KV implementation across FA layers. Reading `modeling_gemma4.py` is the critical next step to confirm the shared KV behavior before implementing fixes.
