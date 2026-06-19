# Code Context

## Files Retrieved
1. `bench/text_generate_gemma4_12b_qat.cu` (lines 1-475, full file) — Gemma 4 12B QAT decode bench. FA/SWA per-layer dispatch, KV cache layout, attention math.
2. `better-inference/gguf_convert.cpp` (lines 1-600, full file) — GGUF→INT4 converter. FA layer handling, per-layer head_dim, norm extraction.
3. `src/kernels/apply_geglu.cu` (lines 1-69, full file) — GeGLU kernel + logit softcap kernel.
4. `src/kernels/decode.cu:398-492` (`attn_batched_kernel`) — batched GQA attention kernel used by bench.
5. `src/kernels/decode.cu:490-575` (`attn_batched_kernel_pos`) — seq_pos-arg variant (same math).
6. `src/kernels/norm.cu:195-228` (`head_norm_kernel`) — per-head RMSNorm for Q/K.
7. `include/blackwell/kernels.h:140-280` — public API for `fused_rope_decode`, `head_norm`, `attention_decode_batched_gqa*`.

---

## 🔴 CRITICAL BUG #1 — attn_batched_kernel truncates dot product at 128 floats (hd=512 broken)

**File**: `src/kernels/decode.cu:445-460` (and identical block in `attn_batched_kernel_pos:541-556`)

```cuda
float Q_reg[4];
{
    const float4* Q4 = reinterpret_cast<const float4*>(smem_Q);
    float4 q4 = Q4[lane_id];                       // ← reads only 4 floats per lane
    Q_reg[0] = q4.x; Q_reg[1] = q4.y; Q_reg[2] = q4.z; Q_reg[3] = q4.w;
}
for (int t = warp_id; t < npos; t += 4) {
    const float4* K4 = reinterpret_cast<const float4*>(K_cache + t * head_dim);
    float4 kv = K4[lane_id];                        // ← only 4 floats of K per lane
    float dot = Q_reg[0]*kv.x + ...                 // 32 lanes × 4 = 128 floats dot
    ...
}
```

- `Q4[lane_id]` → 32 lanes × 4 floats = **128 of head_dim=512** elements in Q·K dot.
- For SWA (`hd_swa=256`): covers 128/256 = half → also broken (50%, not 25%).
- Hardcoded `Q_reg[4]` and single `float4` load assume head_dim ≤ 128.
- **Impact**: FA attention scores computed from 128/512 dims (75% missing). Garbled output guaranteed. This is the root cause of "quality garbled" reported in AGENTS.md for Gemma 4 12B QAT.
- **Fix direction**: loop `Q_reg` over head_dim/128 chunks; or use `attention_decode_kernel_v4` (already exists, used by `attention_decode_gqa_device` for `head_dim > 128`, see `decode.cu:712-718`). Bench calls `attention_decode_batched_gqa` directly (NOT the `_device` variant) which always dispatches `attn_batched_kernel` regardless of head_dim.

**Note**: V weighted-sum loop (`for (int d = tid; d < head_dim; d += blockDim.x)`) correctly strides full head_dim — only Q·K dot is broken.

---

## 🟠 BUG #2 — QK head norm replication 256→512 is wrong assumption (FA layers)

**File**: `bench/text_generate_gemma4_12b_qat.cu:213-229`

```cuda
int nq = (int)fread(w256,4,hd_swa,f);            // reads 256 floats
for (int i = nq; i < l_hd; i++) w256[i] = w256[i % hd_swa];  // FA: replicate 256→512
```

- Bench assumes GGUF QK norm tensors for FA layers are 256-dim, then replicates to fill hd_fa=512.
- **Converter** (`gguf_convert.cpp:456-466`) writes per-layer norm with `copy_hd = min(l_hd_norm, ti.nelements())` and stores `l_hd[l]` entries (512 for FA). If GGUF truly has 512-dim norms for FA, converter writes 512 floats; bench's replication would never trigger (`nq == l_hd`).
- **If GGUF stores 256-dim norms for FA layers**: bench replication assumes `weight[i] = weight[i%256]` (alternating 0,1,2,...). No evidence Gemma 4 FA layers use this pattern — likely the GGUF has either 512-dim or a different layout. **Verify GGUF tensor sizes for FA layers' `attn_q_norm`/`attn_k_norm` before trusting this code.**
- `head_norm_kernel` (`norm.cu:195-222`) correctly applies weight per-head at full head_dim — no bug there.

---

## 🟠 BUG #3 — Logit softcap DISABLED with incorrect reasoning

**File**: `bench/text_generate_gemma4_12b_qat.cu:432-437`

```cuda
// NOTE: final_logit_softcapping (30.0) exists but saturates all logits
// to ±30 with tanh, producing uniform distribution → EOS token.
// INT4 quantization noise pushes raw logits beyond softcap range.
// Skip softcap until quality improves.
// blackwell::kernels::apply_logit_softcap(d_logits, V, FINAL_LOGIT_SOFTCAP, st);
```

- Reasoning is **wrong**: `tanh(x/30)*30` is monotonic in x → preserves argmax ordering. Softcap cannot change "uniform distribution" outcome; it only bounds magnitude.
- `apply_logit_softcap` (`src/kernels/apply_geglu.cu:57-67`) is correct: `data[i] = tanhf(x/cap)*cap`. Symbol present in `libblackwell_kernels.a`.
- Skipping softcap does NOT cause garbled output (argmax identical), but it IS a divergence from reference. **Low-priority fix; enable after BUG #1 resolved.**

---

## 🟡 BUG #4 — GeGLU uses erf-based GELU; HF Gemma uses tanh-approx

**File**: `src/kernels/apply_geglu.cu:18-20`

```cuda
__device__ __forceinline__ float gelu(float x) {
    return x * 0.5f * (1.0f + erff(x * 0.7071067811865475f));  // erf-based
}
```

- HF Gemma 2/3/4 uses `gelu_pytorch_tanh` (tanh approximation): `0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x³)))`.
- Magnitude difference near x=0: ~1e-3. Minor quality impact, not root cause of garbling.
- Same erf-GELU used in `bench/text_generate_gemma.cu:46` (Gemma 3 12B bench).
- **Low priority** — but should switch to tanh-approx for fidelity.

---

## ✅ FA layer handling — correct aspects

### Bench FA dispatch (`text_generate_gemma4_12b_qat.cu:147-185`)
- FA layers correctly identified: `SWA_LAYERS[48]` flag, FA at 5,11,17,23,29,35,41,47.
- Per-layer dims set correctly:
  - FA: `l_hd=512, l_nqh=16, l_nkv=1, l_q_dim=8192, l_k_dim=512, l_v_dim=512, l_o_dim=8192`
  - SWA: `l_hd=256, l_nqh=16, l_nkv=8, l_q_dim=4096, l_k_dim=2048, l_v_dim=2048, l_o_dim=4096`

### K=V sharing (`text_generate_gemma4_12b_qat.cu:195-201`)
```cuda
if (W[l].fa) { W[l].v = W[l].k; }  // K and V share same weight matrix (k_eq_v)
```
Correct — FA layers have no v_proj in GGUF.

### Layer 47 q_proj sharing (`text_generate_gemma4_12b_qat.cu:187-192`)
```cuda
if (l == 47) { W[l].q = W[46].q; }  // GGUF truncates layer 47 q_proj
```
Matches AGENTS.md note. **Quality risk**: layer 47 attention will be wrong (uses layer 46's Q projection), but only one layer affected.

### RoPE per-layer theta and head_dim (`text_generate_gemma4_12b_qat.cu:163-184`)
- Separate RoPE caches: `d_cos_swa`/`d_sin_swa` (hd=256, theta=10000), `d_cos_fa`/`d_sin_fa` (hd=512, theta=1e6).
- Per-layer assignment: `W[l].l_cos_cache = W[l].fa ? d_cos_fa : d_cos_swa`.
- `build_rope_cache` (`text_generate_gemma4_12b_qat.cu:79-88`) computes `theta = pos * powf(base, -2*d/head_dim)` — formula correct.

### KV cache variable-stride layout (`text_generate_gemma4_12b_qat.cu:113-122`)
```cuda
const size_t KV_SLOT_SWA = nkv_swa * MAXSEQ * hd_swa;  // 8*2048*256
const size_t KV_SLOT_FA  = nkv_fa  * MAXSEQ * hd_fa;   // 1*2048*512
for (int l = 0; l < NL; l++) {
    kv_offsets[l] = total_kv;
    total_kv += SWA_LAYERS[l] ? KV_SLOT_SWA : KV_SLOT_FA;
}
```
Correct per-layer offsets. Passed to attention kernel as `kv_layer_elems=kv_off`, `kv_batch_elems=total_kv`.

### FA KV cache write (`text_generate_gemma4_12b_qat.cu:242-251`)
```cuda
if (W[l].fa) {
    for (int h = 0; h < n_heads_k; h++) {  // n_heads_k = 1
        cudaMemcpyAsync(d_kc+kv_off + h*MAXSEQ*l_hd + step*l_hd, d_K + h*l_hd, l_hd*4, ...);
        cudaMemcpyAsync(d_vc+kv_off + h*MAXSEQ*l_hd + step*l_hd, d_V + h*l_hd, l_hd*4, ...);
    }
}
```
Correct — single FA head, D2D copy of K and V (V comes from same weight as K).

### Attention kernel call (`text_generate_gemma4_12b_qat.cu:255-257`)
```cuda
attention_decode_batched_gqa(d_attn,d_Q,d_kc,d_vc,attn_step,l_nqh,l_nkv,l_hd,MAXSEQ,1,
    total_kv,kv_off,st);
```
M=1, kv_batch_elems=total_kv, kv_layer_elems=kv_off. Args correct. **But kernel itself is BUG #1.**

### GQA head mapping (`decode.cu:421`)
```cuda
int kv_head = head * num_kv_heads / num_q_heads;
```
For FA: 16 Q heads, 1 KV head → all Q heads map to kv_head=0. Correct.

### SWA window cap (`text_generate_gemma4_12b_qat.cu:238-239`)
```cuda
if (W[l].swa && attn_step > SWA_WINDOW) attn_step = SWA_WINDOW;
```
SWA_WINDOW=1024. Caps attention length. **Subtle**: this caps `seq_pos` passed to kernel, but KV cache still written at full `step` offset. Kernel reads positions `[0..attn_step]` from cache base — for SWA layers with `attn_step=1024`, reads positions `step-1024..step` would require offset adjustment. **Likely bug**: kernel always reads from cache[0..attn_step], not cache[step-attn_step..step]. SWA windowing reads wrong KV positions when `step > 1024`. Verify if prompts exceed 1024 tokens.

---

## ✅ GGUF converter — FA handling

### Per-layer head_dim (`gguf_convert.cpp:215-235`)
```cpp
for (int i = 0; i < NL; i++) {
    if (l_nkv[i] == 1) l_hd[i] = hd;           // FA: global hd=512
    else l_hd[i] = hd_swa_meta;                 // SWA: 256
}
```
Correct. Reads `attention.key_length` (global=512) and `attention.key_length_swa` (256) from GGUF metadata.

### v_proj absence (`gguf_convert.cpp:467-483`)
```cpp
if (is_gemma4) {
    if (strstr(ti.name.c_str(), "attn_v")) {
        printf("  SKIP: %s (unmapped — k_eq_v layer?)\n", ti.name.c_str());
    }
}
```
FA layer `attn_v.weight` skipped silently if unmapped. **Gap**: converter does NOT verify which layers lack v_proj — any unmapped `attn_v*` tensor is skipped regardless of whether it's actually an FA layer. Could mask real conversion errors.

### FA dim verification (`gguf_convert.cpp:495-521`)
```cpp
if (is_gemma4 && ll >= 0 && l_nkv[ll] == 1) {
    if (strstr(bw_name, "q_proj")) { /* verify N == nqh*exp_hd */ }
    if (strstr(bw_name, "k_proj")) { /* verify N == 1*exp_hd */ }
    if (strstr(bw_name, "o_proj")) { /* verify K == nqh*exp_hd */ }
}
```
Prints NOTE on mismatch but does NOT fail. Good for diagnostics.

### QK norm per-layer head_dim (`gguf_convert.cpp:440-466`)
```cpp
int l_hd_norm = l_hd[l];
if (tensor_nelems > 0 && tensor_nelems < hd) l_hd_norm = tensor_nelems;
int copy_hd = min(l_hd_norm, ti.nelements());
memcpy(&q_norms[qk_off], ..., copy_hd * 4);
```
Reads actual tensor element count. For FA layers with 512-dim norms, writes 512 floats. For 256-dim, writes 256. **Bench replication (BUG #2) only triggers if GGUF has 256-dim norms for FA layers** — converter writes whatever GGUF has.

### QK norm buffer offsets (`gguf_convert.cpp:429-436`)
```cpp
qk_offsets[i] = (i == 0) ? 0 : qk_offsets[i-1] + l_hd[i-1];
```
Variable-stride buffer matching per-layer head_dim. Correct.

### RoPE theta — single global value only (`gguf_convert.cpp:318-365`)
```cpp
float rope_theta = get_meta_f("rope.freq_base", 0.0f);
...
write_file(path, rope_cfg, 8);  // [theta, hd]
```
**Gap**: writes single `rope_theta` to `rope_config.f32`. Gemma 4 needs TWO thetas (1e4 SWA, 1e6 FA). Bench hardcodes both (`rope_theta_swa=10000`, `rope_theta_fa=1000000`) and ignores `rope_config.f32`. Converter's rope export is dead code for Gemma 4.

---

## Softcap implementation — correct

**File**: `src/kernels/apply_geglu.cu:54-67`

```cuda
__global__ void softcap_kernel(float* data, int N, float cap) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    data[idx] = tanhf(data[idx] / cap) * cap;
}
```

- Formula `tanh(x/cap)*cap` matches HF `GemmaForCausalLM.final_logit_softcapping`.
- Monotonic → preserves argmax. Bench comment claiming "uniform distribution" is incorrect.
- Symbol verified in `libblackwell_kernels.a`: `blackwell::kernels::apply_logit_softcap` present.
- Only issue: bench disabled it (BUG #3).

---

## Architecture

```
GGUF (Q4_0 QAT) 
  → gguf_convert.cpp: dequant Q4_0→FP32, transpose [K,N]→[N,K], requant INT4 block-16
  → writes {l}_self_attn.{q,k,v,o}_proj.int4_t + .scale_t (FP16 scales)
  → writes {l}_attn_{q,k}_norm.f32 (per-layer head_dim)
  → writes {l}_{input,post_attn,post_attention,post_ffn}_*.f32 (4 RMSNorms)
  → weights_gemma4_12b_qat/

bench/text_generate_gemma4_12b_qat.cu:
  load weights → per-layer LW4 struct (FA vs SWA dispatch)
  prefill loop: dequant embed → forward_token(step) per prompt token
  decode loop: forward_token → sample_argmax → next token

forward_token (per layer):
  RMSNorm → quant INT4 → q/k/v GEMV → head_norm(Q,K) → RoPE(Q,K) 
  → KV cache write → attention_decode_batched_gqa → quant → o_proj GEMV
  → post_attn RMSNorm → residual add
  → RMSNorm → quant → gate/up GEMV → GeGLU → quant → down GEMV
  → post_ffn RMSNorm → residual add

Final: RMSNorm → quant → lm_head GEMV → [softcap DISABLED] → argmax
```

---

## Start Here

**`src/kernels/decode.cu:445-460`** — **BUG #1** is the root cause of garbled output. `attn_batched_kernel` hardcodes 4-float Q_reg and single float4 load, only computing 128 of head_dim=512 dot product. Fix here first. Alternative: route FA layers to `attention_decode_kernel_v4` (already handles `head_dim > 128`, see `decode.cu:712-718`) by having bench call `attention_decode_gqa_device` instead of `attention_decode_batched_gqa`, OR add a head_dim>128 dispatch path inside `attention_decode_batched_gqa`.

---

## Open Questions

1. **GGUF QK norm tensor size for FA layers**: Are `attn_q_norm`/`attn_k_norm` 256-dim or 512-dim for FA layers in the source GGUF? Determines if bench's replication (BUG #2) actually fires. Check with: `python3 -c "from gguf import GGUFReader; r=GGUFReader('model.gguf'); [print(t.name, t.shape) for t in r.tensors if 'norm' in t.name and 'blk.5.' in t.name]"` or inspect `gguf_convert` output logs.

2. **SWA windowing correctness**: When `step > 1024`, bench passes `attn_step=1024` to attention kernel, but KV cache is written at offset `step`. Kernel reads `cache[0..attn_step]` from base — these are positions 0..1023, NOT step-1024..step-1. SWA window reads stale early positions. Needs cache-base offset shift for SWA layers.

3. **GELU variant**: Confirm HF Gemma 4 config uses `gelu_pytorch_tanh` (not `gelu`). If yes, `apply_geglu.cu` should switch to tanh approximation.

4. **Layer 47 q_proj**: Sharing layer 46's Q projection guarantees layer 47 attention is incorrect. Check if HF/GGUF has the tensor (just truncated in this particular GGUF file) or if model truly omits it.
