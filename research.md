# Research: Blackwell Project Kernel APIs & Architecture

## Summary

Blackwell project implements custom INT8/INT4 LLM inference kernels for RTX 5060 Ti (SM_120a). Key kernel signatures found in `include/blackwell/kernels.h`. Batched architecture uses M×buffer layout with separate KV caches per sequence. Server processes multi-prompt requests sequentially (no GPU batching).

## Key Kernel Signatures

### INT4 GEMV Kernels
```cpp
// Warp-cooperative INT4 GEMV — 1 warp per row, scalar nibble dot product
cudaError_t gemv_int4_warp(
    float* y_out, const void* x_packed, const float* x_scale,
    const void* W_packed, const float* W_scale,
    int K, int N, cudaStream_t stream);

// Batched INT4 GEMV — M sequences, weight reuse across tokens
cudaError_t gemv_int4_batched(
    float* y_out, const uint8_t* x_packed, const float* x_scale,
    const uint8_t* W_packed, const float* W_scale,
    int K, int N, int M, cudaStream_t stream);

// Batched asymmetric INT4 (zero-point)
cudaError_t gemv_int4_asym_batched(
    float* y_out, const uint8_t* x_packed, const float* x_sc_zero,
    const uint8_t* W_packed, const float* W_sc_zero,
    int K, int N, int M, cudaStream_t stream);
```

### INT8 GEMV Kernels
```cpp
// Warp-cooperative INT8 — 1 warp/row, dp4a SIMD, shuffle reduce (PRODUCTION)
cudaError_t gemv_int8_warp(
    float* y_out, const void* x_int8, const float* x_scale,
    const void* W_t_int8, const float* W_t_scale,
    int K, int N, cudaStream_t stream);

// Batched INT8 GEMV — M=1-8
cudaError_t gemv_int8_batched(
    float* y_out, const void* x_int8, const float* x_scale,
    const void* W_t_int8, const float* W_t_scale,
    int K, int N, int M, cudaStream_t stream);

// Fused gate+up INT8 GEMV
cudaError_t gemv_int8_gate_up(
    float* gate_out, float* up_out,
    const int8_t* x_int8, const float* x_scale,
    const int8_t* W_gate, const float* W_gate_sc,
    const int8_t* W_up, const float* W_up_sc,
    int K, int N, cudaStream_t stream);
```

### Attention Kernels
```cpp
// GQA decode attention — single sequence
cudaError_t attention_decode_gqa(
    float* output, const float* Q, const float* K_cache, const float* V_cache,
    int seq_pos, int num_q_heads, int num_kv_heads, int head_dim,
    int max_seq_len, cudaStream_t stream);

// Batched GQA decode — M sequences, kv_batch_elems = stride between seqs
cudaError_t attention_decode_batched_gqa(
    float* output, const float* Q, const float* K_cache, const float* V_cache,
    int seq_pos, int num_q_heads, int num_kv_heads, int head_dim,
    int max_seq_len, int M,
    size_t kv_batch_elems,  // floats between sequences
    size_t kv_layer_elems,  // floats from seq base to current layer
    cudaStream_t stream);
```

### Quantization Kernels
```cpp
// FP32 → INT4 pack with per-block scales (block=16)
cudaError_t quantize_int4(
    void* x_out_packed, float* x_out_sc,
    const float* in_fp32, int K, cudaStream_t stream);

// Batched INT4 quantization
cudaError_t quantize_int4_batched(
    void* x_out_packed, float* x_out_sc,
    const float* in_fp32, int K, int M, cudaStream_t stream);

// FP32 → INT8 pack with per-block scales
cudaError_t quantize_int8(
    void* out_int8, float* out_scale,
    const float* in_fp32, int num_elements, cudaStream_t stream);
```

### Norm/Activation Kernels
```cpp
// Fused RMSNorm + INT8 quant
cudaError_t fused_rmsnorm_quant_int8(
    int8_t* x_out_i8, float* x_out_sc,
    const float* proj, const float* weight,
    int N, float eps, cudaStream_t stream);

// Batched RMSNorm — M sequences, same weight
cudaError_t fused_rmsnorm_batched(
    float* out, const float* inp, const float* weight,
    int H, float eps, int M, cudaStream_t stream);

// SwiGLU activation
cudaError_t apply_swiglu(
    float* out, const float* gate, const float* up,
    int num_elements, cudaStream_t stream);
```

### KV Cache & RoPE
```cpp
// Update KV cache (single sequence)
cudaError_t update_kv_cache(
    float* k_cache, float* v_cache, const float* k_new, const float* v_new,
    int batch_idx, int seq_pos, int num_heads, int head_dim,
    int max_seq_len, cudaStream_t stream);

// Graph-safe variant (CUDA Graph compatible)
cudaError_t update_kv_cache_device(
    float* k_cache, float* v_cache, const float* k_new, const float* v_new,
    int batch_idx, const int* d_seq_pos,
    int num_heads, int head_dim, int max_seq_len, cudaStream_t stream);

// Decode RoPE — reads seq_pos from device pointer (graph-safe)
cudaError_t fused_rope_decode(
    float* out_inplace, const float* cos_cache, const float* sin_cache,
    const int* seq_pos_ptr, int heads, int head_dim, int max_seq_len,
    cudaStream_t stream);
```

### Sampling
```cpp
// GPU sampler — argmax, temperature, top-k
cudaError_t sample_gpu(
    const float* logits, int vocab, float temperature, int top_k,
    int* out_id, unsigned long long rng_seed, int step, cudaStream_t stream);

// Repetition penalty
cudaError_t apply_repetition_penalty(
    float* logits, const int* recent, int num_recent,
    float penalty, int vocab, cudaStream_t stream);
```

## Architecture Decisions

### Batched Benchmark Architecture (text_generate_int4_batched.cu)
- **Buffer layout**: M×buffers (contiguous per-sequence): `[M][K]`, `[M][N]`
- **KV cache**: Separate per-sequence `[M][NL][nkv][MAXSEQ][hd]` — each sequence gets full KV cache
- **GEMV batching**: `gemv_int4_batched` — weight loaded once, reused across M tokens
- **Attention**: Per-sequence calls (batched M>2 non-deterministic)
- **Norm/quant**: Batched kernels `fused_rmsnorm_batched`, `quantize_int4_batched`
- **RoPE**: Per-sequence kernel launch with per-sequence rope_pos

### Server Architecture (inference_server_int4.cu)
- **Sequential processing**: Loops over `str_prompts` array, processes each with `generate()`
- **Single-sequence decode loop**: Uses M=1 buffers, reuses across requests
- **JSON protocol**: stdin→stdout JSON, reads `"prompts":["..."]`, returns `{"tokens":[],"text":[]}`
- **Repetition penalty**: On by default (1.5f), penalizes recent 64 tokens

### Decode Loop (36 layers)
```
for each step:
  embed lookup → quantize → QKV GEMV → head_norm → RoPE
  → update_kv_cache → attention_decode → o_proj → residual add
  → post_attn norm → quantize → MLP (gate+up+swiglu+down) → residual add
  → final_norm → lm_head → sample
```

## Key Data Structures

### DevW4 (INT4 weights)
```cpp
struct DevW4 { int K, N; uint8_t* d; float* sc; };
// .d: [K/2] packed nibbles, .sc: [K/16] block scales
```

### LW4 (Layer weights)
```cpp
struct LW4 {
    DevW4 q,k,v,o,g,u,d;  // projection weights
    float *qn,*kn;        // Q/K head norms
    float *rn_in,*rn_post; // RMSNorm weights
};
```

### ServerState (batched benchmark)
```cpp
struct ServerState {
    int M;
    float *d_x32, *d_xi_f, *d_residual;   // [M][H]
    uint8_t *d_x_i4; float *d_x_i4_sc;   // [M][H/2], [M][H/16]
    float *d_Q,*d_K,*d_V;                // [M][Q/KV]
    float *d_attn; uint8_t *d_attn_i4;    // [M][Q], [M][Q/2]
    float *d_proj, *d_gate, *d_up;        // [M][H], [M][I]
    uint8_t *d_mlp_i4; float *d_mlp_i4_sc;
    float *d_logits; int *d_next_id;      // [M][V], [M]
    float *d_kc, *d_vc;                   // [M][NL][nkv][MAXSEQ][hd]
    float *d_fn; float *d_fn_sc;
};
```

## Model Config (Qwen3-8B INT4)
| Param | Value |
|-------|-------|
| H | 4096 |
| Q (num_q_heads × head_dim) | 32 × 128 = 4096 |
| KV (num_kv_heads × head_dim) | 8 × 128 = 1024 |
| I (intermediate) | 12288 |
| nqh/nkv | 32/8 |
| hd | 128 |
| MAXSEQ | 4096 |
| NL | 36 |
| V | 151936 |

## Notes & Observations

1. **gemv_int8 deprecated**: `gemv_int8` uses 2D block scales [N/16 × K/16] and garbles 28-layer output. Use `gemv_int8_per_row` instead.

2. **Batched attention non-determinism**: `attention_decode_batched_gqa` with M>2 produces non-deterministic output. Root cause: race condition with concurrent blocks. Workaround: per-sequence attention calls.

3. **No CUDA Graph speedup**: Captured 28-layer decode loop with device-side seq_pos. Graph works but same speed as per-kernel (9.4ms/tok) because head_norm+RoPE not fused into capture.

4. **INT4 quality dead for 1.7B**: All sub-8-bit paths (INT4/INT5/FP4) produce garbled output after 28+ layers. 8B INT4 works (PPL 23.52, coherent).

5. **KV cache layout**: `[NL][ms][nkv][hd]` — layer-major, enables efficient layer offset for `update_kv_cache`.

## Sources
- `include/blackwell/kernels.h` — 177 kernel symbols, full API surface
- `bench/text_generate_int4_batched.cu` — batched architecture, 525 lines
- `server/inference_server_int4.cu` — server implementation, 350 lines

## Gaps
- No batched `apply_swiglu` kernel — per-sequence loop required
- No batched `vector_add_fp32` kernel — per-sequence loop required
- CUDA Graph integration deferred until head_norm/RoPE fusion possible