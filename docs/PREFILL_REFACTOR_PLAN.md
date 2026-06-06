# Prefill Integration Plan — Blackwell Server

## Problem Statement

Server decode cache layout `[NL][ms][nkv][hd]` is **incompatible** with batched prefill attention in its current per-token, per-layer processing order. Even for M=1, hidden states differ between decode and prefill paths.

---

## Root Cause Analysis

### 1. Cache Layout Mismatch
**Decode cache**: `[NL][ms][nkv][hd]`
- Server processes tokens SEQUENTIALLY: token 0 through all 28 layers, then token 1...
- At layer l, token m's K/V written to position m
- When `attention_decode_batched_gqa` runs for token m, it reads K/V[0..m] from layer l's cache segment

**Benchmark prefill** (`bench/prefill_decode_benchmark.cu`):
- Also processes tokens sequentially (token 0 through all 28 layers, etc.)
- Uses same cache layout, same attention kernel
- **Produces correct output** because it handles residual flow correctly

### 2. Residual Flow Bug (Server's `batched_prefill`)

**Benchmark pattern** (correct):
```
Input: d_residual (input), d_x (working buffer)
Step 7: d_proj = Wo * attn_out
       d_tmp = d_proj + d_x = attn_out + input
Step 8: MLP runs on d_tmp (RMSNorm reads d_tmp)
        d_proj = Down(MLP)  // d_proj OVERWRITES previous value
Step 9: d_x = d_proj + d_tmp = MLP_out + (attn_out + input)  // correct final
Next layer: d_x used as input (contains attn+input+MLP)
```

**Server pattern** (broken):
```
Input: d_residual[m] = input, d_proj = MLP out
Step 7: d_proj = attn_out + input (in-place)
Step 8: d_proj overwritten by MLP_out
Step 9: d_residual[m] = MLP_out + saved(attn+input) = correct for final
       BUT: next layer's input RMSNorm reads d_proj = MLP_out (wrong!)
```

**Fix**: Use a working buffer (`d_x`) for the combined residual (attn+input+MLP), not `d_residual[m]`.

### 3. Key Insight

The server's `batched_prefill` used `d_residual[m]` as both:
- The INPUT to each layer's RMSNorm
- The OUTPUT buffer for the final hidden state

These are different uses. The benchmark uses `d_residual` only as input and `d_x` as working/output.

---

## Solution: Layer-First Processing

Process each LAYER through all M tokens, not each token through all layers:

```
for l in 0..NL-1:
    for m in 0..M-1:
        # QKV + head_norm + RoPE + cache write
    for m in 0..M-1:
        # Attention for token m (reads K/V[0..m] from this layer)
    # Residual 2 + move to next layer
    # d_x[m] = combined hidden state for token m at this layer
```

This matches the benchmark's data flow and ensures correct residual handling.

---

## Implementation Steps

### Step 1: Add Prefill Cache and Working Buffer

In `struct ServerState`:
```cpp
float* d_x;              // working buffer [M][H] for combined residual
float* d_prefill_kc;     // prefill KV cache [ms][NL][nkv][hd] (or reuse existing)
```

Allocate: `cudaMalloc(&S.d_x, S.M * S.H * 4);`

### Step 2: Rewrite `batched_prefill` with Layer-First Pattern

```cpp
static void batched_prefill(ServerState& S, int M) {
    if (M <= 0) return;
    size_t kv_seq_stride = (size_t)S.nkv * S.hd;

    // Initialize: d_x[m] = token embedding (input)
    for (int m = 0; m < M; m++) {
        cudaMemcpyAsync(S.d_x + m * S.H, S.d_residual[m], S.H * 4, cudaMemcpyDeviceToDevice, S.st);
    }

    for (int l = 0; l < S.NL; l++) {
        size_t kv_layer_off = (size_t)l * S.nkv * S.hd * S.ms;

        // ── QKV for all M tokens ──
        for (int m = 0; m < M; m++) {
            // RMSNorm + quantize using d_x[m] as input
            blackwell::kernels::fused_rmsnorm_quant_int8(
                S.d_xi8 + m * S.H, S.d_xi8s + m * (S.H / 16),
                S.d_x + m * S.H, S.d_rn_in[l], S.H, S.eps, S.st);

            // QKV projections
            blackwell::kernels::gemv_int8_warp(S.d_Q + m * S.Q, ...);
            blackwell::kernels::gemv_int8_warp(S.d_K + m * S.KV, ...);
            blackwell::kernels::gemv_int8_warp(S.d_V + m * S.KV, ...);

            // Q/K head norms
            head_norm_kernel<<<S.nqh, 128, 0, S.st>>>(S.d_Q + m * S.Q, S.layers[l].qn, ...);
            head_norm_kernel<<<S.nkv, 128, 0, S.st>>>(S.d_K + m * S.KV, S.layers[l].kn, ...);

            // RoPE
            int pos = m;
            cudaMemcpyAsync(S.d_seq_pos, &pos, sizeof(int), cudaMemcpyHostToDevice, S.st);
            rope_kernel<<<S.nqh, S.hd / 2, 0, S.st>>>(S.d_Q + m * S.Q, ...);
            rope_kernel<<<S.nkv, S.hd / 2, 0, S.st>>>(S.d_K + m * S.KV, ...);

            // Write K,V to cache at position m, layer l
            size_t kv_off = kv_layer_off + m * kv_seq_stride;
            cudaMemcpyAsync(S.d_kc + kv_off, S.d_K + m * S.KV, S.KV * 4, ...);
            cudaMemcpyAsync(S.d_vc + kv_off, S.d_V + m * S.KV, S.KV * 4, ...);
        }

        // ── Attention for all M tokens ──
        for (int m = 0; m < M; m++) {
            blackwell::kernels::attention_decode_batched_gqa(
                S.d_attn_out + m * S.Q, S.d_Q + m * S.Q,
                S.d_kc + kv_layer_off, S.d_vc + kv_layer_off,
                m, S.nqh, S.nkv, S.hd, S.ms,
                M, (int)kv_seq_stride, 0, S.st);
        }

        // ── Output projection + residual 1 ──
        for (int m = 0; m < M; m++) {
            blackwell::kernels::quantize_int8(S.d_attn_i8 + m * S.Q, ...);
            blackwell::kernels::gemv_int8_warp(S.d_proj + m * S.H, ...);
            // d_proj[m] = attn_out + d_x[m] (input)
            blackwell::kernels::vector_add_fp32(S.d_proj + m * S.H, S.d_proj + m * S.H, S.d_x + m * S.H, S.H, S.st);
            // Save for residual 2: d_proj[m] = attn + input
            cudaMemcpyAsync(S.d_tmp_save, S.d_proj + m * S.H, S.H * 4, ...);
        }

        // ── Post-attention RMSNorm + MLP ──
        for (int m = 0; m < M; m++) {
            blackwell::kernels::fused_rmsnorm_quant_int8(
                S.d_xi8 + m * S.H, S.d_xi8s + m * (S.H / 16),
                S.d_proj + m * S.H, S.d_rn_post[l], S.H, S.eps, S.st);
        }

        blackwell::kernels::gemv_int8_batched(S.d_gate, S.d_xi8, S.d_xi8s, ...);
        blackwell::kernels::gemv_int8_batched(S.d_up, S.d_xi8, S.d_xi8s, ...);
        for (int m = 0; m < M; m++) {
            blackwell::kernels::apply_swiglu(S.d_mlp + m * S.ID, ...);
            blackwell::kernels::quantize_int8(S.d_mlp_i8 + m * S.ID, ...);
        }
        blackwell::kernels::gemv_int8_batched(S.d_proj, S.d_mlp_i8, S.d_mlp_i8s, ...);

        // ── Residual 2: d_proj = MLP_out, add to saved(attn+input) ──
        for (int m = 0; m < M; m++) {
            // d_proj[m] = MLP_out, d_tmp_save = attn + input
            // d_x[m] = MLP_out + (attn + input) = correct final hidden state
            blackwell::kernels::vector_add_fp32(S.d_x + m * S.H, S.d_proj + m * S.H, S.d_tmp_save, S.H, S.st);
        }
    }

    // Copy final hidden state for last token to d_residual[0] for decode
    cudaMemcpyAsync(S.d_residual[0], S.d_x + (M - 1) * S.H, S.H * 4, cudaMemcpyDeviceToDevice, S.st);
}
```

### Step 3: Integrate into Server Main Loop

Replace the decode-based prefill with the new `batched_prefill`:

```cpp
int gen_start = (int)prompts[0].size();
if (gen_start > 0 && gen_start <= M) {
    embed_batch(S, prompts, 0);  // load all M tokens' embeddings to d_residual[m]
    batched_prefill(S, gen_start);
} else if (gen_start > M) {
    // Process in chunks of M
    for (int s = 0; s < gen_start; s += M) {
        int batch = std::min(M, gen_start - s);
        embed_batch(S, prompts, s);
        batched_prefill(S, batch);
    }
}
```

### Step 4: Verify Correctness

1. **M=1 test**: Pre-fill hidden state must match decode hidden state exactly.
2. **M>1 test**: Each token m's final hidden state must match sequential decode for that token.
3. **End-to-end test**: Server output must match benchmark output.

---

## Buffer Sizing

| Buffer | Size | Purpose |
|--------|------|---------|
| `d_x` | `M * H * 4` | Working buffer for combined residual |
| `d_kc/d_vc` | `NL * M * nkv * ms * hd * 4` | Existing decode cache (reused) |
| `d_tmp_save` | `H * 4` | Temp for saving attn+input before MLP |

Total additional allocation: `M * H * 4` bytes = 64KB for M=8, H=2048.

---

## Testing Checklist

- [ ] M=1: prefill hidden state == decode hidden state
- [ ] M=2: token 0 and token 1 hidden states match sequential decode
- [ ] M=8: all 8 tokens process correctly
- [ ] Temperature=0: deterministic output matches benchmark
- [ ] Temperature=1: stochastic output matches benchmark distribution
- [ ] gen_start > M: multiple chunks process correctly
- [ ] HTTP endpoints: all work with prefill enabled

---

## Estimated Time

- Step 1 (add buffer): 10 min
- Step 2 (rewrite function): 45 min
- Step 3 (integrate): 15 min
- Step 4 (verify): 30 min
- Debug if issues: 30-60 min

**Total**: 2-3 hours

---

## Alternative: Skip Prefill Cache

If the above is too complex, an even simpler approach:

1. Keep server decode-only (current state)
2. For prefill requests, fall back to per-token decode processing (current behavior)
3. Document that prefill is processed via decode path

This sacrifices prefill speedup but keeps the server simple and correct.