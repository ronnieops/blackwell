// src/kernels/prefill.cu — Prefill layer orchestration
//
// Orchestrates GEMM + attention for full-sequence prefill.
// Pre-quantizes activations to INT8, then uses __dp4a GEMM for projections.

#include <cuda_runtime.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace blackwell {
namespace kernels {

// Run one prefill layer: GEMM projections + attention + update KV cache.
// hidden_states: [batch_size * seq_len * hidden_dim] FP32 input/output
// W_q, W_k, W_v, W_o: INT8 weight matrices [out_dim × hidden_dim] with scales
// k_cache, v_cache: [num_kv_heads * max_seq_len * head_dim] — written at seq_pos 0..seq_len-1
cudaError_t run_prefill_layer(
    float* hidden_states,
    const int8_t* W_q, const float* W_q_sc,
    const int8_t* W_k, const float* W_k_sc,
    const int8_t* W_v, const float* W_v_sc,
    const int8_t* W_o, const float* W_o_sc,
    float* k_cache, float* v_cache,
    int batch_size, int seq_len, int num_heads, int num_kv_heads,
    int hidden_dim, int head_dim, int max_seq_len,
    cudaStream_t stream)
{
    if (!hidden_states || !W_q || !W_k || !W_v || !W_o)
        return cudaErrorInvalidValue;

    int M = batch_size * seq_len;
    int Q_dim = num_heads * head_dim;
    int KV_dim = num_kv_heads * head_dim;

    // Allocate temporary buffers
    float *d_Q, *d_K, *d_V, *d_attn, *d_proj;
    int8_t *d_a_i8, *d_o_i8;
    float *d_a_sc, *d_o_sc;
    size_t q_size = M * Q_dim * sizeof(float);
    size_t kv_size = M * KV_dim * sizeof(float);
    size_t proj_size = M * hidden_dim * sizeof(float);

    cudaError_t e;
    e = cudaMalloc(&d_Q, q_size); if (e != cudaSuccess) return e;
    e = cudaMalloc(&d_K, kv_size); if (e != cudaSuccess) return e;
    e = cudaMalloc(&d_V, kv_size); if (e != cudaSuccess) return e;
    e = cudaMalloc(&d_attn, q_size); if (e != cudaSuccess) return e;
    e = cudaMalloc(&d_proj, proj_size); if (e != cudaSuccess) return e;

    // Pre-quantized activation buffers
    int a_elems = M * hidden_dim;
    int a_nblks = a_elems / 16;
    e = cudaMalloc(&d_a_i8, a_elems); if (e != cudaSuccess) { cudaFree(d_proj); cudaFree(d_attn); cudaFree(d_V); cudaFree(d_K); cudaFree(d_Q); return e; }
    e = cudaMalloc(&d_a_sc, a_nblks * sizeof(float)); if (e != cudaSuccess) { cudaFree(d_a_i8); cudaFree(d_proj); cudaFree(d_attn); cudaFree(d_V); cudaFree(d_K); cudaFree(d_Q); return e; }

    // Pre-quantize hidden_states to INT8 once (reused for Q, K, V)
    e = quantize_int8(d_a_i8, d_a_sc, hidden_states, a_elems, stream);
    if (e != cudaSuccess) { cudaFree(d_a_sc); cudaFree(d_a_i8); cudaFree(d_proj); cudaFree(d_attn); cudaFree(d_V); cudaFree(d_K); cudaFree(d_Q); return e; }

    // QKV projections via WMMA tensor cores (pre-quantized INT8 × INT8)
    e = gemm_int8_wmma_fast(d_Q, d_a_i8, d_a_sc, W_q, W_q_sc, M, Q_dim, hidden_dim, stream);
    if (e != cudaSuccess) goto cleanup;
    e = gemm_int8_wmma_fast(d_K, d_a_i8, d_a_sc, W_k, W_k_sc, M, KV_dim, hidden_dim, stream);
    if (e != cudaSuccess) goto cleanup;
    e = gemm_int8_wmma_fast(d_V, d_a_i8, d_a_sc, W_v, W_v_sc, M, KV_dim, hidden_dim, stream);
    if (e != cudaSuccess) goto cleanup;

    // Update KV cache: write K/V for all seq positions
    // For prefill, we write seq_len positions starting at seq_pos=0
    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < num_kv_heads; ++h) {
            int src_off = s * num_kv_heads * head_dim + h * head_dim;
            int dst_off = h * max_seq_len * head_dim + s * head_dim;
            cudaMemcpyAsync(k_cache + dst_off, d_K + src_off,
                            head_dim * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(v_cache + dst_off, d_V + src_off,
                            head_dim * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        }
    }

    // Attention
    {
        float sc_at = 1.0f / sqrtf((float)head_dim);
        int qpg = num_heads / num_kv_heads;
        e = attention_prefill(d_attn, d_Q, d_K, d_V,
                              seq_len, head_dim, num_heads, num_kv_heads,
                              qpg, sc_at, stream);
        if (e != cudaSuccess) goto cleanup;
    }

    // Output projection: [M, Q_dim] × [hidden_dim, Q_dim]^T → [M, hidden_dim]
    // Pre-quantize attention output to INT8, then __dp4a GEMM
    {
        int o_elems = M * Q_dim;
        int o_nblks = o_elems / 16;
        e = cudaMalloc(&d_o_i8, o_elems); if (e != cudaSuccess) goto cleanup;
        e = cudaMalloc(&d_o_sc, o_nblks * sizeof(float)); if (e != cudaSuccess) { cudaFree(d_o_i8); goto cleanup; }
        e = quantize_int8(d_o_i8, d_o_sc, d_attn, o_elems, stream);
        if (e != cudaSuccess) { cudaFree(d_o_sc); cudaFree(d_o_i8); goto cleanup; }
        e = gemm_int8_wmma_fast(d_proj, d_o_i8, d_o_sc, W_o, W_o_sc, M, hidden_dim, Q_dim, stream);
        cudaFree(d_o_sc);
        cudaFree(d_o_i8);
    }
    if (e != cudaSuccess) goto cleanup;

    // Residual connection
    e = vector_add_fp32(hidden_states, d_proj, hidden_states, M * hidden_dim, stream);

cleanup:
    cudaFree(d_a_sc);
    cudaFree(d_a_i8);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_attn);
    cudaFree(d_proj);
    return e;
}

} // namespace kernels
} // namespace blackwell
