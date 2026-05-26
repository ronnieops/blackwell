// src/kernels/decode.cu — Decode-path GEMV + KV-cache bandwidth-optimized kernels
#include <cuda_runtime.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace blackwell {
namespace kernels {

// TODO(#6): Load KV-cache for current position — vectorized, coalesced.
//   Cache layout: k_cache[b][head][pos][head_dim] in transposed form for
//   efficient K @ Q^T during attention (pos-contiguous for retrieval, head-contiguous for compute).
//   Use 32-byte aligned loads for the 128-bit bus.

// Load K/V for all heads at current seq_pos into registers
cudaError_t load_kv_cache_qkgv(
    float* Q, float* K_val, float* V_val,
    const float* k_cache, const float* v_cache,
    int batch_idx, int seq_pos, int num_heads,
    int head_dim, int max_seq_len, cudaStream_t stream) {
    (void)Q; (void)K_val; (void)V_val;
    (void)k_cache; (void)v_cache;
    (void)batch_idx; (void)seq_pos; (void)num_heads;
    (void)head_dim; (void)max_seq_len; (void)stream;
    return cudaErrorNotReady;
}

// Update KV-cache with new K/V vectors (at seq_pos position)
cudaError_t update_kv_cache(
    float* k_cache, float* v_cache,
    const float* k_new, const float* v_new,
    int batch_idx, int seq_pos, int num_heads,
    int head_dim, int max_seq_len, cudaStream_t stream) {
    (void)k_cache; (void)v_cache;
    (void)k_new; (void)v_new;
    (void)batch_idx; (void)seq_pos; (void)num_heads;
    (void)head_dim; (void)max_seq_len; (void)stream;
    return cudaErrorNotReady;
}

} // namespace kernels
} // namespace blackwell
