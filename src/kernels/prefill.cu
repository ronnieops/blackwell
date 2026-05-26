// src/kernels/prefill.cu — Separate prefill kernels for GEMM/attention-heavy pass
#include <cuda_runtime.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace blackwell {
namespace kernels {

// TODO(#7): Prefill entry point — orchestrates GEMM + attention for full sequence.
//   Unlike decode, prefill processes seq_len tokens in parallel.
//   Two main paths:
//     1. GEMV-style: for each layer, run attention_fp4 on full seq_len
//     2. GEMM-style: Q/K/V each batched as (batch×seq_len)×head_dim matrix multiply
//   Prefill is compute-bound (unlike decode which is memory-bound).
//   Focus: maximize tensor core utilization via large tile sizes.
//   Use cudaFuncSetPreferredSharedMemoryCarveout for more L1 for GEMM tiles.

cudaError_t run_prefill_layer(
    float* hidden_states,
    const void* Q_fp4, const void* K_fp4, const void* V_fp4,
    const float* Q_scale, const float* K_scale, const float* V_scale,
    float* k_cache, float* v_cache,  // written to for next decode pass
    int batch_size, int seq_len, int num_heads, int head_dim,
    int max_seq_len,
    cudaStream_t stream) {
    (void)hidden_states;
    (void)Q_fp4; (void)K_fp4; (void)V_fp4;
    (void)Q_scale; (void)K_scale; (void)V_scale;
    (void)k_cache; (void)v_cache;
    (void)batch_size; (void)seq_len; (void)num_heads;
    (void)head_dim; (void)max_seq_len; (void)stream;
    return cudaErrorNotReady;
}

} // namespace kernels
} // namespace blackwell
