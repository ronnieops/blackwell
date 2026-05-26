// src/kernels/attention.cu — Flash-style attention for prefill
#include <cuda_runtime.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace blackwell {
namespace kernels {

// TODO(#5/#7): Fused attention prefill.
//   Steps: Q @ K^T → scale → softmax (or softmax-freeOnlineSoftmax) →
//          softmax_out @ V → add residual.
//   KV-cache: per-batch, stored in transpose-optimized layout.
//   Use shared-memory tiling for K/V tiles (respect 99 KB/block).
//   Return logits in FP32 from softcap.

// Computes attention output:  softmax(Q K^T / sqrt(d_k)) V
// Q/K/V already dequantized (FP4 → FP32 in callers or pre-loaded).

cudaError_t attention_fp4(
    float* output, const void* Q_fp4, const void* K_fp4, const void* V_fp4,
    const float* Q_scale, const float* K_scale, const float* V_scale,
    int batch_size, int seq_len, int num_heads, int head_dim,
    float scale, cudaStream_t stream) {
    (void)output; (void)Q_fp4; (void)K_fp4; (void)V_fp4;
    (void)Q_scale; (void)K_scale; (void)V_scale;
    (void)batch_size; (void)seq_len; (void)num_heads;
    (void)head_dim; (void)scale; (void)stream;
    return cudaErrorNotReady;
}

} // namespace kernels
} // namespace blackwell
