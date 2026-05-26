// src/kernels/decode.cu — Decode-path kernels for single-token generation
//
// Decode is memory-bound: 1 token, seq_len grows each step.
//
// KV cache layout:
//   k_cache[head * max_seq_len * head_dim + pos * head_dim + d]
//   v_cache[head * max_seq_len * head_dim + pos * head_dim + d]
//
// Attention decode for one token at seq_pos:
//   1) score[h][t] = Q[h][:] · K_cache[h][t][:]  for t in 0..seq_pos
//   2) w[t] = exp(score[t] / sqrt(head_dim))
//   3) norm = sum w[t]
//   4) O[h][d] = sum_t (w[t]/norm) * V_cache[h][t][d]

#include <cuda_runtime.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"
#include <cfloat>

namespace blackwell {
namespace kernels {
namespace {

// ---------------------------------------------------------------------------
// update_kv_kernel: write K_new/V_new to position seq_pos
// ---------------------------------------------------------------------------
__global__ void update_kv_kernel(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const float* __restrict__ k_new,
    const float* __restrict__ v_new,
    int seq_pos,
    int num_heads,
    int head_dim,
    int max_seq_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * head_dim;
    if (idx >= total) return;
    int h = idx / head_dim;
    int d = idx % head_dim;
    int offset = h * max_seq_len * head_dim + seq_pos * head_dim + d;
    k_cache[offset] = k_new[idx];
    v_cache[offset] = v_new[idx];
}

// ---------------------------------------------------------------------------
// Decode attention kernel: one token × cached KV sequence
//
// Strategy: compute scores[i] = Q · K_cache[i] for all i in 0..seq_pos,
// then softmax, then weighted sum.  Batched per head.
//
// Head parallelism: grid = num_heads, each block handles one head.
// Seq parallelism within each head:
//   thread i in 0..255 handles seq positions i, i+256, i+512, ...
//   accumulates Q_d * K_td for each position.
//   Final accumulate over head_dim via warp reduction.
//
// Shared memory: score_buf[max_seq_len_per_block] for softmax.
// For seq_pos > 256, we tile: process 256 positions at a time.
// ---------------------------------------------------------------------------
__launch_bounds__(256, 1)
__global__ void attention_decode_kernel(
    float* __restrict__ output,     // [num_heads * head_dim]
    const float* __restrict__ Q,    // [num_heads * head_dim]
    const float* __restrict__ K_cache, // [num_heads * max_seq_len * head_dim]
    const float* __restrict__ V_cache, // [num_heads * max_seq_len * head_dim]
    int seq_pos,       // inclusive max index
    int num_heads,
    int head_dim,
    int max_seq_len,
    float scale)       // 1/sqrt(head_dim)
{
    int head = blockIdx.x;
    if (head >= num_heads) return;

    int tid = threadIdx.x;
    int npos = seq_pos + 1;  // number of cached positions
    int h_base = head * max_seq_len * head_dim;

    Q += head * head_dim;
    output += head * head_dim;

    // Shared score buffer: up to 4096 positions (16 KB)
    // Allocate max is enough for typical decode context
    extern __shared__ float scores[];

    // Step 1: Compute scores[t] = Q · K_cache[t] for all t
    // Each thread loads Q[d] once (register), then loops positions
    float Q_reg[4] = {};
    if (tid < head_dim) {
        Q_reg[0] = Q[tid];
    }
    // Q is 128 for Qwen3-1.7B, fits 4 floats per thread
    if (tid < head_dim) {
        // broadcast Q to all threads for K-dim reduction
    }
    // Actually: each thread loads one Q[d], then for each pos t,
    // loads K_cache[t][d] and accumulates dot product.
    // After accumulation, warp-reduce over d.
    // This gives scores[t] for positions handled by this thread.

    // Each thread handles positions with stride 256
    for (int t_start = 0; t_start < npos; t_start += 256) {
        int t = t_start + tid;
        if (t > seq_pos) break;

        float dot = 0.0f;
        // Dot product over head_dim for position t
        for (int d = 0; d < head_dim; ++d) {
            dot += Q[d] * K_cache[h_base + t * head_dim + d];
        }
        scores[t] = dot * scale;
    }
    __syncthreads();

    // Step 2: Softmax over valid scores
    // Find max score for numerical stability
    float maxv = -FLT_MAX;
    for (int t = 0; t < npos; ++t) {
        if (scores[t] > maxv) maxv = scores[t];
    }

    // Compute exp(score - max) and sum
    float sumexp = 0.0f;
    for (int t = 0; t < npos; ++t) {
        float e = __expf(scores[t] - maxv);
        scores[t] = e;  // reuse for weight
        sumexp += e;
    }

    float inv_sum = 1.0f / (sumexp + 1e-9f);

    // Step 3: Weighted sum over V: O[d] = sum_t (scores[t]/sumexp) * V[t][d]
    // Each thread handles one d (or multiple if head_dim > 256)
    float result = 0.0f;
    if (tid < head_dim) {
        for (int t = 0; t < npos; ++t) {
            float w = scores[t] * inv_sum;
            result += w * V_cache[h_base + t * head_dim + tid];
        }
        output[tid] = result;
    }
}

// ===========================================================================
// Single-token decode attention
} // anonymous namespace

// ===========================================================================
// Public API
// ===========================================================================

cudaError_t attention_decode(
    float* output, const float* Q,
    const float* K_cache, const float* V_cache,
    int seq_pos, int num_heads, int head_dim,
    int max_seq_len, cudaStream_t stream) {

    static bool attr_set = false;
    constexpr int smem_bytes = 4096 * 4;  // 16 KB for scores (4096 positions)
    if (!attr_set) {
        cudaError_t e = cudaFuncSetAttribute(
            attention_decode_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes);
        if (e != cudaSuccess) return e;
        attr_set = true;
    }

    float scale = 1.0f / sqrtf((float)head_dim);
    attention_decode_kernel<<<
        dim3(num_heads), dim3(256), smem_bytes, stream>>>(
        output, Q, K_cache, V_cache,
        seq_pos, num_heads, head_dim, max_seq_len, scale);

    return cudaPeekAtLastError();
}

cudaError_t update_kv_cache(
    float* k_cache, float* v_cache,
    const float* k_new, const float* v_new,
    int batch_idx, int seq_pos, int num_heads,
    int head_dim, int max_seq_len, cudaStream_t stream) {

    (void)batch_idx;
    // Only support batch_idx=0 for now
    if (batch_idx != 0) return cudaErrorInvalidValue;

    int total = num_heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    update_kv_kernel<<<blocks, threads, 0, stream>>>(
        k_cache, v_cache,
        k_new, v_new,
        seq_pos, num_heads, head_dim, max_seq_len);

    return cudaPeekAtLastError();
}

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

} // namespace kernels
} // namespace blackwell
