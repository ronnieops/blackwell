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
    const int* seq_pos_ptr,
    int num_heads,
    int head_dim,
    int max_seq_len)
{
    int seq_pos = *seq_pos_ptr;
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
// Decode attention kernel v3: float4 vectorized K/V reads + 128 threads
//
// Same algorithm as v2 (scores in smem, sequential softmax) but with:
// - 128 threads (4 warps) instead of 256
// - Float4 vectorized K cache reads for QK dot product
// - Reduced smem: Q[128] + scores[4096] = 17 KB
// ---------------------------------------------------------------------------
__launch_bounds__(128, 2)
__global__ void attention_decode_kernel(
    float* __restrict__ output,
    const float* __restrict__ Q,
    const float* __restrict__ K_cache,
    const float* __restrict__ V_cache,
    const int* seq_pos_ptr,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    float scale)
{
    int seq_pos = *seq_pos_ptr;
    int head = blockIdx.x;
    if (head >= num_q_heads) return;

    int kv_head = head * num_kv_heads / num_q_heads;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid & 31;
    int npos = seq_pos + 1;
    int h_base = kv_head * max_seq_len * head_dim;

    Q += head * head_dim;
    output += head * head_dim;

    extern __shared__ float smem[];
    float* smem_Q = smem;
    float* scores = smem + head_dim;

    // Load Q into smem (all 128 threads participate)
    if (tid < head_dim) {
        smem_Q[tid] = Q[tid];
    }
    __syncthreads();

    // Load Q into registers: each thread gets Q[lane_id*4 .. lane_id*4+3]
    // 32 threads × 4 = 128 elements = full head_dim
    float Q_reg[4];
    {
        const float4* Q4 = reinterpret_cast<const float4*>(smem_Q);
        float4 q4 = Q4[lane_id];
        Q_reg[0] = q4.x; Q_reg[1] = q4.y; Q_reg[2] = q4.z; Q_reg[3] = q4.w;
    }

    // Compute scores: 4 warps, each handles npos/4 positions
    int num_warps = 4;
    for (int t = warp_id; t < npos; t += num_warps) {
        // Float4 vectorized K read
        const float4* K4 = reinterpret_cast<const float4*>(
            K_cache + h_base + t * head_dim);

        // Each thread computes partial dot for its 4 elements
        // Q_reg[i] = Q[lane_id*4+i], K4[lane_id] = K[lane_id*4..lane_id*4+3]
        float4 kv = K4[lane_id];
        float dot = Q_reg[0] * kv.x + Q_reg[1] * kv.y
                  + Q_reg[2] * kv.z + Q_reg[3] * kv.w;

        // Warp reduce
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, off);

        if (lane_id == 0) {
            scores[t] = dot * scale;
        }
    }
    __syncthreads();

    // Softmax
    float maxv = -FLT_MAX;
    for (int t = 0; t < npos; ++t)
        if (scores[t] > maxv) maxv = scores[t];

    float sumexp = 0.0f;
    for (int t = 0; t < npos; ++t) {
        float e = __expf(scores[t] - maxv);
        scores[t] = e;
        sumexp += e;
    }
    float inv_sum = 1.0f / (sumexp + 1e-9f);

    // V weighted sum: each thread handles elements strided by blockDim.x
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float out_val = 0.0f;
        for (int t = 0; t < npos; ++t) {
            out_val += scores[t] * V_cache[h_base + t * head_dim + d];
        }
        output[d] = out_val * inv_sum;
    }
}

// ===========================================================================
// Single-token decode attention
} // anonymous namespace

// ===========================================================================
// Module-level seq_pos infrastructure (shared across all decode wrappers)
//
// Design for CUDA Graph compatibility:
//   h_seq_pos_pinned — pinned host memory, address stable across graph lifetime
//   d_seq_pos_global — device copy, kernel reads via pointer
//
// Graph captures cudaMemcpyAsync H2D node. Pinned memory guarantee:
// - Address is stable (not stack-allocated)
// - Contents can be updated between graph launches
// - Graph replays read latest pinned value
//
// Thread-safety: alloc_seq_pos() uses atomic flag for one-time init.
// write_seq_pos() is NOT thread-safe — caller must ensure no concurrent
// writes to h_seq_pos_pinned from different threads. In practice, each
// stream's memcpy is ordered, so Stream A's kernel reads Stream A's value.
// ===========================================================================
namespace {
    static int* d_seq_pos_global = nullptr;
    static int* h_seq_pos_pinned = nullptr;
    static volatile int seq_pos_init_flag = 0;  // 0=uninit, 1=initing, 2=done
    static volatile int smem_attr_set = 0;  // 0=uninit, 1=initing, 2=done

    static cudaError_t alloc_seq_pos() {
        if (seq_pos_init_flag == 2) return cudaSuccess;
        // Spin-lock for one-time initialization
        int old = __sync_val_compare_and_swap(&seq_pos_init_flag, 0, 1);
        if (old == 0) {
            cudaError_t e = cudaMalloc(&d_seq_pos_global, sizeof(int));
            if (e != cudaSuccess) { seq_pos_init_flag = 0; return e; }
            e = cudaHostAlloc(&h_seq_pos_pinned, sizeof(int), cudaHostAllocDefault);
            if (e != cudaSuccess) { cudaFree(d_seq_pos_global); d_seq_pos_global = nullptr; seq_pos_init_flag = 0; return e; }
            seq_pos_init_flag = 2;
        } else {
            while (seq_pos_init_flag != 2) {}
        }
        return cudaSuccess;
    }
    // Write seq_pos to pinned memory for graph-safe read by cudaMemcpyAsync
    // NOT thread-safe — caller must serialize.
    static void write_seq_pos(int seq_pos) {
        if (h_seq_pos_pinned) *h_seq_pos_pinned = seq_pos;
    }
}

// ===========================================================================
// Public API
// ===========================================================================

cudaError_t attention_decode(
    float* output, const float* Q,
    const float* K_cache, const float* V_cache,
    int seq_pos, int num_heads, int head_dim,
    int max_seq_len, cudaStream_t stream) {

    cudaError_t e;
    e = alloc_seq_pos();
    if (e != cudaSuccess) return e;
    write_seq_pos(seq_pos);
    e = cudaMemcpyAsync(d_seq_pos_global, h_seq_pos_pinned, sizeof(int),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) return e;

    constexpr int smem_bytes = 4096 * 4;  // 16 KB for scores (4096 positions)
    if (smem_attr_set == 0) {
        int old = __sync_val_compare_and_swap(&smem_attr_set, 0, 1);
        if (old == 0) {
            cudaFuncSetAttribute(
                attention_decode_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_bytes);
            smem_attr_set = 2;
        }
    }

    float scale = 1.0f / sqrtf((float)head_dim);
    attention_decode_kernel<<<
        dim3(num_heads), dim3(128), smem_bytes, stream>>>(
        output, Q, K_cache, V_cache,
        d_seq_pos_global, num_heads, num_heads /* kv_heads = q_heads for non-GQA */,
        head_dim, max_seq_len, scale);

    return cudaPeekAtLastError();
}

cudaError_t attention_decode_gqa(
    float* output, const float* Q,
    const float* K_cache, const float* V_cache,
    int seq_pos, int num_q_heads, int num_kv_heads,
    int head_dim, int max_seq_len, cudaStream_t stream) {

    cudaError_t e;
    e = alloc_seq_pos();
    if (e != cudaSuccess) return e;
    write_seq_pos(seq_pos);
    e = cudaMemcpyAsync(d_seq_pos_global, h_seq_pos_pinned, sizeof(int),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) return e;

    constexpr int smem_bytes = 4096 * 4;
    if (smem_attr_set == 0) {
        int old = __sync_val_compare_and_swap(&smem_attr_set, 0, 1);
        if (old == 0) {
            cudaFuncSetAttribute(
                attention_decode_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_bytes);
            smem_attr_set = 2;
        }
    }

    float scale = 1.0f / sqrtf((float)head_dim);
    attention_decode_kernel<<<
        dim3(num_q_heads), dim3(128), smem_bytes, stream>>>(
        output, Q, K_cache, V_cache,
        d_seq_pos_global, num_q_heads, num_kv_heads,
        head_dim, max_seq_len, scale);

    return cudaPeekAtLastError();
}

cudaError_t update_kv_cache(
    float* k_cache, float* v_cache,
    const float* k_new, const float* v_new,
    int batch_idx, int seq_pos, int num_heads,
    int head_dim, int max_seq_len, cudaStream_t stream) {

    cudaError_t e;
    e = alloc_seq_pos();
    if (e != cudaSuccess) return e;
    write_seq_pos(seq_pos);
    e = cudaMemcpyAsync(d_seq_pos_global, h_seq_pos_pinned, sizeof(int),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) return e;

    (void)batch_idx;
    // Only support batch_idx=0 for now
    if (batch_idx != 0) return cudaErrorInvalidValue;

    int total = num_heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    update_kv_kernel<<<blocks, threads, 0, stream>>>(
        k_cache, v_cache,
        k_new, v_new,
        d_seq_pos_global, num_heads, head_dim, max_seq_len);

    return cudaPeekAtLastError();
}

cudaError_t update_decode_seq_pos(int seq_pos, cudaStream_t stream) {
    cudaError_t e = alloc_seq_pos();
    if (e != cudaSuccess) return e;
    write_seq_pos(seq_pos);
    e = cudaMemcpyAsync(d_seq_pos_global, h_seq_pos_pinned, sizeof(int),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) return e;
    return cudaPeekAtLastError();
}

// Return device pointer to seq_pos (for CUDA Graph RoPE)
cudaError_t get_seq_pos_device_ptr(int** ptr) {
    cudaError_t e = alloc_seq_pos();
    if (e != cudaSuccess) return e;
    *ptr = d_seq_pos_global;
    return cudaSuccess;
}

// Return pinned host pointer to seq_pos (for graph-safe host writes)
cudaError_t get_seq_pos_host_ptr(int** ptr) {
    cudaError_t e = alloc_seq_pos();
    if (e != cudaSuccess) return e;
    *ptr = h_seq_pos_pinned;
    return cudaSuccess;
}

cudaError_t load_kv_cache_qkgv(
    float* Q, float* K_val, float* V_val,
    const float* k_cache, const float* v_cache,
    int batch_idx, int seq_pos, int num_heads,
    int head_dim, int max_seq_len, cudaStream_t stream) {

    if (!Q || !K_val || !V_val || !k_cache || !v_cache)
        return cudaErrorInvalidValue;

    // Read K/V from cache at [batch_idx, seq_pos] → K_val, V_val
    // Q is passed through unchanged.
    int per_head = max_seq_len * head_dim;
    for (int h = 0; h < num_heads; ++h) {
        int offset = h * per_head + seq_pos * head_dim;
        cudaMemcpyAsync(K_val + h * head_dim, k_cache + offset,
                        head_dim * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(V_val + h * head_dim, v_cache + offset,
                        head_dim * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }
    return cudaPeekAtLastError();
}

} // namespace kernels
} // namespace blackwell
