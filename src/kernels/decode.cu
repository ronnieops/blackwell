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
// Decode attention kernel v2: warp-parallel QK dot products + smem-tiled
//
// 8 warps × 32 threads = 256 threads per head block.
//
// QK scores: all 8 warps participate in dot product.
//   smem_Q[128] — Q loaded once by warp 0, broadcast to all via smem.
//   For each position t, all 8 warps load K[t*128..t*128+127] via coalesced
//   float4 reads (32 loads of 4 floats = 128 values), dot with Q in parallel.
//
//   Each warp: 128/32 = 4 elements per thread → dot += Q[d] * K[t][d] for 4 d's.
//   Warp reduce: __shfl_xor_sum over 4 → 1 score per warp.
//   First warp collects 8 scores → writes scores[t] = sum(8 warp scores) * scale.
//
// Softmax: sequential over npos (npos ≪ 32*128 = 4096, fine for typical decode).
//
// V weighted sum: parallel over d. Each thread handles d range.
// ---------------------------------------------------------------------------
__launch_bounds__(256, 1)
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

    // GQA: map Q head to KV head
    int kv_head = head * num_kv_heads / num_q_heads;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid & 31;
    int npos = seq_pos + 1;
    int h_base = kv_head * max_seq_len * head_dim;

    Q += head * head_dim;
    output += head * head_dim;

    // Shared memory: Q[128] + scores[4096]
    // 128 floats for Q (512 B) + 4096 floats for scores (16384 B) = 16896 B
    extern __shared__ float smem[];
    float* smem_Q = smem;
    float* scores = smem + head_dim;

    // Step 0: Load Q into smem (warp 0 does it, broadcast)
    if (warp_id == 0 && lane_id < head_dim) {
        smem_Q[lane_id] = Q[lane_id];
    }
    __syncthreads();

    // Each thread: load 4 consecutive Q elements into registers
    // (head_dim=128 → 4 per thread × 32 threads = 128, all warps get same)
    float Q_reg[4];
    int q_base = (lane_id * 4) % head_dim;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int d = i * 32 + lane_id;
        Q_reg[i] = (d < head_dim) ? Q[d] : 0.0f;
    }

    // Step 1: Compute scores[t] for all t using all 8 warps
    // Each warp covers npos/8 positions, each thread in warp computes
    // dot product for 4 K elements, warp-reduces.
    int warp_stride = 8;  // 8 warps
    for (int t_start = warp_id; t_start < npos; t_start += warp_stride) {
        int t = t_start;
        if (t > seq_pos) break;

        const float* K_t = K_cache + h_base + t * head_dim;

        // Load 4 K values (same positions as Q_reg), multiply-accumulate
        float dot_local = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int d = i * 32 + lane_id;
            float kv = (d < head_dim) ? K_t[d] : 0.0f;
            dot_local += Q_reg[i] * kv;
        }

        // Warp reduction
        for (int off = 16; off > 0; off >>= 1)
            dot_local += __shfl_xor_sync(0xffffffff, dot_local, off);

        // First lane of each warp holds partial score
        if (lane_id == 0) {
            scores[t] = dot_local * scale;
        }
    }
    __syncthreads();

    // First warp: finalize scores by summing partial warp scores
    // Only needed if multiple warps contribute to same position, but
    // we assigned disjoint positions above, so scores[t] is already final.

    // Step 2: Softmax over valid scores
    float maxv = -FLT_MAX;
    for (int t = 0; t < npos; ++t) {
        if (scores[t] > maxv) maxv = scores[t];
    }

    float sumexp = 0.0f;
    for (int t = 0; t < npos; ++t) {
        float e = __expf(scores[t] - maxv);
        scores[t] = e;
        sumexp += e;
    }
    float inv_sum = 1.0f / (sumexp + 1e-9f);

    // Step 3: Weighted sum over V — each thread handles one d (strided)
    // All scores in smem, V is global — sequential over t is fine.
    // head_dim=128, strided over 32 lanes → each thread does 4 elements.
    for (int d = lane_id; d < head_dim; d += 32) {
        float out_val = 0.0f;
        for (int t = 0; t < npos; ++t) {
            float w = scores[t] * inv_sum;
            out_val += w * V_cache[h_base + t * head_dim + d];
        }
        output[d] = out_val;
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
// ===========================================================================
namespace {
    static int* d_seq_pos_global = nullptr;
    static int* h_seq_pos_pinned = nullptr;
    static cudaError_t alloc_seq_pos() {
        if (d_seq_pos_global) return cudaSuccess;
        cudaError_t e = cudaMalloc(&d_seq_pos_global, sizeof(int));
        if (e != cudaSuccess) return e;
        return cudaHostAlloc(&h_seq_pos_pinned, sizeof(int), cudaHostAllocDefault);
    }
    // Write seq_pos to pinned memory for graph-safe read by cudaMemcpyAsync
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

    static bool attr_set = false;
    constexpr int smem_bytes = 4096 * 4;  // 16 KB for scores (4096 positions)
    if (!attr_set) {
        e = cudaFuncSetAttribute(
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

    static bool attr_set = false;
    constexpr int smem_bytes = 4096 * 4;
    if (!attr_set) {
        e = cudaFuncSetAttribute(
            attention_decode_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes);
        if (e != cudaSuccess) return e;
        attr_set = true;
    }

    float scale = 1.0f / sqrtf((float)head_dim);
    attention_decode_kernel<<<
        dim3(num_q_heads), dim3(256), smem_bytes, stream>>>(
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
