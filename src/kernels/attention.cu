// src/kernels/attention.cu — Flash-style attention for prefill
//
// Computes: softmax(Q @ K^T / sqrt(d_k)) @ V
// Strategy: flash attention with online softmax.
// - Q: (M, head_dim) per head
// - K: (M, head_dim) per KV head
// - V: (M, head_dim) per KV head
// - O: (M, head_dim) per group (all query heads sharing same KV)
//
// Kernel: 1 block per KV head. Each block processes all 16 query heads in group.
// M=128, head_dim=64, Q/K/V in FP32 (from upstream GEMM).
//
// Smem layout (max 64 KB per block for Blackwell consumer):
//   K[128×64] + V[128×64] = 32 KB. Q is in registers (~8 KB per warp).
//   Remaining for softmax intermediates (~24 KB).
//
// Build: cmake -B build && cmake --build build --parallel

#include <cuda_runtime.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"
#include <algorithm>
#include <cmath>

namespace blackwell {
namespace kernels {
namespace {

// Flash attention for one KV head + its group of query heads
// Uses attn_coop pattern (proven correct in fused_prefill):
//   - K in shared memory (32 KB)
//   - Q in shared memory (256 B), overlaid per row
//   - Scores S[M] in shared memory
//   - Online softmax in shared memory
//   - V from global (L2 cached, ~1.5% of 32 MB)
//
// Grid: (num_kv_heads × num_q_per_group, M/4, 1)
// Block: 32 threads (1 warp, handles 4 rows)

__global__ void flash_attention_kernel(
    float* __restrict__ O_out,
    const float* __restrict__ Q_in,
    const float* __restrict__ K_in,
    const float* __restrict__ V_in,
    int M, int head_dim, int num_q_heads, int num_kv_heads,
    int num_q_per_group, float scale) {

    // Shared memory layout: K[M*head_dim] + Q[head_dim] + S[M]
    extern __shared__ char smem_base[];
    float* K_s = reinterpret_cast<float*>(smem_base);
    float* Q_s = K_s + M * head_dim;       // Q[head_dim] = 256 B
    float* S_s = Q_s + head_dim;           // S[M] = 512 B
    // Total: 32 KB + 256 B + 512 B ≈ 32.75 KB

    int q_head = blockIdx.x;
    int m0 = blockIdx.y * 4;  // 4 rows per block
    int t = threadIdx.x;
    int lane = t & 31;

    if (q_head >= num_q_heads || m0 >= M) return;

    // Determine KV head for this query head (assumes uniform groups)
    int kv_head = q_head / num_q_per_group;
    const float* K_head = K_in + kv_head * M * head_dim;
    const float* V_head = V_in + kv_head * M * head_dim;

    // Load K[kv_head] into smem
    for (int i = t; i < M * head_dim; i += blockDim.x) {
        K_s[i] = K_head[i];
    }
    __syncthreads();

    const float* Q_row = Q_in + q_head * M * head_dim;
    float* O_row = O_out + q_head * M * head_dim;

    for (int r = 0; r < 4; ++r) {
        int m = m0 + r;
        if (m >= M) continue;

        // Load Q[m] into shared (all 64 elements via 32 threads × 2)
        for (int d = t; d < head_dim; d += blockDim.x) {
            Q_s[d] = Q_row[m * head_dim + d];
        }
        __syncthreads();

        // Compute scores S[j] = Q[m] · K[j] for ALL j=0..M-1
        // 32 lanes, each with 2 Q elements. Process 1 K row at a time.
        // All 32 lanes contribute partials for the SAME K row j.
        // Then shuffle-reduce to get full dot, lane 0 stores to S_s.
        // 128 iterations of 2 MAC + 5 shuffles = manageable.
        for (int j = 0; j < M; ++j) {
            float partial = Q_s[lane * 2] * K_s[j * head_dim + lane * 2]
                          + Q_s[lane * 2 + 1] * K_s[j * head_dim + lane * 2 + 1];
            for (int o = 16; o > 0; o /= 2)
                partial += __shfl_down_sync(0xffffffff, partial, o);
            if (lane == 0) S_s[j] = partial * scale;
        }
        __syncthreads();

        // Online softmax in shared memory
        if (lane == 0) {
            float mx = S_s[0];
            #pragma unroll
            for (int j = 1; j < M; ++j) mx = fmaxf(mx, S_s[j]);
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < M; ++j) {
                S_s[j] = expf(S_s[j] - mx);
                sum += S_s[j];
            }
            #pragma unroll
            for (int j = 0; j < M; ++j) S_s[j] /= sum;
        }
        __syncthreads();

        // Accumulate O[m][d] = Σ_j P[m][j] * V[j][d]
        // Each lane: 2 output elements
        float o0 = 0.0f, o1 = 0.0f;
        if (lane * 2 < head_dim) {
            for (int j = 0; j < M; ++j) {
                float p = S_s[j];
                const float* Vj = V_head + j * head_dim;
                o0 += p * Vj[lane * 2];
                if (lane * 2 + 1 < head_dim) o1 += p * Vj[lane * 2 + 1];
            }
        }
        if (lane * 2 < head_dim) {
            O_row[m * head_dim + lane * 2] = o0;
            if (lane * 2 + 1 < head_dim) O_row[m * head_dim + lane * 2 + 1] = o1;
        }
        __syncthreads();
    }
}

} // anonymous namespace

// Public API
cudaError_t attention_fp4(
    float* output, const void* Q_fp4, const void* K_fp4, const void* V_fp4,
    const float* Q_scale, const float* K_scale, const float* V_scale,
    int batch_size, int seq_len, int num_heads, int head_dim,
    float scale, cudaStream_t stream) {

    if (!output || !Q_fp4 || !K_fp4 || !V_fp4)
        return cudaErrorInvalidValue;
    if (!Q_scale || !K_scale || !V_scale)
        return cudaErrorInvalidValue;

    int total_q = batch_size * seq_len * num_heads * head_dim;
    int num_kv_heads = num_heads;  // assume GQA ratio=1 for FP4 path
    int total_kv = batch_size * seq_len * num_kv_heads * head_dim;

    // Dequantize FP4 → FP32
    float* Q_f32 = nullptr;
    float* K_f32 = nullptr;
    float* V_f32 = nullptr;
    cudaError_t e;
    e = cudaMalloc(&Q_f32, total_q * sizeof(float));
    if (e != cudaSuccess) return e;
    e = cudaMalloc(&K_f32, total_kv * sizeof(float));
    if (e != cudaSuccess) { cudaFree(Q_f32); return e; }
    e = cudaMalloc(&V_f32, total_kv * sizeof(float));
    if (e != cudaSuccess) { cudaFree(Q_f32); cudaFree(K_f32); return e; }

    e = unpack_fp4(Q_f32, Q_fp4, Q_scale, total_q, stream);
    if (e != cudaSuccess) goto cleanup;
    e = unpack_fp4(K_f32, K_fp4, K_scale, total_kv, stream);
    if (e != cudaSuccess) goto cleanup;
    e = unpack_fp4(V_f32, V_fp4, V_scale, total_kv, stream);
    if (e != cudaSuccess) goto cleanup;

    // Run FP32 attention
    e = attention_prefill(output, Q_f32, K_f32, V_f32,
                          seq_len, head_dim, num_heads, num_kv_heads,
                          num_heads / num_kv_heads, scale, stream);

cleanup:
    cudaFree(Q_f32);
    cudaFree(K_f32);
    cudaFree(V_f32);
    return e;
}

// FP32 prefill attention (for use after GEMM outputs FP32 Q/K/V)
cudaError_t attention_prefill(
    float* output,
    const float* Q,  // (num_heads, M, head_dim)
    const float* K,   // (num_kv_heads, M, head_dim)
    const float* V,   // (num_kv_heads, M, head_dim)
    int M, int head_dim, int num_q_heads, int num_kv_heads,
    int num_q_per_group, float scale, cudaStream_t stream) {

    // grid: (num_q_heads, M/4) blocks, 32 threads per block (1 warp, 4 rows)
    // smem: K[M*head_dim] + Q[head_dim] + S[M]
    int blocks_x = num_q_heads;
    int blocks_y = (M + 3) / 4;
    dim3 grid = {static_cast<unsigned>(blocks_x), static_cast<unsigned>(blocks_y), 1};
    int threads = 32;
    int smem = M * head_dim * sizeof(float) + head_dim * sizeof(float) + M * sizeof(float);

    static bool attr_set = false;
    if (!attr_set) {
        cudaError_t e = cudaFuncSetAttribute(flash_attention_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) return e;
        attr_set = true;
    }

    flash_attention_kernel<<<grid, threads, smem, stream>>>(
        output, Q, K, V, M, head_dim, num_q_heads, num_kv_heads, num_q_per_group, scale);

    return cudaPeekAtLastError();
}

} // namespace kernels
} // namespace blackwell