// src/kernels/persistent_attn_block.cu — Persistent QKV GEMV kernel
//
// Single grid launch: 16 blocks (one per Q head), 128 threads/block, 40KB smem.
// Each block stays resident across all num_layers and processes its head.
//
// Fuses QKV GEMV + KV cache update in one launch.
// Attention, Wo, norm handled by separate proven kernels (4 launches/layer).
//
// smem (40 KB):
//   s_Q[2048]   — Q projection output (16 heads × 128)
//   s_K[1024]   — K projection (8 KV heads × 128)
//   s_V[1024]   — V projection
//   s_scr[2056] — softmax scores (2048) + norm scratch (8)
//
// Build: CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake --build build --parallel

#include <cuda_runtime.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {

constexpr int HIDDEN = 2048;
constexpr int KV = 1024;
constexpr int HEAD_DIM = 128;
constexpr int NUM_Q_HEADS = 16;
constexpr int NUM_KV_HEADS = 8;
constexpr int MAX_SEQ = 2048;
constexpr int B = 16;
constexpr int NUM_K_BLKS = HIDDEN / B;  // 128

// Gang-of-4 GEMV: all 128 threads cooperatively compute 4 output rows.
// row_base: starting row within the head
// row: absolute row index = head_base + row_base + r
__device__ __forceinline__ void gemv_gang4(
    const int8_t* __restrict__ W, const float* __restrict__ W_sc,
    const int8_t* __restrict__ x_i8, const float* __restrict__ x_sc,
    int head_base, int row_base,
    float* __restrict__ out_base,
    int lane, int warp_id)
{
    // 4 warps, each computes 1 row. 4 rows total.
    for (int r = 0; r < 4; ++r) {
        int row = head_base + row_base + r;
        float acc = 0.0f;
        for (int kb = lane; kb < NUM_K_BLKS; kb += 32) {
            alignas(16) int8_t w_buf[B];
            const int8_t* w_ptr = W + row * HIDDEN + kb * B;
            #pragma unroll
            for (int i = 0; i < B; ++i) w_buf[i] = w_ptr[i];
            alignas(16) int8_t x_buf[B];
            #pragma unroll
            for (int i = 0; i < B; ++i) x_buf[i] = x_i8[kb * B + i];
            float sc = W_sc[row * NUM_K_BLKS + kb] * x_sc[kb];
            const int* w32 = reinterpret_cast<int*>(w_buf);
            const int* x32 = reinterpret_cast<int*>(x_buf);
            int s = __dp4a(w32[0], x32[0], 0);
            s = __dp4a(w32[1], x32[1], s);
            s = __dp4a(w32[2], x32[2], s);
            s = __dp4a(w32[3], x32[3], s);
            acc += (float)s * sc;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_xor_sync(0xffffffff, acc, off);
        if (lane == 0) out_base[row] = acc;
    }
}

__launch_bounds__(128, 1)
__global__ void persistent_qkv_kernel(
    const int8_t* W_q, const float* W_q_sc,
    const int8_t* W_k, const float* W_k_sc,
    const int8_t* W_v, const float* W_v_sc,
    float* k_cache, float* v_cache,
    const int* seq_pos_ptr,
    const int8_t** layer_x_int8,
    const float** layer_x_sc,
    float* q_out, float* k_out, float* v_out,
    int num_layers)
{
    extern __shared__ char smem_base[];
    float* smem = reinterpret_cast<float*>(smem_base);

    // smem: Q[2048] + K[1024] + V[1024] + scratch[2056] = 40 KB
    float* s_Q   = smem;
    float* s_K   = s_Q + 2048;
    float* s_V   = s_K + 1024;
    float* s_scr = s_V + 1024;

    int q_head = blockIdx.x;  // 0..15
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;

    for (int layer = 0; layer < num_layers; ++layer) {
        const int8_t* x_i8 = layer_x_int8[layer];
        const float* x_sc = layer_x_sc[layer];

        // ===== Q GEMV → s_Q[q_head*HD .. q_head*HD+HD-1] =====
        // 32 gangs of 4 rows (4 warps × 1 row/gang = 4 rows per gang).
        // Each lane computes the same 4 rows but different K blocks.
        // Gang g computes rows: row_base = g, g+4, g+8, g+12...
        for (int gang = 0; gang < 8; ++gang) {
            int row_base = warp_id * 1;  // each warp does 1 row per gang
            gemv_gang4(W_q, W_q_sc, x_i8, x_sc, q_head * HEAD_DIM, row_base, s_Q, lane, warp_id);
            __syncthreads();
        }
        // After: s_Q[q_head*HD .. q_head*HD+HD-1] filled

        // ===== K GEMV → s_K[kv_head*HD .. kv_head*HD+HD-1] =====
        for (int gang = 0; gang < 8; ++gang) {
            int row_base = warp_id * 1;
            gemv_gang4(W_k, W_k_sc, x_i8, x_sc, q_head * HEAD_DIM, row_base, s_K, lane, warp_id);
            __syncthreads();
        }
        // After: s_K[q_head*HD .. q_head*HD+HD-1] filled

        // ===== V GEMV → s_V[kv_head*HD .. kv_head*HD+HD-1] =====
        for (int gang = 0; gang < 8; ++gang) {
            int row_base = warp_id * 1;
            gemv_gang4(W_v, W_v_sc, x_i8, x_sc, q_head * HEAD_DIM, row_base, s_V, lane, warp_id);
            __syncthreads();
        }
        // After: s_V[q_head*HD .. q_head*HD+HD-1] filled

        // ===== Update KV cache =====
        int seq_pos = *seq_pos_ptr;
        int kv_base = q_head * MAX_SEQ * HEAD_DIM;
        int pos_off = seq_pos * HEAD_DIM;
        // Write all 128 elements via float4: lanes 0..31 each write 1 float4
        if (lane < 32) {
            float4* k4_base = reinterpret_cast<float4*>(k_cache + kv_base + pos_off);
            float4* v4_base = reinterpret_cast<float4*>(v_cache + kv_base + pos_off);
            k4_base[lane] = reinterpret_cast<float4*>(s_K)[lane];
            v4_base[lane] = reinterpret_cast<float4*>(s_V)[lane];
        }
        __syncthreads();

        // ===== Copy Q/K/V to output buffers =====
        // q_out: [num_layers * Q_HEADS * HEAD_DIM] = [layer * 2048 + q_head*128 .. +128]
        // Each lane writes 4 floats (float4)
        int q_off = layer * NUM_Q_HEADS * HEAD_DIM + q_head * HEAD_DIM;
        int kv_off = layer * NUM_Q_HEADS * HEAD_DIM + q_head * HEAD_DIM;
        if (lane < 32) {
            reinterpret_cast<float4*>(q_out + q_off)[lane] =
                reinterpret_cast<float4*>(s_Q)[lane];
            reinterpret_cast<float4*>(k_out + kv_off)[lane] =
                reinterpret_cast<float4*>(s_K)[lane];
            reinterpret_cast<float4*>(v_out + kv_off)[lane] =
                reinterpret_cast<float4*>(s_V)[lane];
        }
    }  // end layer loop
}

// Launch wrapper
cudaError_t persistent_qkv_gemv(
    const void* W_q, const float* W_q_sc,
    const void* W_k, const float* W_k_sc,
    const void* W_v, const float* W_v_sc,
    void* k_cache, void* v_cache,
    const int* seq_pos_ptr,
    const int8_t** layer_x_int8,
    const float** layer_x_sc,
    float* q_out, float* k_out, float* v_out,
    int num_layers,
    cudaStream_t stream)
{
    size_t smem = 40 * 1024;
    int grid = NUM_Q_HEADS;
    persistent_qkv_kernel<<<grid, 128, smem, stream>>>(
        static_cast<const int8_t*>(W_q), W_q_sc,
        static_cast<const int8_t*>(W_k), W_k_sc,
        static_cast<const int8_t*>(W_v), W_v_sc,
        static_cast<float*>(k_cache), static_cast<float*>(v_cache),
        seq_pos_ptr, layer_x_int8, layer_x_sc,
        q_out, k_out, v_out, num_layers);
    return cudaPeekAtLastError();
}

}  // namespace kernels
}  // namespace blackwell