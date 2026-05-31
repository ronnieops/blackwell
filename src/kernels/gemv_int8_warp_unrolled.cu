// src/kernels/gemv_int8_warp_unrolled.cu — Warp-cooperative INT8 GEMV with unrolling
//
// Optimization: unroll the stride-32 loop in gemv_int8_warp_kernel.
// Each thread processes multiple K-blocks per iteration for better ILP.
//
// Build:
//   CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake --build build --parallel

#include <cuda_runtime.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {

// Warp-cooperative INT8 GEMV with 4× loop unrolling
// 1 warp (32 threads) per output row, processes 4 K-blocks per iteration
__launch_bounds__(32, 8)
__global__ void gemv_int8_warp_unrolled_kernel(
    float* __restrict__ y_out,
    const int8_t* __restrict__ x_int8,
    const float* __restrict__ x_scale,
    const int8_t* __restrict__ W_t_int8,
    const float* __restrict__ W_t_scale,
    int K, int N)
{
    constexpr int B = 16;
    constexpr int UNROLL = 4;
    int n_out = blockIdx.x;
    int tid = threadIdx.x;

    int num_K_blks = K / B;

    // Multiple accumulators for ILP
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    // Unrolled stride-32 loop: process 4 blocks per iteration
    int kb = tid;
    for (; kb + 3 * 32 < num_K_blks; kb += UNROLL * 32) {
        // Block 0
        {
            const int8_t* w_ptr = &W_t_int8[n_out * K + kb * B];
            alignas(16) int8_t w_buf[B];
            alignas(16) int8_t x_buf[B];
            *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);
            *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_int8 + kb * B);
            
            const int* w32 = reinterpret_cast<const int*>(w_buf);
            const int* x32 = reinterpret_cast<const int*>(x_buf);
            int sumi = 0;
            sumi = __dp4a(w32[0], x32[0], sumi);
            sumi = __dp4a(w32[1], x32[1], sumi);
            sumi = __dp4a(w32[2], x32[2], sumi);
            sumi = __dp4a(w32[3], x32[3], sumi);
            acc0 += static_cast<float>(sumi) * W_t_scale[n_out * num_K_blks + kb] * x_scale[kb];
        }
        
        // Block 1
        {
            const int8_t* w_ptr = &W_t_int8[n_out * K + (kb + 32) * B];
            alignas(16) int8_t w_buf[B];
            alignas(16) int8_t x_buf[B];
            *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);
            *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_int8 + (kb + 32) * B);
            
            const int* w32 = reinterpret_cast<const int*>(w_buf);
            const int* x32 = reinterpret_cast<const int*>(x_buf);
            int sumi = 0;
            sumi = __dp4a(w32[0], x32[0], sumi);
            sumi = __dp4a(w32[1], x32[1], sumi);
            sumi = __dp4a(w32[2], x32[2], sumi);
            sumi = __dp4a(w32[3], x32[3], sumi);
            acc1 += static_cast<float>(sumi) * W_t_scale[n_out * num_K_blks + kb + 32] * x_scale[kb + 32];
        }
        
        // Block 2
        {
            const int8_t* w_ptr = &W_t_int8[n_out * K + (kb + 64) * B];
            alignas(16) int8_t w_buf[B];
            alignas(16) int8_t x_buf[B];
            *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);
            *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_int8 + (kb + 64) * B);
            
            const int* w32 = reinterpret_cast<const int*>(w_buf);
            const int* x32 = reinterpret_cast<const int*>(x_buf);
            int sumi = 0;
            sumi = __dp4a(w32[0], x32[0], sumi);
            sumi = __dp4a(w32[1], x32[1], sumi);
            sumi = __dp4a(w32[2], x32[2], sumi);
            sumi = __dp4a(w32[3], x32[3], sumi);
            acc2 += static_cast<float>(sumi) * W_t_scale[n_out * num_K_blks + kb + 64] * x_scale[kb + 64];
        }
        
        // Block 3
        {
            const int8_t* w_ptr = &W_t_int8[n_out * K + (kb + 96) * B];
            alignas(16) int8_t w_buf[B];
            alignas(16) int8_t x_buf[B];
            *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);
            *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_int8 + (kb + 96) * B);
            
            const int* w32 = reinterpret_cast<const int*>(w_buf);
            const int* x32 = reinterpret_cast<const int*>(x_buf);
            int sumi = 0;
            sumi = __dp4a(w32[0], x32[0], sumi);
            sumi = __dp4a(w32[1], x32[1], sumi);
            sumi = __dp4a(w32[2], x32[2], sumi);
            sumi = __dp4a(w32[3], x32[3], sumi);
            acc3 += static_cast<float>(sumi) * W_t_scale[n_out * num_K_blks + kb + 96] * x_scale[kb + 96];
        }
    }

    // Handle remainder
    for (; kb < num_K_blks; kb += 32) {
        const int8_t* w_ptr = &W_t_int8[n_out * K + kb * B];
        alignas(16) int8_t w_buf[B];
        alignas(16) int8_t x_buf[B];
        *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);
        *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_int8 + kb * B);
        
        const int* w32 = reinterpret_cast<const int*>(w_buf);
        const int* x32 = reinterpret_cast<const int*>(x_buf);
        int sumi = 0;
        sumi = __dp4a(w32[0], x32[0], sumi);
        sumi = __dp4a(w32[1], x32[1], sumi);
        sumi = __dp4a(w32[2], x32[2], sumi);
        sumi = __dp4a(w32[3], x32[3], sumi);
        acc0 += static_cast<float>(sumi) * W_t_scale[n_out * num_K_blks + kb] * x_scale[kb];
    }

    // Combine accumulators
    float acc = acc0 + acc1 + acc2 + acc3;

    // Warp shuffle reduction
    acc += __shfl_xor_sync(0xffffffff, acc, 16);
    acc += __shfl_xor_sync(0xffffffff, acc, 8);
    acc += __shfl_xor_sync(0xffffffff, acc, 4);
    acc += __shfl_xor_sync(0xffffffff, acc, 2);
    acc += __shfl_xor_sync(0xffffffff, acc, 1);

    if (tid == 0) y_out[n_out] = acc;
}

// Launch wrapper
cudaError_t gemv_int8_warp_unrolled(
    float* y_out,
    const void* x_int8,
    const float* x_scale,
    const void* W_t_int8,
    const float* W_t_scale,
    int K, int N,
    cudaStream_t stream)
{
    gemv_int8_warp_unrolled_kernel<<<N, 32, 0, stream>>>(
        y_out, static_cast<const int8_t*>(x_int8), x_scale,
        static_cast<const int8_t*>(W_t_int8), W_t_scale, K, N);
    return cudaGetLastError();
}

}  // namespace kernels
}  // namespace blackwell
