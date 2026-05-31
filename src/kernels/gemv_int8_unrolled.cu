// src/kernels/gemv_int8_unrolled.cu — INT8 GEMV with loop unrolling
//
// Simpler approach: use #pragma unroll on the existing kernel structure.
// This lets the compiler optimize register usage and instruction scheduling.

#include <cuda_runtime.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {

// INT8 GEMV kernel with compiler-guided unrolling
__launch_bounds__(64, 2)
__global__ void gemv_int8_unrolled_kernel(
    float* __restrict__ y_out,
    const int8_t* __restrict__ x_int8,
    const float* __restrict__ x_scale,
    const int8_t* __restrict__ W_t_int8,
    const float* __restrict__ W_t_scale,
    int K, int N)
{
    constexpr int B = 16;
    int tid = threadIdx.x;
    int n_out = blockIdx.x * 64 + tid;
    if (n_out >= N) return;

    int num_K_blks = K / B;

    // Use multiple accumulators for better ILP
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    int kb = 0;
    
    // Process 4 blocks per iteration with separate accumulators
    for (; kb + 3 < num_K_blks; kb += 4) {
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
            const int8_t* w_ptr = &W_t_int8[n_out * K + (kb+1) * B];
            alignas(16) int8_t w_buf[B];
            alignas(16) int8_t x_buf[B];
            *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);
            *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_int8 + (kb+1) * B);
            
            const int* w32 = reinterpret_cast<const int*>(w_buf);
            const int* x32 = reinterpret_cast<const int*>(x_buf);
            int sumi = 0;
            sumi = __dp4a(w32[0], x32[0], sumi);
            sumi = __dp4a(w32[1], x32[1], sumi);
            sumi = __dp4a(w32[2], x32[2], sumi);
            sumi = __dp4a(w32[3], x32[3], sumi);
            acc1 += static_cast<float>(sumi) * W_t_scale[n_out * num_K_blks + kb+1] * x_scale[kb+1];
        }
        
        // Block 2
        {
            const int8_t* w_ptr = &W_t_int8[n_out * K + (kb+2) * B];
            alignas(16) int8_t w_buf[B];
            alignas(16) int8_t x_buf[B];
            *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);
            *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_int8 + (kb+2) * B);
            
            const int* w32 = reinterpret_cast<const int*>(w_buf);
            const int* x32 = reinterpret_cast<const int*>(x_buf);
            int sumi = 0;
            sumi = __dp4a(w32[0], x32[0], sumi);
            sumi = __dp4a(w32[1], x32[1], sumi);
            sumi = __dp4a(w32[2], x32[2], sumi);
            sumi = __dp4a(w32[3], x32[3], sumi);
            acc2 += static_cast<float>(sumi) * W_t_scale[n_out * num_K_blks + kb+2] * x_scale[kb+2];
        }
        
        // Block 3
        {
            const int8_t* w_ptr = &W_t_int8[n_out * K + (kb+3) * B];
            alignas(16) int8_t w_buf[B];
            alignas(16) int8_t x_buf[B];
            *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);
            *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_int8 + (kb+3) * B);
            
            const int* w32 = reinterpret_cast<const int*>(w_buf);
            const int* x32 = reinterpret_cast<const int*>(x_buf);
            int sumi = 0;
            sumi = __dp4a(w32[0], x32[0], sumi);
            sumi = __dp4a(w32[1], x32[1], sumi);
            sumi = __dp4a(w32[2], x32[2], sumi);
            sumi = __dp4a(w32[3], x32[3], sumi);
            acc3 += static_cast<float>(sumi) * W_t_scale[n_out * num_K_blks + kb+3] * x_scale[kb+3];
        }
    }

    // Handle remainder
    for (; kb < num_K_blks; ++kb) {
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

    y_out[n_out] = acc0 + acc1 + acc2 + acc3;
}

// Launch wrapper
cudaError_t gemv_int8_unrolled(
    float* y_out,
    const void* x_int8,
    const float* x_scale,
    const void* W_t_int8,
    const float* W_t_scale,
    int K, int N,
    cudaStream_t stream)
{
    dim3 grid((N + 63) / 64);
    dim3 block(64);
    gemv_int8_unrolled_kernel<<<grid, block, 0, stream>>>(
        y_out, static_cast<const int8_t*>(x_int8), x_scale,
        static_cast<const int8_t*>(W_t_int8), W_t_scale, K, N);
    return cudaGetLastError();
}

}  // namespace kernels
}  // namespace blackwell
