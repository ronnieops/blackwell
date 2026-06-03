// gemv_fp32_int5_asym.cu — FP32 activation × asymmetric INT5 weight GEMV
//
// 5-bit per-block-16 asymmetric quantization: scale=(max-min)/31, zero=round(-min/scale).
// 16 values packed into 10 bytes (80 bits). Weight PSNR ~29 dB vs BF16.
// Scale+zero format: [N][2*K/16] — even=scale, odd=zero (as float).
//
// Grid: N blocks, 32 threads/block. 1 warp/row, stride-32 K-blocks, shuffle reduce.

#include <cuda_runtime.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

// Unpack 16 × 5-bit values from 10 bytes (w0=32-bit, w1=32-bit, w2=16-bit)
__device__ __forceinline__ void unpack_int5_block(
    int* v, uint32_t w0, uint32_t w1, uint16_t w2)
{
    v[0]  = w0 & 0x1F;
    v[1]  = (w0 >> 5) & 0x1F;
    v[2]  = (w0 >> 10) & 0x1F;
    v[3]  = (w0 >> 15) & 0x1F;
    v[4]  = (w0 >> 20) & 0x1F;
    v[5]  = (w0 >> 25) & 0x1F;
    v[6]  = ((w0 >> 30) & 0x03) | ((w1 & 0x07) << 2);
    v[7]  = (w1 >> 3) & 0x1F;
    v[8]  = (w1 >> 8) & 0x1F;
    v[9]  = (w1 >> 13) & 0x1F;
    v[10] = (w1 >> 18) & 0x1F;
    v[11] = (w1 >> 23) & 0x1F;
    v[12] = ((w1 >> 28) & 0x0F) | ((w2 & 0x01) << 4);
    v[13] = (w2 >> 1) & 0x1F;
    v[14] = (w2 >> 6) & 0x1F;
    v[15] = (w2 >> 11) & 0x1F;
}

__launch_bounds__(32, 8)
__global__ void gemv_fp32_int5_asym_kernel(
    float* __restrict__ y_out,
    const float* __restrict__ x_fp32,
    const uint8_t* __restrict__ W_packed,   // [N][K*5/8]
    const float* __restrict__ W_sc_zero,    // [N][2*K/16]
    int K, int N)
{
    constexpr int B = 16, PB = 10;  // 16 elements per block, 10 packed bytes
    int n_out = blockIdx.x;
    int tid = threadIdx.x;

    int num_K_blks = K / B;
    int row_bytes = num_K_blks * PB;

    float acc = 0.0f;

    for (int kb = tid; kb < num_K_blks; kb += 32) {
        // Load 16 FP32 activations
        int x_off = kb * B;
        float xf[B];
        *reinterpret_cast<float4*>(&xf[0])  = reinterpret_cast<const float4*>(&x_fp32[x_off])[0];
        *reinterpret_cast<float4*>(&xf[4])  = reinterpret_cast<const float4*>(&x_fp32[x_off])[1];
        *reinterpret_cast<float4*>(&xf[8])  = reinterpret_cast<const float4*>(&x_fp32[x_off])[2];
        *reinterpret_cast<float4*>(&xf[12]) = reinterpret_cast<const float4*>(&x_fp32[x_off])[3];

        // Load 10 packed bytes (80 bits = 16 × 5-bit values)
        const uint8_t* w_ptr = &W_packed[(size_t)n_out * row_bytes + kb * PB];
        uint32_t w0 = w_ptr[0] | (w_ptr[1] << 8) | (w_ptr[2] << 16) | (w_ptr[3] << 24);
        uint32_t w1 = w_ptr[4] | (w_ptr[5] << 8) | (w_ptr[6] << 16) | (w_ptr[7] << 24);
        uint16_t w2 = w_ptr[8] | (w_ptr[9] << 8);

        // Unpack 16 × 5-bit values
        int v[B];
        unpack_int5_block(v, w0, w1, w2);

        // Load scale + zero for this block
        float w_sc = W_sc_zero[(size_t)n_out * 2 * num_K_blks + kb * 2];
        float w_zero_f = W_sc_zero[(size_t)n_out * 2 * num_K_blks + kb * 2 + 1];
        int w_zero = __float2int_rn(w_zero_f);

        // Dequant and dot product
        float sum_b = 0.0f;
        #pragma unroll
        for (int i = 0; i < B; ++i) {
            float w = static_cast<float>(v[i] - w_zero) * w_sc;
            sum_b += w * xf[i];
        }
        acc += sum_b;
    }

    // Warp shuffle reduction
    acc += __shfl_xor_sync(0xffffffff, acc, 16);
    acc += __shfl_xor_sync(0xffffffff, acc, 8);
    acc += __shfl_xor_sync(0xffffffff, acc, 4);
    acc += __shfl_xor_sync(0xffffffff, acc, 2);
    acc += __shfl_xor_sync(0xffffffff, acc, 1);

    if (tid == 0) y_out[n_out] = acc;
}

} // anonymous namespace

cudaError_t gemv_fp32_int5_asym(
    float*          y_out,
    const float*    x_fp32,
    const uint8_t*  W_packed,
    const float*    W_sc_zero,
    int             K,
    int             N,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || N % 32 != 0)
        return cudaErrorInvalidValue;

    dim3 grid(N);
    gemv_fp32_int5_asym_kernel<<<grid, 32, 0, stream>>>(
        y_out, x_fp32, W_packed, W_sc_zero, K, N);
    return cudaPeekAtLastError();
}

} // namespace kernels
} // namespace blackwell
