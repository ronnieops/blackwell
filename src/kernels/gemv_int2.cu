// src/kernels/gemv_int2.cu — INT2 block-scaled GEMV
//
// INT2 symmetric: values {-2,-1,0,1}, block-16 FP32 scales.
// Packed: 4 weights per byte (2 bits each, offset-binary: val+2 → 0-3).
//
// Weight format: W [N][K/4] packed bytes, W_scale [N][K/16] FP32.
// Activation:    x [M][K/2] INT4 packed, x_scale [M][K/16] FP32.
// Output:        y [M][N] FP32.

#include <cuda_runtime.h>
#include <cstdio>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

constexpr int kINT2Block = 32; // threads per block (1 warp)

// INT2 byte to 4 floats: nibble → (val-2) as float
__device__ __forceinline__ void int2_byte_to_floats(uint8_t b, float& f0, float& f1, float& f2, float& f3) {
    f0 = static_cast<float>(static_cast<int>((b      ) & 0x3) - 2);
    f1 = static_cast<float>(static_cast<int>((b >> 2 ) & 0x3) - 2);
    f2 = static_cast<float>(static_cast<int>((b >> 4 ) & 0x3) - 2);
    f3 = static_cast<float>(static_cast<int>((b >> 6 ) & 0x3) - 2);
}

// INT4 byte to 2 floats (from existing gemv_int8.cu)
__device__ __forceinline__ void int4_byte_to_floats(uint8_t b, float& f0, float& f1) {
    int lo = (b & 0x0F) - 8;
    int hi = ((b >> 4) & 0x0F) - 8;
    f0 = static_cast<float>(lo);
    f1 = static_cast<float>(hi);
}

template<int M>
__launch_bounds__(kINT2Block, 1)
__global__ void gemv_int2_batched_kernel(
    float* __restrict__ y_out,
    const uint8_t* __restrict__ x_packed,   // [M][K/2] INT4
    const float* __restrict__ x_scale,      // [M][K/16]
    const uint8_t* __restrict__ W_packed,   // [N][K/4] INT2
    const float* __restrict__ W_scale,      // [N][K/16]
    int K, int N)
{
    constexpr int B = 16;    // quant block size
    constexpr int WB = 4;    // INT2: 4 weights per byte
    constexpr int XB = 2;    // INT4: 2 weights per byte
    int n_out = blockIdx.x;
    if (n_out >= N) return;
    int tid = threadIdx.x;

    int num_K_blks = K / B;

    float acc[M];
    #pragma unroll
    for (int mi = 0; mi < M; ++mi) acc[mi] = 0.0f;

    // Each thread iterates over K-blocks with stride 32
    for (int kb = tid; kb < num_K_blks; kb += 32) {
        // Load 4 bytes = 16 INT2 weight values
        const uint8_t* w_ptr = &W_packed[(size_t)n_out * (K / 4) + kb * (B / WB)];
        uint32_t w4 = *reinterpret_cast<const uint32_t*>(w_ptr);
        float w_sc = W_scale[(size_t)n_out * num_K_blks + kb];

        #pragma unroll
        for (int mi = 0; mi < M; ++mi) {
            const uint8_t* x_ptr = &x_packed[(size_t)mi * (K / 2) + kb * (B / XB)];
            uint2 x_packed_val = *reinterpret_cast<const uint2*>(x_ptr);
            float x_sc = x_scale[(size_t)mi * num_K_blks + kb];
            float prod_scale = w_sc * x_sc;

            const uint8_t* wb = reinterpret_cast<const uint8_t*>(&w4);
            const uint8_t* xb = reinterpret_cast<const uint8_t*>(&x_packed_val);

            float sum_f = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                // 4 INT2 values from weight byte
                float w0, w1, w2, w3;
                int2_byte_to_floats(wb[j], w0, w1, w2, w3);
                // 2 INT4 values from activation byte
                float x0, x1;
                int4_byte_to_floats(xb[j], x0, x1);
                sum_f += w0 * x0 + w1 * x1;
                // Next pair
                int4_byte_to_floats(xb[j+4], x0, x1);
                sum_f += w2 * x0 + w3 * x1;
            }
            acc[mi] += sum_f * prod_scale;
        }
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int mi = 0; mi < M; ++mi) {
        acc[mi] += __shfl_xor_sync(0xffffffff, acc[mi], 16);
        acc[mi] += __shfl_xor_sync(0xffffffff, acc[mi], 8);
        acc[mi] += __shfl_xor_sync(0xffffffff, acc[mi], 4);
        acc[mi] += __shfl_xor_sync(0xffffffff, acc[mi], 2);
        acc[mi] += __shfl_xor_sync(0xffffffff, acc[mi], 1);
    }

    if (tid == 0) {
        #pragma unroll
        for (int mi = 0; mi < M; ++mi) {
            y_out[(size_t)mi * N + n_out] = acc[mi];
        }
    }
}

} // anonymous namespace

cudaError_t gemv_int2_batched(
    float*          y_out,
    const uint8_t*  x_packed,
    const float*    x_scale,
    const uint8_t*  W_packed,
    const float*    W_scale,
    int             K,
    int             N,
    int             M,
    cudaStream_t    stream)
{
    dim3 grid(N);
    dim3 block(kINT2Block);

    switch(M) {
        case 1: gemv_int2_batched_kernel<1><<<grid, block, 0, stream>>>(y_out, x_packed, x_scale, W_packed, W_scale, K, N); break;
        case 2: gemv_int2_batched_kernel<2><<<grid, block, 0, stream>>>(y_out, x_packed, x_scale, W_packed, W_scale, K, N); break;
        case 4: gemv_int2_batched_kernel<4><<<grid, block, 0, stream>>>(y_out, x_packed, x_scale, W_packed, W_scale, K, N); break;
        case 8: gemv_int2_batched_kernel<8><<<grid, block, 0, stream>>>(y_out, x_packed, x_scale, W_packed, W_scale, K, N); break;
        default:
            fprintf(stderr, "gemv_int2_batched: unsupported M=%d\n", M);
            return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

} // namespace kernels
} // namespace blackwell
