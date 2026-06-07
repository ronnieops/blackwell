// src/kernels/gemv_int8_per_row.cu — Per-row scaled INT8 GEMV
//
// Weight format: INT8 block-16 (same data layout as gemv_int8_warp).
// Scale format:  W_scale [N] FP32 — one scale per output row.
// Activation:    FP32 [K].
//
// 1 warp per output row. Accumulates raw int8 dot product,
// then multiplies by per-row scale at the end.

#include <cuda_runtime.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

__launch_bounds__(32, 4)
__global__ void gemv_int8_per_row_kernel(
    float* __restrict__       y_out,
    const float* __restrict__ x_fp32,
    const int8_t* __restrict__ W_t_int8,
    const float* __restrict__ W_t_scale,
    int K, int N)
{
    constexpr int B = 16;
    int n_out = blockIdx.x;
    int tid = threadIdx.x;
    if (n_out >= N) return;

    int num_K_blks = K / B;
    float row_sc = W_t_scale[n_out];
    float acc = 0.0f;

    for (int kb = tid; kb < num_K_blks; kb += 32) {
        int x_off = kb * B;

        // Load 16 FP32 activation values
        float4 v0 = reinterpret_cast<const float4*>(&x_fp32[x_off])[0];
        float4 v1 = reinterpret_cast<const float4*>(&x_fp32[x_off])[1];
        float4 v2 = reinterpret_cast<const float4*>(&x_fp32[x_off])[2];
        float4 v3 = reinterpret_cast<const float4*>(&x_fp32[x_off])[3];

        // Load 16 INT8 weight values (4 × int)
        const int8_t* w_ptr = &W_t_int8[n_out * K + kb * B];
        int w0 = reinterpret_cast<const int*>(w_ptr)[0];
        int w1 = reinterpret_cast<const int*>(w_ptr)[1];
        int w2 = reinterpret_cast<const int*>(w_ptr)[2];
        int w3 = reinterpret_cast<const int*>(w_ptr)[3];

        #define SE(i) (float)(int8_t)((i) & 0xFF)
        float sum_b = 0.0f;
        sum_b += SE(w0 >>  0) * v0.x;  sum_b += SE(w0 >>  8) * v0.y;
        sum_b += SE(w0 >> 16) * v0.z;  sum_b += SE(w0 >> 24) * v0.w;
        sum_b += SE(w1 >>  0) * v1.x;  sum_b += SE(w1 >>  8) * v1.y;
        sum_b += SE(w1 >> 16) * v1.z;  sum_b += SE(w1 >> 24) * v1.w;
        sum_b += SE(w2 >>  0) * v2.x;  sum_b += SE(w2 >>  8) * v2.y;
        sum_b += SE(w2 >> 16) * v2.z;  sum_b += SE(w2 >> 24) * v2.w;
        sum_b += SE(w3 >>  0) * v3.x;  sum_b += SE(w3 >>  8) * v3.y;
        sum_b += SE(w3 >> 16) * v3.z;  sum_b += SE(w3 >> 24) * v3.w;
        #undef SE
        acc += sum_b;
    }

    // Warp shuffle reduction
    acc += __shfl_xor_sync(0xffffffff, acc, 16);
    acc += __shfl_xor_sync(0xffffffff, acc, 8);
    acc += __shfl_xor_sync(0xffffffff, acc, 4);
    acc += __shfl_xor_sync(0xffffffff, acc, 2);
    acc += __shfl_xor_sync(0xffffffff, acc, 1);

    if (tid == 0) y_out[n_out] = acc * row_sc;
}

} // anonymous namespace

cudaError_t gemv_int8_per_row(
    float*        y_out,
    const float*  x_fp32,
    const void*   W_t_int8,
    const float*  W_t_scale,
    int           K,
    int           N,
    cudaStream_t  stream)
{
    if (N <= 0 || K <= 0) return cudaSuccess;
    gemv_int8_per_row_kernel<<<dim3(N), dim3(32), 0, stream>>>(
        y_out, x_fp32,
        reinterpret_cast<const int8_t*>(W_t_int8),
        W_t_scale, K, N);
    return cudaGetLastError();
}

} // namespace kernels
} // namespace blackwell
