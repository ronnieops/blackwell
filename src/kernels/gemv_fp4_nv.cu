// src/kernels/gemv_fp4_nv.cu — NVF4 scalar GEMV for RTX 5060 Ti Blackwell SM_120
//
// Scalar GEMV path: FP4 E2M1 input, UE4M3 block scales, FP32 output.
// Weight format: W_t [N×K] FP4 E2M1 (transposed row-major).
// Scale format:  W_scale [N/16 × K/16] UE4M3 per block.
// Activation:    x_fp4 [K] FP4 E2M1, x_scale [K/16] UE4M3 per block.

#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace blackwell {
namespace kernels {
namespace {

constexpr int kFP4BlockSize = 16;

// ---------------------------------------------------------------------------
// UE4M3 → float (device)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float ue4m3_to_float(uint8_t v) {
    if (v == 0) return 0.0f;
    int exp = (v >> 3) & 0xF;
    int man = v & 0x7;
    if (exp == 0) return (man / 8.0f) * (1.0f / 64.0f);
    return (1.0f + man / 8.0f) * exp2f((float)(exp - 7));
}

// ---------------------------------------------------------------------------
// Scalar NVF4 GEMV kernel — 256 threads/block, 1 output per thread
// Optimized: vectorized W loads, pre-fused scale product
// ---------------------------------------------------------------------------
__launch_bounds__(256, 1)
__global__ void gemv_fp4_nv_kernel(
    float* __restrict__ y_out,
    const __nv_fp4_e2m1* __restrict__ x_fp4,
    const uint8_t* __restrict__ x_scale,
    const __nv_fp4_e2m1* __restrict__ W_t_fp4,
    const uint8_t* __restrict__ W_t_scale,
    int K, int N)
{
    constexpr int B = 16;
    int tid = threadIdx.x;
    int n_out = blockIdx.x * 256 + tid;
    if (n_out >= N) return;

    int n_blk = n_out / B;
    int num_K = K / B;
    const __nv_fp4_e2m1* w_row = &W_t_fp4[n_out * K];
    float acc = 0.0f;

    // Pre-load x block scales (only K/16 values, fits in registers)
    // x_scale is shared across all N outputs → broadcast via L1
    // w_scale is per N-block → each thread loads its own

    for (int kb = 0; kb < num_K; kb++) {
        // Convert UE4M3 scales to FP32 (once per K-block)
        float x_sc = ue4m3_to_float(x_scale[kb]);
        float w_sc = ue4m3_to_float(W_t_scale[n_blk * num_K + kb]);
        float prod_sc = x_sc * w_sc;

        // Load 16 FP4 values from W_t (vectorized)
        alignas(16) __nv_fp4_e2m1 w_buf[B];
        *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(&w_row[kb * B]);

        // Accumulate 16 elements
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < B; j++)
            sum += static_cast<float>(x_fp4[kb*B+j]) * static_cast<float>(w_buf[j]);
        acc += sum * prod_sc;
    }
    y_out[n_out] = acc;
}

// ===========================================================================
// Optimized NVF4 GEMV kernel — FP32 scales (pre-computed) + FP16 accumulator
// Uses float* scales instead of uint8_t* UE4M3, eliminating ue4m3_to_float().
// Inner loop uses __hfma (FP16 FMA) for higher throughput.
// ===========================================================================
__launch_bounds__(256, 1)
__global__ void gemv_fp4_nv_opt_kernel(
    float* __restrict__ y_out,
    const __nv_fp4_e2m1* __restrict__ x_fp4,
    const float* __restrict__ x_scale,
    const __nv_fp4_e2m1* __restrict__ W_t_fp4,
    const float* __restrict__ W_t_scale,
    int K, int N)
{
    constexpr int B = 16;
    int tid = threadIdx.x;
    int n_out = blockIdx.x * 256 + tid;
    if (n_out >= N) return;

    int n_blk = n_out / B;
    int num_K = K / B;
    const __nv_fp4_e2m1* w_row = &W_t_fp4[n_out * K];
    float acc = 0.0f;

    for (int kb = 0; kb < num_K; kb++) {
        // FP32 scales (pre-converted, no ue4m3_to_float call)
        float prod_sc = x_scale[kb] * W_t_scale[n_blk * num_K + kb];

        // Load 16 FP4 values from W_t (vectorized)
        alignas(16) __nv_fp4_e2m1 w_buf[B];
        *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(&w_row[kb * B]);

        // FP16 dot product via __hfma (2× throughput vs FP32 FMA)
        __half sum_h = __float2half(0.0f);
        #pragma unroll
        for (int j = 0; j < B; j++) {
            __half x_h = __float2half(static_cast<float>(x_fp4[kb*B+j]));
            __half w_h = __float2half(static_cast<float>(w_buf[j]));
            sum_h = __hfma(x_h, w_h, sum_h);
        }
        acc += __half2float(sum_h) * prod_sc;
    }
    y_out[n_out] = acc;
}

} // anonymous namespace

// ===========================================================================
// Public API
// ===========================================================================
cudaError_t gemv_fp4_nv(
    float* y_out, const void* x_fp4, const void* x_scale,
    const void* W_t_fp4, const void* W_t_scale,
    int in_features, int out_features, cudaStream_t stream)
{
    using Fp4 = __nv_fp4_e2m1;
    if (in_features % 16 != 0 || out_features % 16 != 0)
        return cudaErrorInvalidValue;

    int nb = (out_features + 255) / 256;
    gemv_fp4_nv_kernel<<<nb, 256, 0, stream>>>(
        y_out, (const Fp4*)x_fp4, (const uint8_t*)x_scale,
        (const Fp4*)W_t_fp4, (const uint8_t*)W_t_scale,
        in_features, out_features);
    return cudaPeekAtLastError();
}

cudaError_t gemv_fp4_nv_opt(
    float* y_out, const void* x_fp4, const void* x_scale,
    const void* W_t_fp4, const void* W_t_scale,
    int in_features, int out_features, cudaStream_t stream)
{
    using Fp4 = __nv_fp4_e2m1;
    if (in_features % 16 != 0 || out_features % 16 != 0)
        return cudaErrorInvalidValue;

    int nb = (out_features + 255) / 256;
    gemv_fp4_nv_opt_kernel<<<nb, 256, 0, stream>>>(
        y_out, (const Fp4*)x_fp4, (const float*)x_scale,
        (const Fp4*)W_t_fp4, (const float*)W_t_scale,
        in_features, out_features);
    return cudaPeekAtLastError();
}

} // kernels
} // blackwell
