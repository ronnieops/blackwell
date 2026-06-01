// src/kernels/fused_residual_norm.cu — Fused residual add + RMSNorm + INT8 quant
//
// Combines vector_add_fp32 + rmsnorm_quant_int8 into single kernel.
// Saves 1 kernel launch per call (2 per layer).
//
// Build: CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake --build build --parallel

#include <cuda_runtime.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

constexpr int kFusedThreads = 256;
constexpr int kFusedEPT = 16;

__device__ __forceinline__ float warp_reduce_sum_f(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Fused residual add + RMSNorm + INT8 quant
// Identical data layout to rmsnorm_quant_int8_kernel
__launch_bounds__(kFusedThreads, 1)
__global__ void fused_residual_norm_kernel(
    int8_t* __restrict__ x_out_i8,
    float* __restrict__ x_out_scale,
    float* __restrict__ proj,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    int N, float eps)
{
    constexpr int B = 16;
    constexpr int NE = kFusedEPT;
    int tid = threadIdx.x;

    // Phase 1: Residual add + load + sum_sq
    float vals[NE];
    float sum_sq = 0.0f;

    #pragma unroll
    for (int e = 0; e < NE; ++e) {
        int idx = tid + e * kFusedThreads;
        if (idx < N) {
            float v = proj[idx] + residual[idx];
            proj[idx] = v;
            vals[e] = v;
            sum_sq += v * v;
        } else {
            vals[e] = 0.0f;
        }
    }

    sum_sq = warp_reduce_sum_f(sum_sq);

    __shared__ float warp_sums[8];
    if ((tid & 31) == 0) warp_sums[tid >> 5] = sum_sq;
    __syncthreads();

    float block_sum = (tid < 8) ? warp_sums[tid] : 0.0f;
    block_sum = warp_reduce_sum_f(block_sum);

    __shared__ float s_rstd;
    if (tid == 0) s_rstd = rsqrtf(block_sum / (float)N + eps);
    __syncthreads();
    float rstd = s_rstd;

    // Phase 2: normalize + INT8 quant with per-block scales
    // Uses same strided access + warp-shuffle max as rmsnorm_quant_int8_kernel
    #pragma unroll
    for (int e = 0; e < NE; ++e) {
        int idx = tid + e * kFusedThreads;
        if (idx < N) {
            float normed = vals[e] * weight[idx] * rstd;

            // Per-block (16 elements) max reduction via warp shuffles
            float abs_val = fabsf(normed);
            int lane_in_blk = tid % B;

            float d;
            d = __shfl_down_sync(0xffffffff, abs_val, 8);
            if (lane_in_blk < 8) abs_val = fmaxf(abs_val, d);
            d = __shfl_down_sync(0xffffffff, abs_val, 4);
            if (lane_in_blk < 4) abs_val = fmaxf(abs_val, d);
            d = __shfl_down_sync(0xffffffff, abs_val, 2);
            if (lane_in_blk < 2) abs_val = fmaxf(abs_val, d);
            d = __shfl_down_sync(0xffffffff, abs_val, 1);
            if (lane_in_blk == 0) abs_val = fmaxf(abs_val, d);

            if (lane_in_blk == 0) {
                float scale = fmaxf(abs_val / 127.0f, 1e-9f);
                x_out_scale[idx / B] = scale;
            }
            __syncwarp();

            // Quantize to INT8 — matches rmsnorm_quant_int8_kernel exactly
            float sc = x_out_scale[idx / B];
            float v = fdividef(normed, sc);
            v = fminf(127.0f, fmaxf(-127.0f, roundf(v)));
            x_out_i8[idx] = static_cast<int8_t>(static_cast<int>(v));
        }
    }
}

}  // anonymous namespace

cudaError_t fused_residual_norm(
    int8_t* x_out_i8,
    float* x_out_scale,
    float* proj,
    const float* residual,
    const float* norm_w,
    int N, float eps,
    cudaStream_t stream)
{
    if (N % 16 != 0) return cudaErrorInvalidValue;
    
    fused_residual_norm_kernel<<<dim3(1), dim3(kFusedThreads), 0, stream>>>(
        x_out_i8, x_out_scale, proj, residual, norm_w, N, eps);
    
    return cudaPeekAtLastError();
}

}  // namespace kernels
}  // namespace blackwell
