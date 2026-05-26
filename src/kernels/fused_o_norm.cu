// src/kernels/fused_o_norm.cu — Fused O-projection + RMSNorm + FP4 pack
//
// Replaces 3 separate kernels per decode layer:
//   gemv_fp4(W_o) → fused_rmsnorm → pack_fp4(x)
//
// Two-kernel approach:
//   Kernel 1: gemv_fp4 writes FP32 output (same as existing gemv)
//   Kernel 2: rmsnorm + pack (reads FP32, writes FP4)
//
// This eliminates 1 kernel launch AND the intermediate pack step.
// The 2nd kernel is a simple single-block kernel (hidden_dim <= 2048)
// that does RMSNorm reduction + FP4 quantization in one pass.
//
// API: fused_rmsnorm_pack() — the 2nd kernel. Caller still does gemv_fp4
// separately, then calls this instead of fused_rmsnorm + pack_fp4.

#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace blackwell {
namespace kernels {
namespace {

constexpr int kNormPackThreads = 256;
constexpr int kElemsPerThread = 8; // 256 * 8 = 2048

__device__ __forceinline__ float warp_reduce_sum_f(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ---------------------------------------------------------------------------
// Single-block kernel: RMSNorm + FP4 pack
// Input: FP32 projection output (from gemv_fp4)
// Output: FP4 packed x + per-block scales
// ---------------------------------------------------------------------------
__launch_bounds__(kNormPackThreads, 1)
__global__ void rmsnorm_pack_kernel(
    __nv_fp4_e2m1* __restrict__ x_out_fp4,
    float* __restrict__ x_out_scale,
    const float* __restrict__ proj,
    const float* __restrict__ weight,
    int N, float eps)
{
    constexpr int B = 16;
    constexpr int NE = kElemsPerThread;
    int tid = threadIdx.x;
    int num_blks = (N + B - 1) / B;

    // Phase 1: load proj values, compute sum_sq
    float vals[NE];
    float sum_sq = 0.0f;

    #pragma unroll
    for (int e = 0; e < NE; ++e) {
        int idx = tid + e * kNormPackThreads;
        if (idx < N) {
            vals[e] = proj[idx];
            sum_sq += vals[e] * vals[e];
        } else {
            vals[e] = 0.0f;
        }
    }

    // Warp reduce sum_sq
    sum_sq = warp_reduce_sum_f(sum_sq);

    // Cross-warp reduce (256/32 = 8 warps)
    __shared__ float warp_sums[8];
    if ((tid & 31) == 0) warp_sums[tid >> 5] = sum_sq;
    __syncthreads();

    float block_sum = (tid < 8) ? warp_sums[tid] : 0.0f;
    block_sum = warp_reduce_sum_f(block_sum);

    __shared__ float s_rstd;
    if (tid == 0) {
        s_rstd = rsqrtf(block_sum / static_cast<float>(N) + eps);
    }
    __syncthreads();
    float rstd = s_rstd;

    // Phase 2: normalize + pack to FP4
    #pragma unroll
    for (int e = 0; e < NE; ++e) {
        int idx = tid + e * kNormPackThreads;
        if (idx < N) {
            float normed = vals[e] * weight[idx] * rstd;

            // FP4 block-scale: need max(abs) across 16 elements in this block.
            // For tid layout: idx = tid + e*256.
            // FP4 block index = idx / 16.
            // Elements in same FP4 block: idx = blk*16 + j, j=0..15
            //   For fixed e: idx = tid + e*256, so tid values for same FP4 block
            //   are 16 consecutive tids. These are in the same warp (256/32=8 warps).
            //   16 consecutive tids → half-warp. lane_in_blk = tid % 16.

            float abs_val = fabsf(normed);
            int lane_in_blk = tid % B;

            // Half-warp max reduction
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
                float scale = fmaxf(abs_val / 3.0f, 1e-9f);
                x_out_scale[idx / B] = scale;
            }
            __syncwarp();

            float scale = x_out_scale[idx / B];
            float sv = fdividef(normed, scale);
            sv = fmaxf(-3.0f, fminf(3.0f, sv));
            x_out_fp4[idx] = __nv_fp4_e2m1(sv);
        }
    }
}

} // anonymous namespace

// ===========================================================================
// Public API
// ===========================================================================

cudaError_t fused_rmsnorm_pack(
    void*           x_out_fp4,
    float*          x_out_scale,
    const float*    proj,
    const float*    weight,
    int             N,
    float           eps,
    cudaStream_t    stream)
{
    using Fp4 = __nv_fp4_e2m1;
    if (N % 16 != 0) return cudaErrorInvalidValue;
    if (N > kNormPackThreads * kElemsPerThread) return cudaErrorInvalidValue;

    rmsnorm_pack_kernel<<<dim3(1), dim3(kNormPackThreads), 0, stream>>>(
        static_cast<Fp4*>(x_out_fp4), x_out_scale,
        proj, weight, N, eps);

    return cudaPeekAtLastError();
}

// Keep fused_o_norm_pack as a convenience wrapper (2-kernel sequence)
cudaError_t fused_o_norm_pack(
    void*           x_out_fp4,
    float*          x_out_scale,
    float*          /* scratch1 unused */,
    int*            /* scratch2 unused */,
    float*          /* scratch3 unused */,
    const void*     attn_fp4,
    const float*    attn_scale,
    const void*     W_o_fp4,
    const float*    W_o_scale,
    const float*    rmsnorm_weight,
    int             K,
    int             N,
    float           eps,
    cudaStream_t    stream)
{
    // Allocate temp FP32 buffer for GEMV output
    float* d_proj;
    cudaError_t e = cudaMalloc(&d_proj, N * sizeof(float));
    if (e != cudaSuccess) return e;

    // Kernel 1: GEMV
    e = gemv_fp4(d_proj, attn_fp4, attn_scale, W_o_fp4, W_o_scale, K, N, stream);
    if (e != cudaSuccess) { cudaFree(d_proj); return e; }

    // Kernel 2: RMSNorm + pack
    e = fused_rmsnorm_pack(x_out_fp4, x_out_scale, d_proj, rmsnorm_weight, N, eps, stream);
    cudaFree(d_proj);
    return e;
}

} // namespace kernels
} // namespace blackwell
