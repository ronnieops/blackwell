// fused_residual_norm_int4_asym.cu — Fused residual + RMSNorm + ASYMMETRIC INT4 quant
//
// Same as fused_residual_norm_int4 but uses per-block min/max + zero point
// instead of symmetric absmax/7. Output scale format: [2 * N/16] floats
// where even=scale, odd=zero (as float).
//
// For each 16-element block:
//   scale = (max - min) / 15.0f
//   zero  = round(-min / scale)  clipped to [0..15]
//   nib   = round(x / scale) + zero  clipped to [0..15]
//
// Dequant: val = (nib - zero) * scale

#include <cuda_runtime.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

constexpr int kFusedThreads = 512;
constexpr int kFusedREPT = 8;
constexpr int kBlockSize = 16;

__device__ __forceinline__ float warp_reduce_sum_f(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max_f(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float warp_reduce_min_f(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Store scale + zero pair to output array
__device__ __forceinline__ void store_scale_zero(float* out, int blk_idx, float scale, int zero) {
    out[blk_idx * 2 + 0] = scale;
    out[blk_idx * 2 + 1] = (float)zero;
}

__launch_bounds__(kFusedThreads, 1)
__global__ void fused_residual_norm_int4_asym_kernel(
    uint8_t* __restrict__ x_out,       // [N/2] packed INT4 nibbles
    float*   __restrict__ x_sc_zero,   // [2*N/16] scale,zero pairs
    float*   __restrict__ proj,         // in/out: FP32 projection (modified in-place)
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    int N, float eps)
{
    int tid = threadIdx.x;

    // Phase 1: Residual add + sum_sq
    float vals[kFusedREPT];
    float sum_sq = 0.0f;

    #pragma unroll
    for (int r = 0; r < kFusedREPT; ++r) {
        int idx = tid + r * kFusedThreads;
        if (idx < N) {
            float v = proj[idx] + residual[idx];
            proj[idx] = v;
            vals[r] = v;
            sum_sq += v * v;
        } else {
            vals[r] = 0.0f;
        }
    }

    sum_sq = warp_reduce_sum_f(sum_sq);

    __shared__ float warp_sums[16];
    if ((tid & 31) == 0) warp_sums[tid >> 5] = sum_sq;
    __syncthreads();

    float block_sum = (tid < 16) ? warp_sums[tid] : 0.0f;
    block_sum = warp_reduce_sum_f(block_sum);

    __shared__ float s_rstd;
    if (tid == 0) s_rstd = rsqrtf(block_sum / (float)N + eps);
    __syncthreads();
    float rstd = s_rstd;

    // Phase 2: Normalize + per-block min/max (for asymmetric scale)
    float blk_min = 1e38f, blk_max = -1e38f;

    #pragma unroll
    for (int r = 0; r < kFusedREPT; ++r) {
        int idx = tid + r * kFusedThreads;
        if (idx < N) {
            float normed = vals[r] * weight[idx] * rstd;
            vals[r] = normed;
            blk_min = fminf(blk_min, normed);
            blk_max = fmaxf(blk_max, normed);
        }
    }

    // Reduce min/max across 16-lane block
    int blk_ofs = tid % kBlockSize;
    blk_min = warp_reduce_min_f(blk_min);
    blk_max = warp_reduce_max_f(blk_max);

    // Compute and store scale + zero (lane 0 of each block)
    if (blk_ofs == 0) {
        float scale = (blk_max - blk_min) / 15.0f;
        if (scale < 1e-9f) scale = 1e-9f;
        int zero = (int)roundf(-blk_min / scale);
        zero = max(0, min(15, zero));
        store_scale_zero(x_sc_zero, tid / kBlockSize, scale, zero);
    }
    __syncthreads();

    // Phase 3: Quantize with asymmetric scale+zero
    #pragma unroll
    for (int r = 0; r < kFusedREPT; ++r) {
        int idx = tid + r * kFusedThreads;
        if (idx < N) {
            int blk = idx / kBlockSize;
            float scale = x_sc_zero[blk * 2];
            float zf = x_sc_zero[blk * 2 + 1];
            int zero = (int)zf;
            float v = vals[r] / scale;
            int q = (int)roundf(v) + zero;
            q = max(0, min(15, q));
            uint8_t nib = static_cast<uint8_t>(q);

            int byte_addr = idx / 2;
            int nibble_pos = idx & 1;
            if (nibble_pos == 0) {
                uint8_t prev = x_out[byte_addr];
                x_out[byte_addr] = (prev & 0xF0) | nib;
            } else {
                uint8_t prev = x_out[byte_addr];
                x_out[byte_addr] = (prev & 0x0F) | (nib << 4);
            }
        }
    }
}

// FP32 output variant — also writes FP32 normalized values
__launch_bounds__(kFusedThreads, 1)
__global__ void fused_residual_norm_int4_asym_fp32out_kernel(
    uint8_t* __restrict__ x_out,
    float*   __restrict__ x_sc_zero,
    float*   __restrict__ proj_out,      // [N] FP32 normalized (can be nullptr)
    const float* __restrict__ proj_in,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    int N, float eps)
{
    int tid = threadIdx.x;

    // Phase 1: Residual add + sum_sq
    float vals[kFusedREPT];
    float sum_sq = 0.0f;

    #pragma unroll
    for (int r = 0; r < kFusedREPT; ++r) {
        int idx = tid + r * kFusedThreads;
        if (idx < N) {
            float v = proj_in[idx] + residual[idx];
            vals[r] = v;
            sum_sq += v * v;
        } else {
            vals[r] = 0.0f;
        }
    }

    sum_sq = warp_reduce_sum_f(sum_sq);

    __shared__ float warp_sums[16];
    if ((tid & 31) == 0) warp_sums[tid >> 5] = sum_sq;
    __syncthreads();

    float block_sum = (tid < 16) ? warp_sums[tid] : 0.0f;
    block_sum = warp_reduce_sum_f(block_sum);

    __shared__ float s_rstd;
    if (tid == 0) s_rstd = rsqrtf(block_sum / (float)N + eps);
    __syncthreads();
    float rstd = s_rstd;

    // Phase 2: Normalize + per-block min/max
    float blk_min = 1e38f, blk_max = -1e38f;

    #pragma unroll
    for (int r = 0; r < kFusedREPT; ++r) {
        int idx = tid + r * kFusedThreads;
        if (idx < N) {
            float normed = vals[r] * weight[idx] * rstd;
            vals[r] = normed;
            if (proj_out) proj_out[idx] = normed;
            blk_min = fminf(blk_min, normed);
            blk_max = fmaxf(blk_max, normed);
        }
    }

    blk_min = warp_reduce_min_f(blk_min);
    blk_max = warp_reduce_max_f(blk_max);

    if ((tid % kBlockSize) == 0) {
        float scale = (blk_max - blk_min) / 15.0f;
        if (scale < 1e-9f) scale = 1e-9f;
        int zero = (int)roundf(-blk_min / scale);
        zero = max(0, min(15, zero));
        store_scale_zero(x_sc_zero, tid / kBlockSize, scale, zero);
    }
    __syncthreads();

    // Phase 3: Quantize
    #pragma unroll
    for (int r = 0; r < kFusedREPT; ++r) {
        int idx = tid + r * kFusedThreads;
        if (idx < N) {
            int blk = idx / kBlockSize;
            float scale = x_sc_zero[blk * 2];
            float zf = x_sc_zero[blk * 2 + 1];
            int zero = (int)zf;
            float v = vals[r] / scale;
            int q = (int)roundf(v) + zero;
            q = max(0, min(15, q));
            uint8_t nib = static_cast<uint8_t>(q);

            int byte_addr = idx / 2;
            int nibble_pos = idx & 1;
            if (nibble_pos == 0) {
                uint8_t prev = x_out[byte_addr];
                x_out[byte_addr] = (prev & 0xF0) | nib;
            } else {
                uint8_t prev = x_out[byte_addr];
                x_out[byte_addr] = (prev & 0x0F) | (nib << 4);
            }
        }
    }
}

} // anonymous namespace

cudaError_t fused_residual_norm_int4_asym(
    void*           x_out,
    float*          x_sc_zero,
    float*          proj,
    const float*    residual,
    const float*    norm_w,
    int             N,
    float           eps,
    cudaStream_t    stream)
{
    return fused_residual_norm_int4_asym_fp32out(
        x_out, x_sc_zero, nullptr, proj, residual, norm_w, N, eps, stream);
}

cudaError_t fused_residual_norm_int4_asym_fp32out(
    void*           x_out,
    float*          x_sc_zero,
    float*          proj_out_fp32,
    const float*    proj_in,
    const float*    residual,
    const float*    norm_w,
    int             N,
    float           eps,
    cudaStream_t    stream)
{
    if (N % kBlockSize != 0 || N % kFusedThreads != 0)
        return cudaErrorInvalidValue;

    fused_residual_norm_int4_asym_fp32out_kernel<<<1, kFusedThreads, 0, stream>>>(
        static_cast<uint8_t*>(x_out), x_sc_zero, proj_out_fp32,
        proj_in, residual, norm_w, N, eps);

    return cudaPeekAtLastError();
}

} // namespace kernels
} // namespace blackwell