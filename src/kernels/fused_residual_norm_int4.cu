// src/kernels/fused_residual_norm_int4.cu — Fused residual add + RMSNorm + INT4 quant
//
// Combines vector_add_fp32 + fused_rmsnorm + quantize_int4 into single kernel.
// Saves 2 kernel launches per call (3→1 per layer).
//
// Input: proj[N] FP32 (output of projection) + residual[N] FP32 (input x)
// Output: x_out[N/2] packed INT4 (nibbles) + x_sc[N/16] FP32 scales
//
// Build: CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake --build build --parallel

#include <cuda_runtime.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

constexpr int kFusedThreads = 256;
constexpr int kFusedREPT = 8;    // elements per thread (covers H=2048 in 256 threads)
constexpr int kBlockSize = 16;   // quantization block size (matches INT4 format)

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

// Fused residual add + RMSNorm + INT4 quant (symmetric, block=16)
//
// Phase 1: Load proj + residual, write sum → proj, compute sum_sq
// Phase 2: Warp-reduce sum_sq → block rstd
// Phase 3: Normalize, compute per-block absmax, quantize to packed INT4 nibbles
//
// x_out: [N/2] bytes packed INT4 (2 vals/byte, lower=even, upper=odd)
// x_sc:  [N/16] FP32 scales (absmax/7 per block)
__launch_bounds__(kFusedThreads, 1)
__global__ void fused_residual_norm_int4_kernel(
    uint8_t* __restrict__ x_out,
    float*   __restrict__ x_sc,
    float*   __restrict__ proj,       // in/out: FP32 projection output
    const float* __restrict__ residual,
    const float* __restrict__ weight, // RMSNorm weight
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
            proj[idx] = v;   // store back for downstream (or just use local)
            vals[r] = v;
            sum_sq += v * v;
        } else {
            vals[r] = 0.0f;
        }
    }

    // Warp-reduce sum_sq across 8 warps
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

    // Phase 2: Normalize + per-block absmax reduction
    // Each block (16 consecutive threads, tid%16=0..15) computes one absmax
    float absmax = 0.0f;

    #pragma unroll
    for (int r = 0; r < kFusedREPT; ++r) {
        int idx = tid + r * kFusedThreads;
        if (idx < N) {
            float normed = vals[r] * weight[idx] * rstd;
            vals[r] = normed;  // update in-place
            absmax = fmaxf(absmax, fabsf(normed));
        }
    }

    // Warp-reduce absmax across 16 lanes per block
    int blk_ofs = tid % kBlockSize;
    absmax = warp_reduce_max_f(absmax);

    // Write scale for this block (lane 0 of each block)
    if (blk_ofs == 0) {
        float sc = (absmax > 1e-10f) ? (absmax / 7.0f) : (1.0f / 7.0f);
        x_sc[tid / kBlockSize] = sc;
    }
    __syncthreads();

    // Phase 3: Quantize to packed INT4 nibbles
    #pragma unroll
    for (int r = 0; r < kFusedREPT; ++r) {
        int idx = tid + r * kFusedThreads;
        if (idx < N) {
            float sc = x_sc[idx / kBlockSize];
            float v = vals[r] / sc;
            int q = (int)roundf(v);
            q = max(-8, min(7, q));
            uint8_t nib = static_cast<uint8_t>(q + 8);  // 0..15

            // Packed: byte addr = idx/2, nibble position = idx%2
            int byte_addr = idx / 2;
            int nibble_pos = idx & 1;  // 0=lower, 1=upper

            if (nibble_pos == 0) {
                // Lower nibble: clear lower 4 bits, set nib
                uint8_t prev = x_out[byte_addr];
                x_out[byte_addr] = (prev & 0xF0) | nib;
            } else {
                // Upper nibble: clear upper 4 bits, set nib
                uint8_t prev = x_out[byte_addr];
                x_out[byte_addr] = (prev & 0x0F) | (nib << 4);
            }
        }
    }
}

}  // anonymous namespace

cudaError_t fused_residual_norm_int4(
    void*           x_out,          // [N/2] uint8_t, packed INT4
    float*          x_out_sc,       // [N/16] FP32 scales
    float*          proj,           // in/out: FP32 projection (modified in-place)
    const float*    residual,       // input residual (FP32)
    const float*    norm_w,         // RMSNorm weight (FP32)
    int             N,
    float           eps,
    cudaStream_t    stream)
{
    return fused_residual_norm_int4_fp32out(x_out, x_out_sc, nullptr, proj, residual, norm_w, N, eps, stream);
}

// ── FP32 output variant for attention/MLP residual ────────────────────────────
// Same as above but also writes FP32 normalized values to proj_out (for next layer's attn scores).
// Saves 3 kernel launches (vector_add + rmsnorm + quantize_int4 → 1).
//
// x_out: [N/2] uint8_t, packed INT4
// x_sc:  [N/16] FP32 scales
// proj_out: [N] FP32 (normalized, for next layer's d_x32 / attention input)
__launch_bounds__(kFusedThreads, 1)
__global__ void fused_residual_norm_int4_fp32out_kernel(
    uint8_t* __restrict__ x_out,
    float*   __restrict__ x_sc,
    float*   __restrict__ proj_out,
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

    __shared__ float warp_sums[8];
    if ((tid & 31) == 0) warp_sums[tid >> 5] = sum_sq;
    __syncthreads();

    float block_sum = (tid < 8) ? warp_sums[tid] : 0.0f;
    block_sum = warp_reduce_sum_f(block_sum);

    __shared__ float s_rstd;
    if (tid == 0) s_rstd = rsqrtf(block_sum / (float)N + eps);
    __syncthreads();
    float rstd = s_rstd;

    // Phase 2: Normalize + per-block absmax
    float absmax = 0.0f;

    #pragma unroll
    for (int r = 0; r < kFusedREPT; ++r) {
        int idx = tid + r * kFusedThreads;
        if (idx < N) {
            float normed = vals[r] * weight[idx] * rstd;
            vals[r] = normed;
            if (proj_out) proj_out[idx] = normed;
            absmax = fmaxf(absmax, fabsf(normed));
        }
    }

    absmax = warp_reduce_max_f(absmax);

    if ((tid % kBlockSize) == 0) {
        float sc = (absmax > 1e-10f) ? (absmax / 7.0f) : (1.0f / 7.0f);
        x_sc[tid / kBlockSize] = sc;
    }
    __syncthreads();

    // Phase 3: Quantize
    #pragma unroll
    for (int r = 0; r < kFusedREPT; ++r) {
        int idx = tid + r * kFusedThreads;
        if (idx < N) {
            float sc = x_sc[idx / kBlockSize];
            float v = vals[r] / sc;
            int q = (int)roundf(v);
            q = max(-8, min(7, q));
            uint8_t nib = static_cast<uint8_t>(q + 8);
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

cudaError_t fused_residual_norm_int4_fp32out(
    void*           x_out,          // [N/2] uint8_t, packed INT4
    float*          x_out_sc,       // [N/16] FP32 scales
    float*          proj_out_fp32,  // [N] FP32 normalized output (can be nullptr)
    const float*    proj_in,        // input projection (not modified)
    const float*    residual,       // input residual (FP32)
    const float*    norm_w,         // RMSNorm weight (FP32)
    int             N,
    float           eps,
    cudaStream_t    stream)
{
    if (N % kBlockSize != 0) return cudaErrorInvalidValue;
    if (N % kFusedThreads != 0) return cudaErrorInvalidValue;

    fused_residual_norm_int4_fp32out_kernel<<<1, kFusedThreads, 0, stream>>>(
        static_cast<uint8_t*>(x_out),
        x_out_sc,
        proj_out_fp32,
        proj_in,
        residual,
        norm_w,
        N, eps);

    return cudaPeekAtLastError();
}

}  // namespace kernels
}  // namespace blackwell