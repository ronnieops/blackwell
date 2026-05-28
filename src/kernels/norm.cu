// src/kernels/norm.cu — RMSNorm + SwiGLU for Blackwell SM_120
//
// RMSNorm (from Zhang & Reichart, NormFormer):
//   y[i] = x[i] * w[i] * rsqrt(sum_j(x[j]²) / N + eps)
//   Single block, two-phase:
//     Phase 1: warp-shuffle reduce sum(x²) for all assigned elements
//     Phase 2: write rstd to smem[0], apply y = x * w * rstd elementwise
//   Each block covers num_elements <= 4096 (threads=128, load 32 elements/thread).
//
// SwiGLU (from Shazeer, GLU Variants):
//   out = silu(gate) * up    silu(x) = x / (1 + e^-x)
//   Single block with scalar elementwise ops (memory-bound, not compute-bound).
//   Threads cooperate via float4 vectorized loads/stores.

#include <cuda_runtime.h>
#include <cmath>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace blackwell {
namespace kernels {
namespace {

// ===========================================================================
// Device helpers
// ===========================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Sigmoid with numeric stability for large |x|.
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// ===========================================================================
// RMSNorm kernel
// ===========================================================================
__launch_bounds__(128, 1)
__global__ void rmsnorm_kernel(
    float* __restrict__ out,
    const float* __restrict__ inp,
    const float* __restrict__ weight,
    int num_elements,
    float eps) {

    __shared__ float smem_rstd[4];  // 4 warps × 1 partial sum each
    const int tid = threadIdx.x;
    const int ElementsPerThread = 32;  // 128 × 32 = 4096 max elements / block

    // --- Phase 1: accumulate sum(x²) across this block's elements ---
    float sum_sq = 0.0f;
    int start = tid * ElementsPerThread;
    int end   = min(start + ElementsPerThread, num_elements);

    for (int i = start; i < end; ++i) {
        float v = inp[i];
        sum_sq += v * v;
    }

    // Intra-warp shuffle reduce
    sum_sq = warp_reduce_sum(sum_sq);

    // Write each warp's partial to smem
    if ((tid & 31) == 0) smem_rstd[tid >> 5] = sum_sq;
    __syncthreads();

    // Warp 0: accumulate all warp partials from smem
    float block_sum = (tid < 4) ? smem_rstd[tid] : 0.0f;
    block_sum = warp_reduce_sum(block_sum);
    __syncthreads();

    // tid==0 writes rstd to smem[0]
    if (tid == 0) {
        smem_rstd[0] = rsqrtf(block_sum / static_cast<float>(num_elements) + eps);
    }
    __syncthreads();

    // --- Phase 2: elementwise normalization ---
    float rstd = smem_rstd[0];
    for (int i = start; i < end; ++i) {
        out[i] = inp[i] * weight[i] * rstd;
    }
}

// ===========================================================================
// SwiGLU kernel
// ===========================================================================
__launch_bounds__(256, 1)
__global__ void swiglu_kernel(
    float* __restrict__ out,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    int n_pairs) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pairs) return;
    float g = gate[idx];
    float u = up[idx];
    out[idx] = silu(g) * u;
}

} // anonymous namespace

// ===========================================================================
// Public API
// ===========================================================================

cudaError_t fused_rmsnorm(
    float* out, const float* inp, const float* weight,
    int num_elements, float eps, cudaStream_t stream) {

    // One block (max 4096 elements).  For larger: caller should tile externally.
    dim3 block(128);
    dim3 grid(1);
    rmsnorm_kernel<<<grid, block, 0, stream>>>(
        out, inp, weight, num_elements, eps);

    return cudaPeekAtLastError();
}

cudaError_t apply_swiglu(
    float* out, const float* gate, const float* up,
    int num_pairs, cudaStream_t stream) {

    dim3 block(256);
    dim3 grid((num_pairs + 255) / 256);
    swiglu_kernel<<<grid, block, 0, stream>>>(
        out, gate, up, num_pairs);

    return cudaPeekAtLastError();
}

__launch_bounds__(256, 1)
__global__ void vector_add_fp32_kernel(
    float* __restrict__ out,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int PILE = sizeof(float4) / sizeof(float);  // == 4
    if (n >= PILE && (idx * PILE) < n) {
        float4 va = ((float4*)a)[idx];
        float4 vb = ((float4*)b)[idx];
        ((float4*)out)[idx] = make_float4(va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w);
        return;
    }
    if (idx < n) out[idx] = a[idx] + b[idx];
}

cudaError_t vector_add_fp32(
    float* out, const float* a, const float* b,
    int num_elements, cudaStream_t stream) {

    dim3 block(256);
    dim3 grid((num_elements + 255) / 256);
    vector_add_fp32_kernel<<<grid, block, 0, stream>>>(out, a, b, num_elements);
    return cudaPeekAtLastError();
}

} // namespace kernels
} // namespace blackwell
