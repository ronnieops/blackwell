// gemv_fp32.cu — Simple FP32 GEMV kernel for high-precision inference
// y = W * x where W is [N x K] row-major FP32, x is [K] FP32, y is [N] FP32

#include <cuda_runtime.h>
#include <cstdio>

namespace blackwell {
namespace kernels {

// Simple row-parallel GEMV: one thread per output row
__global__ void gemv_fp32_kernel(float* __restrict__ y, const float* __restrict__ W,
                                  const float* __restrict__ x, int K, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const float* row = W + (size_t)n * K;
    float sum = 0.0f;
    for (int k = 0; k < K; k += 4) {
        float4 v = *(float4*)(row + k);
        float4 ix = *(float4*)(x + k);
        sum += v.x * ix.x + v.y * ix.y + v.z * ix.z + v.w * ix.w;
    }
    // Handle remainder
    for (int k = (K & ~3); k < K; k++) {
        sum += row[k] * x[k];
    }
    y[n] = sum;
}

// Warp-cooperative GEMV: 32 threads per row, warp-level reduction
__global__ void gemv_fp32_warp_kernel(float* __restrict__ y, const float* __restrict__ W,
                                       const float* __restrict__ x, int K, int N) {
    int n = blockIdx.x;
    if (n >= N) return;

    const float* row = W + (size_t)n * K;
    int tid = threadIdx.x;

    // Cooperative load in float4
    float sum = 0.0f;
    for (int k = tid * 4; k < K; k += blockDim.x * 4) {
        float4 v = *(float4*)(row + k);
        float4 ix = *(float4*)(x + k);
        sum += v.x * ix.x + v.y * ix.y + v.z * ix.z + v.w * ix.w;
    }

    // Warp reduction
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, off);
    }

    if (tid == 0) {
        y[n] = sum;
    }
}

// Host function
cudaError_t gemv_fp32_launch(
    float*          y_out,
    const float*    W_fp32,
    const float*    x_fp32,
    int             K,
    int             N,
    cudaStream_t    stream) {

    if (N <= 0 || K <= 0) return cudaErrorInvalidValue;

    // Use warp-cooperative kernel for better efficiency
    dim3 grid(N);
    dim3 block(32);
    gemv_fp32_warp_kernel<<<grid, block, 0, stream>>>(y_out, W_fp32, x_fp32, K, N);

    return cudaGetLastError();
}

} // namespace kernels
} // namespace blackwell