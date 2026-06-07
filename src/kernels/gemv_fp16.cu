// gemv_fp16.cu — FP16 × INT8-activation GEMV kernel for mixed-precision inference.
// Uses FP16 weights with INT8 quantized activations (block-16 scales).
// For early layers where weight precision matters more.

#include "blackwell/kernels.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cuda_fp16.h>

namespace blackwell { namespace kernels {

// ── FP16 GEMV: y[row] = sum_i(deq(x[i]) * (float)W_fp16[row*K+i]) ───
// W_fp16: FP16 weight matrix [N, K] row-major
// x_i8: INT8 quantized activations [K]
// x_scales: per-block scales [K/16]
// y: float output [N]
__global__ void gemv_fp16_warp(
    float* __restrict__ y,
    const __half* __restrict__ W_fp16,
    const int8_t* __restrict__ x_i8,
    const float* __restrict__ x_scales,
    int K, int N) {
    
    int row = blockIdx.x;
    if (row >= N) return;
    
    const int warp_id = threadIdx.x / 32;
    const int warp_lane = threadIdx.x % 32;
    const int n_warps = blockDim.x / 32;
    
    float sum = 0.0f;
    
    // Each thread processes elements in its warp×lane pattern
    // Use vectorized loads: process 4 values at a time via float4
    #pragma unroll 4
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        // Dequantize activation: int8 → fp32 × scale
        int blk = i / 16;
        float xv = (float)x_i8[i] * x_scales[blk];
        
        // Load FP16 weight and convert to FP32
        float wv = __half2float(W_fp16[(size_t)row * K + i]);
        sum += xv * wv;
    }
    
    // Warp shuffle reduce (within each warp, 32 lanes)
    for (int o = 16; o > 0; o >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, o);
    
    // Store partial sums to shared memory
    __shared__ float smem[32];  // up to 32 warps (1024 threads)
    if (warp_lane == 0)
        smem[warp_id] = sum;
    __syncthreads();
    
    // Final reduction in warp 0
    if (warp_id == 0) {
        float v = (warp_lane < n_warps) ? smem[warp_lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, o);
        if (warp_lane == 0)
            y[row] = v;
    }
}

// Host launch wrapper
cudaError_t gemv_fp16_warp_launch(
    float* y,
    const void* W_fp16,
    const int8_t* x_i8,
    const float* x_scales,
    int K, int N,
    cudaStream_t st) {
    
    int threads = 128;  // 4 warps
    int blocks = N;
    gemv_fp16_warp<<<blocks, threads, 0, st>>>(
        y,
        (const __half*)W_fp16,
        x_i8, x_scales, K, N);
    return cudaGetLastError();
}

}} // namespace blackwell::kernels
