// src/kernels/fused_unpack_quant.cu — Fused unpack FP4 → quantize to INT8
//
// Combines: unpack_fp4 → pack_int8 into single kernel.
// Saves 1 kernel launch per call (2 per layer: attention + MLP residual).
//
// Note: Current benchmark uses x_residual as FP4, converts to INT8 via:
//   unpack_fp4(fp32) → pack_int8(i8)
// This fusion eliminates the intermediate FP32 buffer writes.
//
// Build: CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake --build build --parallel

#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

// Fused unpack FP4 → INT8 quant kernel
// Each thread handles 16 elements (INT8 block quantization)
// Reads: FP4 E2M1 [N] + FP4 scale [1] + INT8 target scale [N/16]
// Writes: INT8 [N] + INT8 scales [N/16]
__launch_bounds__(256, 1)
__global__ void fused_unpack_fp4_quant_kernel(
    int8_t* __restrict__ out_i8,
    float* __restrict__ out_scale,
    const __nv_fp4_e2m1* __restrict__ in_fp4,
    const float* __restrict__ fp4_scale,
    const float* __restrict__ int8_scale,
    int N)
{
    constexpr int B = 16;
    int blk = blockIdx.x;
    int num_blks = N / B;
    
    if (blk >= num_blks) return;
    
    int lane = threadIdx.x;
    int tid_in_blk = threadIdx.x;
    
    // Compute per-thread local max in this block
    // Each block of 256 threads handles 16 elements (B=16)
    // Threads 0..15 are active, threads 16..255 are idle (set to neutral)
    float local_max = 0.0f;
    if (lane < B) {
        float fp4_sc = fp4_scale[0];
        #pragma unroll 4
        for (int i = 0; i < B / 4; ++i) {
            int idx = blk * B + i;
            float v = static_cast<float>(in_fp4[idx]) * fp4_sc;
            local_max = fmaxf(local_max, fabsf(v));
        }
    }
    
    // Warp shuffle reduce: 16 threads reduce to lane 0
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, off));
    }
    
    // Block 0 lane 0: finalize scale
    if (tid_in_blk == 0) {
        float sc = fmaxf(local_max / 127.0f, 1e-9f);
        out_scale[blk] = sc;
    }
    
    // Barrier ensures scale is written before quantize reads it
    __syncthreads();
    
    float target_sc = out_scale[blk];
    
    if (lane < B) {
        float fp4_sc = fp4_scale[0];
        #pragma unroll 4
        for (int i = 0; i < B / 4; ++i) {
            int idx = blk * B + i;
            float v = static_cast<float>(in_fp4[idx]) * fp4_sc;
            v = v / target_sc;
            v = fminf(127.0f, fmaxf(-127.0f, roundf(v)));
            out_i8[idx] = static_cast<int8_t>(static_cast<int>(v));
        }
    }
}

}  // anonymous namespace

cudaError_t fused_unpack_fp4_quant(
    int8_t* out_i8,
    float* out_scale,
    const void* in_fp4,
    const float* fp4_scale,
    const float* int8_scale,
    int N,
    cudaStream_t stream)
{
    if (N % 16 != 0) return cudaErrorInvalidValue;
    
    int num_blks = N / 16;
    int threads = 256;
    int grid = (num_blks + threads - 1) / threads;
    
    fused_unpack_fp4_quant_kernel<<<grid, threads, 0, stream>>>(
        out_i8, out_scale,
        static_cast<const __nv_fp4_e2m1*>(in_fp4),
        fp4_scale, int8_scale, N);
    
    return cudaPeekAtLastError();
}

}  // namespace kernels
}  // namespace blackwell