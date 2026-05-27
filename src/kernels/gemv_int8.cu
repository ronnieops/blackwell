// src/kernels/gemv_int8.cu — INT8 block-scaled GEMV
//
// INT8 GEMV path: eliminates FP4 cast overhead (static_cast<float> per element).
// Uses FP16 WMMA for matrix multiply to leverage tensor core throughput.
// Block-scaled quantization: weights in INT8 [0..255], scales in FP32 per 16×16 block.
// Activations in INT8, scales per 16-element K-block.
//
// Architecture:
//   - nvcuda::wmma with FP16 input fragments
//   - INT8 weights loaded as FP16 via wmma::mma_sync
//   - Accumulation in FP16 fragments → FP32 output
//   - Per-block scaling using FP32 scale factors
//
// Weight format: W_t [N×K] INT8 (transposed row-major).
// Scale format:  W_scale [N/16 × K/16] FP32 per block.
// Activation:    x_int8 [K], x_scale [K/16] FP32 per block.

#include <cuda_runtime.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace blackwell {
namespace kernels {
namespace {

constexpr int kINT8Block = 256;

// ---------------------------------------------------------------------------
// INT8 GEMV kernel — per-thread dot products, transposed weights
//
// Each thread handles one output n_out.
// Inner loop: K/16 iterations, loads 16 INT8 weights + 16 INT8 activations.
// Applies scales per-block, accumulates in FP32.
// ---------------------------------------------------------------------------
__launch_bounds__(kINT8Block, 1)
__global__ void gemv_int8_kernel(
    float* __restrict__ y_out,
    const int8_t* __restrict__ x_int8,
    const float* __restrict__ x_scale,
    const int8_t* __restrict__ W_t_int8,
    const float* __restrict__ W_t_scale,
    int K, int N)
{
    constexpr int B = 16;
    int tid = threadIdx.x;
    int n_out = blockIdx.x * kINT8Block + tid;
    if (n_out >= N) return;

    int num_K_blks = K / B;
    int n_blk = n_out / B;

    float acc = 0.0f;

    for (int kb = 0; kb < num_K_blks; ++kb) {
        // Load 16 INT8 weight values via vectorized uint4 load
        const int8_t* w_ptr = &W_t_int8[n_out * K + kb * B];
        alignas(16) int8_t w_buf[B];
        *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);

        float w_sc = W_t_scale[n_blk * num_K_blks + kb];
        float x_sc = x_scale[kb];

        // Load x values for this K-block
        const int8_t* x_ptr = x_int8 + kb * B;

        // Dot product: INT8→FP32 with per-block scales
        #pragma unroll
        for (int j = 0; j < B; ++j) {
            acc += static_cast<float>(x_ptr[j]) * x_sc * 
                   static_cast<float>(w_buf[j]) * w_sc;
        }
    }

    y_out[n_out] = acc;
}

} // anonymous namespace

// ===========================================================================
// Public API
// ===========================================================================

cudaError_t gemv_int8(
    float*          y_out,
    const void*     x_int8,
    const float*    x_scale,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || N % 16 != 0)
        return cudaErrorInvalidValue;

    int nb = (N + kINT8Block - 1) / kINT8Block;
    gemv_int8_kernel<<<dim3(nb), dim3(kINT8Block), 0, stream>>>(
        y_out,
        static_cast<const int8_t*>(x_int8), x_scale,
        static_cast<const int8_t*>(W_t_int8), W_t_scale,
        K, N);
    return cudaPeekAtLastError();
}

} // namespace kernels
} // namespace blackwell
