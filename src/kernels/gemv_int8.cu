// src/kernels/gemv_int8.cu — INT8 block-scaled GEMV + pack + transpose
//
// INT8 GEMV path: eliminates FP4 cast overhead (static_cast<float> per element).
// Block-scaled quantization: weights in INT8 [-128..127], scales in FP32 per 16×16 block.
// Activations in INT8, scales per 16-element K-block.
//
// Weight format: W_t [N×K] INT8 (transposed row-major).
// Scale format:  W_scale [N/16 × K/16] FP32 per block.
// Activation:    x_int8 [K], x_scale [K/16] FP32 per block.

#include <cuda_runtime.h>
#include <cuda/std/cmath>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace blackwell {
namespace kernels {
namespace {

constexpr int kINT8Block = 256;

// ---------------------------------------------------------------------------
// INT8 GEMV kernel — per-thread dot products, transposed weights
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

        #pragma unroll
        for (int j = 0; j < B; ++j) {
            acc += static_cast<float>(x_ptr[j]) * x_sc * 
                   static_cast<float>(w_buf[j]) * w_sc;
        }
    }

    y_out[n_out] = acc;
}

// ---------------------------------------------------------------------------
// INT8 pack kernel: FP32 → INT8 with per-block scales
// Block size = 16. Scale = absmax(block) / 127.0
// ---------------------------------------------------------------------------
__global__ void pack_int8_kernel(
    int8_t* __restrict__ out,
    const float* __restrict__ in,
    const float* __restrict__ scales,   // [num_block] pre-computed scales
    int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    int blk = idx / 16;
    float sc = scales[blk];
    float v = in[idx] / sc;
    v = fminf(127.0f, fmaxf(-127.0f, roundf(v)));
    out[idx] = static_cast<int8_t>(static_cast<int>(v));
}

// ---------------------------------------------------------------------------
// INT8 transpose: W (K×N) → W_t (N×K)
// ---------------------------------------------------------------------------
__global__ void transpose_int8_kernel(
    int8_t* __restrict__ dst,      // [N × K]
    const int8_t* __restrict__ src, // [K × N]
    int K, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * N;
    if (idx >= total) return;
    int k = idx / N;
    int n = idx % N;
    dst[n * K + k] = src[k * N + n];
}

// ---------------------------------------------------------------------------
// INT8 scale transpose: W_scale (K/16 × N/16) → W_t_scale (N/16 × K/16)
// ---------------------------------------------------------------------------
__global__ void transpose_scales_int8_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int num_K_blks, int num_N_blks)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_K_blks * num_N_blks;
    if (idx >= total) return;
    int kb = idx / num_N_blks;
    int nb = idx % num_N_blks;
    dst[nb * num_K_blks + kb] = src[kb * num_N_blks + nb];
}

} // anonymous namespace

// ===========================================================================
// Public API
// ===========================================================================

cudaError_t pack_int8(
    void*           out_int8,
    const float*    in_fp32,
    const float*    scale_out,
    int             num_elements,
    cudaStream_t    stream)
{
    if (num_elements <= 0 || num_elements % 16 != 0)
        return cudaErrorInvalidValue;

    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    pack_int8_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<int8_t*>(out_int8), in_fp32, scale_out, num_elements);
    return cudaPeekAtLastError();
}

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

cudaError_t transpose_int8_weights(
    void*           dst,
    float*          dst_scale,
    const void*     src,
    const float*    src_scale,
    int             K,
    int             N,
    cudaStream_t    stream)
{
    int total = K * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    transpose_int8_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<int8_t*>(dst), static_cast<const int8_t*>(src), K, N);

    int num_K_blks = K / 16;
    int num_N_blks = N / 16;
    int total_scales = num_K_blks * num_N_blks;
    blocks = (total_scales + threads - 1) / threads;

    transpose_scales_int8_kernel<<<blocks, threads, 0, stream>>>(
        dst_scale, src_scale, num_K_blks, num_N_blks);

    return cudaPeekAtLastError();
}

} // namespace kernels
} // namespace blackwell
