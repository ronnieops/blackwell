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
#include <cuda_fp4.h>
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
        // Load 16 INT8 weight + activation values via vectorized uint4 loads
        const int8_t* w_ptr = &W_t_int8[n_out * K + kb * B];
        alignas(16) int8_t w_buf[B];
        alignas(16) int8_t x_buf[B];
        *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);
        *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_int8 + kb * B);

        float w_sc = W_t_scale[n_blk * num_K_blks + kb];
        float x_sc = x_scale[kb];
        float prod_scale = w_sc * x_sc;

        // __dp4a: 4-way int8 SIMD dot product per iteration (4 × 4 = 16 total)
        const int* w32 = reinterpret_cast<const int*>(w_buf);
        const int* x32 = reinterpret_cast<const int*>(x_buf);
        int sumi = 0;
        sumi = __dp4a(w32[0], x32[0], sumi);
        sumi = __dp4a(w32[1], x32[1], sumi);
        sumi = __dp4a(w32[2], x32[2], sumi);
        sumi = __dp4a(w32[3], x32[3], sumi);
        acc += static_cast<float>(sumi) * prod_scale;
    }

    y_out[n_out] = acc;
}

// ---------------------------------------------------------------------------
// Fused INT8 GEMV kernel — reads FP4 input, converts to INT8 inline
// Eliminates: unpack_fp4 + pack_int8 + gemv_int8 (3 launches → 1 launch)
// ---------------------------------------------------------------------------
__launch_bounds__(kINT8Block, 1)
__global__ void gemv_int8_from_fp4_kernel(
    float* __restrict__ y_out,
    const __nv_fp4_e2m1* __restrict__ x_fp4,
    const float* __restrict__ x_fp4_scale,      // [K/16]
    const int8_t* __restrict__ W_t_int8,
    const float* __restrict__ W_t_scale,        // [N/16 × K/16]
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
        // Load 16 FP4 input values (16 bytes = uint4)
        alignas(16) __nv_fp4_e2m1 x_buf[B];
        *reinterpret_cast<uint4*>(x_buf) = 
            *reinterpret_cast<const uint4*>(&x_fp4[kb * B]);

        // Apply FP4 scales to get FP32 values and compute INT8 block scale
        float fp4_sc = x_fp4_scale[kb];
        float vals[B];
        float block_max = 0.0f;
        #pragma unroll
        for (int j = 0; j < B; ++j) {
            float v = static_cast<float>(x_buf[j]) * fp4_sc;
            vals[j] = v;
            float av = fabsf(v);
            if (av > block_max) block_max = av;
        }

        // Compute INT8 scale for this 16-element input block
        float i8_sc = block_max / 127.0f;
        if (i8_sc < 1e-10f) i8_sc = 1e-10f;

        // Load 16 INT8 weight values
        const int8_t* w_ptr = &W_t_int8[n_out * K + kb * B];
        alignas(16) int8_t w_buf[B];
        *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);

        float w_sc = W_t_scale[n_blk * num_K_blks + kb];

        // Quantize x values to INT8 and accumulate
        #pragma unroll
        for (int j = 0; j < B; ++j) {
            float x_qf = roundf(vals[j] / i8_sc);
            x_qf = fminf(127.0f, fmaxf(-127.0f, x_qf));
            int8_t x_q = static_cast<int8_t>(static_cast<int>(x_qf));
            acc += static_cast<float>(x_q) * i8_sc *
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
// FP32×INT8 GEMV — FP32 activations × INT8 weights
// ===========================================================================
__launch_bounds__(kINT8Block, 1)
__global__ void gemv_fp32_int8_kernel(
    float* __restrict__ y_out,
    const float* __restrict__ x_fp32,
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
        // Load 16 FP32 activation values (4× float4 = 16 floats)
        int x_off = kb * B;
        float4 v0 = reinterpret_cast<const float4*>(&x_fp32[x_off])[0];
        float4 v1 = reinterpret_cast<const float4*>(&x_fp32[x_off])[1];
        float4 v2 = reinterpret_cast<const float4*>(&x_fp32[x_off])[2];
        float4 v3 = reinterpret_cast<const float4*>(&x_fp32[x_off])[3];

        // Load 16 INT8 weight values (4× int vectors)
        const int8_t* w_ptr = &W_t_int8[n_out * K + kb * B];
        int w0 = reinterpret_cast<const int*>(w_ptr)[0];
        int w1 = reinterpret_cast<const int*>(w_ptr)[1];
        int w2 = reinterpret_cast<const int*>(w_ptr)[2];
        int w3 = reinterpret_cast<const int*>(w_ptr)[3];

        float w_sc = W_t_scale[n_blk * num_K_blks + kb];

        // Unpack int8 (sign-extend) → float and multiply-add (16-wide)
        #define SE(i) (float)(int8_t)((i) & 0xFF)
        float sum_b = 0.0f;
        sum_b += SE(w0 >>  0) * v0.x;  sum_b += SE(w0 >>  8) * v0.y;
        sum_b += SE(w0 >> 16) * v0.z;  sum_b += SE(w0 >> 24) * v0.w;
        sum_b += SE(w1 >>  0) * v1.x;  sum_b += SE(w1 >>  8) * v1.y;
        sum_b += SE(w1 >> 16) * v1.z;  sum_b += SE(w1 >> 24) * v1.w;
        sum_b += SE(w2 >>  0) * v2.x;  sum_b += SE(w2 >>  8) * v2.y;
        sum_b += SE(w2 >> 16) * v2.z;  sum_b += SE(w2 >> 24) * v2.w;
        sum_b += SE(w3 >>  0) * v3.x;  sum_b += SE(w3 >>  8) * v3.y;
        sum_b += SE(w3 >> 16) * v3.z;  sum_b += SE(w3 >> 24) * v3.w;
        #undef SE
        acc += sum_b * w_sc;
    }

    y_out[n_out] = acc;
}

// ---------------------------------------------------------------------------
// Per-row INT8 GEMV kernel — each output row has independent scales.
// Scale layout: W_t_scale [N × K/16] (vs old 2D [N/16 × K/16]).
// ---------------------------------------------------------------------------
__launch_bounds__(kINT8Block, 1)
__global__ void gemv_int8_per_row_kernel(
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

    float acc = 0.0f;

    for (int kb = 0; kb < num_K_blks; ++kb) {
        const int8_t* w_ptr = &W_t_int8[n_out * K + kb * B];
        alignas(16) int8_t w_buf[B];
        alignas(16) int8_t x_buf[B];
        *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);
        *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_int8 + kb * B);

        float w_sc = W_t_scale[n_out * num_K_blks + kb];  // per-row scale
        float x_sc = x_scale[kb];
        float prod_scale = w_sc * x_sc;

        const int* w32 = reinterpret_cast<const int*>(w_buf);
        const int* x32 = reinterpret_cast<const int*>(x_buf);
        int sumi = 0;
        sumi = __dp4a(w32[0], x32[0], sumi);
        sumi = __dp4a(w32[1], x32[1], sumi);
        sumi = __dp4a(w32[2], x32[2], sumi);
        sumi = __dp4a(w32[3], x32[3], sumi);
        acc += static_cast<float>(sumi) * prod_scale;
    }

    y_out[n_out] = acc;
}

// ===========================================================================
// FP32×INT8 per-row GEMV — FP32 activations × INT8 weights with per-row scales
// ===========================================================================
__launch_bounds__(kINT8Block, 1)
__global__ void gemv_fp32_int8_per_row_kernel(
    float* __restrict__ y_out,
    const float* __restrict__ x_fp32,
    const int8_t* __restrict__ W_t_int8,
    const float* __restrict__ W_t_scale,
    int K, int N)
{
    constexpr int B = 16;
    int tid = threadIdx.x;
    int n_out = blockIdx.x * kINT8Block + tid;
    if (n_out >= N) return;

    int num_K_blks = K / B;

    float acc = 0.0f;

    for (int kb = 0; kb < num_K_blks; ++kb) {
        int x_off = kb * B;
        float4 v0 = reinterpret_cast<const float4*>(&x_fp32[x_off])[0];
        float4 v1 = reinterpret_cast<const float4*>(&x_fp32[x_off])[1];
        float4 v2 = reinterpret_cast<const float4*>(&x_fp32[x_off])[2];
        float4 v3 = reinterpret_cast<const float4*>(&x_fp32[x_off])[3];

        const int8_t* w_ptr = &W_t_int8[n_out * K + kb * B];
        int w0 = reinterpret_cast<const int*>(w_ptr)[0];
        int w1 = reinterpret_cast<const int*>(w_ptr)[1];
        int w2 = reinterpret_cast<const int*>(w_ptr)[2];
        int w3 = reinterpret_cast<const int*>(w_ptr)[3];

        float w_sc = W_t_scale[n_out * num_K_blks + kb];  // per-row scale

        #define SE2(i) (float)(int8_t)((i) & 0xFF)
        float sum_b = 0.0f;
        sum_b += SE2(w0 >>  0) * v0.x;  sum_b += SE2(w0 >>  8) * v0.y;
        sum_b += SE2(w0 >> 16) * v0.z;  sum_b += SE2(w0 >> 24) * v0.w;
        sum_b += SE2(w1 >>  0) * v1.x;  sum_b += SE2(w1 >>  8) * v1.y;
        sum_b += SE2(w1 >> 16) * v1.z;  sum_b += SE2(w1 >> 24) * v1.w;
        sum_b += SE2(w2 >>  0) * v2.x;  sum_b += SE2(w2 >>  8) * v2.y;
        sum_b += SE2(w2 >> 16) * v2.z;  sum_b += SE2(w2 >> 24) * v2.w;
        sum_b += SE2(w3 >>  0) * v3.x;  sum_b += SE2(w3 >>  8) * v3.y;
        sum_b += SE2(w3 >> 16) * v3.z;  sum_b += SE2(w3 >> 24) * v3.w;
        #undef SE2
        acc += sum_b * w_sc;
    }

    y_out[n_out] = acc;
}

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

cudaError_t gemv_int8_from_fp4(
    float*          y_out,
    const void*     x_fp4,
    const float*    x_fp4_scale,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || N % 16 != 0)
        return cudaErrorInvalidValue;

    int nb = (N + kINT8Block - 1) / kINT8Block;
    gemv_int8_from_fp4_kernel<<<dim3(nb), dim3(kINT8Block), 0, stream>>>(
        y_out,
        static_cast<const __nv_fp4_e2m1*>(x_fp4), x_fp4_scale,
        static_cast<const int8_t*>(W_t_int8), W_t_scale,
        K, N);
    return cudaPeekAtLastError();
}

cudaError_t gemv_fp32_int8(
    float*          y_out,
    const float*    x_fp32,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || N % 16 != 0)
        return cudaErrorInvalidValue;

    int nb = (N + kINT8Block - 1) / kINT8Block;
    gemv_fp32_int8_kernel<<<dim3(nb), dim3(kINT8Block), 0, stream>>>(
        y_out, x_fp32,
        static_cast<const int8_t*>(W_t_int8), W_t_scale,
        K, N);
    return cudaPeekAtLastError();
}

cudaError_t gemv_fp32_int8_per_row(
    float*          y_out,
    const float*    x_fp32,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || N % 16 != 0)
        return cudaErrorInvalidValue;

    int nb = (N + kINT8Block - 1) / kINT8Block;
    gemv_fp32_int8_per_row_kernel<<<dim3(nb), dim3(kINT8Block), 0, stream>>>(
        y_out, x_fp32,
        static_cast<const int8_t*>(W_t_int8), W_t_scale,
        K, N);
    return cudaPeekAtLastError();
}

cudaError_t gemv_int8_per_row(
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
    gemv_int8_per_row_kernel<<<dim3(nb), dim3(kINT8Block), 0, stream>>>(
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


// ---------------------------------------------------------------------------
// INT8 GEMV Split-K kernel: K split into K_splits, AtomicAdd reduction.
// Grid: (N/256, K_splits). Each block computes partial dot product over
// K/K_splits columns. AtomicAdd to reduce to same output row.
// Caller MUST zero y_out before launch (cudaMemset).
//
// Targets N=6144 down_proj where N/256 = 24 blocks < 36 SMs.
// K_splits=2 → 48 blocks, K_splits=3 → 72 blocks > 36 SMs.
// ---------------------------------------------------------------------------
__launch_bounds__(kINT8Block, 1)
__global__ void gemv_int8_splitk_kernel(
    float* __restrict__ y_out,
    const int8_t* __restrict__ x_int8,
    const float* __restrict__ x_scale,
    const int8_t* __restrict__ W_t_int8,
    const float* __restrict__ W_t_scale,
    int K, int N, int K_splits)
{
    constexpr int B = 16;
    int tid = threadIdx.x;
    int n_out = blockIdx.x * kINT8Block + tid;
    if (n_out >= N) return;

    int split_id = blockIdx.y;
    int num_K_blks = K / B;
    int n_blk = n_out / B;

    // Each split handles K/K_splits columns
    int split_blks = num_K_blks / K_splits;
    int kb_start = split_id * split_blks;
    int kb_end = (split_id == K_splits - 1) ? num_K_blks : (kb_start + split_blks);

    float acc = 0.0f;

    for (int kb = kb_start; kb < kb_end; ++kb) {
        // Load 16 INT8 weight + activation values via vectorized uint4 loads
        const int8_t* w_ptr = &W_t_int8[n_out * K + kb * B];
        alignas(16) int8_t w_buf[B];
        alignas(16) int8_t x_buf[B];
        *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);
        *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_int8 + kb * B);

        float w_sc = W_t_scale[n_blk * num_K_blks + kb];
        float x_sc = x_scale[kb];
        float prod_scale = w_sc * x_sc;

        // __dp4a: 4-way int8 SIMD dot product per iteration (4 × 4 = 16 total)
        const int* w32 = reinterpret_cast<const int*>(w_buf);
        const int* x32 = reinterpret_cast<const int*>(x_buf);
        int sumi = 0;
        sumi = __dp4a(w32[0], x32[0], sumi);
        sumi = __dp4a(w32[1], x32[1], sumi);
        sumi = __dp4a(w32[2], x32[2], sumi);
        sumi = __dp4a(w32[3], x32[3], sumi);
        acc += static_cast<float>(sumi) * prod_scale;
    }

    // AtomicAdd reduction on shared output row
    atomicAdd(&y_out[n_out], acc);
}

cudaError_t gemv_int8_splitk(
    float*          y_out,
    const void*     x_int8,
    const float*    x_scale,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    int             K_splits,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || N % 16 != 0 || K % K_splits != 0)
        return cudaErrorInvalidValue;

    dim3 grid((N + kINT8Block - 1) / kINT8Block, K_splits);
    gemv_int8_splitk_kernel<<<grid, kINT8Block, 0, stream>>>(
        y_out,
        static_cast<const int8_t*>(x_int8), x_scale,
        static_cast<const int8_t*>(W_t_int8), W_t_scale,
        K, N, K_splits);
    return cudaPeekAtLastError();
}

// ---------------------------------------------------------------------------
// INT8 GEMV Persistent kernel: grid-stride loop over N tiles.
// Launches exactly 36 blocks (one per SM). Each block uses atomic work
// scheduling to grab the next available N-tile, processing until all
// N/256 tiles are consumed.
// ---------------------------------------------------------------------------
__launch_bounds__(kINT8Block, 1)
__global__ void gemv_int8_persistent_kernel(
    float* __restrict__ y_out,
    const int8_t* __restrict__ x_int8,
    const float* __restrict__ x_scale,
    const int8_t* __restrict__ W_t_int8,
    const float* __restrict__ W_t_scale,
    int K, int N, int total_tiles)
{
    constexpr int B = 16;
    __shared__ int tile_counter;
    __shared__ int cur_tile;

    if (threadIdx.x == 0)
        tile_counter = 0;
    __syncthreads();

    while (true) {
        if (threadIdx.x == 0)
            cur_tile = atomicAdd(&tile_counter, 1);
        __syncthreads();

        int tile = cur_tile;
        if (tile >= total_tiles) break;

        int n_start = tile * kINT8Block;
        int n_out = n_start + threadIdx.x;

        float acc = 0.0f;
        if (n_out < N) {
            int num_K_blks = K / B;
            int n_blk = n_out / B;

            for (int kb = 0; kb < num_K_blks; ++kb) {
                const int8_t* w_ptr = &W_t_int8[n_out * K + kb * B];
                alignas(16) int8_t w_buf[B];
                alignas(16) int8_t x_buf[B];
                *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);
                *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_int8 + kb * B);

                float w_sc = W_t_scale[n_blk * num_K_blks + kb];
                float x_sc = x_scale[kb];
                float prod_scale = w_sc * x_sc;

                const int* w32 = reinterpret_cast<const int*>(w_buf);
                const int* x32 = reinterpret_cast<const int*>(x_buf);
                int sumi = 0;
                sumi = __dp4a(w32[0], x32[0], sumi);
                sumi = __dp4a(w32[1], x32[1], sumi);
                sumi = __dp4a(w32[2], x32[2], sumi);
                sumi = __dp4a(w32[3], x32[3], sumi);
                acc += static_cast<float>(sumi) * prod_scale;
            }
            y_out[n_out] = acc;
        }
        __syncthreads();
    }
}

cudaError_t gemv_int8_persistent(
    float*          y_out,
    const void*     x_int8,
    const float*    x_scale,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    cudaStream_t    stream)
{
    if (K % 16 != 0)
        return cudaErrorInvalidValue;

    int total_tiles = (N + kINT8Block - 1) / kINT8Block;
    constexpr int kNumSMs = 36;
    gemv_int8_persistent_kernel<<<kNumSMs, kINT8Block, sizeof(int), stream>>>(
        y_out,
        static_cast<const int8_t*>(x_int8), x_scale,
        static_cast<const int8_t*>(W_t_int8), W_t_scale,
        K, N, total_tiles);
    return cudaPeekAtLastError();
}

// ---------------------------------------------------------------------------
// INT8 Batched GEMV: process M tokens simultaneously, reuse weights across them.
// Grid: (ceil(N/256), M). Block: 256 threads.
// Weights loaded once per K-block, activations loaded per-token.
// Eliminates M-1 weight loads vs launching M separate gemv_int8 kernels.
// Best batch sizes: 2-8 tokens (matching llama.cpp MMVQ_MAX_BATCH_SIZE).
// ---------------------------------------------------------------------------
template<int M>
__global__ void gemv_int8_batched_kernel(
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
    int m = blockIdx.y;
    if (n_out >= N) return;

    int num_K_blks = K / B;
    int n_blk = n_out / B;

    float acc = 0.0f;

    for (int kb = 0; kb < num_K_blks; ++kb) {
        // Load 16 weight values once (shared across M tokens via template unrolling)
        const int8_t* w_ptr = &W_t_int8[n_out * K + kb * B];
        alignas(16) int8_t w_buf[B];
        *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);

        float w_sc = W_t_scale[n_blk * num_K_blks + kb];

        // Load this token's activation
        const int8_t* x_ptr = &x_int8[m * K + kb * B];
        alignas(16) int8_t x_buf[B];
        *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_ptr);

        float x_sc = x_scale[m * num_K_blks + kb];
        float prod_scale = w_sc * x_sc;

        const int* w32 = reinterpret_cast<const int*>(w_buf);
        const int* x32 = reinterpret_cast<const int*>(x_buf);
        int sumi = 0;
        sumi = __dp4a(w32[0], x32[0], sumi);
        sumi = __dp4a(w32[1], x32[1], sumi);
        sumi = __dp4a(w32[2], x32[2], sumi);
        sumi = __dp4a(w32[3], x32[3], sumi);
        acc += static_cast<float>(sumi) * prod_scale;
    }

    y_out[m * N + n_out] = acc;
}

// ---------------------------------------------------------------------------
// Batched GEMV dispatch: routes to templated kernel based on M.
// ---------------------------------------------------------------------------
cudaError_t gemv_int8_batched(
    float*          y_out,
    const void*     x_int8,
    const float*    x_scale,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    int             M,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || N % 16 != 0 || M < 1 || M > 8)
        return cudaErrorInvalidValue;

    int nb = (N + kINT8Block - 1) / kINT8Block;
    dim3 grid(nb, M);

    switch (M) {
        case 1:
            gemv_int8_batched_kernel<1><<<grid, kINT8Block, 0, stream>>>(
                y_out, static_cast<const int8_t*>(x_int8), x_scale,
                static_cast<const int8_t*>(W_t_int8), W_t_scale, K, N);
            break;
        case 2:
            gemv_int8_batched_kernel<2><<<grid, kINT8Block, 0, stream>>>(
                y_out, static_cast<const int8_t*>(x_int8), x_scale,
                static_cast<const int8_t*>(W_t_int8), W_t_scale, K, N);
            break;
        case 3:
            gemv_int8_batched_kernel<3><<<grid, kINT8Block, 0, stream>>>(
                y_out, static_cast<const int8_t*>(x_int8), x_scale,
                static_cast<const int8_t*>(W_t_int8), W_t_scale, K, N);
            break;
        case 4:
            gemv_int8_batched_kernel<4><<<grid, kINT8Block, 0, stream>>>(
                y_out, static_cast<const int8_t*>(x_int8), x_scale,
                static_cast<const int8_t*>(W_t_int8), W_t_scale, K, N);
            break;
        case 5:
            gemv_int8_batched_kernel<5><<<grid, kINT8Block, 0, stream>>>(
                y_out, static_cast<const int8_t*>(x_int8), x_scale,
                static_cast<const int8_t*>(W_t_int8), W_t_scale, K, N);
            break;
        case 6:
            gemv_int8_batched_kernel<6><<<grid, kINT8Block, 0, stream>>>(
                y_out, static_cast<const int8_t*>(x_int8), x_scale,
                static_cast<const int8_t*>(W_t_int8), W_t_scale, K, N);
            break;
        case 7:
            gemv_int8_batched_kernel<7><<<grid, kINT8Block, 0, stream>>>(
                y_out, static_cast<const int8_t*>(x_int8), x_scale,
                static_cast<const int8_t*>(W_t_int8), W_t_scale, K, N);
            break;
        case 8:
            gemv_int8_batched_kernel<8><<<grid, kINT8Block, 0, stream>>>(
                y_out, static_cast<const int8_t*>(x_int8), x_scale,
                static_cast<const int8_t*>(W_t_int8), W_t_scale, K, N);
            break;
    }
    return cudaPeekAtLastError();
}

// ===========================================================================
// INT8 GEMM — C[M×N] = A[M×K] × B^T[N×K]
// A is FP32 activations, B is INT8 weights [N×K] with scales [N × K/16]
// Uses 4×4 register tiling with vectorized loads.
// ===========================================================================
__launch_bounds__(256, 1)
__global__ void gemm_int8_kernel(
    float* __restrict__ C,          // [M×N]
    const float* __restrict__ A,    // [M×K] FP32
    const int8_t* __restrict__ B_i8, // [N×K] INT8 transposed
    const float* __restrict__ B_sc,  // [N × K/16] scales
    int M, int N, int K)
{
    constexpr int TILE_M = 4;
    constexpr int TILE_N = 4;
    constexpr int THREADS_M = 16;
    constexpr int THREADS_N = 16;
    constexpr int BSIZE = 16;  // K-block size

    int bm = blockIdx.y * THREADS_M * TILE_M;
    int bn = blockIdx.x * THREADS_N * TILE_N;
    int tm = threadIdx.y;
    int tn = threadIdx.x;
    int m = bm + tm * TILE_M;
    int n = bn + tn * TILE_N;
    int num_K_blks = K / BSIZE;

    float acc[TILE_M][TILE_N] = {};

    for (int kb = 0; kb < num_K_blks; ++kb) {
        float w_sc[TILE_N];
        #pragma unroll
        for (int j = 0; j < TILE_N; ++j) {
            int nj = n + j;
            w_sc[j] = (nj < N) ? B_sc[nj * num_K_blks + kb] : 0.0f;
        }

        float a_vals[TILE_M][BSIZE];
        #pragma unroll
        for (int i = 0; i < TILE_M; ++i) {
            int mi = m + i;
            if (mi < M) {
                const float* a_ptr = &A[mi * K + kb * BSIZE];
                *reinterpret_cast<float4*>(&a_vals[i][0]) = *reinterpret_cast<const float4*>(a_ptr);
                *reinterpret_cast<float4*>(&a_vals[i][4]) = *reinterpret_cast<const float4*>(a_ptr + 4);
                *reinterpret_cast<float4*>(&a_vals[i][8]) = *reinterpret_cast<const float4*>(a_ptr + 8);
                *reinterpret_cast<float4*>(&a_vals[i][12]) = *reinterpret_cast<const float4*>(a_ptr + 12);
            }
        }

        #pragma unroll
        for (int j = 0; j < TILE_N; ++j) {
            int nj = n + j;
            if (nj < N) {
                const int8_t* w_ptr = &B_i8[nj * K + kb * BSIZE];
                alignas(16) int8_t w_buf[BSIZE];
                *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);

                #pragma unroll
                for (int i = 0; i < TILE_M; ++i) {
                    float block_sum = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < BSIZE; ++k) {
                        block_sum += (float)w_buf[k] * a_vals[i][k];
                    }
                    acc[i][j] += block_sum * w_sc[j];
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < TILE_M; ++i) {
        int mi = m + i;
        if (mi < M) {
            #pragma unroll
            for (int j = 0; j < TILE_N; ++j) {
                int nj = n + j;
                if (nj < N) C[mi * N + nj] = acc[i][j];
            }
        }
    }
}

cudaError_t gemm_int8(
    float*          C,              // [M×N] output
    const float*    A,              // [M×K] FP32 activations
    const void*     B_int8,         // [N×K] INT8 transposed weights
    const float*    B_scale,        // [N × K/16] weight scales
    int             M, int N, int K,
    cudaStream_t    stream)
{
    if (K % 16 != 0) return cudaErrorInvalidValue;
    constexpr int TILE = 4;
    constexpr int THREADS = 16;
    dim3 block(THREADS, THREADS);
    dim3 grid((N + THREADS * TILE - 1) / (THREADS * TILE),
              (M + THREADS * TILE - 1) / (THREADS * TILE));
    gemm_int8_kernel<<<grid, block, 0, stream>>>(
        C, A, static_cast<const int8_t*>(B_int8), B_scale, M, N, K);
    return cudaPeekAtLastError();
}
} // namespace kernels
} // namespace blackwell
