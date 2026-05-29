// src/kernels/gemv_v2.cu — Optimized GEMV with vectorized FP4 block loads
//
// Key optimization vs original gemv_fp4_kernel:
//   Original: reads W[k*N + n] — stride N bytes between K iterations.
//             Cache line utilization: 1/128 = 0.8%. Bandwidth: 22 GB/s (4.4% peak).
//
//   This kernel: reads W_t[n*K + k] — TRANSPOSED weight layout.
//             uint4 load at &W_t[n_out*K + k] reads 16 consecutive FP4 values
//             along K → full cache line utilization.
//             K/16 iterations instead of K.
//             Expected: 10× bandwidth improvement.
//
// Weight format: W_t is (N×K) row-major, where row n contains K FP4 values
// for output n. Scale layout: W_scale[n_blk * num_K_blks + k_blk].
//
// Also includes a transpose kernel to convert existing W (K×N) → W_t (N×K).

#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace blackwell {
namespace kernels {
namespace {

constexpr int kGEMVV2Block = 256;

// ---------------------------------------------------------------------------
// Vectorized GEMV kernel with transposed weights
//
// Each thread handles one output n_out.
// Inner loop: K/16 iterations, each loads uint4 (16 FP4 values) sequentially.
// x values loaded once into smem for broadcast.
// ---------------------------------------------------------------------------
__launch_bounds__(kGEMVV2Block, 1)
__global__ void gemv_fp4_v2_kernel(
    float* __restrict__ y_out,
    const __nv_fp4_e2m1* __restrict__ x_fp4,
    const float* __restrict__ x_scale,
    const __nv_fp4_e2m1* __restrict__ W_t_fp4,   // [N × K] transposed
    const float* __restrict__ W_t_scale,           // [N/16 × K/16]
    int K, int N)
{
    constexpr int B = 16;
    int tid = threadIdx.x;
    int n_out = blockIdx.x * kGEMVV2Block + tid;
    if (n_out >= N) return;

    int num_K_blks = K / B;
    int n_blk = n_out / B;

    float acc = 0.0f;

    // K-loop: iterate over blocks of 16 K-values
    // Each iteration: load uint4 (16 bytes = 16 FP4 values) from W_t[n_out*K + kb*16]
    // This is sequential in memory! Full cache line utilization.
    for (int kb = 0; kb < num_K_blks; ++kb) {
        // Load 16 FP4 weight values via uint4
        alignas(16) uint8_t buf[16];
        *reinterpret_cast<uint4*>(buf) = *reinterpret_cast<const uint4*>(
            &W_t_fp4[n_out * K + kb * B]);

        // Load weight scale for this block
        float w_scale = W_t_scale[n_out * num_K_blks + kb];

        // Load x values for this K-block (16 values)
        // x is small (K bytes total) — L1 cached, broadcast via read-only path
        int k_base = kb * B;
        int k_blk = kb;

        // Load x block scale
        float x_sc = x_scale[k_blk];

        // Dequant and accumulate 16 elements
        const __nv_fp4_e2m1* w_vals = reinterpret_cast<const __nv_fp4_e2m1*>(buf);
        #pragma unroll
        for (int j = 0; j < B; ++j) {
            float xv = static_cast<float>(x_fp4[k_base + j]) * x_sc;
            float wv = static_cast<float>(w_vals[j]) * w_scale;
            acc += xv * wv;
        }
    }

    y_out[n_out] = acc;
}

// ---------------------------------------------------------------------------
// Transpose kernel: converts W (K×N) → W_t (N×K)
// Each thread handles one element.
// ---------------------------------------------------------------------------
__global__ void transpose_fp4_kernel(
    __nv_fp4_e2m1* __restrict__ dst,     // [N × K]
    const __nv_fp4_e2m1* __restrict__ src, // [K × N]
    int K, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * N;
    if (idx >= total) return;
    int k = idx / N;
    int n = idx % N;
    dst[n * K + k] = src[k * N + n];
}

// Transpose scales: W_scale [K/16 × N/16] → W_t_scale [N/16 × K/16]
__global__ void transpose_scales_kernel(
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

cudaError_t gemv_fp4_v2(
    float*          y_out,
    const void*     x_fp4,
    const float*    x_scale,
    const void*     W_t_fp4,      // TRANSPOSED: [N × K]
    const float*    W_t_scale,    // TRANSPOSED: [N/16 × K/16]
    int             K,
    int             N,
    cudaStream_t    stream)
{
    using Fp4 = __nv_fp4_e2m1;
    if (K % 16 != 0 || N % 16 != 0)
        return cudaErrorInvalidValue;

    int nb = (N + kGEMVV2Block - 1) / kGEMVV2Block;
    gemv_fp4_v2_kernel<<<dim3(nb), dim3(kGEMVV2Block), 0, stream>>>(
        y_out,
        static_cast<const Fp4*>(x_fp4), x_scale,
        static_cast<const Fp4*>(W_t_fp4), W_t_scale,
        K, N);

    return cudaPeekAtLastError();
}

cudaError_t transpose_fp4_weights(
    void*           dst,          // [N × K] FP4 transposed
    float*          dst_scale,    // [N/16 × K/16] transposed
    const void*     src,          // [K × N] FP4 original
    const float*    src_scale,    // [K/16 × N/16] original
    int             K,
    int             N,
    cudaStream_t    stream)
{
    using Fp4 = __nv_fp4_e2m1;
    int total = K * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    transpose_fp4_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<Fp4*>(dst), static_cast<const Fp4*>(src), K, N);

    int num_K_blks = K / 16;
    int num_N_blks = N / 16;
    int total_scales = num_K_blks * num_N_blks;
    blocks = (total_scales + threads - 1) / threads;

    transpose_scales_kernel<<<blocks, threads, 0, stream>>>(
        dst_scale, src_scale, num_K_blks, num_N_blks);

    return cudaPeekAtLastError();
}

} // namespace kernels
} // namespace blackwell
