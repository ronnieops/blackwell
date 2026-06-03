// gemv_int4_asym_batched.cu — Asymmetric INT4 batched GEMV with zero point
//
// Same structure as gemv_int4_batched but uses per-block zero point.
// Scale format: [2 * num_blocks] where even = scale, odd = zero (as float)
// Zero is stored as float for simplicity, converted to int in kernel.
//
// API: gemv_int4_asym_batched(...)
// Grid: dim3(N, M), 32 threads/block

#include <cuda_runtime.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

// Unpack INT4 nibble with zero point
__device__ __forceinline__ void int4_nibble_unpack(
    uint8_t b, float& f0, float& f1,
    float scale, int zero)
{
    int lo = (b & 0x0F);
    int hi = ((b >> 4) & 0x0F);
    f0 = static_cast<float>(lo - zero) * scale;
    f1 = static_cast<float>(hi - zero) * scale;
}

// Load scale + zero from combined array
__device__ __forceinline__ void load_scale_zero(
    const float* base, int idx,
    float& scale, int& zero)
{
    scale = base[idx * 2];
    float zf = base[idx * 2 + 1];
    zero = __float2int_rn(zf);
}

__launch_bounds__(32, 8)
__global__ void gemv_int4_asym_batched_kernel(
    float* __restrict__ y_out,           // [M][N]
    const uint8_t* __restrict__ x_packed, // [M][K/2] packed INT4
    const float* __restrict__ x_sc_zero,  // [M][2 * K/16] (scale, zero) pairs
    const uint8_t* __restrict__ W_packed, // [N][K/2] packed INT4
    const float* __restrict__ W_sc_zero, // [N][2 * K/16] (scale, zero) pairs
    int K, int N, int M)
{
    constexpr int B = 16, PB = 8;
    int n_out = blockIdx.x;
    int m = blockIdx.y;
    int tid = threadIdx.x;

    int num_K_blks = K / B;

    float acc = 0.0f;
    for (int kb = tid; kb < num_K_blks; kb += 32) {
        // Load weight row (once per K-block)
        const uint8_t* w_ptr = &W_packed[(size_t)n_out * (K / 2) + kb * PB];
        uint2 w_packed = *reinterpret_cast<const uint2*>(w_ptr);

        // Load activation
        const uint8_t* x_ptr = &x_packed[(size_t)m * (K / 2) + kb * PB];
        uint2 x_packed_val = *reinterpret_cast<const uint2*>(x_ptr);

        // Load scales + zeros
        float w_sc; int w_zero;
        load_scale_zero(W_sc_zero, (size_t)n_out * num_K_blks + kb, w_sc, w_zero);
        float x_sc; int x_zero;
        load_scale_zero(x_sc_zero, (size_t)m * num_K_blks + kb, x_sc, x_zero);

        // Unpack and dot
        const uint8_t* wb = reinterpret_cast<const uint8_t*>(&w_packed);
        const uint8_t* xb = reinterpret_cast<const uint8_t*>(&x_packed_val);

        float sum_f = 0.0f;
        #pragma unroll
        for (int j = 0; j < PB; ++j) {
            float w0, w1, x0, x1;
            int4_nibble_unpack(wb[j], w0, w1, w_sc, w_zero);
            int4_nibble_unpack(xb[j], x0, x1, x_sc, x_zero);
            sum_f += w0 * x0 + w1 * x1;
        }
        acc += sum_f;
    }

    // Warp reduction
    acc += __shfl_xor_sync(0xffffffff, acc, 16);
    acc += __shfl_xor_sync(0xffffffff, acc, 8);
    acc += __shfl_xor_sync(0xffffffff, acc, 4);
    acc += __shfl_xor_sync(0xffffffff, acc, 2);
    acc += __shfl_xor_sync(0xffffffff, acc, 1);

    if (tid == 0) {
        y_out[(size_t)m * N + n_out] = acc;
    }
}

} // anonymous namespace

cudaError_t gemv_int4_asym_batched(
    float*          y_out,
    const uint8_t*  x_packed,
    const float*    x_sc_zero,
    const uint8_t*  W_packed,
    const float*    W_sc_zero,
    int             K,
    int             N,
    int             M,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || N % 32 != 0 || M < 1)
        return cudaErrorInvalidValue;

    dim3 grid(N, M);
    gemv_int4_asym_batched_kernel<<<grid, 32, 0, stream>>>(
        y_out, x_packed, x_sc_zero, W_packed, W_sc_zero, K, N, M);
    return cudaPeekAtLastError();
}

} // namespace kernels
} // namespace blackwell