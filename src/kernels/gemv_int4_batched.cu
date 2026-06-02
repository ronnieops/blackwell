// gemv_int4_batched.cu — Batched INT4 GEMV for M sequences
//
// INT4 warp GEMV: 1 warp (32 threads) per output row.
// For batched decode: processes M sequences per layer.
// Weight loaded once per K-block, reused across M tokens.
//
// Grid: (N/32) × M blocks, 32 threads/block
// Thread pattern: stride-32 over K-blocks
//
// API: gemv_int4_batched(y_out[M][N], x_packed[M][K/2], x_scale[M][K/16],
//                           W_packed[N][K/2], W_scale[N][K/16], K, N, M)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

// Unpack 1 byte (2 INT4 nibbles) to 2 floats
__device__ __forceinline__ void int4_byte_to_floats(uint8_t b, float& f0, float& f1) {
    int lo = b & 0x0F; if (lo > 7) lo -= 16;
    int hi = (b >> 4) & 0x0F; if (hi > 7) hi -= 16;
    f0 = static_cast<float>(lo);
    f1 = static_cast<float>(hi);
}

// Load __half from float* with array index (scale stored as __half at stride-1)
__device__ __forceinline__ float load_half_as_float(const float* base, int idx) {
    __half h = *reinterpret_cast<const __half*>(&base[idx]);  // idx = element index (×4 bytes)
    return __half2float(h);
}

// Load scale as float (scale stored as float, for FP32 compatibility)
__device__ __forceinline__ float load_scale_float(const float* base, int idx) {
    return base[idx];
}

__launch_bounds__(32, 8)
__global__ void gemv_int4_batched_kernel(
    float* __restrict__ y_out,          // [M][N]
    const uint8_t* __restrict__ x_packed,    // [M][K/2] packed INT4
    const float* __restrict__ x_scale,       // [M][K/16] scales (stored as __half)
    const uint8_t* __restrict__ W_packed,    // [N][K/2] packed INT4
    const float* __restrict__ W_scale,      // [N][K/16] scales
    int K, int N, int M)
{
    constexpr int B = 16, PB = 8;
    int n_out = blockIdx.x;   // output row index (0..N-1, stride by block)
    int m = blockIdx.y;       // sequence index (0..M-1)
    int tid = threadIdx.x;

    int num_K_blks = K / B;

    // Stride-32 loop: each thread handles K-blocks at indices tid, tid+32, ...
    float acc = 0.0f;
    for (int kb = tid; kb < num_K_blks; kb += 32) {
        // Load weight row (once per K-block, reused across all M tokens)
        const uint8_t* w_ptr = &W_packed[(size_t)n_out * (K / 2) + kb * PB];
        uint2 w_packed = *reinterpret_cast<const uint2*>(w_ptr);

        // Load this token's activation
        const uint8_t* x_ptr = &x_packed[(size_t)m * (K / 2) + kb * PB];
        uint2 x_packed_val = *reinterpret_cast<const uint2*>(x_ptr);

        // Load scales (at byte_idx for __half storage, or idx for float)
        float w_sc = load_scale_float(W_scale, (size_t)n_out * num_K_blks + kb);
        float x_sc = load_scale_float(x_scale, (size_t)m * num_K_blks + kb);
        float prod_scale = w_sc * x_sc;

        // Unpack and dot
        const uint8_t* wb = reinterpret_cast<const uint8_t*>(&w_packed);
        const uint8_t* xb = reinterpret_cast<const uint8_t*>(&x_packed_val);

        float sum_f = 0.0f;
        #pragma unroll
        for (int j = 0; j < PB; ++j) {
            float w0, w1, x0, x1;
            int4_byte_to_floats(wb[j], w0, w1);
            int4_byte_to_floats(xb[j], x0, x1);
            sum_f += w0 * x0 + w1 * x1;
        }
        acc += sum_f * prod_scale;
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

cudaError_t gemv_int4_batched(
    float*          y_out,
    const uint8_t*  x_packed,
    const float*    x_scale,
    const uint8_t*  W_packed,
    const float*    W_scale,
    int             K,
    int             N,
    int             M,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || N % 32 != 0 || M < 1)
        return cudaErrorInvalidValue;

    dim3 grid(N / 32, M);
    gemv_int4_batched_kernel<<<grid, 32, 0, stream>>>(
        y_out, x_packed, x_scale, W_packed, W_scale, K, N, M);
    return cudaPeekAtLastError();
}

} // namespace kernels
} // namespace blackwell