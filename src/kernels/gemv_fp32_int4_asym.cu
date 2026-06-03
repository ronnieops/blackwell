// gemv_fp32_int4_asym.cu — FP32 activation × asymmetric INT4 weight GEMV
//
// Weight-only INT4 quantization: activations stay FP32, only weights quantized.
// Eliminates activation quantization noise — the main quality bottleneck
// in the 28-layer asymmetric INT4 pipeline.
//
// Pattern: 1 warp/row, 32 threads, stride-32 K-block iteration, shuffle reduce.
// Weight format: W_packed [N][K/2] packed INT4 nibbles
// Scale format:  W_sc_zero [N][2*K/16] — even=scale, odd=zero (as float)
// Input:        x_fp32 [K] FP32 activations
// Output:       y_out [N] FP32
//
// Grid: dim3(N), 32 threads/block

#include <cuda_runtime.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

__launch_bounds__(32, 8)
__global__ void gemv_fp32_int4_asym_kernel(
    float* __restrict__ y_out,
    const float* __restrict__ x_fp32,
    const uint8_t* __restrict__ W_packed,   // [N][K/2]
    const float* __restrict__ W_sc_zero,    // [N][2*K/16]
    int K, int N)
{
    constexpr int B = 16, PB = 8;  // 16 elements per K-block, 8 packed bytes
    int n_out = blockIdx.x;
    int tid = threadIdx.x;

    int num_K_blks = K / B;

    float acc = 0.0f;

    // Each thread processes K-blocks at stride-32
    for (int kb = tid; kb < num_K_blks; kb += 32) {
        // Load 16 FP32 activation values into a flat array
        int x_off = kb * B;
        float xf[B];
        *reinterpret_cast<float4*>(&xf[0])  = reinterpret_cast<const float4*>(&x_fp32[x_off])[0];
        *reinterpret_cast<float4*>(&xf[4])  = reinterpret_cast<const float4*>(&x_fp32[x_off])[1];
        *reinterpret_cast<float4*>(&xf[8])  = reinterpret_cast<const float4*>(&x_fp32[x_off])[2];
        *reinterpret_cast<float4*>(&xf[12]) = reinterpret_cast<const float4*>(&x_fp32[x_off])[3];

        // Load 8 packed bytes (16 INT4 values) from weight row
        const uint8_t* w_ptr = &W_packed[(size_t)n_out * (K / 2) + kb * PB];
        uint2 w_packed = *reinterpret_cast<const uint2*>(w_ptr);

        // Load scale + zero for this K-block
        float w_sc = W_sc_zero[(size_t)n_out * 2 * num_K_blks + kb * 2];
        float w_zero_f = W_sc_zero[(size_t)n_out * 2 * num_K_blks + kb * 2 + 1];
        int w_zero = __float2int_rn(w_zero_f);

        // Unpack INT4 nibbles and compute dot product with FP32 activations
        const uint8_t* wb = reinterpret_cast<const uint8_t*>(&w_packed);

        float sum_b = 0.0f;
        #pragma unroll
        for (int j = 0; j < PB; ++j) {
            int lo = (wb[j] & 0x0F);
            int hi = ((wb[j] >> 4) & 0x0F);
            float wl = static_cast<float>(lo - w_zero) * w_sc;
            float wh = static_cast<float>(hi - w_zero) * w_sc;
            sum_b += wl * xf[j * 2] + wh * xf[j * 2 + 1];
        }
        acc += sum_b;
    }

    // Warp shuffle reduction
    acc += __shfl_xor_sync(0xffffffff, acc, 16);
    acc += __shfl_xor_sync(0xffffffff, acc, 8);
    acc += __shfl_xor_sync(0xffffffff, acc, 4);
    acc += __shfl_xor_sync(0xffffffff, acc, 2);
    acc += __shfl_xor_sync(0xffffffff, acc, 1);

    if (tid == 0) y_out[n_out] = acc;
}

} // anonymous namespace

cudaError_t gemv_fp32_int4_asym(
    float*          y_out,
    const float*    x_fp32,
    const uint8_t*  W_packed,
    const float*    W_sc_zero,
    int             K,
    int             N,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || N % 32 != 0)
        return cudaErrorInvalidValue;

    dim3 grid(N);
    gemv_fp32_int4_asym_kernel<<<grid, 32, 0, stream>>>(
        y_out, x_fp32, W_packed, W_sc_zero, K, N);
    return cudaPeekAtLastError();
}

} // namespace kernels
} // namespace blackwell
