// BF16 GEMV kernel — FP32 activations × BF16 weights, FP32 accumulate.
// No quantization error. For quality reference / production decode.
// Weight layout: W_t [N × K] BF16 row-major (from safetensors, already transposed).
//
// Each thread computes one output element: y[n] = sum_k(W_t[n,k] * x[k])
// Uses float2 (half2) loads for 2× vectorization.
// No DP4A — BF16 values are converted to FP32 for accumulation.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>

namespace blackwell {
namespace kernels {
namespace {

constexpr int kBLK = 256;

__launch_bounds__(kBLK, 1)
__global__ void gemv_bf16_kernel(
    float* __restrict__ y_out,
    const float* __restrict__ x_fp32,
    const __nv_bfloat16* __restrict__ W_t_bf16,  // [N × K]
    int K, int N)
{
    int n_out = blockIdx.x * kBLK + threadIdx.x;
    if (n_out >= N) return;

    const __nv_bfloat16* w_row = &W_t_bf16[n_out * K];
    float acc = 0.0f;

    // Process 2 BF16 values per iteration using half2 (nv_bfloat162)
    int num_pairs = K / 2;
    for (int p = 0; p < num_pairs; ++p) {
        // Load 2 BF16 weights as __nv_bfloat162 (4 bytes)
        __nv_bfloat162 w_pair = reinterpret_cast<const __nv_bfloat162*>(w_row)[p];
        // Load 2 FP32 activations
        float2 x_pair = reinterpret_cast<const float2*>(x_fp32)[p];

        // Convert BF16 → FP32 and multiply-accumulate
        acc += __bfloat162float(w_pair.x) * x_pair.x;
        acc += __bfloat162float(w_pair.y) * x_pair.y;
    }

    y_out[n_out] = acc;
}

} // anonymous namespace

cudaError_t gemv_bf16(
    float*          y_out,
    const float*    x_fp32,
    const void*     W_t_bf16,
    int             K,
    int             N,
    cudaStream_t    stream = 0)
{
    if (K % 2 != 0 || N == 0)
        return cudaErrorInvalidValue;

    int nb = (N + kBLK - 1) / kBLK;
    gemv_bf16_kernel<<<dim3(nb), dim3(kBLK), 0, stream>>>(
        y_out, x_fp32,
        static_cast<const __nv_bfloat16*>(W_t_bf16),
        K, N);
    return cudaPeekAtLastError();
}

} // namespace kernels
} // namespace blackwell
