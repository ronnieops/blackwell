// src/kernels/apply_geglu.cu — GeGLU activation for Blackwell SM_120
//
// GeGLU (from Shazeer, GLU Variants, used by Gemma 4):
//   out = GeGLU(gate, up) = gelu(gate) * up
//   gelu(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//   This is the "gelu_pytorch_tanh" approximation used by HF Gemma.
//   erf-based GELU differs by ~1e-3 (low impact but not HF-correct).

#include <cuda_runtime.h>
#include <cmath>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

__device__ __forceinline__ float gelu(float x) {
    // gelu_pytorch_tanh: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // sqrt(2/pi) = 0.7978845608028654
    float x3 = x * x * x;
    float inner = 0.7978845608028654f * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

} // anonymous namespace

// ===========================================================================
// GeGLU kernel
// ===========================================================================
__launch_bounds__(256, 1)
__global__ void geglu_kernel(
    float* __restrict__ out,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    int n_pairs) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pairs) return;
    float g = gate[idx];
    float u = up[idx];
    out[idx] = gelu(g) * u;
}

// ===========================================================================
// Public API
// ===========================================================================
cudaError_t apply_geglu(
    float* out, const float* gate, const float* up,
    int num_pairs, cudaStream_t stream) {

    dim3 block(256);
    dim3 grid((num_pairs + 255) / 256);
    geglu_kernel<<<grid, block, 0, stream>>>(
        out, gate, up, num_pairs);

    return cudaPeekAtLastError();
}

// ===========================================================================
// Logit softcapping: data[i] = tanh(data[i] / cap) * cap
// ===========================================================================
__launch_bounds__(256, 1)
__global__ void softcap_kernel(
    float* __restrict__ data,
    int N,
    float cap) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float x = data[idx];
    data[idx] = tanhf(x / cap) * cap;
}

cudaError_t apply_logit_softcap(
    float* data, int N, float cap, cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((N + 255) / 256);
    softcap_kernel<<<grid, block, 0, stream>>>(data, N, cap);
    return cudaPeekAtLastError();
}

} // namespace kernels
} // namespace blackwell
