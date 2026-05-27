// src/kernels/fused_decode.cu — Fused decode kernel: Q, K, V projections
// Computes: Q = x @ W_q, K = x @ W_k, V = x @ W_v in one kernel
//
// Fixed: multi-block support for output dimensions > 256.
// Grid: (3, ceil(max(q_dim, kv_dim)/256))
//   blockIdx.x: 0=Q, 1=K, 2=V
//   blockIdx.y: output tile index (each tile covers 256 outputs)

#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace blackwell {
namespace kernels {
namespace {

constexpr int kFuseBlockThreads = 256;

__launch_bounds__(kFuseBlockThreads, 1)
__global__ void fused_qkv_kernel(
    float* __restrict__ Q_out,
    float* __restrict__ K_out,
    float* __restrict__ V_out,
    const __nv_fp4_e2m1* __restrict__ x_fp4,
    const float* __restrict__ x_scale,
    const __nv_fp4_e2m1* __restrict__ W_q,
    const float* __restrict__ W_q_scale,
    const __nv_fp4_e2m1* __restrict__ W_k,
    const float* __restrict__ W_k_scale,
    const __nv_fp4_e2m1* __restrict__ W_v,
    const float* __restrict__ W_v_scale,
    int hidden, int q_dim, int kv_dim)
{
    int block_type = blockIdx.x;  // 0=Q, 1=K, 2=V
    int tid = threadIdx.x;
    int out_dim = (block_type == 0) ? q_dim : kv_dim;
    int n_out = blockIdx.y * kFuseBlockThreads + tid;

    if (n_out >= out_dim) return;

    const __nv_fp4_e2m1* W;
    const float* W_scale;
    int width;
    if (block_type == 0) {
        W = W_q; W_scale = W_q_scale; width = q_dim;
    } else if (block_type == 1) {
        W = W_k; W_scale = W_k_scale; width = kv_dim;
    } else {
        W = W_v; W_scale = W_v_scale; width = kv_dim;
    }

    int num_N_blks = (width + 15) / 16;
    int n_blk = n_out / 16;

    float acc = 0.0f;
    for (int k = 0; k < hidden; ++k) {
        int k_blk = k / 16;
        float xv = static_cast<float>(x_fp4[k]) * x_scale[k_blk];
        float wv = static_cast<float>(W[k * width + n_out])
                 * W_scale[k_blk * num_N_blks + n_blk];
        acc += xv * wv;
    }

    float* out = (block_type == 0) ? Q_out : (block_type == 1) ? K_out : V_out;
    out[n_out] = acc;
}

} // anonymous namespace

cudaError_t fused_qkv_gemv(
    float* Q_out, float* K_out, float* V_out,
    const void* x_fp4, const float* x_scale,
    const void* W_q_fp4, const float* W_q_scale,
    const void* W_k_fp4, const float* W_k_scale,
    const void* W_v_fp4, const float* W_v_scale,
    int hidden, int q_dim, int kv_dim,
    cudaStream_t stream) {

    using Fp4 = __nv_fp4_e2m1;
    if (hidden % 16 != 0 || q_dim % 16 != 0 || kv_dim % 16 != 0)
        return cudaErrorInvalidValue;

    int max_dim = (q_dim > kv_dim) ? q_dim : kv_dim;
    int tiles = (max_dim + kFuseBlockThreads - 1) / kFuseBlockThreads;

    dim3 grid(3, tiles);
    fused_qkv_kernel<<<grid, kFuseBlockThreads, 0, stream>>>(
        Q_out, K_out, V_out,
        static_cast<const Fp4*>(x_fp4), x_scale,
        static_cast<const Fp4*>(W_q_fp4), W_q_scale,
        static_cast<const Fp4*>(W_k_fp4), W_k_scale,
        static_cast<const Fp4*>(W_v_fp4), W_v_scale,
        hidden, q_dim, kv_dim);

    return cudaPeekAtLastError();
}

} // namespace kernels
} // namespace blackwell
