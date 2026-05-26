// src/kernels/fused_decode.cu — Fused decode kernel: Q, K, V projections
// Computes: Q = x @ W_q, K = x @ W_k, V = x @ W_v in one kernel
// All three use the same input vector x (16384 float values, 2048 FP4)
// Weight matrices: W_q (2048×2048), W_k (2048×512), W_v (2048×512)
// All weights stored as FP4 E2M1 with per-16 block scales.
//
// Strategy:
//   Grid = (8, 8) for Q, (8, 0) for K/V (2 blocks side-by-side)
//   Each block handles 256 outputs: 256 for Q, 256 for K, 256 for V
//   Shared memory: W tiles loaded from FP4 smem → dequant to FP16
//   Each thread: accumulate dot product over K=2048 in float
//   Tile K in 64-element chunks (using same WMMA tile size from GEMM)
//
// Fused kernel eliminates 2 kernel launches per layer (was 5 GEMVs → 2 GEMVs + 1 fused)

#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace blackwell {
namespace kernels {
namespace {

// Fused QKV GEMV kernel
// Grid: (3, num_layers)  — block 0 = Q, block 1 = K, block 2 = V
// Each block: 256 threads, handles 256 output dimensions
__launch_bounds__(256, 1)
__global__ void fused_qkv_kernel(
    float* __restrict__ Q_out,     // [head_dim * num_q_heads]
    float* __restrict__ K_out,     // [head_dim * num_kv_heads]
    float* __restrict__ V_out,     // [head_dim * num_kv_heads]
    const __nv_fp4_e2m1* __restrict__ x_fp4,
    const float* __restrict__ x_scale,
    const __nv_fp4_e2m1* __restrict__ W_q,   // [hidden × q_dim]
    const float* __restrict__ W_q_scale,
    const __nv_fp4_e2m1* __restrict__ W_k,   // [hidden × kv_dim]
    const float* __restrict__ W_k_scale,
    const __nv_fp4_e2m1* __restrict__ W_v,   // [hidden × kv_dim]
    const float* __restrict__ W_v_scale,
    int hidden, int q_dim, int kv_dim)
{
    int block_type = blockIdx.x;  // 0=Q, 1=K, 2=V
    int tid = threadIdx.x;
    int n_out_this = (block_type == 0) ? q_dim : kv_dim;
    int out_base = block_type * n_out_this;

    // Each thread handles one output element
    // Load x into registers (FP4 dequant per element, 2048×)
    // Each thread: stride=256 over K, accumulate dot
    float acc = 0.0f;

    int out_offset = tid;  // this thread's output index within its block
    if (out_offset >= n_out_this) return;

    const __nv_fp4_e2m1* W;
    const float* W_scale;
    int width;
    int num_N_blks;
    if (block_type == 0) {
        W = W_q; W_scale = W_q_scale; width = q_dim;
    } else if (block_type == 1) {
        W = W_k; W_scale = W_k_scale; width = kv_dim;
    } else {
        W = W_v; W_scale = W_v_scale; width = kv_dim;
    }
    num_N_blks = (width + 15) / 16;
    int out_blk = out_offset / 16;

    // K loop over hidden dimension (2048)
    for (int k = 0; k < hidden; ++k) {
        int k_blk = k / 16;
        float xv = static_cast<float>(x_fp4[k]) * x_scale[k_blk];
        float wv = static_cast<float>(W[k * width + out_offset])
                 * W_scale[k_blk * num_N_blks + out_blk];
        acc += xv * wv;
    }

    float* out = (block_type == 0) ? Q_out : (block_type == 1) ? K_out : V_out;
    out[out_offset] = acc;
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

    dim3 grid(3); // Q, K, V
    fused_qkv_kernel<<<grid, 256, 0, stream>>>(
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
