// src/kernels/fused_mlp.cu — Fused MLP GEMVs: gate + up projections
//
// Computes: gate = x @ W_gate, up = x @ W_up in one kernel
// Both use same input x (2048 FP4), both output intermediate dim (6144).
// Shared K loop: each k iteration computes partial sums for both outputs.
//
// v2 optimized: same vectorized block-load pattern as gemv_fp4_v2.
// Transposed weights: W_t [N×K] row-major.

#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace blackwell {
namespace kernels {
namespace {

constexpr int kFusedMLPBlock = 256;

// ---------------------------------------------------------------------------
// Fused gate + up GEMV with vectorized block loads
//
// Grid: 2 blocks (gate, up), each with kFusedMLPBlock threads
// Each thread handles one output dimension.
// K loop: loads 16 FP4 values via uint4, dequant, accumulate for both outputs.
//
// Memory pattern (transposed W):
//   gate_t[n * K + kb*16 + j] — sequential along K for each n_out
//   up_t[n * K + kb*16 + j]   — same pattern
// ---------------------------------------------------------------------------
__launch_bounds__(kFusedMLPBlock, 1)
__global__ void fused_gate_up_kernel(
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const __nv_fp4_e2m1* __restrict__ x_fp4,
    const float* __restrict__ x_scale,
    const __nv_fp4_e2m1* __restrict__ W_gate_t,
    const float* __restrict__ W_gate_scale,
    const __nv_fp4_e2m1* __restrict__ W_up_t,
    const float* __restrict__ W_up_scale,
    int K, int N)  // N = intermediate dim (6144)
{
    constexpr int B = 16;
    int block_type = blockIdx.x;  // 0=gate, 1=up
    int tid = threadIdx.x;
    int n_out = blockIdx.y * kFusedMLPBlock + tid;
    if (n_out >= N) return;

    int num_K_blks = K / B;
    int n_blk = n_out / B;

    const __nv_fp4_e2m1* W_t = (block_type == 0) ? W_gate_t : W_up_t;
    const float* W_scale = (block_type == 0) ? W_gate_scale : W_up_scale;
    float* out = (block_type == 0) ? gate_out : up_out;

    float acc = 0.0f;
    for (int kb = 0; kb < num_K_blks; ++kb) {
        // Load 16 FP4 weight values via uint4
        alignas(16) uint8_t buf[16];
        *reinterpret_cast<uint4*>(buf) = *reinterpret_cast<const uint4*>(
            &W_t[n_out * K + kb * B]);

        float w_scale = W_scale[n_blk * num_K_blks + kb];
        float x_sc = x_scale[kb];

        const __nv_fp4_e2m1* w_vals = reinterpret_cast<const __nv_fp4_e2m1*>(buf);
        #pragma unroll
        for (int j = 0; j < B; ++j) {
            float xv = static_cast<float>(x_fp4[kb * B + j]) * x_sc;
            float wv = static_cast<float>(w_vals[j]) * w_scale;
            acc += xv * wv;
        }
    }

    out[n_out] = acc;
}

} // anonymous namespace

// ===========================================================================
// Public API
// ===========================================================================

cudaError_t fused_gate_up_gemv(
    float*          gate_out,
    float*          up_out,
    const void*     x_fp4,
    const float*    x_scale,
    const void*     W_gate_t_fp4,
    const float*    W_gate_t_scale,
    const void*     W_up_t_fp4,
    const float*    W_up_t_scale,
    int             K,
    int             N,      // output dim = intermediate (6144)
    cudaStream_t    stream)
{
    using Fp4 = __nv_fp4_e2m1;

    int num_blocks = (N + kFusedMLPBlock - 1) / kFusedMLPBlock;
    // Grid: 2 blocks for gate/up × num_blocks for output tiling
    dim3 grid(2, num_blocks);

    fused_gate_up_kernel<<<grid, kFusedMLPBlock, 0, stream>>>(
        gate_out, up_out,
        static_cast<const Fp4*>(x_fp4), x_scale,
        static_cast<const Fp4*>(W_gate_t_fp4), W_gate_t_scale,
        static_cast<const Fp4*>(W_up_t_fp4), W_up_t_scale,
        K, N);

    return cudaPeekAtLastError();
}

} // namespace kernels
} // namespace blackwell