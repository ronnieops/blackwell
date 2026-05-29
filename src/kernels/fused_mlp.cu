// src/kernels/fused_mlp.cu — Fused MLP GEMVs: gate + up projections
//
// v1: Grid(2, num_blocks) — separate blocks for gate and up.
// v2 (this): Grid(N_blocks), 512 threads/block. Each thread computes BOTH gate and up.
//   Single K-loop, shared x values, two separate accumulators.
//
// Benefits vs v1:
//   - 24 blocks instead of 48 — same SM usage, no block_type branching
//   - 512 threads/block = 2 warps — better GPU occupancy
//   - Single K-loop — x_fp4 loaded once, used for both outputs
//   - Registers: 2 accumulators (gate_acc, up_acc) instead of 1
//
// Memory pattern (transposed W):
//   gate_t[n * K + kb*16 + j] — sequential along K
//   up_t[n * K + kb*16 + j]   — same pattern

#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace blackwell {
namespace kernels {
namespace {

constexpr int kFusedMLPBlockV2 = 512;

// ---------------------------------------------------------------------------
// Fused gate + up GEMV v2: single kernel, 512 threads/block
//
// Grid: ceil(N/256) blocks, each block processes N_outputs_per_block
// Each thread computes gate[n] AND up[n] using same x values.
//
// smem: x_scale buffer for broadcast (optional optimization)
// ---------------------------------------------------------------------------
__launch_bounds__(kFusedMLPBlockV2, 1)
__global__ void fused_gate_up_v2_kernel(
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const __nv_fp4_e2m1* __restrict__ x_fp4,
    const float* __restrict__ x_scale,
    const __nv_fp4_e2m1* __restrict__ W_gate_t,
    const float* __restrict__ W_gate_scale,
    const __nv_fp4_e2m1* __restrict__ W_up_t,
    const float* __restrict__ W_up_scale,
    int K, int N)
{
    using Fp4 = __nv_fp4_e2m1;
    constexpr int B = 16;

    int tid = threadIdx.x;
    // Each thread handles 2 outputs: tid and tid + 256
    // For N=6144: 512 threads → 512 outputs (last 128 unused if N not multiple of 512)
    int n_out0 = blockIdx.x * kFusedMLPBlockV2 + tid;
    int n_out1 = n_out0 + 256;  // second output for this thread

    int num_K_blks = K / B;
    int n_blk0 = n_out0 / B;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    // K-loop: compute both outputs simultaneously
    // Load x_scale once, dequant x once, use for both gate and up
    for (int kb = 0; kb < num_K_blks; ++kb) {
        // Load x block scale (same for both outputs)
        float x_sc = x_scale[kb];

        // Dequant x (16 values) into registers — used for both gate and up
        float x_vals[B];
        #pragma unroll
        for (int j = 0; j < B; ++j) {
            x_vals[j] = static_cast<float>(x_fp4[kb * B + j]) * x_sc;
        }

        // ===== Output 0: gate =====
        if (n_out0 < N) {
            alignas(16) uint8_t buf_gate[16];
            *reinterpret_cast<uint4*>(buf_gate) = *reinterpret_cast<const uint4*>(
                &W_gate_t[n_out0 * K + kb * B]);

            float w_gate_scale = W_gate_scale[n_out0 * num_K_blks + kb];
            const Fp4* w_gate_vals = reinterpret_cast<const Fp4*>(buf_gate);

            #pragma unroll
            for (int j = 0; j < B; ++j) {
                float wv = static_cast<float>(w_gate_vals[j]) * w_gate_scale;
                gate_acc += x_vals[j] * wv;
            }
        }

        // ===== Output 1: up =====
        if (n_out1 < N) {
            alignas(16) uint8_t buf_up[16];
            *reinterpret_cast<uint4*>(buf_up) = *reinterpret_cast<const uint4*>(
                &W_up_t[n_out1 * K + kb * B]);

            int n_blk1 = n_out1 / B;
            float w_up_scale = W_up_scale[n_out1 * num_K_blks + kb];
            const Fp4* w_up_vals = reinterpret_cast<const Fp4*>(buf_up);

            #pragma unroll
            for (int j = 0; j < B; ++j) {
                float wv = static_cast<float>(w_up_vals[j]) * w_up_scale;
                up_acc += x_vals[j] * wv;
            }
        }
    }

    // Write results
    if (n_out0 < N) gate_out[n_out0] = gate_acc;
    if (n_out1 < N) up_out[n_out1] = up_acc;
}

// ---------------------------------------------------------------------------
// Fused gate + up GEMV v1 (original): Grid(2, num_blocks)
// Kept for comparison — may be faster for small N due to simpler control flow
// ---------------------------------------------------------------------------
__launch_bounds__(256, 1)
__global__ void fused_gate_up_kernel(
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const __nv_fp4_e2m1* __restrict__ x_fp4,
    const float* __restrict__ x_scale,
    const __nv_fp4_e2m1* __restrict__ W_gate_t,
    const float* __restrict__ W_gate_scale,
    const __nv_fp4_e2m1* __restrict__ W_up_t,
    const float* __restrict__ W_up_scale,
    int K, int N)
{
    constexpr int B = 16;
    int block_type = blockIdx.x;
    int tid = threadIdx.x;
    int n_out = blockIdx.y * 256 + tid;
    if (n_out >= N) return;

    int num_K_blks = K / B;
    int n_blk = n_out / B;

    const __nv_fp4_e2m1* W_t = (block_type == 0) ? W_gate_t : W_up_t;
    const float* W_scale = (block_type == 0) ? W_gate_scale : W_up_scale;
    float* out = (block_type == 0) ? gate_out : up_out;

    float acc = 0.0f;
    for (int kb = 0; kb < num_K_blks; ++kb) {
        alignas(16) uint8_t buf[16];
        *reinterpret_cast<uint4*>(buf) = *reinterpret_cast<const uint4*>(
            &W_t[n_out * K + kb * B]);

        float w_scale = W_scale[n_out * num_K_blks + kb];
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
    int             N,
    cudaStream_t    stream)
{
    using Fp4 = __nv_fp4_e2m1;

    if (K % 16 != 0 || N % 16 != 0)
        return cudaErrorInvalidValue;

    // Use v2 kernel: Grid(N/256) blocks, 512 threads/block
    // Each thread computes both gate and up outputs
    int num_blocks = (N + 255) / 256;  // ceil(N/256)
    fused_gate_up_v2_kernel<<<dim3(num_blocks), dim3(kFusedMLPBlockV2), 0, stream>>>(
        gate_out, up_out,
        static_cast<const Fp4*>(x_fp4), x_scale,
        static_cast<const Fp4*>(W_gate_t_fp4), W_gate_t_scale,
        static_cast<const Fp4*>(W_up_t_fp4), W_up_t_scale,
        K, N);

    return cudaPeekAtLastError();
}

cudaError_t fused_gate_up_gemv_v1(
    float*          gate_out,
    float*          up_out,
    const void*     x_fp4,
    const float*    x_scale,
    const void*     W_gate_t_fp4,
    const float*    W_gate_t_scale,
    const void*     W_up_t_fp4,
    const float*    W_up_t_scale,
    int             K,
    int             N,
    cudaStream_t    stream)
{
    using Fp4 = __nv_fp4_e2m1;

    if (K % 16 != 0 || N % 16 != 0)
        return cudaErrorInvalidValue;

    int num_blocks = (N + 255) / 256;
    dim3 grid(2, num_blocks);
    fused_gate_up_kernel<<<grid, 256, 0, stream>>>(
        gate_out, up_out,
        static_cast<const Fp4*>(x_fp4), x_scale,
        static_cast<const Fp4*>(W_gate_t_fp4), W_gate_t_scale,
        static_cast<const Fp4*>(W_up_t_fp4), W_up_t_scale,
        K, N);

    return cudaPeekAtLastError();
}

} // namespace kernels
} // namespace blackwell