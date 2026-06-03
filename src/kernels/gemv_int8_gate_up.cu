// gemv_int8_gate_up.cu — Fused gate+up INT8 GEMV
//
// Single kernel: x_int8 × W_gate + x_int8 × W_up → gate_out, up_out
// Reads input ONCE, computes both projections. Saves one full input read (~2KB).
// For MLP: H=2048 → I=6144, two projections. ~12MB weights each.
//
// Grid: N blocks (N = gate_N = up_N), 2 warps/block.
//   Warp 0: computes gate row n
//   Warp 1: computes up row n
// Each warp does stride-32 K-block iteration, shuffle reduce.

#include <cuda_runtime.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

__launch_bounds__(64, 4)
__global__ void gemv_int8_gate_up_kernel(
    float* __restrict__ gate_out,
    float* __restrict__ up_out,
    const int8_t* __restrict__ x_int8,
    const float* __restrict__ x_scale,
    const int8_t* __restrict__ W_gate,   // [gate_N][K]
    const float* __restrict__ W_gate_sc, // [gate_N][K/16]
    const int8_t* __restrict__ W_up,     // [up_N][K]
    const float* __restrict__ W_up_sc,   // [up_N][K/16]
    int K, int N)
{
    constexpr int B = 16;
    int n_out = blockIdx.x;
    int warp_id = threadIdx.x / 32;  // 0 or 1
    int tid = threadIdx.x % 32;

    int num_K_blks = K / B;

    float acc = 0.0f;

    // Warp 0 → gate, Warp 1 → up
    const int8_t* W_t = (warp_id == 0) ? &W_gate[(size_t)n_out * K] : &W_up[(size_t)n_out * K];
    const float* W_sc = (warp_id == 0) ? &W_gate_sc[(size_t)n_out * num_K_blks] : &W_up_sc[(size_t)n_out * num_K_blks];

    for (int kb = tid; kb < num_K_blks; kb += 32) {
        // Load 16 INT8 activation values (all warps read same input — cached in L1)
        const int8_t* x_ptr = x_int8 + kb * B;
        alignas(16) int8_t x_buf[B];
        *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_ptr);

        // Load 16 INT8 weight values
        const int8_t* w_ptr = &W_t[kb * B];
        alignas(16) int8_t w_buf[B];
        *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);

        float x_sc = x_scale[kb];
        float w_sc = W_sc[kb];
        float prod_scale = w_sc * x_sc;

        const int* w32 = reinterpret_cast<const int*>(w_buf);
        const int* x32 = reinterpret_cast<const int*>(x_buf);
        int sumi = 0;
        sumi = __dp4a(w32[0], x32[0], sumi);
        sumi = __dp4a(w32[1], x32[1], sumi);
        sumi = __dp4a(w32[2], x32[2], sumi);
        sumi = __dp4a(w32[3], x32[3], sumi);
        acc += static_cast<float>(sumi) * prod_scale;
    }

    // Warp shuffle reduction
    acc += __shfl_xor_sync(0xffffffff, acc, 16);
    acc += __shfl_xor_sync(0xffffffff, acc, 8);
    acc += __shfl_xor_sync(0xffffffff, acc, 4);
    acc += __shfl_xor_sync(0xffffffff, acc, 2);
    acc += __shfl_xor_sync(0xffffffff, acc, 1);

    if (tid == 0) {
        if (warp_id == 0) gate_out[n_out] = acc;
        else              up_out[n_out] = acc;
    }
}

} // anonymous namespace

cudaError_t gemv_int8_gate_up(
    float*          gate_out,
    float*          up_out,
    const int8_t*   x_int8,
    const float*    x_scale,
    const int8_t*   W_gate,
    const float*    W_gate_sc,
    const int8_t*   W_up,
    const float*    W_up_sc,
    int             K,
    int             N,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || N % 32 != 0)
        return cudaErrorInvalidValue;

    dim3 grid(N);
    gemv_int8_gate_up_kernel<<<grid, 64, 0, stream>>>(
        gate_out, up_out, x_int8, x_scale,
        W_gate, W_gate_sc, W_up, W_up_sc, K, N);
    return cudaPeekAtLastError();
}

} // namespace kernels
} // namespace blackwell
