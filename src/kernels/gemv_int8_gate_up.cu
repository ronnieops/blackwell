// src/kernels/gemv_int8_gate_up.cu — Fused gate+up GEMV kernel
//
// Combines 2 separate GEMV calls into 1 kernel launch.
// Single activation vector shared across gate/up projections.
// Expected: +5-8% improvement for M=1 decode.
//
// Build:
//   CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake --build build --parallel

#include <cuda_runtime.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {

// Fused gate+up GEMV kernel — computes both projections in one launch
// Launch: N blocks (one per output row), 32 threads per block
// gate_out and up_out have same dimensions (H→I)
__launch_bounds__(32, 8)
__global__ void gemv_int8_gate_up_kernel(
    float* __restrict__ gate_out,    // [N] output
    float* __restrict__ up_out,      // [N] output
    const int8_t* __restrict__ x_int8,
    const float* __restrict__ x_scale,
    const int8_t* __restrict__ W_gate, const float* __restrict__ W_gate_sc,
    const int8_t* __restrict__ W_up, const float* __restrict__ W_up_sc,
    int K, int N)
{
    constexpr int B = 16;
    int n_out = blockIdx.x;
    int tid = threadIdx.x;

    int num_K_blks = K / B;

    // Compute gate projection
    float acc_gate = 0.0f;
    for (int kb = tid; kb < num_K_blks; kb += 32) {
        // Load 16 INT8 weight values (gate weights)
        const int8_t* w_ptr = &W_gate[n_out * K + kb * B];
        alignas(16) int8_t w_buf[B];
        *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);

        // Load 16 INT8 activation values
        alignas(16) int8_t x_buf[B];
        *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_int8 + kb * B);

        float w_sc = W_gate_sc[n_out * num_K_blks + kb];
        float x_sc = x_scale[kb];
        float prod_scale = w_sc * x_sc;

        // dp4a dot product
        const int* w32 = reinterpret_cast<const int*>(w_buf);
        const int* x32 = reinterpret_cast<const int*>(x_buf);
        int sumi = 0;
        sumi = __dp4a(w32[0], x32[0], sumi);
        sumi = __dp4a(w32[1], x32[1], sumi);
        sumi = __dp4a(w32[2], x32[2], sumi);
        sumi = __dp4a(w32[3], x32[3], sumi);
        acc_gate += static_cast<float>(sumi) * prod_scale;
    }

    // Warp shuffle reduction for gate
    acc_gate += __shfl_xor_sync(0xffffffff, acc_gate, 16);
    acc_gate += __shfl_xor_sync(0xffffffff, acc_gate, 8);
    acc_gate += __shfl_xor_sync(0xffffffff, acc_gate, 4);
    acc_gate += __shfl_xor_sync(0xffffffff, acc_gate, 2);
    acc_gate += __shfl_xor_sync(0xffffffff, acc_gate, 1);

    if (tid == 0) gate_out[n_out] = acc_gate;

    // Compute up projection
    float acc_up = 0.0f;
    for (int kb = tid; kb < num_K_blks; kb += 32) {
        // Load 16 INT8 weight values (up weights)
        const int8_t* w_ptr = &W_up[n_out * K + kb * B];
        alignas(16) int8_t w_buf[B];
        *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);

        // Load 16 INT8 activation values (same as gate)
        alignas(16) int8_t x_buf[B];
        *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_int8 + kb * B);

        float w_sc = W_up_sc[n_out * num_K_blks + kb];
        float x_sc = x_scale[kb];
        float prod_scale = w_sc * x_sc;

        // dp4a dot product
        const int* w32 = reinterpret_cast<const int*>(w_buf);
        const int* x32 = reinterpret_cast<const int*>(x_buf);
        int sumi = 0;
        sumi = __dp4a(w32[0], x32[0], sumi);
        sumi = __dp4a(w32[1], x32[1], sumi);
        sumi = __dp4a(w32[2], x32[2], sumi);
        sumi = __dp4a(w32[3], x32[3], sumi);
        acc_up += static_cast<float>(sumi) * prod_scale;
    }

    // Warp shuffle reduction for up
    acc_up += __shfl_xor_sync(0xffffffff, acc_up, 16);
    acc_up += __shfl_xor_sync(0xffffffff, acc_up, 8);
    acc_up += __shfl_xor_sync(0xffffffff, acc_up, 4);
    acc_up += __shfl_xor_sync(0xffffffff, acc_up, 2);
    acc_up += __shfl_xor_sync(0xffffffff, acc_up, 1);

    if (tid == 0) up_out[n_out] = acc_up;
}

// Launch wrapper
cudaError_t gemv_int8_gate_up(
    float* gate_out, float* up_out,
    const void* x_int8, const float* x_scale,
    const void* W_gate, const float* W_gate_sc,
    const void* W_up, const float* W_up_sc,
    int K, int N,
    cudaStream_t stream)
{
    if (K % 16 != 0 || N % 16 != 0)
        return cudaErrorInvalidValue;

    gemv_int8_gate_up_kernel<<<dim3(N), dim3(32), 0, stream>>>(
        gate_out, up_out,
        static_cast<const int8_t*>(x_int8), x_scale,
        static_cast<const int8_t*>(W_gate), W_gate_sc,
        static_cast<const int8_t*>(W_up), W_up_sc,
        K, N);
    
    return cudaPeekAtLastError();
}

}  // namespace kernels
}  // namespace blackwell
