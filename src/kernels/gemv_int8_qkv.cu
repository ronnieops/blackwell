// src/kernels/gemv_int8_qkv.cu — Fused Q/K/V GEMV kernel
//
// Combines 3 separate GEMV calls into 1 kernel launch.
// Single activation vector shared across Q/K/V projections.
// Reduces kernel launch overhead and improves GPU utilization.
//
// Build:
//   CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake --build build --parallel

#include <cuda_runtime.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {

// Fused Q/K/V GEMV kernel — computes all 3 projections in one launch
// Launch: max(N_q, N_kv) blocks, 32 threads per block
__launch_bounds__(32, 8)
__global__ void gemv_int8_qkv_kernel(
    float* __restrict__ Q_out,      // [N_q] output
    float* __restrict__ K_out,      // [N_kv] output
    float* __restrict__ V_out,      // [N_kv] output
    const int8_t* __restrict__ x_int8,
    const float* __restrict__ x_scale,
    const int8_t* __restrict__ W_q, const float* __restrict__ W_q_sc,
    const int8_t* __restrict__ W_k, const float* __restrict__ W_k_sc,
    const int8_t* __restrict__ W_v, const float* __restrict__ W_v_sc,
    int K, int N_q, int N_kv)
{
    constexpr int B = 16;
    int n_out = blockIdx.x;
    int tid = threadIdx.x;

    int num_K_blks = K / B;

    // Load activation vector once (shared across Q/K/V)
    // Each thread loads its portion at stride-32
    // Total: 32 threads × (K/32/B × B) = K bytes loaded once
    
    // Compute Q projection if this block is responsible for it
    if (n_out < N_q) {
        float acc_q = 0.0f;
        for (int kb = tid; kb < num_K_blks; kb += 32) {
            // Load 16 INT8 weight values (Q weights)
            const int8_t* w_ptr = &W_q[n_out * K + kb * B];
            alignas(16) int8_t w_buf[B];
            *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);

            // Load 16 INT8 activation values
            alignas(16) int8_t x_buf[B];
            *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_int8 + kb * B);

            float w_sc = W_q_sc[n_out * num_K_blks + kb];
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
            acc_q += static_cast<float>(sumi) * prod_scale;
        }

        // Warp shuffle reduction
        acc_q += __shfl_xor_sync(0xffffffff, acc_q, 16);
        acc_q += __shfl_xor_sync(0xffffffff, acc_q, 8);
        acc_q += __shfl_xor_sync(0xffffffff, acc_q, 4);
        acc_q += __shfl_xor_sync(0xffffffff, acc_q, 2);
        acc_q += __shfl_xor_sync(0xffffffff, acc_q, 1);

        if (tid == 0) Q_out[n_out] = acc_q;
    }

    // Compute K projection if this block is responsible for it
    if (n_out < N_kv) {
        float acc_k = 0.0f;
        for (int kb = tid; kb < num_K_blks; kb += 32) {
            // Load 16 INT8 weight values (K weights)
            const int8_t* w_ptr = &W_k[n_out * K + kb * B];
            alignas(16) int8_t w_buf[B];
            *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);

            // Load 16 INT8 activation values
            alignas(16) int8_t x_buf[B];
            *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_int8 + kb * B);

            float w_sc = W_k_sc[n_out * num_K_blks + kb];
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
            acc_k += static_cast<float>(sumi) * prod_scale;
        }

        // Warp shuffle reduction
        acc_k += __shfl_xor_sync(0xffffffff, acc_k, 16);
        acc_k += __shfl_xor_sync(0xffffffff, acc_k, 8);
        acc_k += __shfl_xor_sync(0xffffffff, acc_k, 4);
        acc_k += __shfl_xor_sync(0xffffffff, acc_k, 2);
        acc_k += __shfl_xor_sync(0xffffffff, acc_k, 1);

        if (tid == 0) K_out[n_out] = acc_k;
    }

    // Compute V projection if this block is responsible for it
    if (n_out < N_kv) {
        float acc_v = 0.0f;
        for (int kb = tid; kb < num_K_blks; kb += 32) {
            // Load 16 INT8 weight values (V weights)
            const int8_t* w_ptr = &W_v[n_out * K + kb * B];
            alignas(16) int8_t w_buf[B];
            *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);

            // Load 16 INT8 activation values
            alignas(16) int8_t x_buf[B];
            *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_int8 + kb * B);

            float w_sc = W_v_sc[n_out * num_K_blks + kb];
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
            acc_v += static_cast<float>(sumi) * prod_scale;
        }

        // Warp shuffle reduction
        acc_v += __shfl_xor_sync(0xffffffff, acc_v, 16);
        acc_v += __shfl_xor_sync(0xffffffff, acc_v, 8);
        acc_v += __shfl_xor_sync(0xffffffff, acc_v, 4);
        acc_v += __shfl_xor_sync(0xffffffff, acc_v, 2);
        acc_v += __shfl_xor_sync(0xffffffff, acc_v, 1);

        if (tid == 0) V_out[n_out] = acc_v;
    }
}

// Launch wrapper
cudaError_t gemv_int8_qkv(
    float* Q_out, float* K_out, float* V_out,
    const void* x_int8, const float* x_scale,
    const void* W_q, const float* W_q_sc,
    const void* W_k, const float* W_k_sc,
    const void* W_v, const float* W_v_sc,
    int K, int N_q, int N_kv,
    cudaStream_t stream)
{
    if (K % 16 != 0 || N_q % 16 != 0 || N_kv % 16 != 0)
        return cudaErrorInvalidValue;

    int max_n = (N_q > N_kv) ? N_q : N_kv;
    
    gemv_int8_qkv_kernel<<<dim3(max_n), dim3(32), 0, stream>>>(
        Q_out, K_out, V_out,
        static_cast<const int8_t*>(x_int8), x_scale,
        static_cast<const int8_t*>(W_q), W_q_sc,
        static_cast<const int8_t*>(W_k), W_k_sc,
        static_cast<const int8_t*>(W_v), W_v_sc,
        K, N_q, N_kv);
    
    return cudaPeekAtLastError();
}

}  // namespace kernels
}  // namespace blackwell
