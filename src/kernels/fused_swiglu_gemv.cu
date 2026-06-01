// src/kernels/fused_swiglu_gemv.cu — Fused SwiGLU + gemv_int8_warp down projection
// Combines: fused_swiglu_quant + gemv_int8_warp(W_down) → 1 kernel. Saves 1 kernel/layer.
#include <cuda_runtime.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

constexpr int B = 16;   // quantization block size

__launch_bounds__(32, 8)
__global__ void fused_swiglu_gemv_kernel(
    float* __restrict__ y_out,
    int8_t* __restrict__ temp_i8,
    float* __restrict__ temp_scale,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const int8_t* __restrict__ W_t_int8,
    const float* __restrict__ W_t_scale,
    int K, int N)
{
    int n_out = blockIdx.x;
    int tid = threadIdx.x;
    int num_K_blks = K / B;

    // Phase 1: SwiGLU + INT8 quant
    for (int kb = tid; kb < num_K_blks; kb += 32) {
        const float* g_ptr = gate + kb * B;
        const float* u_ptr = up + kb * B;
        float4 g0 = *reinterpret_cast<const float4*>(&g_ptr[0]);
        float4 g1 = *reinterpret_cast<const float4*>(&g_ptr[8]);
        float4 u0 = *reinterpret_cast<const float4*>(&u_ptr[0]);
        float4 u1 = *reinterpret_cast<const float4*>(&u_ptr[8]);

        float gate_vals[B], up_vals[B];
        gate_vals[0]=g0.x; gate_vals[1]=g0.y; gate_vals[2]=g0.z; gate_vals[3]=g0.w;
        gate_vals[4]=g1.x; gate_vals[5]=g1.y; gate_vals[6]=g1.z; gate_vals[7]=g1.w;
        up_vals[0]=u0.x;   up_vals[1]=u0.y;   up_vals[2]=u0.z;   up_vals[3]=u0.w;
        up_vals[4]=u1.x;   up_vals[5]=u1.y;   up_vals[6]=u1.z;   up_vals[7]=u1.w;

        float mlp_vals[B];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float g = gate_vals[j];
            float s = 1.0f / (1.0f + expf(-g));
            mlp_vals[j] = g * s * up_vals[j];
        }

        float blk_max = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float av = fabsf(mlp_vals[j]);
            if (av > blk_max) blk_max = av;
        }

        blk_max = fmaxf(blk_max, __shfl_down_sync(0xFFFF, blk_max, 1));
        blk_max = fmaxf(blk_max, __shfl_down_sync(0xFFFF, blk_max, 2));
        blk_max = fmaxf(blk_max, __shfl_down_sync(0xFFFF, blk_max, 4));
        blk_max = fmaxf(blk_max, __shfl_down_sync(0xFFFF, blk_max, 8));
        float sc = (blk_max > 1e-9f) ? (blk_max / 127.0f) : 1e-9f;
        if (tid == 0) temp_scale[kb] = sc;
        __syncwarp();

        float sc_val = temp_scale[kb];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float qf = roundf(mlp_vals[j] / sc_val);
            qf = fminf(127.0f, fmaxf(-127.0f, qf));
            temp_i8[kb * B + j] = static_cast<int8_t>(static_cast<int>(qf));
        }
    }

    __syncthreads();

    // Phase 2: Warp-cooperative GEMV
    float acc = 0.0f;
    for (int kb = tid; kb < num_K_blks; kb += 32) {
        const int8_t* w_ptr = &W_t_int8[(size_t)n_out * K + kb * B];
        alignas(16) int8_t w_buf[B];
        *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);

        alignas(16) int8_t x_buf[B];
        *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(temp_i8 + kb * B);

        float w_sc = W_t_scale[(size_t)n_out * num_K_blks + kb];
        float x_sc = temp_scale[kb];
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

    acc += __shfl_xor_sync(0xffffffff, acc, 16);
    acc += __shfl_xor_sync(0xffffffff, acc, 8);
    acc += __shfl_xor_sync(0xffffffff, acc, 4);
    acc += __shfl_xor_sync(0xffffffff, acc, 2);
    acc += __shfl_xor_sync(0xffffffff, acc, 1);

    if (tid == 0) y_out[n_out] = acc;
}

}  // anonymous namespace

cudaError_t fused_swiglu_gemv(
    float* y_out, int8_t* temp_i8, float* temp_scale,
    const float* gate, const float* up,
    const void* W_t_int8, const float* W_t_scale,
    int K, int N, cudaStream_t stream)
{
    if (K % 16 != 0 || N % 16 != 0) return cudaErrorInvalidValue;
    fused_swiglu_gemv_kernel<<<dim3(N), dim3(32), 0, stream>>>(
        y_out, temp_i8, temp_scale, gate, up,
        static_cast<const int8_t*>(W_t_int8), W_t_scale, K, N);
    return cudaPeekAtLastError();
}

}  // namespace kernels
}  // namespace blackwell