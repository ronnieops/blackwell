// src/kernels/fused_pack_gemv.cu — Fused pack_int8 + gemv_int8_warp output projection
// Combines: pack_int8(attn) + gemv_int8_warp(W_o) → 1 kernel. Saves 1 kernel/layer.
#include <cuda_runtime.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

constexpr int B = 16;   // quantization block size

__launch_bounds__(32, 8)
__global__ void fused_pack_gemv_o_kernel(
    float* __restrict__ y_out,
    int8_t* __restrict__ x_i8,
    float* __restrict__ x_scale,
    const float* __restrict__ x_fp32,
    const int8_t* __restrict__ W_t_int8,
    const float* __restrict__ W_t_scale,
    int K, int N)
{
    int n_out = blockIdx.x;
    int tid = threadIdx.x;
    int num_K_blks = K / B;

    // Phase 1: FP32 → INT8 quant with per-block scales
    for (int kb = tid; kb < num_K_blks; kb += 32) {
        const float* x_ptr = x_fp32 + kb * B;
        float4 v0 = *reinterpret_cast<const float4*>(&x_ptr[0]);
        float4 v1 = *reinterpret_cast<const float4*>(&x_ptr[8]);

        float blk_max = 0.0f;
        float vals[B];
        vals[0]=v0.x; vals[1]=v0.y; vals[2]=v0.z; vals[3]=v0.w;
        vals[4]=v1.x; vals[5]=v1.y; vals[6]=v1.z; vals[7]=v1.w;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float av = fabsf(vals[j]);
            if (av > blk_max) blk_max = av;
        }

        // Warp-wide absmax across 16 lanes
        blk_max = fmaxf(blk_max, __shfl_down_sync(0xFFFF, blk_max, 1));
        blk_max = fmaxf(blk_max, __shfl_down_sync(0xFFFF, blk_max, 2));
        blk_max = fmaxf(blk_max, __shfl_down_sync(0xFFFF, blk_max, 4));
        blk_max = fmaxf(blk_max, __shfl_down_sync(0xFFFF, blk_max, 8));
        float sc = (blk_max > 1e-9f) ? (blk_max / 127.0f) : 1e-9f;
        if (tid == 0) x_scale[kb] = sc;
        __syncwarp();

        float sc_val = x_scale[kb];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float qf = roundf(vals[j] / sc_val);
            qf = fminf(127.0f, fmaxf(-127.0f, qf));
            x_i8[kb * B + j] = static_cast<int8_t>(static_cast<int>(qf));
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
        *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_i8 + kb * B);

        float w_sc = W_t_scale[(size_t)n_out * num_K_blks + kb];
        float x_sc = x_scale[kb];
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

cudaError_t fused_pack_gemv_o(
    float* y_out, int8_t* temp_i8, float* temp_scale,
    const float* x_fp32, const void* W_t_int8, const float* W_t_scale,
    int K, int N, cudaStream_t stream)
{
    if (K % 16 != 0 || N % 16 != 0) return cudaErrorInvalidValue;
    fused_pack_gemv_o_kernel<<<dim3(N), dim3(32), 0, stream>>>(
        y_out, temp_i8, temp_scale, x_fp32,
        static_cast<const int8_t*>(W_t_int8), W_t_scale, K, N);
    return cudaPeekAtLastError();
}

}  // namespace kernels
}  // namespace blackwell