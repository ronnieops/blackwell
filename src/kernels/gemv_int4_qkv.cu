// src/kernels/gemv_int4_qkv.cu — Fused Q/K/V GEMV with inline INT4 quantization
//
// One block per output row, 32 threads (1 warp). Stride-32 loop over K-blocks.
// Each K-block: load x_fp32, warp-reduce scale, quantize, store to smem.
// After all K-blocks quantized: do Q, K, V GEMV using smem x_i4 + x_scale.
//
// Grid: max(N_q, N_kv) blocks. Block n: Q if n < N_q, K/V if n < N_kv.
//
// Build: CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake --build build --parallel

#include <cuda_runtime.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

constexpr int B  = 16;   // quantization block size
constexpr int PB = 8;    // packed bytes per block (B/2)

// Warp-reduce sum (32 lanes)
__device__ __forceinline__ float warp_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16);
    v += __shfl_xor_sync(0xffffffff, v, 8);
    v += __shfl_xor_sync(0xffffffff, v, 4);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    v += __shfl_xor_sync(0xffffffff, v, 1);
    return v;
}

// Scalar INT4 dot product: 8 bytes → 16 values → sum of products
__device__ __forceinline__ float int4_dot_8bytes(uint2 a, uint2 b) {
    const uint8_t* ab = reinterpret_cast<const uint8_t*>(&a);
    const uint8_t* bb = reinterpret_cast<const uint8_t*>(&b);
    float s = 0.0f;
    #pragma unroll
    for (int j = 0; j < PB; ++j) {
        int alo = ab[j] & 0x0F; if (alo > 7) alo -= 16;
        int ahi = (ab[j] >> 4) & 0x0F; if (ahi > 7) ahi -= 16;
        int blo = bb[j] & 0x0F; if (blo > 7) blo -= 16;
        int bhi = (bb[j] >> 4) & 0x0F; if (bhi > 7) bhi -= 16;
        s += (float)alo * (float)blo + (float)ahi * (float)bhi;
    }
    return s;
}

__launch_bounds__(32, 8)
__global__ void fused_qkv_int4_kernel(
    float* __restrict__ Q_out,
    float* __restrict__ K_out,
    float* __restrict__ V_out,
    const float* __restrict__ x_fp32,
    float* __restrict__ x_scale_out,
    const uint8_t* W_q_packed, const float* W_q_scale,
    const uint8_t* W_k_packed, const float* W_k_scale,
    const uint8_t* W_v_packed, const float* W_v_scale,
    int K, int N_q, int N_kv)
{
    int n_out = blockIdx.x;
    int tid   = threadIdx.x;   // 0..31

    int num_K_blks = K / B;    // 128 for K=2048

    // smem layout (byte addressed):
    //   [0 .. num_K_blks*PB)      = x_i4_shared  (1024 bytes)
    //   [num_K_blks*PB .. end)    = x_scale_shared (512 bytes, float-aligned)
    extern __shared__ char smem[];
    uint8_t* x_i4_shared  = reinterpret_cast<uint8_t*>(smem);
    float*   x_sc_shared  = reinterpret_cast<float*>(smem + num_K_blks * PB);

    // ── Phase 1: Quantize x_fp32 → x_i4 in smem ──────────────────────
    // Stride-32: thread tid owns K-blocks tid, tid+32, tid+64, ...
    for (int kb = tid; kb < num_K_blks; kb += 32) {
        int off = kb * B;
        float4 v0 = *reinterpret_cast<const float4*>(&x_fp32[off]);
        float4 v1 = *reinterpret_cast<const float4*>(&x_fp32[off + 4]);
        float4 v2 = *reinterpret_cast<const float4*>(&x_fp32[off + 8]);
        float4 v3 = *reinterpret_cast<const float4*>(&x_fp32[off + 12]);

        // absmax over 16 elements
        float mx = 0.0f;
        mx = fmaxf(mx, fabsf(v0.x)); mx = fmaxf(mx, fabsf(v0.y));
        mx = fmaxf(mx, fabsf(v0.z)); mx = fmaxf(mx, fabsf(v0.w));
        mx = fmaxf(mx, fabsf(v1.x)); mx = fmaxf(mx, fabsf(v1.y));
        mx = fmaxf(mx, fabsf(v1.z)); mx = fmaxf(mx, fabsf(v1.w));
        mx = fmaxf(mx, fabsf(v2.x)); mx = fmaxf(mx, fabsf(v2.y));
        mx = fmaxf(mx, fabsf(v2.z)); mx = fmaxf(mx, fabsf(v2.w));
        mx = fmaxf(mx, fabsf(v3.x)); mx = fmaxf(mx, fabsf(v3.y));
        mx = fmaxf(mx, fabsf(v3.z)); mx = fmaxf(mx, fabsf(v3.w));

        // Each thread computes its own K-block scale (no warp reduce needed)
        float sc = (mx > 1e-10f) ? (mx / 7.0f) : (1.0f / 7.0f);
        x_sc_shared[kb] = sc;

        // Quantize 16 floats → 8 bytes
        int q0  = max(-8, min(7, (int)roundf(v0.x / sc)));
        int q1  = max(-8, min(7, (int)roundf(v0.y / sc)));
        int q2  = max(-8, min(7, (int)roundf(v0.z / sc)));
        int q3  = max(-8, min(7, (int)roundf(v0.w / sc)));
        int q4  = max(-8, min(7, (int)roundf(v1.x / sc)));
        int q5  = max(-8, min(7, (int)roundf(v1.y / sc)));
        int q6  = max(-8, min(7, (int)roundf(v1.z / sc)));
        int q7  = max(-8, min(7, (int)roundf(v1.w / sc)));
        int q8  = max(-8, min(7, (int)roundf(v2.x / sc)));
        int q9  = max(-8, min(7, (int)roundf(v2.y / sc)));
        int q10 = max(-8, min(7, (int)roundf(v2.z / sc)));
        int q11 = max(-8, min(7, (int)roundf(v2.w / sc)));
        int q12 = max(-8, min(7, (int)roundf(v3.x / sc)));
        int q13 = max(-8, min(7, (int)roundf(v3.y / sc)));
        int q14 = max(-8, min(7, (int)roundf(v3.z / sc)));
        int q15 = max(-8, min(7, (int)roundf(v3.w / sc)));

        uint8_t b0 = ((q0+8)&0xF) | (((q1+8)&0xF)<<4);
        uint8_t b1 = ((q2+8)&0xF) | (((q3+8)&0xF)<<4);
        uint8_t b2 = ((q4+8)&0xF) | (((q5+8)&0xF)<<4);
        uint8_t b3 = ((q6+8)&0xF) | (((q7+8)&0xF)<<4);
        uint8_t b4 = ((q8+8)&0xF) | (((q9+8)&0xF)<<4);
        uint8_t b5 = ((q10+8)&0xF) | (((q11+8)&0xF)<<4);
        uint8_t b6 = ((q12+8)&0xF) | (((q13+8)&0xF)<<4);
        uint8_t b7 = ((q14+8)&0xF) | (((q15+8)&0xF)<<4);

        int boff = kb * PB;
        x_i4_shared[boff+0] = b0;
        x_i4_shared[boff+1] = b1;
        x_i4_shared[boff+2] = b2;
        x_i4_shared[boff+3] = b3;
        x_i4_shared[boff+4] = b4;
        x_i4_shared[boff+5] = b5;
        x_i4_shared[boff+6] = b6;
        x_i4_shared[boff+7] = b7;
    }

    __syncthreads();  // all K-blocks quantized before GEMV reads

    // ── Phase 2: Q/K/V GEMV using x_i4 from smem ────────────────────

    // Q projection
    if (n_out < N_q) {
        float acc = 0.0f;
        for (int kb = tid; kb < num_K_blks; kb += 32) {
            const uint8_t* wp = &W_q_packed[(size_t)n_out * (K/2) + kb * PB];
            uint2 wv = *reinterpret_cast<const uint2*>(wp);
            uint2 xv = *reinterpret_cast<const uint2*>(&x_i4_shared[kb * PB]);
            float wsc = W_q_scale[(size_t)n_out * num_K_blks + kb];
            float xsc = x_sc_shared[kb];
            acc += int4_dot_8bytes(wv, xv) * wsc * xsc;
        }
        acc = warp_sum(acc);
        if (tid == 0) Q_out[n_out] = acc;
    }

    // K projection
    if (n_out < N_kv) {
        float acc = 0.0f;
        for (int kb = tid; kb < num_K_blks; kb += 32) {
            const uint8_t* wp = &W_k_packed[(size_t)n_out * (K/2) + kb * PB];
            uint2 wv = *reinterpret_cast<const uint2*>(wp);
            uint2 xv = *reinterpret_cast<const uint2*>(&x_i4_shared[kb * PB]);
            float wsc = W_k_scale[(size_t)n_out * num_K_blks + kb];
            float xsc = x_sc_shared[kb];
            acc += int4_dot_8bytes(wv, xv) * wsc * xsc;
        }
        acc = warp_sum(acc);
        if (tid == 0) K_out[n_out] = acc;
    }

    // V projection
    if (n_out < N_kv) {
        float acc = 0.0f;
        for (int kb = tid; kb < num_K_blks; kb += 32) {
            const uint8_t* wp = &W_v_packed[(size_t)n_out * (K/2) + kb * PB];
            uint2 wv = *reinterpret_cast<const uint2*>(wp);
            uint2 xv = *reinterpret_cast<const uint2*>(&x_i4_shared[kb * PB]);
            float wsc = W_v_scale[(size_t)n_out * num_K_blks + kb];
            float xsc = x_sc_shared[kb];
            acc += int4_dot_8bytes(wv, xv) * wsc * xsc;
        }
        acc = warp_sum(acc);
        if (tid == 0) V_out[n_out] = acc;
    }
}

}  // anonymous namespace

cudaError_t fused_qkv_int4(
    float* Q_out, float* K_out, float* V_out,
    const float* x_fp32, float* x_scale_out,
    const uint8_t* W_q_packed, const float* W_q_scale,
    const uint8_t* W_k_packed, const float* W_k_scale,
    const uint8_t* W_v_packed, const float* W_v_scale,
    int K, int N_q, int N_kv,
    cudaStream_t stream)
{
    if (K % 16 != 0 || N_q % 16 != 0 || N_kv % 16 != 0)
        return cudaErrorInvalidValue;

    int num_K_blks = K / B;
    size_t smem_size = num_K_blks * PB + num_K_blks * sizeof(float);
    smem_size = (smem_size + 3u) & ~3u;  // 4-byte align

    dim3 grid(max(N_q, N_kv));
    fused_qkv_int4_kernel<<<grid, dim3(32), smem_size, stream>>>(
        Q_out, K_out, V_out, x_fp32, x_scale_out,
        W_q_packed, W_q_scale, W_k_packed, W_k_scale, W_v_packed, W_v_scale,
        K, N_q, N_kv);

    return cudaPeekAtLastError();
}

}  // namespace kernels
}  // namespace blackwell
