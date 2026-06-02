// fused_quant_attn_wo.cu — Quantize attention output + Wo GEMV in 1 kernel
//
// Combines: quantize_int4(attn) + gemv_int4_warp(attn_i4, Wo)
// Saves: 1 kernel launch + device memory write/read for attn_i4.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

// Unpack 1 byte (2 INT4 nibbles) to 2 floats
__device__ __forceinline__ void int4_byte_to_floats(uint8_t b, float& f0, float& f1) {
    int lo = b & 0x0F; if (lo > 7) lo -= 16;
    int hi = (b >> 4) & 0x0F; if (hi > 7) hi -= 16;
    f0 = static_cast<float>(lo);
    f1 = static_cast<float>(hi);
}

// Load scale as __half from char* base, convert to float
__device__ __forceinline__ float load_half_as_float(const char* base, int byte_off) {
    __half h = *reinterpret_cast<const __half*>(&base[byte_off]);
    return __half2float(h);
}

__launch_bounds__(32, 8)
__global__ void fused_quant_attn_wo_kernel(
    float* __restrict__ proj_out,
    const float* __restrict__ attn,
    const uint8_t* __restrict__ Wo_packed,
    const float* __restrict__ Wo_scale,
    int Q, int H)
{
    constexpr int B = 16, PB = 8;
    int n_out = blockIdx.x;
    int tid = threadIdx.x;
    int num_K_blks = Q / B;

    // smem: char*-based
    // [0..Q/2-1] = attn_i4 (Q/2 bytes)
    // [Q/2..Q/2+Q/16*2-1] = attn_sc (Q/16 halves × 2 bytes)
    extern __shared__ char smem_[];
    uint8_t* attn_i4_s = reinterpret_cast<uint8_t*>(smem_);
    char*    attn_sc_s  = smem_ + (Q / 2);

    // Phase 1: quantize attn → smem
    for (int kb = tid; kb < num_K_blks; kb += 32) {
        int off = kb * B;
        float4 v0 = reinterpret_cast<const float4*>(&attn[off])[0];
        float4 v1 = reinterpret_cast<const float4*>(&attn[off + 4])[0];
        float4 v2 = reinterpret_cast<const float4*>(&attn[off + 8])[0];
        float4 v3 = reinterpret_cast<const float4*>(&attn[off + 12])[0];

        float mx = fabsf(v0.x); mx = fmaxf(mx, fabsf(v0.y));
        mx = fmaxf(mx, fabsf(v0.z)); mx = fmaxf(mx, fabsf(v0.w));
        mx = fmaxf(mx, fabsf(v1.x)); mx = fmaxf(mx, fabsf(v1.y));
        mx = fmaxf(mx, fabsf(v1.z)); mx = fmaxf(mx, fabsf(v1.w));
        mx = fmaxf(mx, fabsf(v2.x)); mx = fmaxf(mx, fabsf(v2.y));
        mx = fmaxf(mx, fabsf(v2.z)); mx = fmaxf(mx, fabsf(v2.w));
        mx = fmaxf(mx, fabsf(v3.x)); mx = fmaxf(mx, fabsf(v3.y));
        mx = fmaxf(mx, fabsf(v3.z)); mx = fmaxf(mx, fabsf(v3.w));

        float sc = (mx > 1e-10f) ? (mx / 7.0f) : (1.0f / 7.0f);
        *reinterpret_cast<__half*>(&attn_sc_s[kb * 2]) = __float2half(sc);

        int q0 = max(-8, min(7, (int)roundf(v0.x / sc)));
        int q1 = max(-8, min(7, (int)roundf(v0.y / sc)));
        int q2 = max(-8, min(7, (int)roundf(v0.z / sc)));
        int q3 = max(-8, min(7, (int)roundf(v0.w / sc)));
        int q4 = max(-8, min(7, (int)roundf(v1.x / sc)));
        int q5 = max(-8, min(7, (int)roundf(v1.y / sc)));
        int q6 = max(-8, min(7, (int)roundf(v1.z / sc)));
        int q7 = max(-8, min(7, (int)roundf(v1.w / sc)));
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

        int bo = kb * PB;
        attn_i4_s[bo+0] = b0; attn_i4_s[bo+1] = b1;
        attn_i4_s[bo+2] = b2; attn_i4_s[bo+3] = b3;
        attn_i4_s[bo+4] = b4; attn_i4_s[bo+5] = b5;
        attn_i4_s[bo+6] = b6; attn_i4_s[bo+7] = b7;
    }

    __syncthreads();

    // Phase 2: Wo GEMV using smem attn
    float acc = 0.0f;
    for (int kb = tid; kb < num_K_blks; kb += 32) {
        const uint8_t* w_ptr = &Wo_packed[(size_t)n_out * (Q / 2) + kb * PB];
        uint2 w_packed = *reinterpret_cast<const uint2*>(w_ptr);

        const uint8_t* a_ptr = &attn_i4_s[kb * PB];
        uint2 a_packed = *reinterpret_cast<const uint2*>(a_ptr);

        // Load attn scale as __half from smem (written as __half in Phase 1)
        float asc = load_half_as_float(attn_sc_s, kb * 2);

        // Load Wo scale as float (reference path uses float*)
        float wsc = Wo_scale[(size_t)n_out * num_K_blks + kb];

        float prod_scale = wsc * asc;

        const uint8_t* wb = reinterpret_cast<const uint8_t*>(&w_packed);
        const uint8_t* ab = reinterpret_cast<const uint8_t*>(&a_packed);

        float sum_f = 0.0f;
        #pragma unroll
        for (int j = 0; j < PB; ++j) {
            float w0, w1, a0, a1;
            int4_byte_to_floats(wb[j], w0, w1);
            int4_byte_to_floats(ab[j], a0, a1);
            sum_f += w0 * a0 + w1 * a1;
        }
        acc += sum_f * prod_scale;
    }

    acc += __shfl_xor_sync(0xffffffff, acc, 16);
    acc += __shfl_xor_sync(0xffffffff, acc, 8);
    acc += __shfl_xor_sync(0xffffffff, acc, 4);
    acc += __shfl_xor_sync(0xffffffff, acc, 2);
    acc += __shfl_xor_sync(0xffffffff, acc, 1);

    if (tid == 0) proj_out[n_out] = acc;
}

} // anonymous namespace

cudaError_t fused_quant_attn_wo(
    float*          proj_out,
    const float*    attn,
    const uint8_t*  Wo_packed,
    const float*    Wo_scale,
    int             Q,
    int             H,
    cudaStream_t    stream)
{
    size_t smem_sz = Q / 2 + Q / 16 * 2;
    dim3 grid(H), block(32);
    fused_quant_attn_wo_kernel<<<grid, block, smem_sz, stream>>>(
        proj_out, attn, Wo_packed, Wo_scale, Q, H);
    return cudaPeekAtLastError();
}

} // namespace kernels
} // namespace blackwell