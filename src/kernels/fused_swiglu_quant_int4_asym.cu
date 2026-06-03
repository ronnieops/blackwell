// fused_swiglu_quant_int4_asym.cu — Fused SwiGLU + ASYMMETRIC INT4 quant
//
// Same as fused_swiglu_quant_int4 but asymmetric (min/max + zero) format.
// Output: out_packed [N/2] bytes, out_sc_zero [2*N/16] floats (scale,zero pairs)

#include <cuda_runtime.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

__launch_bounds__(16, 16)
__global__ void fused_swiglu_quant_int4_asym_kernel(
    uint8_t* __restrict__ out_packed,
    float* __restrict__ out_sc_zero,   // [2*N/16] scale,zero pairs
    const float* __restrict__ gate,
    const float* __restrict__ up,
    int N)
{
    constexpr int B = 16;
    int blk = blockIdx.x;
    int tid = threadIdx.x;

    int elem_off = blk * B;

    // Load gate and up values, compute SwiGLU
    float g0 = gate[elem_off + tid];
    float u0 = up[elem_off + tid];
    float s = 1.0f / (1.0f + expf(-g0));
    float v = g0 * s * u0;

    // Compute min and max across block
    float blk_min = v;
    float blk_max = v;
    blk_min = fminf(blk_min, __shfl_xor_sync(0xffffffff, blk_min, 1));
    blk_max = fmaxf(blk_max, __shfl_xor_sync(0xffffffff, blk_max, 1));
    blk_min = fminf(blk_min, __shfl_xor_sync(0xffffffff, blk_min, 2));
    blk_max = fmaxf(blk_max, __shfl_xor_sync(0xffffffff, blk_max, 2));
    blk_min = fminf(blk_min, __shfl_xor_sync(0xffffffff, blk_min, 4));
    blk_max = fmaxf(blk_max, __shfl_xor_sync(0xffffffff, blk_max, 4));
    blk_min = fminf(blk_min, __shfl_xor_sync(0xffffffff, blk_min, 8));
    blk_max = fmaxf(blk_max, __shfl_xor_sync(0xffffffff, blk_max, 8));

    if (tid == 0) {
        float scale = (blk_max - blk_min) / 15.0f;
        if (scale < 1e-9f) scale = 1e-9f;
        int zero = (int)roundf(-blk_min / scale);
        zero = max(0, min(15, zero));
        out_sc_zero[blk * 2 + 0] = scale;
        out_sc_zero[blk * 2 + 1] = (float)zero;
    }
    __syncthreads();

    float scale = out_sc_zero[blk * 2];
    float zf = out_sc_zero[blk * 2 + 1];
    int zero = (int)zf;

    // Quantize + pack (8 threads handle 8 bytes)
    if (tid < 8) {
        float v_pair = __shfl_xor_sync(0xffffffff, v, 1);

        if (tid % 2 == 0) {
            int q0 = (int)roundf(v / scale) + zero;
            q0 = max(0, min(15, q0));
            int q1 = (int)roundf(v_pair / scale) + zero;
            q1 = max(0, min(15, q1));
            uint8_t packed = (uint8_t)(q0 | (q1 << 4));
            out_packed[blk * 8 + tid / 2] = packed;
        }
    }
}

} // anonymous namespace

cudaError_t fused_swiglu_quant_int4_asym(
    uint8_t* out_packed,
    float* out_sc_zero,
    const float* gate,
    const float* up,
    int N,
    cudaStream_t stream)
{
    if (N % 16 != 0) return cudaErrorInvalidValue;

    int num_blocks = N / 16;
    fused_swiglu_quant_int4_asym_kernel<<<num_blocks, 16, 0, stream>>>(
        out_packed, out_sc_zero, gate, up, N);
    return cudaPeekAtLastError();
}

} // namespace kernels
} // namespace blackwell