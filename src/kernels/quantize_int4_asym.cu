// quantize_int4_asym.cu — Asymmetric INT4 quantization with zero point
//
// Per-block asymmetric: for 16-element block, compute min, max, then:
//   scale = (max - min) / 15.0f
//   zero  = round(-min / scale)  clipped to [0..15]
//   nib   = round(x / scale) + zero  clipped to [0..15]
//
// Output: x_out_packed [K/2] bytes (2 nibbles/byte, offset-binary with zero)
//         x_out_sc_zero [2 * K/16] floats: even=scale, odd=zero (as float)
//
// For dequant: val = (nib - zero) * scale

#include <cuda_runtime.h>
#include <cstdint>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

constexpr int kBlockSize = 16;
constexpr int kPerBlock = 8;  // bytes per block (2 nibbles/byte)

__global__ void quantize_int4_asym_kernel(
    uint8_t* __restrict__ out_packed,   // [K/2] bytes
    float* __restrict__ out_sc_zero,    // [2 * K/16] floats
    const float* __restrict__ in_fp32,  // [K]
    int K)
{
    int blk = blockIdx.x;          // block index (0..K/16-1)
    int tid = threadIdx.x;         // lane (0..31)
    int num_blocks = K / kBlockSize;
    if (blk >= num_blocks) return;

    int base = blk * kBlockSize;

    // Shared memory for block reduction
    __shared__ float s_min[32], s_max[32];

    // Each thread loads 2 elements and finds local min/max
    float local_min = 1e38f, local_max = -1e38f;
    int idx1 = base + tid;
    int idx2 = base + tid + 32;
    if (idx1 < K) {
        float v1 = in_fp32[idx1];
        local_min = fminf(local_min, v1);
        local_max = fmaxf(local_max, v1);
    }
    if (idx2 < K) {
        float v2 = in_fp32[idx2];
        local_min = fminf(local_min, v2);
        local_max = fmaxf(local_max, v2);
    }

    // Warp reduce min/max
    for (int off = 16; off > 0; off >>= 1) {
        local_min = fminf(local_min, __shfl_xor_sync(0xffffffff, local_min, off));
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));
    }

    if ((tid & 31) == 0) {
        s_min[tid >> 5] = local_min;
        s_max[tid >> 5] = local_max;
    }
    __syncthreads();

    // Reduce across warps (2 warps max for block-16)
    float blk_min = s_min[0], blk_max = s_max[0];
    if (blockDim.x > 32) {
        // Only if we have 2 warps (unlikely for 32-thread block)
        // For a 32-thread block, we have 1 warp. Default launch:
        // threads = min(K, 256) but for block-16, K/16 blocks.
        // Each block handles 16 elements, 32 threads is fine.
        // Actually let's think: grid = (K/16) blocks, each with 32 threads.
        // 32 threads handle 16 elements, so only one warp is active.
        // s_min[0] is the result. No cross-warp reduction needed.
    }

    // Compute scale + zero
    float scale = (blk_max - blk_min) / 15.0f;
    if (scale < 1e-9f) scale = 1e-9f;
    float zf = -blk_min / scale;
    int zero = __float2int_rn(zf);
    zero = max(0, min(15, zero));

    // Store scale + zero
    if (tid < 2) {
        int idx = blk * 2 + tid;
        out_sc_zero[idx] = (tid == 0) ? scale : (float)zero;
    }

    // Pack INT4 nibbles
    if (tid < kPerBlock) {
        int byte_idx = blk * kPerBlock + tid;
        int elem0 = base + tid * 2;
        int elem1 = base + tid * 2 + 1;

        float v0 = (elem0 < K) ? in_fp32[elem0] : 0.0f;
        float v1 = (elem1 < K) ? in_fp32[elem1] : 0.0f;

        int nib0 = __float2int_rn(v0 / scale) + zero;
        int nib1 = __float2int_rn(v1 / scale) + zero;
        nib0 = max(0, min(15, nib0));
        nib1 = max(0, min(15, nib1));

        out_packed[byte_idx] = (uint8_t)(nib0 | (nib1 << 4));
    }
}

} // anonymous namespace

cudaError_t quantize_int4_asym(
    uint8_t*        x_out_packed,
    float*          x_out_sc_zero,    // [2 * K/16] scale,zero pairs
    const float*    in_fp32,
    int             K,
    cudaStream_t    stream)
{
    if (K % kBlockSize != 0)
        return cudaErrorInvalidValue;

    int num_blocks = K / kBlockSize;
    quantize_int4_asym_kernel<<<num_blocks, 32, 0, stream>>>(
        x_out_packed, x_out_sc_zero, in_fp32, K);
    return cudaPeekAtLastError();
}

} // namespace kernels
} // namespace blackwell