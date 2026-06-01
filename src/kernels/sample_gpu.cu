// src/kernels/sample_gpu.cu — GPU-side logit sampling
//
// Eliminates 607 KB GPU→CPU copy per token (151936 logits × 4 bytes).
// Returns token ID via cudaMemcpy (4 bytes instead of 607 KB).
//
// argmax_gpu: fastest path, no random

#include <cuda_runtime.h>
#include <cstdint>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

__launch_bounds__(256, 1)
__global__ void argmax_kernel(const float* logits, int vocab, int* out_id) {
    int tid = threadIdx.x;
    int items_per = (vocab + 255) / 256;
    int start = tid * items_per;
    int end = min(start + items_per, vocab);

    float bv = -1e38f;
    int bi = 0;
    for (int i = start; i < end; ++i) {
        if (logits[i] > bv) { bv = logits[i]; bi = i; }
    }

    // Warp max reduce — track index alongside value
    // Use 32-bit float shuffle for value, manually track best index
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, bv, off);
        int oi = __shfl_xor_sync(0xffffffff, bi, off);
        if (ov > bv) { bv = ov; bi = oi; }
    }

    // Cross-warp reduce via smem
    __shared__ int s_winner[8];
    __shared__ float s_best[8];
    if ((tid & 31) == 0) { s_winner[tid >> 5] = bi; s_best[tid >> 5] = bv; }
    __syncthreads();

    if (tid < 8) {
        bv = s_best[tid]; bi = s_winner[tid];
        for (int off = 4; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, bv, off);
            int oi = __shfl_xor_sync(0xffffffff, bi, off);
            if (ov > bv) { bv = ov; bi = oi; }
        }
        if (tid == 0) *out_id = bi;
    }
}

} // anonymous namespace

cudaError_t sample_argmax_gpu(
    const float* logits, int vocab, int* out_id, cudaStream_t stream) {
    argmax_kernel<<<1, 256, 0, stream>>>(logits, vocab, out_id);
    return cudaGetLastError();
}

} // namespace kernels
} // namespace blackwell