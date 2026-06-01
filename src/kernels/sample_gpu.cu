// src/kernels/sample_gpu.cu — GPU-side logit sampling
//
// Replaces 607 KB cudaMemcpy per token with GPU argmax (4-byte copy).
// Greedy decoding: single-thread find max, 4-byte output.
//
// Currently supports:
//   - argmax (deterministic, temperature < 0.01)
//   - temperature > 0: host fallback (softmax kernel pending fix)
//
// GPU softmax + weighted random select: WIP, not yet working.

#include <cuda_runtime.h>
#include <cstdint>
#include <curand_kernel.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

// ── Argmax kernel ───────────────────────────────────────────────────────────
// Warp-cooperative max-reduce: 256 threads scan vocab, find argmax token.
// 4-byte output copy (not 607 KB).
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

    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, bv, off);
        int oi = __shfl_xor_sync(0xffffffff, bi, off);
        if (ov > bv) { bv = ov; bi = oi; }
    }

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

// ── Public API ──────────────────────────────────────────────────────────────

cudaError_t sample_argmax_gpu(
    const float* logits, int vocab, int* out_id, cudaStream_t stream) {
    argmax_kernel<<<1, 256, 0, stream>>>(logits, vocab, out_id);
    return cudaGetLastError();
}

cudaError_t sample_gpu(
    const float*  logits,
    int           vocab,
    float         temperature,
    int           top_k,
    int*          out_id,
    unsigned long long rng_seed,
    int           step,
    cudaStream_t  stream) {
    // Currently: GPU argmax for greedy decoding only.
    // Temperature + top-k: host fallback.
    // GPU softmax + weighted random select: pending.
    argmax_kernel<<<1, 256, 0, stream>>>(logits, vocab, out_id);
    return cudaGetLastError();
}

} // namespace kernels
} // namespace blackwell