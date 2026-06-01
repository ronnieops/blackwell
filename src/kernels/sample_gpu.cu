// src/kernels/sample_gpu.cu — GPU-side logit sampling
//
// Replaces 607 KB cudaMemcpy per token with on-device softmax + sampling.
// Supports: greedy argmax, temperature sampling, top-k filtering.
//
// Pipeline (temp > 0):
//   1. Max reduction (numerical stability)
//   2. Exp(x - max) + sum reduction
//   3. Normalize → probabilities
//   4. Top-k filter (optional, keeps top-k probs, renormalizes)
//   5. Weighted random sample via cuRAND

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <curand_kernel.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

constexpr int BS = 256;  // block size

// ═══════════════════════════════════════════════════════════════════════════
// Argmax kernel — single-pass warp-cooperative max
// ═══════════════════════════════════════════════════════════════════════════

__launch_bounds__(256, 1)
__global__ void argmax_kernel(const float* logits, int vocab, int* out_id) {
    int tid = threadIdx.x;
    int items = (vocab + 255) / 256;
    int start = tid * items;
    int end = min(start + items, vocab);

    float bv = -1e38f; int bi = 0;
    for (int i = start; i < end; ++i)
        if (logits[i] > bv) { bv = logits[i]; bi = i; }

    // Warp reduce
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, bv, off);
        int   oi = __shfl_xor_sync(0xffffffff, bi, off);
        if (ov > bv) { bv = ov; bi = oi; }
    }
    __shared__ int   s_w[8];
    __shared__ float s_b[8];
    if ((tid & 31) == 0) { s_w[tid >> 5] = bi; s_b[tid >> 5] = bv; }
    __syncthreads();
    if (tid < 8) {
        bv = s_b[tid]; bi = s_w[tid];
        for (int off = 4; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, bv, off);
            int   oi = __shfl_xor_sync(0xffffffff, bi, off);
            if (ov > bv) { bv = ov; bi = oi; }
        }
        if (tid == 0) *out_id = bi;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Softmax phase 1: block-level max reduction (float4 vectorized)
// ═══════════════════════════════════════════════════════════════════════════

__global__ void sm_block_max(const float* __restrict__ logits,
                              float* __restrict__ block_maxes, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * BS + tid;
    int stride = gridDim.x * BS;
    float lmax = -1e38f;

    // Vectorized: 4 floats per load
    int n4 = n / 4;
    const float4* L = reinterpret_cast<const float4*>(logits);
    for (int i = gid; i < n4; i += stride) {
        float4 v = L[i];
        lmax = fmaxf(lmax, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    }
    // Remainder
    for (int i = n4 * 4 + gid; i < n; i += stride)
        lmax = fmaxf(lmax, logits[i]);

    // Warp reduce
    for (int off = 16; off > 0; off >>= 1)
        lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, off));
    __shared__ float wm[8];
    if ((tid & 31) == 0) wm[tid >> 5] = lmax;
    __syncthreads();
    if (tid < 8) {
        float v = wm[tid];
        for (int off = 4; off > 0; off >>= 1)
            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
        if (tid == 0) block_maxes[blockIdx.x] = v;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Softmax phase 2: reduce block maxes → global max
// ═══════════════════════════════════════════════════════════════════════════

__global__ void sm_reduce_max(const float* bm, float* gmax, int nb) {
    int tid = threadIdx.x;
    float v = (tid < nb) ? bm[tid] : -1e38f;
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    __shared__ float wm[8];
    if ((tid & 31) == 0) wm[tid >> 5] = v;
    __syncthreads();
    if (tid < 8) {
        v = wm[tid];
        for (int off = 4; off > 0; off >>= 1)
            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
        if (tid == 0) *gmax = v;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Softmax phase 3: exp(x - max) in-place + block partial sums
// ═══════════════════════════════════════════════════════════════════════════

__global__ void sm_exp_sum(float* logits, float* ps, float gmax, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * BS + tid;
    int stride = gridDim.x * BS;
    float lsum = 0.f;

    int n4 = n / 4;
    float4* L = reinterpret_cast<float4*>(logits);
    for (int i = gid; i < n4; i += stride) {
        float4 v = L[i];
        v.x = expf(v.x - gmax); v.y = expf(v.y - gmax);
        v.z = expf(v.z - gmax); v.w = expf(v.w - gmax);
        L[i] = v;
        lsum += v.x + v.y + v.z + v.w;
    }
    for (int i = n4 * 4 + gid; i < n; i += stride) {
        float val = expf(logits[i] - gmax);
        logits[i] = val;
        lsum += val;
    }

    for (int off = 16; off > 0; off >>= 1)
        lsum += __shfl_xor_sync(0xffffffff, lsum, off);
    __shared__ float ws[8];
    if ((tid & 31) == 0) ws[tid >> 5] = lsum;
    __syncthreads();
    if (tid < 8) {
        float v = ws[tid];
        for (int off = 4; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, off);
        if (tid == 0) ps[blockIdx.x] = v;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Softmax phase 4: reduce partial sums → total sum
// ═══════════════════════════════════════════════════════════════════════════

__global__ void sm_reduce_sum(const float* ps, float* tsum, int nb) {
    int tid = threadIdx.x;
    float v = (tid < nb) ? ps[tid] : 0.f;
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    __shared__ float ws[8];
    if ((tid & 31) == 0) ws[tid >> 5] = v;
    __syncthreads();
    if (tid < 8) {
        v = ws[tid];
        for (int off = 4; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, off);
        if (tid == 0) *tsum = v;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Softmax phase 5: normalize in-place
// ═══════════════════════════════════════════════════════════════════════════

__global__ void sm_normalize(float* logits, float inv_sum, int n) {
    int gid = blockIdx.x * BS + threadIdx.x;
    int stride = gridDim.x * BS;
    int n4 = n / 4;
    float4* L = reinterpret_cast<float4*>(logits);
    for (int i = gid; i < n4; i += stride) {
        float4 v = L[i];
        v.x *= inv_sum; v.y *= inv_sum;
        v.z *= inv_sum; v.w *= inv_sum;
        L[i] = v;
    }
    for (int i = n4 * 4 + gid; i < n; i += stride)
        logits[i] *= inv_sum;
}

// ═══════════════════════════════════════════════════════════════════════════
// Top-k filter: zero out probabilities below threshold
// ═══════════════════════════════════════════════════════════════════════════

__global__ void sm_topk_mask(float* probs, float threshold, int n) {
    int gid = blockIdx.x * BS + threadIdx.x;
    int stride = gridDim.x * BS;
    for (int i = gid; i < n; i += stride)
        if (probs[i] < threshold) probs[i] = 0.f;
}

// ═══════════════════════════════════════════════════════════════════════════
// Weighted random sample via cuRAND
// ═══════════════════════════════════════════════════════════════════════════

__launch_bounds__(256, 1)
__global__ void sm_sample(const float* probs, int* out_id, int vocab,
                           unsigned long long seed, int step) {
    curandState state;
    curand_init(seed, step, 0, &state);
    float r = curand_uniform(&state);  // (0, 1]

    float cumsum = 0.f;
    for (int i = 0; i < vocab; ++i) {
        cumsum += probs[i];
        if (cumsum >= r) { *out_id = i; return; }
    }
    *out_id = vocab - 1;
}

// ═══════════════════════════════════════════════════════════════════════════
// Workspace (cached across calls)
// ═══════════════════════════════════════════════════════════════════════════

struct SmWs {
    float *bm, *ps, *gm, *ts;
    int cap;
    SmWs() : bm(nullptr), ps(nullptr), gm(nullptr), ts(nullptr), cap(0) {}
    void ensure(int nb) {
        if (cap >= nb) return;
        if (bm) { cudaFree(bm); cudaFree(ps); cudaFree(gm); cudaFree(ts); }
        cudaMalloc(&bm, nb * sizeof(float));
        cudaMalloc(&ps, nb * sizeof(float));
        cudaMalloc(&gm, sizeof(float));
        cudaMalloc(&ts, sizeof(float));
        cap = nb;
    }
};

} // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════

cudaError_t sample_argmax_gpu(
    const float* logits, int vocab, int* out_id, cudaStream_t stream) {
    argmax_kernel<<<1, BS, 0, stream>>>(logits, vocab, out_id);
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

    // ── Greedy path: argmax ────────────────────────────────────────────
    if (temperature < 0.01f) {
        argmax_kernel<<<1, BS, 0, stream>>>(logits, vocab, out_id);
        return cudaGetLastError();
    }

    // ── Temperature sampling path ──────────────────────────────────────
    // Use logits buffer as scratch — softmax computed in-place.
    // Caller must not read logits after this call.
    float* probs = const_cast<float*>(logits);

    static SmWs ws;
    int nb = (vocab + BS * 4 - 1) / (BS * 4);  // ~149 for 151K vocab
    ws.ensure(nb);

    // Phase 1: find global max
    sm_block_max<<<nb, BS, 0, stream>>>(probs, ws.bm, vocab);
    sm_reduce_max<<<1, BS, 0, stream>>>(ws.bm, ws.gm, nb);

    // Read global max (sync required — kernel arg must be host value)
    float gmax;
    cudaError_t e = cudaMemcpy(&gmax, ws.gm, sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return e;

    // Phase 2: exp(x - max) + partial sums
    sm_exp_sum<<<nb, BS, 0, stream>>>(probs, ws.ps, gmax, vocab);
    sm_reduce_sum<<<1, BS, 0, stream>>>(ws.ps, ws.ts, nb);

    // Read total sum
    float tsum;
    e = cudaMemcpy(&tsum, ws.ts, sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) return e;
    float inv_sum = 1.0f / fmaxf(tsum, 1e-20f);

    // Phase 3: normalize → probabilities
    sm_normalize<<<nb, BS, 0, stream>>>(probs, inv_sum, vocab);

    // Phase 4: top-k filter (optional)
    if (top_k > 0 && top_k < vocab) {
        // Find k-th largest probability via partial sort on host
        std::vector<float> h_probs(vocab);
        e = cudaMemcpy(h_probs.data(), probs, vocab * sizeof(float), cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) return e;

        std::nth_element(h_probs.begin(), h_probs.begin() + (vocab - top_k), h_probs.end(),
            [](float a, float b) { return a > b; });
        float threshold = h_probs[vocab - top_k];

        // Zero out below threshold on device
        sm_topk_mask<<<nb, BS, 0, stream>>>(probs, threshold, vocab);

        // Recompute sum and renormalize
        e = cudaMemcpy(h_probs.data(), probs, vocab * sizeof(float), cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) return e;
        float new_sum = 0.f;
        for (int i = 0; i < vocab; ++i) new_sum += h_probs[i];
        float new_inv = 1.0f / fmaxf(new_sum, 1e-20f);
        sm_normalize<<<nb, BS, 0, stream>>>(probs, new_inv, vocab);
    }

    // Phase 5: weighted random sample
    sm_sample<<<1, BS, 0, stream>>>(probs, out_id, vocab, rng_seed, step);

    return cudaGetLastError();
}

} // namespace kernels
} // namespace blackwell
