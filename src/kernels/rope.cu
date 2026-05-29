// src/kernels/rope.cu — Rotary Position Embedding (RoPE) for Blackwell SM_120
//
// Standard RoPE (from Su et al., RoFormer):
//   For a vector x[0..d-1] split into pairs:
//     x_rot[2i]   =  x[2i]   * cos(theta_i) - x[2i+1] * sin(theta_i)
//     x_rot[2i+1] =  x[2i]   * sin(theta_i) + x[2i+1] * cos(theta_i)
//   where theta_i = base^( -2i/d ) for i < d/2.
//
// Computation per head:
//   head_dim must be even.  We use 4-element vector loads (128-bit, matches bus).
//
// Data layout convention:
//   Q or K after projection: [batch, seq, num_heads, head_dim]
//   We flatten to 1D: blockIdx.x maps to one head-position: bidx = b * seq * heads + s * heads + h
//   Each block handles one (batch, seq_pos, head) triple.
//   Threads: head_dim / 4 (with element pair = 2 floats → head_dim/2 pairs).
//
// Cos/sin cache layout:
//   Precomputed per position and per head_dim dimension: [max_seq_len, head_dim/2]
//   Access: cos_cache[pos * (head_dim/2) + i], sin likewise.
//   For the standard Llama RoPE base=10000, we precompute externally.

#include <cuda_runtime.h>
#include <cmath>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace blackwell {
namespace kernels {
namespace {

// ===========================================================================
// RoPE kernel
//
// Grid: batch * seq_len * num_heads blocks.
// Each block: head_dim/2 threads (one per rotation pair, 2 floats each).
// Thread layout: Each thread handles one rotation pair:
//   x[2i], x[2i+1] → x_rot[2i], x_rot[2i+1]
// ===========================================================================
__launch_bounds__(64, 1)
__global__ void rope_kernel(
    float* __restrict__ out_inplace,   // same as in-place Q or K: [B, S, H, D]
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int num_heads,
    int seq_len,
    int head_dim,
    int batch_size) {

    // Block index → (batch, seq, head).
    int hp_idx = blockIdx.x;  // flattened: b * seq_len * num_heads + s * num_heads + h
    // Decode position index → batch, seq_pos, head.
    int n_heads_total = num_heads;
    int n_seq_total  = seq_len * num_heads;  // for reuse if seq_len=1

    int h = hp_idx % num_heads;
    int s = (hp_idx / num_heads) % seq_len;
    int b = hp_idx / (num_heads * seq_len);

    if (hp_idx >= batch_size * seq_len * num_heads) return;

    // Shared memory: load cos/sin for this position into fast storage.
    // head_dim/2 cos/sin values per head per position.
    // Each thread loads two floats.
    __shared__ float smem_cos[64];   // max head_dim/2 = 64 (for head_dim=128)
    __shared__ float smem_sin[64];

    // Each thread maps to one rotation pair (floats 2i, 2i+1).
    int tid = threadIdx.x;
    int num_pairs = head_dim / 2;

    // Load cos and sin for this position.  Use full-warp parallel load if needed.
    // cos/sin layout: [max_seq_len][head_dim/2] → row-major by position.
    int cos_sin_idx = s * num_pairs + tid;  // s is seq position

    if (tid < num_pairs) {
        smem_cos[tid] = cos_cache[cos_sin_idx];
        smem_sin[tid] = sin_cache[cos_sin_idx];
    }
    __syncthreads();

    // Compute RoPE for each element pair in the head.
    // Data layout: flat [total_elements = batch * seq * num_heads * head_dim]
    // Index = ((b * seq_len + s) * num_heads + h) * head_dim + elem
    size_t head_base = (static_cast<size_t>(b) * seq_len * num_heads
                        + static_cast<size_t>(s) * num_heads
                        + h) * head_dim;

    // Apply rotation for pair i = tid:
    //   x0 = x[2i], x1 = x[2i+1]
    //   x_new[2i]   = x0 * cos - x1 * sin
    //   x_new[2i+1] = x0 * sin + x1 * cos
    if (tid < num_pairs) {
        int elem_even = head_base + 2 * tid;
        int elem_odd  = elem_even + 1;
        float x0 = out_inplace[elem_even];
        float x1 = out_inplace[elem_odd];
        float cos_t = smem_cos[tid];
        float sin_t = smem_sin[tid];
        float x0_rot = x0 * cos_t - x1 * sin_t;
        float x1_rot = x0 * sin_t + x1 * cos_t;
        out_inplace[elem_even] = x0_rot;
        out_inplace[elem_odd]   = x1_rot;
    }
}

// Variant: compute RoPE directly into output (non-in-place)
__launch_bounds__(64, 1)
__global__ void rope_out_kernel(
    float* __restrict__ rope_out,
    const float* __restrict__ qk_in,    // [B, S, H, D]
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int num_heads,
    int seq_len,
    int head_dim,
    int batch_size) {

    int hp_idx = blockIdx.x;
    if (hp_idx >= batch_size * seq_len * num_heads) return;

    int h = hp_idx % num_heads;
    int s = (hp_idx / num_heads) % seq_len;
    int b = hp_idx / (num_heads * seq_len);

    __shared__ float smem_cos[64];
    __shared__ float smem_sin[64];

    int tid = threadIdx.x;
    int num_pairs = head_dim / 2;
    int cos_sin_idx = s * num_pairs + tid;

    if (tid < num_pairs) {
        smem_cos[tid] = cos_cache[cos_sin_idx];
        smem_sin[tid] = sin_cache[cos_sin_idx];
    }
    __syncthreads();

    size_t head_base = (static_cast<size_t>(b) * seq_len * num_heads
                        + static_cast<size_t>(s) * num_heads
                        + h) * head_dim;

    if (tid < num_pairs) {
        int elem_even = head_base + 2 * tid;
        int elem_odd  = elem_even + 1;
        float x0 = qk_in[elem_even];
        float x1 = qk_in[elem_odd];
        float ct = smem_cos[tid];
        float st = smem_sin[tid];
        rope_out[elem_even] = x0 * ct - x1 * st;
        rope_out[elem_odd]   = x0 * st + x1 * ct;
    }
}

// ===========================================================================
// Decode-specific RoPE kernel: reads seq_pos from device memory.
// Used in CUDA Graph where seq_pos changes per replay via pinned memory.
// cos/sin_cache layout: [MAXSEQ, head_dim/2] — precomputed for all positions.
// ===========================================================================
__launch_bounds__(64, 1)
__global__ void rope_decode_kernel(
    float* __restrict__ out_inplace,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    const int* __restrict__ seq_pos_ptr,
    int num_heads, int head_dim, int max_seq_len) {

    int hp_idx = blockIdx.x;
    int h = hp_idx % num_heads;
    if (hp_idx >= num_heads) return;

    int seq_pos = *seq_pos_ptr;
    int num_pairs = head_dim / 2;

    __shared__ float smem_cos[64];
    __shared__ float smem_sin[64];

    int tid = threadIdx.x;
    int cos_sin_idx = seq_pos * num_pairs + tid;

    if (tid < num_pairs) {
        smem_cos[tid] = cos_cache[cos_sin_idx];
        smem_sin[tid] = sin_cache[cos_sin_idx];
    }
    __syncthreads();

    // Each thread handles one rotation pair
    if (tid < num_pairs) {
        int i2 = tid * 2;
        float* pair = out_inplace + h * head_dim + i2;
        float x = pair[0], y = pair[1];
        float c = smem_cos[tid], s = smem_sin[tid];
        pair[0] = x * c - y * s;
        pair[1] = x * s + y * c;
    }
}

} // anonymous namespace

// ===========================================================================
// Public API
// ===========================================================================

cudaError_t fused_rope(
    float* out_inplace, const float* cos_cache, const float* sin_cache,
    int heads, int seq_len, int head_dim, cudaStream_t stream) {

    // Normalize to 1D: one block per (batch=1, seq_pos, head)
    // For general batch, heads may differ; use simpler 1D grid: heads × seq_len.
    // (Caller must manage batch dimension separately.)
    int total = heads * seq_len;
    dim3 grid(total);
    dim3 block((head_dim / 2 + 63) / 64 * 64);  // ensure at least head_dim/2 threads

    rope_kernel<<<grid, block, 0, stream>>>(
        out_inplace, cos_cache, sin_cache,
        heads, seq_len, head_dim, 1 /* batch_size=1 for sequence decode */);

    return cudaPeekAtLastError();
}

// Decode-specific RoPE: reads seq_pos from device memory (CUDA Graph safe)
cudaError_t fused_rope_decode(
    float* out_inplace,
    const float* cos_cache,
    const float* sin_cache,
    const int* seq_pos_ptr,
    int heads, int head_dim, int max_seq_len,
    cudaStream_t stream) {

    dim3 grid(heads);
    dim3 block((head_dim / 2 + 63) / 64 * 64);

    rope_decode_kernel<<<grid, block, 0, stream>>>(
        out_inplace, cos_cache, sin_cache, seq_pos_ptr,
        heads, head_dim, max_seq_len);

    return cudaPeekAtLastError();
}

} // namespace kernels
} // namespace blackwell
