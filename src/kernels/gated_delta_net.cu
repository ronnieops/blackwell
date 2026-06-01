// src/kernels/gated_delta_net.cu — GatedDeltaNet (linear attention) for Qwen3.5-9B
//
// Implements the recurrent step of the GatedDeltaNet layer used in
// Qwen3.5-9B's 24 linear_attention layers. Single-token decode path.
//
// Architecture (per layer):
//   in_proj_qkv: [8192, 4096] — QKV projection (key_dim=2048, value_dim=4096)
//   in_proj_z: [4096, 4096] — gate projection
//   in_proj_b: [32, 4096] — beta projection (32 value heads)
//   in_proj_a: [32, 4096] — alpha projection (32 value heads)
//   conv1d: [8192, 1, 4] — 1D convolution on QKV (kernel=4)
//   A_log: [32] — SSM decay (log space)
//   dt_bias: [32] — delta time bias
//   out_proj: [4096, 4096] — output projection
//
// Recurrent step (single token):
//   g = -A_log.exp() * softplus(a + dt_bias)
//   state = state * g
//   kv_mem = state^T @ k
//   delta = (v - kv_mem) * beta
//   state = state + k * delta^T
//   out = state^T @ q
//   out = RMSNormGated(out, z) * out_proj

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

constexpr int HD = 128;       // head dimension
constexpr int NK = 16;        // num key heads
constexpr int NV = 32;        // num value heads
constexpr int CONV_DIM = NK * HD * 2 + NV * HD;  // 8192
constexpr int CONV_K = 4;     // conv kernel size

// ═══════════════════════════════════════════════════════════════════════════
// Conv1d update: shift conv_state, insert new QKV, apply SiLU
// conv_state: [CONV_DIM, CONV_K-1] = [8192, 3]
// new_qkv: [CONV_DIM] = [8192]
// Output: [CONV_DIM] (post-SiLU)
// ═══════════════════════════════════════════════════════════════════════════

__global__ void conv1d_update_kernel(
    float* __restrict__ conv_state,   // [CONV_DIM * (CONV_K-1)] in-place
    const float* __restrict__ new_qkv, // [CONV_DIM]
    const float* __restrict__ conv_w,  // [CONV_DIM * CONV_K] (depthwise)
    float* __restrict__ out             // [CONV_DIM]
) {
    int tid = threadIdx.x;
    int dim = blockIdx.x * blockDim.x + tid;
    if (dim >= CONV_DIM) return;

    // Load current state
    float state[CONV_K];
    #pragma unroll
    for (int k = 0; k < CONV_K - 1; ++k) {
        state[k] = conv_state[dim * (CONV_K - 1) + k];
    }
    state[CONV_K - 1] = new_qkv[dim];

    // Update conv_state (shift left, insert new)
    #pragma unroll
    for (int k = 0; k < CONV_K - 2; ++k) {
        conv_state[dim * (CONV_K - 1) + k] = state[k + 1];
    }
    conv_state[dim * (CONV_K - 1) + CONV_K - 2] = new_qkv[dim];

    // Depthwise conv
    float sum = 0.f;
    #pragma unroll
    for (int k = 0; k < CONV_K; ++k) {
        sum += state[k] * conv_w[dim * CONV_K + k];
    }

    // SiLU activation
    out[dim] = sum / (1.f + expf(-sum));
}

// ═══════════════════════════════════════════════════════════════════════════
// GatedDeltaNet recurrent step (single token)
//
// State layout: [batch, NV, HD, HD] — stored in global memory
// Each thread handles one v_dim (0..HD-1) for one head
// ═══════════════════════════════════════════════════════════════════════════

__global__ void gated_delta_recurrent_kernel(
    const float* __restrict__ q,      // [batch, NV, HD] query
    const float* __restrict__ k,      // [batch, NV, HD] key
    const float* __restrict__ v,      // [batch, NV, HD] value
    const float* __restrict__ g,      // [batch, NV] decay
    const float* __restrict__ beta,   // [batch, NV] gate
    float* __restrict__ state,        // [batch, NV, HD, HD]
    float* __restrict__ out,          // [batch, NV, HD]
    int batch_size
) {
    int bh = blockIdx.x;  // which (batch, head) pair
    int batch_idx = bh / NV;
    int head = bh % NV;
    int v_dim = threadIdx.x;  // which v_dim (0..HD-1)

    if (batch_idx >= batch_size || v_dim >= HD) return;

    // Pointers for this (batch, head)
    int qk_base = batch_idx * NV * HD + head * HD;
    int s_base = (batch_idx * NV + head) * HD * HD;
    const float* q_h = q + qk_base;
    const float* k_h = k + qk_base;
    const float* v_h = v + qk_base;
    float* s_h = state + s_base;
    float* o_h = out + batch_idx * NV * HD + head * HD;

    float g_h = g[batch_idx * NV + head];
    float beta_h = beta[batch_idx * NV + head];
    float q_val = q_h[v_dim];
    float k_val = k_h[v_dim];
    float v_val = v_h[v_dim];

    // Step 1: Apply decay to state column
    // state[k][v] *= g for all k
    for (int kk = 0; kk < HD; ++kk) {
        s_h[kk * HD + v_dim] *= g_h;
    }

    // Step 2: Compute kv_mem[v] = sum_k state[k][v] * k[k]
    float kv_mem = 0.f;
    for (int kk = 0; kk < HD; ++kk) {
        kv_mem += s_h[kk * HD + v_dim] * k_h[kk];
    }

    // Step 3: Compute delta[v] = (v[v] - kv_mem[v]) * beta
    float delta = (v_val - kv_mem) * beta_h;

    // Step 4: Update state: state[k][v] += k[k] * delta
    for (int kk = 0; kk < HD; ++kk) {
        s_h[kk * HD + v_dim] += k_h[kk] * delta;
    }

    // Step 5: Compute out[v] = sum_k state[k][v] * q[k]
    float o_val = 0.f;
    for (int kk = 0; kk < HD; ++kk) {
        o_val += s_h[kk * HD + v_dim] * q_h[kk];
    }
    o_h[v_dim] = o_val;
}

// ═══════════════════════════════════════════════════════════════════════════
// RMSNormGated: norm(x) * silu(gate)
// Per-head norm with shared reduction
// ═══════════════════════════════════════════════════════════════════════════

__global__ void rmsnorm_gated_kernel(
    float* __restrict__ out,       // [batch, NV, HD]
    const float* __restrict__ x,   // [batch, NV, HD]
    const float* __restrict__ gate, // [batch, NV, HD]
    const float* __restrict__ w,   // [HD] norm weight
    int total_heads,               // batch * NV
    float eps
) {
    int head = blockIdx.x;
    if (head >= total_heads) return;
    int tid = threadIdx.x;
    int base = head * HD;

    // Compute variance
    float var = 0.f;
    for (int i = tid; i < HD; i += blockDim.x) {
        float val = x[base + i];
        var += val * val;
    }
    // Warp reduce
    for (int off = 16; off > 0; off >>= 1)
        var += __shfl_xor_sync(0xffffffff, var, off);
    __shared__ float warp_var[8];
    if ((tid & 31) == 0) warp_var[tid >> 5] = var;
    __syncthreads();
    // Cross-warp reduce (all threads participate in shuffle)
    if (tid < 32) {
        float v = (tid < 8) ? warp_var[tid] : 0.f;
        for (int off = 4; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, off);
        if (tid == 0) warp_var[0] = v / HD;
    }
    __syncthreads();
    float inv_std = rsqrtf(warp_var[0] + eps);

    // Apply norm + silu gate
    for (int i = tid; i < HD; i += blockDim.x) {
        float normed = x[base + i] * inv_std * w[i];
        float g = gate[base + i];
        float silu = g / (1.f + expf(-g));
        out[base + i] = normed * silu;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Query/key broadcast: NK heads → NV heads (2x repeat)
// q_in: [batch, NK, HD] → q_out: [batch, NV, HD]
// ═══════════════════════════════════════════════════════════════════════════

__global__ void broadcast_qk_kernel(
    const float* __restrict__ q_in,  // [batch, NK, HD]
    float* __restrict__ q_out,       // [batch, NV, HD]
    int batch_size
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * NV * HD;
    if (gid >= total) return;

    int v_dim = gid % HD;
    int head = (gid / HD) % NV;
    int batch = gid / (NV * HD);

    // NV / NK = 2, so each NK head maps to 2 NV heads
    int k_head = head / 2;
    int src_idx = batch * NK * HD + k_head * HD + v_dim;
    q_out[gid] = q_in[src_idx];
}

} // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════

cudaError_t gated_delta_conv1d_update(
    float* conv_state,
    const float* new_qkv,
    const float* conv_w,
    float* out,
    cudaStream_t stream
) {
    int blocks = (CONV_DIM + 255) / 256;
    conv1d_update_kernel<<<blocks, 256, 0, stream>>>(
        conv_state, new_qkv, conv_w, out);
    return cudaGetLastError();
}

cudaError_t gated_delta_recurrent_step(
    const float* q,       // [batch, NK, HD]
    const float* k,       // [batch, NK, HD]
    const float* v,       // [batch, NV, HD]
    const float* g,       // [batch, NV]
    const float* beta,    // [batch, NV]
    float* q_broadcast,   // [batch, NV, HD] temp buffer
    float* k_broadcast,   // [batch, NV, HD] temp buffer
    float* state,         // [batch, NV, HD, HD]
    float* out,           // [batch, NV, HD]
    int batch_size,
    cudaStream_t stream
) {
    // Broadcast Q and K from NK to NV heads
    int total = batch_size * NV * HD;
    int blocks_bc = (total + 255) / 256;
    broadcast_qk_kernel<<<blocks_bc, 256, 0, stream>>>(q, q_broadcast, batch_size);
    broadcast_qk_kernel<<<blocks_bc, 256, 0, stream>>>(k, k_broadcast, batch_size);

    // Recurrent step: one block per (batch, head), HD threads per block
    gated_delta_recurrent_kernel<<<batch_size * NV, HD, 0, stream>>>(
        q_broadcast, k_broadcast, v, g, beta, state, out, batch_size);
    return cudaGetLastError();
}

cudaError_t gated_delta_rmsnorm_gated(
    float* out,
    const float* x,
    const float* gate,
    const float* norm_w,
    int batch_size,
    float eps,
    cudaStream_t stream
) {
    int total_heads = batch_size * NV;
    rmsnorm_gated_kernel<<<total_heads, 256, 0, stream>>>(
        out, x, gate, norm_w, total_heads, eps);
    return cudaGetLastError();
}

} // namespace kernels
} // namespace blackwell
