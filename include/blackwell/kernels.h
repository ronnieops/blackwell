// Kernel declarations — public interface
#pragma once
#ifndef BLACKWELL_KERNELS_H
#define BLACKWELL_KERNELS_H

#include <cuda_runtime.h>

namespace blackwell {
namespace kernels {

// ---------------------------------------------------------------------------
// FP4 block-scaled GEMM
// ---------------------------------------------------------------------------
// Computes C = (A @ B) * scale  where A is (M×K) FP4 with per-K-block scales,
// and B is (K×N) FP4 or FP16.  Result accumulated in FP32.
cudaError_t gemm_fp4_block_scaled(
    float*          C,
    const void*     A_fp4,
    const float*    A_scale,   // [M / kBlockRow × K / kBlockCol] scales
    const void*     B_fp4,
    const float*    B_scale,
    int            M,
    int            N,
    int            K,
    cudaStream_t   stream = 0);

// ---------------------------------------------------------------------------
// FP4 GEMV (decode path)
// ---------------------------------------------------------------------------
cudaError_t gemv_fp4(
    float*          y,
    const void*     x_fp4,
    const float*    x_scale,
    const void*     W_fp4,
    const float*    W_scale,
    int            in_features,
    int            out_features,
    cudaStream_t   stream = 0);

// ---------------------------------------------------------------------------
// Memory ops
// ---------------------------------------------------------------------------
cudaError_t pack_fp4(
    void*           out_fp4,
    const float*    in_fp32,
    const float*    scale_out,
    int            num_elements,
    cudaStream_t   stream = 0);

cudaError_t unpack_fp4(
    float*          out_fp32,
    const void*     in_fp4,
    const float*    scale_in,
    int            num_elements,
    cudaStream_t   stream = 0);

cudaError_t coalesced_copy(
    float*          dst,
    const float*    src,
    int             num_elements,
    cudaStream_t    stream = 0);

// ---------------------------------------------------------------------------
// Fused epilogues
// ---------------------------------------------------------------------------
cudaError_t fused_rmsnorm(
    float*          out,
    const float*    inp,
    const float*    weight,
    int             num_elements,
    float           eps,
    cudaStream_t    stream = 0);

cudaError_t fused_rope(
    float*          out_inplace,
    const float*    cos_cache,
    const float*    sin_cache,
    int             heads,
    int             seq_len,
    int             head_dim,
    cudaStream_t    stream = 0);

cudaError_t apply_swiglu(
    float*          out,
    const float*    gate,
    const float*    up,
    int             num_elements,
    cudaStream_t    stream = 0);

// Elementwise FP32 vector add: out[i] = a[i] + b[i]
// Used for residual connections in transformer decode.
cudaError_t vector_add_fp32(
    float*          out,
    const float*    a,
    const float*    b,
    int             num_elements,
    cudaStream_t    stream = 0);

// ---------------------------------------------------------------------------
// Decode attention (single token × KV cache)
// ---------------------------------------------------------------------------
cudaError_t attention_decode(
    float*          output,      // [num_heads * head_dim] result
    const float*    Q,           // [num_heads * head_dim] query (dequantized)
    const float*    K_cache,     // [num_kv_heads * max_seq_len * head_dim] KV cache
    const float*    V_cache,     // [num_kv_heads * max_seq_len * head_dim]
    int             seq_pos,     // current position (inclusive)
    int             num_heads,   // num Q heads
    int             head_dim,
    int             max_seq_len,
    cudaStream_t    stream = 0);

// GQA-aware version: num_kv_heads may differ from num_heads
cudaError_t attention_decode_gqa(
    float*          output,
    const float*    Q,
    const float*    K_cache,
    const float*    V_cache,
    int             seq_pos,
    int             num_q_heads,
    int             num_kv_heads,
    int             head_dim,
    int             max_seq_len,
    cudaStream_t    stream = 0);

// ---------------------------------------------------------------------------
// Attention (prefill)
// ---------------------------------------------------------------------------
cudaError_t attention_fp4(
    float*          output,
    const void*     Q_fp4,
    const void*     K_fp4,
    const void*     V_fp4,
    const float*    Q_scale,
    const float*    K_scale,
    const float*    V_scale,
    int             batch_size,
    int             seq_len,
    int             num_heads,
    int             head_dim,
    float           scale,
    cudaStream_t    stream = 0);

// ---------------------------------------------------------------------------
// KV-cache (decode)
// ---------------------------------------------------------------------------
cudaError_t update_kv_cache(
    float*          k_cache,
    float*          v_cache,
    const float*    k_new,
    const float*    v_new,
    int             batch_idx,
    int             seq_pos,
    int             num_heads,
    int             head_dim,
    int             max_seq_len,
    cudaStream_t    stream = 0);

cudaError_t load_kv_cache_qkgv(
    float*          Q,
    float*          K_val,
    float*          V_val,
    const float*    k_cache,
    const float*    v_cache,
    int             batch_idx,
    int             seq_pos,
    int             num_heads,
    int             head_dim,
    int             max_seq_len,
    cudaStream_t    stream = 0);

// ---------------------------------------------------------------------------
// Fused QKV decode (single kernel: Q = x@Wq, K = x@Wk, V = x@Wv)
// ---------------------------------------------------------------------------
cudaError_t fused_qkv_gemv(
    float*          Q_out,
    float*          K_out,
    float*          V_out,
    const void*     x_fp4,
    const float*    x_scale,
    const void*     W_q_fp4,
    const float*    W_q_scale,
    const void*     W_k_fp4,
    const float*    W_k_scale,
    const void*     W_v_fp4,
    const float*    W_v_scale,
    int             hidden,
    int             q_dim,
    int             kv_dim,
    cudaStream_t    stream = 0);

// ---------------------------------------------------------------------------
// Prefill vs decode dispatch
// ---------------------------------------------------------------------------
enum class KernelMode { Prefill, Decode };
cudaError_t dispatch_matmul(
    float*          C,
    const void*     A,
    const void*     B,
    const float*    A_scale,
    const float*    B_scale,
    int            M,
    int            N,
    int            K,
    KernelMode     mode,
    cudaStream_t   stream = 0);

// ---------------------------------------------------------------------------
// Fused RMSNorm + FP4 pack (single kernel)
// ---------------------------------------------------------------------------
// Input: FP32 projection output, RMSNorm weight
// Output: FP4 packed + per-block scales
// Replaces: fused_rmsnorm → pack_fp4 (2 kernels → 1 kernel)
cudaError_t fused_rmsnorm_pack(
    void*           x_out_fp4,
    float*          x_out_scale,
    const float*    proj,            // FP32 input (from gemv_fp4)
    const float*    weight,          // RMSNorm weight
    int             N,               // num elements
    float           eps,
    cudaStream_t    stream = 0);

// ---------------------------------------------------------------------------
// Fused O-projection + RMSNorm + FP4 pack (convenience: 2 kernels)
// ---------------------------------------------------------------------------
// Replaces: gemv_fp4(W_o) → fused_rmsnorm → pack_fp4(x) (3 kernels → 2 kernels)
// Allocates internal temp buffer.
cudaError_t fused_o_norm_pack(
    void*           x_out_fp4,
    float*          x_out_scale,
    float*          scratch1,        // unused (kept for API compat)
    int*            scratch2,        // unused
    float*          scratch3,        // unused
    const void*     attn_fp4,
    const float*    attn_scale,
    const void*     W_o_fp4,
    const float*    W_o_scale,
    const float*    rmsnorm_weight,
    int             K,               // input features (q_dim)
    int             N,               // output features (hidden_dim)
    float           eps,
    cudaStream_t    stream = 0);

// ---------------------------------------------------------------------------
// Optimized GEMV v2 (vectorized FP4 block loads, transposed weights)
// ---------------------------------------------------------------------------
// Requires transposed weight layout: W_t [N×K] row-major.
cudaError_t gemv_fp4_v2(
    float*          y_out,
    const void*     x_fp4,
    const float*    x_scale,
    const void*     W_t_fp4,      // TRANSPOSED: [N × K]
    const float*    W_t_scale,    // TRANSPOSED: [N/16 × K/16]
    int             K,
    int             N,
    cudaStream_t    stream = 0);

// GEMV Split-K: K split into K_splits atomic partial sums.
// y_out MUST be initialized to 0 by caller.
// Grid: (N/256, K_splits). Useful for N=6144 where 24 blocks < 36 SMs.
cudaError_t gemv_fp4_splitk(
    float*          y_out,
    const void*     x_fp4,
    const float*    x_scale,
    const void*     W_t_fp4,
    const float*    W_t_scale,
    int             K,
    int             N,
    int             K_splits,
    cudaStream_t    stream = 0);

// GEMV v3: Shared memory tiled kernel for large N (e.g., down_proj 6144).
// Uses smem to load weight tiles and broadcast across threads.
// K must be multiple of 128. N must be multiple of 256.
cudaError_t gemv_fp4_v3(
    float*          y_out,
    const void*     x_fp4,
    const float*    x_scale,
    const void*     W_t_fp4,
    const float*    W_t_scale,
    int             K,
    int             N,
    cudaStream_t    stream = 0);

// GEMV Batched (M×v2): Process M input vectors simultaneously.
// Loads weight matrix once, amortizes across M outputs.
// y_out [M][N], x_fp4 [M][K], x_scale [M][K/16], W [N][K]
// Grid: (ceil(N/256), M). Each block computes 256 outputs for 1 token.
// Best for decode with batch size M (2-4 tokens at once).
cudaError_t gemv_fp4_batched(
    float*          y_out,     // [M * N] output (row-major per token)
    const void*     x_fp4,      // [M * K] FP4 input (row-major per token)
    const float*    x_scale,   // [M * K/16] scales (row-major per token)
    const void*     W_t_fp4,    // [N * K] transposed weights
    const float*    W_t_scale, // [N/16 * K/16] transposed scales
    int             K,
    int             N,
    int             M,         // batch size (number of simultaneous tokens)
    cudaStream_t    stream = 0);

// Pack FP32 to INT8 with per-block scales
cudaError_t pack_int8(
    void*           out_int8,
    const float*    in_fp32,
    const float*    scale_out,
    int             num_elements,
    cudaStream_t    stream = 0);

// INT8 block-scaled GEMV (warp-level dot products, transposed weights)
cudaError_t gemv_int8(
    float*          y_out,
    const void*     x_int8,
    const float*    x_scale,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    cudaStream_t    stream = 0);

cudaError_t transpose_fp4_weights(
    void*           dst,          // [N × K] FP4 transposed
    float*          dst_scale,    // [N/16 × K/16] transposed
    const void*     src,          // [K × N] FP4 original
    const float*    src_scale,    // [K/16 × N/16] original
    int             K,
    int             N,
    cudaStream_t    stream = 0);

// ---------------------------------------------------------------------------
// Fused gate + up MLP GEMV (single kernel)
// ---------------------------------------------------------------------------
// Computes both projections in one kernel. Uses transposed weights.
cudaError_t fused_gate_up_gemv(
    float*          gate_out,
    float*          up_out,
    const void*     x_fp4,
    const float*    x_scale,
    const void*     W_gate_t_fp4,    // TRANSPOSED: [N × K]
    const float*    W_gate_t_scale, // TRANSPOSED
    const void*     W_up_t_fp4,     // TRANSPOSED
    const float*    W_up_t_scale,   // TRANSPOSED
    int             K,
    int             N,               // output dim
    cudaStream_t    stream = 0);

// v1: original Grid(2, N/256) implementation for comparison
cudaError_t fused_gate_up_gemv_v1(
    float*          gate_out,
    float*          up_out,
    const void*     x_fp4,
    const float*    x_scale,
    const void*     W_gate_t_fp4,
    const float*    W_gate_t_scale,
    const void*     W_up_t_fp4,
    const float*    W_up_t_scale,
    int             K,
    int             N,
    cudaStream_t    stream = 0);

// ---------------------------------------------------------------------------
// CUDA Graphs (decode overhead reduction)
// ---------------------------------------------------------------------------
cudaError_t capture_decode_graph(
    void**          graph_out,
    void**          node_out,
    void*           graph_exec_out,
    float*          d_temp_storage,
    size_t          temp_storage_bytes,
    cudaStream_t    stream = 0);

cudaError_t launch_decode_graph(
    void*           graph_exec,
    cudaStream_t    stream = 0);

cudaError_t destroy_decode_graph(
    void*           graph_exec,
    void*           graph);

} // namespace kernels
} // namespace blackwell

#endif // BLACKWELL_KERNELS_H
