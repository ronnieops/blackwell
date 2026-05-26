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
