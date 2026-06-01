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
// Large CTA: 128×128×64, requires M%128==0, N%128==0, K%64==0.
cudaError_t gemm_fp4_block_scaled(
    float*          C,
    const void*     A_fp4,
    const float*    A_scale,
    const void*     B_fp4,
    const float*    B_scale,
    int            M,
    int            N,
    int            K,
    cudaStream_t   stream = 0);

// Small CTA: 64×64×64, 4 warps, 40 KB smem. For M<128 prefill.
// Requires M%64==0, N%64==0, K%64==0.
cudaError_t gemm_fp4_block_scaled_small(
    float*          C,
    const void*     A_fp4,
    const float*    A_scale,
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
// NVF4 GEMV (decode path) — UE4M3 scales, tensor core optimized
// ---------------------------------------------------------------------------
cudaError_t gemv_fp4_nv(
    float*          y,
    const void*     x_fp4,
    const void*     x_scale,      // UE4M3 [K/16]
    const void*     W_t_fp4,      // Transposed FP4 [N × K]
    const void*     W_t_scale,    // UE4M3 [N/16 × K/16]
    int            in_features,
    int            out_features,
    cudaStream_t   stream = 0);

// FP4 GEMV with FP32 pre-computed scales + FP16 accumulator (optimized)
// Same as gemv_fp4_nv but scales are FP32 (not UE4M3), inner loop uses __hfma.
// Scale conversion must be done offline (UE4M3→FP32) before calling.
cudaError_t gemv_fp4_nv_opt(
    float*          y,
    const void*     x_fp4,
    const void*     x_scale,      // FP32 [K/16] (pre-converted)
    const void*     W_t_fp4,      // Transposed FP4 [N × K]
    const void*     W_t_scale,    // FP32 [N/16 × K/16] (pre-converted)
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

// Fused: unpack FP4 → quantize to INT8 in 1 kernel (no intermediate FP32 buffer)
// i8_scales: pre-computed INT8 block scales [num_elements/16]
cudaError_t unpack_fp4_pack_int8(
    void*           out_i8,
    float*          out_scales,     // INT8 block scales (passed through, unused)
    const void*     in_fp4,
    const float*    fp4_scale,
    const float*    i8_scales,
    int             num_elements,
    cudaStream_t    stream = 0);

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

// Decode-specific RoPE: reads seq_pos from device memory (CUDA Graph safe)
cudaError_t fused_rope_decode(
    float*          out_inplace,
    const float*    cos_cache,
    const float*    sin_cache,
    const int*      seq_pos_ptr,
    int             heads,
    int             head_dim,
    int             max_seq_len,
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

// Batched GQA decode attention: process M sequences in parallel
// K_cache layout: [M][total_layers][num_kv_heads][max_seq_len][head_dim]
// kv_batch_elems: stride (in floats) between sequences' data for same layer
// kv_layer_elems: offset (in floats) from seq base to current layer
cudaError_t attention_decode_batched_gqa(
    float*          output,         // [M * num_q_heads * head_dim]
    const float*    Q,              // [M * num_q_heads * head_dim]
    const float*    K_cache,        // base pointer (seq 0, layer 0)
    const float*    V_cache,
    int             seq_pos,
    int             num_q_heads,
    int             num_kv_heads,
    int             head_dim,
    int             max_seq_len,
    int             M,              // batch size
    size_t          kv_batch_elems, // floats between sequences
    size_t          kv_layer_elems, // floats from seq base to current layer
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

// FP32 prefill attention (flash-style, after GEMM outputs FP32 Q/K/V)
cudaError_t attention_prefill(
    float*          output,
    const float*    Q,
    const float*    K,
    const float*    V,
    int             M,
    int             head_dim,
    int             num_q_heads,
    int             num_kv_heads,
    int             num_q_per_group,
    float           scale,
    cudaStream_t    stream = 0);

// ---------------------------------------------------------------------------
// Prefill layer orchestration
// ---------------------------------------------------------------------------
cudaError_t run_prefill_layer(
    float*          hidden_states,
    const int8_t*   W_q, const float* W_q_sc,
    const int8_t*   W_k, const float* W_k_sc,
    const int8_t*   W_v, const float* W_v_sc,
    const int8_t*   W_o, const float* W_o_sc,
    float*          k_cache, float* v_cache,
    int             batch_size, int seq_len,
    int             num_heads, int num_kv_heads,
    int             hidden_dim, int head_dim, int max_seq_len,
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
// Fused RMSNorm + INT8 quant (single kernel)
// ---------------------------------------------------------------------------
// Input: FP32 projection output, RMSNorm weight
// Output: INT8 packed + per-block scales
// Replaces: fused_rmsnorm → pack_int8 (2 kernels → 1 kernel)
cudaError_t fused_rmsnorm_quant_int8(
    int8_t*         x_out_i8,
    float*          x_out_scale,
    const float*    proj,
    const float*    weight,
    int             N,
    float           eps,
    cudaStream_t    stream = 0);

// ---------------------------------------------------------------------------
// Fused O-projection + RMSNorm + FP4 pack (convenience: 2 kernels)
// ---------------------------------------------------------------------------
cudaError_t gemv_int8_from_fp4(
    float*          y_out,
    const void*     x_fp4,        // FP4 input (same as gemv_fp4_v2)
    const float*    x_fp4_scale,  // FP4 per-block scales
    const void*     W_t_int8,     // INT8 transposed: [N × K]
    const float*    W_t_scale,    // INT8 transposed: [N/16 × K/16]
    int             K,
    int             N,
    cudaStream_t    stream = 0);

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

// Fused INT8 quantize: compute absmax scales + pack to INT8 in one kernel
// out_int8: [num_elements] INT8, out_scale: [num_elements/16] FP32 scales
// in_fp32:  [num_elements] FP32 input
// num_elements must be multiple of 16.
cudaError_t quantize_int8(
    void*           out_int8,
    float*          out_scale,
    const float*    in_fp32,
    int             num_elements,
    cudaStream_t    stream = 0);

// INT8 block-scaled GEMV Split-K (K split into K_splits, AtomicAdd reduction)
// Caller MUST zero y_out before launch. Grid: (N/256, K_splits).
// Targets large N with wave quantization (e.g., N=6144: 24 blocks < 36 SMs).
cudaError_t gemv_int8_splitk(
    float*          y_out,
    const void*     x_int8,
    const float*    x_scale,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    int             K_splits,
    cudaStream_t    stream = 0);

// INT8 Batched GEMV: process M tokens simultaneously, reuse weights across them.
// Grid: (ceil(N/256), M). Block: 256 threads.
// y_out [M * N], x_int8 [M * K], x_scale [M * K/16], W_t [N * K], W_t_scale [N/16 * K/16]
// Best M: 2-8 tokens (matching llama.cpp MMVQ_MAX_BATCH_SIZE).
cudaError_t gemv_int8_batched(
    float*          y_out,
    const void*     x_int8,
    const float*    x_scale,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    int             M,          // batch size (1-8)
    cudaStream_t    stream = 0);

// FP32×INT8 block-scaled GEMV — FP32 activations × INT8 weights
// Eliminates activation quantization. Weight format: W_t [N×K] INT8 transposed.
cudaError_t gemv_fp32_int8(
    float*          y_out,
    const float*    x_fp32,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    cudaStream_t    stream = 0);

// BF16 GEMV — FP32 activations × BF16 weights, FP32 accumulate.
// No quantization error. Weight layout: W_t [N × K] BF16 row-major.
cudaError_t gemv_bf16(
    float*          y_out,
    const float*    x_fp32,
    const void*     W_t_bf16,
    int             K,
    int             N,
    cudaStream_t    stream = 0);

// FP32×INT8 per-row GEMV — FP32 activations × INT8 per-row weights.
// Scale layout: W_t_scale [N × K/16]. Higher quality than INT8 activations.
cudaError_t gemv_fp32_int8_per_row(
    float*          y_out,
    const float*    x_fp32,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    cudaStream_t    stream = 0);

// Warp-cooperative INT8 GEMV — 1 warp per output row, shuffle reduction.
// 32 threads cooperatively compute each dot product. Better coalescing
// than per-thread GEMV (all threads read same row → 1 transaction vs 32).
// ~25 regs/thread → 8 blocks/SM occupancy. Best for decode (M=1).
cudaError_t gemv_int8_warp(
    float*          y_out,
    const void*     x_int8,
    const float*    x_scale,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    cudaStream_t   stream = 0);

// Warp-cooperative FP32×INT8 per-row GEMV — 1 warp/row, shuffle reduce.
// FP32 activations × INT8 per-row scaled weights. Same coalescing benefit.
cudaError_t gemv_fp32_int8_per_row_warp(
    float*          y_out,
    const float*    x_fp32,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    cudaStream_t   stream = 0);

// Fused Q/K/V GEMV — single kernel launch for all 3 projections.
// Reads activation vector once, shared across Q/K/V.
// Q_out: [N_q], K_out: [N_kv], V_out: [N_kv]
cudaError_t gemv_int8_qkv(
    float*          Q_out,
    float*          K_out,
    float*          V_out,
    const void*     x_int8,
    const float*    x_scale,
    const void*     W_q, const float* W_q_sc,
    const void*     W_k, const float* W_k_sc,
    const void*     W_v, const float* W_v_sc,
    int             K,
    int             N_q,
    int             N_kv,
    cudaStream_t   stream = 0);

// Fused gate+up GEMV — single kernel for both MLP input projections.
// Reads activation vector once, shared across gate/up.
// gate_out: [N], up_out: [N]
cudaError_t gemv_int8_gate_up(
    float*          gate_out,
    float*          up_out,
    const void*     x_int8,
    const float*    x_scale,
    const void*     W_gate, const float* W_gate_sc,
    const void*     W_up, const float* W_up_sc,
    int             K,
    int             N,
    cudaStream_t   stream = 0);

// Packed FP4 warp GEMV — 2 E2M1 values per byte, 2× less bandwidth than INT8.
// Packed FP4 activations × packed FP4 weights, per-row scales.
// x_packed: [K/2] bytes, x_scale: [K/16] FP32, W_packed: [N][K/2] bytes, W_scale: [N][K/16] FP32
cudaError_t gemv_fp4_warp(
    float*          y_out,
    const void*     x_packed,
    const float*    x_scale,
    const void*     W_packed,
    const float*    W_scale,
    int             K,
    int             N,
    cudaStream_t   stream = 0);

// FP32 activations × packed FP4 weights — mixed precision warp GEMV.
// x_fp32: [K] FP32, W_packed: [N][K/2] bytes, W_scale: [N][K/16] FP32
cudaError_t gemv_fp32_fp4_warp(
    float*          y_out,
    const float*    x_fp32,
    const void*     W_packed,
    const float*    W_scale,
    int             K,
    int             N,
    cudaStream_t   stream = 0);

// Packed INT4 warp GEMV — signed INT4 activations × signed INT4 weights.
// 2× less bandwidth than INT8. Uses __dp4a after nibble→int8 unpack.
// x_packed: [K/2] bytes, x_scale: [K/16] FP32, W_packed: [N][K/2] bytes, W_scale: [N][K/16] FP32
cudaError_t gemv_int4_warp(
    float*          y_out,
    const void*     x_packed,
    const float*    x_scale,
    const void*     W_packed,
    const float*    W_scale,
    int             K,
    int             N,
    cudaStream_t   stream = 0);

// INT8 per-row GEMV — each output row has independent block-16 scales.
// Scale layout: W_t_scale [N × K/16] (not 2D [N/16 × K/16]).
// Fixes quality: per-row scales prevent 16-row quantization error accumulation.
cudaError_t gemv_int8_per_row(
    float*          y_out,
    const void*     x_int8,
    const float*    x_scale,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    cudaStream_t    stream = 0);

// INT8 block-scaled GEMV (warp-level dot products, transposed weights)
// DEPRECATED: uses 2D block scales [N/16 × K/16] — garbles 28-layer output.
// Use gemv_int8_per_row instead.
cudaError_t gemv_int8(
    float*          y_out,
    const void*     x_int8,
    const float*    x_scale,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    cudaStream_t    stream = 0);

// INT8 GEMV with PDL (Programmatic Dependent Launch)
// Overlaps kernel execution for +3-5% speedup.
// Requires CTK >= 12.3, SM >= 90.
cudaError_t gemv_int8_pdl(
    float*          y_out,
    const void*     x_int8,
    const float*    x_scale,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    cudaStream_t    stream = 0);

// INT8 GEMV with FP16 scales (+5-8% speedup)
// Uses FP16 scales instead of FP32, reducing scale memory by 50%.
cudaError_t gemv_int8_fp16sc(
    float*          y_out,
    const void*     x_int8,
    const void*     x_scale,     // __half FP16 scales [K/16]
    const void*     W_t_int8,
    const void*     W_t_scale,   // __half FP16 scales [N × K/16]
    int             K,
    int             N,
    cudaStream_t    stream = 0);

// INT8 GEMV with 4× loop unrolling (+3-5% speedup)
// Processes 4 K-blocks (64 values) per iteration for better ILP.
cudaError_t gemv_int8_unrolled(
    float*          y_out,
    const void*     x_int8,
    const float*    x_scale,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    cudaStream_t    stream = 0);

// Warp-cooperative INT8 GEMV with 4× loop unrolling (+9-45% speedup)
// 1 warp (32 threads) per output row, processes 4 K-blocks per iteration.
cudaError_t gemv_int8_warp_unrolled(
    float*          y_out,
    const void*     x_int8,
    const float*    x_scale,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    cudaStream_t    stream = 0);

// INT8 GEMV with cached FP16 scales (+2-13% speedup)
// Converts FP32 scales to FP16 at load time, reuses cached FP16 scales.
cudaError_t gemv_int8_fp16cached(
    float*          y_out,
    const void*     x_int8,
    const float*    x_scale,     // FP32 scales [K/16]
    const void*     W_t_int8,
    const float*    W_t_scale,   // FP32 scales [N × K/16]
    int             K,
    int             N,
    cudaStream_t    stream = 0);

// Clear FP16 scale caches (call when weights are reloaded)
void clear_fp16_scale_caches();

// Convert FP32 scales to FP16
cudaError_t convert_scales_fp32_to_fp16(
    const float*    fp32_scales,
    void*           fp16_scales,
    int             count,
    cudaStream_t    stream = 0);

// INT8×INT8 GEMM with __dp4a — pre-quantized activations
// C[M×N] = A_i8[M×K] × B_i8[N×K]^T
// Activations must be pre-quantized via pack_int8 or fused_rmsnorm_quant_int8.
// A_i8: INT8 [M×K], A_scale: [M × K/16] FP32 block scales
// B_i8: INT8 [N×K], B_scale: [N × K/16] FP32 block scales
// K must be multiple of 16. Uses __dp4a SIMD dot product.
cudaError_t gemm_int8_dp4a(
    float*          C,              // [M×N] output
    const int8_t*   A_int8,         // [M×K] INT8 pre-quantized activations
    const float*    A_scale,        // [M × K/16] activation scales
    const int8_t*   B_int8,         // [N×K] INT8 transposed weights
    const float*    B_scale,        // [N × K/16] weight scales
    int             M, int N, int K,
    cudaStream_t    stream = 0);

// INT8 GEMM: C[M×N] = A[M×K] × B^T[N×K]
// A is FP32 activations, B is INT8 weights [N×K] with scales [N × K/16]
// Uses 4×4 register tiling. K must be multiple of 16.
cudaError_t gemm_int8(
    float*          C,              // [M×N] output
    const float*    A,              // [M×K] FP32 activations
    const void*     B_int8,         // [N×K] INT8 transposed weights
    const float*    B_scale,        // [N × K/16] weight scales
    int             M, int N, int K,
    cudaStream_t    stream = 0);

// INT8×INT8 GEMM with tensor core mma.sync.aligned.m16n8k32
// C[M×N] = A_i8[M×K] × B_i8[N×K]^T × A_sc × B_sc
// Requires M≥16, N≥8, K≥32.
cudaError_t gemm_int8_mma(
    float*          C,              // [M×N] output
    const void*     A_i8,           // [M×K] INT8 activations
    const float*    A_sc,           // [M × K/16] activation scales
    const void*     B_i8,           // [N×K] INT8 transposed weights
    const float*    B_sc,           // [N × K/16] weight scales
    int             M, int N, int K,
    cudaStream_t    stream = 0);

// INT8×INT8 GEMM with WMMA m16n16k16 tensor cores
// C[M×N] = A_i8[M×K] × B_i8[N×K]^T × A_sc × B_sc
// Requires M≥16, N≥16, K≥16 (multiples of 16).
// 4.8× faster than dp4a for large M.
cudaError_t gemm_int8_wmma(
    float*          C,              // [M×N] output
    const void*     A_i8,           // [M×K] INT8 activations
    const float*    A_sc,           // [M × K/16] activation scales
    const void*     B_i8,           // [N×K] INT8 transposed weights
    const float*    B_sc,           // [N × K/16] weight scales
    int             M, int N, int K,
    cudaStream_t    stream = 0);

// Optimized WMMA: 32×32 tiles, 4 warps, direct FP32 accumulation
// 1.5-2× faster than gemm_int8_wmma for M≥32.
cudaError_t gemm_int8_wmma_fast(
    float*          C,
    const void*     A_i8,
    const float*    A_sc,
    const void*     B_i8,
    const float*    B_sc,
    int             M, int N, int K,
    cudaStream_t    stream = 0);

cudaError_t transpose_fp4_weights(
    void*           dst,          // [N × K] FP4 transposed
    float*          dst_scale,    // [N/16 × K/16] transposed
    const void*     src,          // [K × N] FP4 original
    const float*    src_scale,    // [K/16 × N/16] original
    int             K,
    int             N,
    cudaStream_t    stream = 0);

// INT8 transpose: W (K×N) → W_t (N×K), scales (K/16 × N/16) → (N/16 × K/16)
cudaError_t transpose_int8_weights(
    void*           dst,          // [N × K] INT8 transposed
    float*          dst_scale,    // [N/16 × K/16] transposed
    const void*     src,          // [K × N] INT8 original
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

// ---------------------------------------------------------------------------
// Dynamic seq_pos for CUDA Graph autoregressive decode
// ---------------------------------------------------------------------------
// Updates the device-side seq_pos used by attention_decode_gqa and
// update_kv_cache. Call BEFORE each cudaGraphLaunch in an autoregressive
// loop. Writes to pinned host memory (visible to captured graph memcpy nodes)
// and issues cudaMemcpyAsync to device.
cudaError_t update_decode_seq_pos(
    int             seq_pos,
    cudaStream_t    stream = 0);

// Get device pointer to seq_pos (for CUDA Graph RoPE)
cudaError_t get_seq_pos_device_ptr(int** ptr);

// Get pinned host pointer to seq_pos (for graph-safe host writes)
cudaError_t get_seq_pos_host_ptr(int** ptr);

// ---------------------------------------------------------------------------
// GPU-side logit sampling (eliminates 607 KB copy per token)
// ---------------------------------------------------------------------------
// argmax: deterministic, fastest
cudaError_t sample_argmax_gpu(
    const float*    logits,     // [VOCAB] on-device logits
    int             vocab,      // vocabulary size
    int*            out_id,     // device pointer to single int
    cudaStream_t    stream = 0);

// Unified GPU sampler — handles argmax, temperature, and top-k
// Replaces the 607 KB cudaMemcpy for temperature > 0.01 path
cudaError_t sample_gpu(
    const float*    logits,     // [VOCAB] on-device logits
    int             vocab,       // vocabulary size
    float           temperature,// <0.01 = greedy argmax, >0 = softmax sampling
    int             top_k,      // 0 = disabled, >0 = keep top-k logits
    int*            out_id,     // device pointer to single int
    unsigned long long rng_seed,// curand seed
    int             step,       // step counter (for rng state)
    cudaStream_t    stream = 0);

// ---------------------------------------------------------------------------
// GatedDeltaNet (linear attention) for Qwen3.5-9B
// ---------------------------------------------------------------------------
// Conv1d update: shift state, insert new QKV, depthwise conv + SiLU
cudaError_t gated_delta_conv1d_update(
    float*          conv_state, // [CONV_DIM * (CONV_K-1)] in-place
    const float*    new_qkv,    // [CONV_DIM]
    const float*    conv_w,     // [CONV_DIM * CONV_K]
    float*          out,        // [CONV_DIM]
    cudaStream_t    stream = 0);

// Recurrent step: Q,K broadcast + SSM update
cudaError_t gated_delta_recurrent_step(
    const float*    q,          // [batch, NK, HD]
    const float*    k,          // [batch, NK, HD]
    const float*    v,          // [batch, NV, HD]
    const float*    g,          // [batch, NV] decay
    const float*    beta,       // [batch, NV] gate
    float*          q_broadcast,// [batch, NV, HD] temp buffer
    float*          k_broadcast,// [batch, NV, HD] temp buffer
    float*          state,      // [batch, NV, HD, HD]
    float*          out,        // [batch, NV, HD]
    int             batch_size,
    cudaStream_t    stream = 0);

// RMSNormGated: norm(x) * silu(gate)
cudaError_t gated_delta_rmsnorm_gated(
    float*          out,        // [batch, NV, HD]
    const float*    x,          // [batch, NV, HD]
    const float*    gate,       // [batch, NV, HD]
    const float*    norm_w,     // [HD]
    int             batch_size,
    float           eps = 1e-6f,
    cudaStream_t    stream = 0);

// Fused residual add + RMSNorm + INT8 quant
// Computes residual add + RMSNorm + quantize in single kernel
// Saves 1 kernel launch per call (2 per layer)
cudaError_t fused_residual_norm(
    int8_t* x_out_i8,
    float* x_out_scale,
    float* proj,          // Modified in-place: proj += residual
    const float* residual,
    const float* norm_w,
    int N, float eps,
    cudaStream_t stream = 0);

cudaError_t fused_unpack_fp4_quant(
    int8_t* out_i8,
    float* out_scale,
    const void* in_fp4,
    const float* fp4_scale,
    const float* int8_scale,
    int N,
    cudaStream_t stream = 0);

// Fused: SwiGLU activation + INT8 quant
// Replaces: apply_swiglu → pack_int8 (2 kernels → 1 kernel)
cudaError_t fused_swiglu_quant(
    int8_t* out_i8,
    float* out_scale,
    const float* gate,
    const float* up,
    int N,
    cudaStream_t stream = 0);

// Persistent QKV GEMV kernel
// Single grid launch: 16 blocks (one per Q head), 128 threads/block, 40KB smem.
// Fuses QKV GEMV + KV cache update per layer. 4 launches/layer instead of 14.
// Replaces separate gemv_int8_qkv calls. Uses existing proven attention_decode_gqa.
cudaError_t persistent_qkv_gemv(
    const void* W_q, const float* W_q_sc,
    const void* W_k, const float* W_k_sc,
    const void* W_v, const float* W_v_sc,
    void* k_cache, void* v_cache,
    const int* seq_pos_ptr,
    const int8_t** layer_x_int8,
    const float** layer_x_sc,
    float* q_out, float* k_out, float* v_out,
    int num_layers,
    cudaStream_t stream = 0);

// Fused: pack_int8 + gemv_int8_warp output projection
// Replaces: pack_int8 → gemv_int8_warp (2 kernels → 1 kernel)
cudaError_t fused_pack_gemv_o(
    float* y_out,
    int8_t* temp_i8,
    float* temp_scale,
    const float* x_fp32,
    const void* W_t_int8,
    const float* W_t_scale,
    int K, int N,
    cudaStream_t stream = 0);

// Fused: SwiGLU activation + gemv_int8_warp down projection
// Replaces: fused_swiglu_quant → gemv_int8_warp (2 kernels → 1 kernel)
cudaError_t fused_swiglu_gemv(
    float* y_out,
    int8_t* temp_i8,
    float* temp_scale,
    const float* gate,
    const float* up,
    const void* W_t_int8,
    const float* W_t_scale,
    int K, int N,
    cudaStream_t stream = 0);

} // namespace kernels
} // namespace blackwell

#endif // BLACKWELL_KERNELS_H
