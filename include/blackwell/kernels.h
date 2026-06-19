// Kernel declarations — public interface
#pragma once
#ifndef BLACKWELL_KERNELS_H
#define BLACKWELL_KERNELS_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace blackwell {
namespace kernels {

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

// Batched RMSNorm: processes M sequences of H elements each (same weight per seq).
// Saves M-1 kernel launches vs calling fused_rmsnorm M times.
cudaError_t fused_rmsnorm_batched(
    float*          out,
    const float*    inp,
    const float*    weight,
    int             H,        // elements per sequence (e.g. 4096)
    float           eps,
    int             M,        // number of sequences
    cudaStream_t    stream = 0);


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

// Per-head Q/K RMSNorm (Gemma 4 QK head norms)
cudaError_t head_norm(
    float*          data,
    const float*    weight,
    int             n_heads,
    int             head_dim,
    float           eps,
    cudaStream_t    stream = 0);

cudaError_t apply_geglu(
    float*          out,
    const float*    gate,
    const float*    up,
    int             num_elements,
    cudaStream_t    stream = 0);

cudaError_t apply_swiglu(
    float*          out,
    const float*    gate,
    const float*    up,
    int             num_elements,
    cudaStream_t    stream = 0);

// Apply logit softcapping: out = tanh(x / cap) * cap (in-place)
cudaError_t apply_logit_softcap(
    float*          data,
    int             N_elements,
    float           cap,
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
// FP16→FP32 conversion (element-wise)
// ---------------------------------------------------------------------------
cudaError_t convert_fp16_to_fp32(
    float*          out_f32,
    const __half*   in_f16,
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

// Direct seq_pos variant: no H2D copy, no pinned buffer race.
// Use in prefill loops where multiple positions are processed sequentially.
cudaError_t attention_decode_batched_gqa_pos(
    float*          output,
    const float*    Q,
    const float*    K_cache,
    const float*    V_cache,
    int             seq_pos,         // direct value (no device pointer)
    int             num_q_heads,
    int             num_kv_heads,
    int             head_dim,
    int             max_seq_len,
    int             M,
    size_t          kv_batch_elems,
    size_t          kv_layer_elems,
    cudaStream_t    stream = 0);

// Graph-safe batched GQA: takes device-side seq_pos pointer (no H2D memcpy)
cudaError_t attention_decode_batched_gqa_device(
    float*          output,
    const float*    Q,
    const float*    K_cache,
    const float*    V_cache,
    const int*      d_seq_pos,      // device pointer (skip H2D copy)
    int             num_q_heads,
    int             num_kv_heads,
    int             head_dim,
    int             max_seq_len,
    int             M,
    size_t          kv_batch_elems,
    size_t          kv_layer_elems,
    cudaStream_t    stream = 0);

// Graph-safe single-seq GQA: takes device-side seq_pos pointer
cudaError_t attention_decode_gqa_device(
    float*          output,
    const float*    Q,
    const float*    K_cache,
    const float*    V_cache,
    const int*      d_seq_pos,      // device pointer (skip H2D copy)
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

// Prefill attention for [M, num_heads, head_dim] layout (server layout)
cudaError_t attention_prefill_v2(
    float*          output,
    const float*    Q,
    const float*    K,
    const float*    V,
    int             M,
    int             head_dim,
    int             num_q_heads,
    int             num_kv_heads,
    int             num_q_per_group,
    cudaStream_t    stream = 0);

// Prefill attention v3: [M, num_heads, head_dim] layout, hd=128, M≤16
// Supports larger head_dim (128) and M up to 16.
// Uses 32 threads (1 warp), Q in registers, K in shared memory, causal masking.
cudaError_t attention_prefill_v3(
    float*          output,
    const float*    Q,
    const float*    K,
    const float*    V,
    int             M,
    int             head_dim,
    int             num_q_heads,
    int             num_kv_heads,
    int             num_q_per_group,
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

// Graph-safe variant: no H2D copy, uses device pointer for seq_pos
cudaError_t update_kv_cache_device(
    float*          k_cache,
    float*          v_cache,
    const float*    k_new,
    const float*    v_new,
    int             batch_idx,
    const int*      d_seq_pos,   // device pointer (skip H2D copy)
    int             num_heads,
    int             head_dim,
    int             max_seq_len,
    cudaStream_t    stream = 0);

// Direct seq_pos variant: no H2D copy, no pinned buffer race.
// Use in tight loops where multiple positions are written sequentially.
cudaError_t update_kv_cache_pos(
    float*          k_cache,
    float*          v_cache,
    const float*    k_new,
    const float*    v_new,
    int             seq_pos,     // direct value (no device pointer)
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

// Fused RMSNorm + INT4 quant (single kernel)
// Input: FP32 projection output, RMSNorm weight
// Output: packed INT4 (2 vals/byte) + per-block scales (block-16, absmax/7)
// Replaces: fused_rmsnorm → quantize_int4 (2 kernels → 1 kernel)
cudaError_t fused_rmsnorm_quant_int4(
    uint8_t*        x_out_packed,
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

// Warp-cooperative FP32×INT4 GEMV — 1 warp/row, shuffle reduction.
// FP32 activations × INT4 packed weights (no activation quantization).
// Uses same block-16 INT4 format as gemv_int4_warp_kernel.
cudaError_t gemv_fp32_int4_warp(
    float*          y_out,
    const float*    x_fp32,
    const void*     W_packed,
    const float*    W_scale,
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

// FP32 GEMV — y = W * x where W is [N x K] row-major FP32.
// For high-precision inference with FP16 weights (dequantized to FP32).
cudaError_t gemv_fp32_launch(
    float*          y_out,
    const float*    W_fp32,
    const float*    x_fp32,
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

// FP16 weight-scale variant of gemv_int4_warp.
// Identical math; W_scale points to [N][K/16] __half instead of FP32.
// Halves weight-scale memory traffic (~33% of total GEMV bytes). Use when
// weights were quantized with FP16 scales (scripts/convert_scales_fp16.py).
// Activation scales (x_scale) stay FP32 (tiny, reused across N rows).
cudaError_t gemv_int4_warp_f16wsc(
    float*          y_out,
    const void*     x_packed,
    const float*    x_scale,
    const void*     W_packed,
    const void*     W_scale,       // __half* [N][K/16]
    int             K,
    int             N,
    cudaStream_t    stream = 0);

// Fused Q/K/V GEMV with inline INT4 quantization
// Loads FP32 x once, computes scales, quantizes to INT4 in registers,
// then does Q/K/V GEMV using shared quantized activation.
// Saves: 1× INT4 write + 3× INT4 read + sync vs separate quantize+3×GEMV.
cudaError_t fused_qkv_int4(
    float* Q_out, float* K_out, float* V_out,
    const float* x_fp32, float* x_scale_out,
    const uint8_t* W_q_packed, const float* W_q_scale,
    const uint8_t* W_k_packed, const float* W_k_scale,
    const uint8_t* W_v_packed, const float* W_v_scale,
    int K, int N_q, int N_kv,
    cudaStream_t stream = 0);

// Transpose INT4 weights: W (K×N/2) → W_t (N×K/2), scales transposed
cudaError_t transpose_int4_weights(
    void*           dst,
    float*          dst_scale,
    const void*     src,
    const float*    src_scale,
    int             K,
    int             N,
    cudaStream_t    stream = 0);

// Unpack packed INT4 to FP32 (for non-GEMV usage)
cudaError_t unpack_int4_fp32(
    float*          x_out,
    const void*     x_packed,
    const float*    x_scale,
    int             K,
    cudaStream_t    stream = 0);

// Quantize FP32 → packed INT4 (with per-block scales)
// x_out_packed: [K/2] bytes (packed INT4), x_out_sc: [K/16] FP32 scales
// in_fp32: [K] FP32 input
// Block size = 16. Each 16-element block: absmax scale, quantize to [-7..7], pack.
cudaError_t quantize_int4(
    void*           x_out_packed,
    float*          x_out_sc,
    const float*    in_fp32,
    int             K,
    cudaStream_t    stream = 0);

// Batched INT4 quantization for M sequences
// x_out_packed: [M][K/2] packed INT4, x_out_sc: [M][K/16] scales
// in_fp32: [M][K] FP32 input
cudaError_t quantize_int4_batched(
    void*           x_out_packed,
    float*          x_out_sc,
    const float*    in_fp32,
    int             K,
    int             M,
    cudaStream_t    stream = 0);

// Fused SwiGLU + INT4 quant — replaces apply_swiglu + quantize_int4 (2→1 kernel)
cudaError_t fused_swiglu_quant_int4(
    uint8_t* out_packed,
    float* out_scale,
    const float* gate,
    const float* up,
    int N,
    cudaStream_t stream = 0);

// Fused gate+up INT4 GEMV — replaces two separate gemv_int4_batched_f16wsc calls.
// gate_out [M][N], up_out [M][N]: two INT4×FP32 projections from same activation.
// Loads activation once per K-block, computes both outputs. 1 kernel launch vs 2.
// WScaleT: __half (FP16 scales, half traffic) or float (FP32 scales).
cudaError_t fused_gate_up_int4_f16wsc(
    float*          gate_out,
    float*          up_out,
    const void*     x_packed,     // [M][K/2] packed INT4 activations
    const float*    x_scale,     // [M][K/16] FP32 activation scales
    const void*     W_g_packed,  // [N][K/2] packed INT4 gate weights
    const void*     W_g_scale,   // __half* [N][K/16] gate weight scales
    const void*     W_u_packed,  // [N][K/2] packed INT4 up weights
    const void*     W_u_scale,   // __half* [N][K/16] up weight scales
    int             K,
    int             N,            // = I (hidden dim)
    int             M,            // batch size 1..16
    cudaStream_t    stream = 0);

// FP32-scale variant of fused_gate_up_int4 (for non-FP16-scale weight dirs).
cudaError_t fused_gate_up_int4(
    float*          gate_out,
    float*          up_out,
    const uint8_t*  x_packed,
    const float*    x_scale,
    const uint8_t*  W_g_packed,
    const float*    W_g_scale,
    const uint8_t*  W_u_packed,
    const float*    W_u_scale,
    int             K,
    int             N,
    int             M,
    cudaStream_t    stream = 0);

// Fused QKV INT4 GEMV: 3 projections (Q, K, V) from same activation in 1 kernel.
// Q_out [M][Q_dim], K_out [M][KV_dim], V_out [M][KV_dim].
// Grid: max(Q_dim, KV_dim) blocks. Blocks 0..KV_dim-1 compute Q+K+V (x loaded once);
// blocks KV_dim..Q_dim-1 compute Q only. 1 kernel launch vs 3.
cudaError_t fused_qkv_int4_f16wsc(
    float*          Q_out,
    float*          K_out,
    float*          V_out,
    const void*     x_packed,     // [M][K/2] packed INT4 activations
    const float*    x_scale,     // [M][K/16] FP32 activation scales
    const void*     W_q_packed,  // [Q_dim][K/2] packed INT4 Q weights
    const void*     W_q_scale,   // __half* [Q_dim][K/16] Q weight scales
    const void*     W_k_packed,  // [KV_dim][K/2] packed INT4 K weights
    const void*     W_k_scale,   // __half* [KV_dim][K/16] K weight scales
    const void*     W_v_packed,  // [KV_dim][K/2] packed INT4 V weights
    const void*     W_v_scale,   // __half* [KV_dim][K/16] V weight scales
    int             K,
    int             Q_dim,        // query output dim (4096)
    int             KV_dim,       // key/value output dim (1024)
    int             M,            // batch size 1..16
    cudaStream_t    stream = 0);

// FP32-scale variant of fused_qkv_int4.
cudaError_t fused_qkv_int4(
    float*          Q_out,
    float*          K_out,
    float*          V_out,
    const uint8_t*  x_packed,
    const float*    x_scale,
    const uint8_t*  W_q_packed,
    const float*    W_q_scale,
    const uint8_t*  W_k_packed,
    const float*    W_k_scale,
    const uint8_t*  W_v_packed,
    const float*    W_v_scale,
    int             K,
    int             Q_dim,
    int             KV_dim,
    int             M,
    cudaStream_t    stream = 0);

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

// Apply repetition penalty to logits before sampling.
// logits: [V] in-place
// recent: [num_recent] recent token IDs (circular buffer, newest last)
// penalty: > 1.0, e.g. 1.1 = 10% penalty
cudaError_t apply_repetition_penalty(
    float*          logits,      // [V] in-place
    const int*     recent,      // [num_recent] recent token IDs
    int             num_recent,
    float           penalty,
    int             vocab,
    cudaStream_t    stream = 0);

// Fused: SwiGLU activation + INT8 quant
// Replaces: apply_swiglu → pack_int8 (2 kernels → 1 kernel)
cudaError_t fused_swiglu_quant(
    int8_t* out_i8,
    float* out_scale,
    const float* gate,
    const float* up,
    int N,
    cudaStream_t stream = 0);

// Batched INT4 GEMV: processes M sequences in parallel.
// Grid: N × M blocks, 32 threads/block.
// Weight loaded once per K-block, reused across M tokens.
cudaError_t gemv_int4_batched(
    float*          y_out,
    const uint8_t*  x_packed,
    const float*    x_scale,
    const uint8_t*  W_packed,
    const float*    W_scale,
    int             K,
    int             N,
    int             M,
    cudaStream_t    stream = 0);

// FP16 weight-scale variant of gemv_int4_batched.
// W_scale points to [N][K/16] __half. Halves weight-scale traffic.
// Same M=1..16 template dispatch. Activation scales stay FP32.
cudaError_t gemv_int4_batched_f16wsc(
    float*          y_out,
    const void*     x_packed,
    const float*    x_scale,
    const void*     W_packed,
    const void*     W_scale,       // __half* [N][K/16]
    int             K,
    int             N,
    int             M,
    cudaStream_t    stream = 0);

} // namespace kernels
} // namespace blackwell

#endif // BLACKWELL_KERNELS_H
