// src/kernels/fused_gate_up_int4.cu — Fused gate+up INT4 GEMV
//
// Replaces two separate GEMV calls with one fused kernel:
//   gemv_int4_batched_f16wsc(gate, x, x_sc, W_g, W_g_sc, H, I, M)  // 1 kernel
//   gemv_int4_batched_f16wsc(up,   x, x_sc, W_u, W_u_sc, H, I, M)  // 1 kernel
// → fused_gate_up_int4_f16wsc(gate, up, x, x_sc, W_g, W_g_sc, W_u, W_u_sc, H, I, M)  // 1 kernel
//
// Benefits:
//   - 1 kernel launch instead of 2 (halves kernel overhead)
//   - Activation x loaded once per K-block, used for both gate and up
//   - Weight loaded once per K-block, used for both outputs
//
// K must be multiple of 16. N (=I) must be multiple of 16.
// M must be 1..16.
//
// WScaleT = float (FP32 scales) or __half (FP16 scales, half weight-scale traffic).
// x_scale is always FP32 (activation scales).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

// Load weight scale as FP32 (handles both float and __half)
template <typename WScaleT>
__device__ __forceinline__ float load_wsc(const WScaleT* W_scale, size_t idx);

template <>
__device__ __forceinline__ float load_wsc<float>(const float* W_scale, size_t idx) {
    return W_scale[idx];
}

template <>
__device__ __forceinline__ float load_wsc<__half>(const __half* W_scale, size_t idx) {
    return __half2float(W_scale[idx]);
}

// Unpack 8 packed INT4 bytes into 4 INT32 lanes for dp4a.
// Each lane packs 4 nibbles as 4 INT8 bytes.
// Offset-binary: nibble value 0..15 represents INT4 -8..+7.
__device__ __forceinline__ void int4_8bytes_to_int4lanes(
    const uint8_t* bytes, int& l0, int& l1, int& l2, int& l3)
{
    // Lane 0 <- nibbles 0,1,2,3 (bytes 0,1)
    int b0 = bytes[0], b1 = bytes[1], b2 = bytes[2], b3 = bytes[3];
    int b4 = bytes[4], b5 = bytes[5], b6 = bytes[6], b7 = bytes[7];

    int n00 = (b0 & 0xF) - 8, n01 = ((b0 >> 4) & 0xF) - 8;
    int n02 = (b1 & 0xF) - 8, n03 = ((b1 >> 4) & 0xF) - 8;
    int n04 = (b2 & 0xF) - 8, n05 = ((b2 >> 4) & 0xF) - 8;
    int n06 = (b3 & 0xF) - 8, n07 = ((b3 >> 4) & 0xF) - 8;
    int n08 = (b4 & 0xF) - 8, n09 = ((b4 >> 4) & 0xF) - 8;
    int n10 = (b5 & 0xF) - 8, n11 = ((b5 >> 4) & 0xF) - 8;
    int n12 = (b6 & 0xF) - 8, n13 = ((b6 >> 4) & 0xF) - 8;
    int n14 = (b7 & 0xF) - 8, n15 = ((b7 >> 4) & 0xF) - 8;

    l0 = (n00 & 0xFF) | ((n01 & 0xFF) << 8) | ((n02 & 0xFF) << 16) | ((n03 & 0xFF) << 24);
    l1 = (n04 & 0xFF) | ((n05 & 0xFF) << 8) | ((n06 & 0xFF) << 16) | ((n07 & 0xFF) << 24);
    l2 = (n08 & 0xFF) | ((n09 & 0xFF) << 8) | ((n10 & 0xFF) << 16) | ((n11 & 0xFF) << 24);
    l3 = (n12 & 0xFF) | ((n13 & 0xFF) << 8) | ((n14 & 0xFF) << 16) | ((n15 & 0xFF) << 24);
}

// Grid: I × M blocks, 32 threads/block.
// Each block computes one output row of gate and one output row of up,
// for one of M sequences in the batch.
//
// Key optimization: x_i4 and x_scale loaded ONCE per K-block, used for
// both gate and up. Weight loaded ONCE per K-block for each of gate and up.
template <int M, typename WScaleT>
__launch_bounds__(32, 8)
__global__ void fused_gate_up_int4_kernel(
    float* __restrict__  gate_out,    // [M][I] output
    float* __restrict__  up_out,     // [M][I] output
    const uint8_t* __restrict__ x_packed,    // [M][K/2] packed INT4 activations
    const float* __restrict__ x_scale,       // [M][K/16] FP32 activation scales
    const uint8_t* __restrict__ W_g_packed,   // [I][K/2] packed INT4 gate weights
    const WScaleT* __restrict__ W_g_scale,    // [I][K/16] gate weight scales
    const uint8_t* __restrict__ W_u_packed,   // [I][K/2] packed INT4 up weights
    const WScaleT* __restrict__ W_u_scale,    // [I][K/16] up weight scales
    int K, int N)  // N = I (number of hidden units)
{
    constexpr int B = 16;    // quantization block size
    constexpr int PB = 8;    // packed bytes per block (B/2)

    int n_out = blockIdx.x;   // output row index (0..N-1)
    if (n_out >= N) return;
    int tid = threadIdx.x;

    int num_K_blks = K / B;

    // Accumulators for gate and up, for each of M sequences
    float gate_acc[M];
    float up_acc[M];
    #pragma unroll
    for (int mi = 0; mi < M; ++mi) {
        gate_acc[mi] = 0.0f;
        up_acc[mi] = 0.0f;
    }

    // Stride-32 loop: each thread handles scattered K-blocks
    // Weight data strided by N for gate and up, activation strided by M
    for (int kb = tid; kb < num_K_blks; kb += 32) {
        // ── Load gate weights (one K-block, one output row) ────────────
        const uint8_t* wg_ptr = &W_g_packed[(size_t)n_out * (K / 2) + kb * PB];
        uint2 wg_packed = *reinterpret_cast<const uint2*>(wg_ptr);
        float wg_sc = load_wsc(W_g_scale, (size_t)n_out * num_K_blks + kb);

        // Unpack gate nibbles → 4 INT32 lanes
        const uint8_t* wgb = reinterpret_cast<const uint8_t*>(&wg_packed);
        int wgl0, wgl1, wgl2, wgl3;
        int4_8bytes_to_int4lanes(wgb, wgl0, wgl1, wgl2, wgl3);

        // ── Load up weights (one K-block, same output row) ─────────────
        const uint8_t* wu_ptr = &W_u_packed[(size_t)n_out * (K / 2) + kb * PB];
        uint2 wu_packed = *reinterpret_cast<const uint2*>(wu_ptr);
        float wu_sc = load_wsc(W_u_scale, (size_t)n_out * num_K_blks + kb);

        // Unpack up nibbles → 4 INT32 lanes
        const uint8_t* wub = reinterpret_cast<const uint8_t*>(&wu_packed);
        int wul0, wul1, wul2, wul3;
        int4_8bytes_to_int4lanes(wub, wul0, wul1, wul2, wul3);

        // ── For each of M sequences, compute gate += x·W_g and up += x·W_u ─
        // x_i4 and x_scale loaded ONCE per K-block per sequence
        #pragma unroll
        for (int mi = 0; mi < M; ++mi) {
            // Load activation
            const uint8_t* x_ptr = &x_packed[(size_t)mi * (K / 2) + kb * PB];
            uint2 x_packed_val = *reinterpret_cast<const uint2*>(x_ptr);
            float x_sc = x_scale[(size_t)mi * num_K_blks + kb];
            float prod_scale_g = wg_sc * x_sc;
            float prod_scale_u = wu_sc * x_sc;

            // Unpack activation nibbles
            const uint8_t* xb = reinterpret_cast<const uint8_t*>(&x_packed_val);
            int xl0, xl1, xl2, xl3;
            int4_8bytes_to_int4lanes(xb, xl0, xl1, xl2, xl3);

            // dp4a: gate
            int sg = 0;
            sg = __dp4a(wgl0, xl0, sg);
            sg = __dp4a(wgl1, xl1, sg);
            sg = __dp4a(wgl2, xl2, sg);
            sg = __dp4a(wgl3, xl3, sg);
            gate_acc[mi] += static_cast<float>(sg) * prod_scale_g;

            // dp4a: up
            int su = 0;
            su = __dp4a(wul0, xl0, su);
            su = __dp4a(wul1, xl1, su);
            su = __dp4a(wul2, xl2, su);
            su = __dp4a(wul3, xl3, su);
            up_acc[mi] += static_cast<float>(su) * prod_scale_u;
        }
    }

    // Warp shuffle reduction for gate and up
    #pragma unroll
    for (int mi = 0; mi < M; ++mi) {
        gate_acc[mi] += __shfl_xor_sync(0xffffffff, gate_acc[mi], 16);
        gate_acc[mi] += __shfl_xor_sync(0xffffffff, gate_acc[mi], 8);
        gate_acc[mi] += __shfl_xor_sync(0xffffffff, gate_acc[mi], 4);
        gate_acc[mi] += __shfl_xor_sync(0xffffffff, gate_acc[mi], 2);
        gate_acc[mi] += __shfl_xor_sync(0xffffffff, gate_acc[mi], 1);

        up_acc[mi] += __shfl_xor_sync(0xffffffff, up_acc[mi], 16);
        up_acc[mi] += __shfl_xor_sync(0xffffffff, up_acc[mi], 8);
        up_acc[mi] += __shfl_xor_sync(0xffffffff, up_acc[mi], 4);
        up_acc[mi] += __shfl_xor_sync(0xffffffff, up_acc[mi], 2);
        up_acc[mi] += __shfl_xor_sync(0xffffffff, up_acc[mi], 1);
    }

    // Thread 0 writes all M gate and up outputs
    if (tid == 0) {
        #pragma unroll
        for (int mi = 0; mi < M; ++mi) {
            gate_out[(size_t)mi * N + n_out] = gate_acc[mi];
            up_out[(size_t)mi * N + n_out] = up_acc[mi];
        }
    }
}

}  // anonymous namespace

// =============================================================================
// Public API — FP16 weight scales (__half)
// =============================================================================
// gate_out  [M][I] — SwiGLU gate projection output
// up_out    [M][I] — SwiGLU up projection output
// x_packed  [M][K/2] — packed INT4 activations
// x_scale   [M][K/16] — FP32 activation scales
// W_g       [I][K/2] — packed INT4 gate weights
// W_g_sc    [I][K/16] — __half weight scales
// W_u       [I][K/2] — packed INT4 up weights
// W_u_sc    [I][K/16] — __half weight scales
// K = H (input dim), I = hidden dim (12288 for Qwen3-8B)
cudaError_t fused_gate_up_int4_f16wsc(
    float*          gate_out,
    float*          up_out,
    const void*     x_packed,
    const float*    x_scale,
    const void*     W_g_packed,
    const void*     W_g_scale,     // __half* [I][K/16]
    const void*     W_u_packed,
    const void*     W_u_scale,     // __half* [I][K/16]
    int             K,
    int             N,             // = I (hidden dim)
    int             M,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || N % 16 != 0 || M < 1 || M > 16)
        return cudaErrorInvalidValue;

    const __half* Wgsc = static_cast<const __half*>(W_g_scale);
    const __half* Wusc = static_cast<const __half*>(W_u_scale);
    const uint8_t* xp  = static_cast<const uint8_t*>(x_packed);
    const uint8_t* Wgp = static_cast<const uint8_t*>(W_g_packed);
    const uint8_t* Wup = static_cast<const uint8_t*>(W_u_packed);

    dim3 grid(N, 1);
    switch (M) {
        case  1: fused_gate_up_int4_kernel< 1, __half><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, xp, x_scale, Wgp, Wgsc, Wup, Wusc, K, N); break;
        case  2: fused_gate_up_int4_kernel< 2, __half><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, xp, x_scale, Wgp, Wgsc, Wup, Wusc, K, N); break;
        case  3: fused_gate_up_int4_kernel< 3, __half><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, xp, x_scale, Wgp, Wgsc, Wup, Wusc, K, N); break;
        case  4: fused_gate_up_int4_kernel< 4, __half><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, xp, x_scale, Wgp, Wgsc, Wup, Wusc, K, N); break;
        case  5: fused_gate_up_int4_kernel< 5, __half><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, xp, x_scale, Wgp, Wgsc, Wup, Wusc, K, N); break;
        case  6: fused_gate_up_int4_kernel< 6, __half><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, xp, x_scale, Wgp, Wgsc, Wup, Wusc, K, N); break;
        case  7: fused_gate_up_int4_kernel< 7, __half><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, xp, x_scale, Wgp, Wgsc, Wup, Wusc, K, N); break;
        case  8: fused_gate_up_int4_kernel< 8, __half><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, xp, x_scale, Wgp, Wgsc, Wup, Wusc, K, N); break;
        case  9: fused_gate_up_int4_kernel< 9, __half><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, xp, x_scale, Wgp, Wgsc, Wup, Wusc, K, N); break;
        case 10: fused_gate_up_int4_kernel<10, __half><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, xp, x_scale, Wgp, Wgsc, Wup, Wusc, K, N); break;
        case 11: fused_gate_up_int4_kernel<11, __half><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, xp, x_scale, Wgp, Wgsc, Wup, Wusc, K, N); break;
        case 12: fused_gate_up_int4_kernel<12, __half><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, xp, x_scale, Wgp, Wgsc, Wup, Wusc, K, N); break;
        case 13: fused_gate_up_int4_kernel<13, __half><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, xp, x_scale, Wgp, Wgsc, Wup, Wusc, K, N); break;
        case 14: fused_gate_up_int4_kernel<14, __half><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, xp, x_scale, Wgp, Wgsc, Wup, Wusc, K, N); break;
        case 15: fused_gate_up_int4_kernel<15, __half><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, xp, x_scale, Wgp, Wgsc, Wup, Wusc, K, N); break;
        case 16: fused_gate_up_int4_kernel<16, __half><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, xp, x_scale, Wgp, Wgsc, Wup, Wusc, K, N); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaPeekAtLastError();
}

// =============================================================================
// Public API — FP32 weight scales (float)
// =============================================================================
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
    int             N,             // = I (hidden dim)
    int             M,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || N % 16 != 0 || M < 1 || M > 16)
        return cudaErrorInvalidValue;

    dim3 grid(N, 1);
    switch (M) {
        case  1: fused_gate_up_int4_kernel< 1, float><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, x_packed, x_scale, W_g_packed, W_g_scale, W_u_packed, W_u_scale, K, N); break;
        case  2: fused_gate_up_int4_kernel< 2, float><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, x_packed, x_scale, W_g_packed, W_g_scale, W_u_packed, W_u_scale, K, N); break;
        case  3: fused_gate_up_int4_kernel< 3, float><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, x_packed, x_scale, W_g_packed, W_g_scale, W_u_packed, W_u_scale, K, N); break;
        case  4: fused_gate_up_int4_kernel< 4, float><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, x_packed, x_scale, W_g_packed, W_g_scale, W_u_packed, W_u_scale, K, N); break;
        case  5: fused_gate_up_int4_kernel< 5, float><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, x_packed, x_scale, W_g_packed, W_g_scale, W_u_packed, W_u_scale, K, N); break;
        case  6: fused_gate_up_int4_kernel< 6, float><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, x_packed, x_scale, W_g_packed, W_g_scale, W_u_packed, W_u_scale, K, N); break;
        case  7: fused_gate_up_int4_kernel< 7, float><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, x_packed, x_scale, W_g_packed, W_g_scale, W_u_packed, W_u_scale, K, N); break;
        case  8: fused_gate_up_int4_kernel< 8, float><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, x_packed, x_scale, W_g_packed, W_g_scale, W_u_packed, W_u_scale, K, N); break;
        case  9: fused_gate_up_int4_kernel< 9, float><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, x_packed, x_scale, W_g_packed, W_g_scale, W_u_packed, W_u_scale, K, N); break;
        case 10: fused_gate_up_int4_kernel<10, float><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, x_packed, x_scale, W_g_packed, W_g_scale, W_u_packed, W_u_scale, K, N); break;
        case 11: fused_gate_up_int4_kernel<11, float><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, x_packed, x_scale, W_g_packed, W_g_scale, W_u_packed, W_u_scale, K, N); break;
        case 12: fused_gate_up_int4_kernel<12, float><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, x_packed, x_scale, W_g_packed, W_g_scale, W_u_packed, W_u_scale, K, N); break;
        case 13: fused_gate_up_int4_kernel<13, float><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, x_packed, x_scale, W_g_packed, W_g_scale, W_u_packed, W_u_scale, K, N); break;
        case 14: fused_gate_up_int4_kernel<14, float><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, x_packed, x_scale, W_g_packed, W_g_scale, W_u_packed, W_u_scale, K, N); break;
        case 15: fused_gate_up_int4_kernel<15, float><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, x_packed, x_scale, W_g_packed, W_g_scale, W_u_packed, W_u_scale, K, N); break;
        case 16: fused_gate_up_int4_kernel<16, float><<<grid, dim3(32), 0, stream>>>(gate_out, up_out, x_packed, x_scale, W_g_packed, W_g_scale, W_u_packed, W_u_scale, K, N); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaPeekAtLastError();
}

}  // namespace kernels
}  // namespace blackwell
