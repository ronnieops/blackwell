// src/kernels/fused_qkv_int4.cu — Fused QKV INT4 GEMV
//
// Replaces three separate GEMV calls with one fused kernel:
//   gemv(Q, x, x_sc, W_q, W_q_sc, H, Q_dim)     // 1 kernel
//   gemv(K, x, x_sc, W_k, W_k_sc, H, KV_dim)     // 1 kernel
//   gemv(V, x, x_sc, W_v, W_v_sc, H, KV_dim)     // 1 kernel
// → fused_qkv_int4_f16wsc(Q, K, V, x, x_sc, W_q, W_k, W_v, ..., H, Q_dim, KV_dim, M)
//
// Benefits:
//   - 1 kernel launch instead of 3 (66% fewer launches)
//   - Activation x loaded ONCE per K-block, used for Q + K + V
//
// Grid: max(Q_dim, KV_dim) × 1 blocks, 32 threads/block.
// Block n (0..Q_dim-1): compute Q row n.
// Block n (0..KV_dim-1): compute K row n AND V row n.
// Since Q_dim > KV_dim (4096 > 1024), blocks 0..KV_dim-1 compute Q+K+V (x loaded
// once, used 3×), blocks KV_dim..Q_dim-1 compute Q only.
//
// K must be multiple of 16. Q_dim and KV_dim must be multiples of 16.
// M must be 1..16.
//
// WScaleT = float (FP32 scales) or __half (FP16 scales).

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

// Grid: max(Q_dim, KV_dim) × 1 blocks, 32 threads/block.
// Each block computes:
//   - Q output row blockIdx.x (if blockIdx.x < Q_dim)
//   - K output row blockIdx.x (if blockIdx.x < KV_dim)
//   - V output row blockIdx.x (if blockIdx.x < KV_dim)
//
// Key optimization: x_i4 and x_scale loaded ONCE per K-block, reused across
// all applicable projections (Q, K, V).
template <int M, typename WScaleT>
__launch_bounds__(32, 8)
__global__ void fused_qkv_int4_kernel(
    float* __restrict__       Q_out,    // [M][Q_dim] output
    float* __restrict__       K_out,    // [M][KV_dim] output
    float* __restrict__       V_out,    // [M][KV_dim] output
    const uint8_t* __restrict__ x_packed,    // [M][K/2] packed INT4 activations
    const float* __restrict__ x_scale,       // [M][K/16] FP32 activation scales
    const uint8_t* __restrict__ W_q_packed,   // [Q_dim][K/2] packed INT4 Q weights
    const WScaleT* __restrict__ W_q_scale,    // [Q_dim][K/16] Q weight scales
    const uint8_t* __restrict__ W_k_packed,   // [KV_dim][K/2] packed INT4 K weights
    const WScaleT* __restrict__ W_k_scale,    // [KV_dim][K/16] K weight scales
    const uint8_t* __restrict__ W_v_packed,   // [KV_dim][K/2] packed INT4 V weights
    const WScaleT* __restrict__ W_v_scale,    // [KV_dim][K/16] V weight scales
    int K, int Q_dim, int KV_dim)
{
    constexpr int B = 16;    // quantization block size
    constexpr int PB = 8;    // packed bytes per block (B/2)

    int n = blockIdx.x;   // output row index
    int tid = threadIdx.x;

    bool do_q = (n < Q_dim);
    bool do_kv = (n < KV_dim);
    if (!do_q && !do_kv) return;

    int num_K_blks = K / B;

    // Accumulators for Q, K, V, for each of M sequences
    float q_acc[M], k_acc[M], v_acc[M];
    #pragma unroll
    for (int mi = 0; mi < M; ++mi) {
        q_acc[mi] = 0.0f;
        k_acc[mi] = 0.0f;
        v_acc[mi] = 0.0f;
    }

    // Stride-32 loop: each thread handles scattered K-blocks
    for (int kb = tid; kb < num_K_blks; kb += 32) {
        // ── Load Q weight (one K-block, one output row) ────────────
        float wq_sc = 0.0f;
        int wql0 = 0, wql1 = 0, wql2 = 0, wql3 = 0;
        if (do_q) {
            const uint8_t* wq_ptr = &W_q_packed[(size_t)n * (K / 2) + kb * PB];
            uint2 wq_packed = *reinterpret_cast<const uint2*>(wq_ptr);
            wq_sc = load_wsc(W_q_scale, (size_t)n * num_K_blks + kb);
            const uint8_t* wqb = reinterpret_cast<const uint8_t*>(&wq_packed);
            int4_8bytes_to_int4lanes(wqb, wql0, wql1, wql2, wql3);
        }

        // ── Load K weight (one K-block, same output row) ────────────
        float wk_sc = 0.0f;
        int wkl0 = 0, wkl1 = 0, wkl2 = 0, wkl3 = 0;
        if (do_kv) {
            const uint8_t* wk_ptr = &W_k_packed[(size_t)n * (K / 2) + kb * PB];
            uint2 wk_packed = *reinterpret_cast<const uint2*>(wk_ptr);
            wk_sc = load_wsc(W_k_scale, (size_t)n * num_K_blks + kb);
            const uint8_t* wkb = reinterpret_cast<const uint8_t*>(&wk_packed);
            int4_8bytes_to_int4lanes(wkb, wkl0, wkl1, wkl2, wkl3);
        }

        // ── Load V weight (one K-block, same output row) ────────────
        float wv_sc = 0.0f;
        int wvl0 = 0, wvl1 = 0, wvl2 = 0, wvl3 = 0;
        if (do_kv) {
            const uint8_t* wv_ptr = &W_v_packed[(size_t)n * (K / 2) + kb * PB];
            uint2 wv_packed = *reinterpret_cast<const uint2*>(wv_ptr);
            wv_sc = load_wsc(W_v_scale, (size_t)n * num_K_blks + kb);
            const uint8_t* wvb = reinterpret_cast<const uint8_t*>(&wv_packed);
            int4_8bytes_to_int4lanes(wvb, wvl0, wvl1, wvl2, wvl3);
        }

        // ── For each of M sequences ──
        // x_i4 and x_scale loaded ONCE per K-block per sequence
        #pragma unroll
        for (int mi = 0; mi < M; ++mi) {
            // Load activation (shared across Q, K, V)
            const uint8_t* x_ptr = &x_packed[(size_t)mi * (K / 2) + kb * PB];
            uint2 x_packed_val = *reinterpret_cast<const uint2*>(x_ptr);
            float x_sc = x_scale[(size_t)mi * num_K_blks + kb];

            // Unpack activation nibbles
            const uint8_t* xb = reinterpret_cast<const uint8_t*>(&x_packed_val);
            int xl0, xl1, xl2, xl3;
            int4_8bytes_to_int4lanes(xb, xl0, xl1, xl2, xl3);

            // dp4a: Q
            if (do_q) {
                int sq = 0;
                sq = __dp4a(wql0, xl0, sq);
                sq = __dp4a(wql1, xl1, sq);
                sq = __dp4a(wql2, xl2, sq);
                sq = __dp4a(wql3, xl3, sq);
                q_acc[mi] += static_cast<float>(sq) * (wq_sc * x_sc);
            }

            // dp4a: K
            if (do_kv) {
                int sk = 0;
                sk = __dp4a(wkl0, xl0, sk);
                sk = __dp4a(wkl1, xl1, sk);
                sk = __dp4a(wkl2, xl2, sk);
                sk = __dp4a(wkl3, xl3, sk);
                k_acc[mi] += static_cast<float>(sk) * (wk_sc * x_sc);

                // dp4a: V (same activation, different weight)
                int sv = 0;
                sv = __dp4a(wvl0, xl0, sv);
                sv = __dp4a(wvl1, xl1, sv);
                sv = __dp4a(wvl2, xl2, sv);
                sv = __dp4a(wvl3, xl3, sv);
                v_acc[mi] += static_cast<float>(sv) * (wv_sc * x_sc);
            }
        }
    }

    // Warp shuffle reduction for Q, K, V
    #pragma unroll
    for (int mi = 0; mi < M; ++mi) {
        q_acc[mi] += __shfl_xor_sync(0xffffffff, q_acc[mi], 16);
        q_acc[mi] += __shfl_xor_sync(0xffffffff, q_acc[mi], 8);
        q_acc[mi] += __shfl_xor_sync(0xffffffff, q_acc[mi], 4);
        q_acc[mi] += __shfl_xor_sync(0xffffffff, q_acc[mi], 2);
        q_acc[mi] += __shfl_xor_sync(0xffffffff, q_acc[mi], 1);

        k_acc[mi] += __shfl_xor_sync(0xffffffff, k_acc[mi], 16);
        k_acc[mi] += __shfl_xor_sync(0xffffffff, k_acc[mi], 8);
        k_acc[mi] += __shfl_xor_sync(0xffffffff, k_acc[mi], 4);
        k_acc[mi] += __shfl_xor_sync(0xffffffff, k_acc[mi], 2);
        k_acc[mi] += __shfl_xor_sync(0xffffffff, k_acc[mi], 1);

        v_acc[mi] += __shfl_xor_sync(0xffffffff, v_acc[mi], 16);
        v_acc[mi] += __shfl_xor_sync(0xffffffff, v_acc[mi], 8);
        v_acc[mi] += __shfl_xor_sync(0xffffffff, v_acc[mi], 4);
        v_acc[mi] += __shfl_xor_sync(0xffffffff, v_acc[mi], 2);
        v_acc[mi] += __shfl_xor_sync(0xffffffff, v_acc[mi], 1);
    }

    // Thread 0 writes outputs
    if (tid == 0) {
        #pragma unroll
        for (int mi = 0; mi < M; ++mi) {
            if (do_q) Q_out[(size_t)mi * Q_dim + n] = q_acc[mi];
            if (do_kv) {
                K_out[(size_t)mi * KV_dim + n] = k_acc[mi];
                V_out[(size_t)mi * KV_dim + n] = v_acc[mi];
            }
        }
    }
}

}  // anonymous namespace

// =============================================================================
// Public API — FP16 weight scales (__half)
// =============================================================================
// Q_out     [M][Q_dim] — query projection output
// K_out     [M][KV_dim] — key projection output
// V_out     [M][KV_dim] — value projection output
// x_packed  [M][K/2] — packed INT4 activations
// x_scale   [M][K/16] — FP32 activation scales
// W_q       [Q_dim][K/2] — packed INT4 Q weights
// W_q_sc    [Q_dim][K/16] — __half Q weight scales
// W_k       [KV_dim][K/2] — packed INT4 K weights
// W_k_sc    [KV_dim][K/16] — __half K weight scales
// W_v       [KV_dim][K/2] — packed INT4 V weights
// W_v_sc    [KV_dim][K/16] — __half V weight scales
// K = H (input dim), Q_dim (4096), KV_dim (1024)
cudaError_t fused_qkv_int4_f16wsc(
    float*          Q_out,
    float*          K_out,
    float*          V_out,
    const void*     x_packed,
    const float*    x_scale,
    const void*     W_q_packed,
    const void*     W_q_scale,     // __half* [Q_dim][K/16]
    const void*     W_k_packed,
    const void*     W_k_scale,     // __half* [KV_dim][K/16]
    const void*     W_v_packed,
    const void*     W_v_scale,     // __half* [KV_dim][K/16]
    int             K,
    int             Q_dim,
    int             KV_dim,
    int             M,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || Q_dim % 16 != 0 || KV_dim % 16 != 0 || M < 1 || M > 16)
        return cudaErrorInvalidValue;

    const __half* Wqsc = static_cast<const __half*>(W_q_scale);
    const __half* Wksc = static_cast<const __half*>(W_k_scale);
    const __half* Wvsc = static_cast<const __half*>(W_v_scale);
    const uint8_t* xp  = static_cast<const uint8_t*>(x_packed);
    const uint8_t* Wqp = static_cast<const uint8_t*>(W_q_packed);
    const uint8_t* Wkp = static_cast<const uint8_t*>(W_k_packed);
    const uint8_t* Wvp = static_cast<const uint8_t*>(W_v_packed);

    int grid_dim = (Q_dim > KV_dim) ? Q_dim : KV_dim;
    dim3 grid(grid_dim, 1);
    switch (M) {
        case  1: fused_qkv_int4_kernel< 1, __half><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, xp, x_scale, Wqp, Wqsc, Wkp, Wksc, Wvp, Wvsc, K, Q_dim, KV_dim); break;
        case  2: fused_qkv_int4_kernel< 2, __half><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, xp, x_scale, Wqp, Wqsc, Wkp, Wksc, Wvp, Wvsc, K, Q_dim, KV_dim); break;
        case  3: fused_qkv_int4_kernel< 3, __half><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, xp, x_scale, Wqp, Wqsc, Wkp, Wksc, Wvp, Wvsc, K, Q_dim, KV_dim); break;
        case  4: fused_qkv_int4_kernel< 4, __half><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, xp, x_scale, Wqp, Wqsc, Wkp, Wksc, Wvp, Wvsc, K, Q_dim, KV_dim); break;
        case  5: fused_qkv_int4_kernel< 5, __half><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, xp, x_scale, Wqp, Wqsc, Wkp, Wksc, Wvp, Wvsc, K, Q_dim, KV_dim); break;
        case  6: fused_qkv_int4_kernel< 6, __half><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, xp, x_scale, Wqp, Wqsc, Wkp, Wksc, Wvp, Wvsc, K, Q_dim, KV_dim); break;
        case  7: fused_qkv_int4_kernel< 7, __half><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, xp, x_scale, Wqp, Wqsc, Wkp, Wksc, Wvp, Wvsc, K, Q_dim, KV_dim); break;
        case  8: fused_qkv_int4_kernel< 8, __half><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, xp, x_scale, Wqp, Wqsc, Wkp, Wksc, Wvp, Wvsc, K, Q_dim, KV_dim); break;
        case  9: fused_qkv_int4_kernel< 9, __half><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, xp, x_scale, Wqp, Wqsc, Wkp, Wksc, Wvp, Wvsc, K, Q_dim, KV_dim); break;
        case 10: fused_qkv_int4_kernel<10, __half><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, xp, x_scale, Wqp, Wqsc, Wkp, Wksc, Wvp, Wvsc, K, Q_dim, KV_dim); break;
        case 11: fused_qkv_int4_kernel<11, __half><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, xp, x_scale, Wqp, Wqsc, Wkp, Wksc, Wvp, Wvsc, K, Q_dim, KV_dim); break;
        case 12: fused_qkv_int4_kernel<12, __half><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, xp, x_scale, Wqp, Wqsc, Wkp, Wksc, Wvp, Wvsc, K, Q_dim, KV_dim); break;
        case 13: fused_qkv_int4_kernel<13, __half><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, xp, x_scale, Wqp, Wqsc, Wkp, Wksc, Wvp, Wvsc, K, Q_dim, KV_dim); break;
        case 14: fused_qkv_int4_kernel<14, __half><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, xp, x_scale, Wqp, Wqsc, Wkp, Wksc, Wvp, Wvsc, K, Q_dim, KV_dim); break;
        case 15: fused_qkv_int4_kernel<15, __half><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, xp, x_scale, Wqp, Wqsc, Wkp, Wksc, Wvp, Wvsc, K, Q_dim, KV_dim); break;
        case 16: fused_qkv_int4_kernel<16, __half><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, xp, x_scale, Wqp, Wqsc, Wkp, Wksc, Wvp, Wvsc, K, Q_dim, KV_dim); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaPeekAtLastError();
}

// =============================================================================
// Public API — FP32 weight scales (float)
// =============================================================================
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
    cudaStream_t    stream)
{
    if (K % 16 != 0 || Q_dim % 16 != 0 || KV_dim % 16 != 0 || M < 1 || M > 16)
        return cudaErrorInvalidValue;

    int grid_dim = (Q_dim > KV_dim) ? Q_dim : KV_dim;
    dim3 grid(grid_dim, 1);
    switch (M) {
        case  1: fused_qkv_int4_kernel< 1, float><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, x_packed, x_scale, W_q_packed, W_q_scale, W_k_packed, W_k_scale, W_v_packed, W_v_scale, K, Q_dim, KV_dim); break;
        case  2: fused_qkv_int4_kernel< 2, float><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, x_packed, x_scale, W_q_packed, W_q_scale, W_k_packed, W_k_scale, W_v_packed, W_v_scale, K, Q_dim, KV_dim); break;
        case  3: fused_qkv_int4_kernel< 3, float><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, x_packed, x_scale, W_q_packed, W_q_scale, W_k_packed, W_k_scale, W_v_packed, W_v_scale, K, Q_dim, KV_dim); break;
        case  4: fused_qkv_int4_kernel< 4, float><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, x_packed, x_scale, W_q_packed, W_q_scale, W_k_packed, W_k_scale, W_v_packed, W_v_scale, K, Q_dim, KV_dim); break;
        case  5: fused_qkv_int4_kernel< 5, float><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, x_packed, x_scale, W_q_packed, W_q_scale, W_k_packed, W_k_scale, W_v_packed, W_v_scale, K, Q_dim, KV_dim); break;
        case  6: fused_qkv_int4_kernel< 6, float><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, x_packed, x_scale, W_q_packed, W_q_scale, W_k_packed, W_k_scale, W_v_packed, W_v_scale, K, Q_dim, KV_dim); break;
        case  7: fused_qkv_int4_kernel< 7, float><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, x_packed, x_scale, W_q_packed, W_q_scale, W_k_packed, W_k_scale, W_v_packed, W_v_scale, K, Q_dim, KV_dim); break;
        case  8: fused_qkv_int4_kernel< 8, float><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, x_packed, x_scale, W_q_packed, W_q_scale, W_k_packed, W_k_scale, W_v_packed, W_v_scale, K, Q_dim, KV_dim); break;
        case  9: fused_qkv_int4_kernel< 9, float><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, x_packed, x_scale, W_q_packed, W_q_scale, W_k_packed, W_k_scale, W_v_packed, W_v_scale, K, Q_dim, KV_dim); break;
        case 10: fused_qkv_int4_kernel<10, float><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, x_packed, x_scale, W_q_packed, W_q_scale, W_k_packed, W_k_scale, W_v_packed, W_v_scale, K, Q_dim, KV_dim); break;
        case 11: fused_qkv_int4_kernel<11, float><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, x_packed, x_scale, W_q_packed, W_q_scale, W_k_packed, W_k_scale, W_v_packed, W_v_scale, K, Q_dim, KV_dim); break;
        case 12: fused_qkv_int4_kernel<12, float><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, x_packed, x_scale, W_q_packed, W_q_scale, W_k_packed, W_k_scale, W_v_packed, W_v_scale, K, Q_dim, KV_dim); break;
        case 13: fused_qkv_int4_kernel<13, float><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, x_packed, x_scale, W_q_packed, W_q_scale, W_k_packed, W_k_scale, W_v_packed, W_v_scale, K, Q_dim, KV_dim); break;
        case 14: fused_qkv_int4_kernel<14, float><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, x_packed, x_scale, W_q_packed, W_q_scale, W_k_packed, W_k_scale, W_v_packed, W_v_scale, K, Q_dim, KV_dim); break;
        case 15: fused_qkv_int4_kernel<15, float><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, x_packed, x_scale, W_q_packed, W_q_scale, W_k_packed, W_k_scale, W_v_packed, W_v_scale, K, Q_dim, KV_dim); break;
        case 16: fused_qkv_int4_kernel<16, float><<<grid, dim3(32), 0, stream>>>(Q_out, K_out, V_out, x_packed, x_scale, W_q_packed, W_q_scale, W_k_packed, W_k_scale, W_v_packed, W_v_scale, K, Q_dim, KV_dim); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaPeekAtLastError();
}

}  // namespace kernels
}  // namespace blackwell
