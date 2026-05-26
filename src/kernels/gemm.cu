// src/kernels/gemm.cu — FP4 GEMM / GEMV for RTX 5060 Ti Blackwell SM_120
//
// GEMM (prefill path):
//   C(M×N) = A(M×K) @ B(K×N)
//   CTA tile: 128×128×64, 8 warps (256 threads)
//   Each warp: 2 M-frags × 4 N-frags = 8 output fragments per warp
//   K=64 tiled: 4× m16n16k16 MMA per K-slice, loop over K in 64-element chunks
//   Vectorized uint4 FP4 global loads → register dequant → FP16 smem
//
// GEMV (decode path):
//   y(1×N) = x(1×K) @ W(K×N)
//   K=64 fixed (single tile). Dynamic K tiling separated into gemv_fp4_batched.

#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <mma.h>
#include <cuda_pipeline.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace blackwell {
namespace kernels {
namespace {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int kNumWarpsGEMV = 2;
static constexpr int kThreadsGEMV  = kNumWarpsGEMV * 32; // 64
static constexpr int kGEMVOutputsPerBlock = kThreadsGEMV; // 64

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ float dequant_fp4(__nv_fp4_e2m1 v, float scale) {
    return static_cast<float>(v) * scale;
}

// Load 16 FP4 values via uint4 vector load, dequant with block-scale, store FP16.
// Each uint4 = 16 bytes = 16 __nv_fp4_e2m1 values.
// All 16 values share the same scale (belong to same 16-element block).
template <int TK>
__device__ __forceinline__ void load_dequant_stripe(
    __half* __restrict__ smem_dst,   // FP16 output in smem
    const __nv_fp4_e2m1* __restrict__ src_base,  // FP4 global base
    int row_global, int col_start, int row_stride,
    float scale, int M_max, int K_max) {

    // 16 consecutive FP4 values = one uint4 load
    alignas(16) uint8_t buf[16];
    const uint4* src_vec = reinterpret_cast<const uint4*>(
        src_base + row_global * row_stride + col_start);
    *reinterpret_cast<uint4*>(buf) = *src_vec;

    __nv_fp4_e2m1* vals = reinterpret_cast<__nv_fp4_e2m1*>(buf);
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        smem_dst[j] = __float2half(static_cast<float>(vals[j]) * scale);
    }
}

// ---------------------------------------------------------------------------
// GEMM kernel: 128×128×64 CTA, 8 warps, vectorized FP4 loads, register dequant
//
// Shared memory:
//   smem_A[128×64] halves (16384 B) — dequantized FP16 from A tile
//   smem_B[64×128] halves (16384 B) — dequantized FP16 from B tile
//   Total: 32768 B (32 KB) — well under 99 KB limit
//
// Warp-to-fragment mapping:
//   8 warps → 4 along M × 2 along N
//   Each warp: (128/16)/4 = 2 M-frags × (128/16)/2 = 4 N-frags = 8 output frags
// ---------------------------------------------------------------------------
__launch_bounds__(blackwell::kGEMMThreads, 1)
__global__ void gemm_fp4_kernel(
    float* __restrict__ C_out,
    const __nv_fp4_e2m1* __restrict__ A_fp4,
    const __nv_fp4_e2m1* __restrict__ B_fp4,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    int M, int N_, int K_) {

    constexpr int TM = blackwell::kGEMMTileM;  // 128
    constexpr int TN = blackwell::kGEMMTileN;  // 128
    constexpr int TK = blackwell::kGEMMTileK;  // 64
    constexpr int B  = blackwell::kFP4BlockSize; // 16

    // Warp/fragment decomposition
    constexpr int FM_PER_WARP = blackwell::kFragsPerWarpM;   // 2
    constexpr int FN_PER_WARP = blackwell::kFragsPerWarpN;   // 4

    int block_row = blockIdx.x * TM;
    int block_col = blockIdx.y * TN;
    if (block_row >= M || block_col >= N_) return;

    // Shared memory: dequantized FP16 tiles
    __shared__ __half smem_A[TM * TK];  // 128×64 = 16384 B
    __shared__ __half smem_B[TK * TN];  // 64×128 = 16384 B

    int warp_id = threadIdx.x / 32;
    int warp_m = warp_id % 4;
    int warp_n = warp_id / 4;

    int frag_m_start = warp_m * FM_PER_WARP;  // 0, 2, 4, or 6
    int frag_n_start = warp_n * FN_PER_WARP;  // 0 or 4

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    namespace wmma = nvcuda::wmma;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c[FM_PER_WARP * FN_PER_WARP];

    // Initialize accumulators
    #pragma unroll
    for (int i = 0; i < FM_PER_WARP * FN_PER_WARP; ++i) {
        wmma::fill_fragment(frag_c[i], 0.0f);
    }

    // Number of scale blocks along K
    int num_K_blks = (K_ + B - 1) / B;

    // Outer K-tiling loop
    for (int k_start = 0; k_start < K_; k_start += TK) {
        // ----- Load A tile [TM × TK]: 8192 FP4 values -----
        // Each thread: 2× uint4 (32 FP4 values), each uint4 = one 16-element scale block
        for (int i = threadIdx.x; i < (TM * TK) / 16; i += blackwell::kGEMMThreads) {
            int flat16 = i * 16;
            int m_loc = flat16 / TK;
            int k_loc = flat16 % TK;
            int m_glob = block_row + m_loc;
            int k_glob = k_start + k_loc;

            if (m_glob < M && k_glob < K_) {
                int blk_m = m_glob / B;
                int blk_k = k_glob / B;
                float sc = A_scale[blk_m * num_K_blks + blk_k];
                load_dequant_stripe<TK>(
                    &smem_A[m_loc * TK + k_loc],
                    A_fp4, m_glob, k_glob, K_, sc, M, K_);
            }
        }
        __syncthreads();

        // ----- Load B tile [TK × TN]: 8192 FP4 values -----
        int num_N_blks = (N_ + B - 1) / B;
        for (int i = threadIdx.x; i < (TK * TN) / 16; i += blackwell::kGEMMThreads) {
            int flat16 = i * 16;
            int k_loc = flat16 / TN;
            int n_loc = flat16 % TN;
            int k_glob = k_start + k_loc;
            int n_glob = block_col + n_loc;

            if (k_glob < K_ && n_glob < N_) {
                int blk_k = k_glob / B;
                int blk_n = n_glob / B;
                float sc = B_scale[blk_k * num_N_blks + blk_n];
                load_dequant_stripe<TN>(
                    &smem_B[k_loc * TN + n_loc],
                    B_fp4, k_glob, n_glob, N_, sc, K_, N_);
            }
        }
        __syncthreads();

        // ----- 4× m16n16k16 WMMA (cover K=64) -----
        for (int k_s = 0; k_s < TK; k_s += 16) {
            #pragma unroll
            for (int fm = 0; fm < FM_PER_WARP; ++fm) {
                int abs_fm = frag_m_start + fm;
                #pragma unroll
                for (int fn = 0; fn < FN_PER_WARP; ++fn) {
                    int abs_fn = frag_n_start + fn;

                    // A fragment: rows [abs_fm*16, abs_fm*16+16), cols [k_s, k_s+16)
                    wmma::load_matrix_sync(frag_a,
                        smem_A + abs_fm * 16 * TK + k_s, TK);
                    // B fragment: rows [k_s, k_s+16), cols [abs_fn*16, abs_fn*16+16)
                    wmma::load_matrix_sync(frag_b,
                        smem_B + k_s * TN + abs_fn * 16, TN);

                    wmma::mma_sync(frag_c[fm * FN_PER_WARP + fn],
                        frag_a, frag_b,
                        frag_c[fm * FN_PER_WARP + fn]);
                }
            }
        }
        __syncthreads();
    }

    // ----- Store results: each warp writes 8× 16×16 output fragments -----
    #pragma unroll
    for (int fm = 0; fm < FM_PER_WARP; ++fm) {
        int abs_fm = frag_m_start + fm;
        #pragma unroll
        for (int fn = 0; fn < FN_PER_WARP; ++fn) {
            int abs_fn = frag_n_start + fn;
            float* out_ptr = C_out
                + (block_row + abs_fm * 16) * N_
                + (block_col + abs_fn * 16);
            wmma::store_matrix_sync(
                out_ptr, frag_c[fm * FN_PER_WARP + fn],
                N_, wmma::mem_row_major);
        }
    }
#else
    // Fallback for non-SM_120: write zeros
    for (int i = threadIdx.x; i < TM * TN; i += blackwell::kGEMMThreads) {
        int m_l = i / TN, n_l = i % TN;
        if (block_row + m_l < M && block_col + n_l < N_)
            C_out[(block_row + m_l) * N_ + block_col + n_l] = 0.0f;
    }
    (void)A_fp4; (void)B_fp4; (void)A_scale; (void)B_scale;
#endif
}

// ===========================================================================
// GEMV kernel: decode path — one token × weight matrix
// ===========================================================================
__launch_bounds__(kThreadsGEMV, 1)
__global__ void gemv_fp4_kernel(
    float* __restrict__ y_out,
    const __nv_fp4_e2m1* __restrict__ x_fp4,
    const float* __restrict__ x_scale,
    const __nv_fp4_e2m1* __restrict__ W_fp4,
    const float* __restrict__ W_scale,
    int in_features, int out_features) {

    using Fp4 = __nv_fp4_e2m1;
    constexpr int K = blackwell::kGEMMTileK;   // 64
    constexpr int B  = blackwell::kFP4BlockSize; // 16
    static_assert(K % B == 0, "");

    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int n_out   = warp_id * 32 + lane_id;  // 0..63
    if (n_out >= out_features) return;

    __half w_hp[K / B];
    #pragma unroll
    for (int kb = 0; kb < K / B; ++kb) {
        int k_idx = kb * B + lane_id;
        int w_idx = n_out * in_features + k_idx;
        float scale = W_scale[n_out / B];
        float v = (k_idx < in_features)
                  ? dequant_fp4(W_fp4[w_idx], scale)
                  : 0.0f;
        w_hp[kb] = __float2half(v);
    }
    __syncthreads();

    __half x_hp[K / B];
    if (lane_id < K / B) {
        float v = dequant_fp4(x_fp4[lane_id * B], x_scale[lane_id]);
        x_hp[lane_id] = __float2half(v);
    }
    #pragma unroll
    for (int kb = 0; kb < K / B; ++kb) {
        x_hp[kb] = __shfl_sync(0xffffffff, x_hp[kb], 0);
    }
    __syncthreads();

    float acc = 0.0f;
    #pragma unroll
    for (int kb = 0; kb < K / B; ++kb) {
        acc += __half2float(x_hp[kb]) * __half2float(w_hp[kb]);
    }
    y_out[n_out] = acc;
}

} // anonymous namespace

// ===========================================================================
// Public API
// ===========================================================================

cudaError_t gemm_fp4_block_scaled(
    float* C, const void* A_fp4, const float* A_scale,
    const void* B_fp4, const float* B_scale,
    int M_arg, int N_arg, int K_arg, cudaStream_t stream) {

    using Fp4 = __nv_fp4_e2m1;
    // TM=128, TN=128. Validate multiples.
    if (M_arg % blackwell::kGEMMTileM != 0 ||
        N_arg % blackwell::kGEMMTileN != 0 ||
        K_arg % blackwell::kGEMMTileK != 0) {
        return cudaErrorInvalidValue;
    }

    dim3 grid(
        M_arg / blackwell::kGEMMTileM,
        N_arg / blackwell::kGEMMTileN
    );

    gemm_fp4_kernel<<<grid, blackwell::kGEMMThreads, 0, stream>>>(
        C,
        static_cast<const Fp4*>(A_fp4),
        static_cast<const Fp4*>(B_fp4),
        A_scale, B_scale, M_arg, N_arg, K_arg);

    return cudaPeekAtLastError();
}

cudaError_t gemv_fp4(
    float* y_out, const void* x_fp4, const float* x_scale,
    const void* W_fp4, const float* W_scale,
    int in_features_arg, int out_features, cudaStream_t stream) {

    using Fp4 = __nv_fp4_e2m1;
    if (in_features_arg % blackwell::kFP4BlockSize != 0 ||
        in_features_arg != blackwell::kGEMMTileK) {
        return cudaErrorInvalidValue;
    }

    int nb = (out_features + kGEMVOutputsPerBlock - 1) / kGEMVOutputsPerBlock;
    gemv_fp4_kernel<<<dim3(nb), dim3(kThreadsGEMV), 0, stream>>>(
        y_out,
        static_cast<const Fp4*>(x_fp4), x_scale,
        static_cast<const Fp4*>(W_fp4), W_scale,
        in_features_arg, out_features);

    return cudaPeekAtLastError();
}

cudaError_t dispatch_matmul(
    float* C, const void* A, const void* B,
    const float* A_scale, const float* B_scale,
    int M_arg, int N_arg, int K_arg, KernelMode mode, cudaStream_t stream) {

    if (mode == KernelMode::Prefill) {
        return gemm_fp4_block_scaled(C, A, A_scale, B, B_scale, M_arg, N_arg, K_arg, stream);
    } else {
        return gemv_fp4(C, A, A_scale, B, B_scale, K_arg, N_arg, stream);
    }
}

} // namespace kernels
} // namespace blackwell
