// src/kernels/gemm.cu — FP4 GEMM / GEMV for RTX 5060 Ti Blackwell SM_120
//
// GEMM (prefill path):
//   C(M×N) = A(M×K) @ B(K×N)
//   Templated CTA sizes: (128×128×64, 8 warps) and (64×64×64, 4 warps)
//   Each warp: 2 M-frags × 4/2 N-frags = 8/4 output fragments per warp
//   K=64 tiled: 4×/4× m16n16k16 MMA per K-slice, loop over K in 64-element chunks
//   Vectorized uint4 FP4 global loads → register dequant → FP16 smem
//
// GEMV (decode path):
//   y(1×N) = x(1×K) @ W(K×N)
//   256 threads/block, grid = ceil(N/256).

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
static constexpr int kGEMVBlock = 256;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ void dequant_stripe_from_smem(
    __half* __restrict__ fp16_dst,
    const uint8_t* __restrict__ raw_src,
    float scale) {
    alignas(16) uint8_t buf[16];
    *reinterpret_cast<uint4*>(buf) = *reinterpret_cast<const uint4*>(raw_src);
    __nv_fp4_e2m1* vals = reinterpret_cast<__nv_fp4_e2m1*>(buf);
    #pragma unroll
    for (int j = 0; j < 16; ++j)
        fp16_dst[j] = __float2half(static_cast<float>(vals[j]) * scale);
}

template <int TK>
__device__ __forceinline__ void load_dequant_stripe(
    __half* __restrict__ smem_dst,
    const __nv_fp4_e2m1* __restrict__ src_base,
    int row_global, int col_start, int row_stride,
    float scale, int, int) {
    alignas(16) uint8_t buf[16];
    const uint4* src_vec = reinterpret_cast<const uint4*>(
        src_base + row_global * row_stride + col_start);
    *reinterpret_cast<uint4*>(buf) = *src_vec;
    __nv_fp4_e2m1* vals = reinterpret_cast<__nv_fp4_e2m1*>(buf);
    #pragma unroll
    for (int j = 0; j < 16; ++j)
        smem_dst[j] = __float2half(static_cast<float>(vals[j]) * scale);
}

// ---------------------------------------------------------------------------
// Templated GEMM kernel
//   TM, TN: CTA tile dimensions (multiple of 16)
//   TK: K-tile dimension (64)
//   WARPS_M, WARPS_N: warp grid within CTA
//   THREADS = WARPS_M * WARPS_N * 32
//
// Shared memory layout (dynamic):
//   raw_A[TM×TK]  — raw FP4 A (bytes)
//   raw_B[TK×TN]  — raw FP4 B (bytes)
//   FP16 A double-buffer [2×TM×TK×2 bytes]
//   FP16 B double-buffer [2×TK×TN×2 bytes]
// ---------------------------------------------------------------------------
template <int TM, int TN, int TK, int WARPS_M, int WARPS_N>
__launch_bounds__(WARPS_M * WARPS_N * 32, 1)
__global__ void gemm_fp4_kernel_tmpl(
    float* __restrict__ C_out,
    const __nv_fp4_e2m1* __restrict__ A_fp4,
    const __nv_fp4_e2m1* __restrict__ B_fp4,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    int M, int N_, int K_) {

    constexpr int B   = blackwell::kFP4BlockSize; // 16
    constexpr int THR = WARPS_M * WARPS_N * 32;
    constexpr int FM_PER_WARP = TM / 16 / WARPS_M;
    constexpr int FN_PER_WARP = TN / 16 / WARPS_N;

    int block_row = blockIdx.x * TM;
    int block_col = blockIdx.y * TN;
    if (block_row >= M || block_col >= N_) return;

    extern __shared__ uint8_t smem_dyn[];
    uint8_t* smem_raw_A = smem_dyn;
    uint8_t* smem_raw_B = smem_dyn + TM * TK;
    __half* smem_A0 = reinterpret_cast<__half*>(smem_dyn + TM * TK + TK * TN);
    __half* smem_A1 = smem_A0 + TM * TK;
    __half* smem_B0 = smem_A1 + TM * TK;
    __half* smem_B1 = smem_B0 + TK * TN;

    int warp_id = threadIdx.x / 32;
    int warp_m = warp_id % WARPS_M;
    int warp_n = warp_id / WARPS_M;
    int frag_m_start = warp_m * FM_PER_WARP;
    int frag_n_start = warp_n * FN_PER_WARP;
    int num_tiles = K_ / TK;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    namespace wmma = nvcuda::wmma;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c[FM_PER_WARP * FN_PER_WARP];

    #pragma unroll
    for (int i = 0; i < FM_PER_WARP * FN_PER_WARP; ++i)
        wmma::fill_fragment(frag_c[i], 0.0f);

    int num_K_blks = (K_ + B - 1) / B;
    int num_N_blks = (N_ + B - 1) / B;
    const int tid = threadIdx.x;

    // ----- First K-tile: load + dequant → smem_A0/smem_B0 -----
    for (int i = tid; i < (TM * TK) / 16; i += THR) {
        int flat16 = i * 16;
        int m_loc = flat16 / TK, k_loc = flat16 % TK;
        int m_glob = block_row + m_loc;
        float sc = A_scale[(m_glob / B) * num_K_blks + (k_loc / B)];
        load_dequant_stripe<TK>(&smem_A0[m_loc * TK + k_loc],
            A_fp4, m_glob, k_loc, K_, sc, M, K_);
    }
    for (int i = tid; i < (TK * TN) / 16; i += THR) {
        int flat16 = i * 16;
        int k_loc = flat16 / TN, n_loc = flat16 % TN;
        int n_glob = block_col + n_loc;
        float sc = B_scale[(k_loc / B) * num_N_blks + (n_glob / B)];
        load_dequant_stripe<TN>(&smem_B0[k_loc * TN + n_loc],
            B_fp4, k_loc, n_glob, N_, sc, K_, N_);
    }
    __syncthreads();

    // ----- Pipelined K-tiles -----
    int pipe_stage = 0;
    for (int t = 0; t < num_tiles; ++t) {
        int next_k_start = (t + 1) * TK;

        if (next_k_start < K_) {
            for (int i = tid; i < (TM * TK) / 16; i += THR) {
                int flat16 = i * 16;
                int m_loc = flat16 / TK, k_loc = flat16 % TK;
                int k_glob = next_k_start + k_loc;
                const void* src = A_fp4 + (block_row + m_loc) * K_ + k_glob;
                __pipeline_memcpy_async(smem_raw_A + m_loc * TK + k_loc, src, 16);
            }
            for (int i = tid; i < (TK * TN) / 16; i += THR) {
                int flat16 = i * 16;
                int k_loc = flat16 / TN, n_loc = flat16 % TN;
                int k_glob = next_k_start + k_loc;
                int n_glob = block_col + n_loc;
                const void* src = B_fp4 + k_glob * N_ + n_glob;
                __pipeline_memcpy_async(smem_raw_B + k_loc * TN + n_loc, src, 16);
            }
            __pipeline_commit();
        }

        // WMMA on current buffer
        __half* cur_A = (pipe_stage == 0) ? smem_A0 : smem_A1;
        __half* cur_B = (pipe_stage == 0) ? smem_B0 : smem_B1;
        for (int k_s = 0; k_s < TK; k_s += 16) {
            #pragma unroll
            for (int fm = 0; fm < FM_PER_WARP; ++fm) {
                int abs_fm = frag_m_start + fm;
                #pragma unroll
                for (int fn = 0; fn < FN_PER_WARP; ++fn) {
                    int abs_fn = frag_n_start + fn;
                    wmma::load_matrix_sync(frag_a,
                        cur_A + abs_fm * 16 * TK + k_s, TK);
                    wmma::load_matrix_sync(frag_b,
                        cur_B + k_s * TN + abs_fn * 16, TN);
                    wmma::mma_sync(frag_c[fm * FN_PER_WARP + fn],
                        frag_a, frag_b, frag_c[fm * FN_PER_WARP + fn]);
                }
            }
        }

        // Dequant next
        if (next_k_start < K_) {
            __pipeline_wait_prior(0);
            __syncthreads();
            __half* next_A = (pipe_stage == 0) ? smem_A1 : smem_A0;
            __half* next_B = (pipe_stage == 0) ? smem_B1 : smem_B0;
            for (int i = tid; i < (TM * TK) / 16; i += THR) {
                int flat16 = i * 16;
                int m_loc = flat16 / TK, k_loc = flat16 % TK;
                int k_glob = next_k_start + k_loc;
                int m_glob = block_row + m_loc;
                float sc = A_scale[(m_glob / B) * num_K_blks + (k_glob / B)];
                dequant_stripe_from_smem(&next_A[m_loc * TK + k_loc],
                    &smem_raw_A[m_loc * TK + k_loc], sc);
            }
            for (int i = tid; i < (TK * TN) / 16; i += THR) {
                int flat16 = i * 16;
                int k_loc = flat16 / TN, n_loc = flat16 % TN;
                int k_glob = next_k_start + k_loc;
                int n_glob = block_col + n_loc;
                float sc = B_scale[(k_glob / B) * num_N_blks + (n_glob / B)];
                dequant_stripe_from_smem(&next_B[k_loc * TN + n_loc],
                    &smem_raw_B[k_loc * TN + n_loc], sc);
            }
            __syncthreads();
        }
        pipe_stage ^= 1;
    }

    // ----- Store results -----
    #pragma unroll
    for (int fm = 0; fm < FM_PER_WARP; ++fm) {
        int abs_fm = frag_m_start + fm;
        #pragma unroll
        for (int fn = 0; fn < FN_PER_WARP; ++fn) {
            int abs_fn = frag_n_start + fn;
            wmma::store_matrix_sync(
                C_out + (block_row + abs_fm * 16) * N_ + (block_col + abs_fn * 16),
                frag_c[fm * FN_PER_WARP + fn], N_, wmma::mem_row_major);
        }
    }
#else
    for (int i = threadIdx.x; i < TM * TN; i += THR) {
        int m_l = i / TN, n_l = i % TN;
        if (block_row + m_l < M && block_col + n_l < N_)
            C_out[(block_row + m_l) * N_ + block_col + n_l] = 0.0f;
    }
    (void)A_fp4; (void)B_fp4; (void)A_scale; (void)B_scale;
#endif
}

// Explicit instantiations
template __global__ void gemm_fp4_kernel_tmpl<128,128,64,4,2>(
    float*, const __nv_fp4_e2m1*, const __nv_fp4_e2m1*,
    const float*, const float*, int, int, int);
template __global__ void gemm_fp4_kernel_tmpl<64,64,64,2,2>(
    float*, const __nv_fp4_e2m1*, const __nv_fp4_e2m1*,
    const float*, const float*, int, int, int);

// ===========================================================================
// GEMV kernel: decode path
// ===========================================================================
__launch_bounds__(kGEMVBlock, 1)
__global__ void gemv_fp4_kernel(
    float* __restrict__ y_out,
    const __nv_fp4_e2m1* __restrict__ x_fp4,
    const float* __restrict__ x_scale,
    const __nv_fp4_e2m1* __restrict__ W_fp4,
    const float* __restrict__ W_scale,
    int K, int N) {

    constexpr int B = blackwell::kFP4BlockSize;
    int n_out = blockIdx.x * kGEMVBlock + threadIdx.x;
    if (n_out >= N) return;

    int n_blk = n_out / B;
    int num_N_blks = (N + B - 1) / B;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        int k_blk = k / B;
        float xv = static_cast<float>(x_fp4[k]) * x_scale[k_blk];
        float wv = static_cast<float>(W_fp4[k * N + n_out])
                 * W_scale[k_blk * num_N_blks + n_blk];
        acc += xv * wv;
    }
    y_out[n_out] = acc;
}

// Large CTA kernel: use gemm_fp4_kernel_tmpl<128,128,64,4,2>

} // anonymous namespace

// ===========================================================================
// Public API
// ===========================================================================

// Large CTA (128×128×64) — for M≥128 prefill
cudaError_t gemm_fp4_block_scaled(
    float* C, const void* A_fp4, const float* A_scale,
    const void* B_fp4, const float* B_scale,
    int M_arg, int N_arg, int K_arg, cudaStream_t stream) {

    using Fp4 = __nv_fp4_e2m1;
    constexpr int TM = 128, TN = 128, TK = 64;
    if (M_arg % TM != 0 || N_arg % TN != 0 || K_arg % TK != 0)
        return cudaErrorInvalidValue;

    dim3 grid(M_arg / TM, N_arg / TN);
    constexpr int kSmem = TM * TK + TK * TN  // raw
                        + 2 * TM * TK * 2    // FP16 A double
                        + 2 * TK * TN * 2;   // FP16 B double

    static bool attr_set = false;
    if (!attr_set) {
        cudaError_t e = cudaFuncSetAttribute(
            gemm_fp4_kernel_tmpl<128,128,64,4,2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
        if (e != cudaSuccess) return e;
        attr_set = true;
    }

    gemm_fp4_kernel_tmpl<128,128,64,4,2><<<grid, 256, kSmem, stream>>>(
        C, static_cast<const Fp4*>(A_fp4), static_cast<const Fp4*>(B_fp4),
        A_scale, B_scale, M_arg, N_arg, K_arg);
    return cudaPeekAtLastError();
}

// Small CTA (64×64×64) — for M<128 prefill, better SM util on small N
cudaError_t gemm_fp4_block_scaled_small(
    float* C, const void* A_fp4, const float* A_scale,
    const void* B_fp4, const float* B_scale,
    int M_arg, int N_arg, int K_arg, cudaStream_t stream) {

    using Fp4 = __nv_fp4_e2m1;
    constexpr int TM = 64, TN = 64, TK = 64;
    if (M_arg % TM != 0 || N_arg % TN != 0 || K_arg % TK != 0)
        return cudaErrorInvalidValue;

    dim3 grid(M_arg / TM, N_arg / TN);
    constexpr int kSmem = TM * TK + TK * TN
                        + 2 * TM * TK * 2
                        + 2 * TK * TN * 2;

    static bool attr_set = false;
    if (!attr_set) {
        cudaError_t e = cudaFuncSetAttribute(
            gemm_fp4_kernel_tmpl<64,64,64,2,2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
        if (e != cudaSuccess) return e;
        attr_set = true;
    }

    gemm_fp4_kernel_tmpl<64,64,64,2,2><<<grid, 128, kSmem, stream>>>(
        C, static_cast<const Fp4*>(A_fp4), static_cast<const Fp4*>(B_fp4),
        A_scale, B_scale, M_arg, N_arg, K_arg);
    return cudaPeekAtLastError();
}

cudaError_t gemv_fp4(
    float* y_out, const void* x_fp4, const float* x_scale,
    const void* W_fp4, const float* W_scale,
    int in_features_arg, int out_features, cudaStream_t stream) {

    using Fp4 = __nv_fp4_e2m1;
    if (in_features_arg % blackwell::kFP4BlockSize != 0)
        return cudaErrorInvalidValue;

    int nb = (out_features + kGEMVBlock - 1) / kGEMVBlock;
    gemv_fp4_kernel<<<dim3(nb), dim3(kGEMVBlock), 0, stream>>>(
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
        // Route: small CTA for N<4096 (better SM util on small-N GEMMs),
        // large CTA for N>=4096 (gate/up with N=6144 need more compute)
        if (N_arg < 4096 || M_arg < 128)
            return gemm_fp4_block_scaled_small(C, A, A_scale, B, B_scale,
                                              M_arg, N_arg, K_arg, stream);
        else
            return gemm_fp4_block_scaled(C, A, A_scale, B, B_scale,
                                        M_arg, N_arg, K_arg, stream);
    } else {
        return gemv_fp4(C, A, A_scale, B, B_scale, K_arg, N_arg, stream);
    }
}

} // namespace kernels
} // namespace blackwell