// src/kernels/gemm.cu — FP4 GEMM / GEMV for RTX 5060 Ti Blackwell SM_120
//
// GEMV (decode path):
//   y(1×N) = x(1×K) @ W(K×N)
//   x and W stored as FP4 E2M1 with per-block scales.
//   Dequant to FP16 in registers, accumulate in FP32.
//   Each block: 2 warps × 32 threads = 64 output elements.  K=64 fixed.
//
// GEMM (prefill path):
//   C(M×N) = A(M×K) @ B(K×N)
//   Grid: (M/16 × N/16) blocks, each 16×16 WMMA output.
//   K=64 tiled: 4× m16n16k16 MMA.

#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <mma.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace blackwell {
namespace kernels {
namespace {

__device__ __forceinline__ float dequant_fp4(__nv_fp4_e2m1 v, float scale) {
    return static_cast<float>(v) * scale;
}

static constexpr int kNumWarpsGEMM = 4;
static constexpr int kThreadsGEMM  = kNumWarpsGEMM * 32; // 128

static constexpr int kNumWarpsGEMV = 2;
static constexpr int kThreadsGEMV  = kNumWarpsGEMV * 32; // 64
static constexpr int kGEMVOutputsPerBlock = kThreadsGEMV; // 64

// ===========================================================================
// GEMM kernel: 16×16 output per block, K=64 tiled (4× WMMA m16n16k16).
//
// Shared mem: smem_a[16×64] + smem_b[64×16] = 2560 B  < 99 KB  ✓
// Row-major layout in smem for both A and B.
// WMMA row_major for both frag_a and frag_b.
// ===========================================================================
__launch_bounds__(kThreadsGEMM, 1)
__global__ void gemm_fp4_kernel(
    float* __restrict__ C_out,
    const __nv_fp4_e2m1* __restrict__ A_fp4,
    const __nv_fp4_e2m1* __restrict__ B_fp4,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    int M, int N_arg, int K_arg, const float* /*C_scale*/) {

    constexpr int TM = blackwell::kGEMMTileM;  // 16
    constexpr int TN = blackwell::kGEMMTileN;  // 16
    constexpr int TK = blackwell::kGEMMTileK;  // 64
    constexpr int B  = blackwell::kFP4BlockSize; // 16

    int block_row = blockIdx.x * TM;
    int block_col = blockIdx.y * TN;
    if (block_row >= M || block_col >= N_arg) return;

    __shared__ __half smem_a[TM * TK];  // 16×64 = 2048 B
    __shared__ __half smem_b[TK * TN];  // 64×16 = 2048 B

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    namespace wmma = nvcuda::wmma;

    // Row-major A and B: natural for row-major smem storage.
    wmma::fragment<wmma::matrix_a, TM, TN, 16, __half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, TM, TN, 16, __half, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, TM, TN, 16, float> frag_c;

    wmma::fill_fragment(frag_c, 0.0f);

    // K-tiling: TK=64 → 4 MMA iterations (each 16 in K).
    for (int k_start = 0; k_start < K_arg; k_start += TK) {
        // Load A[TM×TK] = 16×64: each thread covers (TK*TM)/128 = 8 elements.
        for (int i = threadIdx.x; i < TK * TM; i += kThreadsGEMM) {
            int m_l = i / TK;
            int k_l = i % TK;
            int m_g = block_row + m_l;
            int k_g = k_start + k_l;
            if (m_g < M && k_g < K_arg) {
                int idx = m_g * K_arg + k_g;
                int Nb = (K_arg + B - 1) / B;
                float s = A_scale[(m_g / B) * Nb + (k_g / B)];
                smem_a[m_l * TK + k_l] = __float2half(dequant_fp4(A_fp4[idx], s));
            }
        }
        __syncthreads();

        // Load B[TK×TN] = 64×16: each thread covers (TK*TN)/128 = 8 elements.
        for (int i = threadIdx.x; i < TK * TN; i += kThreadsGEMM) {
            int k_l = i / TN;
            int n_l = i % TN;
            int k_g = k_start + k_l;
            int n_g = block_col + n_l;
            if (k_g < K_arg && n_g < N_arg) {
                int idx = k_g * N_arg + n_g;
                int Nb = (N_arg + B - 1) / B;
                float s = B_scale[(k_g / B) * Nb + (n_g / B)];
                smem_b[k_l * TN + n_l] = __float2half(dequant_fp4(B_fp4[idx], s));
            }
        }
        __syncthreads();

        // 4× m16n16k16 MMA, each covering 16 in K.
        // smem_a row-major 16×64: &smem_a[col] with ldg=TK=64 loads rows 0..15, columns col..col+15.
        // smem_b row-major 64×16: &smem_b[row*TN] with ldg=TN=16 loads rows row..row+15, cols 0..15.
        for (int k_s = 0; k_s < TK; k_s += 16) {
            wmma::load_matrix_sync(frag_a, smem_a + k_s, TK);
            wmma::load_matrix_sync(frag_b, smem_b + k_s * TN, TN);
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }
        __syncthreads();
    }

    wmma::store_matrix_sync(
        C_out + block_row * N_arg + block_col,
        frag_c, N_arg, wmma::mem_row_major);

#else
    for (int i = threadIdx.x; i < TM * TN; i += kThreadsGEMM) {
        int m_l = i / TN, n_l = i % TN;
        if (block_row + m_l < M && block_col + n_l < N_arg)
            C_out[(block_row + m_l) * N_arg + block_col + n_l] = 0.0f;
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
    // TM=16, TN=16.  Validate that M,N are multiples of 16.
    if (M_arg % blackwell::kGEMMTileM != 0 ||
        N_arg % blackwell::kGEMMTileN != 0 ||
        K_arg % blackwell::kGEMMTileK != 0) {
        return cudaErrorInvalidValue;
    }

    dim3 grid(
        M_arg / blackwell::kGEMMTileM,
        N_arg / blackwell::kGEMMTileN
    );

    gemm_fp4_kernel<<<grid, kThreadsGEMM, 0, stream>>>(
        C,
        static_cast<const Fp4*>(A_fp4),
        static_cast<const Fp4*>(B_fp4),
        A_scale, B_scale, M_arg, N_arg, K_arg, nullptr);

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
