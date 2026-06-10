// src/kernels/fused_int4_ops.cu — Fused INT4 decode kernels
//
// Replaces per-layer kernel sequences with single-fusion kernels:
//   fused_rmsnorm_quant_int4:  fused_rmsnorm + quantize_int4  (2→1)
//   fused_swiglu_quant_int4:   apply_swiglu + quantize_int4  (2→1)
//
// Build: cmake --build build --parallel

#include <cuda_runtime.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

constexpr int THREADS = 256;
constexpr int B = 16;         // INT4 block size
constexpr int EPT = 16;        // elements per thread (256*16 = 4096)

// ─────────────────────────────────────────────────────────────────────────
// fused_rmsnorm_quant_int4_kernel
//
// Single-block kernel: RMSNorm + INT4 pack (block-16, absmax/7)
//
// Input:  proj [N] FP32, weight [N] FP32
// Output: x_out [N/2] packed INT4, x_out_scale [N/16] FP32 scales
//
// Replaces: fused_rmsnorm + quantize_int4  (2 kernels → 1)
//
// Phase 1: load values, compute sum_sq for RMSNorm
// Phase 2: normalize, find absmax per block of 16, compute scale, quantize
// ─────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_max_f(float val) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, off));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum_f(float val) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    return val;
}

__launch_bounds__(THREADS, 1)
__global__ void fused_rmsnorm_quant_int4_kernel(
    uint8_t* __restrict__ x_out,
    float* __restrict__ x_out_scale,
    const float* __restrict__ proj,
    const float* __restrict__ weight,
    int N, float eps)
{
    int tid = threadIdx.x;
    constexpr int NE = EPT;

    // Phase 1: load + sum_sq
    float vals[NE];
    float sum_sq = 0.0f;

    #pragma unroll
    for (int e = 0; e < NE; ++e) {
        int idx = tid + e * THREADS;
        if (idx < N) {
            vals[e] = proj[idx];
            sum_sq += vals[e] * vals[e];
        } else {
            vals[e] = 0.0f;
        }
    }

    sum_sq = warp_reduce_sum_f(sum_sq);

    // Cross-warp reduce (8 warps)
    __shared__ float warp_sums[8];
    if ((tid & 31) == 0) warp_sums[tid >> 5] = sum_sq;
    __syncthreads();

    float block_sum = (tid < 8) ? warp_sums[tid] : 0.0f;
    block_sum = warp_reduce_sum_f(block_sum);

    __shared__ float s_rstd;
    if (tid == 0) {
        s_rstd = rsqrtf(block_sum / static_cast<float>(N) + eps);
    }
    __syncthreads();
    float rstd = s_rstd;

    // Phase 2: normalize, find block absmax, quantize to INT4
    #pragma unroll
    for (int e = 0; e < NE; ++e) {
        int idx = tid + e * THREADS;
        if (idx < N) {
            float normed = vals[e] * weight[idx] * rstd;

            // Find absmax per 16-element block (lane_in_blk = idx % 16)
            float abs_val = fabsf(normed);
            int lane_in_blk = tid & (B - 1);

            // Half-warp max reduction (16 consecutive lanes = 1 warp half)
            float d;
            d = __shfl_down_sync(0xffffffff, abs_val, 8);
            if (lane_in_blk < 8) abs_val = fmaxf(abs_val, d);
            d = __shfl_down_sync(0xffffffff, abs_val, 4);
            if (lane_in_blk < 4) abs_val = fmaxf(abs_val, d);
            d = __shfl_down_sync(0xffffffff, abs_val, 2);
            if (lane_in_blk < 2) abs_val = fmaxf(abs_val, d);
            d = __shfl_down_sync(0xffffffff, abs_val, 1);
            if (lane_in_blk == 0) abs_val = fmaxf(abs_val, d);

            if (lane_in_blk == 0) {
                float scale = fmaxf(abs_val / 7.0f, 1e-9f);
                x_out_scale[idx / B] = scale;
            }
            __syncwarp();

            // Quantize to INT4 (range -7..7)
            float sc = x_out_scale[idx / B];
            float q = normed / sc;
            q = fminf(7.0f, fmaxf(-7.0f, roundf(q)));
            int nib = static_cast<int>(q) + 8;  // offset-binary: -7→1, 0→8, 7→15

            // Pack: byte at idx/2, low nib if even, high nib if odd
            int byte_idx = idx / 2;
            if (lane_in_blk == 0) {
                // This thread handles the low nib of byte[byte_idx]
                // But we need all 16 lanes to cooperate for the byte
                // Alternative: each thread writes its own nib to a shared byte
            }
        }
    }

    // Phase 3: parallel byte packing
    // Each thread handles 2 elements (1 byte) at byte_idx = tid + e*THREADS
    // Need to make sure even/odd elements don't collide
    #pragma unroll
    for (int e = 0; e < NE; ++e) {
        int byte_idx = tid + e * THREADS;  // byte index
        if (byte_idx >= N / 2) continue;

        // Two elements: lo = byte_idx*2, hi = byte_idx*2+1
        int lo_idx = byte_idx * 2;
        int hi_idx = lo_idx + 1;

        // Recompute normalization (can't reuse from phase 2 without smem)
        // Alternative: store normalized values in registers via smem
        // Better: do both norm+quantize in one pass, store to smem, then pack

        // For simplicity: use smem to store quantized values
    }
}

// Simpler approach: 2-pass with smem
__launch_bounds__(THREADS, 1)
__global__ void fused_rmsnorm_quant_int4_v2_kernel(
    uint8_t* __restrict__ x_out,
    float* __restrict__ x_out_scale,
    const float* __restrict__ proj,
    const float* __restrict__ weight,
    int N, float eps)
{
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    constexpr int NE = EPT;
    constexpr int NUM_BLKS = THREADS * NE / B;  // 256*16/16 = 256 blocks

    // Phase 1: load + sum_sq
    float vals[NE];
    float sum_sq = 0.0f;

    #pragma unroll
    for (int e = 0; e < NE; ++e) {
        int idx = tid + e * THREADS;
        if (idx < N) {
            vals[e] = proj[idx];
            sum_sq += vals[e] * vals[e];
        } else {
            vals[e] = 0.0f;
        }
    }

    sum_sq = warp_reduce_sum_f(sum_sq);

    __shared__ float warp_sums[8];
    if ((tid & 31) == 0) warp_sums[tid >> 5] = sum_sq;
    __syncthreads();

    float block_sum = (tid < 8) ? warp_sums[tid] : 0.0f;
    block_sum = warp_reduce_sum_f(block_sum);

    __shared__ float s_rstd;
    if (tid == 0) s_rstd = rsqrtf(block_sum / static_cast<float>(N) + eps);
    __syncthreads();
    float rstd = s_rstd;

    // Store normalized values to smem
    float* smem_norm = smem;
    #pragma unroll
    for (int e = 0; e < NE; ++e) {
        int idx = tid + e * THREADS;
        smem_norm[idx] = (idx < N) ? vals[e] * weight[idx] * rstd : 0.0f;
    }
    __syncthreads();

    // Phase 2: compute block scales from NORMALIZED values (smem_norm)
    int num_blocks = (N + B - 1) / B;
    for (int blk_id = tid; blk_id < num_blocks; blk_id += THREADS) {
        int blk_start = blk_id * B;
        float absmax = 0.0f;
        for (int i = 0; i < B && blk_start + i < N; ++i) {
            absmax = fmaxf(absmax, fabsf(smem_norm[blk_start + i]));
        }
        x_out_scale[blk_id] = fmaxf(absmax / 7.0f, 1e-9f);
    }
    __syncthreads();

    // Phase 3: quantize + pack (each thread handles 2 elements = 1 byte)
    int num_bytes = N / 2;
    #pragma unroll
    for (int e = 0; e < NE; ++e) {
        int byte_idx = tid + e * THREADS;
        if (byte_idx >= num_bytes) continue;

        int lo_idx = byte_idx * 2;
        int hi_idx = lo_idx + 1;

        float lo_val = (lo_idx < N) ? smem_norm[lo_idx] : 0.0f;
        float hi_val = (hi_idx < N) ? smem_norm[hi_idx] : 0.0f;

        float lo_sc = x_out_scale[lo_idx / B];
        float hi_sc = x_out_scale[hi_idx / B];

        float lo_q = fminf(7.0f, fmaxf(-7.0f, roundf(lo_val / lo_sc)));
        float hi_q = fminf(7.0f, fmaxf(-7.0f, roundf(hi_val / hi_sc)));

        uint8_t lo_nib = static_cast<uint8_t>(lo_q + 8);  // -7..7 → 1..15
        uint8_t hi_nib = static_cast<uint8_t>(hi_q + 8);

        x_out[byte_idx] = lo_nib | (hi_nib << 4);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// fused_swiglu_quant_int4_kernel
//
// Single-block kernel: SwiGLU activation + INT4 quant (block-16, absmax/7)
//
// Input:  gate [N] FP32, up [N] FP32
// Output: x_out [N/2] packed INT4, x_out_scale [N/16] FP32 scales
//
// Replaces: apply_swiglu + quantize_int4  (2 kernels → 1)
// ─────────────────────────────────────────────────────────────────────────
__launch_bounds__(THREADS, 1)
__global__ void fused_swiglu_quant_int4_kernel(
    uint8_t* __restrict__ x_out,
    float* __restrict__ x_out_scale,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    int N)
{
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    constexpr int NE = EPT;

    // Phase 1: SwiGLU activation + store to smem
    float vals[NE];
    #pragma unroll
    for (int e = 0; e < NE; ++e) {
        int idx = tid + e * THREADS;
        if (idx < N) {
            float g = gate[idx];
            float u = up[idx];
            // silu(x) = x * sigmoid(x)
            float s = 1.0f / (1.0f + expf(-g));
            vals[e] = g * s * u;
        } else {
            vals[e] = 0.0f;
        }
    }

    // Store to smem for block scale computation
    float* smem_act = smem;
    #pragma unroll
    for (int e = 0; e < NE; ++e) {
        int idx = tid + e * THREADS;
        smem_act[idx] = vals[e];
    }
    __syncthreads();

    // Phase 2: compute block scales
    int num_blocks = (N + B - 1) / B;
    for (int blk_id = tid; blk_id < num_blocks; blk_id += THREADS) {
        int blk_start = blk_id * B;
        float absmax = 0.0f;
        for (int i = 0; i < B && blk_start + i < N; ++i) {
            absmax = fmaxf(absmax, fabsf(smem_act[blk_start + i]));
        }
        x_out_scale[blk_id] = fmaxf(absmax / 7.0f, 1e-9f);
    }
    __syncthreads();

    // Phase 3: quantize + pack
    int num_bytes = N / 2;
    #pragma unroll
    for (int e = 0; e < NE; ++e) {
        int byte_idx = tid + e * THREADS;
        if (byte_idx >= num_bytes) continue;

        int lo_idx = byte_idx * 2;
        int hi_idx = lo_idx + 1;

        float lo_val = (lo_idx < N) ? smem_act[lo_idx] : 0.0f;
        float hi_val = (hi_idx < N) ? smem_act[hi_idx] : 0.0f;

        float lo_sc = x_out_scale[lo_idx / B];
        float hi_sc = x_out_scale[hi_idx / B];

        float lo_q = fminf(7.0f, fmaxf(-7.0f, roundf(lo_val / lo_sc)));
        float hi_q = fminf(7.0f, fmaxf(-7.0f, roundf(hi_val / hi_sc)));

        uint8_t lo_nib = static_cast<uint8_t>(lo_q + 8);
        uint8_t hi_nib = static_cast<uint8_t>(hi_q + 8);

        x_out[byte_idx] = lo_nib | (hi_nib << 4);
    }
}

} // anonymous namespace

// ===========================================================================
// Public API
// ===========================================================================

// fused_rmsnorm_quant_int4 — RMSNorm + INT4 pack (single kernel)
// Input:  proj [N] FP32, weight [N] FP32 RMSNorm weight
// Output: x_out [N/2] packed INT4, x_out_scale [N/16] FP32 scales
// N must be multiple of 32 (for 256 threads × 16 elements).
cudaError_t fused_rmsnorm_quant_int4(
    uint8_t* x_out,
    float* x_out_scale,
    const float* proj,
    const float* weight,
    int N,
    float eps,
    cudaStream_t stream)
{
    if (N % 32 != 0) return cudaErrorInvalidValue;
    if (N > THREADS * EPT) {
        // For large N (e.g., H=4096, I=12288), need multiple blocks
        // Use grid-stride loop: each block handles THREADS*EPT elements
        int smem_bytes = THREADS * EPT * sizeof(float);  // for smem_norm
        int grid = (N + THREADS * EPT - 1) / (THREADS * EPT);
        fused_rmsnorm_quant_int4_v2_kernel<<<grid, THREADS, smem_bytes, stream>>>(
            x_out, x_out_scale, proj, weight, N, eps);
    } else {
        // Single block for small N
        int smem_bytes = N * sizeof(float);
        fused_rmsnorm_quant_int4_v2_kernel<<<dim3(1), dim3(THREADS), smem_bytes, stream>>>(
            x_out, x_out_scale, proj, weight, N, eps);
    }
    return cudaPeekAtLastError();
}

// fused_swiglu_quant_int4 — SwiGLU + INT4 quant (single kernel)
// Input:  gate [N] FP32, up [N] FP32
// Output: x_out [N/2] packed INT4, x_out_scale [N/16] FP32 scales
cudaError_t fused_swiglu_quant_int4(
    uint8_t* x_out,
    float* x_out_scale,
    const float* gate,
    const float* up,
    int N,
    cudaStream_t stream)
{
    if (N % 32 != 0) return cudaErrorInvalidValue;
    int smem_bytes = THREADS * EPT * sizeof(float);

    if (N <= THREADS * EPT) {
        fused_swiglu_quant_int4_kernel<<<dim3(1), dim3(THREADS), smem_bytes, stream>>>(
            x_out, x_out_scale, gate, up, N);
    } else {
        // Multi-block for large N (e.g., I=12288)
        int grid = (N + THREADS * EPT - 1) / (THREADS * EPT);
        fused_swiglu_quant_int4_kernel<<<grid, dim3(THREADS), smem_bytes, stream>>>(
            x_out, x_out_scale, gate, up, N);
    }
    return cudaPeekAtLastError();
}

} // namespace kernels
} // namespace blackwell