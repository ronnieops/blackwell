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

// Simpler approach: single-block with grid-stride over N chunks of THREADS*EPT.
// RMSNorm requires a GLOBAL sum_sq over all N elements. Multi-block would need a
// cross-block reduction (2-pass + global memory or cooperative groups), which is
// overkill for these tiny (~1-5µs) ops. One block looping over N in chunks keeps
// the reduction trivial and still finishes in microseconds.
//
// Works for any N (multiple of 32): H=4096 (1 chunk), I=12288 (3 chunks), etc.
// N must be multiple of THREADS*EPT/2 = 2048 for clean byte packing, but we handle
// the tail explicitly. We require N % 32 == 0 at the API boundary.
__launch_bounds__(THREADS, 1)
__global__ void fused_rmsnorm_quant_int4_v2_kernel(
    uint8_t* __restrict__ x_out,
    float* __restrict__ x_out_scale,
    const float* __restrict__ proj,
    const float* __restrict__ weight,
    int N, float eps)
{
    extern __shared__ float smem[];   // size = N floats (caller allocates N*4)
    int tid = threadIdx.x;
    constexpr int NE = EPT;
    constexpr int CHUNK = THREADS * NE;   // 4096 elements per grid-stride iteration

    // Phase 1: load all N values into smem, accumulate global sum_sq.
    // Grid-stride over chunks of CHUNK elements.
    float my_sum_sq = 0.0f;
    for (int base = 0; base < N; base += CHUNK) {
        #pragma unroll
        for (int e = 0; e < NE; ++e) {
            int idx = base + tid + e * THREADS;
            if (idx < N) {
                float v = proj[idx];
                smem[idx] = v;
                my_sum_sq += v * v;
            }
        }
    }

    // Block-wide reduce of my_sum_sq
    my_sum_sq = warp_reduce_sum_f(my_sum_sq);
    __shared__ float warp_sums[8];
    if ((tid & 31) == 0) warp_sums[tid >> 5] = my_sum_sq;
    __syncthreads();
    float block_sum = (tid < 8) ? warp_sums[tid] : 0.0f;
    block_sum = warp_reduce_sum_f(block_sum);

    __shared__ float s_rstd;
    if (tid == 0) s_rstd = rsqrtf(block_sum / static_cast<float>(N) + eps);
    __syncthreads();
    float rstd = s_rstd;

    // Phase 2: compute block scales (absmax/7 per 16 elements) over normalized values.
    // Write normalized values back into smem in place (we no longer need raw values).
    // Match production quantize_int4: scale = (absmax>1e-10) ? absmax/7 : 1/7,
    // quant range [-8,7] (4-bit signed, 15 levels).
    int num_blocks = (N + B - 1) / B;
    for (int blk_id = tid; blk_id < num_blocks; blk_id += THREADS) {
        int blk_start = blk_id * B;
        float absmax = 0.0f;
        #pragma unroll
        for (int i = 0; i < B; ++i) {
            int idx = blk_start + i;
            if (idx < N) {
                float normed = smem[idx] * weight[idx] * rstd;
                smem[idx] = normed;           // store normalized back
                absmax = fmaxf(absmax, fabsf(normed));
            }
        }
        x_out_scale[blk_id] = (absmax > 1e-10f) ? (absmax / 7.0f) : (1.0f / 7.0f);
    }
    __syncthreads();

    // Phase 3: quantize + pack (each thread handles 2 elements = 1 byte).
    // Range [-8,7], offset-binary (q+8), low nibble first.
    int num_bytes = N / 2;
    for (int byte_idx = tid; byte_idx < num_bytes; byte_idx += THREADS) {
        int lo_idx = byte_idx * 2;
        int hi_idx = lo_idx + 1;

        float lo_val = smem[lo_idx];
        float hi_val = (hi_idx < N) ? smem[hi_idx] : 0.0f;

        float lo_sc = x_out_scale[lo_idx / B];
        float hi_sc = x_out_scale[hi_idx / B];

        int lo_q = (int)roundf(lo_val / lo_sc);
        int hi_q = (int)roundf(hi_val / hi_sc);
        lo_q = max(-8, min(7, lo_q));
        hi_q = max(-8, min(7, hi_q));

        uint8_t lo_nib = (uint8_t)((lo_q + 8) & 0x0F);
        uint8_t hi_nib = (uint8_t)((hi_q + 8) & 0x0F);

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
// Single-block kernel: SwiGLU activation + INT4 quant (block-16, absmax/7).
// Grid-stride over N chunks of THREADS*EPT so any N works with one block.
//
// Input:  gate [N] FP32, up [N] FP32
// Output: x_out [N/2] packed INT4, x_out_scale [N/16] FP32 scales
//
// Replaces: apply_swiglu + quantize_int4  (2 kernels → 1)
__launch_bounds__(THREADS, 1)
__global__ void fused_swiglu_quant_int4_kernel(
    uint8_t* __restrict__ x_out,
    float* __restrict__ x_out_scale,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    int N)
{
    extern __shared__ float smem[];   // size = N floats
    int tid = threadIdx.x;
    constexpr int CHUNK = THREADS * EPT;

    // Phase 1: SwiGLU activation over all N, store to smem.
    // silu(g) * u, where silu(x) = x * sigmoid(x).
    for (int base = 0; base < N; base += CHUNK) {
        #pragma unroll
        for (int e = 0; e < EPT; ++e) {
            int idx = base + tid + e * THREADS;
            if (idx < N) {
                float g = gate[idx];
                float s = 1.0f / (1.0f + expf(-g));
                smem[idx] = g * s * up[idx];
            }
        }
    }
    __syncthreads();

    // Phase 2: compute block scales (absmax/7 per 16 elements).
    // Match production quantize_int4: scale clamp + range [-8,7].
    int num_blocks = (N + B - 1) / B;
    for (int blk_id = tid; blk_id < num_blocks; blk_id += THREADS) {
        int blk_start = blk_id * B;
        float absmax = 0.0f;
        #pragma unroll
        for (int i = 0; i < B; ++i) {
            int idx = blk_start + i;
            if (idx < N) absmax = fmaxf(absmax, fabsf(smem[idx]));
        }
        x_out_scale[blk_id] = (absmax > 1e-10f) ? (absmax / 7.0f) : (1.0f / 7.0f);
    }
    __syncthreads();

    // Phase 3: quantize + pack (each thread handles 2 elements = 1 byte).
    int num_bytes = N / 2;
    for (int byte_idx = tid; byte_idx < num_bytes; byte_idx += THREADS) {
        int lo_idx = byte_idx * 2;
        int hi_idx = lo_idx + 1;

        float lo_val = smem[lo_idx];
        float hi_val = (hi_idx < N) ? smem[hi_idx] : 0.0f;

        float lo_sc = x_out_scale[lo_idx / B];
        float hi_sc = x_out_scale[hi_idx / B];

        int lo_q = (int)roundf(lo_val / lo_sc);
        int hi_q = (int)roundf(hi_val / hi_sc);
        lo_q = max(-8, min(7, lo_q));
        hi_q = max(-8, min(7, hi_q));

        uint8_t lo_nib = (uint8_t)((lo_q + 8) & 0x0F);
        uint8_t hi_nib = (uint8_t)((hi_q + 8) & 0x0F);

        x_out[byte_idx] = lo_nib | (hi_nib << 4);
    }
}

} // anonymous namespace

// ===========================================================================
// Public API
// ===========================================================================

// fused_rmsnorm_quant_int4 — RMSNorm + INT4 pack (single kernel, single block).
// Input:  proj [N] FP32, weight [N] FP32 RMSNorm weight
// Output: x_out [N/2] packed INT4, x_out_scale [N/16] FP32 scales
// N must be multiple of 32. Single block with grid-stride handles any N (H=4096,
// I=12288, etc.). smem = N floats for the in-place normalization buffer.
// I=12288 → 48 KB smem; larger sizes need opt-in (one-time cudaFuncSetAttribute).
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
    if (N * sizeof(float) > 100 * 1024) return cudaErrorInvalidValue;  // 100 KB Blackwell smem
    int smem_bytes = N * sizeof(float);
    // Opt-in to >48KB dynamic smem (needed for I=12288 → 48KB, exactly at default limit).
    if (smem_bytes + 256 > 48 * 1024) {  // +256 for static smem slack
        cudaFuncSetAttribute(fused_rmsnorm_quant_int4_v2_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes + 256);
    }
    fused_rmsnorm_quant_int4_v2_kernel<<<dim3(1), dim3(THREADS), smem_bytes, stream>>>(
        x_out, x_out_scale, proj, weight, N, eps);
    return cudaPeekAtLastError();
}

// fused_swiglu_quant_int4 — SwiGLU + INT4 quant (single kernel, single block).
// Input:  gate [N] FP32, up [N] FP32
// Output: x_out [N/2] packed INT4, x_out_scale [N/16] FP32 scales
// N must be multiple of 32. Single block with grid-stride handles any N.
cudaError_t fused_swiglu_quant_int4(
    uint8_t* x_out,
    float* x_out_scale,
    const float* gate,
    const float* up,
    int N,
    cudaStream_t stream)
{
    if (N % 32 != 0) return cudaErrorInvalidValue;
    if (N * sizeof(float) > 100 * 1024) return cudaErrorInvalidValue;  // 100 KB Blackwell smem
    int smem_bytes = N * sizeof(float);
    if (smem_bytes + 256 > 48 * 1024) {  // +256 for static smem slack
        cudaFuncSetAttribute(fused_swiglu_quant_int4_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes + 256);
    }
    fused_swiglu_quant_int4_kernel<<<dim3(1), dim3(THREADS), smem_bytes, stream>>>(
        x_out, x_out_scale, gate, up, N);
    return cudaPeekAtLastError();
}

} // namespace kernels
} // namespace blackwell