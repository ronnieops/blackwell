// src/kernels/gemv_int4_qkv.cu — Fused Q/K/V GEMV with inline INT4 quantization
//
// Combines: quantize(x_fp32) + 3× gemv_int4_warp(Q/K/V) → 1 kernel launch.
//
// Flow (per block = 1 output row):
//   1. Load FP32 activation x_fp32[K] (once, shared across Q/K/V)
//   2. Compute per-block scales and quantize x → x_i4 in registers
//   3. For each of Q, K, V: dot(x_i4, W_i4) → output (use same x_i4, different weights)
//   4. Unload Q, K, V outputs
//
// Saves: 1× INT4 write + 3× INT4 read + sync (vs separate quantize + 3× GEMV).
// Load reduction: 3× K bytes → 1× K bytes (activation loaded once).
//
// Build:
//   CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake --build build --parallel

#include <cuda_runtime.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

constexpr int B = 16;      // quantization block size
constexpr int PB = 8;     // packed bytes per block (B/2)

// Fused Q/K/V with inline INT4 quantization
// Grid: (N_q) blocks for Q, (N_kv) blocks for K/V... but we need Q/K/V together.
// Use max(N_q, N_kv) blocks. Each block computes Q if n < N_q, K/V if n < N_kv.
// x_fp32: [K] FP32 input (not packed!)
// x_scale_out: [K/16] FP32 scales (computed in kernel)
__launch_bounds__(32, 8)
__global__ void fused_qkv_int4_kernel(
    float* __restrict__ Q_out,      // [N_q] output
    float* __restrict__ K_out,      // [N_kv] output
    float* __restrict__ V_out,      // [N_kv] output
    const float* __restrict__ x_fp32,  // [K] FP32 input
    float* __restrict__ x_scale_out,  // [K/16] scales (written for Q)
    // Q weights
    const uint8_t* __restrict__ W_q_packed,
    const float* __restrict__ W_q_scale,
    // K weights
    const uint8_t* __restrict__ W_k_packed,
    const float* __restrict__ W_k_scale,
    // V weights
    const uint8_t* __restrict__ W_v_packed,
    const float* __restrict__ W_v_scale,
    int K, int N_q, int N_kv)
{
    int n_out = blockIdx.x;  // output row index
    int tid = threadIdx.x;   // 0..31

    int num_K_blks = K / B;

    // Load activation + quantize to INT4 (shared across Q/K/V)
    // Each thread loads its K-blocks at stride-32
    // After loading, we have x_fp32 values for this thread's K-blocks.
    // We need to compute per-block scales and store quantized values.
    //
    // But: different threads own different K-blocks. For scale computation,
    // we need the max abs across ALL threads within each block.
    //
    // Approach: 
    //   Phase 1: Load FP32 into registers, compute local absmax per K-block
    //   Phase 2: Warp-reduce absmax to get block scale
    //   Phase 3: With scale known, quantize and store x_i4 (or keep in registers)
    //   Phase 4: Do GEMV for Q, K, V using quantized values
    //
    // Storage: keep x_i4 in registers for all 32 lanes × 128 bytes = 4KB.
    // That's tight but doable (32 threads × 16 elements × 1 byte = 512 bytes per thread).

    // We have num_K_blks K-blocks total. Each thread handles (num_K_blks/32) blocks.
    // But for scale + quantize, we need the max per K-block.
    // Let's allocate registers for x_vals[128] (max 2048/16 = 128 blocks, worst case).

    // Actually: with stride-32, each thread handles blocks tid, tid+32, tid+64...
    // So thread 0 handles blocks 0,32,64,... thread 1 handles 1,33,67,...
    // For each block, we need to compute absmax over B=16 elements.
    //
    // Let each thread store its block's elements in registers, then reduce.

    // Register storage for FP32 activation values (8 per thread per iteration)
    // We process blocks tid, tid+32, ... so each thread has num_K_blks/32 blocks.
    // For each block: B=16 elements → need 16 float registers per thread.
    // Process all blocks sequentially, or buffer them.

    // Simpler: load one block at a time, quantize, then do all 3 GEMV projections.
    // For block kb (owned by tid):
    //   - Load x_fp32[kb*B + 0..15] into registers
    //   - Compute block absmax
    //   - Sync to get final scale
    //   - Quantize to int8_t
    //   - Do Q, K, V GEMV for this block
    //   - Repeat for next block

    // We'll use smem for intermediate x_i4 (8 bytes per block = 512 bytes total)
    extern __shared__ float smem[];
    // x_i4_shared: [num_K_blks][PB] bytes = num_K_blks * 8 bytes
    // x_scale_shared: [num_K_blks] floats
    uint8_t* x_i4_shared = reinterpret_cast<uint8_t*>(smem);
    float* x_scale_shared = smem + num_K_blks * PB;

    // Phase 1: Load FP32, compute scales, quantize (one block at a time)
    for (int kb_base = 0; kb_base < num_K_blks; kb_base += 32) {
        int kb = kb_base + tid;  // which block this thread handles
        if (kb >= num_K_blks) continue;

        // Load 16 FP32 values into registers
        int off = kb * B;
        float4 x0 = *reinterpret_cast<const float4*>(&x_fp32[off]);
        float4 x1 = *reinterpret_cast<const float4*>(&x_fp32[off + 8]);

        float vals[16];
        vals[0]=x0.x; vals[1]=x0.y; vals[2]=x0.z; vals[3]=x0.w;
        vals[4]=x1.x; vals[5]=x1.y; vals[6]=x1.z; vals[7]=x1.w;
        vals[8]=x0.x; vals[9]=x0.y; vals[10]=x0.z; vals[11]=x0.w;  // duplicated but we'll fix
        // Actually let's just use float4 directly

        // Compute absmax for this block
        float blk_max = fmaxf(fabsf(x0.x), fabsf(x0.y));
        blk_max = fmaxf(blk_max, fabsf(x0.z));
        blk_max = fmaxf(blk_max, fabsf(x0.w));
        blk_max = fmaxf(blk_max, fabsf(x1.x));
        blk_max = fmaxf(blk_max, fabsf(x1.y));
        blk_max = fmaxf(blk_max, fabsf(x1.z));
        blk_max = fmaxf(blk_max, fabsf(x1.w));

        // Warp reduce: get max across all 32 threads for this block
        blk_max = fmaxf(blk_max, __shfl_down_sync(0xffffffff, blk_max, 16));
        blk_max = fmaxf(blk_max, __shfl_down_sync(0xffffffff, blk_max, 8));
        blk_max = fmaxf(blk_max, __shfl_down_sync(0xffffffff, blk_max, 4));
        blk_max = fmaxf(blk_max, __shfl_down_sync(0xffffffff, blk_max, 2));
        blk_max = fmaxf(blk_max, __shfl_down_sync(0xffffffff, blk_max, 1));

        // Now we know the block scale (lane 0 has it)
        if (tid == 0) {
            x_scale_shared[kb] = (blk_max > 1e-10f) ? (blk_max / 7.0f) : (1.0f / 7.0f);
        }
        __syncthreads();

        float sc = x_scale_shared[kb];

        // Quantize: pack 16 float values into 8 bytes (2 values/byte)
        // Lower nibble = even index, upper = odd
        int q0 = (int)roundf(x0.x / sc); q0 = max(-8, min(7, q0));
        int q1 = (int)roundf(x0.y / sc); q1 = max(-8, min(7, q1));
        int q2 = (int)roundf(x0.z / sc); q2 = max(-8, min(7, q2));
        int q3 = (int)roundf(x0.w / sc); q3 = max(-8, min(7, q3));
        int q4 = (int)roundf(x1.x / sc); q4 = max(-8, min(7, q4));
        int q5 = (int)roundf(x1.y / sc); q5 = max(-8, min(7, q5));
        int q6 = (int)roundf(x1.z / sc); q6 = max(-8, min(7, q6));
        int q7 = (int)roundf(x1.w / sc); q7 = max(-8, min(7, q7));

        uint8_t byte0 = ((q0 + 8) & 0x0F) | (((q1 + 8) & 0x0F) << 4);
        uint8_t byte1 = ((q2 + 8) & 0x0F) | (((q3 + 8) & 0x0F) << 4);
        uint8_t byte2 = ((q4 + 8) & 0x0F) | (((q5 + 8) & 0x0F) << 4);
        uint8_t byte3 = ((q6 + 8) & 0x0F) | (((q7 + 8) & 0x0F) << 4);

        // Store to smem (8 bytes per block)
        int byte_off = kb * PB;
        x_i4_shared[byte_off + 0] = byte0;
        x_i4_shared[byte_off + 1] = byte1;
        x_i4_shared[byte_off + 2] = byte2;
        x_i4_shared[byte_off + 3] = byte3;

        __syncthreads();

        // Now do GEMV for Q, K, V using x_i4 from smem
        // For each projection, load weight bytes + x_i4, unpack, multiply, accumulate

        // Q projection
        if (n_out < N_q) {
            float acc_q = 0.0f;
            for (int jb = tid; jb < num_K_blks; jb += 32) {
                // Load weight bytes for this block (16 nibbles = 8 bytes)
                const uint8_t* w_ptr = &W_q_packed[(size_t)n_out * (K / 2) + jb * PB];
                uint2 w_p = *reinterpret_cast<const uint2*>(w_ptr);

                // Load x_i4 bytes
                int x_off = jb * PB;
                uint2 x_p = *reinterpret_cast<const uint2*>(&x_i4_shared[x_off]);

                // Scales
                float w_sc = W_q_scale[(size_t)n_out * num_K_blks + jb];
                float x_sc = x_scale_shared[jb];
                float prod_sc = w_sc * x_sc;

                // Unpack + dot product (16 elements per block)
                const uint8_t* wb = reinterpret_cast<const uint8_t*>(&w_p);
                const uint8_t* xb = reinterpret_cast<const uint8_t*>(&x_p);

                float sum_f = 0.0f;
                #pragma unroll
                for (int jj = 0; jj < PB; ++jj) {
                    float w0, w1, x0, x1;
                    int lo_w = wb[jj] & 0x0F; if (lo_w > 7) lo_w -= 16;
                    int hi_w = (wb[jj] >> 4) & 0x0F; if (hi_w > 7) hi_w -= 16;
                    int lo_x = xb[jj] & 0x0F; if (lo_x > 7) lo_x -= 16;
                    int hi_x = (xb[jj] >> 4) & 0x0F; if (hi_x > 7) hi_x -= 16;
                    sum_f += (float)lo_w * (float)lo_x + (float)hi_w * (float)hi_x;
                }
                acc_q += sum_f * prod_sc;
            }

            // Warp reduce
            acc_q += __shfl_xor_sync(0xffffffff, acc_q, 16);
            acc_q += __shfl_xor_sync(0xffffffff, acc_q, 8);
            acc_q += __shfl_xor_sync(0xffffffff, acc_q, 4);
            acc_q += __shfl_xor_sync(0xffffffff, acc_q, 2);
            acc_q += __shfl_xor_sync(0xffffffff, acc_q, 1);

            if (tid == 0) Q_out[n_out] = acc_q;
        }

        // K projection
        if (n_out < N_kv) {
            float acc_k = 0.0f;
            for (int jb = tid; jb < num_K_blks; jb += 32) {
                const uint8_t* w_ptr = &W_k_packed[(size_t)n_out * (K / 2) + jb * PB];
                uint2 w_p = *reinterpret_cast<const uint2*>(w_ptr);
                int x_off = jb * PB;
                uint2 x_p = *reinterpret_cast<const uint2*>(&x_i4_shared[x_off]);

                float w_sc = W_k_scale[(size_t)n_out * num_K_blks + jb];
                float x_sc = x_scale_shared[jb];
                float prod_sc = w_sc * x_sc;

                const uint8_t* wb = reinterpret_cast<const uint8_t*>(&w_p);
                const uint8_t* xb = reinterpret_cast<const uint8_t*>(&x_p);

                float sum_f = 0.0f;
                #pragma unroll
                for (int jj = 0; jj < PB; ++jj) {
                    float w0, w1, x0, x1;
                    int lo_w = wb[jj] & 0x0F; if (lo_w > 7) lo_w -= 16;
                    int hi_w = (wb[jj] >> 4) & 0x0F; if (hi_w > 7) hi_w -= 16;
                    int lo_x = xb[jj] & 0x0F; if (lo_x > 7) lo_x -= 16;
                    int hi_x = (xb[jj] >> 4) & 0x0F; if (hi_x > 7) hi_x -= 16;
                    sum_f += (float)lo_w * (float)lo_x + (float)hi_w * (float)hi_x;
                }
                acc_k += sum_f * prod_sc;
            }

            acc_k += __shfl_xor_sync(0xffffffff, acc_k, 16);
            acc_k += __shfl_xor_sync(0xffffffff, acc_k, 8);
            acc_k += __shfl_xor_sync(0xffffffff, acc_k, 4);
            acc_k += __shfl_xor_sync(0xffffffff, acc_k, 2);
            acc_k += __shfl_xor_sync(0xffffffff, acc_k, 1);

            if (tid == 0) K_out[n_out] = acc_k;
        }

        // V projection
        if (n_out < N_kv) {
            float acc_v = 0.0f;
            for (int jb = tid; jb < num_K_blks; jb += 32) {
                const uint8_t* w_ptr = &W_v_packed[(size_t)n_out * (K / 2) + jb * PB];
                uint2 w_p = *reinterpret_cast<const uint2*>(w_ptr);
                int x_off = jb * PB;
                uint2 x_p = *reinterpret_cast<const uint2*>(&x_i4_shared[x_off]);

                float w_sc = W_v_scale[(size_t)n_out * num_K_blks + jb];
                float x_sc = x_scale_shared[jb];
                float prod_sc = w_sc * x_sc;

                const uint8_t* wb = reinterpret_cast<const uint8_t*>(&w_p);
                const uint8_t* xb = reinterpret_cast<const uint8_t*>(&x_p);

                float sum_f = 0.0f;
                #pragma unroll
                for (int jj = 0; jj < PB; ++jj) {
                    float w0, w1, x0, x1;
                    int lo_w = wb[jj] & 0x0F; if (lo_w > 7) lo_w -= 16;
                    int hi_w = (wb[jj] >> 4) & 0x0F; if (hi_w > 7) hi_w -= 16;
                    int lo_x = xb[jj] & 0x0F; if (lo_x > 7) lo_x -= 16;
                    int hi_x = (xb[jj] >> 4) & 0x0F; if (hi_x > 7) hi_x -= 16;
                    sum_f += (float)lo_w * (float)lo_x + (float)hi_w * (float)hi_x;
                }
                acc_v += sum_f * prod_sc;
            }

            acc_v += __shfl_xor_sync(0xffffffff, acc_v, 16);
            acc_v += __shfl_xor_sync(0xffffffff, acc_v, 8);
            acc_v += __shfl_xor_sync(0xffffffff, acc_v, 4);
            acc_v += __shfl_xor_sync(0xffffffff, acc_v, 2);
            acc_v += __shfl_xor_sync(0xffffffff, acc_v, 1);

            if (tid == 0) V_out[n_out] = acc_v;
        }

        __syncthreads();
    }
}

}  // anonymous namespace

cudaError_t fused_qkv_int4(
    float* Q_out,
    float* K_out,
    float* V_out,
    const float* x_fp32,
    float* x_scale_out,
    const uint8_t* W_q_packed, const float* W_q_scale,
    const uint8_t* W_k_packed, const float* W_k_scale,
    const uint8_t* W_v_packed, const float* W_v_scale,
    int K, int N_q, int N_kv,
    cudaStream_t stream)
{
    if (K % 16 != 0 || N_q % 16 != 0 || N_kv % 16 != 0)
        return cudaErrorInvalidValue;

    int num_K_blks = K / B;
    size_t smem_size = num_K_blks * PB * sizeof(uint8_t) + num_K_blks * sizeof(float);

    dim3 grid(max(N_q, N_kv));
    fused_qkv_int4_kernel<<<grid, dim3(32), smem_size, stream>>>(
        Q_out, K_out, V_out,
        x_fp32, x_scale_out,
        W_q_packed, W_q_scale,
        W_k_packed, W_k_scale,
        W_v_packed, W_v_scale,
        K, N_q, N_kv);

    return cudaPeekAtLastError();
}

}  // namespace kernels
}  // namespace blackwell