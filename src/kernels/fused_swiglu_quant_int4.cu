// src/kernels/fused_swiglu_quant_int4.cu — Fused SwiGLU + INT4 quant
//
// Combines: apply_swiglu(gate, up) → quantize_int4(mlp)
// Saves 1 kernel launch per MLP (swiglu + quantize = 2 → 1).
//
// Grid: N/16 blocks, 16 threads per block
// Thread t handles byte t (2 elements: 2t, 2t+1)
//
// Build:
//   CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake --build build --parallel

#include <cuda_runtime.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

__launch_bounds__(16, 16)
__global__ void fused_swiglu_quant_int4_kernel(
    uint8_t* __restrict__ out_packed,
    float* __restrict__ out_scale,
    const float* __restrict__ gate,     // [N] FP32
    const float* __restrict__ up,        // [N] FP32
    int N)
{
    constexpr int B = 16;  // block size

    int blk = blockIdx.x;
    int tid = threadIdx.x;  // 0..15

    // Each block handles B=16 elements
    int elem_off = blk * B;

    // Load gate and up values
    float g0 = gate[elem_off + tid];
    float u0 = up[elem_off + tid];

    // SiLU: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    float s = 1.0f / (1.0f + expf(-g0));
    float v = g0 * s * u0;

    // Compute absmax for this block (across all 16 threads)
    float blk_max = fabsf(v);
    blk_max = fmaxf(blk_max, __shfl_down_sync(0xffffffff, blk_max, 8));
    blk_max = fmaxf(blk_max, __shfl_down_sync(0xffffffff, blk_max, 4));
    blk_max = fmaxf(blk_max, __shfl_down_sync(0xffffffff, blk_max, 2));
    blk_max = fmaxf(blk_max, __shfl_down_sync(0xffffffff, blk_max, 1));

    // Thread 0 writes scale
    if (tid == 0) {
        out_scale[blk] = (blk_max > 1e-10f) ? (blk_max / 7.0f) : (1.0f / 7.0f);
    }
    __syncthreads();

    float sc = out_scale[blk];

    // Quantize to [-7..7], pack 2 values per byte
    int q = (int)roundf(v / sc);
    q = max(-8, min(7, q));

    // Thread t handles byte t: lower nibble = element 2t, upper = 2t+1
    uint8_t nib0 = (uint8_t)((q + 8) & 0x0F);
    // For the second element (2t+1), we need value from another thread
    // Actually, each thread handles ONE byte = 2 elements. But we have 16 threads
    // and 8 bytes per block. So threads 0-7 handle bytes 0-7.
    // Thread t: if t < 8, pack element t (as lower nibble) + element t+8 (as upper)
    // This gives us 8 bytes per block with 16 threads (some idle).
    //
    // Alternative: use 8 threads per block (1 warp), each packs 1 byte
    // Threads 0-7: handle bytes 0-7. Threads 8-15: idle.
    //
    // Let's use 8 threads.

    if (tid < 8) {
        // Lower nibble: element 2*tid
        // Upper nibble: element 2*tid + 1
        // But element 2*tid+1 is computed by another thread! We need the value.
        //
        // Actually, we can use shfl to get the other half.
        // Thread tid has element[2*tid]. We need element[2*tid+1].
        // Element[2*tid+1] is held by thread tid if 2*tid+1 < 16... no, each thread has 1 element.
        //
        // Correction: with 16 threads and B=16 elements, each thread has 1 element.
        // To pack 2 elements per byte, we need pairs: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15)
        // Thread 0 pairs with thread 1, thread 2 with 3, etc.
        //
        // Use shfl to pair: thread tid gets element from thread tid^1
        float v_pair = __shfl_xor_sync(0xffffffff, v, 1);

        // Now v has our element, v_pair has the paired element
        // For tid = 0: v = elem[0], v_pair = elem[1]
        // For tid = 1: v = elem[1], v_pair = elem[0]
        // etc.

        if (tid % 2 == 0) {
            // Even thread: v is lower nibble, v_pair is upper
            int q0 = (int)roundf(v / sc);
            q0 = max(-8, min(7, q0));
            int q1 = (int)roundf(v_pair / sc);
            q1 = max(-8, min(7, q1));
            uint8_t packed = ((q0 + 8) & 0x0F) | (((q1 + 8) & 0x0F) << 4);
            out_packed[blk * 8 + tid / 2] = packed;
        }
    }
}

}  // anonymous namespace

cudaError_t fused_swiglu_quant_int4(
    uint8_t* out_packed,
    float* out_scale,
    const float* gate,
    const float* up,
    int N,
    cudaStream_t stream)
{
    if (N % 16 != 0) return cudaErrorInvalidValue;

    int num_blocks = N / 16;
    fused_swiglu_quant_int4_kernel<<<num_blocks, 16, 0, stream>>>(
        out_packed, out_scale, gate, up, N);

    return cudaPeekAtLastError();
}

}  // namespace kernels
}  // namespace blackwell