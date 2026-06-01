// src/kernels/fused_swiglu_quant.cu — Fused SwiGLU + INT8 quant
//
// Combines: apply_swiglu(gate, up) → pack_int8(output)
// Saves 1 kernel launch per call (2 per MLP: swiglu + pack_mlp).
//
// Flow:
//   gate_fp32 = x @ W_gate  (pre-computed)
//   up_fp32 = x @ W_up      (pre-computed)
//   mlp_fp32 = silu(gate) * up  (elementwise)
//   mlp_i8, mlp_scale = quantize(mlp_fp32)
//
// Build: CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake --build build --parallel

#include <cuda_runtime.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

constexpr int THREADS = 256;
constexpr int B = 16;         // INT8 block size
constexpr int EPT = 8;         // elements per thread

__launch_bounds__(THREADS, 1)
__global__ void fused_swiglu_quant_kernel(
    int8_t* __restrict__ out_i8,
    float* __restrict__ out_scale,
    const float* __restrict__ gate,     // [N] FP32
    const float* __restrict__ up,        // [N] FP32
    int N)
{
    int blk = blockIdx.x;
    int tid = threadIdx.x;
    
    if (blk * (THREADS * EPT) >= N) return;  // bounds check
    
    // SwiGLU + per-block scale computation
    // Each thread computes EPT elements + local max
    float local_max = 0.0f;
    float vals[EPT];
    
    #pragma unroll
    for (int e = 0; e < EPT; ++e) {
        int idx = blk * THREADS * EPT + tid + e * THREADS;
        if (idx < N) {
            float g = gate[idx];
            float u = up[idx];
            // SiLU: silu(x) = x * sigmoid(x)
            float s = 1.0f / (1.0f + expf(-g));
            float v = g * s * u;
            vals[e] = v;
            local_max = fmaxf(local_max, fabsf(v));
        } else {
            vals[e] = 0.0f;
        }
    }
    
    // Warp reduce max
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, off));
    
    int block_elem_base = blk * THREADS * EPT;
    int num_blks_total = (N + B - 1) / B;
    int my_blk = (block_elem_base + tid) / B;  // which block this thread contributes to
    
    // Compute block scale (thread 0 of block writes scale)
    int tid_in_blk = (block_elem_base + tid) % B;
    if (tid_in_blk == 0 && my_blk < num_blks_total) {
        out_scale[my_blk] = fmaxf(local_max / 127.0f, 1e-9f);
    }
    __syncthreads();
    
    // Now quantize
    float sc = out_scale[my_blk];
    
    #pragma unroll
    for (int e = 0; e < EPT; ++e) {
        int idx = blk * THREADS * EPT + tid + e * THREADS;
        if (idx < N) {
            float q = vals[e] / sc;
            q = fminf(127.0f, fmaxf(-127.0f, roundf(q)));
            out_i8[idx] = static_cast<int8_t>(q);
        }
    }
}

}  // anonymous namespace

cudaError_t fused_swiglu_quant(
    int8_t* out_i8,
    float* out_scale,
    const float* gate,
    const float* up,
    int N,
    cudaStream_t stream)
{
    if (N % 16 != 0) return cudaErrorInvalidValue;
    
    int grid = (N + THREADS * EPT - 1) / (THREADS * EPT);
    fused_swiglu_quant_kernel<<<grid, THREADS, 0, stream>>>(
        out_i8, out_scale, gate, up, N);
    
    return cudaPeekAtLastError();
}

}  // namespace kernels
}  // namespace blackwell