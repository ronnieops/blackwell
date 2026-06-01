// src/kernels/gemm_int8_wmma_fast.cu — Optimized INT8 GEMM with WMMA
//
// Key optimizations over gemm_int8_wmma.cu:
// 1. 32×32 tiles (4 WMMA blocks per CTA)
// 2. 4 warps per CTA for better occupancy
// 3. Correct shared memory layout for WMMA store

#include <cuda_runtime.h>
#include <mma.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {

using namespace nvcuda;

// 32×32 output tile per CTA, 4 warps (128 threads)
__global__ __launch_bounds__(128, 1)
void gemm_int8_wmma_fast_kernel(
    float* __restrict__ C,
    const int8_t* __restrict__ A,
    const float* __restrict__ A_sc,
    const int8_t* __restrict__ B,
    const float* __restrict__ B_sc,
    int M, int N, int K)
{
    int cm = blockIdx.y * 32;
    int cn = blockIdx.x * 32;
    int num_K_blks = K / 16;
    
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Determine which 16×16 tile this warp handles
    int tile_row = (warp_id / 2) * 16;  // 0 or 16
    int tile_col = (warp_id % 2) * 16;  // 0 or 16
    int wm = cm + tile_row;
    int wn = cn + tile_col;
    
    // Shared memory: scales only (no INT32 SMEM buffer needed)
    __shared__ float smem_a_sc[32];
    __shared__ float smem_b_sc[32];
    
    // FP32 accumulators (8 per thread)
    float acc[8] = {0.0f};
    
    // Loop over K blocks
    for (int kb = 0; kb < num_K_blks; kb++) {
        // Load scales cooperatively
        if (tid < 32) {
            int m = cm + tid;
            int n = cn + tid;
            smem_a_sc[tid] = (m < M) ? A_sc[m * num_K_blks + kb] : 0.0f;
            smem_b_sc[tid] = (n < N) ? B_sc[n * num_K_blks + kb] : 0.0f;
        }
        __syncthreads();
        
        if (wm < M && wn < N) {
            int k = kb * 16;
            
            // Load A fragment (row_major)
            wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a_frag;
            wmma::load_matrix_sync(a_frag, A + wm * K + k, K);
            
            // Load B fragment (col_major)
            wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> b_frag;
            wmma::load_matrix_sync(b_frag, B + wn * K + k, K);
            
            // MMA accumulate
            wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
            wmma::fill_fragment(c_frag, 0);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            
            // Direct dequant from c_frag.x[i] — NO SMEM round-trip
            // WMMA m16n16k16 int accumulator layout:
            // thread owns 8 values at:
            //   row = (lane_id/4)*2 + (i/4), col = (lane_id%4)*4 + (i%4)
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int row_in_tile = (lane_id / 4) * 2 + (i / 4);
                int col_in_tile = (lane_id % 4) * 4 + (i % 4);
                int m_out = wm + row_in_tile;
                int n_out = wn + col_in_tile;
                float a_scale = smem_a_sc[tile_row + row_in_tile];
                float b_scale = smem_b_sc[tile_col + col_in_tile];
                acc[i] += (float)c_frag.x[i] * a_scale * b_scale;
            }
        }
        // No __syncthreads — no SMEM buffer between warps
    }
    
    // Write final result
    if (wm < M && wn < N) {
        for (int i = 0; i < 8; i++) {
            int row_in_tile = (lane_id / 4) * 2 + (i / 4);
            int col_in_tile = (lane_id % 4) * 4 + (i % 4);
            int m = wm + row_in_tile;
            int n = wn + col_in_tile;
            if (m < M && n < N) {
                C[m * N + n] = acc[i];
            }
        }
    }
}

// Launch wrapper
cudaError_t gemm_int8_wmma_fast(
    float* C,
    const void* A_i8,
    const float* A_sc,
    const void* B_i8,
    const float* B_sc,
    int M, int N, int K,
    cudaStream_t stream)
{
    if (M < 16 || N < 16 || K < 16)
        return cudaErrorInvalidValue;
    
    // 32×32 tiles, 128 threads (4 warps)
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    dim3 block(128);
    gemm_int8_wmma_fast_kernel<<<grid, block, 0, stream>>>(
        C,
        static_cast<const int8_t*>(A_i8),
        A_sc,
        static_cast<const int8_t*>(B_i8),
        B_sc,
        M, N, K);
    
    return cudaGetLastError();
}

}  // namespace kernels
}  // namespace blackwell
