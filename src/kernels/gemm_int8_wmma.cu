// src/kernels/gemm_int8_wmma.cu — INT8 GEMM with WMMA tensor cores
//
// C[M×N] = sum_k( A[M×K] × B[N×K]^T × A_sc[k] × B_sc[k] )
// Uses wmma::mma_sync with m16n16k16 INT8 fragments.
// Per-block scales applied during accumulation via FP32.
//
// Design choices:
//   - Separate compilation unit to avoid WMMA fragment ABI issues
//   - __launch_bounds__(32, 2) to control register pressure
//   - Shared memory for dequant to avoid global memory round-trip

#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {

// Use nvcuda::wmma in this TU only (avoids conflicts in other TUs)
using namespace nvcuda;

// Launch bounds: 32 threads/warp, 2 blocks per SM (64 threads total)
// This limits registers to ~128 per thread (256 regs per SM / 2 blocks / 32 threads)
__global__ __launch_bounds__(32, 2)
void gemm_int8_wmma_kernel(
    float* __restrict__ C,
    const int8_t* __restrict__ A,
    const float* __restrict__ A_sc,
    const int8_t* __restrict__ B,
    const float* __restrict__ B_sc,
    int M, int N, int K)
{
    int wm = blockIdx.y * 16;
    int wn = blockIdx.x * 16;
    int num_K_blks = K / 16;
    
    // Shared memory for scales and intermediate int32 tile
    __shared__ float smem_a_sc[16];
    __shared__ float smem_b_sc[16];
    __shared__ int smem_int[16][16];
    
    int tid = threadIdx.x;
    // Per-thread float accumulator (8 elements per thread)
    float acc[8] = {0.0f};
    
    // Loop over K blocks
    for (int kb = 0; kb < num_K_blks; kb++) {
        // Load scales for this block
        if (tid < 16) {
            int m = wm + tid;
            int n = wn + tid;
            smem_a_sc[tid] = (m < M) ? A_sc[m * num_K_blks + kb] : 0.0f;
            smem_b_sc[tid] = (n < N) ? B_sc[n * num_K_blks + kb] : 0.0f;
        }
        __syncthreads();
        
        // WMMA fragments
        wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
        
        wmma::fill_fragment(c_frag, 0);
        
        if (wm < M && wn < N) {
            int k = kb * 16;
            wmma::load_matrix_sync(a_frag, A + wm * K + k, K);
            wmma::load_matrix_sync(b_frag, B + wn * K + k, K);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            
            // Store int32 tile to shared memory
            wmma::store_matrix_sync(&smem_int[0][0], c_frag, 16, wmma::mem_row_major);
        }
        __syncthreads();
        
        // Convert int32 tile to float and scale
        // WMMA m16n16k16 accumulator layout: thread tid owns 8 elements at:
        //   row = (tid/4)*2 + (i/4), col = (tid%4)*4 + (i%4)
        for (int i = 0; i < 8; i++) {
            int row = (tid / 4) * 2 + (i / 4);
            int col = (tid % 4) * 4 + (i % 4);
            int m = wm + row;
            int n = wn + col;
            if (m < M && n < N) {
                float a_sc = smem_a_sc[row];
                float b_sc = smem_b_sc[col];
                acc[i] += (float)smem_int[row][col] * a_sc * b_sc;
            }
        }
        __syncthreads();
    }
    
    // Write final result
    if (wm < M && wn < N) {
        for (int i = 0; i < 8; i++) {
            int row = (tid / 4) * 2 + (i / 4);
            int col = (tid % 4) * 4 + (i % 4);
            int m = wm + row;
            int n = wn + col;
            if (m < M && n < N) {
                C[m * N + n] = acc[i];
            }
        }
    }
}

// Version 2: Use global temp buffer for INT32, then dequant separately
__global__ __launch_bounds__(32, 2)
void gemm_int8_wmma_raw_kernel(
    int* __restrict__ C_int,
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int M, int N, int K)
{
    int wm = blockIdx.y * 16;
    int wn = blockIdx.x * 16;
    int num_K_blks = K / 16;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
    
    wmma::fill_fragment(c_frag, 0);
    
    for (int kb = 0; kb < num_K_blks; kb++) {
        int k = kb * 16;
        if (wm < M && wn < N) {
            wmma::load_matrix_sync(a_frag, A + wm * K + k, K);
            wmma::load_matrix_sync(b_frag, B + wn * K + k, K);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }
    
    if (wm < M && wn < N) {
        wmma::store_matrix_sync(C_int + wm * N + wn, c_frag, N, wmma::mem_row_major);
    }
}

// Dequantize INT32 → FP32 with per-block scales
__global__ __launch_bounds__(256, 4)
void dequant_int32_to_float_kernel(
    float* __restrict__ C,
    const int* __restrict__ C_int,
    const float* __restrict__ A_sc,
    const float* __restrict__ B_sc,
    int M, int N, int num_K_blks)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int m = idx / N;
        int n = idx % N;
        float a_sc = A_sc[m * num_K_blks];
        float b_sc = B_sc[n * num_K_blks];
        C[idx] = (float)C_int[idx] * a_sc * b_sc;
    }
}

// Public API: INT8 GEMM with WMMA
// Caller must provide temp buffer C_int of size M*N*4 bytes (will be zeroed)
cudaError_t gemm_int8_wmma(
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
    
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    dim3 block(32);
    gemm_int8_wmma_kernel<<<grid, block, 0, stream>>>(
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
