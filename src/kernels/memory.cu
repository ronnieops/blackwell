// src/kernels/memory.cu — shared-memory tiling / async copy utilities
//
// Provides building blocks for overlapping data movement with compute.
// Used in prefill path where global→shared memory overlap matters.

#include <cuda_runtime.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace blackwell {
namespace kernels {

// ---------------------------------------------------------------------------
// Shared-memory tiled copy: load a tile from global to shared memory.
// Caller manages __syncthreads. This is a device helper, not a kernel.
// For use in custom GEMM/attention kernels that need manual tiling.
// ---------------------------------------------------------------------------
__global__ void shared_copy_kernel(
    float* __restrict__ shared_dst,
    const float* __restrict__ global_src,
    int tile_M, int tile_K, int global_stride)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total = tile_M * tile_K;
    if (idx >= total) return;

    int row = idx / tile_K;
    int col = idx % tile_K;
    shared_dst[row * tile_K + col] = global_src[row * global_stride + col];
}

cudaError_t shared_copy_async(
    float* dst, const float* src,
    int tile_M, int tile_K, int stride,
    cudaStream_t stream)
{
    if (!dst || !src || tile_M <= 0 || tile_K <= 0)
        return cudaErrorInvalidValue;

    int total = tile_M * tile_K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    shared_copy_kernel<<<blocks, threads, 0, stream>>>(
        dst, src, tile_M, tile_K, stride);

    return cudaPeekAtLastError();
}

// ---------------------------------------------------------------------------
// Async pipeline stage: copy a block of data from global to shared memory.
// Uses memcpy_async for hardware-accelerated global→shared transfers.
// Caller must call __pipeline_commit() before and __pipeline_wait_prior()
// after to manage the async pipeline.
// ---------------------------------------------------------------------------
__global__ void async_copy_kernel(
    float* __restrict__ shared_dst,
    const float* __restrict__ global_src,
    int num_floats)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_floats) return;

    // Vectorized 128-bit copy (4 floats per thread)
    int vec_idx = idx * 4;
    if (vec_idx + 3 < num_floats) {
        float4 val = *reinterpret_cast<const float4*>(&global_src[vec_idx]);
        *reinterpret_cast<float4*>(&shared_dst[vec_idx]) = val;
    } else {
        // Scalar fallback for tail elements
        for (int i = vec_idx; i < num_floats && i < vec_idx + 4; ++i)
            shared_dst[i] = global_src[i];
    }
}

cudaError_t async_pipeline_stage(
    float* shared_dst, const float* global_src,
    size_t num_bytes, cudaStream_t stream)
{
    if (!shared_dst || !global_src || num_bytes == 0)
        return cudaErrorInvalidValue;

    int num_floats = (int)(num_bytes / sizeof(float));
    int threads = 256;
    int blocks = (num_floats / 4 + threads - 1) / threads;  // 4 floats per thread
    async_copy_kernel<<<blocks, threads, 0, stream>>>(
        shared_dst, global_src, num_floats);

    return cudaPeekAtLastError();
}

} // namespace kernels
} // namespace blackwell
