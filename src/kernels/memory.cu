// src/kernels/memory.cu — shared-memory tiling / DSM utilities
#include <cuda_runtime.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace blackwell {
namespace kernels {

// TODO(#4): Shared-memory tiled copy with 99 KB/block limit enforced.
//   CC 12.0: 128 KB shared mem/SM total, 99 KB max per thread block.
//   Tile M×K for A, tile K×N for B.  Both tiles must fit within 99 KB.
//   Example: 128×128×2 bytes float16 = 32768 B  → one A or B tile OK.
//   Beware: A+B+scales+metadata must all fit.

cudaError_t shared_copy_async(
    float* dst, const float* src,
    int tile_M, int tile_K, int stride,
    cudaStream_t stream) {
    (void)dst; (void)src; (void)tile_M; (void)tile_K;
    (void)stride; (void)stream;
    return cudaErrorNotReady;
}

// TODO(#8): Async copy / cuda::pipeline for prefill staging.
//   Overlap global→shared movement with compute.
//   Use memcpy_async where possible.

cudaError_t async_pipeline_stage(
    float* shared_dst, const float* global_src,
    size_t num_bytes, cudaStream_t stream) {
    (void)shared_dst; (void)global_src; (void)num_bytes; (void)stream;
    return cudaErrorNotReady;
}

} // namespace kernels
} // namespace blackwell
