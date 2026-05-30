// src/kernels/gemm_int8_mma.cu — INT8 GEMM with WMMA tensor cores (stub)
//
// WMMA m16n16k16 INT8 provides 4.8× speedup over dp4a for GEMM.
// Full integration pending (register pressure / ABI compatibility issues).
// Standalone benchmark: bench/bench_mma_standalone.cu

#include <cuda_runtime.h>
#include <cstdint>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {

cudaError_t gemm_int8_mma(
    float* C,
    const void* A_i8,
    const float* A_sc,
    const void* B_i8,
    const float* B_sc,
    int M, int N, int K,
    cudaStream_t stream)
{
    // Stub: not yet integrated (use gemm_int8_dp4a instead)
    return cudaErrorNotSupported;
}

}  // namespace kernels
}  // namespace blackwell
