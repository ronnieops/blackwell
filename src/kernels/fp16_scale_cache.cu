// src/kernels/fp16_scale_cache.cu — FP16 scale cache for GEMV
//
// Converts FP32 scales to FP16 at load time and caches them.
// Reduces scale memory by 50% during inference.
//
// Build:
//   CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake --build build --parallel

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {

// Global cache for FP16 scales (per weight matrix)
struct FP16ScaleCache {
    __half* d_scales_fp16;
    int count;
    bool initialized;
    
    FP16ScaleCache() : d_scales_fp16(nullptr), count(0), initialized(false) {}
    
    ~FP16ScaleCache() {
        if (d_scales_fp16) cudaFree(d_scales_fp16);
    }
    
    // Initialize cache from FP32 scales
    cudaError_t init(const float* fp32_scales, int num_scales, cudaStream_t stream) {
        if (initialized && count == num_scales) return cudaSuccess;
        
        if (d_scales_fp16) cudaFree(d_scales_fp16);
        
        cudaError_t err = cudaMalloc(&d_scales_fp16, num_scales * sizeof(__half));
        if (err != cudaSuccess) return err;
        
        // Convert FP32 to FP16
        err = convert_scales_fp32_to_fp16(fp32_scales, d_scales_fp16, num_scales, stream);
        if (err != cudaSuccess) return err;
        
        count = num_scales;
        initialized = true;
        return cudaSuccess;
    }
    
    // Get FP16 scales
    const __half* get() const { return d_scales_fp16; }
};

// Static cache instances (one per weight matrix type)
static FP16ScaleCache g_w_scale_cache;
static FP16ScaleCache g_x_scale_cache;

// Convert FP32 scales to FP16
__global__ void fp32_to_fp16_kernel(const float* in, __half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

cudaError_t convert_scales_fp32_to_fp16(
    const float* fp32_scales,
    __half* fp16_scales,
    int count,
    cudaStream_t stream)
{
    fp32_to_fp16_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
        fp32_scales, fp16_scales, count);
    return cudaGetLastError();
}

// INT8 GEMV with cached FP16 scales
// Converts scales at first call, reuses cached FP16 scales thereafter
cudaError_t gemv_int8_fp16cached(
    float* y_out,
    const void* x_int8,
    const float* x_scale,      // FP32 scales [K/16]
    const void* W_t_int8,
    const float* W_t_scale,    // FP32 scales [N × K/16]
    int K, int N,
    cudaStream_t stream)
{
    // Convert weight scales to FP16 (cached)
    int num_w_scales = N * (K / 16);
    cudaError_t err = g_w_scale_cache.init(W_t_scale, num_w_scales, stream);
    if (err != cudaSuccess) return err;
    
    // Convert activation scales to FP16 (cached)
    int num_x_scales = K / 16;
    err = g_x_scale_cache.init(x_scale, num_x_scales, stream);
    if (err != cudaSuccess) return err;
    
    // Launch GEMV with FP16 scales
    return gemv_int8_fp16sc(
        y_out, x_int8, g_x_scale_cache.get(),
        W_t_int8, g_w_scale_cache.get(), K, N, stream);
}

// Clear scale caches (call when weights are reloaded)
void clear_fp16_scale_caches() {
    g_w_scale_cache = FP16ScaleCache();
    g_x_scale_cache = FP16ScaleCache();
}

}  // namespace kernels
}  // namespace blackwell
