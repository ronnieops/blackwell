// src/kernels/gemv_int8_pdl.cu — INT8 GEMV with PDL (Programmatic Dependent Launch)
//
// PDL allows the next kernel to start before the current one finishes,
// overlapping compute and memory operations for +3-5% speedup.
//
// Requires: CUDA Toolkit >= 12.3, SM >= 90 (Hopper/Blackwell)
//
// Build:
//   CUDACXX=/usr/local/cuda-13.3/bin/nvcc nvcc -std=c++17 -O3 \
//     -arch=sm_120a -I include src/kernels/gemv_int8_pdl.cu \
//     -L build -lblackwell_kernels -lcudart

#include <cuda_runtime.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {

// PDL support check
#if CUDART_VERSION >= 12030
#define BLACKWELL_USE_PDL 1
#else
#define BLACKWELL_USE_PDL 0
#endif

// Device-side PDL sync (call at kernel start)
static __device__ __forceinline__ void pdl_sync() {
#if BLACKWELL_USE_PDL
    cudaGridDependencySynchronize();
#endif
}

// Device-side PDL trigger (call at kernel end)
static __device__ __forceinline__ void pdl_trigger() {
#if BLACKWELL_USE_PDL
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// INT8 GEMV kernel with PDL
// Same as gemv_int8_kernel but with PDL sync/trigger
__launch_bounds__(64, 2)
__global__ void gemv_int8_pdl_kernel(
    float* __restrict__ y_out,
    const int8_t* __restrict__ x_int8,
    const float* __restrict__ x_scale,
    const int8_t* __restrict__ W_t_int8,
    const float* __restrict__ W_t_scale,
    int K, int N)
{
    // PDL sync: wait for previous kernel to be ready
    pdl_sync();
    
    constexpr int B = 16;
    int tid = threadIdx.x;
    int n_out = blockIdx.x * 64 + tid;
    if (n_out >= N) return;

    int num_K_blks = K / B;
    float acc = 0.0f;

    for (int kb = 0; kb < num_K_blks; ++kb) {
        const int8_t* w_ptr = &W_t_int8[n_out * K + kb * B];
        alignas(16) int8_t w_buf[B];
        alignas(16) int8_t x_buf[B];
        *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);
        *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_int8 + kb * B);

        int sumi = 0;
        #pragma unroll
        for (int i = 0; i < B; ++i) {
            sumi += __dp4a(static_cast<int>(w_buf[i]), static_cast<int>(x_buf[i]), 0);
        }

        float w_scale = W_t_scale[n_out * num_K_blks + kb];
        float x_scale_val = x_scale[kb];
        acc += static_cast<float>(sumi) * w_scale * x_scale_val;
    }

    y_out[n_out] = acc;
    
    // PDL trigger: signal next kernel can start
    pdl_trigger();
}

// Launch wrapper for PDL GEMV
cudaError_t gemv_int8_pdl(
    float* y_out,
    const void* x_int8,
    const float* x_scale,
    const void* W_t_int8,
    const float* W_t_scale,
    int K, int N,
    cudaStream_t stream)
{
#if BLACKWELL_USE_PDL
    // Check if PDL is enabled via environment variable
    static bool pdl_enabled = []() {
        const char* env = getenv("BLACKWELL_PDL");
        return env == nullptr || atoi(env) != 0;
    }();
    
    if (pdl_enabled) {
        // Use cudaLaunchKernelEx for PDL
        dim3 grid((N + 63) / 64);
        dim3 block(64);
        
        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr.val.programmaticStreamSerializationAllowed = 1;
        
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = grid;
        cfg.blockDim = block;
        cfg.dynamicSmemBytes = 0;
        cfg.stream = stream;
        cfg.attrs = &attr;
        cfg.numAttrs = 1;
        
        void* args[] = {&y_out, &x_int8, &x_scale, &W_t_int8, &W_t_scale, &K, &N};
        cudaError_t err = cudaLaunchKernelEx(&cfg, gemv_int8_pdl_kernel,
            y_out, static_cast<const int8_t*>(x_int8), x_scale, static_cast<const int8_t*>(W_t_int8), W_t_scale, K, N);
        return err;
    }
#endif
    
    // Fallback: regular launch
    dim3 grid((N + 63) / 64);
    dim3 block(64);
    gemv_int8_pdl_kernel<<<grid, block, 0, stream>>>(
        y_out, static_cast<const int8_t*>(x_int8), x_scale, static_cast<const int8_t*>(W_t_int8), W_t_scale, K, N);
    return cudaGetLastError();
}

}  // namespace kernels
}  // namespace blackwell
