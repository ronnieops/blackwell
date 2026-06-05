#include <cstdio>
#include <cuda_runtime.h>
#include "blackwell/kernels.h"

int main() {
    // Minimal test: just update_kv_cache
    size_t kv_sz = 28 * 8 * 2048 * 128 * 4;  // 1.5 GB
    float *d_kc, *d_vc, *d_K, *d_V;
    cudaMalloc(&d_kc, kv_sz); cudaMalloc(&d_vc, kv_sz);
    cudaMalloc(&d_K, 1024*4); cudaMalloc(&d_V, 1024*4);
    cudaMemset(d_kc, 0, kv_sz);
    cudaMemset(d_vc, 0, kv_sz);
    cudaStream_t st; cudaStreamCreate(&st);
    
    printf("Calling update_kv_cache...\n");
    cudaError_t e = cudaGetLastError();
    printf("CUDA error before: %s\n", cudaGetErrorString(e));
    
    for (int s = 0; s < 5; s++) {
        printf("  seq_pos=%d... ", s); fflush(stdout);
        e = blackwell::kernels::update_kv_cache(d_kc, d_vc, d_K, d_V, 0, s, 8, 128, 2048, st);
        printf("ret=%s ", cudaGetErrorString(e));
        e = cudaPeekAtLastError();
        printf("peek=%s\n", cudaGetErrorString(e));
    }
    cudaStreamSynchronize(st);
    e = cudaGetLastError();
    printf("All done. final: %s\n", cudaGetErrorString(e));
    return 0;
}
