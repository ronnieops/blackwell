#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "blackwell/kernels.h"

int main() {
    float *d_x32, *d_y;
    void *d_x4, *d_W4;
    float *d_xs, *d_Ws;
    
    const int hidden = 2048;
    const int q_dim = 2048;
    const int kv_dim = 512;
    
    cudaMalloc(&d_x32, hidden * 4);
    cudaMalloc(&d_x4, hidden);
    cudaMalloc(&d_y, q_dim * 4);
    cudaMalloc(&d_W4, (long)hidden * q_dim);
    cudaMalloc(&d_xs, (hidden/16) * 4);
    cudaMalloc(&d_Ws, (hidden/16) * (q_dim/16) * 4);
    
    std::vector<float> x32(hidden, 1.0f);
    std::vector<float> xs(hidden/16, 1.0f/3.0f);
    std::vector<float> Ws((hidden/16)*(q_dim/16), 1.0f/3.0f);
    cudaMemcpy(d_x32, x32.data(), hidden*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_xs, xs.data(), (hidden/16)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ws, Ws.data(), (hidden/16)*(q_dim/16)*4, cudaMemcpyHostToDevice);
    blackwell::kernels::pack_fp4(d_x4, d_x32, d_xs, hidden, 0);
    
    // Allocate W_q with uniform 1.0
    float *d_W32;
    cudaMalloc(&d_W32, (long)hidden * q_dim * 4);
    std::vector<float> W((long)hidden * q_dim, 1.0f);
    cudaMemcpy(d_W32, W.data(), (long)hidden * q_dim * 4, cudaMemcpyHostToDevice);
    blackwell::kernels::pack_fp4(d_W4, d_W32, d_Ws, hidden * q_dim, 0);
    
    int warm = 10, bench = 200;
    for (int i = 0; i < warm; ++i)
        blackwell::kernels::gemv_fp4(d_y, d_x4, d_xs, d_W4, d_Ws, hidden, q_dim, 0);
    cudaDeviceSynchronize();
    
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s, 0);
    for (int i = 0; i < bench; ++i)
        blackwell::kernels::gemv_fp4(d_y, d_x4, d_xs, d_W4, d_Ws, hidden, q_dim, 0);
    cudaEventRecord(e, 0);
    cudaEventSynchronize(e);
    float ms;
    cudaEventElapsedTime(&ms, s, e);
    ms /= bench;
    printf("GEMV 2048x2048: %.4f ms\n", ms);
    
    // Also pack_fp4
    for (int i = 0; i < warm; ++i)
        blackwell::kernels::pack_fp4(d_x4, d_x32, d_xs, q_dim, 0);
    cudaDeviceSynchronize();
    cudaEventRecord(s, 0);
    for (int i = 0; i < bench; ++i)
        blackwell::kernels::pack_fp4(d_x4, d_x32, d_xs, q_dim, 0);
    cudaEventRecord(e, 0);
    cudaEventSynchronize(e);
    cudaEventElapsedTime(&ms, s, e);
    ms /= bench;
    printf("pack_fp4 2048:   %.4f ms\n", ms);
    
    // RMSNorm 2048
    float *d_rn;
    cudaMalloc(&d_rn, hidden * 4);
    for (int i = 0; i < warm; ++i)
        blackwell::kernels::fused_rmsnorm(d_y, d_x32, d_rn, hidden, 1e-5f, 0);
    cudaDeviceSynchronize();
    cudaEventRecord(s, 0);
    for (int i = 0; i < bench; ++i)
        blackwell::kernels::fused_rmsnorm(d_y, d_x32, d_rn, hidden, 1e-5f, 0);
    cudaEventRecord(e, 0);
    cudaEventSynchronize(e);
    cudaEventElapsedTime(&ms, s, e);
    ms /= bench;
    printf("rmsnorm 2048:    %.4f ms\n", ms);
    
    // Update KV cache 4 heads × 128
    float *d_kv;
    cudaMalloc(&d_kv, 4 * 2048 * 128 * 4);
    for (int i = 0; i < warm; ++i)
        blackwell::kernels::update_kv_cache(d_kv, d_kv, d_x32, d_x32, 0, 0, 4, 128, 2048, 0);
    cudaDeviceSynchronize();
    cudaEventRecord(s, 0);
    for (int i = 0; i < bench; ++i)
        blackwell::kernels::update_kv_cache(d_kv, d_kv, d_x32, d_x32, 0, 0, 4, 128, 2048, 0);
    cudaEventRecord(e, 0);
    cudaEventSynchronize(e);
    cudaEventElapsedTime(&ms, s, e);
    ms /= bench;
    printf("update_kv 4h:    %.4f ms\n", ms);
    
    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(d_x32); cudaFree(d_x4); cudaFree(d_y);
    cudaFree(d_W4); cudaFree(d_W32);
    cudaFree(d_xs); cudaFree(d_Ws); cudaFree(d_kv); cudaFree(d_rn);
    return 0;
}
