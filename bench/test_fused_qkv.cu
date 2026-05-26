#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "blackwell/kernels.h"

int main() {
    const int hidden = 2048, q_dim = 2048, kv_dim = 512;
    
    void *d_x4, *d_Wq, *d_Wk, *d_Wv;
    float *d_xs, *d_Wqs, *d_Wks, *d_Wvs;
    float *d_Q, *d_K, *d_V;
    float *d_x32;
    
    cudaMalloc(&d_x32, hidden * 4);
    cudaMalloc(&d_x4, hidden);
    cudaMalloc(&d_Wq, (long)hidden * q_dim);
    cudaMalloc(&d_Wk, (long)hidden * kv_dim);
    cudaMalloc(&d_Wv, (long)hidden * kv_dim);
    cudaMalloc(&d_xs, (hidden/16)*4);
    cudaMalloc(&d_Wqs, (long)(hidden/16)*(q_dim/16)*4);
    cudaMalloc(&d_Wks, (long)(hidden/16)*(kv_dim/16)*4);
    cudaMalloc(&d_Wvs, (long)(hidden/16)*(kv_dim/16)*4);
    cudaMalloc(&d_Q, q_dim*4);
    cudaMalloc(&d_K, kv_dim*4);
    cudaMalloc(&d_V, kv_dim*4);
    
    // Fill with uniform 1.0
    std::vector<float> xs(hidden/16, 1/3.f);
    std::vector<float> wqs((long)(hidden/16)*(q_dim/16), 1/3.f);
    std::vector<float> wks((long)(hidden/16)*(kv_dim/16), 1/3.f);
    std::vector<float> x32(hidden, 1.0f);
    cudaMemcpy(d_xs, xs.data(), (hidden/16)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wqs, wqs.data(), (long)(hidden/16)*(q_dim/16)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wks, wks.data(), (long)(hidden/16)*(kv_dim/16)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wvs, wks.data(), (long)(hidden/16)*(kv_dim/16)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x32, x32.data(), hidden*4, cudaMemcpyHostToDevice);
    blackwell::kernels::pack_fp4(d_x4, d_x32, d_xs, hidden, 0);
    
    // Fill weights
    std::vector<float> Wq((long)hidden * q_dim, 1.0f);
    std::vector<float> Wkv((long)hidden * kv_dim, 1.0f);
    float *d_Wf;
    cudaMalloc(&d_Wf, (long)hidden * q_dim * 4);
    cudaMemcpy(d_Wf, Wq.data(), (long)hidden * q_dim * 4, cudaMemcpyHostToDevice);
    blackwell::kernels::pack_fp4(d_Wq, d_Wf, d_Wqs, hidden * q_dim, 0);
    cudaFree(d_Wf);
    
    cudaMalloc(&d_Wf, (long)hidden * kv_dim * 4);
    cudaMemcpy(d_Wf, Wkv.data(), (long)hidden * kv_dim * 4, cudaMemcpyHostToDevice);
    blackwell::kernels::pack_fp4(d_Wk, d_Wf, d_Wks, hidden * kv_dim, 0);
    blackwell::kernels::pack_fp4(d_Wv, d_Wf, d_Wvs, hidden * kv_dim, 0);
    cudaFree(d_Wf);
    
    int warm = 10, bench = 200;
    
    // Benchmark: 3 separate GEMVs
    for (int i = 0; i < warm; ++i) {
        blackwell::kernels::gemv_fp4(d_Q, d_x4, d_xs, d_Wq, d_Wqs, hidden, q_dim, 0);
        blackwell::kernels::gemv_fp4(d_K, d_x4, d_xs, d_Wk, d_Wks, hidden, kv_dim, 0);
        blackwell::kernels::gemv_fp4(d_V, d_x4, d_xs, d_Wv, d_Wvs, hidden, kv_dim, 0);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s, 0);
    for (int i = 0; i < bench; ++i) {
        blackwell::kernels::gemv_fp4(d_Q, d_x4, d_xs, d_Wq, d_Wqs, hidden, q_dim, 0);
        blackwell::kernels::gemv_fp4(d_K, d_x4, d_xs, d_Wk, d_Wks, hidden, kv_dim, 0);
        blackwell::kernels::gemv_fp4(d_V, d_x4, d_xs, d_Wv, d_Wvs, hidden, kv_dim, 0);
    }
    cudaEventRecord(e, 0);
    cudaEventSynchronize(e);
    float ms_sep;
    cudaEventElapsedTime(&ms_sep, s, e);
    ms_sep /= bench;
    printf("3 separate GEMVs: %.4f ms\n", ms_sep);
    
    // Benchmark: 1 fused QKV
    for (int i = 0; i < warm; ++i) {
        blackwell::kernels::fused_qkv_gemv(d_Q, d_K, d_V,
            d_x4, d_xs, d_Wq, d_Wqs, d_Wk, d_Wks, d_Wv, d_Wvs,
            hidden, q_dim, kv_dim, 0);
    }
    cudaDeviceSynchronize();
    
    cudaEventRecord(s, 0);
    for (int i = 0; i < bench; ++i) {
        blackwell::kernels::fused_qkv_gemv(d_Q, d_K, d_V,
            d_x4, d_xs, d_Wq, d_Wqs, d_Wk, d_Wks, d_Wv, d_Wvs,
            hidden, q_dim, kv_dim, 0);
    }
    cudaEventRecord(e, 0);
    cudaEventSynchronize(e);
    float ms_fused;
    cudaEventElapsedTime(&ms_fused, s, e);
    ms_fused /= bench;
    printf("1 fused QKV:      %.4f ms\n", ms_fused);
    printf("Speedup:          %.1f%%\n", (ms_sep / ms_fused) * 100.0f);
    
    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(d_x32); cudaFree(d_x4);
    cudaFree(d_Wq); cudaFree(d_Wk); cudaFree(d_Wv);
    cudaFree(d_xs); cudaFree(d_Wqs); cudaFree(d_Wks); cudaFree(d_Wvs);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    return 0;
}
