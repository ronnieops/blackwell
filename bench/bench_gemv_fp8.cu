// bench_gemv_fp8.cu — FP8 GEMV throughput benchmark
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include "blackwell/kernels.h"

#define AL(e) do{if((e)!=cudaSuccess){fprintf(stderr,"FAIL %s:%d\n",__FILE__,__LINE__);exit(1);}}while(0)

int main(int argc, char** argv) {
    int K = 2048, N = 2048;
    int iters = argc > 1 ? atoi(argv[1]) : 1000;
    printf("FP8 GEMV benchmark: K=%d N=%d iters=%d\n", K, N, iters);

    // Allocate
    float *d_x, *d_y, *d_sc;
    uint8_t *d_w;
    AL(cudaMalloc(&d_x, K * 4));
    AL(cudaMalloc(&d_y, N * 4));
    AL(cudaMalloc(&d_w, (size_t)N * K));
    AL(cudaMalloc(&d_sc, N * 4));

    // Init with dummy data
    AL(cudaMemset(d_w, 0x38, (size_t)N * K)); // FP8 1.0
    float one = 1.0f;
    for (int i = 0; i < N; i++) AL(cudaMemcpy(&d_sc[i], &one, 4, cudaMemcpyHostToDevice));
    float val = 1.0f;
    for (int i = 0; i < K; i++) AL(cudaMemcpy(&d_x[i], &val, 4, cudaMemcpyHostToDevice));

    cudaStream_t st; AL(cudaStreamCreate(&st));

    // Warmup
    for (int i = 0; i < 10; i++)
        AL(blackwell::kernels::gemv_fp8_fp32act(d_y, d_x, d_w, d_sc, K, N, st));
    AL(cudaStreamSynchronize(st));

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
        AL(blackwell::kernels::gemv_fp8_fp32act(d_y, d_x, d_w, d_sc, K, N, st));
    AL(cudaStreamSynchronize(st));
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    // Bytes read: K*N (weights) + K (activation) + N (scale) = K*(N+1) + N
    size_t bytes = (size_t)K * N + K * 4 + (size_t)N * 4 + (size_t)N * 4;  // weights + act + scale + output
    double bw = bytes / (ms * 1e-3) / 1e9;

    printf("  %.3f ms/iter, %.1f GB/s effective BW\n", ms, bw);

    // Compare: INT8 GEMV
    int8_t *d_x_i8; float *d_x_sc;
    int8_t *d_w_i8; float *d_w_sc;
    AL(cudaMalloc(&d_x_i8, K));
    AL(cudaMalloc(&d_x_sc, (K/16)*4));
    AL(cudaMalloc(&d_w_i8, (size_t)N * K));
    AL(cudaMalloc(&d_w_sc, (size_t)(N * (K/16)) * 4));
    AL(cudaMemset(d_x_i8, 1, K));
    float sc = 0.01f;
    for (int i = 0; i < K/16; i++) AL(cudaMemcpy(&d_x_sc[i], &sc, 4, cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < 10; i++)
        AL(blackwell::kernels::gemv_int8_warp(d_y, d_x_i8, d_x_sc, d_w_i8, d_w_sc, K, N, st));
    AL(cudaStreamSynchronize(st));

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++)
        AL(blackwell::kernels::gemv_int8_warp(d_y, d_x_i8, d_x_sc, d_w_i8, d_w_sc, K, N, st));
    AL(cudaStreamSynchronize(st));
    t1 = std::chrono::high_resolution_clock::now();
    double ms_i8 = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    size_t bytes_i8 = (size_t)K * N + K + (K/16)*4 + (size_t)N*(K/16)*4 + (size_t)N*4;
    double bw_i8 = bytes_i8 / (ms_i8 * 1e-3) / 1e9;

    printf("  INT8: %.3f ms/iter, %.1f GB/s\n", ms_i8, bw_i8);
    printf("  FP8/INT8 ratio: %.2f×\n", ms / ms_i8);

    return 0;
}
