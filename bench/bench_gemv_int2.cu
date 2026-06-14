// INT2 GEMV benchmark: measure throughput vs INT4
// Usage: ./bench/bench_gemv_int2
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "blackwell/kernels.h"

#define CK(x) do { auto _e = (x); if(_e != cudaSuccess) { fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1); } } while(0)

int main() {
    constexpr double peak_bw = 500.0;
    int K = 4096, N = 4096, M = 1, iters = 200;

    cudaDeviceProp P; cudaGetDeviceProperties(&P, 0);
    printf("Device: %s (CC %d.%d)\n\n", P.name, P.major, P.minor);

    // Weight: INT2 [N][K/4] + scales [N][K/16]
    size_t w2 = (size_t)N * K / 4;
    size_t ws = (size_t)N * (K/16) * 4;
    // Activation: INT4 [M][K/2] + scales
    size_t x4 = (size_t)M * K / 2;
    size_t xs = (size_t)M * (K/16) * 4;
    size_t yb = (size_t)M * N * 4;

    uint8_t *d_W2, *d_x4; float *d_Wsc, *d_xsc, *d_y;
    CK(cudaMalloc(&d_W2, w2)); CK(cudaMalloc(&d_Wsc, ws));
    CK(cudaMalloc(&d_x4, x4)); CK(cudaMalloc(&d_xsc, xs));
    CK(cudaMalloc(&d_y, yb));

    // Fill with test data
    uint8_t* h_W2 = new uint8_t[w2]; memset(h_W2, 0x55, w2); // alternating 01 pattern
    float* h_Wsc = new float[ws/4]; for(size_t i=0;i<ws/4;i++) h_Wsc[i] = 0.1f;
    uint8_t* h_x4 = new uint8_t[x4]; memset(h_x4, 0x88, x4);
    float* h_xsc = new float[xs/4]; for(size_t i=0;i<xs/4;i++) h_xsc[i] = 0.1f;
    CK(cudaMemcpy(d_W2, h_W2, w2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_Wsc, h_Wsc, ws, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_x4, h_x4, x4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_xsc, h_xsc, xs, cudaMemcpyHostToDevice));

    cudaStream_t st; CK(cudaStreamCreate(&st));
    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));

    // Warmup
    for(int i=0;i<10;i++) CK(blackwell::kernels::gemv_int2_batched(d_y, d_x4, d_xsc, d_W2, d_Wsc, K, N, M, st));
    CK(cudaStreamSynchronize(st));

    // INT2 benchmark
    CK(cudaEventRecord(t0, st));
    for(int i=0;i<iters;i++) CK(blackwell::kernels::gemv_int2_batched(d_y, d_x4, d_xsc, d_W2, d_Wsc, K, N, M, st));
    CK(cudaEventRecord(t1, st));
    CK(cudaStreamSynchronize(st));
    float ms_int2; CK(cudaEventElapsedTime(&ms_int2, t0, t1));

    // INT4 benchmark (for comparison)
    size_t w4 = (size_t)N * K / 2;
    uint8_t *d_W4;
    CK(cudaMalloc(&d_W4, w4));
    uint8_t* h_W4 = new uint8_t[w4]; memset(h_W4, 0x12, w4);
    CK(cudaMemcpy(d_W4, h_W4, w4, cudaMemcpyHostToDevice));

    for(int i=0;i<10;i++) CK(blackwell::kernels::gemv_int4_batched(d_y, d_x4, d_xsc, d_W4, d_Wsc, K, N, M, st));
    CK(cudaStreamSynchronize(st));
    CK(cudaEventRecord(t0, st));
    for(int i=0;i<iters;i++) CK(blackwell::kernels::gemv_int4_batched(d_y, d_x4, d_xsc, d_W4, d_Wsc, K, N, M, st));
    CK(cudaEventRecord(t1, st));
    CK(cudaStreamSynchronize(st));
    float ms_int4; CK(cudaEventElapsedTime(&ms_int4, t0, t1));

    double us_int2 = ms_int2 * 1000.0 / iters;
    double us_int4 = ms_int4 * 1000.0 / iters;
    double data_int2 = (double)(w2 + ws + x4 + xs + yb);
    double data_int4 = (double)(w4 + ws + x4 + xs + yb);

    printf("── INT2 vs INT4 GEMV (M=1, K=%d, N=%d) ──\n", K, N);
    printf("  INT2: %7.1f µs  %.1f GB/s  (%.0f KB data)\n", us_int2, data_int2/(us_int2*1e-6)/1e9, data_int2/1024);
    printf("  INT4: %7.1f µs  %.1f GB/s  (%.0f KB data)\n", us_int4, data_int4/(us_int4*1e-6)/1e9, data_int4/1024);
    printf("  Speedup: %.2fx\n", us_int4 / us_int2);

    // Test 8B model layer sizes
    printf("\n── 8B Model Layer GEMV (INT2 vs INT4) ──\n");
    struct { const char* name; int k, n; } sizes[] = {
        {"q_proj", 4096, 4096},
        {"k_proj", 4096, 1024},
        {"gate_proj", 4096, 14336},
        {"lm_head", 4096, 151936},
    };
    for(auto& s : sizes) {
        // Alloc for this size
        size_t sw2 = (size_t)s.n * s.k / 4;
        size_t sw4 = (size_t)s.n * s.k / 2;
        size_t ss = (size_t)s.n * (s.k/16) * 4;
        size_t sx4 = (size_t)M * s.k / 2;
        size_t sxs = (size_t)M * (s.k/16) * 4;
        size_t sy = (size_t)M * s.n * 4;
        uint8_t *dsw2, *dsw4, *dsx4; float *dssc, *dsxsc, *dsy;
        CK(cudaMalloc(&dsw2, sw2)); CK(cudaMalloc(&dsw4, sw4));
        CK(cudaMalloc(&dssc, ss)); CK(cudaMalloc(&dsx4, sx4));
        CK(cudaMalloc(&dsxsc, sxs)); CK(cudaMalloc(&dsy, sy));

        // INT2
        for(int i=0;i<5;i++) CK(blackwell::kernels::gemv_int2_batched(dsy, dsx4, dsxsc, dsw2, dssc, s.k, s.n, M, st));
        CK(cudaStreamSynchronize(st));
        CK(cudaEventRecord(t0, st));
        for(int i=0;i<iters;i++) CK(blackwell::kernels::gemv_int2_batched(dsy, dsx4, dsxsc, dsw2, dssc, s.k, s.n, M, st));
        CK(cudaEventRecord(t1, st)); CK(cudaStreamSynchronize(st));
        float ms2; CK(cudaEventElapsedTime(&ms2, t0, t1));

        // INT4
        for(int i=0;i<5;i++) CK(blackwell::kernels::gemv_int4_batched(dsy, dsx4, dsxsc, dsw4, dssc, s.k, s.n, M, st));
        CK(cudaStreamSynchronize(st));
        CK(cudaEventRecord(t0, st));
        for(int i=0;i<iters;i++) CK(blackwell::kernels::gemv_int4_batched(dsy, dsx4, dsxsc, dsw4, dssc, s.k, s.n, M, st));
        CK(cudaEventRecord(t1, st)); CK(cudaStreamSynchronize(st));
        float ms4; CK(cudaEventElapsedTime(&ms4, t0, t1));

        printf("  %-12s: INT2 %7.1f µs  INT4 %7.1f µs  ratio %.2fx\n",
            s.name, ms2*1000/iters, ms4*1000/iters, ms4/ms2);

        CK(cudaFree(dsw2)); CK(cudaFree(dsw4)); CK(cudaFree(dssc));
        CK(cudaFree(dsx4)); CK(cudaFree(dsxsc)); CK(cudaFree(dsy));
    }

    CK(cudaFree(d_W2)); CK(cudaFree(d_W4)); CK(cudaFree(d_Wsc));
    CK(cudaFree(d_x4)); CK(cudaFree(d_xsc)); CK(cudaFree(d_y));
    return 0;
}
