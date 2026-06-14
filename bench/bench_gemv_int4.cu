// Microbenchmark: INT4 GEMV kernel performance
// Measures: throughput, effective bandwidth, compute utilization
// Usage: ./bench/bench_gemv_int4 [K] [N] [M] [iterations]
// Default: K=4096, N=4096, M=1, 100 iters (matches q_proj for 8B)

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "blackwell/kernels.h"

#define CK(x) do { auto _e = (x); if(_e != cudaSuccess) { fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1); } } while(0)

int main(int argc, char** argv) {
    int K = argc > 1 ? atoi(argv[1]) : 4096;
    int N = argc > 2 ? atoi(argv[2]) : 4096;
    int M = argc > 3 ? atoi(argv[3]) : 1;
    int iters = argc > 4 ? atoi(argv[4]) : 100;

    cudaDeviceProp P; cudaGetDeviceProperties(&P, 0);
    constexpr double peak_bw = 500.0; // RTX 5060 Ti GDDR7, ~500 GB/s
    fprintf(stderr, "Device: %s (CC %d.%d, %d SMs, %.0f GB/s peak BW)\n",
        P.name, P.major, P.minor, P.multiProcessorCount, peak_bw);

    fprintf(stderr, "GEMV: M=%d, K=%d, N=%d, iters=%d\n", M, K, N, iters);

    // Weight data: [N][K/2] packed INT4 + [N][K/16] scales
    size_t w_bytes = (size_t)N * K / 2;
    size_t w_sc_bytes = (size_t)N * (K / 16) * 4;
    size_t x_bytes = (size_t)M * K / 2;
    size_t x_sc_bytes = (size_t)M * (K / 16) * 4;
    size_t y_bytes = (size_t)M * N * 4;

    uint8_t *d_W, *d_x; float *d_Wsc, *d_xsc, *d_y;
    CK(cudaMalloc(&d_W, w_bytes));
    CK(cudaMalloc(&d_Wsc, w_sc_bytes));
    CK(cudaMalloc(&d_x, x_bytes));
    CK(cudaMalloc(&d_xsc, x_sc_bytes));
    CK(cudaMalloc(&d_y, y_bytes));

    // Fill with random data
    uint8_t* h_W = new uint8_t[w_bytes]; memset(h_W, 0x12, w_bytes);
    float* h_Wsc = new float[w_sc_bytes / 4]; for(size_t i=0;i<w_sc_bytes/4;i++) h_Wsc[i] = 0.1f;
    CK(cudaMemcpy(d_W, h_W, w_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_Wsc, h_Wsc, w_sc_bytes, cudaMemcpyHostToDevice));

    cudaStream_t st; CK(cudaStreamCreate(&st));
    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));

    // Warmup
    for(int i=0;i<5;i++) CK(blackwell::kernels::gemv_int4_batched(d_y, d_x, d_xsc, d_W, d_Wsc, K, N, M, st));
    CK(cudaStreamSynchronize(st));

    // Benchmark
    CK(cudaEventRecord(t0, st));
    for(int i=0;i<iters;i++) {
        CK(blackwell::kernels::gemv_int4_batched(d_y, d_x, d_xsc, d_W, d_Wsc, K, N, M, st));
    }
    CK(cudaEventRecord(t1, st));
    CK(cudaStreamSynchronize(st));

    float ms;
    CK(cudaEventElapsedTime(&ms, t0, t1));

    double per_call_us = ms * 1000.0 / iters;
    double total_bytes = (double)(w_bytes + w_sc_bytes + x_bytes + x_sc_bytes + y_bytes);
    double bw_gb_s = total_bytes / (per_call_us * 1e-6) / 1e9;
    constexpr double peak_bw_f = 500.0;
    double ops = 2.0 * M * K * N; // multiply-adds
    double tops = ops / (per_call_us * 1e-6) / 1e12;

    printf("── INT4 GEMV Benchmark ──\n");
    printf("  Config: M=%d, K=%d, N=%d\n", M, K, N);
    printf("  Avg time: %.1f µs/call (%d iters)\n", per_call_us, iters);
    printf("  Data read: %.1f KB/call\n", total_bytes / 1024.0);
    printf("  Bandwidth: %.1f GB/s (%.0f%% of %.0f GB/s peak)\n", bw_gb_s, 100*bw_gb_s/peak_bw_f, peak_bw_f);
    printf("  TOPS: %.2f\n", tops);

    // Also test different sizes for 8B model layers
    printf("\n── 8B Model GEMV Sizes ──\n");
    struct { const char* name; int k, n; } sizes[] = {
        {"q_proj", 4096, 4096},
        {"k_proj", 4096, 1024},
        {"v_proj", 4096, 1024},
        {"o_proj", 4096, 4096},
        {"gate_proj", 4096, 14336},
        {"up_proj", 4096, 14336},
        {"down_proj", 14336, 4096},
        {"lm_head", 4096, 151936},
    };
    int ns = sizeof(sizes)/sizeof(sizes[0]);

    for(int s=0; s<ns; s++) {
        int Sk = sizes[s].k, Sn = sizes[s].n;
        size_t Sw = (size_t)Sn * Sk / 2;
        size_t Ss = (size_t)Sn * (Sk/16) * 4;
        uint8_t *dSw; float *dSsc;
        CK(cudaMalloc(&dSw, Sw));
        CK(cudaMalloc(&dSsc, Ss));
        uint8_t* htmp = new uint8_t[Sw]; memset(htmp, 0x12, Sw);
        CK(cudaMemcpy(dSw, htmp, Sw, cudaMemcpyHostToDevice));
        float* htmpf = new float[Ss/4]; for(size_t i=0;i<Ss/4;i++) htmpf[i]=0.1f;
        CK(cudaMemcpy(dSsc, htmpf, Ss, cudaMemcpyHostToDevice));
        delete[] htmp; delete[] htmpf;

        // Alloc act buffers for correct size
        size_t Sx = (size_t)M * Sk / 2;
        size_t Sxsc = (size_t)M * (Sk/16) * 4;
        size_t Sy = (size_t)M * Sn * 4;
        uint8_t *dSx; float *dSxsc, *dSy;
        CK(cudaMalloc(&dSx, Sx));
        CK(cudaMalloc(&dSxsc, Sxsc));
        CK(cudaMalloc(&dSy, Sy));

        // Warmup
        for(int i=0;i<3;i++) CK(blackwell::kernels::gemv_int4_batched(dSy, dSx, dSxsc, dSw, dSsc, Sk, Sn, M, st));
        CK(cudaStreamSynchronize(st));

        CK(cudaEventRecord(t0, st));
        for(int i=0;i<iters;i++) {
            CK(blackwell::kernels::gemv_int4_batched(dSy, dSx, dSxsc, dSw, dSsc, Sk, Sn, M, st));
        }
        CK(cudaEventRecord(t1, st));
        CK(cudaStreamSynchronize(st));
        CK(cudaEventElapsedTime(&ms, t0, t1));
        double us = ms * 1000.0 / iters;
        double data = (double)(Sw + Ss + Sx + Sxsc + Sy);
        printf("  %-12s K=%5d N=%6d: %7.1f µs  %.1f GB/s (%4.0f KB)\n",
            sizes[s].name, Sk, Sn, us, data/(us*1e-6)/1e9, data/1024.0);

        CK(cudaFree(dSw)); CK(cudaFree(dSsc));
        CK(cudaFree(dSx)); CK(cudaFree(dSxsc)); CK(cudaFree(dSy));
    }

    CK(cudaFree(d_W)); CK(cudaFree(d_Wsc));
    CK(cudaFree(d_x)); CK(cudaFree(d_xsc)); CK(cudaFree(d_y));
    return 0;
}
