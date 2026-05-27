// bench/gemv_v2_bench.cu — Compare original GEMV vs vectorized v2
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120,code=sm_120 \
//     -I include bench/gemv_v2_bench.cu build/libblackwell_kernels.a \
//     -o bench/gemv_v2_bench

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "blackwell/kernels.h"

struct Timer {
    cudaEvent_t s, e;
    Timer() { cudaEventCreate(&s); cudaEventCreate(&e); }
    ~Timer() { cudaEventDestroy(s); cudaEventDestroy(e); }
    void start() { cudaEventRecord(s, 0); }
    float stop() { cudaEventRecord(e, 0); cudaEventSynchronize(e);
                   float ms; cudaEventElapsedTime(&ms, s, e); return ms; }
};

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("# GEMV v1 vs v2 Benchmark\n");
    printf("Device: %s (CC %d.%d)\n\n", p.name, p.major, p.minor);

    const float s13 = 1.0f / 3.0f;
    const int B = 16;

    struct Test { int K, N; const char* label; };
    Test tests[] = {
        {2048, 2048, "O-proj (attn)"},
        {2048, 1024, "K/V-proj"},
        {2048, 6144, "gate/up-proj"},
        {6144, 2048, "down-proj"},
        {4096, 4096, "large square"},
    };
    int ntests = sizeof(tests)/sizeof(tests[0]);
    int warmup = 5, bench = 50;

    printf("| %-16s | %6s | %6s | %8s | %8s | %7s | %8s | %7s | %6s |\n",
           "Test", "K", "N", "v1 ms", "v2 ms", "speedup", "v1 GB/s", "v2 GB/s", "%peak");
    printf("|%s|\n", "-----------------------------------------------------------------------------------------------");

    for (int t = 0; t < ntests; ++t) {
        int K = tests[t].K, N = tests[t].N;
        int num_K_blks = K / B, num_N_blks = N / B;

        // Allocate original (K×N) layout
        void* d_W; float* d_Ws;
        cudaMalloc(&d_W, (size_t)K * N);
        cudaMalloc(&d_Ws, (size_t)num_K_blks * num_N_blks * 4);

        // Allocate transposed (N×K) layout
        void* d_Wt; float* d_Wts;
        cudaMalloc(&d_Wt, (size_t)N * K);
        cudaMalloc(&d_Wts, (size_t)num_N_blks * num_K_blks * 4);

        // Allocate x, y
        void* d_x; float* d_xs, *d_y1, *d_y2;
        cudaMalloc(&d_x, K);
        cudaMalloc(&d_xs, num_K_blks * 4);
        cudaMalloc(&d_y1, N * 4);
        cudaMalloc(&d_y2, N * 4);

        // Init weights (all 1.0 packed)
        {
            float* d_tmp;
            cudaMalloc(&d_tmp, (size_t)K * N * 4);
            std::vector<float> ones(K * N, 1.0f);
            cudaMemcpy(d_tmp, ones.data(), (size_t)K * N * 4, cudaMemcpyHostToDevice);
            std::vector<float> ws(num_K_blks * num_N_blks, s13);
            cudaMemcpy(d_Ws, ws.data(), ws.size() * 4, cudaMemcpyHostToDevice);
            blackwell::kernels::pack_fp4(d_W, d_tmp, d_Ws, K * N, 0);
            cudaFree(d_tmp);
        }

        // Transpose weights
        blackwell::kernels::transpose_fp4_weights(d_Wt, d_Wts, d_W, d_Ws, K, N, 0);

        // Init x (all 1.0)
        {
            float* d_tmp;
            cudaMalloc(&d_tmp, K * 4);
            std::vector<float> ones(K, 1.0f);
            cudaMemcpy(d_tmp, ones.data(), K * 4, cudaMemcpyHostToDevice);
            std::vector<float> xs(num_K_blks, s13);
            cudaMemcpy(d_xs, xs.data(), num_K_blks * 4, cudaMemcpyHostToDevice);
            blackwell::kernels::pack_fp4(d_x, d_tmp, d_xs, K, 0);
            cudaFree(d_tmp);
        }

        // Warmup v1
        for (int i = 0; i < warmup; ++i)
            blackwell::kernels::gemv_fp4(d_y1, d_x, d_xs, d_W, d_Ws, K, N, 0);
        cudaDeviceSynchronize();

        Timer t1;
        t1.start();
        for (int i = 0; i < bench; ++i)
            blackwell::kernels::gemv_fp4(d_y1, d_x, d_xs, d_W, d_Ws, K, N, 0);
        float ms1 = t1.stop() / bench;

        // Warmup v2
        for (int i = 0; i < warmup; ++i)
            blackwell::kernels::gemv_fp4_v2(d_y2, d_x, d_xs, d_Wt, d_Wts, K, N, 0);
        cudaDeviceSynchronize();

        Timer t2;
        t2.start();
        for (int i = 0; i < bench; ++i)
            blackwell::kernels::gemv_fp4_v2(d_y2, d_x, d_xs, d_Wt, d_Wts, K, N, 0);
        float ms2 = t2.stop() / bench;

        // Verify correctness
        std::vector<float> y1(N), y2(N);
        cudaMemcpy(y1.data(), d_y1, N * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(y2.data(), d_y2, N * 4, cudaMemcpyDeviceToHost);
        float max_err = 0.f;
        for (int i = 0; i < N; ++i) {
            float d = fabsf(y1[i]) > 0.01f ? fabsf(y1[i]) : 0.01f;
            float e = fabsf(y1[i] - y2[i]) / d;
            if (e > max_err) max_err = e;
        }

        // BW calculation
        size_t total_bytes = K + num_K_blks*4 + (size_t)K*N + (size_t)num_K_blks*num_N_blks*4 + N*4;
        double bw1 = (double)total_bytes / 1e9 / (ms1 / 1e3);
        double bw2 = (double)total_bytes / 1e9 / (ms2 / 1e3);
        double speedup = ms1 / ms2;
        double pct_peak = bw2 / 500.0 * 100.0;

        printf("| %-16s | %6d | %6d | %8.3f | %8.3f | %7.2fx | %8.1f | %7.1f | %5.1f%% | err=%.4f\n",
               tests[t].label, K, N, ms1, ms2, speedup, bw1, bw2, pct_peak, max_err);

        cudaFree(d_W); cudaFree(d_Ws);
        cudaFree(d_Wt); cudaFree(d_Wts);
        cudaFree(d_x); cudaFree(d_xs);
        cudaFree(d_y1); cudaFree(d_y2);
    }

    return 0;
}
