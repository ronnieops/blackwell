// bench/gemv_char.cu — GEMV characterization: bandwidth, scaling, occupancy
//
// Measures GEMV throughput for various (K, N) sizes to understand
// memory bandwidth utilization and scaling behavior.
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120,code=sm_120 \
//     -I include bench/gemv_char.cu build/libblackwell_kernels.a \
//     -o bench/gemv_char

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
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
    const double peak_bw = 500.0; // GB/s approximate GDDR7 peak for RTX 5060 Ti
    printf("# GEMV Characterization\n");
    printf("Device: %s (CC %d.%d)\n", p.name, p.major, p.minor);
    printf("Peak BW (approx): %.0f GB/s\n\n", peak_bw);

    const float s13 = 1.0f / 3.0f;
    const int B = 16;

    // Test various (K, N) combinations
    struct Test { int K, N; const char* label; };
    Test tests[] = {
        {2048, 2048, "O-proj (attn)"},
        {2048, 1024, "K/V-proj"},
        {2048, 6144, "gate/up-proj (MLP)"},
        {6144, 2048, "down-proj (MLP)"},
        {4096, 4096, "large square"},
        {1024, 2048, "small K"},
        {2048, 512,  "small N"},
        {2048, 4096, "mid N"},
        {4096, 6144, "large MLP"},
    };
    int ntests = sizeof(tests) / sizeof(tests[0]);

    int warmup = 5, bench = 50;

    printf("| %-20s | %8s | %8s | %8s | %8s | %7s | %7s | %7s |\n",
           "Test", "K", "N", "ms", "GB/s", "%peak", "W_read", "W_size");
    printf("|%s|\n", "------------------------------------------------------------------------------------------------------------------");

    for (int t = 0; t < ntests; ++t) {
        int K = tests[t].K, N = tests[t].N;
        int num_K_blks = (K + B - 1) / B;
        int num_N_blks = (N + B - 1) / B;

        // Allocate
        void* d_x_fp4; float *d_x_scale, *d_y;
        void* d_W_fp4; float* d_W_scale;
        cudaMalloc(&d_x_fp4, K);       // 1 byte per FP4
        cudaMalloc(&d_x_scale, num_K_blks * 4);
        cudaMalloc(&d_W_fp4, (size_t)K * N);
        cudaMalloc(&d_W_scale, (size_t)num_K_blks * num_N_blks * 4);
        cudaMalloc(&d_y, N * 4);

        // Init with uniform data
        std::vector<float> scales_k(num_K_blks, s13);
        cudaMemcpy(d_x_scale, scales_k.data(), num_K_blks * 4, cudaMemcpyHostToDevice);
        std::vector<float> scales_all(num_K_blks * num_N_blks, s13);
        cudaMemcpy(d_W_scale, scales_all.data(), num_K_blks * num_N_blks * 4, cudaMemcpyHostToDevice);

        // Pack x and W (all 1.0)
        {
            float* d_tmp;
            cudaMalloc(&d_tmp, K * 4);
            std::vector<float> ones(K, 1.0f);
            cudaMemcpy(d_tmp, ones.data(), K * 4, cudaMemcpyHostToDevice);
            blackwell::kernels::pack_fp4(d_x_fp4, d_tmp, d_x_scale, K, 0);
            cudaFree(d_tmp);
        }
        {
            float* d_tmp;
            cudaMalloc(&d_tmp, (size_t)K * N * 4);
            std::vector<float> ones(K * N, 1.0f);
            cudaMemcpy(d_tmp, ones.data(), K * N * 4, cudaMemcpyHostToDevice);
            blackwell::kernels::pack_fp4(d_W_fp4, d_tmp, d_W_scale, K * N, 0);
            cudaFree(d_tmp);
        }

        // Warmup
        for (int i = 0; i < warmup; ++i) {
            blackwell::kernels::gemv_fp4(d_y, d_x_fp4, d_x_scale, d_W_fp4, d_W_scale, K, N, 0);
        }
        cudaDeviceSynchronize();

        // Benchmark
        Timer timer;
        timer.start();
        for (int i = 0; i < bench; ++i) {
            blackwell::kernels::gemv_fp4(d_y, d_x_fp4, d_x_scale, d_W_fp4, d_W_scale, K, N, 0);
        }
        float ms = timer.stop() / bench;

        // Calculate bandwidth
        // GEMV reads: x (K bytes) + x_scale (K/16 * 4) + W (K*N bytes) + W_scale (K/16 * N/16 * 4) + writes y (N * 4)
        size_t x_bytes = K;                          // FP4 input
        size_t xs_bytes = num_K_blks * 4;            // input scales
        size_t w_bytes = (size_t)K * N;              // FP4 weights
        size_t ws_bytes = (size_t)num_K_blks * num_N_blks * 4; // weight scales
        size_t y_bytes = N * 4;                      // FP32 output
        size_t total_bytes = x_bytes + xs_bytes + w_bytes + ws_bytes + y_bytes;
        double bw = (double)total_bytes / 1e9 / (ms / 1e3);
        double pct_peak = bw / peak_bw * 100.0;
        double w_gb = (double)w_bytes / 1e9;

        printf("| %-20s | %8d | %8d | %8.3f | %8.1f | %6.1f%% | %6.1f | %6zu |\n",
               tests[t].label, K, N, ms, bw, pct_peak, w_gb, total_bytes);

        cudaFree(d_x_fp4); cudaFree(d_x_scale);
        cudaFree(d_W_fp4); cudaFree(d_W_scale);
        cudaFree(d_y);
    }

    printf("\n# Notes:\n");
    printf("# Peak BW ~500 GB/s (GDDR7, RTX 5060 Ti 16GB)\n");
    printf("# GEMV is memory-bound: each output reads K weight elements (no reuse)\n");
    printf("# W_read = weight matrix size in GB (dominant data)\n");
    printf("# %%peak = achieved %% of peak memory bandwidth\n");

    return 0;
}
