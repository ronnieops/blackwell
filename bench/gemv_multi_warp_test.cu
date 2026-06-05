// bench/gemv_multi_warp_test.cu — Compare single-warp vs multi-warp GEMV
// Compile: nvcc -O3 -std=c++17 -arch=sm_120a -I include \
//   bench/gemv_multi_warp_test.cu build/libblackwell_kernels.a \
//   -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lcudart -lcublas -o bench/gemv_multi_warp_test

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "blackwell/kernels.h"

using namespace blackwell::kernels;

#define CHECK(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// Qwen3-1.7B: H=2048, Q=H, I=H
const int H = 2048;
const int Q = 2048;

struct Timer {
    cudaEvent_t start, stop;
    Timer() { cudaEventCreate(&start); cudaEventCreate(&stop); }
    float elapsed_ms() {
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop); return ms;
    }
};

int main() {
    CHECK(cudaSetDevice(0));
    cudaStream_t st;
    CHECK(cudaStreamCreate(&st));

    // Allocate
    void* d_x8; CHECK(cudaMalloc(&d_x8, H * sizeof(int8_t)));
    void* d_xsc; CHECK(cudaMalloc(&d_xsc, (H/16) * sizeof(float)));
    void* d_wq; CHECK(cudaMalloc(&d_wq, Q * H * sizeof(int8_t)));
    void* d_wq_sc; CHECK(cudaMalloc(&d_wq_sc, (Q/16) * (H/16) * sizeof(float)));
    float* d_y1; CHECK(cudaMalloc(&d_y1, Q * sizeof(float)));
    float* d_y2; CHECK(cudaMalloc(&d_y2, Q * sizeof(float)));
    float* h_y1 = (float*)malloc(Q * sizeof(float));
    float* h_y2 = (float*)malloc(Q * sizeof(float));

    // Random init
    std::vector<int8_t> h_x8(H);
    std::vector<float> h_xsc(H/16);
    std::vector<int8_t> h_wq(Q * H);
    std::vector<float> h_wq_sc((Q/16)*(H/16));
    for (int i = 0; i < H; ++i) h_x8[i] = (rand() % 256) - 128;
    for (int i = 0; i < H/16; ++i) h_xsc[i] = 0.1f + (rand() % 100) / 1000.0f;
    for (int i = 0; i < Q * H; ++i) h_wq[i] = (rand() % 256) - 128;
    for (int i = 0; i < (Q/16)*(H/16); ++i) h_wq_sc[i] = 0.1f + (rand() % 100) / 1000.0f;
    CHECK(cudaMemcpy(d_x8, h_x8.data(), H * sizeof(int8_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_xsc, h_xsc.data(), (H/16) * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_wq, h_wq.data(), Q * H * sizeof(int8_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_wq_sc, h_wq_sc.data(), (Q/16)*(H/16) * sizeof(float), cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < 5; ++i) {
        gemv_int8_warp(d_y1, d_x8, h_xsc.data(), d_wq, h_wq_sc.data(), H, Q, st);
        gemv_int8_multi_warp(d_y2, d_x8, h_xsc.data(), d_wq, h_wq_sc.data(), H, Q, st);
    }
    CHECK(cudaStreamSynchronize(st));

    const int ITERS = 100;
    Timer t;

    // Single-warp
    cudaEventRecord(t.start, st);
    for (int i = 0; i < ITERS; ++i) gemv_int8_warp(d_y1, d_x8, h_xsc.data(), d_wq, h_wq_sc.data(), H, Q, st);
    cudaEventRecord(t.stop, st);
    float ms1 = t.elapsed_ms() / ITERS;

    // Multi-warp
    cudaEventRecord(t.start, st);
    for (int i = 0; i < ITERS; ++i) gemv_int8_multi_warp(d_y2, d_x8, h_xsc.data(), d_wq, h_wq_sc.data(), H, Q, st);
    cudaEventRecord(t.stop, st);
    float ms2 = t.elapsed_ms() / ITERS;

    // Verify
    CHECK(cudaMemcpy(h_y1, d_y1, Q * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_y2, d_y2, Q * sizeof(float), cudaMemcpyDeviceToHost));
    float max_diff = 0.0f;
    for (int i = 0; i < Q; ++i) {
        float d = fabsf(h_y1[i] - h_y2[i]);
        if (d > max_diff) max_diff = d;
    }

    double ops = (double)Q * H * 2;
    printf("=== GEMV Comparison (Qwen3-1.7B, H=%d, K=%d) ===\n", H, H);
    printf("  gemv_int8_warp (32 threads):    %.3f ms  →  %.1f t/s\n", ms1, ops / (ms1 * 1e6));
    printf("  gemv_int8_multi_warp (128 threads): %.3f ms  →  %.1f t/s\n", ms2, ops / (ms2 * 1e6));
    printf("  Speedup: %.2fx\n", ms1 / ms2);
    printf("  Max diff: %.6f %s\n", max_diff, max_diff < 1e-4 ? "✅" : "⚠️");

    CHECK(cudaFree(d_x8)); CHECK(cudaFree(d_xsc));
    CHECK(cudaFree(d_wq)); CHECK(cudaFree(d_wq_sc));
    CHECK(cudaFree(d_y1)); CHECK(cudaFree(d_y2));
    free(h_y1); free(h_y2);
    return 0;
}