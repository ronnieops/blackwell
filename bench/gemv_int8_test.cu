// bench/gemv_int8_test.cu — INT8 GEMV benchmark using GPU pack + transpose
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120,code=sm_120 \
//     -I include bench/gemv_int8_test.cu build/libblackwell_kernels.a -o bench/gemv_int8_test

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "blackwell/kernels.h"

struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin() { cudaEventRecord(start, 0); }
    float end() { cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
                  float ms=0; cudaEventElapsedTime(&ms, start, stop); return ms; }
};

void cpu_gemv_ref(const float* W, const float* x, float* y, int K, int N) {
    for (int n = 0; n < N; ++n) {
        float s = 0.f;
        for (int k = 0; k < K; ++k) s += W[n*K + k] * x[k];
        y[n] = s;
    }
}

float max_rel_err(const float* a, const float* b, int n, float eps=1e-5f) {
    float me = 0.f;
    for (int i = 0; i < n; ++i) {
        float d = fabsf(a[i]) > eps ? fabsf(a[i]) : eps;
        float e = fabsf(a[i]-b[i]) / d;
        if (e > me) me = e;
    }
    return me;
}

double bw_gbps(size_t bytes, float ms) {
    return (static_cast<double>(bytes) / 1e9) / (ms / 1e3);
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("# INT8 GEMV — GPU pack + transpose verification\n\n");
    printf("Device: %s (CC %d.%d)\n\n", p.name, p.major, p.minor);

    srand(42);
    const int K = 2048;
    const int N = 6144;
    const int warm = 10, bench = 100;
    int nw = K * N;
    int nb_x = K / 16;
    int nb_W = (K / 16) * (N / 16);

    // Device memory
    float *d_W32, *d_x32, *d_y;
    int8_t *d_W8_orig, *d_W8_t, *d_x8;
    float *d_W8_scale_orig, *d_W8_scale_t, *d_x8_scale;

    cudaMalloc(&d_W32, nw*4); cudaMalloc(&d_x32, K*4); cudaMalloc(&d_y, N*4);
    cudaMalloc(&d_W8_orig, nw); cudaMalloc(&d_W8_t, nw); cudaMalloc(&d_x8, K);
    cudaMalloc(&d_W8_scale_orig, nb_W*4); cudaMalloc(&d_W8_scale_t, nb_W*4);
    cudaMalloc(&d_x8_scale, nb_x*4);

    // Init weights (K×N) — random [-1..1]
    std::vector<float> W32_h(nw);
    for (int i = 0; i < nw; ++i) W32_h[i] = (rand() % 100 - 50) / 50.f;
    cudaMemcpy(d_W32, W32_h.data(), nw*4, cudaMemcpyHostToDevice);

    // Init x (uniform 0.5)
    std::vector<float> x32_h(K, 0.5f);
    cudaMemcpy(d_x32, x32_h.data(), K*4, cudaMemcpyHostToDevice);

    // Compute per-block weight scales (on host, then upload)
    std::vector<float> W_scales_h(nb_W);
    for (int nb = 0; nb < N/16; ++nb) {
        for (int kb = 0; kb < K/16; ++kb) {
            float blk_max = 0.f;
            for (int j = 0; j < 16; ++j)
                for (int i = 0; i < 16; ++i) {
                    float v = fabsf(W32_h[(kb*16+i)*N + (nb*16+j)]);
                    if (v > blk_max) blk_max = v;
                }
            W_scales_h[kb*(N/16) + nb] = (blk_max > 1e-10f) ? blk_max / 127.f : 1.f/127.f;
        }
    }
    cudaMemcpy(d_W8_scale_orig, W_scales_h.data(), nb_W*4, cudaMemcpyHostToDevice);

    // x scales — uniform
    float x_scale_val = 0.5f / 127.f;
    std::vector<float> x_scales_h(nb_x, x_scale_val);
    cudaMemcpy(d_x8_scale, x_scales_h.data(), nb_x*4, cudaMemcpyHostToDevice);

    // === 1. Test GPU pack_int8 ===
    printf("## 1. GPU pack_int8 + transpose_int8_weights\n\n");
    printf("  Packing %d elements (K=%d, N=%d)...\n", nw, K, N);

    cudaError_t e = blackwell::kernels::pack_int8(d_W8_orig, d_W32, d_W8_scale_orig, nw, 0);
    if (e != cudaSuccess) { printf("FAIL: pack_int8: %s\n", cudaGetErrorString(e)); return 1; }
    printf("  pack_int8: OK\n");

    // Also pack x
    e = blackwell::kernels::pack_int8(d_x8, d_x32, d_x8_scale, K, 0);
    if (e != cudaSuccess) { printf("FAIL: pack_int8(x): %s\n", cudaGetErrorString(e)); return 1; }
    printf("  pack_int8(x): OK\n");

    // === 2. Test GPU transpose_int8_weights ===
    e = blackwell::kernels::transpose_int8_weights(d_W8_t, d_W8_scale_t, d_W8_orig, d_W8_scale_orig, K, N, 0);
    if (e != cudaSuccess) { printf("FAIL: transpose_int8_weights: %s\n", cudaGetErrorString(e)); return 1; }
    printf("  transpose_int8_weights: OK\n");

    // === 3. Test gemv_int8 ===
    printf("\n## 2. gemv_int8 benchmark (K=%d, N=%d)\n\n", K, N);

    for (int i = 0; i < warm; ++i)
        blackwell::kernels::gemv_int8_warp(d_y, d_x8, d_x8_scale, d_W8_t, d_W8_scale_t, K, N, 0);
    cudaDeviceSynchronize();

    GpuTimer t;
    t.begin();
    for (int i = 0; i < bench; ++i)
        blackwell::kernels::gemv_int8_warp(d_y, d_x8, d_x8_scale, d_W8_t, d_W8_scale_t, K, N, 0);
    float ms = t.end() / bench;

    size_t total = K + nw + nb_x*4 + nb_W*4 + N*4;
    double gbps = bw_gbps(total, ms);
    printf("  gemv_int8: %.3f ms, %.1f GB/s\n", ms, gbps);

    // Verify correctness
    std::vector<float> y_gpu(N);
    cudaMemcpy(y_gpu.data(), d_y, N*4, cudaMemcpyDeviceToHost);

    // CPU INT8 reference: same path
    // First download GPU-packed data to verify match
    std::vector<int8_t> W8_t_h(nw), x8_h(K);
    std::vector<float> W8_scale_t_h(nb_W);
    cudaMemcpy(W8_t_h.data(), d_W8_t, nw, cudaMemcpyDeviceToHost);
    cudaMemcpy(x8_h.data(), d_x8, K, cudaMemcpyDeviceToHost);
    cudaMemcpy(W8_scale_t_h.data(), d_W8_scale_t, nb_W*4, cudaMemcpyDeviceToHost);

    std::vector<float> y_int8_cpu(N, 0.f);
    for (int n = 0; n < N; ++n) {
        int n_blk = n / 16;
        for (int kb = 0; kb < nb_x; ++kb) {
            float w_sc = W8_scale_t_h[n_blk * nb_x + kb];
            float x_sc = x_scales_h[kb];
            for (int j = 0; j < 16; ++j) {
                y_int8_cpu[n] += static_cast<float>(W8_t_h[n*K + kb*16 + j]) * w_sc *
                                 static_cast<float>(x8_h[kb*16 + j]) * x_sc;
            }
        }
    }

    float gpu_err = max_rel_err(y_int8_cpu.data(), y_gpu.data(), N);
    printf("  GPU vs CPU INT8 rel err: %.6e  (kernel correctness)\n", gpu_err);

    // FP32 reference (for info)
    std::vector<float> y_cpu(N);
    cpu_gemv_ref(W32_h.data(), x32_h.data(), y_cpu.data(), K, N);
    printf("  First 8 y_gpu: ");
    for (int i = 0; i < 8; ++i) printf("%.4f ", y_gpu[i]);
    printf("\n");
    printf("  First 8 y_fp32: ");
    for (int i = 0; i < 8; ++i) printf("%.4f ", y_cpu[i]);
    printf("\n");

    // Result
    printf("\n## Result\n\n");
    if (gpu_err < 1e-3f) {
        printf("  ✅ PASS GPU = CPU INT8 (err=%.6e)\n", gpu_err);
        printf("  INT8 throughput: %.1f GB/s\n", gbps);
    } else {
        printf("  ❌ FAIL GPU != CPU INT8 (err=%.6e)\n", gpu_err);
    }

    cudaFree(d_W32); cudaFree(d_x32); cudaFree(d_y);
    cudaFree(d_W8_orig); cudaFree(d_W8_t); cudaFree(d_x8);
    cudaFree(d_W8_scale_orig); cudaFree(d_W8_scale_t); cudaFree(d_x8_scale);
    return 0;
}
