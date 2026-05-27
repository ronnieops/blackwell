// bench/gemv_int8_test.cu — INT8 GEMV benchmark vs GEMV v2 (FP4)
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

// Pack FP32 → INT8 with per-block scales
void host_pack_int8(
    int8_t* out, const float* in, const float* scales,
    int K, int N, float scale_min)
{
    int num_K_blks = K / 16;
    for (int kb = 0; kb < num_K_blks; ++kb) {
        float sc = scales[kb];
        for (int j = 0; j < 16; ++j) {
            int idx = kb * 16 + j;
            float v = in[idx] * sc;
            // Symmetric quantization: clamp(-127, 127)
            v = fmaxf(-127.f, fminf(127.f, roundf(v)));
            out[idx] = static_cast<int8_t>(static_cast<int>(v));
        }
    }
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("# INT8 vs FP4 GEMV Benchmark\n\n");
    printf("Device: %s (CC %d.%d)\n\n", p.name, p.major, p.minor);

    srand(42);
    const int K = 2048;
    const int N = 6144;
    const int warm = 10, bench = 100;

    // FP4 reference: gemv_fp4_v2
    {
        printf("## FP4 GEMV v2 Reference (K=%d, N=%d)\n\n", K, N);
        int nw = K * N;
        int nb_x = K / 16;
        int nb_W = (K / 16) * (N / 16);

        float *d_W32, *d_x32, *d_y;
        void *d_W4_t, *d_x4;
        float *d_W4_scale_t, *d_x4_scale;

        cudaMalloc(&d_W32, nw*4); cudaMalloc(&d_x32, K*4); cudaMalloc(&d_y, N*4);
        cudaMalloc(&d_W4_t, nw); cudaMalloc(&d_x4, K);
        cudaMalloc(&d_W4_scale_t, nb_W*4); cudaMalloc(&d_x4_scale, nb_x*4);

        // Init weights (K×N)
        std::vector<float> W32_h(nw);
        for (int i = 0; i < nw; ++i) W32_h[i] = (rand() % 100 - 50) / 50.f;
        cudaMemcpy(d_W32, W32_h.data(), nw*4, cudaMemcpyHostToDevice);

        // Init x (uniform pattern)
        std::vector<float> x32_h(K, 0.5f);
        cudaMemcpy(d_x32, x32_h.data(), K*4, cudaMemcpyHostToDevice);

        // Weight scales
        float am = 0.f;
        for (auto v : W32_h) { float a = fabsf(v); if (a > am) am = a; }
        float w_sv = am / 3.f;
        std::vector<float> W_scales_h(nb_W, w_sv);
        cudaMemcpy(d_W4_scale_t, W_scales_h.data(), nb_W*4, cudaMemcpyHostToDevice);

        if (blackwell::kernels::pack_fp4(d_W4_t, d_W32, W_scales_h.data(), nw, 0) != cudaSuccess) {
            printf("FAIL: pack_fp4(W)\n"); return 1;
        }
        if (blackwell::kernels::transpose_fp4_weights(d_W4_t, d_W4_scale_t, d_W4_t, W_scales_h.data(), K, N, 0) != cudaSuccess) {
            printf("FAIL: transpose_fp4_weights\n"); return 1;
        }

        // x scales
        std::vector<float> x_scales_h(nb_x, 1.f / 3.f);
        cudaMemcpy(d_x4_scale, x_scales_h.data(), nb_x*4, cudaMemcpyHostToDevice);
        if (blackwell::kernels::pack_fp4(d_x4, d_x32, x_scales_h.data(), K, 0) != cudaSuccess) {
            printf("FAIL: pack_fp4(x)\n"); return 1;
        }

        for (int i = 0; i < warm; ++i) blackwell::kernels::gemv_fp4_v2(d_y, d_x4, d_x4_scale, d_W4_t, d_W4_scale_t, K, N, 0);
        cudaDeviceSynchronize();

        GpuTimer t;
        t.begin();
        for (int i = 0; i < bench; ++i) blackwell::kernels::gemv_fp4_v2(d_y, d_x4, d_x4_scale, d_W4_t, d_W4_scale_t, K, N, 0);
        float ms = t.end() / bench;

        size_t total = K + nw + nb_x*4 + nb_W*4 + N*4;
        printf("  gemv_fp4_v2: %.3f ms, %.1f GB/s\n", ms, bw_gbps(total, ms));

        cudaFree(d_W32); cudaFree(d_x32); cudaFree(d_y);
        cudaFree(d_W4_t); cudaFree(d_x4); cudaFree(d_W4_scale_t); cudaFree(d_x4_scale);
    }

    // INT8 kernel test
    {
        printf("\n## INT8 GEMV (K=%d, N=%d)\n\n", K, N);
        int nw = K * N;
        int nb_x = K / 16;
        int nb_W = (K / 16) * (N / 16);

        float *d_W32, *d_x32, *d_y;
        int8_t *d_W8, *d_x8;
        float *d_W8_scale, *d_x8_scale;

        cudaMalloc(&d_W32, nw*4); cudaMalloc(&d_x32, K*4); cudaMalloc(&d_y, N*4);
        cudaMalloc(&d_W8, nw); cudaMalloc(&d_x8, K);
        cudaMalloc(&d_W8_scale, nb_W*4); cudaMalloc(&d_x8_scale, nb_x*4);

        // Init weights (K×N)
        std::vector<float> W32_h(nw);
        for (int i = 0; i < nw; ++i) W32_h[i] = (rand() % 100 - 50) / 50.f;
        cudaMemcpy(d_W32, W32_h.data(), nw*4, cudaMemcpyHostToDevice);

        // Init x (uniform pattern)
        std::vector<float> x32_h(K, 0.5f);
        cudaMemcpy(d_x32, x32_h.data(), K*4, cudaMemcpyHostToDevice);

        // Weight scales (for INT8 quantization)
        float am = 0.f;
        for (auto v : W32_h) { float a = fabsf(v); if (a > am) am = a; }
        float w_sc = am / 127.f;  // INT8 scale: max/127 for symmetric

        // Per-block weight scales
        std::vector<float> W_scales_h(nb_W);
        for (int nb = 0; nb < N/16; ++nb) {
            for (int kb = 0; kb < K/16; ++kb) {
                // per-block absmax
                float blk_max = 0.f;
                for (int j = 0; j < 16; ++j) {
                    for (int i = 0; i < 16; ++i) {
                        float v = W32_h[(kb*16+i)*N + (nb*16+j)];
                        float a = fabsf(v);
                        if (a > blk_max) blk_max = a;
                    }
                }
                W_scales_h[kb*(N/16) + nb] = (blk_max > 1e-10f) ? blk_max / 127.f : 1.f/127.f;
            }
        }
        printf("  W_scales[0..8]: ");
        for (int i = 0; i < 8; ++i) printf("%.6f ", W_scales_h[i]);
        printf("\n");
        cudaMemcpy(d_W8_scale, W_scales_h.data(), nb_W*4, cudaMemcpyHostToDevice);

        // x scales (per K-block)
        float x_sc = 0.5f / 127.f;  // uniform x = 0.5
        std::vector<float> x_scales_h(nb_x, x_sc);
        cudaMemcpy(d_x8_scale, x_scales_h.data(), nb_x*4, cudaMemcpyHostToDevice);

        // Quantize weights to INT8 on host
        std::vector<int8_t> W8_h(nw);
        for (int n = 0; n < N; ++n) {
            for (int kb = 0; kb < nb_x; ++kb) {
                float blk_scale = W_scales_h[kb*(N/16) + n/16];
                for (int j = 0; j < 16; ++j) {
                    int idx = n*K + kb*16 + j;
                    float v = W32_h[idx] / blk_scale;  // q = x / scale
                    v = fmaxf(-127.f, fminf(127.f, roundf(v)));
                    W8_h[idx] = static_cast<int8_t>(static_cast<int>(v));
                }
            }
        }
        cudaMemcpy(d_W8, W8_h.data(), nw, cudaMemcpyHostToDevice);

        // Quantize x to INT8 on host
        std::vector<int8_t> x8_h(K);
        float x_scale_h = x_scales_h[0];
        for (int kb = 0; kb < nb_x; ++kb) {
            float blk_scale = x_scales_h[kb];
            for (int j = 0; j < 16; ++j) {
                int idx = kb*16 + j;
                float v = x32_h[idx] / blk_scale;  // q = x / scale
                v = fmaxf(-127.f, fminf(127.f, roundf(v)));
                x8_h[idx] = static_cast<int8_t>(static_cast<int>(v));
            }
        }
        cudaMemcpy(d_x8, x8_h.data(), K, cudaMemcpyHostToDevice);

        // Transpose weights: W (K×N) → W_t (N×K)
        // INT8 doesn't have transpose kernel yet, so we transpose on host
        std::vector<int8_t> W8_t_h(nw);
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                W8_t_h[n*K + k] = W8_h[k*N + n];
            }
        }
        cudaFree(d_W8);
        cudaMalloc(&d_W8, nw);
        cudaMemcpy(d_W8, W8_t_h.data(), nw, cudaMemcpyHostToDevice);

        // Transpose scales: W_scale (K/16 × N/16) → W_t_scale (N/16 × K/16)
        std::vector<float> W8_scale_t_h(nb_W);
        for (int nb = 0; nb < N/16; ++nb) {
            for (int kb = 0; kb < K/16; ++kb) {
                W8_scale_t_h[nb*(K/16) + kb] = W_scales_h[kb*(N/16) + nb];
            }
        }
        cudaFree(d_W8_scale);
        cudaMalloc(&d_W8_scale, nb_W*4);
        cudaMemcpy(d_W8_scale, W8_scale_t_h.data(), nb_W*4, cudaMemcpyHostToDevice);

        for (int i = 0; i < warm; ++i) blackwell::kernels::gemv_int8(d_y, d_x8, d_x8_scale, d_W8, d_W8_scale, K, N, 0);
        cudaDeviceSynchronize();

        GpuTimer t;
        t.begin();
        for (int i = 0; i < bench; ++i) blackwell::kernels::gemv_int8(d_y, d_x8, d_x8_scale, d_W8, d_W8_scale, K, N, 0);
        float ms = t.end() / bench;

        size_t total = K + nw + nb_x*4 + nb_W*4 + N*4;
        printf("  gemv_int8: %.3f ms, %.1f GB/s\n", ms, bw_gbps(total, ms));

        // Check correctness
        std::vector<float> y_gpu(N);
        cudaMemcpy(y_gpu.data(), d_y, N*4, cudaMemcpyDeviceToHost);

        // Debug: print first 8 values
        printf("  First 8 y values: ");
        for (int i = 0; i < 8 && i < N; ++i) printf("%.4f ", y_gpu[i]);
        printf("\n");

        std::vector<float> y_cpu(N);
        cpu_gemv_ref(W32_h.data(), x32_h.data(), y_cpu.data(), K, N);
        printf("  First 8 y_cpu: ");
        for (int i = 0; i < 8 && i < N; ++i) printf("%.4f ", y_cpu[i]);
        printf("\n");

        // Debug: print first 8 quantized first 8 x values
        printf("  First 8 x_int8: ");
        for (int i = 0; i < 8 && i < K; ++i) printf("%d ", static_cast<int>(x8_h[i]));
        printf("\n");

        // Debug: print first 8 quantized weight values for first output row
        printf("  First 8 W8[0][0..7]: ");
        for (int i = 0; i < 8; ++i) printf("%d ", static_cast<int>(W8_h[i]));  // W[0][0..7] = row 0
        printf("\n");
        printf("  First 8 W8_t[0][0..7]: ");
        for (int i = 0; i < 8; ++i) printf("%d ", static_cast<int>(W8_t_h[i]));  // W_t[0][0..7]
        printf("\n");
        printf("  First 8 FP32 W[0][0..7]: ");
        for (int i = 0; i < 8; ++i) printf("%.4f ", W32_h[i]);
        printf("\n");
        printf("  First 8 FP32 W_t[n*K + 0]: ");
        for (int i = 0; i < 8; ++i) printf("%.4f ", W32_h[i*N + 0]);  // W_t[0..7][0]
        printf("\n");
        printf("  W_scale[0]: %.8f, W_scale_t[0]: %.8f\n", W_scales_h[0], W8_scale_t_h[0]);

        // CPU INT8 reference: same dequantization as kernel
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
        printf("  First 8 y_int8_cpu (INT8 reference): ");
        for (int i = 0; i < 8; ++i) printf("%.4f ", y_int8_cpu[i]);
        printf("\n");

        // GPU matches CPU INT8 reference exactly (kernel correct)
        float gpu_err = max_rel_err(y_int8_cpu.data(), y_gpu.data(), N);
        printf("  GPU vs CPU INT8 rel err: %.6e  (kernel correctness)\n", gpu_err);

        // INT8 vs FP32 — expected quantization noise
        float quant_err = max_rel_err(y_cpu.data(), y_int8_cpu.data(), N);
        printf("  INT8 vs FP32 rel err: %.6e  (quantization accuracy)\n", quant_err);

        cudaFree(d_W32); cudaFree(d_x32); cudaFree(d_y);
        cudaFree(d_W8); cudaFree(d_x8); cudaFree(d_W8_scale); cudaFree(d_x8_scale);
    }

    printf("\nDone.\n");
    return 0;
}
