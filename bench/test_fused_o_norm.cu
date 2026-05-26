// bench/test_fused_o_norm.cu — Test fused O-proj + RMSNorm + FP4 pack
//
// Verify correctness against separate gemv_fp4 → fused_rmsnorm → pack_fp4
// Then benchmark both paths and compare.
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120,code=sm_120 \
//     -I include bench/test_fused_o_norm.cu build/libblackwell_kernels.a \
//     -o bench/test_fused_o_norm

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

static bool check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        printf("FAIL: %s: %s\n", msg, cudaGetErrorString(e));
        return false;
    }
    return true;
}

float max_rel_err(const float* a, const float* b, int n, float eps=1e-3f) {
    float me = 0.f;
    for (int i = 0; i < n; ++i) {
        float d = fabsf(a[i]) > eps ? fabsf(a[i]) : eps;
        float e = fabsf(a[i]-b[i]) / d;
        if (e > me) me = e;
    }
    return me;
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("# Fused O-proj + RMSNorm + Pack Test\n");
    printf("Device: %s (CC %d.%d)\n\n", p.name, p.major, p.minor);

    constexpr int K = 2048;   // q_dim
    constexpr int N = 2048;   // hidden_dim
    constexpr int B = 16;     // FP4 block size
    constexpr float scale_1_3 = 1.0f / 3.0f;
    constexpr int num_N_blks = (N + B - 1) / B;
    constexpr int num_K_blks = (K + B - 1) / B;

    // Allocate weights (all ones)
    void* d_W_fp4;
    float* d_W_scale;
    cudaMalloc(&d_W_fp4, K * N);
    cudaMalloc(&d_W_scale, num_K_blks * num_N_blks * 4);
    std::vector<float> wscales(num_K_blks * num_N_blks, scale_1_3);
    cudaMemcpy(d_W_scale, wscales.data(), wscales.size() * 4, cudaMemcpyHostToDevice);
    {
        std::vector<float> tmp(K * N, 1.0f);
        float* d_tmp;
        cudaMalloc(&d_tmp, K * N * 4);
        cudaMemcpy(d_tmp, tmp.data(), K * N * 4, cudaMemcpyHostToDevice);
        blackwell::kernels::pack_fp4(d_W_fp4, d_tmp, d_W_scale, K * N, 0);
        cudaFree(d_tmp);
    }

    // Allocate input (all ones)
    float* d_attn_fp32;
    void* d_attn_fp4;
    float* d_attn_scale;
    cudaMalloc(&d_attn_fp32, K * 4);
    cudaMalloc(&d_attn_fp4, K);
    cudaMalloc(&d_attn_scale, num_K_blks * 4);
    {
        std::vector<float> a(K, 1.0f);
        cudaMemcpy(d_attn_fp32, a.data(), K * 4, cudaMemcpyHostToDevice);
        std::vector<float> as(num_K_blks, scale_1_3);
        cudaMemcpy(d_attn_scale, as.data(), num_K_blks * 4, cudaMemcpyHostToDevice);
        blackwell::kernels::pack_fp4(d_attn_fp4, d_attn_fp32, d_attn_scale, K, 0);
    }

    // RMSNorm weight (all ones)
    float* d_rn_weight;
    cudaMalloc(&d_rn_weight, N * 4);
    {
        std::vector<float> rn(N, 1.0f);
        cudaMemcpy(d_rn_weight, rn.data(), N * 4, cudaMemcpyHostToDevice);
    }

    // ---- Baseline: separate kernels ----
    float *d_proj_baseline, *d_norm_baseline;
    void *d_x_fp4_baseline;
    float *d_x_scale_baseline;
    cudaMalloc(&d_proj_baseline, N * 4);
    cudaMalloc(&d_norm_baseline, N * 4);
    cudaMalloc(&d_x_fp4_baseline, N);
    cudaMalloc(&d_x_scale_baseline, num_N_blks * 4);
    {
        std::vector<float> xs(num_N_blks, scale_1_3);
        cudaMemcpy(d_x_scale_baseline, xs.data(), num_N_blks * 4, cudaMemcpyHostToDevice);
    }

    check(blackwell::kernels::gemv_fp4(d_proj_baseline, d_attn_fp4, d_attn_scale,
        d_W_fp4, d_W_scale, K, N, 0), "baseline gemv");
    check(blackwell::kernels::fused_rmsnorm(d_norm_baseline, d_proj_baseline,
        d_rn_weight, N, 1e-5f, 0), "baseline rmsnorm");
    check(blackwell::kernels::pack_fp4(d_x_fp4_baseline, d_norm_baseline,
        d_x_scale_baseline, N, 0), "baseline pack");

    // Read baseline results
    std::vector<float> proj_baseline(N), norm_baseline(N);
    cudaMemcpy(proj_baseline.data(), d_proj_baseline, N * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(norm_baseline.data(), d_norm_baseline, N * 4, cudaMemcpyDeviceToHost);
    printf("Baseline proj[0..3]: ");
    for (int i = 0; i < 4; ++i) printf("%.4f ", proj_baseline[i]);
    printf("\n");
    printf("Baseline norm[0..3]: ");
    for (int i = 0; i < 4; ++i) printf("%.4f ", norm_baseline[i]);
    printf("\n");

    // ---- Fused kernel (2-kernel: gemv + rmsnorm_pack) ----
    float *d_proj_fused;
    void *d_x_fp4_fused;
    float *d_x_scale_fused;
    cudaMalloc(&d_proj_fused, N * 4);
    cudaMalloc(&d_x_fp4_fused, N);
    cudaMalloc(&d_x_scale_fused, num_N_blks * 4);

    check(blackwell::kernels::gemv_fp4(d_proj_fused, d_attn_fp4, d_attn_scale,
        d_W_fp4, d_W_scale, K, N, 0), "fused gemv");
    check(blackwell::kernels::fused_rmsnorm_pack(
        d_x_fp4_fused, d_x_scale_fused,
        d_proj_fused, d_rn_weight, N, 1e-5f, 0), "fused rmsnorm_pack");

    // Unpack fused output for comparison
    float* d_fused_unpacked;
    cudaMalloc(&d_fused_unpacked, N * 4);
    check(blackwell::kernels::unpack_fp4(d_fused_unpacked, d_x_fp4_fused,
        d_x_scale_fused, N, 0), "unpack fused");

    std::vector<float> fused_out(N);
    cudaMemcpy(fused_out.data(), d_fused_unpacked, N * 4, cudaMemcpyDeviceToHost);

    printf("Fused   norm[0..3]: ");
    for (int i = 0; i < 4; ++i) printf("%.4f ", fused_out[i]);
    printf("\n");

    float err = max_rel_err(norm_baseline.data(), fused_out.data(), N);
    printf("\nMax relative error: %.4f\n", err);
    if (err < 0.15f) {
        printf("PASS: fused output matches baseline within FP4 quantization tolerance\n");
    } else {
        printf("FAIL: fused output differs significantly from baseline\n");
    }

    // ---- Benchmark ----
    int warmup = 5, bench = 100;
    printf("\nBenchmarking %d iterations...\n", bench);

    // Baseline: 3 kernels
    GpuTimer t1;
    for (int i = 0; i < warmup; ++i) {
        blackwell::kernels::gemv_fp4(d_proj_baseline, d_attn_fp4, d_attn_scale,
            d_W_fp4, d_W_scale, K, N, 0);
        blackwell::kernels::fused_rmsnorm(d_norm_baseline, d_proj_baseline,
            d_rn_weight, N, 1e-5f, 0);
        blackwell::kernels::pack_fp4(d_x_fp4_baseline, d_norm_baseline,
            d_x_scale_baseline, N, 0);
    }
    cudaDeviceSynchronize();
    t1.begin();
    for (int i = 0; i < bench; ++i) {
        blackwell::kernels::gemv_fp4(d_proj_baseline, d_attn_fp4, d_attn_scale,
            d_W_fp4, d_W_scale, K, N, 0);
        blackwell::kernels::fused_rmsnorm(d_norm_baseline, d_proj_baseline,
            d_rn_weight, N, 1e-5f, 0);
        blackwell::kernels::pack_fp4(d_x_fp4_baseline, d_norm_baseline,
            d_x_scale_baseline, N, 0);
    }
    float baseline_ms = t1.end() / bench;

    // Fused: 2 kernels (gemv + rmsnorm_pack)
    for (int i = 0; i < warmup; ++i) {
        blackwell::kernels::gemv_fp4(d_proj_fused, d_attn_fp4, d_attn_scale,
            d_W_fp4, d_W_scale, K, N, 0);
        blackwell::kernels::fused_rmsnorm_pack(
            d_x_fp4_fused, d_x_scale_fused,
            d_proj_fused, d_rn_weight, N, 1e-5f, 0);
    }
    cudaDeviceSynchronize();
    GpuTimer t2;
    t2.begin();
    for (int i = 0; i < bench; ++i) {
        blackwell::kernels::gemv_fp4(d_proj_fused, d_attn_fp4, d_attn_scale,
            d_W_fp4, d_W_scale, K, N, 0);
        blackwell::kernels::fused_rmsnorm_pack(
            d_x_fp4_fused, d_x_scale_fused,
            d_proj_fused, d_rn_weight, N, 1e-5f, 0);
    }
    float fused_ms = t2.end() / bench;

    printf("\n=== Benchmark Results (K=%d, N=%d) ===\n", K, N);
    printf("  Baseline (3 kernels): %.3f ms/call\n", baseline_ms);
    printf("  Fused    (2 kernels): %.3f ms/call\n", fused_ms);
    printf("  Speedup:              %.2fx\n", baseline_ms / fused_ms);

    // Cleanup
    cudaFree(d_W_fp4); cudaFree(d_W_scale);
    cudaFree(d_attn_fp32); cudaFree(d_attn_fp4); cudaFree(d_attn_scale);
    cudaFree(d_rn_weight);
    cudaFree(d_proj_baseline); cudaFree(d_norm_baseline);
    cudaFree(d_x_fp4_baseline); cudaFree(d_x_scale_baseline);
    cudaFree(d_proj_fused);
    cudaFree(d_x_fp4_fused); cudaFree(d_x_scale_fused);
    cudaFree(d_fused_unpacked);

    return 0;
}
