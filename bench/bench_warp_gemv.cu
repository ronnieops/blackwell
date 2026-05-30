// bench/bench_warp_gemv.cu — Compare gemv_int8 vs gemv_int8_warp
//
// Build:
//   nvcc -O3 -std=c++17 -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/bench_warp_gemv.cu build/libblackwell_kernels.a \
//     -o bench/bench_warp_gemv

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "blackwell/kernels.h"

struct GpuTimer {
    cudaEvent_t s, e;
    GpuTimer() { cudaEventCreate(&s); cudaEventCreate(&e); }
    ~GpuTimer() { cudaEventDestroy(s); cudaEventDestroy(e); }
    void start(cudaStream_t st=0) { cudaEventRecord(s, st); }
    float stop(cudaStream_t st=0) {
        cudaEventRecord(e, st); cudaEventSynchronize(e);
        float ms=0; cudaEventElapsedTime(&ms, s, e); return ms;
    }
};

struct LoadedW { int K, N; std::vector<int8_t> d; std::vector<float> sc; };
static LoadedW load_int8_w(const char* prefix) {
    char p[256]; snprintf(p,256,"%s.int8_t",prefix);
    FILE* f = fopen(p,"rb"); int h[5]; fread(h,4,5,f);
    LoadedW w; w.K=h[0]; w.N=h[1]; w.d.resize(h[0]*h[1]); fread(w.d.data(),1,w.d.size(),f); fclose(f);
    snprintf(p,256,"%s.scale_t",prefix); f=fopen(p,"rb"); fread(h,4,5,f);
    w.sc.resize(h[3]*h[4]); fread(w.sc.data(),4,w.sc.size(),f); fclose(f);
    return w;
}

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("# Warp GEMV Benchmark — %s (%d.%d)\n\n", p.name, p.major, p.minor);

    const float s13 = 1.f/3.f;

    // Load Q projection weights for layer 0 (K=2048, N=2048)
    auto wq = load_int8_w("weights_int8_bf16/0_self_attn.q_proj");
    printf("Q proj: K=%d N=%d\n", wq.K, wq.N);

    int8_t* d_w; float* d_ws;
    cudaMalloc(&d_w, wq.K*wq.N);
    cudaMemcpy(d_w, wq.d.data(), wq.K*wq.N, cudaMemcpyHostToDevice);
    int nsc = wq.sc.size();
    cudaMalloc(&d_ws, nsc*4);
    cudaMemcpy(d_ws, wq.sc.data(), nsc*4, cudaMemcpyHostToDevice);

    // Activation: INT8 x (K=2048)
    int K = wq.K, N = wq.N;
    std::vector<float> xh(K, 1.f);
    std::vector<float> xsh(K/16, s13);
    float *d_x32, *d_xs;
    int8_t *d_xi8;
    cudaMalloc(&d_x32, K*4);
    cudaMalloc(&d_xs, (K/16)*4);
    cudaMalloc(&d_xi8, K);
    cudaMemcpy(d_x32, xh.data(), K*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_xs, xsh.data(), (K/16)*4, cudaMemcpyHostToDevice);
    blackwell::kernels::pack_int8(d_xi8, d_x32, d_xs, K, 0);

    // Output buffers
    float *d_y_old, *d_y_warp;
    cudaMalloc(&d_y_old, N*4);
    cudaMalloc(&d_y_warp, N*4);

    // Warmup
    for (int i = 0; i < 10; ++i) {
        blackwell::kernels::gemv_int8(d_y_old, d_xi8, d_xs, d_w, d_ws, K, N, 0);
        blackwell::kernels::gemv_int8_warp(d_y_warp, d_xi8, d_xs, d_w, d_ws, K, N, 0);
    }
    cudaDeviceSynchronize();

    // Correctness check
    std::vector<float> y_old(N), y_warp(N);
    cudaMemcpy(y_old.data(), d_y_old, N*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(y_warp.data(), d_y_warp, N*4, cudaMemcpyDeviceToHost);

    float max_diff = 0, cos_sim = 0, norm_old = 0, norm_warp = 0;
    for (int i = 0; i < N; ++i) {
        float d = fabsf(y_old[i] - y_warp[i]);
        if (d > max_diff) max_diff = d;
        cos_sim += y_old[i] * y_warp[i];
        norm_old += y_old[i] * y_old[i];
        norm_warp += y_warp[i] * y_warp[i];
    }
    cos_sim /= sqrtf(norm_old) * sqrtf(norm_warp);

    printf("\n=== Correctness ===\n");
    printf("  Old[0..3]:  %.6f %.6f %.6f %.6f\n", y_old[0], y_old[1], y_old[2], y_old[3]);
    printf("  Warp[0..3]: %.6f %.6f %.6f %.6f\n", y_warp[0], y_warp[1], y_warp[2], y_warp[3]);
    printf("  Max diff: %.8f\n", max_diff);
    printf("  Cosine:   %.8f %s\n", cos_sim,
        cos_sim > 0.9999 ? "✅" : cos_sim > 0.99 ? "⚠️" : "❌");

    // Benchmark
    int iters = 1000;
    printf("\n=== Throughput (%d iters) ===\n", iters);

    // Old kernel
    GpuTimer t1;
    t1.start();
    for (int i = 0; i < iters; ++i)
        blackwell::kernels::gemv_int8(d_y_old, d_xi8, d_xs, d_w, d_ws, K, N, 0);
    float ms_old = t1.stop();

    // Warp kernel
    GpuTimer t2;
    t2.start();
    for (int i = 0; i < iters; ++i)
        blackwell::kernels::gemv_int8_warp(d_y_warp, d_xi8, d_xs, d_w, d_ws, K, N, 0);
    float ms_warp = t2.stop();

    float bw_old = (float)(2LL*K*N) * iters / ms_old / 1e6;  // GB/s
    float bw_warp = (float)(2LL*K*N) * iters / ms_warp / 1e6;

    printf("  %-15s  %7.3f ms  %7.1f GB/s\n", "gemv_int8", ms_old, bw_old);
    printf("  %-15s  %7.3f ms  %7.1f GB/s\n", "gemv_int8_warp", ms_warp, bw_warp);
    printf("  Speedup: %.2fx\n", ms_old / ms_warp);

    // Also test larger MLP dimensions
    printf("\n--- MLP gate_proj (K=2048, N=6144) ---\n");
    auto wg = load_int8_w("weights_int8_bf16/0_mlp.gate_proj");
    printf("Gate proj: K=%d N=%d\n", wg.K, wg.N);
    int K2 = wg.K, N2 = wg.N;

    int8_t* d_wg; float* d_wgs;
    cudaMalloc(&d_wg, K2*N2);
    cudaMemcpy(d_wg, wg.d.data(), K2*N2, cudaMemcpyHostToDevice);
    int nsc2 = wg.sc.size();
    cudaMalloc(&d_wgs, nsc2*4);
    cudaMemcpy(d_wgs, wg.sc.data(), nsc2*4, cudaMemcpyHostToDevice);

    float *d_y2_old, *d_y2_warp;
    cudaMalloc(&d_y2_old, N2*4);
    cudaMalloc(&d_y2_warp, N2*4);

    // Need activation for K=2048
    int8_t *d_xi8_2;
    float *d_xs_2;
    cudaMalloc(&d_xi8_2, K2);
    cudaMalloc(&d_xs_2, (K2/16)*4);
    cudaMemcpy(d_xi8_2, d_xi8, K2, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_xs_2, d_xs, (K2/16)*4, cudaMemcpyDeviceToDevice);

    // Warmup
    for (int i = 0; i < 10; ++i) {
        blackwell::kernels::gemv_int8(d_y2_old, d_xi8_2, d_xs_2, d_wg, d_wgs, K2, N2, 0);
        blackwell::kernels::gemv_int8_warp(d_y2_warp, d_xi8_2, d_xs_2, d_wg, d_wgs, K2, N2, 0);
    }
    cudaDeviceSynchronize();

    // Correctness
    std::vector<float> y2_old(N2), y2_warp(N2);
    cudaMemcpy(y2_old.data(), d_y2_old, N2*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(y2_warp.data(), d_y2_warp, N2*4, cudaMemcpyDeviceToHost);

    float max_diff2 = 0, cos2 = 0, n2o = 0, n2w = 0;
    for (int i = 0; i < N2; ++i) {
        float d = fabsf(y2_old[i] - y2_warp[i]);
        if (d > max_diff2) max_diff2 = d;
        cos2 += y2_old[i] * y2_warp[i];
        n2o += y2_old[i]*y2_old[i];
        n2w += y2_warp[i]*y2_warp[i];
    }
    cos2 /= sqrtf(n2o) * sqrtf(n2w);
    printf("  Max diff: %.8f  Cosine: %.8f %s\n", max_diff2, cos2,
        cos2 > 0.9999 ? "✅" : cos2 > 0.99 ? "⚠️" : "❌");

    GpuTimer t3, t4;
    t3.start();
    for (int i = 0; i < iters; ++i)
        blackwell::kernels::gemv_int8(d_y2_old, d_xi8_2, d_xs_2, d_wg, d_wgs, K2, N2, 0);
    float ms_old2 = t3.stop();

    t4.start();
    for (int i = 0; i < iters; ++i)
        blackwell::kernels::gemv_int8_warp(d_y2_warp, d_xi8_2, d_xs_2, d_wg, d_wgs, K2, N2, 0);
    float ms_warp2 = t4.stop();

    float bw_o2 = (float)(2LL*K2*N2) * iters / ms_old2 / 1e6;
    float bw_w2 = (float)(2LL*K2*N2) * iters / ms_warp2 / 1e6;
    printf("  %-15s  %7.3f ms  %7.1f GB/s\n", "gemv_int8", ms_old2, bw_o2);
    printf("  %-15s  %7.3f ms  %7.1f GB/s\n", "gemv_int8_warp", ms_warp2, bw_w2);
    printf("  Speedup: %.2fx\n", ms_old2 / ms_warp2);

    // Cleanup
    cudaFree(d_w); cudaFree(d_ws);
    cudaFree(d_x32); cudaFree(d_xs); cudaFree(d_xi8);
    cudaFree(d_y_old); cudaFree(d_y_warp);
    cudaFree(d_wg); cudaFree(d_wgs);
    cudaFree(d_xi8_2); cudaFree(d_xs_2);
    cudaFree(d_y2_old); cudaFree(d_y2_warp);

    return 0;
}
