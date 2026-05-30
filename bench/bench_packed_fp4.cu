// bench/bench_packed_fp4.cu — Benchmark packed FP4 warp GEMV vs INT8
//
// Build:
//   nvcc -O3 -std=c++17 -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/bench_packed_fp4.cu build/libblackwell_kernels.a \
//     -o bench/bench_packed_fp4

#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
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

// Load INT8 weights
struct LoadedW { int K, N; std::vector<int8_t> d; std::vector<float> sc; };
static LoadedW load_int8_w(const char* prefix) {
    char p[256]; snprintf(p,256,"%s.int8_t",prefix);
    FILE* f = fopen(p,"rb"); if(!f){fprintf(stderr,"FAIL open %s\n",p);exit(1);}
    int h[5]; fread(h,4,5,f);
    LoadedW w; w.K=h[0]; w.N=h[1]; w.d.resize(h[0]*h[1]); fread(w.d.data(),1,w.d.size(),f); fclose(f);
    snprintf(p,256,"%s.scale_t",prefix); f=fopen(p,"rb"); fread(h,4,5,f);
    w.sc.resize(h[3]*h[4]); fread(w.sc.data(),4,w.sc.size(),f); fclose(f);
    return w;
}

// Load packed FP4 weights
struct LoadedFP4 { int K, N; std::vector<uint8_t> packed; std::vector<float> sc; };
static LoadedFP4 load_fp4_w(const char* prefix) {
    char p[256]; snprintf(p,256,"%s.packed_fp4",prefix);
    FILE* f = fopen(p,"rb"); if(!f){fprintf(stderr,"FAIL open %s\n",p);exit(1);}
    int h[5]; fread(h,4,5,f);
    LoadedFP4 w; w.K=h[0]; w.N=h[1];
    w.packed.resize(h[4]); fread(w.packed.data(),1,w.packed.size(),f); fclose(f);
    int n_sc = h[3] * (h[0]/16);
    w.sc.resize(n_sc); 
    // scales are after the packed data in file
    // Re-read to get scales
    f = fopen(p,"rb"); fread(h,4,5,f);
    fseek(f, 20 + h[4], SEEK_SET); // skip header + packed data
    w.sc.resize(h[3] * (h[0]/16));
    fread(w.sc.data(),4,w.sc.size(),f); fclose(f);
    return w;
}

int main(int argc, char** argv) {
    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("# Packed FP4 GEMV Benchmark — %s (%d.%d)\n\n", p.name, p.major, p.minor);

    const float s13 = 1.f/3.f;
    int iters = 1000;

    // ── Test 1: Q projection (K=2048, N=2048) ────────────────────────────────
    printf("=== Q Projection (K=2048, N=2048) ===\n");

    auto w_i8 = load_int8_w("weights_int8_bf16/0_self_attn.q_proj");
    auto w_fp4 = load_fp4_w("weights_packed_fp4/0_self_attn.q_proj");
    int K = w_i8.K, N = w_i8.N;

    // Device - INT8
    int8_t *d_w_i8; float *d_ws_i8;
    cudaMalloc(&d_w_i8, K*N); cudaMemcpy(d_w_i8, w_i8.d.data(), K*N, cudaMemcpyHostToDevice);
    cudaMalloc(&d_ws_i8, w_i8.sc.size()*4); cudaMemcpy(d_ws_i8, w_i8.sc.data(), w_i8.sc.size()*4, cudaMemcpyHostToDevice);

    // Device - FP4 packed
    uint8_t *d_w_fp4; float *d_ws_fp4;
    cudaMalloc(&d_w_fp4, w_fp4.packed.size()); cudaMemcpy(d_w_fp4, w_fp4.packed.data(), w_fp4.packed.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&d_ws_fp4, w_fp4.sc.size()*4); cudaMemcpy(d_ws_fp4, w_fp4.sc.data(), w_fp4.sc.size()*4, cudaMemcpyHostToDevice);

    // Activations: INT8 + FP4
    int8_t *d_xi8; float *d_xs_i8;
    uint8_t *d_xfp4; float *d_xs_fp4;
    cudaMalloc(&d_xi8, K); cudaMalloc(&d_xs_i8, (K/16)*4);
    cudaMalloc(&d_xfp4, K/2); cudaMalloc(&d_xs_fp4, (K/16)*4);

    // Fill with uniform values
    std::vector<int8_t> xh_i8(K, 42);
    std::vector<float> xsh_i8(K/16, s13);
    cudaMemcpy(d_xi8, xh_i8.data(), K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_xs_i8, xsh_i8.data(), (K/16)*4, cudaMemcpyHostToDevice);

    // Packed FP4 activation (from INT8)
    std::vector<uint8_t> xh_fp4(K/2);
    for (int j = 0; j < K/2; j++) {
        __nv_fp4_e2m1 f0(0.5f), f1(0.5f);
        uint8_t b0, b1;
        memcpy(&b0, &f0, 1); memcpy(&b1, &f1, 1);
        xh_fp4[j] = (b0 & 0x0F) | ((b1 & 0x0F) << 4);
    }
    std::vector<float> xsh_fp4(K/16, 0.166666f);
    cudaMemcpy(d_xfp4, xh_fp4.data(), K/2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_xs_fp4, xsh_fp4.data(), (K/16)*4, cudaMemcpyHostToDevice);

    float *d_y_i8, *d_y_fp4;
    cudaMalloc(&d_y_i8, N*4); cudaMalloc(&d_y_fp4, N*4);

    // Warmup
    for (int i = 0; i < 10; i++) {
        blackwell::kernels::gemv_int8_warp(d_y_i8, d_xi8, d_xs_i8, d_w_i8, d_ws_i8, K, N, 0);
        blackwell::kernels::gemv_fp4_warp(d_y_fp4, d_xfp4, d_xs_fp4, d_w_fp4, d_ws_fp4, K, N, 0);
    }
    cudaDeviceSynchronize();

    // Correctness
    std::vector<float> y_i8(N), y_fp4(N);
    cudaMemcpy(y_i8.data(), d_y_i8, N*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(y_fp4.data(), d_y_fp4, N*4, cudaMemcpyDeviceToHost);

    float max_diff = 0, cos_sim = 0, n_i8 = 0, n_fp4 = 0;
    for (int i = 0; i < N; i++) {
        float d = fabsf(y_i8[i] - y_fp4[i]);
        if (d > max_diff) max_diff = d;
        cos_sim += y_i8[i] * y_fp4[i];
        n_i8 += y_i8[i] * y_i8[i];
        n_fp4 += y_fp4[i] * y_fp4[i];
    }
    cos_sim /= sqrtf(n_i8) * sqrtf(n_fp4);
    printf("  INT8[0..3]:  %.6f %.6f %.6f %.6f\n", y_i8[0], y_i8[1], y_i8[2], y_i8[3]);
    printf("  FP4[0..3]:  %.6f %.6f %.6f %.6f\n", y_fp4[0], y_fp4[1], y_fp4[2], y_fp4[3]);
    printf("  Max diff: %.6f  Cosine: %.6f %s\n", max_diff, cos_sim,
        cos_sim > 0.999 ? "✅" : cos_sim > 0.9 ? "⚠️" : "❌");

    // Throughput
    GpuTimer t1, t2;
    t1.start();
    for (int i = 0; i < iters; i++)
        blackwell::kernels::gemv_int8_warp(d_y_i8, d_xi8, d_xs_i8, d_w_i8, d_ws_i8, K, N, 0);
    float ms_i8 = t1.stop();

    t2.start();
    for (int i = 0; i < iters; i++)
        blackwell::kernels::gemv_fp4_warp(d_y_fp4, d_xfp4, d_xs_fp4, d_w_fp4, d_ws_fp4, K, N, 0);
    float ms_fp4 = t2.stop();

    // Bandwidth: INT8 = K*N bytes, FP4 = K*N/2 bytes
    float bw_i8 = (float)(K*N) * iters / ms_i8 / 1e6;
    float bw_fp4 = (float)(K*N/2) * iters / ms_fp4 / 1e6;
    printf("  %-15s  %7.3f ms  %7.1f GB/s\n", "INT8 warp", ms_i8, bw_i8);
    printf("  %-15s  %7.3f ms  %7.1f GB/s\n", "FP4 packed", ms_fp4, bw_fp4);
    printf("  Speedup: %.2fx\n", ms_i8 / ms_fp4);

    cudaFree(d_w_i8); cudaFree(d_ws_i8);
    cudaFree(d_w_fp4); cudaFree(d_ws_fp4);
    cudaFree(d_xi8); cudaFree(d_xs_i8);
    cudaFree(d_xfp4); cudaFree(d_xs_fp4);
    cudaFree(d_y_i8); cudaFree(d_y_fp4);

    // ── Test 2: MLP gate (K=2048, N=6144) ────────────────────────────────────
    printf("\n=== MLP Gate Projection (K=2048, N=6144) ===\n");

    w_i8 = load_int8_w("weights_int8_bf16/0_mlp.gate_proj");
    w_fp4 = load_fp4_w("weights_packed_fp4/0_mlp.gate_proj");
    K = w_i8.K; N = w_i8.N;

    cudaMalloc(&d_w_i8, K*N); cudaMemcpy(d_w_i8, w_i8.d.data(), K*N, cudaMemcpyHostToDevice);
    cudaMalloc(&d_ws_i8, w_i8.sc.size()*4); cudaMemcpy(d_ws_i8, w_i8.sc.data(), w_i8.sc.size()*4, cudaMemcpyHostToDevice);
    cudaMalloc(&d_w_fp4, w_fp4.packed.size()); cudaMemcpy(d_w_fp4, w_fp4.packed.data(), w_fp4.packed.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&d_ws_fp4, w_fp4.sc.size()*4); cudaMemcpy(d_ws_fp4, w_fp4.sc.data(), w_fp4.sc.size()*4, cudaMemcpyHostToDevice);

    // K=2048 activation same as above
    cudaMalloc(&d_xi8, K); cudaMemcpy(d_xi8, xh_i8.data(), K, cudaMemcpyHostToDevice);
    cudaMalloc(&d_xs_i8, (K/16)*4); cudaMemcpy(d_xs_i8, xsh_i8.data(), (K/16)*4, cudaMemcpyHostToDevice);
    cudaMalloc(&d_xfp4, K/2); cudaMemcpy(d_xfp4, xh_fp4.data(), K/2, cudaMemcpyHostToDevice);
    cudaMalloc(&d_xs_fp4, (K/16)*4); cudaMemcpy(d_xs_fp4, xsh_fp4.data(), (K/16)*4, cudaMemcpyHostToDevice);

    cudaMalloc(&d_y_i8, N*4); cudaMalloc(&d_y_fp4, N*4);

    // Warmup
    for (int i = 0; i < 10; i++) {
        blackwell::kernels::gemv_int8_warp(d_y_i8, d_xi8, d_xs_i8, d_w_i8, d_ws_i8, K, N, 0);
        blackwell::kernels::gemv_fp4_warp(d_y_fp4, d_xfp4, d_xs_fp4, d_w_fp4, d_ws_fp4, K, N, 0);
    }
    cudaDeviceSynchronize();

    // Correctness
    std::vector<float> y2_i8(N), y2_fp4(N);
    cudaMemcpy(y2_i8.data(), d_y_i8, N*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(y2_fp4.data(), d_y_fp4, N*4, cudaMemcpyDeviceToHost);

    max_diff = 0; cos_sim = 0; n_i8 = 0; n_fp4 = 0;
    for (int i = 0; i < N; i++) {
        float d = fabsf(y2_i8[i] - y2_fp4[i]);
        if (d > max_diff) max_diff = d;
        cos_sim += y2_i8[i] * y2_fp4[i];
        n_i8 += y2_i8[i] * y2_i8[i];
        n_fp4 += y2_fp4[i] * y2_fp4[i];
    }
    cos_sim /= sqrtf(n_i8) * sqrtf(n_fp4);
    printf("  Max diff: %.6f  Cosine: %.6f %s\n", max_diff, cos_sim,
        cos_sim > 0.999 ? "✅" : cos_sim > 0.9 ? "⚠️" : "❌");

    GpuTimer t3, t4;
    t3.start();
    for (int i = 0; i < iters; i++)
        blackwell::kernels::gemv_int8_warp(d_y_i8, d_xi8, d_xs_i8, d_w_i8, d_ws_i8, K, N, 0);
    float ms_i8_2 = t3.stop();

    t4.start();
    for (int i = 0; i < iters; i++)
        blackwell::kernels::gemv_fp4_warp(d_y_fp4, d_xfp4, d_xs_fp4, d_w_fp4, d_ws_fp4, K, N, 0);
    float ms_fp4_2 = t4.stop();

    float bw_i8_2 = (float)(K*N) * iters / ms_i8_2 / 1e6;
    float bw_fp4_2 = (float)(K*N/2) * iters / ms_fp4_2 / 1e6;
    printf("  %-15s  %7.3f ms  %7.1f GB/s\n", "INT8 warp", ms_i8_2, bw_i8_2);
    printf("  %-15s  %7.3f ms  %7.1f GB/s\n", "FP4 packed", ms_fp4_2, bw_fp4_2);
    printf("  Speedup: %.2fx\n", ms_i8_2 / ms_fp4_2);

    cudaFree(d_w_i8); cudaFree(d_ws_i8);
    cudaFree(d_w_fp4); cudaFree(d_ws_fp4);
    cudaFree(d_xi8); cudaFree(d_xs_i8);
    cudaFree(d_xfp4); cudaFree(d_xs_fp4);
    cudaFree(d_y_i8); cudaFree(d_y_fp4);

    return 0;
}
