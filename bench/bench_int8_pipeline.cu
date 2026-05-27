// bench/bench_int8_pipeline.cu — INT8 vs FP4 pipeline throughput comparison
//
// Benchmarks gemv_fp4_v2 vs gemv_int8 on ALL layer-0 weights.
// Measures individual and aggregate bandwidth.
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120,code=sm_120 \
//     -I include bench/bench_int8_pipeline.cu build/libblackwell_kernels.a \
//     -o bench/bench_int8_pipeline

#include <cuda_runtime.h>
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
    void start() { cudaEventRecord(s); }
    float stop() { cudaEventRecord(e); cudaEventSynchronize(e);
                   float ms=0; cudaEventElapsedTime(&ms, s, e); return ms; }
};

static void chk(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) { printf("FAIL: %s: %s\n", msg, cudaGetErrorString(e)); exit(1); }
}

// Load int8_t weight file: header + N×K int8 data
static std::vector<int8_t> load_int8(const char* prefix, int& K, int& N) {
    char path[256]; snprintf(path,256,"%s.int8_t",prefix);
    FILE* f = fopen(path,"rb");
    int h[5]; fread(h,4,5,f); K=h[0]; N=h[1];
    std::vector<int8_t> d(K*N); fread(d.data(),1,K*N,f); fclose(f);
    return d;
}
static std::vector<float> load_scales(const char* prefix, int& nKb, int& nNb) {
    char path[256]; snprintf(path,256,"%s.scale_t",prefix);
    FILE* f = fopen(path,"rb");
    int h[5]; fread(h,4,5,f); nKb=h[3]; nNb=h[4];
    std::vector<float> d(nKb*nNb); fread(d.data(),4,nKb*nNb,f); fclose(f);
    return d;
}

struct Weight {
    int K, N;
    int8_t *d_w;
    float *d_sc;
};

Weight upload_int8(const char* prefix) {
    int K,N; auto d=load_int8(prefix,K,N);
    int nKb,nNb; auto sc=load_scales(prefix,nKb,nNb);
    Weight w{K,N};
    cudaMalloc(&w.d_w, K*N); cudaMalloc(&w.d_sc, nKb*nNb*4);
    cudaMemcpy(w.d_w, d.data(), K*N, cudaMemcpyHostToDevice);
    cudaMemcpy(w.d_sc, sc.data(), nKb*nNb*4, cudaMemcpyHostToDevice);
    return w;
}

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("# INT8 Pipeline Bandwidth — Qwen3-1.7B Layer 0\n");
    printf("Device: %s\n\n", p.name);

    // Weights
    struct Wspec { const char* name; const char* prefix; int K, N; };
    Wspec ws[] = {
        {"q_proj",  "weights_int8_bf16/0_self_attn.q_proj", 2048, 2048},
        {"k_proj",  "weights_int8_bf16/0_self_attn.k_proj", 2048, 1024},
        {"v_proj",  "weights_int8_bf16/0_self_attn.v_proj", 2048, 1024},
        {"o_proj",  "weights_int8_bf16/0_self_attn.o_proj", 2048, 2048},
        {"gate",    "weights_int8_bf16/0_mlp.gate_proj",    2048, 6144},
        {"up",      "weights_int8_bf16/0_mlp.up_proj",      2048, 6144},
        {"down",    "weights_int8_bf16/0_mlp.down_proj",   6144, 2048},
    };
    const int NW = sizeof(ws)/sizeof(ws[0]);

    Weight w[NW];
    for (int i = 0; i < NW; ++i)
        w[i] = upload_int8(ws[i].prefix);

    // FP4 input x (shared across all)
    void *d_x4; float *d_xs;
    cudaMalloc(&d_x4, 2048); cudaMalloc(&d_xs, 128*4);
    float xs_val = 1.f/3.f;
    std::vector<float> xs_h(128, xs_val);
    cudaMemcpy(d_xs, xs_h.data(), 128*4, cudaMemcpyHostToDevice);
    float *d_x32; cudaMalloc(&d_x32, 2048*4);
    std::vector<float> x32_h(2048, 0.5f);
    cudaMemcpy(d_x32, x32_h.data(), 2048*4, cudaMemcpyHostToDevice);
    blackwell::kernels::pack_fp4(d_x4, d_x32, xs_h.data(), 2048, 0);

    // INT8 input x (FP4 → unpack → pack_int8)
    int8_t *d_i8_x;
    float *d_i8_xs;
    cudaMalloc(&d_i8_x, 2048);
    cudaMalloc(&d_i8_xs, 128*4);
    float i8_xv = 0.5f/127.f;
    std::vector<float> i8_xs_h(128, i8_xv);
    cudaMemcpy(d_i8_xs, i8_xs_h.data(), 128*4, cudaMemcpyHostToDevice);
    blackwell::kernels::pack_int8(d_i8_x, d_x32, d_i8_xs, 2048, 0);

    cudaDeviceSynchronize();

    // Benchmark each GEMV
    const int warm = 50, iter = 500;

    printf("%-12s %8s %8s %8s %8s %8s   %s\n",
           "weight", "K", "N", "fp4_ms", "i8_ms", "speedup", "int8_gbps");

    for (int i = 0; i < NW; ++i) {
        // FP4 v2: use 2048-input x_fp4 and FP4 weights
        // We don't have FP4 weights transposed for all — skip FP4 bench here
        // Just bench INT8

        // Warmup
        for (int j = 0; j < warm; ++j) {
            float *d_y;
            cudaMalloc(&d_y, ws[i].N*4);
            blackwell::kernels::gemv_int8(d_y, d_i8_x, d_i8_xs,
                w[i].d_w, w[i].d_sc, ws[i].K, ws[i].N, 0);
            cudaFree(d_y);
        }

        float *d_y;
        cudaMalloc(&d_y, ws[i].N*4);

        // INT8 bench
        GpuTimer ti;
        ti.start();
        for (int j = 0; j < iter; ++j)
            blackwell::kernels::gemv_int8(d_y, d_i8_x, d_i8_xs,
                w[i].d_w, w[i].d_sc, ws[i].K, ws[i].N, 0);
        float ms_i8 = ti.stop() / iter;
        cudaDeviceSynchronize();
        cudaFree(d_y);

        // Bandwidth: weights(K*N) + weight_scales(nKb*nNb*4)
        //           + input(K) + input_scales(K/16*4) + output(N*4)
        size_t w_bytes = (size_t)ws[i].K * ws[i].N;
        size_t ws_bytes = (size_t)(ws[i].K/16) * (ws[i].N/16) * 4;
        size_t total = w_bytes + ws_bytes + ws[i].K + (ws[i].K/16)*4 + ws[i].N*4;
        double gbps = (double)total / 1e9 / (ms_i8 / 1e3);

        printf("%-12s %8d %8d %8s %8.4f %8.2fx %8.1f\n",
               ws[i].name, ws[i].K, ws[i].N,
               "-", ms_i8,
               0.0,  // FP4 speedup placeholder
               gbps);
    }

    printf("\nFP4 speedup requires FP4 weights; INT8 bandwidth shown above.\n");

    // Cleanup
    for (int i = 0; i < NW; ++i) {
        cudaFree(w[i].d_w); cudaFree(w[i].d_sc);
    }
    cudaFree(d_x4); cudaFree(d_xs); cudaFree(d_x32);
    cudaFree(d_i8_x); cudaFree(d_i8_xs);
    return 0;
}