// bench/bench_gemm_dp4a.cu — Compare FP32×INT8 (scalar) vs INT8×INT8 (__dp4a) GEMM
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>
#include "blackwell/kernels.h"

static void die(cudaError_t e, const char* m) {
    if(e!=cudaSuccess){printf("FAIL %s: %s\n",m,cudaGetErrorString(e));exit(1);}
}

struct GpuTimer {
    cudaEvent_t s,e;
    GpuTimer(){cudaEventCreate(&s);cudaEventCreate(&e);}
    ~GpuTimer(){cudaEventDestroy(s);cudaEventDestroy(e);}
    void start(){cudaEventRecord(s,0);}
    float stop(){cudaEventRecord(e,0);cudaEventSynchronize(e);float m=0;cudaEventElapsedTime(&m,s,e);return m;}
};

struct LW { std::vector<int8_t> d; std::vector<float> sc; };
static LW lw(const char* p) {
    char x[256]; snprintf(x,256,"%s.int8_t",p);
    FILE* f=fopen(x,"rb"); (void)f; if(!f){printf("FAIL open %s\n",x);exit(1);}
    int h[5]; (void)fread(h,4,5,f); LW w;
    w.d.resize(h[0]*h[1]); (void)fread(w.d.data(),1,w.d.size(),f); fclose(f);
    snprintf(x,256,"%s.scale_t",p); f=fopen(x,"rb"); (void)fread(h,4,5,f);
    w.sc.resize(h[3]*h[4]); (void)fread(w.sc.data(),4,w.sc.size(),f); fclose(f);
    return w;
}

struct TestCase { const char* name; int N; int K; const char* path; };

// Benchmark one GEMM method: return avg ms
float bench_gemm_old(TestCase& tc, int M, float* d_A, float* d_C, int8_t* d_W, float* d_Wsc) {
    GpuTimer t;
    int warm = 5, iter = 20;
    for (int i = 0; i < warm; ++i)
        die(blackwell::kernels::gemm_int8(d_C, d_A, d_W, d_Wsc, M, tc.N, tc.K, 0), "gemm_int8");
    t.start();
    for (int i = 0; i < iter; ++i)
        die(blackwell::kernels::gemm_int8(d_C, d_A, d_W, d_Wsc, M, tc.N, tc.K, 0), "gemm_int8");
    float ms = t.stop();
    return ms / iter;
}

float bench_gemm_new(TestCase& tc, int M, float* d_A,
                     int8_t* d_Ai8, float* d_Asc, float* d_C,
                     int8_t* d_W, float* d_Wsc,
                     float* out_quant_ms, float* out_gemm_ms) {
    GpuTimer tq, tg;
    int warm = 5, iter = 20;
    // Measure quantize separately
    for (int i = 0; i < warm; ++i)
        die(blackwell::kernels::quantize_int8(d_Ai8, d_Asc, d_A, M * tc.K, 0), "quantize_int8");
    tq.start();
    for (int i = 0; i < iter; ++i)
        die(blackwell::kernels::quantize_int8(d_Ai8, d_Asc, d_A, M * tc.K, 0), "quantize_int8");
    *out_quant_ms = tq.stop() / iter;

    // Ensure quantized data is ready
    die(blackwell::kernels::quantize_int8(d_Ai8, d_Asc, d_A, M * tc.K, 0), "quantize_int8_warm");
    // Measure dp4a GEMM separately
    for (int i = 0; i < warm; ++i)
        die(blackwell::kernels::gemm_int8_dp4a(d_C, d_Ai8, d_Asc, d_W, d_Wsc, M, tc.N, tc.K, 0), "gemm_int8_dp4a");
    tg.start();
    for (int i = 0; i < iter; ++i)
        die(blackwell::kernels::gemm_int8_dp4a(d_C, d_Ai8, d_Asc, d_W, d_Wsc, M, tc.N, tc.K, 0), "gemm_int8_dp4a");
    *out_gemm_ms = tg.stop() / iter;
    return *out_quant_ms + *out_gemm_ms;
}

const int H=2048, QD=2048, KV=1024, ID=6144;

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 128;

    printf("=== GEMM Throughput: FP32×INT8 (scalar) vs INT8×INT8 (__dp4a) ===\n");
    printf("  M=%d (prefill batch size)\n\n", M);

    TestCase tests[] = {
        {"Q proj", QD, H, "weights_int8_bf16/0_self_attn.q_proj"},
        {"K proj", KV, H, "weights_int8_bf16/0_self_attn.k_proj"},
        {"V proj", KV, H, "weights_int8_bf16/0_self_attn.v_proj"},
        {"O proj", H, QD, "weights_int8_bf16/0_self_attn.o_proj"},
        {"gate",   ID, H, "weights_int8_bf16/0_mlp.gate_proj"},
        {"up",     ID, H, "weights_int8_bf16/0_mlp.up_proj"},
        {"down",   H, ID, "weights_int8_bf16/0_mlp.down_proj"},
    };
    int n = sizeof(tests)/sizeof(tests[0]);
    int maxK = ID, maxN = ID;

    // FP32 input A
    float *d_A, *d_C;
    die(cudaMalloc(&d_A, M * maxK * sizeof(float)), "d_A");
    die(cudaMalloc(&d_C, M * maxN * sizeof(float)), "d_C");

    // Quantized buffer (for __dp4a path)
    int8_t *d_Ai8;
    float *d_Asc;
    int a_elems = M * maxK;
    int a_nblks = a_elems / 16;
    die(cudaMalloc(&d_Ai8, a_elems), "d_Ai8");
    die(cudaMalloc(&d_Asc, a_nblks * sizeof(float)), "d_Asc");

    // Fill input with synthetic data
    std::vector<float> h_A(M * maxK);
    for (int i = 0; i < M * maxK; ++i)
        h_A[i] = ((i * 31 + 7) % 127 - 63) * 0.01f;
    die(cudaMemcpy(d_A, h_A.data(), M * maxK * sizeof(float), cudaMemcpyHostToDevice), "cpy_A");

    printf("%-12s %8s %8s %8s %8s %8s %8s %8s\n",
           "Projection", "N", "K", "Old(ms)", "Quant(ms)", "DP4A(ms)", "New(ms)", "Speedup");

    for (int t = 0; t < n; ++t) {
        auto& tc = tests[t];
        int N = tc.N, K = tc.K;
        LW w = lw(tc.path);

        int8_t* d_W; float* d_Wsc;
        die(cudaMalloc((void**)&d_W, w.d.size()), "d_W");
        die(cudaMalloc((void**)&d_Wsc, w.sc.size()*sizeof(float)), "d_Wsc");
        die(cudaMemcpy(d_W, w.d.data(), w.d.size(), cudaMemcpyHostToDevice), "cpy_W");
        die(cudaMemcpy(d_Wsc, w.sc.data(), w.sc.size()*sizeof(float), cudaMemcpyHostToDevice), "cpy_Wsc");

        // Pre-quantize activations to INT8 (warm + part of timed path)
        float ms_old = bench_gemm_old(tc, M, d_A, d_C, d_W, d_Wsc);
        float quant_ms = 0, gemm_ms = 0;
        float ms_new = bench_gemm_new(tc, M, d_A, d_Ai8, d_Asc, d_C, d_W, d_Wsc, &quant_ms, &gemm_ms);
        float speedup = (ms_new > 0) ? ms_old / ms_new : 0;

        printf("%-12s %8d %8d %8.3f %8.3f %8.3f %8.3f %7.2fx\n",
               tc.name, N, K, ms_old, quant_ms, gemm_ms, ms_new, speedup);

        cudaFree(d_W);
        cudaFree(d_Wsc);
    }

    cudaFree(d_A);
    cudaFree(d_Ai8);
    cudaFree(d_Asc);
    cudaFree(d_C);
    return 0;
}
