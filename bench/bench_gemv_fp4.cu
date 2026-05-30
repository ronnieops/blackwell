// bench/bench_gemv_fp4.cu — Compare FP4 GEMV: original vs optimized (FP32 scales + FP16 acc)
#include <cuda_runtime.h>
#include <cuda_fp4.h>
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

// UE4M3 → float (host)
float ue4m3_to_f(uint8_t v) {
    if (v == 0) return 0.0f;
    int exp = (v >> 3) & 0xF;
    int man = v & 0x7;
    if (exp == 0) return (man / 8.0f) * (1.0f / 64.0f);
    return (1.0f + man / 8.0f) * exp2f((float)(exp - 7));
}

struct LW {
    std::vector<__nv_fp4_e2m1> d_fp4;  // W_t FP4 [N × K]
    std::vector<uint8_t> sc_ue4;       // UE4M3 scales
    std::vector<float> sc_fp32;        // FP32 scales (pre-converted)
};

// Load weights in FP4 format (Qwen3 projection)
LW load_fp4(const char* prefix) {
    LW w;
    // Load scale file first to get dimensions
    char p[256]; snprintf(p,256,"%s.scale_t",prefix);
    FILE* f=fopen(p,"rb"); if(!f){printf("Cannot open %s\n",p);exit(1);}
    int h[5]; (void)fread(h,4,5,f);
    int K=h[0], N=h[1], BLK=h[2], nKb=h[3], nNb=h[4];  // nKb=K/16, nNb=N (per-row scales)
    fclose(f);

    // Load scale data (UE4M3 format, but stored as float32 — need to re-read)
    // Actually the scale_t file stores FP32, not UE4M3. UE4M3 is for NVF4.
    // Let's load them as FP32 and also create UE4M3 equivalents for the old path.
    snprintf(p,256,"%s.scale_t",prefix);
    f=fopen(p,"rb"); if(!f){printf("Cannot open %s\n",p);exit(1);}
    (void)fread(h,4,5,f);
    size_t ns = (size_t)h[3]*h[4];  // nKb * N
    w.sc_fp32.resize(ns);
    (void)fread(w.sc_fp32.data(),4,ns,f);
    fclose(f);

    // Convert FP32 scales to UE4M3 (quantized approximation)
    w.sc_ue4.resize(ns);
    for (size_t i = 0; i < ns; ++i) {
        float v = w.sc_fp32[i];
        // Simple quantize to UE4M3: 3-bit exponent + 3-bit mantissa, range ~0.0156 to 448
        if (v <= 0.0f) { w.sc_ue4[i] = 0; continue; }
        int e = (int)floorf(log2f(v)) + 7;  // bias 7
        if (e < 0) { w.sc_ue4[i] = 0; continue; }
        if (e > 14) { e = 14; }  // clamp to max
        float norm = v / exp2f((float)(e - 7));
        int m = (int)roundf((norm - 1.0f) * 8.0f);
        if (m < 0) m = 0;
        if (m > 7) m = 7;
        w.sc_ue4[i] = (uint8_t)((e << 3) | m);
    }

    // Load FP4 weight data
    snprintf(p,256,"%s.int8_t",prefix);
    f=fopen(p,"rb"); if(!f){printf("Cannot open %s\n",p);exit(1);}
    (void)fread(h,4,5,f);
    size_t ne = (size_t)K * N;
    w.d_fp4.resize(ne);
    (void)fread(w.d_fp4.data(),1,ne,f);
    fclose(f);

    printf("  Loaded %s: K=%d N=%d scales=%zu data=%zu\n", prefix, K, N, ns, ne);
    return w;
}

const int H=2048, QD=2048, KV=1024, ID=6144;

int main() {
    printf("=== FP4 GEMV Benchmark: original vs optimized ===\n\n");

    struct TestCase { const char* name; int N; int K; const char* path; };
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

    // Input FP4 vector + scales
    int maxK = ID;
    std::vector<__nv_fp4_e2m1> h_x(maxK);
    std::vector<float> h_xsc_fp32(maxK / 16);
    std::vector<uint8_t> h_xsc_ue4(maxK / 16);
    for (int i = 0; i < maxK; ++i) {
        // Generate synthetic FP4 values
        float v = ((i * 31 + 7) % 127 - 63) * 0.01f;
        h_x[i] = __nv_fp4_e2m1(v);
    }
    for (size_t i = 0; i < h_xsc_fp32.size(); ++i) {
        float sc = 0.5f;  // fixed scale for synthetic data
        h_xsc_fp32[i] = sc;
        // UE4M3 quantize
        if (sc <= 0.0f) { h_xsc_ue4[i] = 0; continue; }
        int e = (int)floorf(log2f(sc)) + 7;
        if (e < 0) { h_xsc_ue4[i] = 0; continue; }
        if (e > 14) e = 14;
        float norm = sc / exp2f((float)(e - 7));
        int m = (int)roundf((norm - 1.0f) * 8.0f);
        if (m < 0) m = 0; if (m > 7) m = 7;
        h_xsc_ue4[i] = (uint8_t)((e << 3) | m);
    }

    printf("%-12s %8s %8s %10s %10s %8s\n",
           "Projection", "N", "K", "Old(ms)", "New(ms)", "Speedup");

    for (int t = 0; t < n; ++t) {
        auto& tc = tests[t];
        int N = tc.N, K = tc.K;

        // Load weights
        LW w = load_fp4(tc.path);

        // Device buffers
        __nv_fp4_e2m1 *d_W, *d_x;
        uint8_t *d_Wsc_ue4, *d_xsc_ue4;
        float *d_Wsc_fp32, *d_xsc_fp32, *d_y;
        die(cudaMalloc(&d_W, K * N * sizeof(__nv_fp4_e2m1)), "d_W");
        die(cudaMalloc(&d_Wsc_ue4, w.sc_ue4.size()), "d_Wsc_ue4");
        die(cudaMalloc(&d_Wsc_fp32, w.sc_fp32.size() * sizeof(float)), "d_Wsc_fp32");
        die(cudaMalloc(&d_x, K * sizeof(__nv_fp4_e2m1)), "d_x");
        die(cudaMalloc(&d_xsc_ue4, K/16), "d_xsc_ue4");
        die(cudaMalloc(&d_xsc_fp32, (K/16) * sizeof(float)), "d_xsc_fp32");
        die(cudaMalloc(&d_y, N * sizeof(float)), "d_y");
        die(cudaMemcpy(d_W, w.d_fp4.data(), K*N*sizeof(__nv_fp4_e2m1), cudaMemcpyHostToDevice), "cpy_W");
        die(cudaMemcpy(d_Wsc_ue4, w.sc_ue4.data(), w.sc_ue4.size(), cudaMemcpyHostToDevice), "cpy_Wsc_u");
        die(cudaMemcpy(d_Wsc_fp32, w.sc_fp32.data(), w.sc_fp32.size()*sizeof(float), cudaMemcpyHostToDevice), "cpy_Wsc_f");
        die(cudaMemcpy(d_x, h_x.data(), K*sizeof(__nv_fp4_e2m1), cudaMemcpyHostToDevice), "cpy_x");
        die(cudaMemcpy(d_xsc_ue4, h_xsc_ue4.data(), K/16, cudaMemcpyHostToDevice), "cpy_xsc_u");
        die(cudaMemcpy(d_xsc_fp32, h_xsc_fp32.data(), (K/16)*sizeof(float), cudaMemcpyHostToDevice), "cpy_xsc_f");

        // Warm-up
        for (int w = 0; w < 3; ++w) {
            blackwell::kernels::gemv_fp4_nv(d_y, d_x, d_xsc_ue4, d_W, d_Wsc_ue4, K, N, 0);
            blackwell::kernels::gemv_fp4_nv_opt(d_y, d_x, d_xsc_fp32, d_W, d_Wsc_fp32, K, N, 0);
        }
        cudaDeviceSynchronize();

        // Benchmark old
        GpuTimer to; to.start();
        int iter = 50;
        for (int i = 0; i < iter; ++i)
            blackwell::kernels::gemv_fp4_nv(d_y, d_x, d_xsc_ue4, d_W, d_Wsc_ue4, K, N, 0);
        cudaDeviceSynchronize();
        float ms_old = to.stop() / iter;

        // Benchmark new
        GpuTimer tn; tn.start();
        for (int i = 0; i < iter; ++i)
            blackwell::kernels::gemv_fp4_nv_opt(d_y, d_x, d_xsc_fp32, d_W, d_Wsc_fp32, K, N, 0);
        cudaDeviceSynchronize();
        float ms_new = tn.stop() / iter;

        float speedup = (ms_new > 0) ? ms_old / ms_new : 0;
        printf("%-12s %8d %8d %8.4f  %8.4f  %7.2fx\n",
               tc.name, N, K, ms_old, ms_new, speedup);

        cudaFree(d_W); cudaFree(d_Wsc_ue4); cudaFree(d_Wsc_fp32);
        cudaFree(d_x); cudaFree(d_xsc_ue4); cudaFree(d_xsc_fp32); cudaFree(d_y);
    }
    return 0;
}
