// bench/phase_a.cu — Phase A benchmark: FP4 kernel throughput on RTX 5060 Ti / SM_120
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120,code=sm_120 \
//     -I include \
//     bench/phase_a.cu \
//     build/libblackwell_kernels.a \
//     -o bench/phase_a

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <cassert>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin() { cudaEventRecord(start, 0); }
    float end() { cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
                  float ms=0; cudaEventElapsedTime(&ms, start, stop); return ms; }
};

void cpu_gemv_ref(const float* W, const float* x, float* y, int in_, int out_) {
    for (int o = 0; o < out_; ++o) {
        float s = 0.f;
        for (int i = 0; i < in_; ++i) s += W[o * in_ + i] * x[i];
        y[o] = s;
    }
}

float max_rel_err(const float* a, const float* b, int n, float eps=1e-6f) {
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

std::string fmt(int n) {
    std::string s = std::to_string(n);
    int p = (int)s.length() - 3;
    while (p > 0) { s.insert(p, ","); p -= 3; }
    return s;
}

static bool check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        printf("  FAIL: %s: %s\n", msg, cudaGetErrorString(e));
        return false;
    }
    return true;
}

// ===== test pack/unpack =====
void test_pack(int n) {
    printf("\n## 1. FP4 Pack/Unpack Correctness\n\n");
    std::vector<float> h(n);
    for (int i = 0; i < n; ++i) h[i] = 2.f * static_cast<float>(rand()) / RAND_MAX - 1.f;

    float *d_fp32, *d_rec; void *d_fp4; float *d_scale;
    cudaMalloc(&d_fp32, n*4); cudaMalloc(&d_rec, n*4);
    cudaMalloc(&d_fp4, n); cudaMalloc(&d_scale, 4);
    cudaMemcpy(d_fp32, h.data(), n*4, cudaMemcpyHostToDevice);

    float amax = 0.f;
    for (auto v : h) { float a = fabsf(v); if (a > amax) amax = a; }
    float sc = amax / 3.f;
    cudaMemcpy(d_scale, &sc, 4, cudaMemcpyHostToDevice);

    if (!check(blackwell::kernels::pack_fp4(d_fp4, d_fp32, d_scale, n, 0), "pack_fp4")) return;
    if (!check(blackwell::kernels::unpack_fp4(d_rec, d_fp4, d_scale, n, 0), "unpack_fp4")) return;
    std::vector<float> r(n);
    cudaMemcpy(r.data(), d_rec, n*4, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    printf("| Operation | Elements | Max Rel Error |\n| --- | --- | --- |\n");
    printf("| pack+unpack | %s | %.4e |\n\n", fmt(n).c_str(), max_rel_err(h.data(), r.data(), n));
    cudaFree(d_fp32); cudaFree(d_rec); cudaFree(d_fp4); cudaFree(d_scale);
}

// ===== GEMV =====
double bench_gemv(const float* W_h, int in_, int out_, int warm, int bench, double* gbps_out) {
    *gbps_out = 0;
    int nw = in_ * out_;
    int nb_scales_x = in_ / 16;  // 1 per 16-element FP4 block
    float *d_W32, *d_x32, *d_y;
    void   *d_W4, *d_x4;
    float  *d_ws, *d_xs;
    cudaMalloc(&d_W32, nw*4); cudaMalloc(&d_x32, in_*4); cudaMalloc(&d_y, out_*4);
    cudaMalloc(&d_W4, nw); cudaMalloc(&d_x4, in_);
    cudaMalloc(&d_ws, 4); cudaMalloc(&d_xs, nb_scales_x*4);

    cudaMemcpy(d_W32, W_h, nw*4, cudaMemcpyHostToDevice);
    std::vector<float> x_h(in_, 1.f);
    cudaMemcpy(d_x32, x_h.data(), in_*4, cudaMemcpyHostToDevice);

    float am = 0.f;
    for (int i = 0; i < nw; ++i) { float a = fabsf(W_h[i]); if (a > am) am = a; }
    float sv = am / 3.f;
    cudaMemcpy(d_ws, &sv, 4, cudaMemcpyHostToDevice);
    if (!check(blackwell::kernels::pack_fp4(d_W4, d_W32, d_ws, nw, 0), "pack_fp4(W)")) return 0;

    // Pack x to FP4 with per-block scales
    float x_sv = 1.f / 3.f;  // uniform input = 1.0
    std::vector<float> x_scales_h(nb_scales_x, x_sv);
    cudaMemcpy(d_xs, x_scales_h.data(), nb_scales_x*4, cudaMemcpyHostToDevice);
    if (!check(blackwell::kernels::pack_fp4(d_x4, d_x32, d_xs, in_, 0), "pack_fp4(x)")) return 0;

    for (int i = 0; i < warm; ++i)
        check(blackwell::kernels::gemv_fp4(d_y, d_x4, d_xs, d_W4, d_ws, in_, out_, 0), "gemv_fp4(warm)");
    cudaDeviceSynchronize();

    GpuTimer t;
    t.begin();
    for (int i = 0; i < bench; ++i) {
        cudaError_t e = blackwell::kernels::gemv_fp4(d_y, d_x4, d_xs, d_W4, d_ws, in_, out_, 0);
        if (e != cudaSuccess) { printf("  FAIL iter %d: %s\n", i, cudaGetErrorString(e)); break; }
    }
    float ms = t.end() / bench;

    size_t total = in_ + nw + nb_scales_x*4 + 4 + out_*4;
    *gbps_out = bw_gbps(total, ms);

    std::vector<float> y_gpu(out_);
    cudaMemcpy(y_gpu.data(), d_y, out_*4, cudaMemcpyDeviceToHost);
    std::vector<float> y_cpu(out_);
    cpu_gemv_ref(W_h, x_h.data(), y_cpu.data(), in_, out_);
    float re = max_rel_err(y_cpu.data(), y_gpu.data(), out_);

    printf("| gemv_fp4 | %s x %s | %.3f | %.1f | %.4e |\n",
           fmt(out_).c_str(), fmt(in_).c_str(), ms, *gbps_out, re);

    cudaFree(d_W32); cudaFree(d_x32); cudaFree(d_y);
    cudaFree(d_W4); cudaFree(d_x4); cudaFree(d_ws); cudaFree(d_xs);
    return ms;
}

// ===== GEMM =====
double bench_gemm(int M, int N, int K, int warm, int bench, double* gbps_out) {
    *gbps_out = 0;
    int nA = M*K, nB = K*N, nC = M*N;
    int nblk_A = (M/16)*(K/16), nblk_B = (K/16)*(N/16);  // per-block scales
    float *d_A32, *d_B32, *d_C, *d_As, *d_Bs;
    void   *d_A4, *d_B4;
    cudaMalloc(&d_A32, nA*4); cudaMalloc(&d_B32, nB*4); cudaMalloc(&d_C, nC*4);
    cudaMalloc(&d_A4, nA); cudaMalloc(&d_B4, nB);
    cudaMalloc(&d_As, nblk_A*4); cudaMalloc(&d_Bs, nblk_B*4);

    std::vector<float> A_h(nA, 1.f), B_h(nB, 1.f);
    cudaMemcpy(d_A32, A_h.data(), nA*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B32, B_h.data(), nB*4, cudaMemcpyHostToDevice);

    // Per-block scales: uniform = 1.0 / 3.0
    float sv = 1.f / 3.f;
    std::vector<float> A_scales(nblk_A, sv), B_scales(nblk_B, sv);
    cudaMemcpy(d_As, A_scales.data(), nblk_A*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bs, B_scales.data(), nblk_B*4, cudaMemcpyHostToDevice);

    if (!check(blackwell::kernels::pack_fp4(d_A4, d_A32, d_As, nA, 0), "pack_fp4(A)")) { cudaFree(d_A32); cudaFree(d_B32); cudaFree(d_C); cudaFree(d_A4); cudaFree(d_B4); cudaFree(d_As); cudaFree(d_Bs); return 0; }
    if (!check(blackwell::kernels::pack_fp4(d_B4, d_B32, d_Bs, nB, 0), "pack_fp4(B)")) { cudaFree(d_A32); cudaFree(d_B32); cudaFree(d_C); cudaFree(d_A4); cudaFree(d_B4); cudaFree(d_As); cudaFree(d_Bs); return 0; }

    for (int i = 0; i < warm; ++i)
        check(blackwell::kernels::gemm_fp4_block_scaled(d_C, d_A4, d_As, d_B4, d_Bs, M, N, K, 0), "gemm(warm)");
    cudaDeviceSynchronize();

    GpuTimer t;
    t.begin();
    for (int i = 0; i < bench; ++i) {
        cudaError_t e = blackwell::kernels::gemm_fp4_block_scaled(d_C, d_A4, d_As, d_B4, d_Bs, M, N, K, 0);
        if (e != cudaSuccess) { printf("  FAIL iter %d: %s\n", i, cudaGetErrorString(e)); break; }
    }
    float ms = t.end() / bench;

    size_t br = nA + nB, bw = nC*4;
    *gbps_out = bw_gbps(br + bw, ms);

    printf("| gemm_fp4 | %s x %s x %s | %.3f | %.1f |\n",
           fmt(M).c_str(), fmt(N).c_str(), fmt(K).c_str(), ms, *gbps_out);

    cudaFree(d_A32); cudaFree(d_B32); cudaFree(d_C);
    cudaFree(d_A4); cudaFree(d_B4); cudaFree(d_As); cudaFree(d_Bs);
    return ms;
}

// ===== RMSNorm =====
void bench_rmsnorm(int n, int warm, int bench) {
    std::vector<float> in(n, 1.f), wt(n, 1.f);
    float *d_in, *d_wt, *d_out;
    cudaMalloc(&d_in, n*4); cudaMalloc(&d_wt, n*4); cudaMalloc(&d_out, n*4);
    cudaMemcpy(d_in, in.data(), n*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wt, wt.data(), n*4, cudaMemcpyHostToDevice);

    for (int i = 0; i < warm; ++i)
        blackwell::kernels::fused_rmsnorm(d_out, d_in, d_wt, n, 1e-5f, 0);
    cudaDeviceSynchronize();

    GpuTimer t;
    t.begin();
    for (int i = 0; i < bench; ++i)
        blackwell::kernels::fused_rmsnorm(d_out, d_in, d_wt, n, 1e-5f, 0);
    float ms = t.end() / bench;
    double gb = bw_gbps(n*4*3, ms);
    printf("| fused_rmsnorm | %s | %.3f | %.1f |\n", fmt(n).c_str(), ms, gb);
    cudaFree(d_in); cudaFree(d_wt); cudaFree(d_out);
}

// ===== SwiGLU =====
void bench_swiglu(int n, int warm, int bench) {
    std::vector<float> g(n, 1.f), u(n, 1.f);
    float *d_g, *d_u, *d_o;
    cudaMalloc(&d_g, n*4); cudaMalloc(&d_u, n*4); cudaMalloc(&d_o, n*4);
    cudaMemcpy(d_g, g.data(), n*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u.data(), n*4, cudaMemcpyHostToDevice);

    for (int i = 0; i < warm; ++i)
        blackwell::kernels::apply_swiglu(d_o, d_g, d_u, n, 0);
    cudaDeviceSynchronize();

    GpuTimer t;
    t.begin();
    for (int i = 0; i < bench; ++i)
        blackwell::kernels::apply_swiglu(d_o, d_g, d_u, n, 0);
    float ms = t.end() / bench;
    double gb = bw_gbps(n*4*3, ms);
    printf("| apply_swiglu | %s | %.3f | %.1f |\n", fmt(n).c_str(), ms, gb);
    cudaFree(d_g); cudaFree(d_u); cudaFree(d_o);
}

// ===== Main =====
int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("# Blackwell Phase A Benchmark\n\n");
    printf("Device: %s (CC %d.%d, %.0f MiB VRAM, %d SMs)\n\n",
           p.name, p.major, p.minor, p.totalGlobalMem/1048576., p.multiProcessorCount);

    srand(42);
    const int W = 10, B = 100;

    // 1. Pack/Unpack correctness
    test_pack(512);

    // 2. GEMV
    printf("\n## 2. FP4 GEMV (Decode Path)\n\n");
    printf("| Op | Shape (out x in) | Lat (ms) | GB/s | Rel Err |\n| --- | --- | --- | --- | --- |\n");
    int gemv_shapes[][2] = {{64,64},{128,64},{2048,64},{6144,64}};
    for (int i = 0; i < 4; ++i) {
        int o = gemv_shapes[i][0], k = gemv_shapes[i][1];
        std::vector<float> W(o * k, 1.f);
        double gb = 0;
        bench_gemv(W.data(), k, o, 10, 100, &gb);
    }

    // 3. GEMM
    printf("\n## 3. FP4 GEMM (Prefill Path)\n\n");
    printf("| Op | Shape | Lat (ms) | GB/s |\n| --- | --- | --- | --- |\n");
    struct {int M,N,K;} gemm_shapes[] = {{512,2048,2048},{512,6144,2048},{512,2048,6144},{2048,2048,2048}};
    for (auto s : gemm_shapes) {
        double gb = 0;
        bench_gemm(s.M, s.N, s.K, W, B/10, &gb);
    }

    // 4. Fused epilogues
    printf("\n## 4. Fused Epilogues\n\n");
    printf("| Op | Elements | Lat (ms) | GB/s |\n| --- | --- | --- | --- |\n");
    bench_rmsnorm(4096, W, B);
    bench_rmsnorm(2048, W, B);
    bench_swiglu(2048*256, W, B);
    bench_swiglu(6144*256, W, B);

    printf("\n## 5. Summary\n\n");
    printf("Phase A establishes baseline FP4 kernel throughput on RTX 5060 Ti.\n");
    printf("Compare with llama-bench: Qwen3.5-4B Q4_K_M on same hardware:\n");
    printf("  prefill: ~4560 t/s (pp512)  decode: ~114 t/s (tg128)\n\n");
    return 0;
}
