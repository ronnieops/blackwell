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
    int nb_scales_x = in_ / 16;
    int nb_scales_W = (in_ / 16) * (out_ / 16);  // 2D block scales: (K/16) × (N/16)
    float *d_W32, *d_x32, *d_y;
    void   *d_W4, *d_x4;
    float  *d_ws, *d_xs;
    cudaMalloc(&d_W32, nw*4); cudaMalloc(&d_x32, in_*4); cudaMalloc(&d_y, out_*4);
    cudaMalloc(&d_W4, nw); cudaMalloc(&d_x4, in_);
    cudaMalloc(&d_ws, nb_scales_W*4); cudaMalloc(&d_xs, nb_scales_x*4);

    cudaMemcpy(d_W32, W_h, nw*4, cudaMemcpyHostToDevice);
    std::vector<float> x_h(in_, 1.f);
    cudaMemcpy(d_x32, x_h.data(), in_*4, cudaMemcpyHostToDevice);

    // W_scale: one scale per (16,16) block, matching GEMM convention
    // W_scale[(k/16) * (N/16) + (n/16)]
    float am = 0.f;
    for (int i = 0; i < nw; ++i) { float a = fabsf(W_h[i]); if (a > am) am = a; }
    float sv = am / 3.f;
    std::vector<float> W_scales_h(nb_scales_W, sv);
    cudaMemcpy(d_ws, W_scales_h.data(), nb_scales_W*4, cudaMemcpyHostToDevice);

    // pack_fp4 uses uniform scale (scale_in[0])
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

    size_t total = in_ + nw + nb_scales_x*4 + nb_scales_W*4 + out_*4;
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

// ===== GEMV v2 (vectorized with transposed weights) =====
// gemv_fp4_v2 requires transposed weight layout: W_t [N x K] row-major.
// Weights are transposed in-place on GPU (W -> W_t) then benchmarked.
double bench_gemv_v2(const float* W_h, int in_, int out_, int warm, int bench, double* gbps_out) {
    *gbps_out = 0;
    int nw = in_ * out_;
    int nb_scales_x = in_ / 16;
    int num_K_blks = in_ / 16;
    int num_N_blks = out_ / 16;
    int nb_scales_W = num_K_blks * num_N_blks;  // (K/16) x (N/16)

    float *d_W32, *d_x32, *d_y;
    void   *d_W4_orig, *d_W4_t;   // original + transposed
    float  *d_W4_scale_orig, *d_W4_scale_t;  // original + transposed scales
    void   *d_x4;
    float  *d_xs;

    cudaMalloc(&d_W32, nw*4); cudaMalloc(&d_x32, in_*4); cudaMalloc(&d_y, out_*4);
    cudaMalloc(&d_W4_orig, nw); cudaMalloc(&d_W4_t, nw);
    cudaMalloc(&d_W4_scale_orig, nb_scales_W*4); cudaMalloc(&d_W4_scale_t, nb_scales_W*4);
    cudaMalloc(&d_x4, in_); cudaMalloc(&d_xs, nb_scales_x*4);

    cudaMemcpy(d_W32, W_h, nw*4, cudaMemcpyHostToDevice);
    std::vector<float> x_h(in_, 1.f);
    cudaMemcpy(d_x32, x_h.data(), in_*4, cudaMemcpyHostToDevice);

    // W_scale: (K/16) x (N/16) row-major, uniform for all-1s input
    float am = 0.f;
    for (int i = 0; i < nw; ++i) { float a = fabsf(W_h[i]); if (a > am) am = a; }
    float sv = am / 3.f;
    std::vector<float> W_scales_h(nb_scales_W, sv);
    cudaMemcpy(d_W4_scale_orig, W_scales_h.data(), nb_scales_W*4, cudaMemcpyHostToDevice);

    if (!check(blackwell::kernels::pack_fp4(d_W4_orig, d_W32, d_W4_scale_orig, nw, 0), "pack_fp4(W)")) return 0;
    // Transpose weights: W (KxN) -> W_t (NxK)
    if (!check(blackwell::kernels::transpose_fp4_weights(d_W4_t, d_W4_scale_t, d_W4_orig, d_W4_scale_orig, in_, out_, 0), "transpose_fp4_weights")) return 0;

    // Pack x to FP4 with per-block scales
    float x_sv = 1.f / 3.f;
    std::vector<float> x_scales_h(nb_scales_x, x_sv);
    cudaMemcpy(d_xs, x_scales_h.data(), nb_scales_x*4, cudaMemcpyHostToDevice);
    if (!check(blackwell::kernels::pack_fp4(d_x4, d_x32, d_xs, in_, 0), "pack_fp4(x)")) return 0;

    for (int i = 0; i < warm; ++i)
        check(blackwell::kernels::gemv_fp4_v2(d_y, d_x4, d_xs, d_W4_t, d_W4_scale_t, in_, out_, 0), "gemv_fp4_v2(warm)");
    cudaDeviceSynchronize();

    GpuTimer t;
    t.begin();
    for (int i = 0; i < bench; ++i) {
        cudaError_t e = blackwell::kernels::gemv_fp4_v2(d_y, d_x4, d_xs, d_W4_t, d_W4_scale_t, in_, out_, 0);
        if (e != cudaSuccess) { printf("  FAIL iter %d: %s\n", i, cudaGetErrorString(e)); break; }
    }
    float ms = t.end() / bench;

    // Bandwidth: input(x) + weights(W) + scales + output(y)
    size_t total = in_ + nw + nb_scales_x*4 + nb_scales_W*4 + out_*4;
    *gbps_out = bw_gbps(total, ms);

    // CPU reference uses original (non-transposed) W layout
    std::vector<float> y_gpu(out_);
    cudaMemcpy(y_gpu.data(), d_y, out_*4, cudaMemcpyDeviceToHost);
    std::vector<float> y_cpu(out_);
    cpu_gemv_ref(W_h, x_h.data(), y_cpu.data(), in_, out_);
    float re = max_rel_err(y_cpu.data(), y_gpu.data(), out_);

    printf("| gemv_fp4_v2 | %s x %s | %.3f | %.1f | %.4e |\n",
           fmt(out_).c_str(), fmt(in_).c_str(), ms, *gbps_out, re);

    cudaFree(d_W32); cudaFree(d_x32); cudaFree(d_y);
    cudaFree(d_W4_orig); cudaFree(d_W4_t);
    cudaFree(d_W4_scale_orig); cudaFree(d_W4_scale_t);
    cudaFree(d_x4); cudaFree(d_xs);
    return ms;
}

double bench_gemv_splitk(const float* W_h, int in_, int out_, int K_splits, int warm, int bench, double* gbps_out) {
    *gbps_out = 0;
    int nw = in_ * out_;
    int num_K_blks = in_ / 16;
    int num_N_blks = out_ / 16;
    int nb_scales_W = num_K_blks * num_N_blks;
    int nb_scales_x = num_K_blks;

    float *d_y;  // output (zero-initialized for atomic adds)
    void *d_x4;
    float *d_xs, *d_W4_t, *d_W4_scale_t;

    cudaMalloc(&d_y, out_ * 4);
    cudaMalloc(&d_x4, in_);
    cudaMalloc(&d_xs, nb_scales_x * 4);
    cudaMalloc(&d_W4_t, nw);
    cudaMalloc(&d_W4_scale_t, nb_scales_W * 4);

    float x_sv = 1.f / 3.f;
    std::vector<float> x_scales_h(nb_scales_x, x_sv);
    std::vector<float> x32_h(in_, 1.f);
    cudaMemcpy(d_xs, x_scales_h.data(), nb_scales_x * 4, cudaMemcpyHostToDevice);
    if (!check(blackwell::kernels::pack_fp4(d_x4, x32_h.data(), x_scales_h.data(), in_, 0), "pack")) return 0;

    // Synthesize weight
    float *d_W4_orig, *d_W4_orig_v, *d_W4_scale_orig;
    cudaMalloc(&d_W4_orig, nw * 4);
    cudaMalloc(&d_W4_orig_v, nw);
    cudaMalloc(&d_W4_scale_orig, nb_scales_W * 4);
    std::vector<float> W32_h(nw, 1.f);
    cudaMemcpy(d_W4_orig, W32_h.data(), nw * 4, cudaMemcpyHostToDevice);
    std::vector<float> W_scales_h(nb_scales_W, 3.f / 3.f);
    cudaMemcpy(d_W4_scale_orig, W_scales_h.data(), nb_scales_W * 4, cudaMemcpyHostToDevice);
    if (!check(blackwell::kernels::pack_fp4(d_W4_orig_v, d_W4_orig, d_W4_scale_orig, nw, 0), "pack")) return 0;
    if (!check(blackwell::kernels::transpose_fp4_weights(d_W4_t, d_W4_scale_t, d_W4_orig_v, d_W4_scale_orig, in_, out_, 0), "transpose")) return 0;
    cudaFree(d_W4_orig); cudaFree(d_W4_orig_v); cudaFree(d_W4_scale_orig);

    for (int i = 0; i < warm; ++i) {
        cudaMemset(d_y, 0, out_ * 4);
        check(blackwell::kernels::gemv_fp4_splitk(d_y, d_x4, d_xs, d_W4_t, d_W4_scale_t, in_, out_, K_splits, 0), "gemv_splitk(warm)");
    }
    cudaDeviceSynchronize();

    GpuTimer t;
    t.begin();
    for (int i = 0; i < bench; ++i) {
        cudaMemset(d_y, 0, out_ * 4);
        cudaError_t e = blackwell::kernels::gemv_fp4_splitk(d_y, d_x4, d_xs, d_W4_t, d_W4_scale_t, in_, out_, K_splits, 0);
        if (e != cudaSuccess) { printf("  FAIL iter %d: %s\n", i, cudaGetErrorString(e)); break; }
    }
    float ms = t.end() / bench;

    size_t total = in_ + nw + nb_scales_x*4 + nb_scales_W*4 + out_*4;
    *gbps_out = bw_gbps(total, ms);

    std::vector<float> y_gpu(out_);
    cudaMemcpy(y_gpu.data(), d_y, out_ * 4, cudaMemcpyDeviceToHost);
    std::vector<float> y_cpu(out_);
    cpu_gemv_ref(W32_h.data(), x32_h.data(), y_cpu.data(), in_, out_);
    float re = max_rel_err(y_cpu.data(), y_gpu.data(), out_);

    printf("| gemv_fp4_splitk (K_splits=%d) | %s x %s | %.3f | %.1f | %.4e |\n",
           K_splits, fmt(out_).c_str(), fmt(in_).c_str(), ms, *gbps_out, re);

    cudaFree(d_y); cudaFree(d_x4); cudaFree(d_xs);
    cudaFree(d_W4_t); cudaFree(d_W4_scale_t);
    return ms;
}

// ===== GEMV v3 (shared memory tiled) =====
double bench_gemv_v3(const float* W_h, int in_, int out_, int warm, int bench, double* gbps_out) {
    *gbps_out = 0;
    int nw = in_ * out_;
    int nb_scales_x = in_ / 16;
    int num_K_blks = in_ / 16;
    int num_N_blks = out_ / 16;
    int nb_scales_W = num_K_blks * num_N_blks;

    float *d_W32, *d_x32, *d_y;
    void   *d_W4_orig, *d_W4_t;
    float  *d_W4_scale_orig, *d_W4_scale_t;
    void   *d_x4;
    float  *d_xs;

    cudaMalloc(&d_W32, nw*4); cudaMalloc(&d_x32, in_*4); cudaMalloc(&d_y, out_*4);
    cudaMalloc(&d_W4_orig, nw); cudaMalloc(&d_W4_t, nw);
    cudaMalloc(&d_W4_scale_orig, nb_scales_W*4); cudaMalloc(&d_W4_scale_t, nb_scales_W*4);
    cudaMalloc(&d_x4, in_); cudaMalloc(&d_xs, nb_scales_x*4);

    cudaMemcpy(d_W32, W_h, nw*4, cudaMemcpyHostToDevice);
    std::vector<float> x_h(in_, 1.f);
    cudaMemcpy(d_x32, x_h.data(), in_*4, cudaMemcpyHostToDevice);

    float am = 0.f;
    for (int i = 0; i < nw; ++i) { float a = fabsf(W_h[i]); if (a > am) am = a; }
    float sv = am / 3.f;
    std::vector<float> W_scales_h(nb_scales_W, sv);
    cudaMemcpy(d_W4_scale_orig, W_scales_h.data(), nb_scales_W*4, cudaMemcpyHostToDevice);

    if (!check(blackwell::kernels::pack_fp4(d_W4_orig, d_W32, d_W4_scale_orig, nw, 0), "pack_fp4(W)")) return 0;
    if (!check(blackwell::kernels::transpose_fp4_weights(d_W4_t, d_W4_scale_t, d_W4_orig, d_W4_scale_orig, in_, out_, 0), "transpose_fp4_weights")) return 0;

    float x_sv = 1.f / 3.f;
    std::vector<float> x_scales_h(nb_scales_x, x_sv);
    cudaMemcpy(d_xs, x_scales_h.data(), nb_scales_x*4, cudaMemcpyHostToDevice);
    if (!check(blackwell::kernels::pack_fp4(d_x4, d_x32, d_xs, in_, 0), "pack_fp4(x)")) return 0;

    for (int i = 0; i < warm; ++i)
        check(blackwell::kernels::gemv_fp4_v3(d_y, d_x4, d_xs, d_W4_t, d_W4_scale_t, in_, out_, 0), "gemv_fp4_v3(warm)");
    cudaDeviceSynchronize();

    GpuTimer t;
    t.begin();
    for (int i = 0; i < bench; ++i) {
        cudaError_t e = blackwell::kernels::gemv_fp4_v3(d_y, d_x4, d_xs, d_W4_t, d_W4_scale_t, in_, out_, 0);
        if (e != cudaSuccess) { printf("  FAIL iter %d: %s\n", i, cudaGetErrorString(e)); break; }
    }
    float ms = t.end() / bench;

    size_t total = in_ + nw + nb_scales_x*4 + nb_scales_W*4 + out_*4;
    *gbps_out = bw_gbps(total, ms);

    std::vector<float> y_gpu(out_);
    cudaMemcpy(y_gpu.data(), d_y, out_*4, cudaMemcpyDeviceToHost);
    std::vector<float> y_cpu(out_);
    cpu_gemv_ref(W_h, x_h.data(), y_cpu.data(), in_, out_);
    float re = max_rel_err(y_cpu.data(), y_gpu.data(), out_);

    printf("| gemv_fp4_v3 | %s x %s | %.3f | %.1f | %.4e |\n",
           fmt(out_).c_str(), fmt(in_).c_str(), ms, *gbps_out, re);

    cudaFree(d_W32); cudaFree(d_x32); cudaFree(d_y);
    cudaFree(d_W4_orig); cudaFree(d_W4_t);
    cudaFree(d_W4_scale_orig); cudaFree(d_W4_scale_t);
    cudaFree(d_x4); cudaFree(d_xs);
    return ms;
}

double bench_gemv_batched(const float* W_h, int in_, int out_, int M, int warm, int bench, double* gbps_out) {
    *gbps_out = 0;
    int nw = in_ * out_;
    int num_K_blks = in_ / 16;
    int num_N_blks = out_ / 16;
    int nb_scales_W = num_K_blks * num_N_blks;
    int nb_scales_x = num_K_blks;

    float *d_W32;
    void *d_W4_orig, *d_W4_t;
    float *d_W4_scale_orig, *d_W4_scale_t;

    // Batch inputs: M copies of x
    float *d_x32_batch, *d_x4_batch, *d_xs_batch, *d_y_batch;
    void *d_x4_ptr_arr[4];
    float *d_xs_ptr_arr[4];

    cudaMalloc(&d_W32, nw*4);
    cudaMalloc(&d_W4_orig, nw);
    cudaMalloc(&d_W4_t, nw);
    cudaMalloc(&d_W4_scale_orig, nb_scales_W*4);
    cudaMalloc(&d_W4_scale_t, nb_scales_W*4);

    cudaMalloc(&d_x32_batch, M * in_ * 4);
    cudaMalloc(&d_x4_batch, M * in_);
    cudaMalloc(&d_xs_batch, M * nb_scales_x * 4);
    cudaMalloc(&d_y_batch, M * out_ * 4);

    cudaMemcpy(d_W32, W_h, nw*4, cudaMemcpyHostToDevice);

    // Pack weights
    float am = 0.f;
    for (int i = 0; i < nw; ++i) { float a = fabsf(W_h[i]); if (a > am) am = a; }
    float sv = am / 3.f;
    std::vector<float> W_scales_h(nb_scales_W, sv);
    cudaMemcpy(d_W4_scale_orig, W_scales_h.data(), nb_scales_W*4, cudaMemcpyHostToDevice);
    if (!check(blackwell::kernels::pack_fp4(d_W4_orig, d_W32, d_W4_scale_orig, nw, 0), "pack_fp4(W)")) return 0;
    if (!check(blackwell::kernels::transpose_fp4_weights(d_W4_t, d_W4_scale_t, d_W4_orig, d_W4_scale_orig, in_, out_, 0), "transpose")) return 0;

    // Init M input vectors
    float x_sv = 1.f / 3.f;
    std::vector<float> x32_batch_h;
    x32_batch_h.reserve(M * in_);
    for (int m = 0; m < M; ++m) {
        for (int i = 0; i < in_; ++i) x32_batch_h.push_back(((i + m*37) % 17) / 16.f);
    }
    cudaMemcpy(d_x32_batch, x32_batch_h.data(), M*in_*4, cudaMemcpyHostToDevice);
    std::vector<float> xs_batch_h(M * nb_scales_x, x_sv);
    cudaMemcpy(d_xs_batch, xs_batch_h.data(), M*nb_scales_x*4, cudaMemcpyHostToDevice);

    // Pack all M x vectors (inline per-vector loop)
    for (int m = 0; m < M; ++m) {
        void* x4_ptr = (char*)d_x4_batch + m * in_;
        float* xs_ptr = d_xs_batch + m * nb_scales_x;
        const float* x32_ptr = d_x32_batch + m * in_;
        if (!check(blackwell::kernels::pack_fp4(x4_ptr, x32_ptr, xs_ptr, in_, 0), "pack_fp4(x)")) return 0;
    }

    for (int i = 0; i < warm; ++i)
        check(blackwell::kernels::gemv_fp4_batched(d_y_batch, d_x4_batch, d_xs_batch, d_W4_t, d_W4_scale_t, in_, out_, M, 0), "gemv_batched(warm)");
    cudaDeviceSynchronize();

    GpuTimer t;
    t.begin();
    for (int i = 0; i < bench; ++i) {
        cudaError_t e = blackwell::kernels::gemv_fp4_batched(d_y_batch, d_x4_batch, d_xs_batch, d_W4_t, d_W4_scale_t, in_, out_, M, 0);
        if (e != cudaSuccess) { printf("  FAIL iter %d: %s\n", i, cudaGetErrorString(e)); break; }
    }
    float ms = t.end() / bench;

    size_t total = M*in_ + nw + M*nb_scales_x*4 + nb_scales_W*4 + M*out_*4;
    *gbps_out = bw_gbps(total, ms);

    std::vector<float> y_gpu(M*out_);
    cudaMemcpy(y_gpu.data(), d_y_batch, M*out_*4, cudaMemcpyDeviceToHost);
    float max_re = 0.f;
    for (int m = 0; m < M; ++m) {
        std::vector<float> y_cpu(out_);
        cpu_gemv_ref(W_h, &x32_batch_h[m*in_], y_cpu.data(), in_, out_);
        float re = max_rel_err(y_cpu.data(), &y_gpu[m*out_], out_);
        if (re > max_re) max_re = re;
    }

    printf("| gemv_fp4_batched (M=%d) | %s x %s | %.3f | %.1f | %.4e |\n",
           M, fmt(out_).c_str(), fmt(in_).c_str(), ms, *gbps_out, max_re);

    cudaFree(d_W32); cudaFree(d_W4_orig); cudaFree(d_W4_t);
    cudaFree(d_W4_scale_orig); cudaFree(d_W4_scale_t);
    cudaFree(d_x32_batch); cudaFree(d_x4_batch);
    cudaFree(d_xs_batch); cudaFree(d_y_batch);
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

    // 2. GEMV — small K (legacy, backward-compat)
    printf("\n## 2a. FP4 GEMV (Decode Path) — K=64\n\n");
    printf("| Op | Shape (out x in) | Lat (ms) | GB/s | Rel Err |\n| --- | --- | --- | --- | --- |\n");
    int gemv_shapes_smallK[][2] = {{64,64},{128,64},{2048,64},{6144,64}};
    for (int i = 0; i < 4; ++i) {
        int o = gemv_shapes_smallK[i][0], k = gemv_shapes_smallK[i][1];
        std::vector<float> W(o * k, 1.f);
        double gb = 0;
        bench_gemv(W.data(), k, o, 10, 100, &gb);
    }

    // 2b. GEMV — dynamic K (real model hidden dim)
    printf("\n## 2b. FP4 GEMV (Decode Path) — Dynamic K\n\n");
    printf("| Op | Shape (out x in) | Lat (ms) | GB/s | Rel Err |\n| --- | --- | --- | --- | --- |\n");
    int gemv_shapes_largeK[][2] = {
        {64,2048}, {128,2048}, {2048,2048}, {6144,2048},
        {2048,4096}
    };
    for (int i = 0; i < 5; ++i) {
        int o = gemv_shapes_largeK[i][0], k = gemv_shapes_largeK[i][1];
        std::vector<float> W(o * k, 1.f);
        double gb = 0;
        bench_gemv(W.data(), k, o, 10, 100, &gb);
    }

    // 2c. GEMV v2 — vectorized with transposed weights
    printf("\n## 2c. FP4 GEMV v2 (Transposed + Vectorized) — Dynamic K\n\n");
    printf("| Op | Shape (out x in) | Lat (ms) | GB/s | Rel Err |\n| --- | --- | --- | --- | --- |\n");
    int gemv_v2_shapes[][2] = {
        {64,2048}, {128,2048}, {2048,2048}, {6144,2048},
        {2048,4096}
    };
    for (int i = 0; i < 5; ++i) {
        int o = gemv_v2_shapes[i][0], k = gemv_v2_shapes[i][1];
        std::vector<float> W(o * k, 1.f);
        double gb = 0;
        bench_gemv_v2(W.data(), k, o, 10, 100, &gb);
    }


    // 2c2. Split-K: N=6144 SM saturation test (K=2048, K_splits=2,4)
    printf("\n## 2c2. Split-K GEMV \u2014 N=6144 SM Saturation\n\n");
    printf("N=6144 \u2192 24 blocks \u2192 12 SMs idle. Splitting K=2048 into 2\u00d72=48 blocks.\n\n");
    printf("| Op | Shape (out x in) | Lat (ms) | GB/s | Rel Err \n| --- | --- | --- | --- | --- |\n");
    {
        int o = 6144, k = 2048;
        std::vector<float> W(o * k, 1.f);
        double gb = 0;
        bench_gemv_v2(W.data(), k, o, 10, 100, &gb);  // v2 baseline
        bench_gemv_splitk(W.data(), k, o, 2, 10, 100, &gb);  // K_splits=2
        bench_gemv_splitk(W.data(), k, o, 4, 10, 100, &gb);  // K_splits=4
    }
    // 2d. GEMV v3 — shared memory tiled (requires K multiple of 128, N multiple of 256)
    printf("\n## 2d. FP4 GEMV v3 (Shared Memory Tiled) — K=128 tile\n\n");
    printf("| Op | Shape (out x in) | Lat (ms) | GB/s | Rel Err |\n| --- | --- | --- | --- | --- |\n");
    int gemv_v3_shapes[][2] = {
        {256,2048}, {2048,2048}, {6144,2048},
        {2048,4096}
    };
    for (int i = 0; i < 4; ++i) {
        int o = gemv_v3_shapes[i][0], k = gemv_v3_shapes[i][1];
        std::vector<float> W(o * k, 1.f);
        double gb = 0;
        bench_gemv_v3(W.data(), k, o, 10, 100, &gb);
    }

    // 2e. GEMV Batched (M×GEMV v2)
    printf("\n## 2e. FP4 GEMV Batched (M×v2 simultaneous) — M=2,4\n\n");
    printf("| Op | Shape (out x in) | Lat (ms) | GB/s | Rel Err \n| --- | --- | --- | --- | --- |\n");
    int batched_shapes[][2] = {{6144, 2048}, {2048, 2048}};
    int batch_sizes[] = {2, 4};
    for (auto& s : batched_shapes) {
        int o = s[0], k = s[1];
        std::vector<float> W(o * k, 1.f);
        for (int Mi : batch_sizes) {
            double gb = 0;
            bench_gemv_batched(W.data(), k, o, Mi, 5, 100, &gb);
        }
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
