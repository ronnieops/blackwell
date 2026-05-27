// bench/verify_int8_pipeline.cu — End-to-end INT8 pipeline correctness check
//
// Loads FP4 weight file, converts to INT8 (same as convert_weights_int8),
// runs gemv_fp4_v2 and gemv_int8, compares outputs.
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120,code=sm_120 \
//     -I include bench/verify_int8_pipeline.cu build/libblackwell_kernels.a \
//     -o bench/verify_int8_pipeline

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>
#include "blackwell/kernels.h"

static void chk(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) { printf("FAIL: %s: %s\n", msg, cudaGetErrorString(e)); exit(1); }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <fp4_prefix> <int8_prefix>\n", argv[0]);
        return 1;
    }

    // Read FP4 header
    char path[256];
    snprintf(path,256,"%s.fp4",argv[1]);
    int h[5]; FILE* f = fopen(path,"rb"); fread(h,4,5,f); fclose(f);
    int K = h[0], N = h[1], nKb = h[3], nNb = h[4];
    size_t nw = (size_t)K * N;
    printf("Weights: K=%d N=%d nKb=%d nNb=%d\n", K, N, nKb, nNb);

    // Load FP4 data+scales
    std::vector<uint8_t> fp4_w(nw);
    std::vector<float> fp4_sc(nKb*nNb);
    f = fopen(path,"rb"); fseek(f,20,SEEK_SET);
    fread(fp4_w.data(),1,nw,f); fread(fp4_sc.data(),4,nKb*nNb,f); fclose(f);

    // GPU buffers
    void *d_w, *d_w_t;
    float *d_sc, *d_sc_t;
    cudaMalloc(&d_w, nw); cudaMalloc(&d_w_t, nw);
    cudaMalloc(&d_sc, nKb*nNb*4); cudaMalloc(&d_sc_t, nKb*nNb*4);
    cudaMemcpy(d_w, fp4_w.data(), nw, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sc, fp4_sc.data(), nKb*nNb*4, cudaMemcpyHostToDevice);
    chk(blackwell::kernels::transpose_fp4_weights(d_w_t, d_sc_t, d_w, d_sc, K, N, 0), "transpose_fp4");

    // Load INT8 transposed
    snprintf(path,256,"%s.int8_t",argv[2]);
    f = fopen(path,"rb"); fseek(f,20,SEEK_SET);
    std::vector<int8_t> i8_w_t(nw); fread(i8_w_t.data(),1,nw,f); fclose(f);
    snprintf(path,256,"%s.scale_t",argv[2]);
    f = fopen(path,"rb"); fseek(f,20,SEEK_SET);
    std::vector<float> i8_sc_t(nKb*nNb); fread(i8_sc_t.data(),4,nKb*nNb,f); fclose(f);

    int8_t *d_i8_t;
    float *d_i8_sc_t;
    cudaMalloc(&d_i8_t, nw); cudaMalloc(&d_i8_sc_t, nKb*nNb*4);
    cudaMemcpy(d_i8_t, i8_w_t.data(), nw, cudaMemcpyHostToDevice);
    cudaMemcpy(d_i8_sc_t, i8_sc_t.data(), nKb*nNb*4, cudaMemcpyHostToDevice);

    // FP4 input x
    void *d_x4; float *d_xs;
    cudaMalloc(&d_x4, K); cudaMalloc(&d_xs, (K/16)*4);
    float xs_val = 1.f/3.f; std::vector<float> xs_h(K/16, xs_val);
    cudaMemcpy(d_xs, xs_h.data(), (K/16)*4, cudaMemcpyHostToDevice);
    float *d_x32; cudaMalloc(&d_x32, K*4);
    std::vector<float> x32_h(K, 0.5f); cudaMemcpy(d_x32, x32_h.data(), K*4, cudaMemcpyHostToDevice);
    blackwell::kernels::pack_fp4(d_x4, d_x32, xs_h.data(), K, 0);

    // INT8 input x
    int8_t *d_i8_x; float *d_i8_xs;
    cudaMalloc(&d_i8_x, K); cudaMalloc(&d_i8_xs, (K/16)*4);
    float i8_xv = 0.5f/127.f; std::vector<float> i8_xs_h(K/16, i8_xv);
    cudaMemcpy(d_i8_xs, i8_xs_h.data(), (K/16)*4, cudaMemcpyHostToDevice);
    blackwell::kernels::pack_int8(d_i8_x, d_x32, d_i8_xs, K, 0);

    // Output buffers
    float *d_y4, *d_y8;
    cudaMalloc(&d_y4, N*4); cudaMalloc(&d_y8, N*4);

    // Run both GEMVs
    chk(blackwell::kernels::gemv_fp4_v2(d_y4, d_x4, d_xs, d_w_t, d_sc_t, K, N, 0), "gemv_fp4_v2");
    chk(blackwell::kernels::gemv_int8(d_y8, d_i8_x, d_i8_xs, d_i8_t, d_i8_sc_t, K, N, 0), "gemv_int8");
    cudaDeviceSynchronize();

    // Compare
    std::vector<float> y4(N), y8(N);
    cudaMemcpy(y4.data(), d_y4, N*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(y8.data(), d_y8, N*4, cudaMemcpyDeviceToHost);

    float y_max = 0.f;
    for (int i = 0; i < N; ++i) { float ay = fabsf(y4[i]); if (ay > y_max) y_max = ay; }
    float eps = fmaxf(y_max, 1e-6f);
    float max_e = 0.f, sum_e = 0.f;
    for (int i = 0; i < N; ++i) {
        float e = fabsf(y4[i] - y8[i]) / eps;
        if (e > max_e) max_e = e; sum_e += e;
    }

    printf("\n=== INT8 vs FP4 v2 Pipeline ===\n");
    printf("  y[0]: fp4=%.6f  int8=%.6f\n", y4[0], y8[0]);
    printf("  Max rel err: %.6e\n", max_e);
    printf("  Mean rel err: %.6e\n", sum_e/N);
    printf("  %s\n", max_e < 0.01f ? "PASS (err < 1%)" :
                     max_e < 0.1f  ? "WARN (err 1-10%)" :
                                     "FAIL (err > 10%)");

    cudaFree(d_w); cudaFree(d_w_t); cudaFree(d_sc); cudaFree(d_sc_t);
    cudaFree(d_i8_t); cudaFree(d_i8_sc_t);
    cudaFree(d_x4); cudaFree(d_xs); cudaFree(d_x32);
    cudaFree(d_i8_x); cudaFree(d_i8_xs);
    cudaFree(d_y4); cudaFree(d_y8);
    return 0;
}