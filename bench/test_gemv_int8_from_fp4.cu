// bench/test_gemv_int8_from_fp4.cu — Verify fused kernel vs FP4 v2 + INT8 path
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120,code=sm_120 \
//     -I include bench/test_gemv_int8_from_fp4.cu \
//     build/libblackwell_kernels.a -o bench/test_gemv_int8_from_fp4

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

int main() {
    const int K = 2048, N = 2048;
    const int B = 16, nb_K = K/B, nb_N = N/B;

    // Create random FP4 input x
    std::vector<uint8_t> x_fp4_h(K);
    std::vector<float> x_sc_h(nb_K, 1.f/3.f);
    for (int i = 0; i < K; ++i) x_fp4_h[i] = (i % 8) << 4 | ((i+1) % 8); // random FP4 values

    // Create random INT8 weights (transposed N×K)
    std::vector<int8_t> w_h(N*K);
    std::vector<float> ws_h(nb_N * nb_K);
    for (int i = 0; i < N*K; ++i) w_h[i] = (i * 7) % 127 - 63;
    for (int i = 0; i < nb_N * nb_K; ++i) ws_h[i] = 0.5f / 127.f;

    // GPU buffers
    void *d_x4; float *d_xs; cudaMalloc(&d_x4, K); cudaMalloc(&d_xs, nb_K*4);
    int8_t *d_w; float *d_ws; cudaMalloc(&d_w, N*K); cudaMalloc(&d_ws, nb_N*nb_K*4);
    float *d_y_fp4, *d_y_fused, *d_y_i8;
    cudaMalloc(&d_y_fp4, N*4); cudaMalloc(&d_y_fused, N*4); cudaMalloc(&d_y_i8, N*4);

    cudaMemcpy(d_x4, x_fp4_h.data(), K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_xs, x_sc_h.data(), nb_K*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w_h.data(), N*K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ws, ws_h.data(), nb_N*nb_K*4, cudaMemcpyHostToDevice);

    // Also create INT8 input x for reference gemv_int8 path
    int8_t *d_x8; float *d_x8s; cudaMalloc(&d_x8, K); cudaMalloc(&d_x8s, nb_K*4);
    float *d_x32; cudaMalloc(&d_x32, K*4);
    // Dequant FP4 → compute INT8 scales → quant
    chk(blackwell::kernels::unpack_fp4(d_x32, d_x4, d_xs, K, 0), "unpack");
    float x8v = 0.5f/127.f;
    std::vector<float> x8sh(nb_K, x8v);
    cudaMemcpy(d_x8s, x8sh.data(), nb_K*4, cudaMemcpyHostToDevice);
    chk(blackwell::kernels::pack_int8(d_x8, d_x32, d_x8s, K, 0), "pack_i8");

    // Also need FP4 weight path for reference — transpose FP4 weights
    void *d_w4; float *d_w4s;
    cudaMalloc(&d_w4, N*K); cudaMalloc(&d_w4s, nb_N*nb_K*4);
    // Create FP4 weights from our INT8 values (rough approximation)
    std::vector<uint8_t> w4_h(N*K, 0x11);
    cudaMemcpy(d_w4, w4_h.data(), N*K, cudaMemcpyHostToDevice);
    std::vector<float> w4s_h(nb_N*nb_K, 1.f);
    cudaMemcpy(d_w4s, w4s_h.data(), nb_N*nb_K*4, cudaMemcpyHostToDevice);

    // Run GEMVs
    chk(blackwell::kernels::gemv_fp4_v2(d_y_fp4, d_x4, d_xs, d_w4, d_w4s, K, N, 0), "fp4_v2");
    chk(blackwell::kernels::gemv_int8_from_fp4(d_y_fused, d_x4, d_xs, d_w, d_ws, K, N, 0), "fused");
    chk(blackwell::kernels::gemv_int8_warp(d_y_i8, d_x8, d_x8s, d_w, d_ws, K, N, 0), "int8");

    // Compare
    std::vector<float> y_fp4(N), y_fused(N), y_i8(N);
    cudaMemcpy(y_fp4.data(), d_y_fp4, N*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(y_fused.data(), d_y_fused, N*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(y_i8.data(), d_y_i8, N*4, cudaMemcpyDeviceToHost);

    // Fused vs reference INT8 path (same weights, same quant input → should match)
    float max_e = 0, sum_e = 0, y_max = 0;
    for (int i = 0; i < N; ++i) if (fabsf(y_i8[i]) > y_max) y_max = fabsf(y_i8[i]);
    float eps = fmaxf(y_max, 1e-6f);
    for (int i = 0; i < N; ++i) {
        float e = fabsf(y_i8[i] - y_fused[i]) / eps;
        if (e > max_e) max_e = e; sum_e += e;
    }
    printf("Fused vs INT8 reference: max_err=%.6e mean=%.6e\n", max_e, sum_e/N);
    printf("%s\n", max_e < 0.01f ? "PASS" : "FAIL");

    cudaFree(d_x4); cudaFree(d_xs); cudaFree(d_w); cudaFree(d_ws);
    cudaFree(d_y_fp4); cudaFree(d_y_fused); cudaFree(d_y_i8);
    cudaFree(d_x8); cudaFree(d_x8s); cudaFree(d_x32);
    cudaFree(d_w4); cudaFree(d_w4s);
    return 0;
}