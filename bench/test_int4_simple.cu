// bench/test_int4_simple.cu — Minimal INT4 warp GEMV smoke test
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include "blackwell/kernels.h"

int main() {
    printf("INT4 simple test\\n"); fflush(stdout);
    int K = 256, N = 256;
    int num_K_blks = K / 16;
    size_t packed_sz = (size_t)N * (K / 2);
    size_t scale_sz = (size_t)N * num_K_blks;

    // Create packed INT4 data (all zeros = signed value 0 after dequant)
    std::vector<uint8_t> w_packed(packed_sz, 0x00); // nibbles = 0 → signed 0
    std::vector<float> w_scales(scale_sz, 1.0f);
    std::vector<uint8_t> x_packed(K / 2, 0x00);
    std::vector<float> x_scales(num_K_blks, 1.0f);
    std::vector<float> y(N, -1.0f);

    uint8_t *d_wp, *d_xp;
    float *d_ws, *d_xs, *d_y;
    cudaMalloc(&d_wp, packed_sz);
    cudaMalloc(&d_ws, scale_sz * 4);
    cudaMalloc(&d_xp, K / 2);
    cudaMalloc(&d_xs, num_K_blks * 4);
    cudaMalloc(&d_y, N * 4);

    cudaMemcpy(d_wp, w_packed.data(), packed_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ws, w_scales.data(), scale_sz * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_xp, x_packed.data(), K / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_xs, x_scales.data(), num_K_blks * 4, cudaMemcpyHostToDevice);

    cudaError_t e = blackwell::kernels::gemv_int4_warp(d_y, d_xp, d_xs, d_wp, d_ws, K, N, 0);
    printf("gemv_int4_warp: %s\\n", cudaGetErrorString(e));
    cudaDeviceSynchronize();
    e = cudaGetLastError();
    printf("after sync: %s\\n", cudaGetErrorString(e));

    cudaMemcpy(y.data(), d_y, N * 4, cudaMemcpyDeviceToHost);
    printf("y[0..7] = ");
    for (int i = 0; i < 8; i++) printf("%.4f ", y[i]);
    printf("\\n");

    // All-zero packed nibbles (0x8 → signed value 0) → dot product = 0
    int zero_count = 0;
    for (int i = 0; i < N; i++) if (y[i] == 0.0f) zero_count++;
    printf("Zero outputs: %d / %d\\n", zero_count, N);

    cudaFree(d_wp); cudaFree(d_ws); cudaFree(d_xp); cudaFree(d_xs); cudaFree(d_y);
    printf("Done.\\n");
    return (zero_count == N) ? 0 : 1;
}
