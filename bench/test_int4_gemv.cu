// bench/test_int4_gemv.cu — Test INT4 warp GEMV correctness
//
// Compares INT4 packed GEMV output vs INT8 GEMV output on same weights.
// INT4 is a quantized approximation of INT8, so outputs won't match exactly,
// but cosine similarity should be high (>0.99).
//
// Build:
//   CUDACXX=/usr/local/cuda-13.3/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/test_int4_gemv.cu build/libblackwell_kernels.a \
//     -o bench/test_int4_gemv

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include <cstdint>
#include "blackwell/kernels.h"

static void chk(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) { fprintf(stderr, "FAIL: %s: %s\n", msg, cudaGetErrorString(e)); exit(1); }
}

int main() {
    printf("# INT4 Warp GEMV Correctness Test\n\n");
    fflush(stdout);

    // Test with layer 0 q_proj: K=2048, N=2048
    const char* prefix = "weights_int8_bf16/0_self_attn.q_proj";

    // Load INT8 weights
    char p[256];
    snprintf(p, 256, "%s.int8_t", prefix);
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", p); return 1; }
    int h[5]; fread(h, 4, 5, f);
    int K = h[0], N = h[1];
    printf("INT8 Weight: K=%d N=%d\n", K, N);
    std::vector<int8_t> i8_data((size_t)K * N);
    fread(i8_data.data(), 1, i8_data.size(), f);
    fclose(f);

    snprintf(p, 256, "%s.scale_t", prefix);
    f = fopen(p, "rb");
    fread(h, 4, 5, f);
    // Header: {K, N, block, num_K_blks, N} — h[3] = num_K_blks, h[4] = N
    int num_K_blks = h[3];
    printf("INT8 scales: h[3]=%d h[4]=%d → num_K_blks=%d\n", h[3], h[4], num_K_blks);
    std::vector<float> i8_scales((size_t)h[4] * num_K_blks);  // N * num_K_blks
    fread(i8_scales.data(), 4, i8_scales.size(), f);
    fclose(f);

    // Load INT4 weights
    snprintf(p, 256, "weights_int4_packed/0_self_attn.q_proj.int4_packed");
    f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", p); return 1; }
    fread(h, 4, 5, f);
    printf("INT4 header: K=%d N=%d block=%d nKb=%d nNb=%d\n", h[0], h[1], h[2], h[3], h[4]);
    size_t packed_size = (size_t)N * (K / 2);
    printf("INT4 packed_size: %zu bytes\n", packed_size);
    std::vector<uint8_t> i4_packed(packed_size);
    fread(i4_packed.data(), 1, packed_size, f);
    fclose(f);

    snprintf(p, 256, "weights_int4_packed/0_self_attn.q_proj.scale");
    f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", p); return 1; }
    fread(h, 4, 5, f);
    // INT4 scale file: same header as INT8
    size_t scale_count = (size_t)h[4] * h[3];  // N * num_K_blks
    printf("INT4 scales: h[3]=%d h[4]=%d → %zu floats\n", h[3], h[4], scale_count);
    std::vector<float> i4_scales(scale_count);
    fread(i4_scales.data(), 4, i4_scales.size(), f);
    fclose(f);

    // Create random INT8 activation vector
    std::vector<int8_t> x_i8(K);
    std::vector<float> x_scales(num_K_blks);
    srand(42);
    for (int i = 0; i < K; i++) x_i8[i] = (int8_t)(rand() % 256 - 128);
    // Compute activation scales (block_size=16)
    for (int kb = 0; kb < num_K_blks; kb++) {
        float block_max = 0;
        for (int j = 0; j < 16; j++) {
            float val = fabsf((float)x_i8[kb * 16 + j]);
            if (val > block_max) block_max = val;
        }
        x_scales[kb] = (block_max > 1e-10f) ? block_max / 127.f : 1.f / 127.f;
    }

    // Pack activation to INT4
    std::vector<uint8_t> x_i4_packed(K / 2);
    for (int kb = 0; kb < num_K_blks; kb++) {
        for (int j = 0; j < 8; j++) {
            int idx0 = kb * 16 + j * 2;
            int idx1 = kb * 16 + j * 2 + 1;
            float sc = x_scales[kb];
            int q0 = (sc > 1e-10f) ? (int)lroundf((float)x_i8[idx0] / sc) : 0;
            int q1 = (sc > 1e-10f) ? (int)lroundf((float)x_i8[idx1] / sc) : 0;
            q0 = max(-8, min(7, q0));
            q1 = max(-8, min(7, q1));
            uint8_t n0 = (uint8_t)((q0 + 16) & 0x0F);
            uint8_t n1 = (uint8_t)((q1 + 16) & 0x0F);
            x_i4_packed[kb * 8 + j] = (n0 & 0x0F) | ((n1 & 0x0F) << 4);
        }
    }

    // Upload to GPU
    int8_t* d_i8; float* d_i8_sc;
    uint8_t* d_i4; float* d_i4_sc;
    int8_t* d_x_i8; float* d_x_i8_sc;
    uint8_t* d_x_i4; float* d_x_i4_sc;
    float* d_y_i8; float* d_y_i4;

    chk(cudaMalloc(&d_i8, K * N), "malloc i8");
    chk(cudaMalloc(&d_i8_sc, N * num_K_blks * 4), "malloc i8_sc");
    chk(cudaMalloc(&d_i4, packed_size), "malloc i4");
    chk(cudaMalloc(&d_i4_sc, N * num_K_blks * 4), "malloc i4_sc");
    chk(cudaMalloc(&d_x_i8, K), "malloc x_i8");
    chk(cudaMalloc(&d_x_i8_sc, num_K_blks * 4), "malloc x_i8_sc");
    chk(cudaMalloc(&d_x_i4, K / 2), "malloc x_i4");
    chk(cudaMalloc(&d_x_i4_sc, num_K_blks * 4), "malloc x_i4_sc");
    chk(cudaMalloc(&d_y_i8, N * 4), "malloc y_i8");
    chk(cudaMalloc(&d_y_i4, N * 4), "malloc y_i4");

    chk(cudaMemcpy(d_i8, i8_data.data(), K * N, cudaMemcpyHostToDevice), "cpy i8");
    chk(cudaMemcpy(d_i8_sc, i8_scales.data(), N * num_K_blks * 4, cudaMemcpyHostToDevice), "cpy i8_sc");
    chk(cudaMemcpy(d_i4, i4_packed.data(), packed_size, cudaMemcpyHostToDevice), "cpy i4");
    chk(cudaMemcpy(d_i4_sc, i4_scales.data(), N * num_K_blks * 4, cudaMemcpyHostToDevice), "cpy i4_sc");
    chk(cudaMemcpy(d_x_i8, x_i8.data(), K, cudaMemcpyHostToDevice), "cpy x_i8");
    chk(cudaMemcpy(d_x_i8_sc, x_scales.data(), num_K_blks * 4, cudaMemcpyHostToDevice), "cpy x_i8_sc");
    chk(cudaMemcpy(d_x_i4, x_i4_packed.data(), K / 2, cudaMemcpyHostToDevice), "cpy x_i4");
    chk(cudaMemcpy(d_x_i4_sc, x_scales.data(), num_K_blks * 4, cudaMemcpyHostToDevice), "cpy x_i4_sc");

    // Run INT8 GEMV (reference)
    printf("\nRunning INT8 warp GEMV...\n");
    chk(blackwell::kernels::gemv_int8_warp(d_y_i8, d_x_i8, d_x_i8_sc, d_i8, d_i8_sc, K, N, 0), "gemv_int8_warp");
    cudaDeviceSynchronize();

    // Run INT4 GEMV
    printf("Running INT4 warp GEMV...\n");
    chk(blackwell::kernels::gemv_int4_warp(d_y_i4, d_x_i4, d_x_i4_sc, d_i4, d_i4_sc, K, N, 0), "gemv_int4_warp");
    cudaDeviceSynchronize();

    // Download results
    std::vector<float> y_i8(N), y_i4(N);
    chk(cudaMemcpy(y_i8.data(), d_y_i8, N * 4, cudaMemcpyDeviceToHost), "dwn y_i8");
    chk(cudaMemcpy(y_i4.data(), d_y_i4, N * 4, cudaMemcpyDeviceToHost), "dwn y_i4");

    // Compare
    float l1 = 0, l2 = 0;
    float dot = 0, norm_i8 = 0, norm_i4 = 0;
    for (int i = 0; i < N; i++) {
        float diff = y_i8[i] - y_i4[i];
        l1 += fabsf(diff);
        l2 += diff * diff;
        dot += y_i8[i] * y_i4[i];
        norm_i8 += y_i8[i] * y_i8[i];
        norm_i4 += y_i4[i] * y_i4[i];
    }
    l1 /= N;
    l2 = sqrtf(l2 / N);
    float cosine = dot / (sqrtf(norm_i8) * sqrtf(norm_i4) + 1e-10f);

    printf("\n=== Results ===\n");
    printf("L1 error:  %.6f\n", l1);
    printf("L2 error:  %.6f\n", l2);
    printf("Cosine:    %.6f\n", cosine);
    printf("\nINT8 y[0..7]: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", y_i8[i]);
    printf("\nINT4 y[0..7]: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", y_i4[i]);
    printf("\n");

    // Benchmark
    printf("\nBenchmarking (1000 iterations)...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // INT8
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++)
        blackwell::kernels::gemv_int8_warp(d_y_i8, d_x_i8, d_x_i8_sc, d_i8, d_i8_sc, K, N, 0);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms_i8; cudaEventElapsedTime(&ms_i8, start, stop);
    printf("INT8: %.3f ms total, %.3f us per GEMV\n", ms_i8, ms_i8 * 1000 / 1000);

    // INT4
    cudaEventRecord(start);
    for (int i = 0; i < 1000; i++)
        blackwell::kernels::gemv_int4_warp(d_y_i4, d_x_i4, d_x_i4_sc, d_i4, d_i4_sc, K, N, 0);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms_i4; cudaEventElapsedTime(&ms_i4, start, stop);
    printf("INT4: %.3f ms total, %.3f us per GEMV\n", ms_i4, ms_i4 * 1000 / 1000);
    printf("INT4/INT8 speedup: %.2fx\n", ms_i8 / ms_i4);

    // Cleanup
    cudaFree(d_i8); cudaFree(d_i8_sc);
    cudaFree(d_i4); cudaFree(d_i4_sc);
    cudaFree(d_x_i8); cudaFree(d_x_i8_sc);
    cudaFree(d_x_i4); cudaFree(d_x_i4_sc);
    cudaFree(d_y_i8); cudaFree(d_y_i4);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    printf("\nDone.\n");
    return (cosine > 0.99f) ? 0 : 1;
}
