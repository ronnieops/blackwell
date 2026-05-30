// bench/test_int4_real.cu — Test INT4 GEMV with real weights
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include "blackwell/kernels.h"

static void chk(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) { fprintf(stderr, "FAIL: %s: %s\n", msg, cudaGetErrorString(e)); exit(1); }
}

// Reference: dequant INT8 → FP32 and compute dot product on CPU
static float cpu_gemv_ref(
    const int8_t* x_i8, const float* x_sc,
    const int8_t* W_i8, const float* W_sc,
    int K, int N, int n_out)
{
    int num_K_blks = K / 16;
    float acc = 0.0f;
    for (int kb = 0; kb < num_K_blks; kb++) {
        float w_sc = W_sc[n_out * num_K_blks + kb];
        float x_s = x_sc[kb];
        for (int j = 0; j < 16; j++) {
            float w_val = (float)W_i8[n_out * K + kb * 16 + j] * w_sc;
            float x_val = (float)x_i8[kb * 16 + j] * x_s;
            acc += w_val * x_val;
        }
    }
    return acc;
}

int main() {
    printf("# INT4 GEMV Real Weights Test\n\n");

    const char* prefix = "weights_int8_bf16/0_self_attn.q_proj";

    // Load INT8 weights (reference)
    char p[256];
    snprintf(p, 256, "%s.int8_t", prefix);
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", p); return 1; }
    int h[5]; fread(h, 4, 5, f);
    int K = h[0], N = h[1];
    printf("K=%d N=%d\n", K, N);
    std::vector<int8_t> i8_w(K * N);
    fread(i8_w.data(), 1, i8_w.size(), f);
    fclose(f);

    snprintf(p, 256, "%s.scale_t", prefix);
    f = fopen(p, "rb");
    fread(h, 4, 5, f);
    // Header: {K, N, block, num_K_blks, N}
    int num_K_blks = h[3];
    printf("num_K_blks=%d\n", num_K_blks);
    std::vector<float> i8_sc((size_t)h[4] * num_K_blks);
    fread(i8_sc.data(), 4, i8_sc.size(), f);
    fclose(f);

    // Load INT4 weights
    snprintf(p, 256, "weights_int4_packed/0_self_attn.q_proj.int4_packed");
    f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", p); return 1; }
    fread(h, 4, 5, f);
    size_t packed_sz = (size_t)N * (K / 2);
    std::vector<uint8_t> i4_w(packed_sz);
    fread(i4_w.data(), 1, packed_sz, f);
    fclose(f);

    snprintf(p, 256, "weights_int4_packed/0_self_attn.q_proj.scale");
    f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", p); return 1; }
    fread(h, 4, 5, f);
    std::vector<float> i4_sc(h[3] * h[4]);
    fread(i4_sc.data(), 4, i4_sc.size(), f);
    fclose(f);

    // Create INT8 activation (constant 1 for simplicity)
    std::vector<int8_t> x_i8(K, 1);
    std::vector<float> x_sc(num_K_blks, 1.0f / 127.0f); // scale so dequant → ~0.00787

    // Pack to INT4
    std::vector<uint8_t> x_i4(K / 2);
    for (int kb = 0; kb < num_K_blks; kb++) {
        for (int j = 0; j < 8; j++) {
            // All activations are 1, so INT4 quantized = 1 (since scale = 1/127)
            // Actually, x_i8[idx] = 1, dequant = 1 * (1/127) ≈ 0.00787
            // For INT4, we need to quantize the raw INT8 value, not the dequant value
            // x_i8[idx] = 1, INT4 scale = 1/7 (absmax=1), so quantized = round(1 / (1/7)) = 7
            // But we want the dequant to match INT8: 7 * (1/7) = 1.0 vs 1 * (1/127) = 0.00787
            // These don't match because INT4 and INT8 have different scale conventions

            // For fair comparison, quantize INT8 values to INT4 directly
            int idx0 = kb * 16 + j * 2;
            int idx1 = kb * 16 + j * 2 + 1;
            int v0 = x_i8[idx0]; // = 1
            int v1 = x_i8[idx1]; // = 1
            // INT4 scale = max(|v|) / 7 = 1/7
            int q0 = (v0 > 0) ? 7 : (v0 < 0) ? -8 : 0; // round(1 / (1/7)) = 7
            int q1 = (v1 > 0) ? 7 : (v1 < 0) ? -8 : 0;
            uint8_t n0 = (uint8_t)((q0 + 16) & 0x0F);
            uint8_t n1 = (uint8_t)((q1 + 16) & 0x0F);
            x_i4[kb * 8 + j] = (n0 & 0x0F) | ((n1 & 0x0F) << 4);
        }
    }
    // INT4 activation scale = 1/7
    std::vector<float> x_i4_sc(num_K_blks, 1.0f / 7.0f);

    // Upload
    int8_t *d_i8_w; float *d_i8_sc;
    uint8_t *d_i4_w; float *d_i4_sc;
    int8_t *d_x_i8; float *d_x_i8_sc;
    uint8_t *d_x_i4; float *d_x_i4_sc;
    float *d_y_i8; float *d_y_i4;

    chk(cudaMalloc(&d_i8_w, (size_t)K * N), "i8_w");
    chk(cudaMalloc(&d_i8_sc, (size_t)N * num_K_blks * 4), "i8_sc");
    chk(cudaMalloc(&d_i4_w, packed_sz), "i4_w");
    chk(cudaMalloc(&d_i4_sc, (size_t)i4_sc.size() * 4), "i4_sc");
    chk(cudaMalloc(&d_x_i8, K), "x_i8");
    chk(cudaMalloc(&d_x_i8_sc, num_K_blks * 4), "x_i8_sc");
    chk(cudaMalloc(&d_x_i4, K / 2), "x_i4");
    chk(cudaMalloc(&d_x_i4_sc, num_K_blks * 4), "x_i4_sc");
    chk(cudaMalloc(&d_y_i8, N * 4), "y_i8");
    chk(cudaMalloc(&d_y_i4, N * 4), "y_i4");

    chk(cudaMemcpy(d_i8_w, i8_w.data(), (size_t)K * N, cudaMemcpyHostToDevice), "cpy");
    chk(cudaMemcpy(d_i8_sc, i8_sc.data(), (size_t)i8_sc.size() * 4, cudaMemcpyHostToDevice), "cpy");
    chk(cudaMemcpy(d_i4_w, i4_w.data(), packed_sz, cudaMemcpyHostToDevice), "cpy");
    chk(cudaMemcpy(d_i4_sc, i4_sc.data(), (size_t)i4_sc.size() * 4, cudaMemcpyHostToDevice), "cpy");
    chk(cudaMemcpy(d_x_i8, x_i8.data(), K, cudaMemcpyHostToDevice), "cpy");
    chk(cudaMemcpy(d_x_i8_sc, x_sc.data(), num_K_blks * 4, cudaMemcpyHostToDevice), "cpy");
    chk(cudaMemcpy(d_x_i4, x_i4.data(), K / 2, cudaMemcpyHostToDevice), "cpy");
    chk(cudaMemcpy(d_x_i4_sc, x_i4_sc.data(), num_K_blks * 4, cudaMemcpyHostToDevice), "cpy");

    // INT8 GEMV
    chk(blackwell::kernels::gemv_int8_warp(d_y_i8, d_x_i8, d_x_i8_sc, d_i8_w, d_i8_sc, K, N, 0), "gemv_int8");
    cudaDeviceSynchronize();

    // INT4 GEMV
    chk(blackwell::kernels::gemv_int4_warp(d_y_i4, d_x_i4, d_x_i4_sc, d_i4_w, d_i4_sc, K, N, 0), "gemv_int4");
    cudaDeviceSynchronize();

    // Download
    std::vector<float> y_i8(N), y_i4(N);
    chk(cudaMemcpy(y_i8.data(), d_y_i8, N * 4, cudaMemcpyDeviceToHost), "dwn");
    chk(cudaMemcpy(y_i4.data(), d_y_i4, N * 4, cudaMemcpyDeviceToHost), "dwn");

    // CPU reference
    std::vector<float> y_ref(N);
    for (int n = 0; n < N; n++)
        y_ref[n] = cpu_gemv_ref(x_i8.data(), x_sc.data(), i8_w.data(), i8_sc.data(), K, N, n);

    // Compare INT8 GPU vs CPU reference
    float max_err_i8 = 0;
    for (int n = 0; n < N; n++) {
        float err = fabsf(y_i8[n] - y_ref[n]);
        if (err > max_err_i8) max_err_i8 = err;
    }
    printf("INT8 GPU vs CPU ref: max_err = %.6f\n", max_err_i8);

    // Compare INT4 vs INT8 (quantization difference expected)
    float dot = 0, norm_i8 = 0, norm_i4 = 0;
    float l1 = 0;
    for (int n = 0; n < N; n++) {
        float diff = y_i8[n] - y_i4[n];
        l1 += fabsf(diff);
        dot += y_i8[n] * y_i4[n];
        norm_i8 += y_i8[n] * y_i8[n];
        norm_i4 += y_i4[n] * y_i4[n];
    }
    float cosine = dot / (sqrtf(norm_i8) * sqrtf(norm_i4) + 1e-10f);
    printf("INT4 vs INT8: L1=%.4f cosine=%.6f\n", l1 / N, cosine);
    printf("INT8 y[0..3]: %.6f %.6f %.6f %.6f\n", y_i8[0], y_i8[1], y_i8[2], y_i8[3]);
    printf("INT4 y[0..3]: %.6f %.6f %.6f %.6f\n", y_i4[0], y_i4[1], y_i4[2], y_i4[3]);
    printf("REF y[0..3]:  %.6f %.6f %.6f %.6f\n", y_ref[0], y_ref[1], y_ref[2], y_ref[3]);

    // Benchmark
    printf("\nBenchmark (1000 iters)...\n");
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < 1000; i++)
        blackwell::kernels::gemv_int8_warp(d_y_i8, d_x_i8, d_x_i8_sc, d_i8_w, d_i8_sc, K, N, 0);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms_i8; cudaEventElapsedTime(&ms_i8, s, e);

    cudaEventRecord(s);
    for (int i = 0; i < 1000; i++)
        blackwell::kernels::gemv_int4_warp(d_y_i4, d_x_i4, d_x_i4_sc, d_i4_w, d_i4_sc, K, N, 0);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms_i4; cudaEventElapsedTime(&ms_i4, s, e);

    printf("INT8: %.3f us/GEMV\n", ms_i8 * 1000 / 1000);
    printf("INT4: %.3f us/GEMV\n", ms_i4 * 1000 / 1000);
    printf("Speedup: %.2fx\n", ms_i8 / ms_i4);

    cudaFree(d_i8_w); cudaFree(d_i8_sc);
    cudaFree(d_i4_w); cudaFree(d_i4_sc);
    cudaFree(d_x_i8); cudaFree(d_x_i8_sc);
    cudaFree(d_x_i4); cudaFree(d_x_i4_sc);
    cudaFree(d_y_i8); cudaFree(d_y_i4);
    cudaEventDestroy(s); cudaEventDestroy(e);

    printf("\nDone.\n");
    return (cosine > 0.95f) ? 0 : 1;
}
