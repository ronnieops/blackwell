// bench/test_nvfp4_gemv.cu — Test NVF4 GEMV kernel against scalar FP4 reference
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/test_nvfp4_gemv.cu build/libblackwell_kernels.a \
//     -o bench/test_nvfp4_gemv

#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <cstdint>
#include "blackwell/kernels.h"

struct GpuTimer {
    cudaEvent_t s, e;
    GpuTimer() { cudaEventCreate(&s); cudaEventCreate(&e); }
    ~GpuTimer() { cudaEventDestroy(s); cudaEventDestroy(e); }
    void start(cudaStream_t st = 0) { cudaEventRecord(s, st); }
    float stop(cudaStream_t st = 0) { 
        cudaEventRecord(e, st); 
        cudaEventSynchronize(e); 
        float ms = 0; 
        cudaEventElapsedTime(&ms, s, e); 
        return ms; 
    }
};

static void chk(cudaError_t e, const char* m) {
    if (e != cudaSuccess) { printf("FAIL %s: %s\n", m, cudaGetErrorString(e)); exit(1); }
}

// UE4M3 ↔ float conversion (host)
static float ue4m3_to_float_h(uint8_t v) {
    if (v == 0) return 0.0f;
    int exp = (v >> 3) & 0xF;
    int man = v & 0x7;
    if (exp == 0) return (man / 8.0f) * (1.0f / 64.0f);
    return (1.0f + man / 8.0f) * ldexpf(1.0f, exp - 7);
}

static uint8_t float_to_ue4m3_h(float val) {
    if (val <= 0) return 0;
    val = fminf(val, 448.0f);
    int exp_unbiased = (int)floorf(log2f(val));
    int exp_biased = exp_unbiased + 7;
    
    if (exp_biased <= 0) {
        int man = (int)roundf(val * 512.0f);
        return (uint8_t)(max(0, min(7, man)));
    }
    if (exp_biased >= 15) {
        if (exp_biased > 15) return (uint8_t)((15 << 3) | 6);
        float mf = val / ldexpf(1.0f, exp_unbiased) - 1.0f;
        int man = (int)roundf(mf * 8.0f);
        return (uint8_t)((exp_biased << 3) | max(0, min(6, man)));
    }
    float mf = val / ldexpf(1.0f, exp_unbiased) - 1.0f;
    int man = (int)roundf(mf * 8.0f);
    return (uint8_t)((exp_biased << 3) | max(0, min(7, man)));
}

// Scalar FP4 GEMV reference (uses __nv_fp4_e2m1 for proper conversion)
__global__ void gemv_fp4_scalar_ref(
    float* __restrict__ y_out,
    const __nv_fp4_e2m1* __restrict__ x_fp4,
    const float* __restrict__ x_scale,
    const __nv_fp4_e2m1* __restrict__ W_t_fp4,
    const float* __restrict__ W_t_scale,
    int K, int N) {
    
    constexpr int B = 16;
    int tid = threadIdx.x;
    int n_out = blockIdx.x * blockDim.x + tid;
    if (n_out >= N) return;
    
    int num_K_blks = K / B;
    int n_blk = n_out / B;
    
    float acc = 0.0f;
    for (int kb = 0; kb < num_K_blks; ++kb) {
        const __nv_fp4_e2m1* w_ptr = &W_t_fp4[n_out * K + kb * B];
        alignas(16) __nv_fp4_e2m1 w_buf[B], x_buf[B];
        *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);
        *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_fp4 + kb * B);
        
        float w_sc = W_t_scale[n_blk * num_K_blks + kb];
        float x_sc = x_scale[kb];
        float prod_sc = w_sc * x_sc;
        
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < B; ++j) {
            sum += static_cast<float>(x_buf[j]) * static_cast<float>(w_buf[j]);
        }
        acc += sum * prod_sc;
    }
    y_out[n_out] = acc;
}

int main(int argc, char** argv) {
    int K = 2048, N = 2048;
    if (argc > 1) K = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    
    printf("NVF4 GEMV Test: K=%d, N=%d\n", K, N);
    printf("Block size: 16, Scale format: UE4M3 (1 byte)\n\n");
    
    // Allocate host data
    int num_K_blks = K / 16;
    int num_N_blks = N / 16;
    
    using Fp4 = __nv_fp4_e2m1;
    std::vector<Fp4> h_x_fp4(K);
    std::vector<float> h_x_scale(K / 16);
    std::vector<uint8_t> h_x_scale_ue4m3(K / 16);
    std::vector<Fp4> h_W_fp4(N * K);
    std::vector<float> h_W_scale(num_N_blks * num_K_blks);
    std::vector<uint8_t> h_W_scale_ue4m3(num_N_blks * num_K_blks);
    
    // Initialize with random FP4 E2M1 values (0-15 raw encoding)
    srand(42);
    for (int i = 0; i < K; ++i) {
        uint8_t raw = rand() % 16;
        h_x_fp4[i] = *reinterpret_cast<Fp4*>(&raw);
    }
    for (int i = 0; i < N * K; ++i) {
        uint8_t raw = rand() % 16;
        h_W_fp4[i] = *reinterpret_cast<Fp4*>(&raw);
    }
    
    // Generate scales in FP32 range [0.01, 0.3]
    for (int i = 0; i < K / 16; ++i) {
        h_x_scale[i] = 0.01f + (rand() % 1000) / 10000.0f;
        h_x_scale_ue4m3[i] = float_to_ue4m3_h(h_x_scale[i]);
    }
    for (int i = 0; i < num_N_blks * num_K_blks; ++i) {
        h_W_scale[i] = 0.01f + (rand() % 1000) / 10000.0f;
        h_W_scale_ue4m3[i] = float_to_ue4m3_h(h_W_scale[i]);
    }
    
    printf("Scale conversion accuracy:\n");
    float max_rel_err = 0;
    for (int i = 0; i < K / 16; ++i) {
        float ref = h_x_scale[i];
        float recon = ue4m3_to_float_h(h_x_scale_ue4m3[i]);
        float rel_err = fabsf(recon - ref) / (ref + 1e-10f);
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }
    printf("  Max x_scale relative error: %.6f\n", max_rel_err);
    
    max_rel_err = 0;
    for (int i = 0; i < num_N_blks * num_K_blks; ++i) {
        float ref = h_W_scale[i];
        float recon = ue4m3_to_float_h(h_W_scale_ue4m3[i]);
        float rel_err = fabsf(recon - ref) / (ref + 1e-10f);
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }
    printf("  Max W_scale relative error: %.6f\n\n", max_rel_err);
    
    // Allocate device memory
    Fp4 *d_x_fp4, *d_W_fp4;
    uint8_t *d_x_scale_ue4m3, *d_W_scale_ue4m3;
    float *d_x_scale, *d_W_scale, *d_y_ref, *d_y_nvfp4;
    
    chk(cudaMalloc(&d_x_fp4, K * sizeof(Fp4)), "malloc x_fp4");
    chk(cudaMalloc(&d_x_scale, (K/16) * sizeof(float)), "malloc x_scale");
    chk(cudaMalloc(&d_x_scale_ue4m3, K/16), "malloc x_scale_ue4m3");
    chk(cudaMalloc(&d_W_fp4, N * K * sizeof(Fp4)), "malloc W_fp4");
    chk(cudaMalloc(&d_W_scale, (num_N_blks * num_K_blks) * sizeof(float)), "malloc W_scale");
    chk(cudaMalloc(&d_W_scale_ue4m3, num_N_blks * num_K_blks), "malloc W_scale_ue4m3");
    chk(cudaMalloc(&d_y_ref, N * sizeof(float)), "malloc y_ref");
    chk(cudaMalloc(&d_y_nvfp4, N * sizeof(float)), "malloc y_nvfp4");
    
    // Copy to device
    chk(cudaMemcpy(d_x_fp4, h_x_fp4.data(), K * sizeof(Fp4), cudaMemcpyHostToDevice), "copy x_fp4");
    chk(cudaMemcpy(d_x_scale, h_x_scale.data(), (K/16) * sizeof(float), cudaMemcpyHostToDevice), "copy x_scale");
    chk(cudaMemcpy(d_x_scale_ue4m3, h_x_scale_ue4m3.data(), K/16, cudaMemcpyHostToDevice), "copy x_scale_ue4m3");
    chk(cudaMemcpy(d_W_fp4, h_W_fp4.data(), N * K * sizeof(Fp4), cudaMemcpyHostToDevice), "copy W_fp4");
    chk(cudaMemcpy(d_W_scale, h_W_scale.data(), (num_N_blks * num_K_blks) * sizeof(float), cudaMemcpyHostToDevice), "copy W_scale");
    chk(cudaMemcpy(d_W_scale_ue4m3, h_W_scale_ue4m3.data(), num_N_blks * num_K_blks, cudaMemcpyHostToDevice), "copy W_scale_ue4m3");
    
    // Run reference kernel (scalar FP4 with FP32 scales)
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    gemv_fp4_scalar_ref<<<blocks, threads>>>(
        d_y_ref, d_x_fp4, d_x_scale, d_W_fp4, d_W_scale, K, N);
    chk(cudaGetLastError(), "ref kernel");
    
    // Run NVF4 kernel (scalar with UE4M3 scales)
    blackwell::kernels::gemv_fp4_nv(
        d_y_nvfp4, d_x_fp4, d_x_scale_ue4m3, d_W_fp4, d_W_scale_ue4m3, K, N, 0);
    chk(cudaGetLastError(), "nvfp4 kernel");
    
    // Copy results back
    std::vector<float> h_y_ref(N), h_y_nvfp4(N);
    chk(cudaMemcpy(h_y_ref.data(), d_y_ref, N * sizeof(float), cudaMemcpyDeviceToHost), "copy y_ref");
    chk(cudaMemcpy(h_y_nvfp4.data(), d_y_nvfp4, N * sizeof(float), cudaMemcpyDeviceToHost), "copy y_nvfp4");
    
    // Compare results
    float max_diff = 0;
    float sum_ref = 0, sum_diff = 0;
    for (int i = 0; i < N; ++i) {
        float diff = fabsf(h_y_ref[i] - h_y_nvfp4[i]);
        if (diff > max_diff) max_diff = diff;
        sum_ref += h_y_ref[i] * h_y_ref[i];
        sum_diff += diff * diff;
    }
    float cosine_sim = 1.0f - sum_diff / (sqrtf(sum_ref) * sqrtf(sum_diff + sum_ref));
    
    printf("Correctness check:\n");
    printf("  Max absolute diff: %.6e\n", max_diff);
    printf("  Cosine similarity: %.8f\n", cosine_sim);
    printf("  Result match: %s\n\n", (cosine_sim > 0.999f) ? "PASS" : "FAIL");
    
    // Benchmark
    int bench_iters = 100;
    GpuTimer timer;
    
    // Reference kernel
    timer.start();
    for (int i = 0; i < bench_iters; ++i) {
        gemv_fp4_scalar_ref<<<blocks, threads>>>(
            d_y_ref, d_x_fp4, d_x_scale, d_W_fp4, d_W_scale, K, N);
    }
    float ref_ms = timer.stop();
    
    // NVF4 kernel
    timer.start();
    for (int i = 0; i < bench_iters; ++i) {
        blackwell::kernels::gemv_fp4_nv(
            d_y_nvfp4, d_x_fp4, d_x_scale_ue4m3, d_W_fp4, d_W_scale_ue4m3, K, N, 0);
    }
    float nv_ms = timer.stop();
    
    // Calculate bandwidth
    // Read: x[K] + W[N*K] + x_scale[K/16] + W_scale[N*K/16]
    size_t bytes_read = K * sizeof(Fp4) + (size_t)N * K * sizeof(Fp4) + K/16 * sizeof(float) + (size_t)(N/16) * (K/16) * sizeof(float);
    size_t bytes_read_nv = K * sizeof(Fp4) + (size_t)N * K * sizeof(Fp4) + K/16 + (size_t)(N/16) * (K/16);
    // Write: y[N]
    size_t bytes_write = N * sizeof(float);
    
    float ref_gb_s = (float)(bytes_read + bytes_write) * bench_iters / (ref_ms / 1000.0f) / 1e9f;
    float nv_gb_s = (float)(bytes_read_nv + bytes_write) * bench_iters / (nv_ms / 1000.0f) / 1e9f;
    
    printf("Benchmark (%d iterations):\n", bench_iters);
    printf("  Reference (scalar FP4 + FP32 scales): %.3f ms = %.1f GB/s\n", ref_ms / bench_iters, ref_gb_s);
    printf("  NVF4 (scalar FP4 + UE4M3 scales):     %.3f ms = %.1f GB/s\n", nv_ms / bench_iters, nv_gb_s);
    printf("  Speedup: %.2fx\n", ref_ms / nv_ms);
    
    // Cleanup
    cudaFree(d_x_fp4);
    cudaFree(d_x_scale);
    cudaFree(d_x_scale_ue4m3);
    cudaFree(d_W_fp4);
    cudaFree(d_W_scale);
    cudaFree(d_W_scale_ue4m3);
    cudaFree(d_y_ref);
    cudaFree(d_y_nvfp4);
    
    return 0;
}
