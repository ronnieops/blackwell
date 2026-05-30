// bench/bench_fused_unpack_pack.cu — Compare unpack_fp4+pack_int8 vs fused
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <cstring>
#include "blackwell/kernels.h"

struct GpuTimer {
    cudaEvent_t s, e;
    GpuTimer() { cudaEventCreate(&s); cudaEventCreate(&e); }
    ~GpuTimer() { cudaEventDestroy(s); cudaEventDestroy(e); }
    void start(cudaStream_t st=0) { cudaEventRecord(s, st); }
    float stop(cudaStream_t st=0) {
        cudaEventRecord(e, st); cudaEventSynchronize(e);
        float ms=0; cudaEventElapsedTime(&ms, s, e); return ms;
    }
};

int main() {
    const int H = 2048;
    const float s13 = 1.f/3.f;

    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("# Fused unpack_fp4 + pack_int8 benchmark\nDevice: %s (%d.%d)\n", p.name, p.major, p.minor);
    printf("Elements: %d\n\n", H);

    // Setup: FP4 input + scales
    void *d_fp4; float *d_fp4s, *d_f32, *d_i8s;
    int8_t *d_i8;
    cudaMalloc(&d_fp4, H);
    cudaMalloc(&d_fp4s, (H/16)*4);
    cudaMalloc(&d_f32, H*4);
    cudaMalloc(&d_i8, H);
    cudaMalloc(&d_i8s, (H/16)*4);

    std::vector<float> h_f32(H, 1.f), h_fp4s(H/16, s13), h_i8s(H/16, 1.f/127.f);
    cudaMemcpy(d_f32, h_f32.data(), H*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fp4s, h_fp4s.data(), (H/16)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_i8s, h_i8s.data(), (H/16)*4, cudaMemcpyHostToDevice);
    blackwell::kernels::pack_fp4(d_fp4, d_f32, d_fp4s, H, 0);

    int warm = 100, bench = 1000;

    // Old: unpack_fp4 + pack_int8 (2 kernels, 1 intermediate buffer)
    printf("Benchmarking unpack_fp4 + pack_int8 (old)...\n");
    for (int w = 0; w < warm; ++w) {
        blackwell::kernels::unpack_fp4(d_f32, d_fp4, d_fp4s, H, 0);
        blackwell::kernels::pack_int8(d_i8, d_f32, d_i8s, H, 0);
    }
    cudaDeviceSynchronize();
    GpuTimer t1; t1.start();
    for (int i = 0; i < bench; ++i) {
        blackwell::kernels::unpack_fp4(d_f32, d_fp4, d_fp4s, H, 0);
        blackwell::kernels::pack_int8(d_i8, d_f32, d_i8s, H, 0);
    }
    float ms1 = t1.stop();
    printf("  Total: %.3f ms, Per-call: %.3f us\n", ms1, ms1/bench*1000);

    // New: unpack_fp4_pack_int8 (1 fused kernel)
    printf("Benchmarking unpack_fp4_pack_int8 (fused)...\n");
    for (int w = 0; w < warm; ++w) {
        blackwell::kernels::unpack_fp4_pack_int8(d_i8, d_i8s, d_fp4, d_fp4s, d_i8s, H, 0);
    }
    cudaDeviceSynchronize();
    GpuTimer t2; t2.start();
    for (int i = 0; i < bench; ++i) {
        blackwell::kernels::unpack_fp4_pack_int8(d_i8, d_i8s, d_fp4, d_fp4s, d_i8s, H, 0);
    }
    float ms2 = t2.stop();
    printf("  Total: %.3f ms, Per-call: %.3f us\n", ms2, ms2/bench*1000);

    printf("\n  Speedup: %.2fx\n", ms1/ms2);

    // Correctness
    std::vector<int8_t> i8_old(H), i8_new(H);
    cudaMemcpy(i8_old.data(), d_i8, H, cudaMemcpyDeviceToHost);
    blackwell::kernels::unpack_fp4_pack_int8(d_i8, d_i8s, d_fp4, d_fp4s, d_i8s, H, 0);
    cudaMemcpy(i8_new.data(), d_i8, H, cudaMemcpyDeviceToHost);
    int max_diff = 0;
    for (int i = 0; i < H; ++i) {
        int d = abs(i8_old[i] - i8_new[i]);
        if (d > max_diff) max_diff = d;
    }
    printf("  Max INT8 diff: %d %s\n", max_diff, max_diff == 0 ? "✅ MATCH" : "❌");

    cudaFree(d_fp4); cudaFree(d_fp4s); cudaFree(d_f32);
    cudaFree(d_i8); cudaFree(d_i8s);
    return 0;
}
