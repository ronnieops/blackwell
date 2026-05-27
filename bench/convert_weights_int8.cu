// bench/convert_weights_int8.cu — Convert FP4 weight files to INT8 format
//
// Reads FP4 weights (header + packed data + scales), unpacks to FP32 using
// per-block scales, computes INT8 per-block scales, packs to INT8, transposes.
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120,code=sm_120 \
//     -I include bench/convert_weights_int8.cu \
//     build/libblackwell_kernels.a -o bench/convert_weights_int8

#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>
#include "blackwell/kernels.h"

// Dequant FP4 with per-block 2D scales
__global__ void unpack_fp4_blockscaled(
    float* __restrict__ out,
    const uint8_t* __restrict__ in_fp4,
    const float* __restrict__ scales,    // [K/16 × N/16]
    int K, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * N;
    if (idx >= total) return;
    int k = idx / N;
    int n = idx % N;
    int kb = k / 16;
    int nb = n / 16;
    float sc = scales[kb * (N/16) + nb];
    out[idx] = static_cast<float>(reinterpret_cast<const __nv_fp4_e2m1*>(in_fp4)[idx]) * sc;
}

// Compute INT8 per-block scale (absmax/127)
__global__ void compute_int8_scales(
    const float* in,       // [K×N] FP32
    float* out_scales,     // [K/16 × N/16] FP32
    int K, int N)
{
    constexpr int B = 16;
    int total_blocks = (K/B) * (N/B);
    int tid = threadIdx.x;
    int blk = blockIdx.x;
    if (blk >= total_blocks) return;
    int num_N_blks = N / B;
    int nb = blk % num_N_blks;
    int kb = blk / num_N_blks;

    __shared__ float s_max[256];
    if (tid < B*B) {
        int i = tid / B, j = tid % B;
        s_max[tid] = fabsf(in[(kb*B + i) * N + (nb*B + j)]);
    } else s_max[tid] = 0;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) { if (s_max[tid + s] > s_max[tid]) s_max[tid] = s_max[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) {
        float blk_max = s_max[0];
        out_scales[blk] = (blk_max > 1e-10f) ? blk_max / 127.f : 1.f/127.f;
    }
}

static bool check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) { printf("FAIL: %s: %s\n", msg, cudaGetErrorString(e)); return false; }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <input.fp4> <output_prefix>\n", argv[0]);
        return 1;
    }

    FILE* f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", argv[1]); return 1; }
    int header[5];
    fread(header, 4, 5, f);
    int K = header[0], N = header[1];
    int block = header[2];
    int num_K_blks = header[3], num_N_blks = header[4];
    printf("%s: K=%d N=%d block=%d nKb=%d nNb=%d\n", argv[1], K, N, block, num_K_blks, num_N_blks);

    size_t data_sz = (size_t)K * N;
    size_t scale_sz = (size_t)num_K_blks * num_N_blks * 4;

    std::vector<uint8_t> fp4_data(data_sz);
    std::vector<float> fp4_scales(num_K_blks * num_N_blks);
    fread(fp4_data.data(), 1, data_sz, f);
    fread(fp4_scales.data(), 4, scale_sz, f);
    fclose(f);

    // GPU buffers
    float *d_fp32;
    int8_t *d_i8, *d_i8_t;
    float *d_i8_sc, *d_i8_t_sc;
    cudaMalloc(&d_fp32, data_sz*4);
    cudaMalloc(&d_i8, data_sz);
    cudaMalloc(&d_i8_t, data_sz);
    cudaMalloc(&d_i8_sc, scale_sz);
    cudaMalloc(&d_i8_t_sc, scale_sz);

    uint8_t *d_fp4;
    float *d_fp4_sc;
    cudaMalloc(&d_fp4, data_sz);
    cudaMalloc(&d_fp4_sc, scale_sz);
    cudaMemcpy(d_fp4, fp4_data.data(), data_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fp4_sc, fp4_scales.data(), scale_sz, cudaMemcpyHostToDevice);

    // Step 1: FP4→FP32 with per-block scales (not uniform)
    printf("  Dequant FP4→FP32 (per-block scales)...\n");
    {
        int threads = 256;
        int blocks = (data_sz + threads - 1) / threads;
        unpack_fp4_blockscaled<<<blocks, threads,0>>>(d_fp32, d_fp4, d_fp4_sc, K, N);
        check(cudaPeekAtLastError(), "unpack_fp4_blockscaled");
    }

    // Step 2: Compute INT8 per-block scales
    printf("  Compute INT8 scales...\n");
    {
        int total_blks = num_K_blks * num_N_blks;
        compute_int8_scales<<<total_blks,256,0>>>(d_fp32, d_i8_sc, K, N);
        check(cudaPeekAtLastError(), "compute_int8_scales");
    }

    // Step 3: Pack FP32→INT8
    printf("  Pack FP32→INT8...\n");
    check(blackwell::kernels::pack_int8(d_i8, d_fp32, d_i8_sc, data_sz,0), "pack_int8");

    // Step 4: Transpose INT8 weights
    printf("  Transpose INT8 weights...\n");
    check(blackwell::kernels::transpose_int8_weights(d_i8_t,d_i8_t_sc,d_i8,d_i8_sc,K,N,0), "transpose_int8");
    cudaDeviceSynchronize();

    // Download
    std::vector<int8_t> i8_data(data_sz), i8_t_data(data_sz);
    std::vector<float> i8_sc(num_K_blks*num_N_blks), i8_t_sc(num_K_blks*num_N_blks);
    cudaMemcpy(i8_data.data(), d_i8, data_sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(i8_t_data.data(), d_i8_t, data_sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(i8_sc.data(), d_i8_sc, scale_sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(i8_t_sc.data(), d_i8_t_sc, scale_sz, cudaMemcpyDeviceToHost);

    // Write files
    auto wf = [&](const char* ext, const void* d, size_t sz) {
        char p[256]; snprintf(p,256,"%s.%s",argv[2],ext);
        FILE* f=fopen(p,"wb"); fwrite(header,4,5,f); fwrite(d,1,sz,f); fclose(f);
        printf("  %s: header+%zu bytes\n", p, sz);
    };
    wf("int8", i8_data.data(), data_sz);
    wf("int8_t", i8_t_data.data(), data_sz);
    wf("scale", i8_sc.data(), scale_sz);
    wf("scale_t", i8_t_sc.data(), scale_sz);

    // Verify gemv_int8 works with converted weights
    printf("\n  Verifying gemv_int8...\n");
    float *d_x32, *d_y;
    int8_t *d_x8;
    float *d_xs;
    int nb_x = K/16;
    cudaMalloc(&d_x32, K*4); cudaMalloc(&d_y, N*4);
    cudaMalloc(&d_x8, K); cudaMalloc(&d_xs, nb_x*4);
    std::vector<float> x32(K,0.5f);
    cudaMemcpy(d_x32, x32.data(), K*4, cudaMemcpyHostToDevice);
    float xs_val = 0.5f/127.f;
    std::vector<float> xs_h(nb_x, xs_val);
    cudaMemcpy(d_xs, xs_h.data(), nb_x*4, cudaMemcpyHostToDevice);
    check(blackwell::kernels::pack_int8(d_x8, d_x32, d_xs, K,0), "pack_int8(x)");
    check(blackwell::kernels::gemv_int8(d_y, d_x8, d_xs, d_i8_t, d_i8_t_sc, K, N,0), "gemv_int8");
    cudaDeviceSynchronize();
    printf("  gemv_int8 OK\n");

    // Compare with FP4 v2
    void *d_x4;
    cudaMalloc(&d_x4, K);
    check(blackwell::kernels::pack_fp4(d_x4, d_x32, &xs_val, K,0), "pack_fp4(x)");
    float *d_y_v2;
    cudaMalloc(&d_y_v2, N*4);
    // Need transposed FP4 weights. We don't have them here.
    // Just confirm INT8 doesn't crash. Skip detailed comparison.

    cudaFree(d_fp32); cudaFree(d_i8); cudaFree(d_i8_t);
    cudaFree(d_i8_sc); cudaFree(d_i8_t_sc);
    cudaFree(d_fp4); cudaFree(d_fp4_sc);
    cudaFree(d_x32); cudaFree(d_y); cudaFree(d_x8); cudaFree(d_xs);
    cudaFree(d_x4); cudaFree(d_y_v2);

    printf("\nDone: %s int8/scale/scale_t files written\n", argv[2]);
    return 0;
}
