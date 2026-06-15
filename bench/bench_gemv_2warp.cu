// bench/bench_gemv_2warp.cu — Microbench: 1-warp vs 2-warp INT4 GEMV
//
// Tests whether doubling threads-per-output-row (32→64) helps the
// memory-bound M=1 GEMV case. 1 warp = 32 threads stride-32 over K-blocks.
// 2 warp = 64 threads stride-64 over K-blocks (each thread does fewer K-blocks,
// more in-flight memory requests per thread).
//
// Expected outcome: no gain (kernel is memory-saturated at 95% of peak BW
// per AGENTS.md). But user wants empirical confirmation.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <cstdint>

namespace {

constexpr int B = 16;    // INT4 block size
constexpr int PB = 8;    // packed bytes per block

__device__ __forceinline__ void int4_8bytes_to_int4lanes(
    const uint8_t* bytes, int& l0, int& l1, int& l2, int& l3)
{
    int b0=bytes[0],b1=bytes[1],b2=bytes[2],b3=bytes[3];
    int b4=bytes[4],b5=bytes[5],b6=bytes[6],b7=bytes[7];
    int n00=(b0&0xF)-8,n01=((b0>>4)&0xF)-8,n02=(b1&0xF)-8,n03=((b1>>4)&0xF)-8;
    int n04=(b2&0xF)-8,n05=((b2>>4)&0xF)-8,n06=(b3&0xF)-8,n07=((b3>>4)&0xF)-8;
    int n08=(b4&0xF)-8,n09=((b4>>4)&0xF)-8,n10=(b5&0xF)-8,n11=((b5>>4)&0xF)-8;
    int n12=(b6&0xF)-8,n13=((b6>>4)&0xF)-8,n14=(b7&0xF)-8,n15=((b7>>4)&0xF)-8;
    l0=(n00&0xFF)|((n01&0xFF)<<8)|((n02&0xFF)<<16)|((n03&0xFF)<<24);
    l1=(n04&0xFF)|((n05&0xFF)<<8)|((n06&0xFF)<<16)|((n07&0xFF)<<24);
    l2=(n08&0xFF)|((n09&0xFF)<<8)|((n10&0xFF)<<16)|((n11&0xFF)<<24);
    l3=(n12&0xFF)|((n13&0xFF)<<8)|((n14&0xFF)<<16)|((n15&0xFF)<<24);
}

// ── 1-warp variant (32 threads) — production baseline ──
template <typename WScaleT>
__launch_bounds__(32, 8)
__global__ void gemv_int4_1warp(
    float* __restrict__ y,
    const uint8_t* __restrict__ x_packed,
    const float* __restrict__ x_scale,
    const uint8_t* __restrict__ W_packed,
    const WScaleT* __restrict__ W_scale,
    int K, int N)
{
    int n = blockIdx.x;
    int tid = threadIdx.x;
    int num_K_blks = K / B;
    float acc = 0.0f;

    for (int kb = tid; kb < num_K_blks; kb += 32) {
        const uint8_t* w_ptr = &W_packed[(size_t)n*(K/2) + kb*PB];
        uint2 w_packed = *reinterpret_cast<const uint2*>(w_ptr);
        const uint8_t* x_ptr = &x_packed[kb*PB];
        uint2 x_packed_val = *reinterpret_cast<const uint2*>(x_ptr);
        float w_sc = WScaleT(0);
        if constexpr (std::is_same_v<WScaleT, float>)
            w_sc = W_scale[(size_t)n*num_K_blks + kb];
        else
            w_sc = __half2float(W_scale[(size_t)n*num_K_blks + kb]);
        float x_sc = x_scale[kb];
        float prod = w_sc * x_sc;

        const uint8_t* wb = reinterpret_cast<const uint8_t*>(&w_packed);
        const uint8_t* xb = reinterpret_cast<const uint8_t*>(&x_packed_val);
        int wl0,wl1,wl2,wl3,xl0,xl1,xl2,xl3;
        int4_8bytes_to_int4lanes(wb,wl0,wl1,wl2,wl3);
        int4_8bytes_to_int4lanes(xb,xl0,xl1,xl2,xl3);
        int s=0;
        s=__dp4a(wl0,xl0,s);s=__dp4a(wl1,xl1,s);s=__dp4a(wl2,xl2,s);s=__dp4a(wl3,xl3,s);
        acc += static_cast<float>(s)*prod;
    }
    // Warp reduce
    acc += __shfl_xor_sync(0xffffffff,acc,16);
    acc += __shfl_xor_sync(0xffffffff,acc,8);
    acc += __shfl_xor_sync(0xffffffff,acc,4);
    acc += __shfl_xor_sync(0xffffffff,acc,2);
    acc += __shfl_xor_sync(0xffffffff,acc,1);
    if (tid==0) y[n] = acc;
}

// ── 2-warp variant (64 threads) — experiment ──
// 2 warps cooperate on K-reduction. warp 0 = threads 0-31, warp 1 = threads 32-63.
// Each thread strides by 64 over K-blocks. Cross-warp reduction via shared memory.
template <typename WScaleT>
__launch_bounds__(64, 4)
__global__ void gemv_int4_2warp(
    float* __restrict__ y,
    const uint8_t* __restrict__ x_packed,
    const float* __restrict__ x_scale,
    const uint8_t* __restrict__ W_packed,
    const WScaleT* __restrict__ W_scale,
    int K, int N)
{
    int n = blockIdx.x;
    int tid = threadIdx.x;
    int num_K_blks = K / B;
    float acc = 0.0f;

    for (int kb = tid; kb < num_K_blks; kb += 64) {
        const uint8_t* w_ptr = &W_packed[(size_t)n*(K/2) + kb*PB];
        uint2 w_packed = *reinterpret_cast<const uint2*>(w_ptr);
        const uint8_t* x_ptr = &x_packed[kb*PB];
        uint2 x_packed_val = *reinterpret_cast<const uint2*>(x_ptr);
        float w_sc;
        if constexpr (std::is_same_v<WScaleT, float>)
            w_sc = W_scale[(size_t)n*num_K_blks + kb];
        else
            w_sc = __half2float(W_scale[(size_t)n*num_K_blks + kb]);
        float x_sc = x_scale[kb];
        float prod = w_sc * x_sc;

        const uint8_t* wb = reinterpret_cast<const uint8_t*>(&w_packed);
        const uint8_t* xb = reinterpret_cast<const uint8_t*>(&x_packed_val);
        int wl0,wl1,wl2,wl3,xl0,xl1,xl2,xl3;
        int4_8bytes_to_int4lanes(wb,wl0,wl1,wl2,wl3);
        int4_8bytes_to_int4lanes(xb,xl0,xl1,xl2,xl3);
        int s=0;
        s=__dp4a(wl0,xl0,s);s=__dp4a(wl1,xl1,s);s=__dp4a(wl2,xl2,s);s=__dp4a(wl3,xl3,s);
        acc += static_cast<float>(s)*prod;
    }
    // Intra-warp reduce (each warp reduces to lane 0)
    acc += __shfl_xor_sync(0xffffffff,acc,16);
    acc += __shfl_xor_sync(0xffffffff,acc,8);
    acc += __shfl_xor_sync(0xffffffff,acc,4);
    acc += __shfl_xor_sync(0xffffffff,acc,2);
    acc += __shfl_xor_sync(0xffffffff,acc,1);

    // Cross-warp reduce via shared memory
    __shared__ float warp_sum[2];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) warp_sum[warp_id] = acc;
    __syncthreads();
    if (tid == 0) y[n] = warp_sum[0] + warp_sum[1];
}

} // namespace

int main(int argc, char** argv) {
    int K = 4096;   // input dim
    int N = 12288;  // output dim (MLP gate/up)
    int iters = 1000;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    fprintf(stderr, "# GEMV 2-warp microbench — %s\n", prop.name);
    fprintf(stderr, "# K=%d N=%d iters=%d\n", K, N, iters);

    // Allocate fake weights and activations
    size_t w_bytes = (size_t)N * (K/2);
    size_t w_sc_bytes = (size_t)N * (K/16) * sizeof(__half);
    size_t x_bytes = K/2;
    size_t x_sc_bytes = (K/16) * sizeof(float);

    uint8_t* h_W = new uint8_t[w_bytes];
    __half* h_Wsc = new __half[N * (K/16)];
    uint8_t* h_x = new uint8_t[x_bytes];
    float* h_xsc = new float[K/16];
    // Fill with random-ish data
    for (size_t i = 0; i < w_bytes; ++i) h_W[i] = rand() & 0xFF;
    for (int i = 0; i < N*(K/16); ++i) h_Wsc[i] = __float2half(0.01f + 0.001f*(rand()%100));
    for (size_t i = 0; i < x_bytes; ++i) h_x[i] = rand() & 0xFF;
    for (int i = 0; i < K/16; ++i) h_xsc[i] = 0.01f + 0.001f*(rand()%100);

    uint8_t *d_W, *d_x;
    __half *d_Wsc;
    float *d_xsc, *d_y;
    cudaMalloc(&d_W, w_bytes); cudaMemcpy(d_W, h_W, w_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_Wsc, w_sc_bytes); cudaMemcpy(d_Wsc, h_Wsc, w_sc_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_x, x_bytes); cudaMemcpy(d_x, h_x, x_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_xsc, x_sc_bytes); cudaMemcpy(d_xsc, h_xsc, x_sc_bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_y, N*sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < 10; ++i) {
        gemv_int4_1warp<__half><<<N, 32>>>(d_y, d_x, d_xsc, d_W, d_Wsc, K, N);
        gemv_int4_2warp<__half><<<N, 64>>>(d_y, d_x, d_xsc, d_W, d_Wsc, K, N);
    }
    cudaDeviceSynchronize();

    // Bench 1-warp
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i)
        gemv_int4_1warp<__half><<<N, 32>>>(d_y, d_x, d_xsc, d_W, d_Wsc, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms1 = 0;
    cudaEventElapsedTime(&ms1, start, stop);

    // Bench 2-warp
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i)
        gemv_int4_2warp<__half><<<N, 64>>>(d_y, d_x, d_xsc, d_W, d_Wsc, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms2 = 0;
    cudaEventElapsedTime(&ms2, start, stop);

    // Calculate BW (weight bytes + activation bytes per call)
    double bytes_per_call = (double)w_bytes + w_sc_bytes + x_bytes + x_sc_bytes;
    double us1 = ms1 * 1000.0 / iters;
    double us2 = ms2 * 1000.0 / iters;
    double bw1 = bytes_per_call / (us1 * 1e-6) / 1e9;  // GB/s
    double bw2 = bytes_per_call / (us2 * 1e-6) / 1e9;

    printf("── Results (K=%d, N=%d, %d iters) ──\n", K, N, iters);
    printf("  1-warp (32 thr): %8.1f µs/call, %6.1f GB/s\n", us1, bw1);
    printf("  2-warp (64 thr): %8.1f µs/call, %6.1f GB/s\n", us2, bw2);
    printf("  Speedup 2w/1w:   %.3fx\n", us1/us2);
    printf("  Peak BW:         ~448 GB/s (GDDR7)\n");

    // Also test with K=4096, N=4096 (Q proj)
    int N2 = 4096;
    size_t w_bytes2 = (size_t)N2 * (K/2);
    size_t w_sc_bytes2 = (size_t)N2 * (K/16) * sizeof(__half);
    uint8_t* d_W2; __half* d_Wsc2; float* d_y2;
    cudaMalloc(&d_W2, w_bytes2); cudaMemcpy(d_W2, h_W, w_bytes2, cudaMemcpyHostToDevice);
    cudaMalloc(&d_Wsc2, w_sc_bytes2); cudaMemcpy(d_Wsc2, h_Wsc, w_sc_bytes2, cudaMemcpyHostToDevice);
    cudaMalloc(&d_y2, N2*sizeof(float));

    // Warmup
    for (int i = 0; i < 10; ++i) {
        gemv_int4_1warp<__half><<<N2, 32>>>(d_y2, d_x, d_xsc, d_W2, d_Wsc2, K, N2);
        gemv_int4_2warp<__half><<<N2, 64>>>(d_y2, d_x, d_xsc, d_W2, d_Wsc2, K, N2);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i)
        gemv_int4_1warp<__half><<<N2, 32>>>(d_y2, d_x, d_xsc, d_W2, d_Wsc2, K, N2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms1, start, stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i)
        gemv_int4_2warp<__half><<<N2, 64>>>(d_y2, d_x, d_xsc, d_W2, d_Wsc2, K, N2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms2, start, stop);

    double bytes2 = (double)w_bytes2 + w_sc_bytes2 + x_bytes + x_sc_bytes;
    us1 = ms1 * 1000.0 / iters;
    us2 = ms2 * 1000.0 / iters;
    bw1 = bytes2 / (us1 * 1e-6) / 1e9;
    bw2 = bytes2 / (us2 * 1e-6) / 1e9;

    printf("── Results (K=%d, N=%d [Q proj], %d iters) ──\n", K, N2, iters);
    printf("  1-warp (32 thr): %8.1f µs/call, %6.1f GB/s\n", us1, bw1);
    printf("  2-warp (64 thr): %8.1f µs/call, %6.1f GB/s\n", us2, bw2);
    printf("  Speedup 2w/1w:   %.3fx\n", us1/us2);

    return 0;
}
