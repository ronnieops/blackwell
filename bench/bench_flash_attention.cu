// bench_flash_attention.cu — Benchmark library attention_prefill vs attn1 baseline
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/bench_flash_attention.cu build/libblackwell_kernels.a \
//     -o bench/bench_flash_attention

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include "blackwell/kernels.h"

static void die(cudaError_t e) {
    if (e != cudaSuccess) { printf("FAIL: %s\n", cudaGetErrorString(e)); exit(1); }
}

// attn1 correctness baseline (1 thr per output element)
__global__ void attn1(float*O,const float*Q,const float*K,const float*V,int M,int H,int QH,float sc){
    int h=blockIdx.x,m=blockIdx.y,d=blockIdx.z*blockDim.x+threadIdx.x;
    if(h>=QH||m>=M||d>=H)return;
    const float*Qm=Q+h*M*H+m*H,*Kh=K+h*M*H,*Vh=V+h*M*H;
    float mx=-1e9f,sum=0,acc=0,s[128];
    for(int j=0;j<M;++j){float dot=0;for(int k=0;k<H;++k)dot+=Qm[k]*Kh[j*H+k];s[j]=dot*sc;mx=fmaxf(mx,dot*sc);}
    for(int j=0;j<M;++j){s[j]=expf(s[j]-mx);sum+=s[j];}
    for(int j=0;j<M;++j)acc+=s[j]*Vh[j*H+d];
    O[h*M*H+m*H+d]=acc/sum;
}

int main(int argc, char** argv) {
    int IT = 20;
    if (argc > 1) IT = atoi(argv[1]);

    cudaDeviceProp P;
    cudaGetDeviceProperties(&P, 0);
    printf("# Flash Attention Benchmark — %s  SM:%d\n", P.name, P.multiProcessorCount);

    cudaStream_t st;
    die(cudaStreamCreate(&st));
    cudaEvent_t s, e;
    die(cudaEventCreate(&s));
    die(cudaEventCreate(&e));

    const int M = 128, H = 64, QH = 12, kvH = 1;  // Qwen3: 12 Q heads, 1 KV head
    float scale = 1.0f / sqrtf((float)H);
    size_t N = QH * M * H;       // Q: (12, 128, 64)
    size_t N_kv = kvH * M * H;   // K/V: (1, 128, 64)

    // Generate test data (single KV head)
    std::vector<float> Q(N), K(N_kv), V(N_kv);
    for (int i = 0; i < (int)N; ++i) Q[i] = ((i * 17 + 13) % 127 - 63) * 0.01f;
    for (int i = 0; i < (int)N_kv; ++i) { K[i] = ((i * 23 + 7) % 127 - 63) * 0.01f; V[i] = ((i * 31 + 11) % 127 - 63) * 0.01f; }

    float *dQ, *dK, *dV, *dO;
    die(cudaMalloc(&dQ, N * 4));
    // Allocate K/V for ALL Q heads (attn1 baseline doesn't support GQA)
    die(cudaMalloc(&dK, N * 4));
    die(cudaMalloc(&dV, N * 4));
    die(cudaMalloc(&dO, N * 4));
    die(cudaMemcpy(dQ, Q.data(), N * 4, cudaMemcpyHostToDevice));
    // Replicate single KV head across all Q heads for attn1 comparison
    for (int h = 0; h < QH; ++h) {
        die(cudaMemcpy(dK + h * M * H, K.data(), N_kv * 4, cudaMemcpyHostToDevice));
        die(cudaMemcpy(dV + h * M * H, V.data(), N_kv * 4, cudaMemcpyHostToDevice));
    }

    int q_per_group = QH / kvH;  // 12

    // ─── attn1 (baseline) ──────────────────────────────
    dim3 g1(QH, M, (H + 255) / 256), b1(256);
    attn1<<<g1, b1, 0, st>>>(dO, dQ, dK, dV, M, H, QH, scale);
    die(cudaStreamSynchronize(st));
    die(cudaPeekAtLastError());

    // Read baseline output for correctness check
    std::vector<float> O_baseline(N);
    die(cudaMemcpy(O_baseline.data(), dO, N * 4, cudaMemcpyDeviceToHost));

    die(cudaEventRecord(s, st));
    for (int i = 0; i < IT; ++i)
        attn1<<<g1, b1, 0, st>>>(dO, dQ, dK, dV, M, H, QH, scale);
    die(cudaEventRecord(e, st));
    die(cudaEventSynchronize(e));
    float t1;
    die(cudaEventElapsedTime(&t1, s, e));
    double ms1 = t1 / IT;
    double gf1 = 2.0 * QH * M * M * H * 1e-9 / (ms1 / 1000.0);
    printf("\nattn1 (1 thr/elem):      %.3f ms  (%7.1f GFLOPS)\n", ms1, gf1);

    // ─── library attention_prefill (flash_attention_kernel) ───
    die(cudaMemset(dO, 0, N * 4));
    die(blackwell::kernels::attention_prefill(dO, dQ, dK, dV,
        M, H, QH, kvH, q_per_group, scale, st));
    die(cudaStreamSynchronize(st));
    die(cudaPeekAtLastError());

    die(cudaEventRecord(s, st));
    for (int i = 0; i < IT; ++i)
        die(blackwell::kernels::attention_prefill(dO, dQ, dK, dV,
            M, H, QH, kvH, q_per_group, scale, st));
    die(cudaEventRecord(e, st));
    die(cudaEventSynchronize(e));
    float t2;
    die(cudaEventElapsedTime(&t2, s, e));
    double ms2 = t2 / IT;
    double gf2 = 2.0 * QH * M * M * H * 1e-9 / (ms2 / 1000.0);
    printf("attention_prefill (flash): %.3f ms  (%7.1f GFLOPS)  %6.2fx\n", ms2, gf2, ms1 / ms2);

    // ─── Correctness check ─────────────────────────────
    std::vector<float> O_flash(N);
    die(cudaMemcpy(O_flash.data(), dO, N * 4, cudaMemcpyDeviceToHost));

    int nans = 0, mismatches = 0;
    float max_diff = 0, max_rel = 0;
    for (int i = 0; i < (int)N; ++i) {
        if (isnan(O_flash[i]) || isnan(O_baseline[i])) { nans++; continue; }
        float diff = fabsf(O_flash[i] - O_baseline[i]);
        float rel = diff / (fabsf(O_baseline[i]) + 1e-9f);
        max_diff = fmaxf(max_diff, diff);
        max_rel = fmaxf(max_rel, rel);
        if (rel > 1e-3f) mismatches++;
    }
    printf("\nCorrectness (vs attn1):\n");
    printf("  max diff: %.2e, max rel err: %.2e\n", max_diff, max_rel);
    printf("  nans: %d, mismatches (>0.1%%): %d / %zu\n", nans, mismatches, N);

    // ─── Per-layer breakdown ───────────────────────────
    printf("\nPer-layer (M=128, H=64, QH=12, kvH=1):\n");
    printf("  GEMM (from prefill_benchmark): %.3f ms\n", 0.863);
    printf("  Attention (this kernel): %.3f ms\n", ms2);
    printf("  Total per layer:           %.3f ms\n", 0.863 + ms2);
    printf("  28-layer total:            %.1f ms\n", 28 * (0.863 + ms2));

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    cudaEventDestroy(s); cudaEventDestroy(e); cudaStreamDestroy(st);
    return 0;
}