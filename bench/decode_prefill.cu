// bench/decode_prefill.cu — GEMM prefill benchmark + optimization
//
// Problem: Current GEMM at 40 GB/s (8.1% of 500 GB/s peak).
// Root cause: CTA tile 128×128 too large for M≤128. Only 1 CTA runs.
// Fix: Smaller CTA tiles + more parallelism.
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/decode_prefill.cu build/libblackwell_kernels.a \
//     -o bench/decode_prefill

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include "blackwell/kernels.h"

static void die(cudaError_t e, const char* m){
    if(e!=cudaSuccess){printf("FAIL %s %s\n",m,cudaGetErrorString(e));::exit(1);}}
using Clock = std::chrono::high_resolution_clock;

struct LW { std::vector<int8_t> d; std::vector<float> sc; };
struct DW { int8_t*d; float*sc; };
struct L { DW q,k,v,o,g,u,d; };

static LW lw(const char*p){
    char x[256]; snprintf(x,256,"%s.int8_t",p); FILE*f=fopen(x,"rb");
    int h[5]; fread(h,4,5,f); LW w;
    w.d.resize(h[0]*h[1]); fread(w.d.data(),1,w.d.size(),f); fclose(f);
    snprintf(x,256,"%s.scale_t",p); f=fopen(x,"rb"); fread(h,4,5,f);
    w.sc.resize(h[3]*h[4]); fread(w.sc.data(),4,w.sc.size(),f); fclose(f); return w;
}
static DW dw(const LW& w){
    DW d;
    cudaMalloc(&d.d,w.d.size());    cudaMemcpy(d.d,w.d.data(),w.d.size(),cudaMemcpyHostToDevice);
    cudaMalloc(&d.sc,w.sc.size()*4);cudaMemcpy(d.sc,w.sc.data(),w.sc.size()*4,cudaMemcpyHostToDevice); return d;
}

// INT8 GEMM using current kernel
// Measures throughput for different M (prefill sequence length)
void bench_current_gemm(const char* label, int M, int K, int N, int IT, cudaStream_t st){
    // Allocate A (M×K), B (K×N), C (M×N)
    std::vector<int8_t> A(M*K);
    std::vector<float> A_sc((M+15)/16*(K+15)/16);
    std::vector<int8_t> B(K*N);
    std::vector<float> B_sc((K+15)/16*(N+15)/16);
    for(int i=0;i<M*K;++i)A[i]=((i*17+13)%127)-64;
    for(int i=0;i<K*N;++i)B[i]=((i*23+7)%127)-64;
    for(auto& s:A_sc)s=1.f/127.f;
    for(auto& s:B_sc)s=1.f/127.f;

    int8_t*d_A; float*d_Asc; int8_t*d_B; float*d_Bsc; float*d_C;
    cudaMalloc(&d_A,M*K); cudaMalloc(&d_Asc,A_sc.size()*4);
    cudaMalloc(&d_B,K*N); cudaMalloc(&d_Bsc,B_sc.size()*4); cudaMalloc(&d_C,M*N*4);
    cudaMemcpy(d_A,A.data(),M*K,cudaMemcpyHostToDevice);
    cudaMemcpy(d_Asc,A_sc.data(),A_sc.size()*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B.data(),K*N,cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bsc,B_sc.data(),B_sc.size()*4,cudaMemcpyHostToDevice);

    // Warmup
    for(int w=0;w<5;++w)
        blackwell::kernels::dispatch_matmul(d_C,d_A,d_B,d_Asc,d_Bsc,M,N,K,
            blackwell::kernels::KernelMode::Prefill,st);
    cudaStreamSynchronize(st);

    // Benchmark
    auto t0=Clock::now();
    for(int i=0;i<IT;++i)
        blackwell::kernels::dispatch_matmul(d_C,d_A,d_B,d_Asc,d_Bsc,M,N,K,
            blackwell::kernels::KernelMode::Prefill,st);
    cudaStreamSynchronize(st);
    auto t1=Clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();

    double gbps = 2.0*M*K*N*4.0 / (ms/1000.0) / 1e9;
    double pct = gbps / 500.0 * 100.0;
    printf("  %-25s M=%-4d K=%-4d N=%-4d: %7.1f GB/s (%5.1f%% peak)  (%.2f ms)\n",
        label, M, K, N, gbps, pct, ms/IT);

    cudaFree(d_A);cudaFree(d_Asc);cudaFree(d_B);cudaFree(d_Bsc);cudaFree(d_C);
}

// INT8 GEMM using naive CPU fallback (for correctness check)
void cpu_gemm(float*C, const int8_t*A, const float*Asc,
             const int8_t*B, const float*Bsc, int M, int K, int N){
    // C = A @ B, per-block quantized
    for(int m=0;m<M;++m){
        for(int n=0;n<N;++n){
            float acc=0;
            for(int k=0;k<K;++k){
                int mb=m/16, kb=k/16, nb=n/16;
                float a=static_cast<float>(A[m*K+k])*Asc[mb*(K+15)/16+kb];
                float b=static_cast<float>(B[k*N+n])*Bsc[kb*(N+15)/16+nb];
                acc+=a*b;
            }
            C[m*N+n]=acc;
        }
    }
}

// INT8 GEMM using naive GPU (1 thread per output element)
// For comparison with CTA-based kernel
__global__ void naive_gemm_kernel(float*C, const int8_t*A, const float*Asc,
    const int8_t*B, const float*Bsc, int M, int K, int N){
    int m=blockIdx.x*blockDim.x+threadIdx.x;
    int n=blockIdx.y*blockDim.y+threadIdx.y;
    if(m>=M||n>=N) return;
    int nb_m=(M+15)/16, nb_k=(K+15)/16, nb_n=(N+15)/16;
    float acc=0;
    for(int k=0;k<K;++k){
        int mb=m/16, kb=k/16, nb=n/16;
        float a=static_cast<float>(A[m*K+k])*Asc[mb*nb_k+kb];
        float b=static_cast<float>(B[k*N+n])*Bsc[kb*nb_n+nb];
        acc+=a*b;
    }
    C[m*N+n]=acc;
}

void bench_naive_gpu(const char* label, int M, int K, int N, int IT, cudaStream_t st){
    std::vector<int8_t> A(M*K);
    std::vector<float> A_sc((M+15)/16*(K+15)/16);
    std::vector<int8_t> B(K*N);
    std::vector<float> B_sc((K+15)/16*(N+15)/16);
    for(int i=0;i<M*K;++i)A[i]=((i*17+13)%127)-64;
    for(int i=0;i<K*N;++i)B[i]=((i*23+7)%127)-64;
    for(auto& s:A_sc)s=1.f/127.f;
    for(auto& s:B_sc)s=1.f/127.f;

    int8_t*d_A; float*d_Asc; int8_t*d_B; float*d_Bsc; float*d_C;
    cudaMalloc(&d_A,M*K); cudaMalloc(&d_Asc,A_sc.size()*4);
    cudaMalloc(&d_B,K*N); cudaMalloc(&d_Bsc,B_sc.size()*4); cudaMalloc(&d_C,M*N*4);
    cudaMemcpy(d_A,A.data(),M*K,cudaMemcpyHostToDevice);
    cudaMemcpy(d_Asc,A_sc.data(),A_sc.size()*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B.data(),K*N,cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bsc,B_sc.data(),B_sc.size()*4,cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid((M+15)/16,(N+15)/16);

    cudaStreamSynchronize(st);
    auto t0=Clock::now();
    for(int i=0;i<IT;++i){
        naive_gemm_kernel<<<grid,block,0,st>>>(d_C,d_A,d_Asc,d_B,d_Bsc,M,K,N);
    }
    cudaStreamSynchronize(st);
    auto t1=Clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    double gbps=2.0*M*K*N*4.0/(ms/IT)/1e9;
    printf("  %-25s M=%-4d K=%-4d N=%-4d: %7.1f GB/s (%5.1f%% peak)  (%.2f ms)\n",
        label,M,K,N,gbps,gbps/500.0*100,ms/IT);
    cudaFree(d_A);cudaFree(d_Asc);cudaFree(d_B);cudaFree(d_Bsc);cudaFree(d_C);
}

// GEMV decode baseline (single row × weight matrix)
void bench_gemv(const char* label, int K, int N, int IT, cudaStream_t st){
    std::vector<int8_t> x(K);
    std::vector<float> x_sc((K+15)/16);
    std::vector<int8_t> W(K*N);
    std::vector<float> W_sc((K+15)/16*(N+15)/16);
    for(int i=0;i<K;++i)x[i]=((i*17+13)%127)-64;
    for(int i=0;i<K*N;++i)W[i]=((i*23+7)%127)-64;
    for(auto& s:x_sc)s=1.f/127.f;
    for(auto& s:W_sc)s=1.f/127.f;

    int8_t*d_x; float*d_xsc; int8_t*d_W; float*d_Wsc; float*d_y;
    cudaMalloc(&d_x,K); cudaMalloc(&d_xsc,x_sc.size()*4);
    cudaMalloc(&d_W,K*N); cudaMalloc(&d_Wsc,W_sc.size()*4); cudaMalloc(&d_y,N*4);
    cudaMemcpy(d_x,x.data(),K,cudaMemcpyHostToDevice);
    cudaMemcpy(d_xsc,x_sc.data(),x_sc.size()*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_W,W.data(),K*N,cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wsc,W_sc.data(),W_sc.size()*4,cudaMemcpyHostToDevice);

    cudaStreamSynchronize(st);
    auto t0=Clock::now();
    for(int i=0;i<IT;++i)
        blackwell::kernels::gemv_int8_warp(d_y,d_x,d_xsc,d_W,d_Wsc,K,N,st);
    cudaStreamSynchronize(st);
    auto t1=Clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    double gbps=2.0*K*N*4.0/(ms/IT)/1e9;
    printf("  %-25s K=%-4d N=%-4d: %7.0f GB/s (%5.1f%% peak)  (%.1f us)\n",
        label,K,N,gbps,gbps/500.0*100,ms*1000.0/IT);
    cudaFree(d_x);cudaFree(d_xsc);cudaFree(d_W);cudaFree(d_Wsc);cudaFree(d_y);
}

// Correctness check: current GEMM vs CPU
void correctness_check(int M, int K, int N){
    printf("\n  Correctness check (M=%d K=%d N=%d)...\n",M,K,N);
    std::vector<int8_t> A(M*K);
    std::vector<float> A_sc((M+15)/16*(K+15)/16);
    std::vector<int8_t> B(K*N);
    std::vector<float> B_sc((K+15)/16*(N+15)/16);
    for(int i=0;i<M*K;++i)A[i]=((i*17+13)%127)-64;
    for(int i=0;i<K*N;++i)B[i]=((i*23+7)%127)-64;
    for(auto& s:A_sc)s=1.f/127.f;
    for(auto& s:B_sc)s=1.f/127.f;

    int8_t*d_A; float*d_Asc; int8_t*d_B; float*d_Bsc; float*d_C;
    cudaMalloc(&d_A,M*K); cudaMalloc(&d_Asc,A_sc.size()*4);
    cudaMalloc(&d_B,K*N); cudaMalloc(&d_Bsc,B_sc.size()*4); cudaMalloc(&d_C,M*N*4);
    cudaMemcpy(d_A,A.data(),M*K,cudaMemcpyHostToDevice);
    cudaMemcpy(d_Asc,A_sc.data(),A_sc.size()*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B.data(),K*N,cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bsc,B_sc.data(),B_sc.size()*4,cudaMemcpyHostToDevice);

    cudaStream_t st; cudaStreamCreate(&st);
    blackwell::kernels::dispatch_matmul(d_C,d_A,d_B,d_Asc,d_Bsc,M,N,K,
        blackwell::kernels::KernelMode::Prefill,st);
    cudaStreamSynchronize(st);

    std::vector<float> C_gpu(M*N);
    cudaMemcpy(C_gpu.data(),d_C,M*N*4,cudaMemcpyDeviceToHost);

    std::vector<float> C_cpu(M*N);
    cpu_gemm(C_cpu.data(),A.data(),A_sc.data(),B.data(),B_sc.data(),M,K,N);

    float max_diff=0,max_val=0;
    for(int i=0;i<M*N;++i){
        float d=fabs(C_gpu[i]-C_cpu[i]);
        if(d>max_diff)max_diff=d;
        if(fabs(C_cpu[i])>max_val)max_val=fabs(C_cpu[i]);
    }
    printf("  Max diff: %.6f  Max val: %.6f  Ratio: %.6f\n",max_diff,max_val,max_diff/(max_val+1e-10));

    cudaFree(d_A);cudaFree(d_Asc);cudaFree(d_B);cudaFree(d_Bsc);cudaFree(d_C);
    cudaStreamDestroy(st);
}

int main(int argc, char** argv){
    int IT=20;
    if(argc>1)IT=atoi(argv[1]);

    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    printf("# GEMM Prefill Benchmark — %s\n  Peak: 500 GB/s GDDR7\n\n",P.name);

    cudaStream_t st; die(cudaStreamCreate(&st),"stream");

    // Qwen3-1.7B dimensions
    const int H=2048,I=6144,Q=2048,KV=1024;

    printf("=== GEMM Prefill (INT8 block-scaled) ===\n");
    printf("  %-25s %6s  %6s  %6s  %9s  %6s\n",
        "Config","M","K","N","GB/s","%peak");
    printf("  %-25s %6s  %6s  %6s  %9s  %6s\n",
        "-------","---","---","---","----","-----");

    // Real Qwen3 shapes
    bench_current_gemm("Wo (Q×H)", 128, 2048, 2048, IT, st);
    bench_current_gemm("Q (H×H)", 128, 2048, 2048, IT, st);
    bench_current_gemm("K (H×KV)", 128, 2048, 1024, IT, st);
    bench_current_gemm("V (H×KV)", 128, 2048, 1024, IT, st);
    bench_current_gemm("gate (H×I)", 128, 2048, 6144, IT, st);
    bench_current_gemm("up (H×I)", 128, 2048, 6144, IT, st);
    bench_current_gemm("down (I×H)", 128, 6144, 2048, IT, st);

    printf("\n  Varied M (H=2048, I=6144):\n");
    for(int M: {1, 4, 8, 16, 32, 64, 128}){
        char label[32]; snprintf(label,sizeof(label),"M=%d (H×I)",M);
        bench_current_gemm(label, M, 2048, 6144, IT, st);
    }

    printf("\n=== Naive GPU GEMM (1 thr/output) ===\n");
    bench_naive_gpu("Naive M=128", 128, 2048, 6144, IT, st);
    bench_naive_gpu("Naive M=32", 32, 2048, 6144, IT, st);
    bench_naive_gpu("Naive M=8", 8, 2048, 6144, IT, st);
    bench_naive_gpu("Naive M=4", 4, 2048, 6144, IT, st);
    bench_naive_gpu("Naive M=2", 2, 2048, 6144, IT, st);

    printf("\n=== GEMV Decode baseline (M=1, for reference) ===\n");
    bench_gemv("gate GEMV (2048×6144)", 2048, 6144, IT*100, st);
    bench_gemv("Wo GEMV (2048×2048)", 2048, 2048, IT*100, st);

    printf("\n=== Correctness checks ===\n");
    correctness_check(128, 2048, 2048);
    correctness_check(32, 2048, 6144);
    correctness_check(8, 2048, 6144);
    correctness_check(2, 2048, 6144);

    printf("\n=== Analysis ===\n");
    printf("  Peak: 500 GB/s. GEMM kernel: 40 GB/s (8.1%%).\n");
    printf("  If below 50%% peak: likely grid/CTA sizing issue.\n");
    printf("  Fix: smaller CTA tiles (64×64, 32×32) for better parallelism.\n");

    cudaStreamDestroy(st);
    return 0;
}