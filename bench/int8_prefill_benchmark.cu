// bench/int8_prefill_benchmark.cu — INT8 GEMM prefill benchmark
//
// Measures INT8 WMMA GEMM throughput with correct weight layout.
// Uses real Qwen3-1.7B weight shapes.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include "blackwell/kernels.h"

static void chk(cudaError_t e){if(e){printf("CUDA err %d: %s\n",e,cudaGetErrorString(e));exit(1);}}

struct IW { int K,N; int8_t*d; float*ds; };
static IW load_iw(const char* d, const char* n){
    char p[256]; snprintf(p,256,"%s/%s.int8_t",d,n); FILE*f=fopen(p,"rb");
    if(!f){printf("Cannot open %s\n",p);exit(1);}
    int h[5]; fread(h,4,5,f);
    IW w{h[0],h[1],nullptr,nullptr};
    std::vector<int8_t> tmp((size_t)w.K*w.N);
    fread(tmp.data(),1,(size_t)w.K*w.N,f); fclose(f);
    cudaMalloc(&w.d,(size_t)w.K*w.N);
    cudaMemcpy(w.d,tmp.data(),(size_t)w.K*w.N,cudaMemcpyHostToDevice);
    snprintf(p,256,"%s/%s.scale_t",d,n); f=fopen(p,"rb");
    fread(h,4,5,f);
    size_t ns=(size_t)h[3]*h[4];
    std::vector<float> tmp_s(ns);
    fread(tmp_s.data(),4,ns,f); fclose(f);
    cudaMalloc(&w.ds,ns*4);
    cudaMemcpy(w.ds,tmp_s.data(),ns*4,cudaMemcpyHostToDevice);
    return w;
}

void bench(const char* label, int M, int K, int N, IW& w, int IT, cudaStream_t st){
    // A: [M×K] activations
    std::vector<int8_t> A(M*K);
    std::vector<float> A_sc((M+15)/16*(K+15)/16);
    for(int i=0;i<M*K;++i) A[i]=((i*17+13)%255)-128;
    for(auto& s:A_sc) s=1.f/127.f;
    
    int8_t*d_A; float*d_Asc;
    chk(cudaMalloc(&d_A,M*K));
    chk(cudaMalloc(&d_Asc,A_sc.size()*4));
    cudaMemcpy(d_A,A.data(),M*K,cudaMemcpyHostToDevice);
    cudaMemcpy(d_Asc,A_sc.data(),A_sc.size()*4,cudaMemcpyHostToDevice);
    
    float*d_C;
    chk(cudaMalloc(&d_C,M*N*4));
    
    // Warmup
    for(int i=0;i<3;++i)
        blackwell::kernels::gemm_int8_wmma_fast(d_C,d_A,d_Asc,w.d,w.ds,M,N,K,st);
    cudaStreamSynchronize(st);
    
    // Benchmark
    auto t0=std::chrono::high_resolution_clock::now();
    for(int i=0;i<IT;++i)
        blackwell::kernels::gemm_int8_wmma_fast(d_C,d_A,d_Asc,w.d,w.ds,M,N,K,st);
    cudaStreamSynchronize(st);
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/IT;
    
    double gflops=2.0*M*N*K/(ms/1000.0)/1e9;
    double gbps=(M*K+K*N+M*N)*4.0/(ms/1000.0)/1e9;
    
    printf("  %-15s M=%-4d K=%-4d N=%-4d: %7.1f GFLOPS  %6.1f GB/s  %.3f ms\n",
        label,M,N,K,gflops,gbps,ms);
    
    cudaFree(d_A);cudaFree(d_Asc);cudaFree(d_C);
}

int main(int argc, char** argv){
    int IT=20;
    if(argc>1)IT=atoi(argv[1]);
    
    cudaDeviceProp p;cudaGetDeviceProperties(&p,0);
    printf("# INT8 WMMA GEMM Prefill — %s\n",p.name);
    printf("  Peak: ~50 TOPS INT8, 500 GB/s GDDR7\n\n");
    
    cudaStream_t st;cudaStreamCreate(&st);
    
    const char* dir="weights_int8_bf16";
    printf("Loading Qwen3-1.7B weights...\n");
    
    // Load one layer's weights
    IW q_k=load_iw(dir,"0_self_attn.q_proj");  // [2048×2048]
    IW gate=load_iw(dir,"0_mlp.gate_proj");     // [6144×2048]
    IW down=load_iw(dir,"0_mlp.down_proj");     // [2048×6144]
    
    printf("\n=== INT8 WMMA GEMM Prefill (real weights) ===\n");
    for(int M : {16, 32, 64, 128}){
        printf("\n  M=%d:\n",M);
        bench("Q proj",M,2048,2048,q_k,IT,st);
        bench("gate",M,2048,6144,gate,IT,st);
        bench("down",M,6144,2048,down,IT,st);
    }
    
    cudaFree(q_k.d);cudaFree(q_k.ds);
    cudaFree(gate.d);cudaFree(gate.ds);
    cudaFree(down.d);cudaFree(down.ds);
    cudaStreamDestroy(st);
    return 0;
}
