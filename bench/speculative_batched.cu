// bench/speculative_batched.cu — Batched GEMV speedup for speculative decoding
//
// Key insight: batched GEMV for MLP (gate+up+down) amortizes weight loads.
// This test measures the isolated speedup of batched GEMV vs M× single GEMV.
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/speculative_batched.cu build/libblackwell_kernels.a \
//     -o bench/speculative_batched

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>
#include "blackwell/kernels.h"

struct GpuTimer { cudaEvent_t s,e; GpuTimer(){cudaEventCreate(&s);cudaEventCreate(&e);} ~GpuTimer(){cudaEventDestroy(s);cudaEventDestroy(e);} void start(){cudaEventRecord(s,0);} float stop(){cudaEventRecord(e,0);cudaEventSynchronize(e);float m=0;cudaEventElapsedTime(&m,s,e);return m;} };

struct IW { int K,N; int8_t*d; float*ds; };
static IW load_iw(const char* d, const char* n){
    char p[256]; snprintf(p,256,"%s/%s.int8_t",d,n); FILE*f=fopen(p,"rb");
    if(!f){printf("Cannot open %s\n",p);exit(1);}
    int h[5]; size_t r=fread(h,4,5,f); if(r!=5){printf("Bad header %s\n",p);exit(1);}
    IW w{h[0],h[1],nullptr,nullptr};
    std::vector<int8_t> tmp((size_t)w.K*w.N);
    size_t nr=fread(tmp.data(),1,(size_t)w.K*w.N,f);
    if(nr!=(size_t)w.K*w.N){printf("Bad data %s\n",p);exit(1);}
    fclose(f);
    cudaMalloc(&w.d,(size_t)w.K*w.N);
    cudaMemcpy(w.d,tmp.data(),(size_t)w.K*w.N,cudaMemcpyHostToDevice);
    snprintf(p,256,"%s/%s.scale_t",d,n); f=fopen(p,"rb");
    if(!f){printf("Cannot open %s\n",p);exit(1);}
    r=fread(h,4,5,f); if(r!=5){printf("Bad scale header %s\n",p);exit(1);}
    size_t ns=(size_t)h[3]*h[4];
    std::vector<float> tmp_s(ns);
    nr=fread(tmp_s.data(),4,ns,f);
    if(nr!=ns){printf("Bad scale data %s\n",p);exit(1);}
    fclose(f);
    cudaMalloc(&w.ds,ns*4);
    cudaMemcpy(w.ds,tmp_s.data(),ns*4,cudaMemcpyHostToDevice);
    return w;
}

int main(int argc, char** argv) {
    int M=4, iters=100;
    if(argc>1)M=atoi(argv[1]);
    if(argc>2)iters=atoi(argv[2]);

    cudaDeviceProp p;cudaGetDeviceProperties(&p,0);
    printf("# Batched GEMV Speedup — M=%d drafts\nDevice: %s\n",M,p.name);

    // Load gate+up+down weights
    IW gate=load_iw("weights_int8_bf16","0_mlp.gate_proj");
    IW up=load_iw("weights_int8_bf16","0_mlp.up_proj");
    IW down=load_iw("weights_int8_bf16","0_mlp.down_proj");

    const int H=2048,I=6144;
    float ixv=1.f/127.f;

    // Batch inputs (M tokens, same input for simplicity)
    int8_t*xM;float*xMs;
    cudaMalloc(&xM,M*H);cudaMalloc(&xMs,M*(H/16)*4);
    std::vector<float>xms(M*H/16,ixv);
    cudaMemcpy(xMs,xms.data(),xms.size()*4,cudaMemcpyHostToDevice);
    std::vector<int8_t>xm(M*H,127);
    cudaMemcpy(xM,xm.data(),M*H,cudaMemcpyHostToDevice);

    // Outputs
    float*gM,*uM;cudaMalloc(&gM,M*I*4);cudaMalloc(&uM,M*I*4);
    float*projM;cudaMalloc(&projM,M*H*4);
    float*g,*u;cudaMalloc(&g,I*4);cudaMalloc(&u,I*4);
    float*proj;cudaMalloc(&proj,H*4);

    // Mode A: M × single GEMV
    GpuTimer ta;ta.start();
    for(int i=0;i<iters;++i){
        for(int m=0;m<M;++m){
            blackwell::kernels::gemv_int8(g+m*I,xM+m*H,xMs+m*(H/16),gate.d,gate.ds,H,I,0);
            blackwell::kernels::gemv_int8(u+m*I,xM+m*H,xMs+m*(H/16),up.d,up.ds,H,I,0);
        }
    }
    cudaDeviceSynchronize();float ms_a=ta.stop();

    // Mode B: batched GEMV
    GpuTimer tb;tb.start();
    for(int i=0;i<iters;++i){
        blackwell::kernels::gemv_int8_batched(gM,xM,xMs,gate.d,gate.ds,H,I,M,0);
        blackwell::kernels::gemv_int8_batched(uM,xM,xMs,up.d,up.ds,H,I,M,0);
    }
    cudaDeviceSynchronize();float ms_b=tb.stop();

    // Mode C: M × single down_proj
    GpuTimer tc;tc.start();
    for(int i=0;i<iters;++i){
        for(int m=0;m<M;++m){
            blackwell::kernels::gemv_int8(proj+m*H,xM+m*H,xMs+m*(H/16),down.d,down.ds,H,H,0);
        }
    }
    cudaDeviceSynchronize();float ms_c=tc.stop();

    // Mode D: batched down_proj
    int8_t*mi8M;float*mi8sM;cudaMalloc(&mi8M,M*I);cudaMalloc(&mi8sM,M*(I/16)*4);
    std::vector<float>mims(M*I/16,ixv);cudaMemcpy(mi8sM,mims.data(),mims.size()*4,cudaMemcpyHostToDevice);
    std::vector<int8_t>mim(M*I,127);cudaMemcpy(mi8M,mim.data(),M*I,cudaMemcpyHostToDevice);
    GpuTimer td;td.start();
    for(int i=0;i<iters;++i){
        blackwell::kernels::gemv_int8_batched(projM,mi8M,mi8sM,down.d,down.ds,I,H,M,0);
    }
    cudaDeviceSynchronize();float ms_d=td.stop();

    printf("\n=== GEMV Speedup (M=%d) ===\n",M);
    printf("  %-15s  %8s  %8s  %8s\n","Project","Time","Per-M","Speedup");
    printf("  %-15s  %7.2fus  %7.2fus  %7.2fx\n","M×gemv gate+up",ms_a/iters*1000,ms_a/iters/M*1000,0.f);
    printf("  %-15s  %7.2fus  %7.2fus  %7.2fx\n","Batched gate+up",ms_b/iters*1000,ms_b/iters*1000,ms_a/ms_b);
    printf("  %-15s  %7.2fus  %7.2fus  %7.2fx\n","M×gemv down",ms_c/iters*1000,ms_c/iters/M*1000,0.f);
    printf("  %-15s  %7.2fus  %7.2fus  %7.2fx\n","Batched down",ms_d/iters*1000,ms_d/iters*1000,ms_c/ms_d);
    printf("\n  Batched gate+up speedup: %.2fx\n",ms_a/ms_b);
    printf("  Batched down speedup:     %.2fx\n",ms_c/ms_d);
    printf("  Total batched speedup:    %.2fx\n",(ms_a+ms_c)/(ms_b+ms_d));

    cudaFree(gate.d);cudaFree(gate.ds);cudaFree(up.d);cudaFree(up.ds);cudaFree(down.d);cudaFree(down.ds);
    cudaFree(xM);cudaFree(xMs);cudaFree(gM);cudaFree(uM);cudaFree(projM);cudaFree(g);cudaFree(u);cudaFree(proj);
    cudaFree(mi8M);cudaFree(mi8sM);
    return 0;
}