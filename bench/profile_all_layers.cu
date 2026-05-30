// bench/profile_all_layers.cu — Layer-by-layer GEMV perf
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/profile_all_layers.cu build/libblackwell_kernels.a \
//     -o bench/profile_all_layers

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <cstdint>
#include "blackwell/kernels.h"

struct GpuTimer { cudaEvent_t s,e; GpuTimer(){cudaEventCreate(&s);cudaEventCreate(&e);}
    ~GpuTimer(){cudaEventDestroy(s);cudaEventDestroy(e);}
    void start(cudaStream_t st=0){cudaEventRecord(s,st);}
    float stop(cudaStream_t st=0){cudaEventRecord(e,st);cudaEventSynchronize(e);float m=0;cudaEventElapsedTime(&m,s,e);return m;} };

struct IW { int K,N; int8_t*d; float*ds; };
static IW load_iw(const char* d, const char* n){
    char p[256]; snprintf(p,256,"%s/%s.int8_t",d,n); FILE*f=fopen(p,"rb");
    int h[5]; fread(h,4,5,f); IW w{h[0],h[1],nullptr,nullptr};
    std::vector<int8_t>tmp(w.K*w.N); fread(tmp.data(),1,w.K*w.N,f); fclose(f);
    cudaMalloc(&w.d,w.K*w.N); cudaMemcpy(w.d,tmp.data(),w.K*w.N,cudaMemcpyHostToDevice);
    snprintf(p,256,"%s/%s.scale_t",d,n); f=fopen(p,"rb"); fread(h,4,5,f);
    size_t ns=h[3]*h[4]; std::vector<float>tmp_s(ns); fread(tmp_s.data(),4,ns,f); fclose(f);
    cudaMalloc(&w.ds,ns*4); cudaMemcpy(w.ds,tmp_s.data(),ns*4,cudaMemcpyHostToDevice);
    return w;
}

int main(int argc, char** argv){
    int L=28, iters=10;
    if(argc>1)L=atoi(argv[1]);
    if(argc>2)iters=atoi(argv[2]);

    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("# Layer-by-layer GEMV profile (%d layers, %d iters)\nDevice: %s\n\n",L,iters,p.name);

    const int H=2048,I=6144,KV=1024;
    std::vector<IW> g(L), u(L), d(L);
    for(int l=0;l<L;++l){
        char b[256];
        snprintf(b,256,"%d_mlp.gate_proj",l); g[l]=load_iw("weights_int8_bf16",b);
        snprintf(b,256,"%d_mlp.up_proj",l);   u[l]=load_iw("weights_int8_bf16",b);
        snprintf(b,256,"%d_mlp.down_proj",l); d[l]=load_iw("weights_int8_bf16",b);
    }

    // Shared input (all non-zero to avoid trivial-zero optimizations)
    float ixv=1.f/127.f;
    int8_t*x8;float*xs8;cudaMalloc(&x8,H);cudaMalloc(&xs8,H/16*4);
    std::vector<int8_t> xv(H,127);cudaMemcpy(x8,xv.data(),H,cudaMemcpyHostToDevice);
    std::vector<float> x8s(H/16,ixv);cudaMemcpy(xs8,x8s.data(),H/16*4,cudaMemcpyHostToDevice);
    float*out;cudaMalloc(&out,I*4);

    printf("  L     gate      up      down  |  gate_CV  up_CV  down_CV  (us)\n");
    printf("  --   ------   ------   ------ |  ------  ------  ------\n");

    // Per-layer timing, each GEMV isolated (no loop, just individual times)
    std::vector<float> t_gate(L), t_up(L), t_down(L);
    for(int l=0;l<L;++l){
        GpuTimer t;
        t.start(); for(int i=0;i<iters;++i) blackwell::kernels::gemv_int8_warp(out,x8,xs8,g[l].d,g[l].ds,H,I,0);
        t_gate[l]=t.stop()/iters*1e6;
        t.start(); for(int i=0;i<iters;++i) blackwell::kernels::gemv_int8_warp(out,x8,xs8,u[l].d,u[l].ds,H,I,0);
        t_up[l]=t.stop()/iters*1e6;
        t.start(); for(int i=0;i<iters;++i) blackwell::kernels::gemv_int8_warp(out,x8,xs8,d[l].d,d[l].ds,H,I,0);
        t_down[l]=t.stop()/iters*1e6;
        printf("  %2d  %7.2f  %7.2f  %7.2f\n",l,t_gate[l],t_up[l],t_down[l]);
    }

    // Stats
    float gl=0,gu=0,gd=0,gl0=t_gate[0],gu0=t_up[0],gd0=t_down[0];
    for(int l=0;l<L;++l){ gl+=t_gate[l]; gu+=t_up[l]; gd+=t_down[l]; }
    gl/=L;gu/=L;gd/=L;
    float gsd=0,usd=0,dsd=0;
    for(int l=0;l<L;++l){ gsd+=(t_gate[l]-gl)*(t_gate[l]-gl); usd+=(t_up[l]-gu)*(t_up[l]-gu); dsd+=(t_down[l]-gd)*(t_down[l]-gd); }
    gsd=sqrt(gsd/L);usd=sqrt(usd/L);dsd=sqrt(dsd/L);

    printf("\n  Summary:\n");
    printf("  gate  avg=%.1fus (stdev=%.1f) | L0=%.1f | slowdown L0→avg=%.2fx\n", gl,gsd,gl0,gl/gl0);
    printf("  up    avg=%.1fus (stdev=%.1f) | L0=%.1f | slowdown L0→avg=%.2fx\n", gu,usd,gu0,gu/gu0);
    printf("  down  avg=%.1fus (stdev=%.1f) | L0=%.1f | slowdown L0→avg=%.2fx\n", gd,dsd,gd0,gd/gd0);
    printf("  \n");
    printf("  gate+up+down total: %.2fms (L0), %.2fms (avg)\n", (gl0+gu0+gd0)*1e-3, (gl+gu+gd)*1e-3);
    printf("  vs decode_int8_cgraph layer time: ~210us\n");
    printf("  BERT: if all 28 layers had avg times → per-token=%.1fus\n", (gl+gu+gd)/3*28*1e-3);

    // Sequential warmup: time each layer after warming up previous ones
    printf("\n  Sequential L2 warmup test (iters=%d):\n", iters);
    GpuTimer t;
    t.start();
    for(int i=0;i<iters;++i){
        for(int l=0;l<L;++l){
            blackwell::kernels::gemv_int8_warp(out,x8,xs8,g[l].d,g[l].ds,H,I,0);
            blackwell::kernels::gemv_int8_warp(out,x8,xs8,u[l].d,u[l].ds,H,I,0);
            blackwell::kernels::gemv_int8_warp(out,x8,xs8,d[l].d,d[l].ds,H,I,0);
        }
    }
    cudaDeviceSynchronize();
    float ms_seq=t.stop();
    printf("  Sequential (all 28L) total: %.2fms (%.3fms per layer)\n", ms_seq, ms_seq/L/iters*1e3);
    printf("  Sequential vs isolated average: %.2fx\n", ms_seq*3*L/(L*(gl+gu+gd)*iters*1e-3));

    cudaFree(x8);cudaFree(xs8);cudaFree(out);
    for(int l=0;l<L;++l){
        cudaFree(g[l].d);cudaFree(g[l].ds);cudaFree(u[l].d);cudaFree(u[l].ds);cudaFree(d[l].d);cudaFree(d[l].ds);
    }
    return 0;
}
