// bench/decode_int8_pipeline.cu — INT8-only decode pipeline (no FP4)
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/decode_int8_pipeline.cu build/libblackwell_kernels.a \
//     -o bench/decode_int8_pipeline

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include "blackwell/kernels.h"

struct GpuTimer {
    cudaEvent_t s,e;
    GpuTimer(){cudaEventCreate(&s);cudaEventCreate(&e);}
    ~GpuTimer(){cudaEventDestroy(s);cudaEventDestroy(e);}
    void start(){cudaEventRecord(s,0);}
    float stop(){cudaEventRecord(e,0);cudaEventSynchronize(e);float m=0;cudaEventElapsedTime(&m,s,e);return m;}
};
static void chk(cudaError_t e,const char* m){
    if(e!=cudaSuccess){printf("FAIL %s %s\n",m,cudaGetErrorString(e));::exit(1);}}

struct LW { int K,N; std::vector<int8_t> d; std::vector<float> sc; };
static LW lw(const char* p){
    char x[256]; snprintf(x,256,"%s.int8_t",p); FILE*f=fopen(x,"rb");
    int h[5]; fread(h,4,5,f);
    LW w; w.K=h[0]; w.N=h[1];
    w.d.resize(h[0]*h[1]); fread(w.d.data(),1,w.d.size(),f); fclose(f);
    snprintf(x,256,"%s.scale_t",p); f=fopen(x,"rb"); fread(h,4,5,f);
    w.sc.resize(h[3]*h[4]); fread(w.sc.data(),4,w.sc.size(),f); fclose(f); return w;
}
struct DW { int K,N; int8_t*d; float*sc; };
static DW dw(const LW& w){
    DW d; d.K=w.K; d.N=w.N;
    cudaMalloc(&d.d,w.d.size());    cudaMemcpy(d.d,w.d.data(),w.d.size(),cudaMemcpyHostToDevice);
    cudaMalloc(&d.sc,w.sc.size()*4);cudaMemcpy(d.sc,w.sc.data(),w.sc.size()*4,cudaMemcpyHostToDevice); return d;
}
struct L { DW q,k,v,o,g,u,d; };

int main(int argc, char** argv){
    int NL=2, IT=100;
    if(argc>1)NL=atoi(argv[1]);
    if(argc>2)IT=atoi(argv[2]);

    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    printf("# INT8 Pipeline (no FP4) — Qwen3-1.7B\n  Device: %s\n  Layers:%d  Iters:%d\n\n",P.name,NL,IT);

    const int H=2048,Q=2048,KV=1024,I=6144,nqh=12,nkv=12,hd=64,SEQ=128;
    const float IXV=1.f/127.f;
    size_t kv_sz=(size_t)NL*nkv*SEQ*hd*4;

    float *d_x,*d_res,*d_Q,*d_K,*d_V,*d_attn,*d_gate,*d_up,*d_mlp,*d_proj,*d_rn;
    int8_t *d_xi,*d_ai,*d_mi;
    float *d_xs,*d_as,*d_ms_out;
    float *d_kc,*d_vc;

    cudaMalloc(&d_x,H*4); cudaMalloc(&d_res,H*4); cudaMalloc(&d_Q,Q*4);
    cudaMalloc(&d_K,KV*4); cudaMalloc(&d_V,KV*4); cudaMalloc(&d_attn,Q*4);
    cudaMalloc(&d_gate,I*4); cudaMalloc(&d_up,I*4); cudaMalloc(&d_mlp,I*4);
    cudaMalloc(&d_proj,H*4); cudaMalloc(&d_rn,H*4);
    cudaMalloc(&d_xi,H); cudaMalloc(&d_ai,Q); cudaMalloc(&d_mi,I);
    cudaMalloc(&d_xs,(H/16)*4); cudaMalloc(&d_as,(Q/16)*4); cudaMalloc(&d_ms_out,(I/16)*4);
    cudaMalloc(&d_kc,kv_sz); cudaMalloc(&d_vc,kv_sz);

    std::vector<float> ix(H); for(int i=0;i<H;++i)ix[i]=(i%17-8)*0.01f;
    cudaMemcpy(d_x,ix.data(),H*4,cudaMemcpyHostToDevice);
    std::vector<float> xv={IXV},av={IXV},mv={IXV};
    cudaMemcpy(d_xs,xv.data(),(H/16)*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_as,av.data(),(Q/16)*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_ms_out,mv.data(),(I/16)*4,cudaMemcpyHostToDevice);
    cudaMemset(d_kc,0,kv_sz);cudaMemset(d_vc,0,kv_sz);
    std::vector<float> rn(H,1.f);cudaMemcpy(d_rn,rn.data(),H*4,cudaMemcpyHostToDevice);

    printf("Loading %d layers...\n",NL);
    std::vector<L> L(NL); char p[256];
    for(int l=0;l<NL;++l){
        snprintf(p,256,"weights_int8_bf16/%d_self_attn.q_proj",l);L[l].q=dw(lw(p));
        snprintf(p,256,"weights_int8_bf16/%d_self_attn.k_proj",l);L[l].k=dw(lw(p));
        snprintf(p,256,"weights_int8_bf16/%d_self_attn.v_proj",l);L[l].v=dw(lw(p));
        snprintf(p,256,"weights_int8_bf16/%d_self_attn.o_proj",l);L[l].o=dw(lw(p));
        snprintf(p,256,"weights_int8_bf16/%d_mlp.gate_proj",l);  L[l].g=dw(lw(p));
        snprintf(p,256,"weights_int8_bf16/%d_mlp.up_proj",l);    L[l].u=dw(lw(p));
        snprintf(p,256,"weights_int8_bf16/%d_mlp.down_proj",l);  L[l].d=dw(lw(p));
    }
    printf("Loaded.\n\n");

    auto do_layer=[&](int l){
        int kb=l*nkv*SEQ*hd;
        chk(blackwell::kernels::fused_rmsnorm_quant_int8(d_xi,d_xs,d_x,d_rn,H,1e-6f,0),"rn1");
        blackwell::kernels::gemv_int8(d_Q,d_xi,d_xs,L[l].q.d,L[l].q.sc,H,Q,0);
        blackwell::kernels::gemv_int8(d_K,d_xi,d_xs,L[l].k.d,L[l].k.sc,H,KV,0);
        blackwell::kernels::gemv_int8(d_V,d_xi,d_xs,L[l].v.d,L[l].v.sc,H,KV,0);
        blackwell::kernels::update_kv_cache(d_kc+kb,d_vc+kb,d_K,d_V,0,SEQ-1,nkv,hd,SEQ,0);
        blackwell::kernels::attention_decode_gqa(d_attn,d_Q,d_kc+kb,d_vc+kb,SEQ-1,nqh,nkv,hd,SEQ,0);
        blackwell::kernels::pack_int8(d_ai,d_attn,d_as,Q,0);
        blackwell::kernels::gemv_int8(d_proj,d_ai,d_as,L[l].o.d,L[l].o.sc,Q,H,0);
        blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_x,H,0);
        cudaMemcpy(d_res,d_proj,H*4,cudaMemcpyDeviceToDevice);
        chk(blackwell::kernels::fused_rmsnorm_quant_int8(d_xi,d_xs,d_proj,d_rn,H,1e-6f,0),"rn2");
        blackwell::kernels::gemv_int8(d_gate,d_xi,d_xs,L[l].g.d,L[l].g.sc,H,I,0);
        blackwell::kernels::gemv_int8(d_up,d_xi,d_xs,L[l].u.d,L[l].u.sc,H,I,0);
        blackwell::kernels::apply_swiglu(d_mlp,d_gate,d_up,I,0);
        blackwell::kernels::pack_int8(d_mi,d_mlp,d_ms_out,I,0);
        blackwell::kernels::gemv_int8(d_proj,d_mi,d_ms_out,L[l].d.d,L[l].d.sc,I,H,0);
        blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_res,H,0);
        cudaMemcpy(d_x,d_proj,H*4,cudaMemcpyDeviceToDevice);
    };

    // Warmup
    for(int w=0;w<5;++w)for(int l=0;l<NL;++l)do_layer(l);
    cudaDeviceSynchronize();

    // Per-kernel profiling (1 layer, isolated timing)
    printf("=== Per-Kernel Profile (1 layer) ===\n");
    GpuTimer T; struct KT{const char*n;float us;}kt[32];int NK=0;
    cudaMemcpy(d_x,ix.data(),H*4,cudaMemcpyHostToDevice);
    cudaMemset(d_kc,0,kv_sz);cudaMemset(d_vc,0,kv_sz);
    for(int w=0;w<5;++w)do_layer(0);
    cudaDeviceSynchronize();
    cudaMemcpy(d_x,ix.data(),H*4,cudaMemcpyHostToDevice);
    cudaMemset(d_kc,0,kv_sz);cudaMemset(d_vc,0,kv_sz);

    T.start(); blackwell::kernels::fused_rmsnorm_quant_int8(d_xi,d_xs,d_x,d_rn,H,1e-6f,0);
    kt[NK].n="rn_q"; kt[NK++].us=T.stop()*1e6;
    T.start(); blackwell::kernels::gemv_int8(d_Q,d_xi,d_xs,L[0].q.d,L[0].q.sc,H,Q,0);
    kt[NK].n="Q"; kt[NK++].us=T.stop()*1e6;
    T.start(); blackwell::kernels::gemv_int8(d_K,d_xi,d_xs,L[0].k.d,L[0].k.sc,H,KV,0);
    kt[NK].n="K"; kt[NK++].us=T.stop()*1e6;
    T.start(); blackwell::kernels::gemv_int8(d_V,d_xi,d_xs,L[0].v.d,L[0].v.sc,H,KV,0);
    kt[NK].n="V"; kt[NK++].us=T.stop()*1e6;
    T.start(); blackwell::kernels::update_kv_cache(d_kc,d_vc,d_K,d_V,0,SEQ-1,nkv,hd,SEQ,0);
    kt[NK].n="kv"; kt[NK++].us=T.stop()*1e6;
    T.start(); blackwell::kernels::attention_decode_gqa(d_attn,d_Q,d_kc,d_vc,SEQ-1,nqh,nkv,hd,SEQ,0);
    kt[NK].n="attn"; kt[NK++].us=T.stop()*1e6;
    T.start(); blackwell::kernels::pack_int8(d_ai,d_attn,d_as,Q,0);
    kt[NK].n="pack_a"; kt[NK++].us=T.stop()*1e6;
    T.start(); blackwell::kernels::gemv_int8(d_proj,d_ai,d_as,L[0].o.d,L[0].o.sc,Q,H,0);
    kt[NK].n="Wo"; kt[NK++].us=T.stop()*1e6;
    T.start(); blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_x,H,0);
    kt[NK].n="add1"; kt[NK++].us=T.stop()*1e6;
    cudaMemcpy(d_res,d_proj,H*4,cudaMemcpyDeviceToDevice);
    T.start(); blackwell::kernels::fused_rmsnorm_quant_int8(d_xi,d_xs,d_proj,d_rn,H,1e-6f,0);
    kt[NK].n="rn_m"; kt[NK++].us=T.stop()*1e6;
    T.start(); blackwell::kernels::gemv_int8(d_gate,d_xi,d_xs,L[0].g.d,L[0].g.sc,H,I,0);
    kt[NK].n="gate"; kt[NK++].us=T.stop()*1e6;
    T.start(); blackwell::kernels::gemv_int8(d_up,d_xi,d_xs,L[0].u.d,L[0].u.sc,H,I,0);
    kt[NK].n="up"; kt[NK++].us=T.stop()*1e6;
    T.start(); blackwell::kernels::apply_swiglu(d_mlp,d_gate,d_up,I,0);
    kt[NK].n="swiglu"; kt[NK++].us=T.stop()*1e6;
    T.start(); blackwell::kernels::pack_int8(d_mi,d_mlp,d_ms_out,I,0);
    kt[NK].n="pack_m"; kt[NK++].us=T.stop()*1e6;
    T.start(); blackwell::kernels::gemv_int8(d_proj,d_mi,d_ms_out,L[0].d.d,L[0].d.sc,I,H,0);
    kt[NK].n="down"; kt[NK++].us=T.stop()*1e6;
    T.start(); blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_res,H,0);
    kt[NK].n="add2"; kt[NK++].us=T.stop()*1e6;
    cudaDeviceSynchronize();

    float total=0;for(int i=0;i<NK;++i)total+=kt[i].us;
    printf("  1 layer=%.1f us\n",total);
    for(int i=0;i<NK;++i)printf("  %-7s %6.1f us %5.1f%%\n",kt[i].n,kt[i].us,100*kt[i].us/total);

    auto gbw=[](const char*n,int K,int N,float us){
        printf("  %-7s %4dx%-4d: %7.0f GB/s  (%.1f us)\n",n,K,N,2.f*K*N*4/(us*1e-3),us);
    };
    printf("\n  GEMV bandwidth:\n");
    gbw("Q",H,Q,kt[1].us); gbw("K",H,KV,kt[2].us); gbw("V",H,KV,kt[3].us);
    gbw("Wo",Q,H,kt[7].us); gbw("gate",H,I,kt[10].us); gbw("up",H,I,kt[11].us); gbw("down",I,H,kt[14].us);

    const char*G[]={"Q","K","V","Wo","gate","up","down"};
    float gv=0;for(int i=0;i<NK;++i)for(int j=0;j<7;++j)if(!strcmp(kt[i].n,G[j]))gv+=kt[i].us;
    float mlp_gv=kt[10].us+kt[11].us+kt[14].us;
    printf("\n  GEMV:          %.1f us (%.1f%%)\n",gv,100*gv/total);
    printf("  Non-GEMV:       %.1f us (%.1f%%)\n",total-gv,100*(total-gv)/total);
    printf("  MLP GEMVs:     %.1f us (%.1f%% of all GEMV, %.1f%% of layer)\n",
        mlp_gv,100*mlp_gv/gv,100*mlp_gv/total);

    // Wall-clock benchmark
    cudaMemcpy(d_x,ix.data(),H*4,cudaMemcpyHostToDevice);
    cudaMemset(d_kc,0,kv_sz);cudaMemset(d_vc,0,kv_sz);
    GpuTimer TW; TW.start();
    for(int i=0;i<IT;++i)for(int l=0;l<NL;++l)do_layer(l);
    cudaDeviceSynchronize();
    float ms=TW.stop();
    float pt=ms/IT;
    float tps=1000.f/(pt*28.f/NL);
    printf("\n=== Results ===\n");
    printf("  Per-token: %.2f ms  =>  %.1f t/s (28L)\n",pt,tps);
    printf("  CUDA Graph: 122.7 t/s  Speedup: %.2fx\n",tps/122.7f);
    printf("  Kernels/layer: %d  (was 20 with FP4 path)\n",NK);

    cudaFree(d_x);cudaFree(d_res);cudaFree(d_Q);cudaFree(d_K);cudaFree(d_V);
    cudaFree(d_attn);cudaFree(d_gate);cudaFree(d_up);cudaFree(d_mlp);cudaFree(d_proj);
    cudaFree(d_xi);cudaFree(d_ai);cudaFree(d_mi);
    cudaFree(d_xs);cudaFree(d_as);cudaFree(d_ms_out);
    cudaFree(d_kc);cudaFree(d_vc);cudaFree(d_rn);
    for(int l=0;l<NL;++l){
        cudaFree(L[l].q.d);cudaFree(L[l].q.sc);cudaFree(L[l].k.d);cudaFree(L[l].k.sc);
        cudaFree(L[l].v.d);cudaFree(L[l].v.sc);cudaFree(L[l].o.d);cudaFree(L[l].o.sc);
        cudaFree(L[l].g.d);cudaFree(L[l].g.sc);cudaFree(L[l].u.d);cudaFree(L[l].u.sc);
        cudaFree(L[l].d.d);cudaFree(L[l].d.sc);
    }
    return 0;
}
