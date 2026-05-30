// bench/decode_batched_gemv_cgraph.cu — All GEMVs batched (M sequences per call)
//
// Key change: gemv_int8 → gemv_int8_batched for ALL GEMVs.
// Each GEMV call processes M sequences simultaneously.
// Expected: ~M× throughput increase vs single-sequence CUDA Graph.
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/decode_batched_gemv_cgraph.cu build/libblackwell_kernels.a \
//     -o bench/decode_batched_gemv_cgraph

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "blackwell/kernels.h"

static void die(cudaError_t e, const char* m){
    if(e!=cudaSuccess){printf("FAIL %s %s\n",m,cudaGetErrorString(e));::exit(1);}}
struct Tmr { cudaEvent_t s_,e_; Tmr(){cudaEventCreate(&s_);cudaEventCreate(&e_);}
    ~Tmr(){cudaEventDestroy(s_);cudaEventDestroy(e_);}
    void start(){cudaEventRecord(s_,0);}
    float stop(){cudaEventRecord(e_,0);cudaEventSynchronize(e_);float ms;cudaEventElapsedTime(&ms,s_,e_);return ms;} };

struct LW { std::vector<int8_t> d; std::vector<float> sc; };
static LW lw(const char*p){
    char x[256]; snprintf(x,256,"%s.int8_t",p); FILE*f=fopen(x,"rb");
    int h[5]; fread(h,4,5,f); LW w;
    w.d.resize(h[0]*h[1]); fread(w.d.data(),1,w.d.size(),f); fclose(f);
    snprintf(x,256,"%s.scale_t",p); f=fopen(x,"rb"); fread(h,4,5,f);
    w.sc.resize(h[3]*h[4]); fread(w.sc.data(),4,w.sc.size(),f); fclose(f); return w;
}
struct DW { int8_t*d; float*sc; };
static DW dw(const LW& w){
    DW d;
    cudaMalloc(&d.d,w.d.size());    cudaMemcpy(d.d,w.d.data(),w.d.size(),cudaMemcpyHostToDevice);
    cudaMalloc(&d.sc,w.sc.size()*4);cudaMemcpy(d.sc,w.sc.data(),w.sc.size()*4,cudaMemcpyHostToDevice); return d;
}
struct L { DW q,k,v,o,g,u,d; };

int main(int argc, char** argv){
    int NL=2, M=4, IT=50;
    if(argc>1)NL=atoi(argv[1]);
    if(argc>2)M=atoi(argv[2]);
    if(argc>3)IT=atoi(argv[3]);
    if(M>8){printf("M max 8\n");return 0;}

    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    printf("# All-Batched GEMV CUDA Graph — Qwen3-1.7B\n");
    printf("  Device: %s\n  L:%d M:%d IT:%d\n\n",P.name,NL,M,IT);

    const int H=2048,Q=2048,KV=1024,I=6144,nqh=12,nkv=12,hd=64,SEQ=128;
    const float IXV=1.f/127.f;

    // Batched buffers: contiguous M×H, M×Q, etc.
    float *d_x;      // M×H FP32 input (contiguous)
    int8_t *d_xi;    // M×H INT8
    float *d_xs;     // M×(H/16) scale
    float *d_res;    // M×H residual
    float *d_Q;      // M×Q Q projections
    float *d_K;      // M×KV K projections
    float *d_V;      // M×KV V projections
    float *d_attn;   // M×Q attention output
    int8_t *d_ai;    // M×Q attn INT8
    float *d_as;     // M×(Q/16) scale
    float *d_gate;   // M×I gate output
    float *d_up;     // M×I up output
    float *d_mlp;    // M×I mlp (swiglu output)
    int8_t *d_mi;    // M×I mlp INT8
    float *d_ms;     // M×(I/16) scale
    float *d_proj;   // M×H projected output

    #define ALLOC(p,n) die(cudaMalloc(&(p),(n)),#p)
    ALLOC(d_x,M*H*4);ALLOC(d_xi,M*H);ALLOC(d_xs,M*(H/16)*4);
    ALLOC(d_res,M*H*4);ALLOC(d_Q,M*Q*4);ALLOC(d_K,M*KV*4);ALLOC(d_V,M*KV*4);
    ALLOC(d_attn,M*Q*4);ALLOC(d_ai,M*Q);ALLOC(d_as,M*(Q/16)*4);
    ALLOC(d_gate,M*I*4);ALLOC(d_up,M*I*4);ALLOC(d_mlp,M*I*4);
    ALLOC(d_mi,M*I);ALLOC(d_ms,M*(I/16)*4);ALLOC(d_proj,M*H*4);
    float*d_kc,*d_vc;ALLOC(d_kc,nkv*SEQ*hd*4);ALLOC(d_vc,nkv*SEQ*hd*4);
    float*d_rn;ALLOC(d_rn,H*4);
    #undef ALLOC

    std::vector<float> x0(H); for(int i=0;i<H;++i)x0[i]=(i%17-8)*0.01f;
    std::vector<float> xv={(float)IXV};
    std::vector<float> rn(H,1.f);
    cudaMemcpy(d_rn,rn.data(),H*4,cudaMemcpyHostToDevice);
    // Copy x0 into all M slots of d_x
    for(int m=0;m<M;++m) cudaMemcpy((float*)d_x+m*H,x0.data(),H*4,cudaMemcpyHostToDevice);
    // Copy scales to all batch positions
    for(int m=0;m<M;++m){
        cudaMemcpy((float*)d_xs+m*(H/16),xv.data(),(H/16)*4,cudaMemcpyHostToDevice);
        cudaMemcpy((float*)d_as+m*(Q/16),xv.data(),(Q/16)*4,cudaMemcpyHostToDevice);
        cudaMemcpy((float*)d_ms+m*(I/16),xv.data(),(I/16)*4,cudaMemcpyHostToDevice);
    }
    cudaMemset(d_kc,0,nkv*SEQ*hd*4);cudaMemset(d_vc,0,nkv*SEQ*hd*4);

    printf("Loading %d layers...\n",NL);
    std::vector<L> W(NL); char p[256];
    for(int l=0;l<NL;++l){
        snprintf(p,256,"weights_int8_bf16/%d_self_attn.q_proj",l);W[l].q=dw(lw(p));
        snprintf(p,256,"weights_int8_bf16/%d_self_attn.k_proj",l);W[l].k=dw(lw(p));
        snprintf(p,256,"weights_int8_bf16/%d_self_attn.v_proj",l);W[l].v=dw(lw(p));
        snprintf(p,256,"weights_int8_bf16/%d_self_attn.o_proj",l);W[l].o=dw(lw(p));
        snprintf(p,256,"weights_int8_bf16/%d_mlp.gate_proj",l);  W[l].g=dw(lw(p));
        snprintf(p,256,"weights_int8_bf16/%d_mlp.up_proj",l);    W[l].u=dw(lw(p));
        snprintf(p,256,"weights_int8_bf16/%d_mlp.down_proj",l);  W[l].d=dw(lw(p));
    }
    printf("Loaded.\n");

    cudaStream_t st; die(cudaStreamCreate(&st),"stream");

    // Fully-batched layer pass
    auto do_batched_layer=[&](int l){
        // RMSNorm + quantize (batched: one kernel processes all M inputs)
        blackwell::kernels::fused_rmsnorm_quant_int8(
            (int8_t*)d_xi,(float*)d_xs,(float*)d_x,d_rn,H,1e-5f,st);
        // Q, K, V GEMVs — batched
        blackwell::kernels::gemv_int8_batched(
            (float*)d_Q,(int8_t*)d_xi,(float*)d_xs,W[l].q.d,W[l].q.sc,M,H,Q,st);
        blackwell::kernels::gemv_int8_batched(
            (float*)d_K,(int8_t*)d_xi,(float*)d_xs,W[l].k.d,W[l].k.sc,M,H,KV,st);
        blackwell::kernels::gemv_int8_batched(
            (float*)d_V,(int8_t*)d_xi,(float*)d_xs,W[l].v.d,W[l].v.sc,M,H,KV,st);
        // KV cache update — per sequence (M separate slots)
        for(int m=0;m<M;++m)
            blackwell::kernels::update_kv_cache(
                (float*)d_kc+m*nkv*SEQ*hd,(float*)d_vc+m*nkv*SEQ*hd,
                (float*)d_K+m*KV*4,(float*)d_V+m*KV*4,0,SEQ-1,nkv,hd,SEQ,st);
        // Attention — per sequence (each seq has its own KV cache slot)
        for(int m=0;m<M;++m)
            blackwell::kernels::attention_decode_gqa(
                (float*)d_attn+m*Q*4,(float*)d_Q+m*Q*4,
                (float*)d_kc+m*nkv*SEQ*hd,(float*)d_vc+m*nkv*SEQ*hd,
                SEQ-1,nqh,nkv,hd,SEQ,st);
        // Wo — batched
        for(int m=0;m<M;++m)
            blackwell::kernels::pack_int8((int8_t*)d_ai+m*Q,(float*)d_attn+m*Q*4,(float*)d_as+m*(Q/16),Q,st);
        blackwell::kernels::gemv_int8_batched(
            (float*)d_proj,(int8_t*)d_ai,(float*)d_as,W[l].o.d,W[l].o.sc,M,Q,H,st);
        for(int m=0;m<M;++m)
            blackwell::kernels::vector_add_fp32((float*)d_proj+m*H,(float*)d_proj+m*H,(float*)d_x+m*H,H,st);
        // Save residual for MLP
        for(int m=0;m<M;++m)
            blackwell::kernels::vector_add_fp32((float*)d_res+m*H,(float*)d_proj+m*H,(float*)d_x+m*H,H,st);
        // MLP RMSNorm + GEMVs — batched
        blackwell::kernels::fused_rmsnorm_quant_int8(
            (int8_t*)d_xi,(float*)d_xs,(float*)d_proj,d_rn,H,1e-5f,st);
        blackwell::kernels::gemv_int8_batched(
            (float*)d_gate,(int8_t*)d_xi,(float*)d_xs,W[l].g.d,W[l].g.sc,M,H,I,st);
        blackwell::kernels::gemv_int8_batched(
            (float*)d_up,(int8_t*)d_xi,(float*)d_xs,W[l].u.d,W[l].u.sc,M,H,I,st);
        for(int m=0;m<M;++m)
            blackwell::kernels::apply_swiglu((float*)d_mlp+m*I*4,(float*)d_gate+m*I*4,(float*)d_up+m*I*4,I,st);
        for(int m=0;m<M;++m)
            blackwell::kernels::pack_int8((int8_t*)d_mi+m*I,(float*)d_mlp+m*I*4,(float*)d_ms+m*(I/16),I,st);
        blackwell::kernels::gemv_int8_batched(
            (float*)d_proj,(int8_t*)d_mi,(float*)d_ms,W[l].d.d,W[l].d.sc,M,I,H,st);
        for(int m=0;m<M;++m)
            blackwell::kernels::vector_add_fp32((float*)d_proj+m*H,(float*)d_proj+m*H,(float*)d_res+m*H,H,st);
        // Copy proj to x (async, for next layer)
        for(int m=0;m<M;++m)
            cudaMemcpyAsync((float*)d_x+m*H,(float*)d_proj+m*H,H*4,cudaMemcpyDeviceToDevice,st);
    };

    // Per-kernel baseline (M× separate calls, like decode_batched_cgraph)
    auto do_per_seq_layer=[&](int l){
        for(int m=0;m<M;++m){
            blackwell::kernels::fused_rmsnorm_quant_int8(
                (int8_t*)d_xi+m*H,(float*)d_xs+m*(H/16),(float*)d_x+m*H,d_rn,H,1e-6f,st);
            blackwell::kernels::gemv_int8_warp(
                (float*)d_Q+m*Q*4,(int8_t*)d_xi+m*H,(float*)d_xs+m*(H/16),W[l].q.d,W[l].q.sc,H,Q,st);
            blackwell::kernels::gemv_int8_warp(
                (float*)d_K+m*KV*4,(int8_t*)d_xi+m*H,(float*)d_xs+m*(H/16),W[l].k.d,W[l].k.sc,H,KV,st);
            blackwell::kernels::gemv_int8_warp(
                (float*)d_V+m*KV*4,(int8_t*)d_xi+m*H,(float*)d_xs+m*(H/16),W[l].v.d,W[l].v.sc,H,KV,st);
            int kb_m=m*nkv*SEQ*hd;
            blackwell::kernels::update_kv_cache(
                (float*)d_kc+kb_m,(float*)d_vc+kb_m,
                (float*)d_K+m*KV*4,(float*)d_V+m*KV*4,0,SEQ-1,nkv,hd,SEQ,st);
            blackwell::kernels::attention_decode_gqa(
                (float*)d_attn+m*Q*4,(float*)d_Q+m*Q*4,
                (float*)d_kc+kb_m,(float*)d_vc+kb_m,SEQ-1,nqh,nkv,hd,SEQ,st);
            blackwell::kernels::pack_int8((int8_t*)d_ai+m*Q,(float*)d_attn+m*Q*4,(float*)d_as+m*(Q/16),Q,st);
            blackwell::kernels::gemv_int8_warp(
                (float*)d_proj+m*H,(int8_t*)d_ai+m*Q,(float*)d_as+m*(Q/16),W[l].o.d,W[l].o.sc,Q,H,st);
            blackwell::kernels::vector_add_fp32((float*)d_proj+m*H,(float*)d_proj+m*H,(float*)d_x+m*H,H,st);
            blackwell::kernels::vector_add_fp32((float*)d_res+m*H,(float*)d_proj+m*H,(float*)d_x+m*H,H,st);
            blackwell::kernels::fused_rmsnorm_quant_int8(
                (int8_t*)d_xi+m*H,(float*)d_xs+m*(H/16),(float*)d_proj+m*H,d_rn,H,1e-6f,st);
            blackwell::kernels::gemv_int8_warp(
                (float*)d_gate+m*I*4,(int8_t*)d_xi+m*H,(float*)d_xs+m*(H/16),W[l].g.d,W[l].g.sc,H,I,st);
            blackwell::kernels::gemv_int8_warp(
                (float*)d_up+m*I*4,(int8_t*)d_xi+m*H,(float*)d_xs+m*(H/16),W[l].u.d,W[l].u.sc,H,I,st);
            blackwell::kernels::apply_swiglu((float*)d_mlp+m*I*4,(float*)d_gate+m*I*4,(float*)d_up+m*I*4,I,st);
            blackwell::kernels::pack_int8((int8_t*)d_mi+m*I,(float*)d_mlp+m*I*4,(float*)d_ms+m*(I/16),I,st);
            blackwell::kernels::gemv_int8_warp(
                (float*)d_proj+m*H,(int8_t*)d_mi+m*I,(float*)d_ms+m*(I/16),W[l].d.d,W[l].d.sc,I,H,st);
            blackwell::kernels::vector_add_fp32((float*)d_proj+m*H,(float*)d_proj+m*H,(float*)d_res+m*H,H,st);
            cudaMemcpyAsync((float*)d_x+m*H,(float*)d_proj+m*H,H*4,cudaMemcpyDeviceToDevice,st);
        }
    };

    auto reset_x=[&](){
        for(int m=0;m<M;++m) cudaMemcpy((float*)d_x+m*H,x0.data(),H*4,cudaMemcpyHostToDevice);
    };

    // Mode A: Per-sequence baseline
    reset_x(); cudaStreamSynchronize(st);
    Tmr TA; TA.start();
    for(int i=0;i<IT;++i) for(int l=0;l<NL;++l) do_per_seq_layer(l);
    cudaStreamSynchronize(st);
    float ms_A=TA.stop();
    float pt_A=ms_A/(IT*M);
    printf("\n=== Per-sequence (M×, per-kernel) ===\n");
    printf("  %.3fms/token  %.1f t/s  (batch %.1f t/s)\n",
        pt_A, 1000.f/pt_A, M*1000.f/pt_A);

    // Mode B: Batched GEMV per-kernel
    reset_x(); cudaStreamSynchronize(st);
    Tmr TB; TB.start();
    for(int i=0;i<IT;++i) for(int l=0;l<NL;++l) do_batched_layer(l);
    cudaStreamSynchronize(st);
    float ms_B=TB.stop();
    float pt_B=ms_B/(IT*M);
    printf("\n=== Batched GEMV (per-kernel, M=%d) ===\n",M);
    printf("  %.3fms/token  %.1f t/s  (batch %.1f t/s)\n",
        pt_B, 1000.f/pt_B, M*1000.f/pt_B);

    // Mode C: Batched GEMV + CUDA Graph
    reset_x(); cudaStreamSynchronize(st);
    cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);
    for(int l=0;l<NL;++l) do_batched_layer(l);
    cudaGraph_t graph; die(cudaStreamEndCapture(st,&graph),"capture");
    cudaGraphExec_t ge; die(cudaGraphInstantiate(&ge,graph,NULL,NULL,0),"inst");
    for(int w=0;w<5;++w){ die(cudaGraphLaunch(ge,st),"warmup"); }
    cudaStreamSynchronize(st);
    reset_x(); cudaStreamSynchronize(st);
    Tmr TC; TC.start();
    for(int i=0;i<IT;++i) die(cudaGraphLaunch(ge,st),"launch");
    cudaStreamSynchronize(st);
    float ms_C=TC.stop();
    float pt_C=ms_C/(IT*M);
    printf("\n=== Batched GEMV + CUDA Graph (M=%d) ===\n",M);
    printf("  %.3fms/token  %.1f t/s  (batch %.1f t/s)\n",
        pt_C, 1000.f/pt_C, M*1000.f/pt_C);

    printf("\n=== Summary ===\n");
    printf("  %-32s  %8s  %10s\n","Method","ms/token","batch t/s");
    printf("  %-32s  %7.3fms  %10.1f\n","Per-sequence (M×)",pt_A,M*1000.f/pt_A);
    printf("  %-32s  %7.3fms  %10.1f\n","Batched GEMV (per-kernel)",pt_B,M*1000.f/pt_B);
    printf("  %-32s  %7.3fms  %10.1f\n","Batched GEMV + CUDA Graph",pt_C,M*1000.f/pt_C);
    printf("\n  Batched speedup vs per-seq: %.2fx\n", pt_A/pt_B);
    printf("  CUDA Graph speedup: %.2fx\n", ms_B/ms_C);
    printf("  Total: %.2fx\n", pt_A/pt_C);
    printf("  Effective: %.1f t/s\n",M*1000.f/pt_C);
    printf("  vs single-seq CUDA Graph: 122.7 t/s\n");
    printf("  vs llama.cpp: 114.0 t/s\n");
    printf("  Per-sequence: %.1f t/s\n", 1000.f/pt_C);

    cudaGraphDestroy(graph); cudaGraphExecDestroy(ge);
    cudaStreamDestroy(st);
    cudaFree(d_x);cudaFree(d_xi);cudaFree(d_xs);cudaFree(d_res);
    cudaFree(d_Q);cudaFree(d_K);cudaFree(d_V);cudaFree(d_attn);
    cudaFree(d_ai);cudaFree(d_as);cudaFree(d_gate);cudaFree(d_up);cudaFree(d_mlp);cudaFree(d_mi);cudaFree(d_ms);cudaFree(d_proj);
    cudaFree(d_kc);cudaFree(d_vc);cudaFree(d_rn);
    for(int l=0;l<NL;++l){cudaFree(W[l].q.d);cudaFree(W[l].q.sc);cudaFree(W[l].k.d);cudaFree(W[l].k.sc);
        cudaFree(W[l].v.d);cudaFree(W[l].v.sc);cudaFree(W[l].o.d);cudaFree(W[l].o.sc);
        cudaFree(W[l].g.d);cudaFree(W[l].g.sc);cudaFree(W[l].u.d);cudaFree(W[l].u.sc);cudaFree(W[l].d.d);cudaFree(W[l].d.sc);}
    return 0;
}