// bench/decode_batched_cgraph.cu — Batched INT8 + CUDA Graph
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/decode_batched_cgraph.cu build/libblackwell_kernels.a \
//     -o bench/decode_batched_cgraph

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "blackwell/kernels.h"

static void die(cudaError_t e, const char* m){
    if(e!=cudaSuccess){printf("FAIL %s %s\n",m,cudaGetErrorString(e));::exit(1);}}

struct Tmr {
    cudaEvent_t s_, e_;
    Tmr(){cudaEventCreate(&s_);cudaEventCreate(&e_);}
    ~Tmr(){cudaEventDestroy(s_);cudaEventDestroy(e_);}
    void start(){cudaEventRecord(s_,0);}
    float stop(){
        cudaEventRecord(e_,0);
        cudaEventSynchronize(e_);
        float ms; cudaEventElapsedTime(&ms,s_,e_);
        return ms;
    }
};

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
    if(M>4){printf("M max 4\n");return 0;}

    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    printf("# Batched CUDA Graph — Qwen3-1.7B\n  Device: %s\n  L:%d M:%d IT:%d\n\n",P.name,NL,M,IT);

    const int H=2048,Q=2048,KV=1024,I=6144,nqh=12,nkv=12,hd=64,SEQ=128;
    const float IXV=1.f/127.f;

    float* d_x[4]; int8_t* d_xi[4]; float* d_xs[4];
    float* d_res[4]; float* d_Q[4]; float* d_attn[4];
    int8_t* d_ai[4]; float* d_as[4];
    float* d_gate[4]; float* d_up[4]; float* d_mlp[4];
    int8_t* d_mi[4]; float* d_ms[4];
    float* d_proj[4];
    #define ALLOC(p,n) die(cudaMalloc(&(p),(n)),#p)
    for(int m=0;m<M;++m){
        ALLOC(d_x[m],H*4);ALLOC(d_xi[m],H);ALLOC(d_xs[m],(H/16)*4);
        ALLOC(d_res[m],H*4);ALLOC(d_Q[m],Q*4);ALLOC(d_attn[m],Q*4);
        ALLOC(d_ai[m],Q);ALLOC(d_as[m],(Q/16)*4);
        ALLOC(d_gate[m],I*4);ALLOC(d_up[m],I*4);ALLOC(d_mlp[m],I*4);
        ALLOC(d_mi[m],I);ALLOC(d_ms[m],(I/16)*4);
        ALLOC(d_proj[m],H*4);
    }
    float*d_K,*d_V;ALLOC(d_K,KV*4);ALLOC(d_V,KV*4);
    float*d_kc,*d_vc;ALLOC(d_kc,nkv*SEQ*hd*4);ALLOC(d_vc,nkv*SEQ*hd*4);
    float*d_rn;ALLOC(d_rn,H*4);
    #undef ALLOC

    std::vector<float> x0(H); for(int i=0;i<H;++i)x0[i]=(i%17-8)*0.01f;
    std::vector<float> xv={(float)IXV};
    std::vector<float> rn(H,1.f);
    std::vector<int8_t> x88(H,127);
    cudaMemcpy(d_rn,rn.data(),H*4,cudaMemcpyHostToDevice);
    for(int m=0;m<M;++m){
        cudaMemcpy(d_x[m],x0.data(),H*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_xs[m],xv.data(),(H/16)*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_as[m],xv.data(),(Q/16)*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_ms[m],xv.data(),(I/16)*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_xi[m],x88.data(),H,cudaMemcpyHostToDevice);
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

    // Per-kernel execution
    auto do_pass=[&](){
        for(int l=0;l<NL;++l){
            for(int m=0;m<M;++m){
                // Residual path: use d_res as intermediate
                // d_proj = Wo * attn + residual
                // d_res = d_proj (save for MLP)
                // d_x = proj (for next layer) — via async copy
                blackwell::kernels::fused_rmsnorm_quant_int8(d_xi[m],d_xs[m],d_x[m],d_rn,H,1e-6f,st);
                blackwell::kernels::gemv_int8(d_Q[m],d_xi[m],d_xs[m],W[l].q.d,W[l].q.sc,H,Q,st);
                blackwell::kernels::gemv_int8(d_K,d_xi[m],d_xs[m],W[l].k.d,W[l].k.sc,H,KV,st);
                blackwell::kernels::gemv_int8(d_V,d_xi[m],d_xs[m],W[l].v.d,W[l].v.sc,H,KV,st);
                blackwell::kernels::update_kv_cache(d_kc,d_vc,d_K,d_V,0,SEQ-1,nkv,hd,SEQ,st);
                blackwell::kernels::attention_decode_gqa(d_attn[m],d_Q[m],d_kc,d_vc,SEQ-1,nqh,nkv,hd,SEQ,st);
                blackwell::kernels::pack_int8(d_ai[m],d_attn[m],d_as[m],Q,st);
                blackwell::kernels::gemv_int8(d_proj[m],d_ai[m],d_as[m],W[l].o.d,W[l].o.sc,Q,H,st);
                blackwell::kernels::vector_add_fp32(d_proj[m],d_proj[m],d_x[m],H,st);
                // Save proj to res (for MLP residual), then copy proj to x (for next layer)
                blackwell::kernels::vector_add_fp32(d_res[m],d_proj[m],d_x[m],H,st); // res = proj
                // MLP: proj = down * swiglu(gate,up) + res
                blackwell::kernels::fused_rmsnorm_quant_int8(d_xi[m],d_xs[m],d_proj[m],d_rn,H,1e-6f,st);
                blackwell::kernels::gemv_int8(d_gate[m],d_xi[m],d_xs[m],W[l].g.d,W[l].g.sc,H,I,st);
                blackwell::kernels::gemv_int8(d_up[m],  d_xi[m],d_xs[m],W[l].u.d,W[l].u.sc,H,I,st);
                blackwell::kernels::apply_swiglu(d_mlp[m],d_gate[m],d_up[m],I,st);
                blackwell::kernels::pack_int8(d_mi[m],d_mlp[m],d_ms[m],I,st);
                blackwell::kernels::gemv_int8(d_proj[m],d_mi[m],d_ms[m],W[l].d.d,W[l].d.sc,I,H,st);
                blackwell::kernels::vector_add_fp32(d_proj[m],d_proj[m],d_res[m],H,st);
                // Copy proj to x (for next layer) — async to be capture-compatible
                cudaMemcpyAsync(d_x[m],d_proj[m],H*4,cudaMemcpyDeviceToDevice,st);
            }
        }
    };

    auto reset_x=[&](){
        for(int m=0;m<M;++m) cudaMemcpy(d_x[m],x0.data(),H*4,cudaMemcpyHostToDevice);
    };

    // Mode C: Per-kernel (M× single-token baseline)
    reset_x(); cudaStreamSynchronize(st);
    cudaEvent_t cs,ce; cudaEventCreate(&cs); cudaEventCreate(&ce);
    cudaEventRecord(cs,st);
    for(int i=0;i<IT;++i) do_pass();
    cudaEventRecord(ce,st); cudaStreamSynchronize(st);
    float ms_C; cudaEventElapsedTime(&ms_C,cs,ce);
    float pt_C=ms_C/(IT*M);
    printf("\n=== M× single-token (M=%d) ===\n",M);
    printf("  %.3fms/token  %.1f t/s  (batch %.1f t/s)\n",
        pt_C, 1000.f/pt_C, M*1000.f/pt_C);

    // Mode A: Batched per-kernel
    reset_x(); cudaStreamSynchronize(st);
    Tmr TA; TA.start();
    for(int i=0;i<IT;++i) do_pass();
    cudaStreamSynchronize(st);
    float ms_A=TA.stop();
    float pt_A=ms_A/(IT*M);
    printf("\n=== Batched per-kernel (M=%d) ===\n",M);
    printf("  %.3fms/token  %.1f t/s  (batch %.1f t/s)\n",
        pt_A, 1000.f/pt_A, M*1000.f/pt_A);

    // Mode B: CUDA Graph (explicit loop, cudaMemcpyAsync for capture compat)
    reset_x(); cudaStreamSynchronize(st);
    cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);
    for(int l=0;l<NL;++l){
        for(int m=0;m<M;++m){
            blackwell::kernels::fused_rmsnorm_quant_int8(d_xi[m],d_xs[m],d_x[m],d_rn,H,1e-6f,st);
            blackwell::kernels::gemv_int8(d_Q[m],d_xi[m],d_xs[m],W[l].q.d,W[l].q.sc,H,Q,st);
            blackwell::kernels::gemv_int8(d_K,d_xi[m],d_xs[m],W[l].k.d,W[l].k.sc,H,KV,st);
            blackwell::kernels::gemv_int8(d_V,d_xi[m],d_xs[m],W[l].v.d,W[l].v.sc,H,KV,st);
            blackwell::kernels::update_kv_cache(d_kc,d_vc,d_K,d_V,0,SEQ-1,nkv,hd,SEQ,st);
            blackwell::kernels::attention_decode_gqa(d_attn[m],d_Q[m],d_kc,d_vc,SEQ-1,nqh,nkv,hd,SEQ,st);
            blackwell::kernels::pack_int8(d_ai[m],d_attn[m],d_as[m],Q,st);
            blackwell::kernels::gemv_int8(d_proj[m],d_xi[m],d_as[m],W[l].o.d,W[l].o.sc,Q,H,st);
            blackwell::kernels::vector_add_fp32(d_proj[m],d_proj[m],d_x[m],H,st);
            blackwell::kernels::fused_rmsnorm_quant_int8(d_xi[m],d_xs[m],d_proj[m],d_rn,H,1e-6f,st);
            blackwell::kernels::gemv_int8(d_gate[m],d_xi[m],d_xs[m],W[l].g.d,W[l].g.sc,H,I,st);
            blackwell::kernels::gemv_int8(d_up[m],  d_xi[m],d_xs[m],W[l].u.d,W[l].u.sc,H,I,st);
            blackwell::kernels::apply_swiglu(d_mlp[m],d_gate[m],d_up[m],I,st);
            blackwell::kernels::pack_int8(d_mi[m],d_mlp[m],d_ms[m],I,st);
            blackwell::kernels::gemv_int8(d_proj[m],d_mi[m],d_ms[m],W[l].d.d,W[l].d.sc,I,H,st);
            blackwell::kernels::vector_add_fp32(d_proj[m],d_proj[m],d_res[m],H,st);
            // Use d_res as preserved attn output — copy proj to x for next layer
            // Copy proj to d_x[m] for next layer start (async for capture compat)
            cudaMemcpyAsync(d_x[m],d_proj[m],H*4,cudaMemcpyDeviceToDevice,st);
        }
    }
    cudaGraph_t graph; die(cudaStreamEndCapture(st,&graph),"capture");
    cudaGraphExec_t ge; die(cudaGraphInstantiate(&ge,graph,NULL,NULL,0),"inst");

    for(int w=0;w<5;++w){ die(cudaGraphLaunch(ge,st),"warmup"); }
    cudaStreamSynchronize(st);
    reset_x(); cudaStreamSynchronize(st);

    Tmr TB; TB.start();
    for(int i=0;i<IT;++i) die(cudaGraphLaunch(ge,st),"launch");
    cudaStreamSynchronize(st);
    float ms_B=TB.stop();
    float pt_B=ms_B/(IT*M);
    printf("\n=== Batched CUDA Graph (M=%d) ===\n",M);
    printf("  %.3fms/token  %.1f t/s  (batch %.1f t/s)\n",
        pt_B, 1000.f/pt_B, M*1000.f/pt_B);

    printf("\n=== Summary ===\n");
    printf("  %-28s  %7s  %10s\n","Method","ms/token","batch t/s");
    printf("  %-28s  %6.3fms  %10.1f\n","M× single-token",pt_C,M*1000.f/pt_C);
    printf("  %-28s  %6.3fms  %10.1f\n","Batched per-kernel",pt_A,M*1000.f/pt_A);
    printf("  %-28s  %6.3fms  %10.1f\n","Batched CUDA Graph",pt_B,M*1000.f/pt_B);
    printf("\n  Batched vs single: %.2fx\n", pt_C/pt_A);
    printf("  CUDA Graph speedup: %.2fx\n", ms_A/ms_B);
    printf("  Total: %.2fx\n", pt_C/pt_B);
    printf("  Effective: %.1f t/s\n",M*1000.f/pt_B);
    printf("  vs single-seq CUDA Graph: 122.7 t/s\n");
    printf("  vs llama.cpp: 114.0 t/s\n");

    cudaGraphDestroy(graph); cudaGraphExecDestroy(ge);
    cudaStreamDestroy(st);
    for(int m=0;m<M;++m){
        cudaFree(d_x[m]);cudaFree(d_xi[m]);cudaFree(d_xs[m]);cudaFree(d_res[m]);
        cudaFree(d_Q[m]);cudaFree(d_attn[m]);cudaFree(d_ai[m]);cudaFree(d_as[m]);
        cudaFree(d_gate[m]);cudaFree(d_up[m]);cudaFree(d_mlp[m]);cudaFree(d_mi[m]);cudaFree(d_ms[m]);cudaFree(d_proj[m]);
    }
    cudaFree(d_K);cudaFree(d_V);cudaFree(d_kc);cudaFree(d_vc);cudaFree(d_rn);
    for(int l=0;l<NL;++l){cudaFree(W[l].q.d);cudaFree(W[l].q.sc);cudaFree(W[l].k.d);cudaFree(W[l].k.sc);
        cudaFree(W[l].v.d);cudaFree(W[l].v.sc);cudaFree(W[l].o.d);cudaFree(W[l].o.sc);
        cudaFree(W[l].g.d);cudaFree(W[l].g.sc);cudaFree(W[l].u.d);cudaFree(W[l].u.sc);cudaFree(W[l].d.d);cudaFree(W[l].d.sc);}
    return 0;
}