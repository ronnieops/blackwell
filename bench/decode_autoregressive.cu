// bench/decode_autoregressive.cu — Autoregressive decode with dynamic seq_pos
//
// Demonstrates CUDA Graph capture with variable seq_pos.
// Captures graph once, then loops tokens updating seq_pos between launches
// via update_decode_seq_pos().
//
// Key: seq_pos is read from device pointer by attention_decode_gqa and
// update_kv_cache. Wrappers use pinned host memory (graph-safe). Update
// pinned value between launches = variable seq_pos without re-capturing.
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/decode_autoregressive.cu build/libblackwell_kernels.a \
//     -o bench/decode_autoregressive

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
    int NL=1, IT=50, MAX_POS=128;
    if(argc>1)NL=atoi(argv[1]);
    if(argc>2)IT=atoi(argv[2]);
    if(IT>MAX_POS){printf("IT max %d\n",MAX_POS);return 0;}

    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    printf("# Autoregressive Decode — Dynamic seq_pos with CUDA Graph\n");
    printf("  Device: %s\n  L:%d IT:%d MAX_POS:%d\n\n",P.name,NL,IT,MAX_POS);

    const int H=2048,Q=2048,KV=1024,I=6144,nqh=12,nkv=12,hd=64,SEQ=MAX_POS;
    const float IXV=1.f/127.f;

    float *d_x, *d_res, *d_Q, *d_K, *d_V, *d_attn, *d_gate, *d_up, *d_mlp, *d_proj;
    int8_t *d_xi, *d_ai, *d_mi;
    float *d_xs, *d_as, *d_ms, *d_rn;
    float *d_kc, *d_vc;

    #define ALLOC(p,n) die(cudaMalloc(&(p),(n)),#p)
    ALLOC(d_x,H*4);ALLOC(d_xi,H);ALLOC(d_xs,(H/16)*4);
    ALLOC(d_res,H*4);ALLOC(d_Q,Q*4);ALLOC(d_K,KV*4);ALLOC(d_V,KV*4);
    ALLOC(d_attn,Q*4);ALLOC(d_ai,Q);ALLOC(d_as,(Q/16)*4);
    ALLOC(d_gate,I*4);ALLOC(d_up,I*4);ALLOC(d_mlp,I*4);
    ALLOC(d_mi,I);ALLOC(d_ms,(I/16)*4);ALLOC(d_proj,H*4);
    ALLOC(d_kc,nkv*SEQ*hd*4);ALLOC(d_vc,nkv*SEQ*hd*4);
    ALLOC(d_rn,H*4);
    #undef ALLOC

    std::vector<float> x0(H); for(int i=0;i<H;++i)x0[i]=(i%17-8)*0.01f;
    std::vector<float> xv={(float)IXV};
    std::vector<float> rn(H,1.f);
    cudaMemcpy(d_rn,rn.data(),H*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,x0.data(),H*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_xs,xv.data(),(H/16)*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_as,xv.data(),(Q/16)*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_ms,xv.data(),(I/16)*4,cudaMemcpyHostToDevice);
    cudaMemset(d_kc,0,nkv*SEQ*hd*4);
    cudaMemset(d_vc,0,nkv*SEQ*hd*4);

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

    auto do_layer=[&](int l){
        blackwell::kernels::fused_rmsnorm_quant_int8(
            (int8_t*)d_xi,(float*)d_xs,d_x,d_rn,H,1e-6f,st);
        blackwell::kernels::gemv_int8(
            d_Q,d_xi,d_xs,W[l].q.d,W[l].q.sc,H,Q,st);
        blackwell::kernels::gemv_int8(
            d_K,d_xi,d_xs,W[l].k.d,W[l].k.sc,H,KV,st);
        blackwell::kernels::gemv_int8(
            d_V,d_xi,d_xs,W[l].v.d,W[l].v.sc,H,KV,st);
        blackwell::kernels::update_kv_cache(
            d_kc,d_vc,d_K,d_V,0,0,nkv,hd,SEQ,st);
        blackwell::kernels::attention_decode_gqa(
            d_attn,d_Q,d_kc,d_vc,0,nqh,nkv,hd,SEQ,st);
        blackwell::kernels::pack_int8(d_ai,d_attn,d_as,Q,st);
        blackwell::kernels::gemv_int8(
            d_proj,d_ai,d_as,W[l].o.d,W[l].o.sc,Q,H,st);
        blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_x,H,st);
        blackwell::kernels::vector_add_fp32(d_res,d_proj,d_x,H,st);
        blackwell::kernels::fused_rmsnorm_quant_int8(
            d_xi,d_xs,d_proj,d_rn,H,1e-6f,st);
        blackwell::kernels::gemv_int8(
            d_gate,d_xi,d_xs,W[l].g.d,W[l].g.sc,H,I,st);
        blackwell::kernels::gemv_int8(
            d_up,d_xi,d_xs,W[l].u.d,W[l].u.sc,H,I,st);
        blackwell::kernels::apply_swiglu(d_mlp,d_gate,d_up,I,st);
        blackwell::kernels::pack_int8(d_mi,d_mlp,d_ms,I,st);
        blackwell::kernels::gemv_int8(
            d_proj,d_mi,d_ms,W[l].d.d,W[l].d.sc,I,H,st);
        blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_res,H,st);
        cudaMemcpyAsync(d_x,d_proj,H*4,cudaMemcpyDeviceToDevice,st);
    };

    // Pre-fill KV cache (each step writes K,V at position t)
    printf("Pre-filling KV cache (%d tokens)...\n",IT);
    cudaMemcpy(d_x,x0.data(),H*4,cudaMemcpyHostToDevice);
    for(int t=0;t<IT;++t){
        die(blackwell::kernels::update_decode_seq_pos(t,st),"prefill seq_pos");
        for(int l=0;l<NL;++l) do_layer(l);
        cudaStreamSynchronize(st);
        cudaMemcpy(d_x,x0.data(),H*4,cudaMemcpyHostToDevice);
    }
    printf("KV cache filled.\n\n");

    // === Baseline: Per-kernel autoregressive ===
    cudaMemcpy(d_x,x0.data(),H*4,cudaMemcpyHostToDevice);
    cudaStreamSynchronize(st);
    Tmr TA; TA.start();
    for(int t=0;t<IT;++t){
        die(blackwell::kernels::update_decode_seq_pos(t,st),"per-kernel seq_pos");
        for(int l=0;l<NL;++l) do_layer(l);
        cudaStreamSynchronize(st);
        cudaMemcpy(d_x,x0.data(),H*4,cudaMemcpyHostToDevice);
    }
    float ms_A=TA.stop();
    float pt_A=ms_A/IT;
    printf("=== Per-kernel autoregressive ===\n");
    printf("  %.3fms/token  %.1f t/s\n", pt_A, 1000.f/pt_A);

    // === Captured: CUDA Graph autoregressive ===
    cudaMemcpy(d_x,x0.data(),H*4,cudaMemcpyHostToDevice);
    cudaStreamSynchronize(st);

    die(blackwell::kernels::update_decode_seq_pos(0,st),"capture seq_pos");
    cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);
    for(int l=0;l<NL;++l) do_layer(l);
    cudaGraph_t graph; die(cudaStreamEndCapture(st,&graph),"capture");
    cudaGraphExec_t ge; die(cudaGraphInstantiate(&ge,graph,NULL,NULL,0),"inst");

    for(int w=0;w<5;++w){
        die(blackwell::kernels::update_decode_seq_pos(0,st),"warmup");
        die(cudaGraphLaunch(ge,st),"warmup");
    }
    cudaStreamSynchronize(st);

    // Benchmark: autoregressive loop with INCREASING seq_pos
    cudaMemcpy(d_x,x0.data(),H*4,cudaMemcpyHostToDevice);
    cudaStreamSynchronize(st);
    Tmr TB; TB.start();
    for(int t=0;t<IT;++t){
        // seq_pos increments each step — full autoregressive behavior
        die(blackwell::kernels::update_decode_seq_pos(t,st),"dyn seq_pos");
        die(cudaGraphLaunch(ge,st),"launch");
        cudaStreamSynchronize(st);
        cudaMemcpy(d_x,x0.data(),H*4,cudaMemcpyHostToDevice);
    }
    float ms_B=TB.stop();
    float pt_B=ms_B/IT;

    printf("\n=== CUDA Graph autoregressive (dynamic seq_pos) ===\n");
    printf("  %.3fms/token  %.1f t/s\n", pt_B, 1000.f/pt_B);
    printf("\n=== Summary ===\n");
    printf("  %-40s  %8s\n","Method","t/s");
    printf("  %-40s  %8.1f\n","Per-kernel (autoregressive)",1000.f/pt_A);
    printf("  %-40s  %8.1f\n","CUDA Graph (dynamic seq_pos)",1000.f/pt_B);
    printf("  Speedup: %.2fx\n\n", pt_A/pt_B);

    // Cleanup
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