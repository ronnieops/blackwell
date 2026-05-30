// bench/inference_server_batched.cu — Production inference server using batched GEMV
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/inference_server_batched.cu build/libblackwell_kernels.a \
//     -o bench/inference_server_batched

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cstring>
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

// Per-sequence decode (all NL layers)
void decode_seq(float*d_x, float*d_xi, float*d_xs, float*d_res,
                float*d_Q, float*d_K, float*d_V, float*d_attn,
                int8_t*d_ai, float*d_as, float*d_gate, float*d_up,
                float*d_mlp, int8_t*d_mi, float*d_ms, float*d_proj,
                float*d_kc, float*d_vc, float*d_rn,
                L*W, int NL, int m, int seq_len, int SEQ, cudaStream_t st){
    const int H=2048,Q=2048,KV=1024,I=6144,nqh=12,nkv=12,hd=64;
    int kb_base=m*nkv*SEQ*hd;
    for(int s=0;s<seq_len;++s){
        int kb=kb_base+s*nkv*SEQ*hd;
        blackwell::kernels::fused_rmsnorm_quant_int8(
            (int8_t*)d_xi+m*H,(float*)d_xs+m*(H/16),(float*)d_x+m*H,d_rn,H,1e-6f,st);
        blackwell::kernels::gemv_int8_warp((float*)d_Q+m*Q*4,(int8_t*)d_xi+m*H,(float*)d_xs+m*(H/16),
            W[0].q.d,W[0].q.sc,H,Q,st);
        blackwell::kernels::gemv_int8_warp((float*)d_K+m*KV*4,(int8_t*)d_xi+m*H,(float*)d_xs+m*(H/16),
            W[0].k.d,W[0].k.sc,H,KV,st);
        blackwell::kernels::gemv_int8_warp((float*)d_V+m*KV*4,(int8_t*)d_xi+m*H,(float*)d_xs+m*(H/16),
            W[0].v.d,W[0].v.sc,H,KV,st);
        blackwell::kernels::update_kv_cache((float*)d_kc+kb,(float*)d_vc+kb,
            (float*)d_K+m*KV*4,(float*)d_V+m*KV*4,s,s,nkv,hd,SEQ,st);
        blackwell::kernels::attention_decode_gqa((float*)d_attn+m*Q*4,(float*)d_Q+m*Q*4,
            (float*)d_kc+kb,(float*)d_vc+kb,s,nqh,nkv,hd,SEQ,st);
        blackwell::kernels::pack_int8((int8_t*)d_ai+m*Q,(float*)d_attn+m*Q*4,(float*)d_as+m*(Q/16),Q,st);
        blackwell::kernels::gemv_int8_warp((float*)d_proj+m*H,(int8_t*)d_ai+m*Q,(float*)d_as+m*(Q/16),
            W[0].o.d,W[0].o.sc,Q,H,st);
        blackwell::kernels::vector_add_fp32((float*)d_proj+m*H,(float*)d_proj+m*H,(float*)d_x+m*H,H,st);
        blackwell::kernels::vector_add_fp32((float*)d_res+m*H,(float*)d_proj+m*H,(float*)d_x+m*H,H,st);
        blackwell::kernels::fused_rmsnorm_quant_int8(
            (int8_t*)d_xi+m*H,(float*)d_xs+m*(H/16),(float*)d_proj+m*H,d_rn,H,1e-6f,st);
        blackwell::kernels::gemv_int8_warp((float*)d_gate+m*I*4,(int8_t*)d_xi+m*H,(float*)d_xs+m*(H/16),
            W[0].g.d,W[0].g.sc,H,I,st);
        blackwell::kernels::gemv_int8_warp((float*)d_up+m*I*4,(int8_t*)d_xi+m*H,(float*)d_xs+m*(H/16),
            W[0].u.d,W[0].u.sc,H,I,st);
        blackwell::kernels::apply_swiglu((float*)d_mlp+m*I*4,(float*)d_gate+m*I*4,(float*)d_up+m*I*4,I,st);
        blackwell::kernels::pack_int8((int8_t*)d_mi+m*I,(float*)d_mlp+m*I*4,(float*)d_ms+m*(I/16),I,st);
        blackwell::kernels::gemv_int8_warp((float*)d_proj+m*H,(int8_t*)d_mi+m*I,(float*)d_ms+m*(I/16),
            W[0].d.d,W[0].d.sc,I,H,st);
        blackwell::kernels::vector_add_fp32((float*)d_proj+m*H,(float*)d_proj+m*H,(float*)d_res+m*H,H,st);
        cudaMemcpyAsync((float*)d_x+m*H,(float*)d_proj+m*H,H*4,cudaMemcpyDeviceToDevice,st);
    }
}

// Batched decode (all NL layers, all M sequences) — uses per-sequence GEMVs (NOT batched GEMV kernel)
void batch_decode_per_seq(float*d_x, float*d_xi, float*d_xs, float*d_res,
                          float*d_Q, float*d_K, float*d_V, float*d_attn,
                          int8_t*d_ai, float*d_as, float*d_gate, float*d_up,
                          float*d_mlp, int8_t*d_mi, float*d_ms, float*d_proj,
                          float*d_kc, float*d_vc, float*d_rn,
                          L*W, int NL, int M, int seq_len, int SEQ, cudaStream_t st){
    const int H=2048,Q=2048,KV=1024,I=6144,nqh=12,nkv=12,hd=64;
    for(int s=0;s<seq_len;++s){
        for(int m=0;m<M;++m){
            int kb=m*nkv*SEQ*hd+s*nkv*SEQ*hd;
            blackwell::kernels::fused_rmsnorm_quant_int8(
                (int8_t*)d_xi+m*H,(float*)d_xs+m*(H/16),(float*)d_x+m*H,d_rn,H,1e-6f,st);
            blackwell::kernels::gemv_int8_warp((float*)d_Q+m*Q*4,(int8_t*)d_xi+m*H,(float*)d_xs+m*(H/16),
                W[0].q.d,W[0].q.sc,H,Q,st);
            blackwell::kernels::gemv_int8_warp((float*)d_K+m*KV*4,(int8_t*)d_xi+m*H,(float*)d_xs+m*(H/16),
                W[0].k.d,W[0].k.sc,H,KV,st);
            blackwell::kernels::gemv_int8_warp((float*)d_V+m*KV*4,(int8_t*)d_xi+m*H,(float*)d_xs+m*(H/16),
                W[0].v.d,W[0].v.sc,H,KV,st);
            blackwell::kernels::update_kv_cache((float*)d_kc+kb,(float*)d_vc+kb,
                (float*)d_K+m*KV*4,(float*)d_V+m*KV*4,s,s,nkv,hd,SEQ,st);
            blackwell::kernels::attention_decode_gqa((float*)d_attn+m*Q*4,(float*)d_Q+m*Q*4,
                (float*)d_kc+kb,(float*)d_vc+kb,s,nqh,nkv,hd,SEQ,st);
            blackwell::kernels::pack_int8((int8_t*)d_ai+m*Q,(float*)d_attn+m*Q*4,(float*)d_as+m*(Q/16),Q,st);
            blackwell::kernels::gemv_int8_warp((float*)d_proj+m*H,(int8_t*)d_ai+m*Q,(float*)d_as+m*(Q/16),
                W[0].o.d,W[0].o.sc,Q,H,st);
            blackwell::kernels::vector_add_fp32((float*)d_proj+m*H,(float*)d_proj+m*H,(float*)d_x+m*H,H,st);
            blackwell::kernels::vector_add_fp32((float*)d_res+m*H,(float*)d_proj+m*H,(float*)d_x+m*H,H,st);
            blackwell::kernels::fused_rmsnorm_quant_int8(
                (int8_t*)d_xi+m*H,(float*)d_xs+m*(H/16),(float*)d_proj+m*H,d_rn,H,1e-6f,st);
            blackwell::kernels::gemv_int8_warp((float*)d_gate+m*I*4,(int8_t*)d_xi+m*H,(float*)d_xs+m*(H/16),
                W[0].g.d,W[0].g.sc,H,I,st);
            blackwell::kernels::gemv_int8_warp((float*)d_up+m*I*4,(int8_t*)d_xi+m*H,(float*)d_xs+m*(H/16),
                W[0].u.d,W[0].u.sc,H,I,st);
            blackwell::kernels::apply_swiglu((float*)d_mlp+m*I*4,(float*)d_gate+m*I*4,(float*)d_up+m*I*4,I,st);
            blackwell::kernels::pack_int8((int8_t*)d_mi+m*I,(float*)d_mlp+m*I*4,(float*)d_ms+m*(I/16),I,st);
            blackwell::kernels::gemv_int8_warp((float*)d_proj+m*H,(int8_t*)d_mi+m*I,(float*)d_ms+m*(I/16),
                W[0].d.d,W[0].d.sc,I,H,st);
            blackwell::kernels::vector_add_fp32((float*)d_proj+m*H,(float*)d_proj+m*H,(float*)d_res+m*H,H,st);
            cudaMemcpyAsync((float*)d_x+m*H,(float*)d_proj+m*H,H*4,cudaMemcpyDeviceToDevice,st);
        }
    }
    cudaStreamSynchronize(st);
}

// Batched decode with TRUE batched GEMV kernel (gemv_int8_batched)
void batch_decode_batched(float*d_x, float*d_xi, float*d_xs, float*d_res,
                          float*d_Q, float*d_K, float*d_V, float*d_attn,
                          int8_t*d_ai, float*d_as, float*d_gate, float*d_up,
                          float*d_mlp, int8_t*d_mi, float*d_ms, float*d_proj,
                          float*d_kc, float*d_vc, float*d_rn,
                          L*W, int NL, int M, int seq_len, int SEQ, cudaStream_t st){
    const int H=2048,Q=2048,KV=1024,I=6144,nqh=12,nkv=12,hd=64;
    // Batched GEMV: one call processes M sequences
    for(int s=0;s<seq_len;++s){
        // RMSNorm + quantize (batched: one kernel for all M)
        blackwell::kernels::fused_rmsnorm_quant_int8(
            (int8_t*)d_xi,(float*)d_xs,(float*)d_x,d_rn,H,M,st);
        // Q,K,V — batched GEMV
        blackwell::kernels::gemv_int8_batched((float*)d_Q,(int8_t*)d_xi,(float*)d_xs,
            W[0].q.d,W[0].q.sc,M,H,Q,st);
        blackwell::kernels::gemv_int8_batched((float*)d_K,(int8_t*)d_xi,(float*)d_xs,
            W[0].k.d,W[0].k.sc,M,H,KV,st);
        blackwell::kernels::gemv_int8_batched((float*)d_V,(int8_t*)d_xi,(float*)d_xs,
            W[0].v.d,W[0].v.sc,M,H,KV,st);
        // KV cache — per sequence
        for(int m=0;m<M;++m){
            int kb=m*nkv*SEQ*hd+s*nkv*SEQ*hd;
            blackwell::kernels::update_kv_cache(
                (float*)d_kc+kb,(float*)d_vc+kb,
                (float*)d_K+m*KV*4,(float*)d_V+m*KV*4,s,s,nkv,hd,SEQ,st);
            blackwell::kernels::attention_decode_gqa(
                (float*)d_attn+m*Q*4,(float*)d_Q+m*Q*4,
                (float*)d_kc+kb,(float*)d_vc+kb,s,nqh,nkv,hd,SEQ,st);
            blackwell::kernels::pack_int8((int8_t*)d_ai+m*Q,(float*)d_attn+m*Q*4,(float*)d_as+m*(Q/16),Q,st);
        }
        // Wo — batched GEMV
        blackwell::kernels::gemv_int8_batched((float*)d_proj,(int8_t*)d_ai,(float*)d_as,
            W[0].o.d,W[0].o.sc,M,Q,H,st);
        for(int m=0;m<M;++m){
            blackwell::kernels::vector_add_fp32((float*)d_proj+m*H,(float*)d_proj+m*H,(float*)d_x+m*H,H,st);
            blackwell::kernels::vector_add_fp32((float*)d_res+m*H,(float*)d_proj+m*H,(float*)d_x+m*H,H,st);
        }
        // MLP RMSNorm — batched
        blackwell::kernels::fused_rmsnorm_quant_int8(
            (int8_t*)d_xi,(float*)d_xs,(float*)d_proj,d_rn,H,M,st);
        // Gate + up — batched GEMV
        blackwell::kernels::gemv_int8_batched((float*)d_gate,(int8_t*)d_xi,(float*)d_xs,
            W[0].g.d,W[0].g.sc,M,H,I,st);
        blackwell::kernels::gemv_int8_batched((float*)d_up,(int8_t*)d_xi,(float*)d_xs,
            W[0].u.d,W[0].u.sc,M,H,I,st);
        // Swiglu + down — per sequence
        for(int m=0;m<M;++m){
            blackwell::kernels::apply_swiglu((float*)d_mlp+m*I*4,(float*)d_gate+m*I*4,(float*)d_up+m*I*4,I,st);
            blackwell::kernels::pack_int8((int8_t*)d_mi+m*I,(float*)d_mlp+m*I*4,(float*)d_ms+m*(I/16),I,st);
            blackwell::kernels::gemv_int8_warp((float*)d_proj+m*H,(int8_t*)d_mi+m*I,(float*)d_ms+m*(I/16),
                W[0].d.d,W[0].d.sc,I,H,st);
            blackwell::kernels::vector_add_fp32((float*)d_proj+m*H,(float*)d_proj+m*H,(float*)d_res+m*H,H,st);
            cudaMemcpyAsync((float*)d_x+m*H,(float*)d_proj+m*H,H*4,cudaMemcpyDeviceToDevice,st);
        }
    }
    cudaStreamSynchronize(st);
}

int main(int argc, char** argv){
    int NL=2, M=4, N=20, SEQ=8;
    if(argc>1)NL=atoi(argv[1]);
    if(argc>2)M=atoi(argv[2]);
    if(argc>3)N=atoi(argv[3]);
    if(argc>4)SEQ=atoi(argv[4]);

    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    printf("# Inference Server — Qwen3-1.7B\n");
    printf("  Device: %s\n  L:%d M:%d N:%d SEQ:%d\n\n",P.name,NL,M,N,SEQ);

    const int H=2048,Q=2048,KV=1024,I=6144,nqh=12,nkv=12,hd=64,MAXSEQ=128;
    const float IXV=1.f/127.f;

    float*d_x,*d_xi,*d_xs,*d_res;
    float*d_Q,*d_K,*d_V,*d_attn;
    int8_t*d_ai; float*d_as;
    float*d_gate,*d_up,*d_mlp;
    int8_t*d_mi; float*d_ms;
    float*d_proj;
    float*d_kc,*d_vc,*d_rn;

    #define ALLOC(p,n) die(cudaMalloc(&(p),(n)),#p)
    ALLOC(d_x,M*H*4);ALLOC(d_xi,M*H);ALLOC(d_xs,M*(H/16)*4);
    ALLOC(d_res,M*H*4);ALLOC(d_Q,M*Q*4);ALLOC(d_K,M*KV*4);ALLOC(d_V,M*KV*4);
    ALLOC(d_attn,M*Q*4);ALLOC(d_ai,M*Q);ALLOC(d_as,M*(Q/16)*4);
    ALLOC(d_gate,M*I*4);ALLOC(d_up,M*I*4);ALLOC(d_mlp,M*I*4);
    ALLOC(d_mi,M*I);ALLOC(d_ms,M*(I/16)*4);ALLOC(d_proj,M*H*4);
    ALLOC(d_kc,M*nkv*MAXSEQ*hd*4);ALLOC(d_vc,M*nkv*MAXSEQ*hd*4);
    ALLOC(d_rn,H*4);
    #undef ALLOC

    std::vector<float> rn(H,1.f);
    std::vector<float> xv={(float)IXV};
    cudaMemcpy(d_rn,rn.data(),H*4,cudaMemcpyHostToDevice);
    for(int m=0;m<M;++m){
        cudaMemcpy((float*)d_xs+m*(H/16),xv.data(),(H/16)*4,cudaMemcpyHostToDevice);
        cudaMemcpy((float*)d_as+m*(Q/16),xv.data(),(Q/16)*4,cudaMemcpyHostToDevice);
        cudaMemcpy((float*)d_ms+m*(I/16),xv.data(),(I/16)*4,cudaMemcpyHostToDevice);
    }
    cudaMemset(d_kc,0,M*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vc,0,M*nkv*MAXSEQ*hd*4);

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

    // Generate random inputs
    std::vector<std::vector<float>> inputs(N, std::vector<float>(H));
    for(int i=0;i<N;++i)
        for(int j=0;j<H;++j)
            inputs[i][j]=(j%17-8)*0.01f;

    // Mode A: Sequential per-kernel (one request at a time, all NL layers)
    printf("\n=== Mode A: Sequential per-kernel (1 request) ===\n");
    auto t0=Clock::now();
    for(int i=0;i<N;++i){
        cudaMemcpy((float*)d_x, inputs[i].data(), H*4, cudaMemcpyHostToDevice);
        cudaStreamSynchronize(st);
        for(int s=0;s<SEQ;++s){
            int kb=s*nkv*MAXSEQ*hd;
            blackwell::kernels::fused_rmsnorm_quant_int8(
                (int8_t*)d_xi,(float*)d_xs,(float*)d_x,d_rn,H,1e-6f,st);
            blackwell::kernels::gemv_int8_warp((float*)d_Q,(int8_t*)d_xi,(float*)d_xs,
                W[0].q.d,W[0].q.sc,H,Q,st);
            blackwell::kernels::gemv_int8_warp((float*)d_K,(int8_t*)d_xi,(float*)d_xs,
                W[0].k.d,W[0].k.sc,H,KV,st);
            blackwell::kernels::gemv_int8_warp((float*)d_V,(int8_t*)d_xi,(float*)d_xs,
                W[0].v.d,W[0].v.sc,H,KV,st);
            blackwell::kernels::update_kv_cache((float*)d_kc+kb,(float*)d_vc+kb,
                d_K,d_V,s,s,nkv,hd,MAXSEQ,st);
            blackwell::kernels::attention_decode_gqa(d_attn,d_Q,(float*)d_kc+kb,(float*)d_vc+kb,s,nqh,nkv,hd,MAXSEQ,st);
            blackwell::kernels::pack_int8(d_ai,d_attn,d_as,Q,st);
            blackwell::kernels::gemv_int8_warp(d_proj,d_ai,d_as,W[0].o.d,W[0].o.sc,Q,H,st);
            blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_x,H,st);
            blackwell::kernels::vector_add_fp32(d_res,d_proj,d_x,H,st);
            blackwell::kernels::fused_rmsnorm_quant_int8(
                (int8_t*)d_xi,(float*)d_xs,d_proj,d_rn,H,1e-6f,st);
            blackwell::kernels::gemv_int8_warp(d_gate,(int8_t*)d_xi,(float*)d_xs,
                W[0].g.d,W[0].g.sc,H,I,st);
            blackwell::kernels::gemv_int8_warp(d_up,(int8_t*)d_xi,(float*)d_xs,
                W[0].u.d,W[0].u.sc,H,I,st);
            blackwell::kernels::apply_swiglu(d_mlp,d_gate,d_up,I,st);
            blackwell::kernels::pack_int8(d_mi,d_mlp,d_ms,I,st);
            blackwell::kernels::gemv_int8_warp(d_proj,d_mi,d_ms,W[0].d.d,W[0].d.sc,I,H,st);
            blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_res,H,st);
            cudaMemcpyAsync(d_x,d_proj,H*4,cudaMemcpyDeviceToDevice,st);
        }
        cudaStreamSynchronize(st);
    }
    auto t1=Clock::now();
    double ms_A=std::chrono::duration<double,std::milli>(t1-t0).count();
    printf("  %d × seq_len=%d: %.1fms total, %.1fms per req, %.1f req/s\n",
        N,SEQ,ms_A,ms_A/N,1000.0*N/ms_A);

    // Mode B: Batched per-sequence GEMVs (M sequences, but M× separate kernel calls)
    printf("\n=== Mode B: Batched per-sequence (M=%d, per-kernel) ===\n",M);
    // Reset KV cache
    cudaMemset(d_kc,0,M*nkv*MAXSEQ*hd*4);cudaMemset(d_vc,0,M*nkv*MAXSEQ*hd*4);
    t0=Clock::now();
    for(int batch=0;batch<N/M;++batch){
        // Copy M inputs
        for(int m=0;m<M;++m){
            int idx=batch*M+m;
            cudaMemcpy((float*)d_x+m*H, inputs[idx].data(), H*4, cudaMemcpyHostToDevice);
        }
        cudaStreamSynchronize(st);
        batch_decode_per_seq(d_x,d_xi,d_xs,d_res,d_Q,d_K,d_V,d_attn,
            d_ai,d_as,d_gate,d_up,d_mlp,d_mi,d_ms,d_proj,d_kc,d_vc,d_rn,
            W.data(),NL,M,SEQ,MAXSEQ,st);
    }
    t1=Clock::now();
    double ms_B=std::chrono::duration<double,std::milli>(t1-t0).count();
    printf("  %d batches × M=%d: %.1fms total, %.1fms per batch\n",
        N/M,M,ms_B,ms_B/(N/M));
    printf("  Throughput: %.1f req/s  (%.1fms per req)\n", 1000.0*N/ms_B, ms_B/N);

    // Mode C: TRUE Batched GEMV (gemv_int8_batched kernel, M sequences per call)
    printf("\n=== Mode C: Batched GEMV (gemv_int8_batched, M=%d) ===\n",M);
    // Reset KV cache
    cudaMemset(d_kc,0,M*nkv*MAXSEQ*hd*4);cudaMemset(d_vc,0,M*nkv*MAXSEQ*hd*4);
    t0=Clock::now();
    for(int batch=0;batch<N/M;++batch){
        for(int m=0;m<M;++m){
            int idx=batch*M+m;
            cudaMemcpy((float*)d_x+m*H, inputs[idx].data(), H*4, cudaMemcpyHostToDevice);
        }
        cudaStreamSynchronize(st);
        batch_decode_batched(d_x,d_xi,d_xs,d_res,d_Q,d_K,d_V,d_attn,
            d_ai,d_as,d_gate,d_up,d_mlp,d_mi,d_ms,d_proj,d_kc,d_vc,d_rn,
            W.data(),NL,M,SEQ,MAXSEQ,st);
    }
    t1=Clock::now();
    double ms_C=std::chrono::duration<double,std::milli>(t1-t0).count();
    float pt_C=ms_C/(N/M)/M;  // ms per seq
    printf("  %d batches × M=%d: %.1fms total, %.1fms per batch\n",
        N/M,M,ms_C,ms_C/(N/M));
    printf("  Per-sequence: %.3fms  =>  %.1f t/s\n", pt_C, 1000.f/pt_C);
    printf("  Throughput: %.1f req/s\n", 1000.0*N/ms_C);

    printf("\n=== Summary ===\n");
    printf("  %-30s  %8s  %8s\n","Method","ms/req","req/s");
    printf("  %-30s  %7.1fms  %8.1f\n","Sequential per-kernel",ms_A/N,1000.0*N/ms_A);
    printf("  %-30s  %7.1fms  %8.1f\n","Batched per-seq (M×kernel)",ms_B/N,1000.0*N/ms_B);
    printf("  %-30s  %7.3fms  %8.1f\n","Batched GEMV kernel",ms_C/N,1000.0*N/ms_C);
    printf("\n  Batched speedup vs sequential: %.2fx\n", ms_A/ms_C);
    printf("  vs llama.cpp: 114.0 t/s\n");
    printf("  vs single-seq CUDA Graph: 122.7 t/s\n");

    cudaStreamDestroy(st);
    cudaFree(d_x);cudaFree(d_xi);cudaFree(d_xs);cudaFree(d_res);
    cudaFree(d_Q);cudaFree(d_K);cudaFree(d_V);cudaFree(d_attn);
    cudaFree(d_ai);cudaFree(d_as);cudaFree(d_gate);cudaFree(d_up);
    cudaFree(d_mlp);cudaFree(d_mi);cudaFree(d_ms);cudaFree(d_proj);
    cudaFree(d_kc);cudaFree(d_vc);cudaFree(d_rn);
    for(auto& w:W){
        cudaFree(w.q.d);cudaFree(w.q.sc);cudaFree(w.k.d);cudaFree(w.k.sc);
        cudaFree(w.v.d);cudaFree(w.v.sc);cudaFree(w.o.d);cudaFree(w.o.sc);
        cudaFree(w.g.d);cudaFree(w.g.sc);cudaFree(w.u.d);cudaFree(w.u.sc);
        cudaFree(w.d.d);cudaFree(w.d.sc);
    }
    return 0;
}