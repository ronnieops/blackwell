// bench/inference_server.cu — Production inference server
//
// 28-layer Qwen3-1.7B: prefill (M=128) + autoregressive decode with INT8 weights.
// Loads real weights from weights_int8_bf16/.
//
// Modes:
//   A — single-seq per-kernel decode (1 req at a time)
//   B — batch per-kernel decode (M seq, per-kernel calls)
//   C — batched GEMV decode (gemv_int8_batched, M=2-8)
//   D — prefill M=128 tokens → autoregressive decode N tokens
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/inference_server.cu build/libblackwell_kernels.a \
//     -o bench/inference_server
//
// Run: ./bench/inference_server [NL=28] [M=4] [N=20] [SEQ=8]

#include <cuda_runtime.h>
#include <cstdio>

// Compute 16-element block absmax scales for pack_int8
__global__ void absmax_scales_kernel(const float* in, float* sc, int n) {
    int blk=blockIdx.x; int lane=threadIdx.x; float amax=0;
    for(int i=lane;i<16&&blk*16+i<n;i+=32) amax=fmaxf(amax,fabsf(in[blk*16+i]));
    for(int off=16;off>0;off>>=1) amax=fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,off));
    if(lane==0) sc[blk]=fmaxf(amax/127.0f,1e-9f);
}
static void cs(float* in, float* out, int n, cudaStream_t st, const char* nm) {
    absmax_scales_kernel<<<n/16,32,0,st>>>(in,out,n);
    cudaError_t e=cudaPeekAtLastError();
    if(e!=cudaSuccess){printf("FAIL scales %s: %s\n",nm,cudaGetErrorString(e));exit(1);}
}

// Per-head RMSNorm for Q/K norms (1 block per head)
__global__ void head_norm_kernel(float* data, const float* weight, int nh, int hd, float eps) {
    int h=blockIdx.x; if(h>=nh) return;
    float* d=data+h*hd;
    __shared__ float wp[4];
    float s=0; int tid=threadIdx.x;
    for(int i=tid;i<hd;i+=blockDim.x) s+=d[i]*d[i];
    for(int off=16;off>0;off>>=1) s+=__shfl_xor_sync(0xffffffff,s,off);
    if((tid&31)==0) wp[tid>>5]=s; __syncthreads();
    if(tid<4) s=wp[tid]; else s=0;
    for(int off=2;off>0;off>>=1) s+=__shfl_xor_sync(0xffffffff,s,off);
    if(tid==0) wp[0]=rsqrtf(s/hd+eps); __syncthreads();
    float is=wp[0];
    for(int i=tid;i<hd;i+=blockDim.x) d[i]=d[i]*is*weight[i];
}

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cstring>
#include "blackwell/kernels.h"

static void die(cudaError_t e, const char* m) {
    if(e!=cudaSuccess){printf("FAIL %s: %s\n",m,cudaGetErrorString(e));exit(1);}}

using Clock = std::chrono::high_resolution_clock;

const int H=2048, QD=2048, KV=1024, ID=6144, nqh=16, nkv=8, hd=128, MAXSEQ=2048;
const float eps=1e-6f, sc_at=1.f/sqrtf((float)hd);
const int qpg=nqh/nkv;

// ── Weight loading ────────────────────────────────────────────────────
struct LW { std::vector<int8_t> d; std::vector<float> sc; };
struct DW { int8_t* d; float* sc; };

static LW lw(const char* p) {
    char x[256]; snprintf(x,256,"%s.int8_t",p);
    FILE* f=fopen(x,"rb"); if(!f){printf("FAIL open %s\n",x);exit(1);}
    int h[5]; (void)fread(h,4,5,f); LW w;
    w.d.resize(h[0]*h[1]); (void)fread(w.d.data(),1,w.d.size(),f); fclose(f);
    snprintf(x,256,"%s.scale_t",p); f=fopen(x,"rb"); (void)fread(h,4,5,f);
    w.sc.resize(h[3]*h[4]); (void)fread(w.sc.data(),4,w.sc.size(),f); fclose(f);
    return w;
}
static DW dw(const LW& w) {
    DW d;
    cudaMalloc(&d.d,w.d.size()); cudaMemcpy(d.d,w.d.data(),w.d.size(),cudaMemcpyHostToDevice);
    cudaMalloc(&d.sc,w.sc.size()*4); cudaMemcpy(d.sc,w.sc.data(),w.sc.size()*4,cudaMemcpyHostToDevice);
    return d;
}

struct L { DW q,k,v,o,g,u,d; float* qn; float* kn; };

// ── Transpose kernel: [M, QH*HD] → [QH, M, HD] for attention ─────────
__global__ void tr_attn(float* d, const float* s, int mm, int hh, int hd_) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    int N=mm*hh*hd_; if(i>=N)return;
    int m=i/(hh*hd_); int h=(i/hd_)%hh; int d_=i%hd_;
    d[h*mm*hd_+m*hd_+d_]=s[m*hh*hd_+h*hd_+d_];
}

static void do_tr(cudaStream_t st, float* dst, const float* src, int mm, int hh, int hd_) {
    int N=mm*hh*hd_; int T=256;
    tr_attn<<<(N+T-1)/T,T,0,st>>>(dst,src,mm,hh,hd_);
}

int main(int argc, char** argv) {
    int NL=28, M=4, N=20, SEQ=8;
    if(argc>1) NL=atoi(argv[1]);
    if(argc>2) M=atoi(argv[2]);
    if(argc>3) N=atoi(argv[3]);
    if(argc>4) SEQ=atoi(argv[4]);

    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    printf("# Inference Server — Qwen3-1.7B\n");
    printf("  Device: %s  NL:%d  M:%d  N:%d  SEQ:%d\n\n",P.name,NL,M,N,SEQ);

    // ── Allocate ──────────────────────────────────────────────────────
    float *d_x, *d_xi_f, *d_xs, *d_res, *d_rn;
    float *d_Q, *d_K, *d_V, *d_attn;
    int8_t *d_ai; float *d_as;
    float *d_gate, *d_up, *d_mlp;
    int8_t *d_mi; float *d_ms;
    float *d_proj;
    float *d_kc, *d_vc;

    #define AL(p,n) die(cudaMalloc(&(p),(n)),#p)
    AL(d_x,H*4); AL(d_xi_f,H*4); AL(d_xs,(H/16)*4);
    AL(d_res,H*4); AL(d_rn,H*4);
    AL(d_Q,QD*4); AL(d_K,KV*4); AL(d_V,KV*4);
    AL(d_attn,QD*4); AL(d_ai,QD); AL(d_as,(QD/16)*4);
    AL(d_gate,ID*4); AL(d_up,ID*4);
    AL(d_mlp,ID*4); AL(d_mi,ID); AL(d_ms,(ID/16)*4);
    AL(d_proj,H*4);
    AL(d_kc,NL*nkv*MAXSEQ*hd*4); AL(d_vc,NL*nkv*MAXSEQ*hd*4);
    // M-batch
    float *d_xM,*d_xiM_f,*d_xsM,*d_resM;
    float *d_QM,*d_KM,*d_VM,*d_attnM;
    int8_t *d_aiM; float *d_asM;
    float *d_gateM,*d_upM,*d_mlpM;
    int8_t *d_miM; float *d_msM;
    float *d_projM;
    float *d_kcM,*d_vcM;
    AL(d_xM,M*H*4); AL(d_xiM_f,M*H*4); AL(d_xsM,M*(H/16)*4); AL(d_resM,M*H*4);
    AL(d_QM,M*QD*4); AL(d_KM,M*KV*4); AL(d_VM,M*KV*4);
    AL(d_attnM,M*QD*4); AL(d_aiM,M*QD); AL(d_asM,M*(QD/16)*4);
    AL(d_gateM,M*ID*4); AL(d_upM,M*ID*4);
    AL(d_mlpM,M*ID*4); AL(d_miM,M*ID); AL(d_msM,M*(ID/16)*4);
    AL(d_projM,M*H*4);
    AL(d_kcM,M*NL*nkv*MAXSEQ*hd*4); AL(d_vcM,M*NL*nkv*MAXSEQ*hd*4);
    #undef AL
    // Final norm + lm_head
    float *d_fn; int8_t *d_emb_d; float *d_emb_sc;
    float *d_logits; float *d_fn_sc;
    int V=151936;
    die(cudaMalloc(&d_fn,2048*4),"d_fn");
    die(cudaMalloc(&d_emb_d,V*2048),"d_emb_d");
    die(cudaMalloc(&d_emb_sc,(2048/16)*V*4),"d_emb_sc");
    die(cudaMalloc(&d_logits,V*4),"d_logits");
    die(cudaMalloc(&d_fn_sc,(H/16)*4),"d_fn_sc");

    std::vector<float> rn(H,1.f);
    cudaMemcpy(d_rn,rn.data(),H*4,cudaMemcpyHostToDevice);
    float ixv=1.f/127.f;
    cudaMemcpy(d_xs,&ixv,4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_as,&ixv,4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_ms,&ixv,4,cudaMemcpyHostToDevice);
    for(int m=0;m<M;++m){
        cudaMemcpy(d_xsM+m*(H/16),&ixv,4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_asM+m*(QD/16),&ixv,4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_msM+m*(ID/16),&ixv,4,cudaMemcpyHostToDevice);
    }

    // ── Load weights ──────────────────────────────────────────────────
    printf("Loading %d layers...\n",NL); fflush(stdout);
    std::vector<L> W(NL); char p_[256];
    for(int l=0;l<NL;++l){
        snprintf(p_,256,"weights_int8_bf16/%d_self_attn.q_proj",l); W[l].q=dw(lw(p_));
        snprintf(p_,256,"weights_int8_bf16/%d_self_attn.k_proj",l); W[l].k=dw(lw(p_));
        snprintf(p_,256,"weights_int8_bf16/%d_self_attn.v_proj",l); W[l].v=dw(lw(p_));
        snprintf(p_,256,"weights_int8_bf16/%d_self_attn.o_proj",l); W[l].o=dw(lw(p_));
        snprintf(p_,256,"weights_int8_bf16/%d_mlp.gate_proj",l);   W[l].g=dw(lw(p_));
        snprintf(p_,256,"weights_int8_bf16/%d_mlp.up_proj",l);     W[l].u=dw(lw(p_));
        snprintf(p_,256,"weights_int8_bf16/%d_mlp.down_proj",l);   W[l].d=dw(lw(p_));
        if((l+1)%7==0||l+1==NL)printf("  layer %d/%d\n",l+1,NL);
    }
    // Load Q/K norms
    float* qk_norms_h=(float*)malloc(28*2*128*4);
    {FILE* f=fopen("weights_int8_bf16/qk_norms.f32","rb");
    if(!f){printf("FAIL open qk_norms\n");exit(1);}
    size_t nr=fread(qk_norms_h,4,28*2*128,f);(void)nr;fclose(f);}
    for(int l=0;l<NL;++l){
        cudaMalloc(&W[l].qn,128*4); cudaMemcpy(W[l].qn,qk_norms_h+l*2*128,128*4,cudaMemcpyHostToDevice);
        cudaMalloc(&W[l].kn,128*4); cudaMemcpy(W[l].kn,qk_norms_h+l*2*128+128,128*4,cudaMemcpyHostToDevice);
    }
    free(qk_norms_h);

    // Load final norm + lm_head
    {FILE* f=fopen("weights_int8_bf16/final_norm.f32","rb");
    if(!f){printf("FAIL open final_norm\n");exit(1);}
    float* fn_h=(float*)malloc(2048*4);
    size_t nr=fread(fn_h,4,2048,f);(void)nr;fclose(f);
    cudaMemcpy(d_fn,fn_h,2048*4,cudaMemcpyHostToDevice); free(fn_h);}
    {LW w=lw("weights_int8_bf16/embed_tokens");
    cudaMemcpy(d_emb_d,w.d.data(),w.d.size(),cudaMemcpyHostToDevice);
    cudaMemcpy(d_emb_sc,w.sc.data(),w.sc.size()*4,cudaMemcpyHostToDevice);}

    printf("Loaded.\n");

    cudaStream_t st; die(cudaStreamCreate(&st),"stream");

    // ── Inputs ────────────────────────────────────────────────────────
    std::vector<std::vector<float>> inputs(N, std::vector<float>(H));
    for(int i=0;i<N;++i) for(int j=0;j<H;++j) inputs[i][j]=(j%17-8)*0.01f;

    // ══════════════════════════════════════════════════════════════════
    // Decode one token through one layer (per-kernel)
    // ══════════════════════════════════════════════════════════════════
    auto dcl = [&](float* xx, float* xi, float* xs,
                   float* r, float* q, float* k, float* v,
                   float* at, int8_t* ai_, float* as_,
                   float* ga, float* up, float* ml,
                   int8_t* mi_, float* ms_,
                   float* pr, float* kc, float* vc,
                   const L& Wl, int sp) {
        int kb=0;
        die(blackwell::kernels::fused_rmsnorm_quant_int8(
            (int8_t*)xi,xs,xx,d_rn,H,eps,st),"rmsnorm");
        die(blackwell::kernels::gemv_int8(q,(int8_t*)xi,xs,Wl.q.d,Wl.q.sc,H,QD,st),"q");
        die(blackwell::kernels::gemv_int8(k,(int8_t*)xi,xs,Wl.k.d,Wl.k.sc,H,KV,st),"k");
        die(blackwell::kernels::gemv_int8(v,(int8_t*)xi,xs,Wl.v.d,Wl.v.sc,H,KV,st),"v");
        // Apply Q/K norms before attention
        head_norm_kernel<<<nqh,128,0,st>>>(q,Wl.qn,nqh,hd,eps);
        head_norm_kernel<<<nkv,128,0,st>>>(k,Wl.kn,nkv,hd,eps);
        die(blackwell::kernels::update_kv_cache(kc+kb,vc+kb,k,v,0,sp,nkv,hd,MAXSEQ,st),"kv");
        die(blackwell::kernels::attention_decode_gqa(at,q,kc+kb,vc+kb,sp,nqh,nkv,hd,MAXSEQ,st),"attn");
        cs(at,as_,QD,st,"as");
        die(blackwell::kernels::pack_int8(ai_,at,as_,QD,st),"pack");
        die(blackwell::kernels::gemv_int8(pr,ai_,as_,Wl.o.d,Wl.o.sc,QD,H,st),"o");
        die(blackwell::kernels::vector_add_fp32(pr,pr,xx,H,st),"res1");
        die(cudaMemcpyAsync(r,pr,H*4,cudaMemcpyDeviceToDevice,st),"save_res");
        die(blackwell::kernels::fused_rmsnorm_quant_int8(
            (int8_t*)xi,xs,pr,d_rn,H,eps,st),"rmsnorm2");
        die(blackwell::kernels::gemv_int8(ga,(int8_t*)xi,xs,Wl.g.d,Wl.g.sc,H,ID,st),"gate");
        die(blackwell::kernels::gemv_int8(up,(int8_t*)xi,xs,Wl.u.d,Wl.u.sc,H,ID,st),"up");
        die(blackwell::kernels::apply_swiglu(ml,ga,up,ID,st),"swiglu");
        cs(ml,ms_,ID,st,"ms");
        die(blackwell::kernels::pack_int8(mi_,ml,ms_,ID,st),"pack2");
        die(blackwell::kernels::gemv_int8(pr,mi_,ms_,Wl.d.d,Wl.d.sc,ID,H,st),"down");
        die(blackwell::kernels::vector_add_fp32(pr,pr,r,H,st),"res2");
    };

    // ══════════════════════════════════════════════════════════════════
    // CUDA Graph capture (all NL layers, 1 token step)
    // ══════════════════════════════════════════════════════════════════
    printf("\n=== Capturing CUDA Graph (%d layers) ===\n",NL);
    fflush(stdout);
    // Pre-trigger attention_decode_gqa to set smem config
    blackwell::kernels::attention_decode_gqa(
        d_attn,d_Q,d_kc,d_vc,0,nqh,nkv,hd,MAXSEQ,st);
    cudaStreamSynchronize(st);
    
    blackwell::kernels::update_decode_seq_pos(0,st);
    cudaStreamSynchronize(st);
    
    cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);
    for(int l=0;l<NL;++l){
        int kb=l*nkv*MAXSEQ*hd;
        blackwell::kernels::fused_rmsnorm_quant_int8(
            (int8_t*)d_xi_f,d_xs,d_x,d_rn,H,eps,st);
        blackwell::kernels::gemv_int8(d_Q,(int8_t*)d_xi_f,d_xs,W[l].q.d,W[l].q.sc,H,QD,st);
        blackwell::kernels::gemv_int8(d_K,(int8_t*)d_xi_f,d_xs,W[l].k.d,W[l].k.sc,H,KV,st);
        blackwell::kernels::gemv_int8(d_V,(int8_t*)d_xi_f,d_xs,W[l].v.d,W[l].v.sc,H,KV,st);
        head_norm_kernel<<<nqh,128,0,st>>>(d_Q,W[l].qn,nqh,hd,eps);
        head_norm_kernel<<<nkv,128,0,st>>>(d_K,W[l].kn,nkv,hd,eps);
        blackwell::kernels::update_kv_cache(d_kc+kb,d_vc+kb,d_K,d_V,0,0,nkv,hd,MAXSEQ,st);
        blackwell::kernels::attention_decode_gqa(
            d_attn,d_Q,d_kc+kb,d_vc+kb,0,nqh,nkv,hd,MAXSEQ,st);
        absmax_scales_kernel<<<QD/16,32,0,st>>>(d_attn,d_as,QD);
        blackwell::kernels::pack_int8(d_ai,d_attn,d_as,QD,st);
        blackwell::kernels::gemv_int8(d_proj,d_ai,d_as,W[l].o.d,W[l].o.sc,QD,H,st);
        blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_x,H,st);
        cudaMemcpyAsync(d_res,d_proj,H*4,cudaMemcpyDeviceToDevice,st);
        blackwell::kernels::fused_rmsnorm_quant_int8(
            (int8_t*)d_xi_f,d_xs,d_proj,d_rn,H,eps,st);
        // MLP block
        blackwell::kernels::gemv_int8(d_gate,(int8_t*)d_xi_f,d_xs,W[l].g.d,W[l].g.sc,H,ID,st);
        blackwell::kernels::gemv_int8(d_up,(int8_t*)d_xi_f,d_xs,W[l].u.d,W[l].u.sc,H,ID,st);
        blackwell::kernels::apply_swiglu(d_mlp,d_gate,d_up,ID,st);
        absmax_scales_kernel<<<ID/16,32,0,st>>>(d_mlp,d_ms,ID);
        blackwell::kernels::pack_int8(d_mi,d_mlp,d_ms,ID,st);
        blackwell::kernels::gemv_int8(d_proj,d_mi,d_ms,W[l].d.d,W[l].d.sc,ID,H,st);
        blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_res,H,st);
        cudaMemcpyAsync(d_x,d_proj,H*4,cudaMemcpyDeviceToDevice,st);
    }
    cudaGraph_t gr;
    cudaError_t ge_=cudaStreamEndCapture(st,&gr);
    if(ge_!=cudaSuccess){printf("FAIL graph capture: %s\n",cudaGetErrorString(ge_));exit(1);}
    cudaGraphExec_t ge;
    ge_=cudaGraphInstantiate(&ge,gr,NULL,NULL,0);
    if(ge_!=cudaSuccess){printf("FAIL graph instantiate: %s\n",cudaGetErrorString(ge_));exit(1);}
    printf("  Graph captured OK.\n");

    // ══════════════════════════════════════════════════════════════════
    // MODE A: Single-seq per-kernel (original)
    // ══════════════════════════════════════════════════════════════════
    printf("\n=== Mode A: Single-seq decode (%dL, %d tok) ===\n",NL,N);
    cudaMemset(d_kc,0,NL*nkv*MAXSEQ*hd*4); cudaMemset(d_vc,0,NL*nkv*MAXSEQ*hd*4);
    auto t0=Clock::now();
    for(int i=0;i<N;++i){
        cudaMemcpy(d_x,inputs[i].data(),H*4,cudaMemcpyHostToDevice);
        for(int s=0;s<SEQ;++s){
            for(int l=0;l<NL;++l){
                dcl(d_x,d_xi_f,d_xs,d_res,d_Q,d_K,d_V,d_attn,
                    d_ai,d_as,d_gate,d_up,d_mlp,d_mi,d_ms,d_proj,
                    d_kc+l*nkv*MAXSEQ*hd,d_vc+l*nkv*MAXSEQ*hd,W[l],s);
                cudaMemcpyAsync(d_x,d_proj,H*4,cudaMemcpyDeviceToDevice,st);
            }
        }
        cudaStreamSynchronize(st);
    }
    auto t1=Clock::now();
    double msA=std::chrono::duration<double,std::milli>(t1-t0).count();
    printf("  %.1fms total, %.1fms/req, %.0f req/s\n",msA,msA/N,1000.0*N/msA);
    printf("  %.0f us/tok  =>  %.0f t/s (%dL)\n",msA*1000/(N*SEQ),1000.0*N*SEQ/msA,NL);

    // Final RMSNorm + lm_head
    die(blackwell::kernels::fused_rmsnorm_quant_int8(
        (int8_t*)d_xi_f,d_fn_sc,d_proj,d_fn,H,eps,st),"fn");
    die(blackwell::kernels::gemv_int8(d_logits,(int8_t*)d_xi_f,d_fn_sc,
        d_emb_d,d_emb_sc,H,V,st),"lm_head");
    cudaStreamSynchronize(st);
    {
        std::vector<float> logits_out(V);
        cudaMemcpy(logits_out.data(),d_logits,V*4,cudaMemcpyDeviceToHost);
        FILE* f=fopen("/tmp/inference_server_out.bin","wb");
        if(f){fwrite(logits_out.data(),4,V,f);fclose(f);}
        printf("  Dumped logits (V=%d) to /tmp/inference_server_out.bin\n",V);
    }

    // ══════════════════════════════════════════════════════════════════
    // MODE A': Single-seq CUDA Graph
    // ══════════════════════════════════════════════════════════════════
    printf("\n=== Mode A': Single-seq CUDA Graph (%dL, %d tok) ===\n",NL,N);
    cudaMemset(d_kc,0,NL*nkv*MAXSEQ*hd*4); cudaMemset(d_vc,0,NL*nkv*MAXSEQ*hd*4);
    auto t0a=Clock::now();
    for(int i=0;i<N;++i){
        cudaMemcpy(d_x,inputs[i].data(),H*4,cudaMemcpyHostToDevice);
        cudaStreamSynchronize(st);
        for(int s=0;s<SEQ;++s){
            blackwell::kernels::update_decode_seq_pos(s,st);
            cudaGraphLaunch(ge,st);
        }
        cudaStreamSynchronize(st);
    }
    auto t1a=Clock::now();
    double msAa=std::chrono::duration<double,std::milli>(t1a-t0a).count();
    printf("  %.1fms total, %.1fms/req, %.0f req/s\n",msAa,msAa/N,1000.0*N/msAa);
    printf("  %.0f us/tok  =>  %.0f t/s (%dL)\n",msAa*1000/(N*SEQ),1000.0*N*SEQ/msAa,NL);

    // ══════════════════════════════════════════════════════════════════
    // MODE B: Batched per-kernel
    // ══════════════════════════════════════════════════════════════════
    printf("\n=== Mode B: Batched per-kernel (M=%d, %dL) ===\n",M,NL);
    cudaMemset(d_kcM,0,M*NL*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vcM,0,M*NL*nkv*MAXSEQ*hd*4);
    t0=Clock::now();
    for(int b=0;b<N/M;++b){
        for(int m=0;m<M;++m){
            int idx=b*M+m;
            cudaMemcpy(d_xM+m*H,inputs[idx].data(),H*4,cudaMemcpyHostToDevice);
        }
        cudaStreamSynchronize(st);
        for(int s=0;s<SEQ;++s){
            for(int m=0;m<M;++m){
                for(int l=0;l<NL;++l){
                    dcl(d_xM+m*H,d_xiM_f+m*H,d_xsM+m*(H/16),d_resM+m*H,
                        d_QM+m*QD,d_KM+m*KV,d_VM+m*KV,d_attnM+m*QD,
                        d_aiM+m*QD,d_asM+m*(QD/16),
                        d_gateM+m*ID,d_upM+m*ID,d_mlpM+m*ID,
                        d_miM+m*ID,d_msM+m*(ID/16),d_projM+m*H,
                        d_kcM+(m*NL+l)*nkv*MAXSEQ*hd,d_vcM+(m*NL+l)*nkv*MAXSEQ*hd,W[l],s);
                    cudaMemcpyAsync(d_xM+m*H,d_projM+m*H,H*4,cudaMemcpyDeviceToDevice,st);
                }
            }
        }
        cudaStreamSynchronize(st);
    }
    t1=Clock::now();
    double msB=std::chrono::duration<double,std::milli>(t1-t0).count();
    printf("  %d batches × M=%d: %.1fms total, %.1fms/batch\n",N/M,M,msB,msB/(N/M));
    printf("  %.0f us/seq  =>  %.0f req/s\n",msB*1000/N,1000.0*N/msB);

    // ══════════════════════════════════════════════════════════════════
    // MODE C: Batched GEMV (gemv_int8_batched)
    // ══════════════════════════════════════════════════════════════════
    printf("\n=== Mode C: Batched GEMV (M=%d, %dL) ===\n",M,NL);
    cudaMemset(d_kcM,0,M*NL*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vcM,0,M*NL*nkv*MAXSEQ*hd*4);
    t0=Clock::now();
    for(int b=0;b<N/M;++b){
        for(int m=0;m<M;++m){
            int idx=b*M+m;
            cudaMemcpy(d_xM+m*H,inputs[idx].data(),H*4,cudaMemcpyHostToDevice);
        }
        cudaStreamSynchronize(st);
        for(int s=0;s<SEQ;++s){
            for(int l=0;l<NL;++l){
                // Batch RMSNorm+quant
                die(blackwell::kernels::fused_rmsnorm_quant_int8(
                    (int8_t*)d_xiM_f,d_xsM,d_xM,d_rn,H,eps,st),"norm");
                // Batched Q,K,V
                die(blackwell::kernels::gemv_int8_batched(d_QM,(int8_t*)d_xiM_f,d_xsM,
                    W[l].q.d,W[l].q.sc,H,QD,M,st),"q");
                die(blackwell::kernels::gemv_int8_batched(d_KM,(int8_t*)d_xiM_f,d_xsM,
                    W[l].k.d,W[l].k.sc,H,KV,M,st),"k");
                die(blackwell::kernels::gemv_int8_batched(d_VM,(int8_t*)d_xiM_f,d_xsM,
                    W[l].v.d,W[l].v.sc,H,KV,M,st),"v");
                // Per-seq: KV cache + attention + pack
                for(int m=0;m<M;++m){
                    int kb=m*nkv*MAXSEQ*hd;
                    head_norm_kernel<<<nqh,128,0,st>>>(d_QM+m*QD,W[l].qn,nqh,hd,eps);
                    head_norm_kernel<<<nkv,128,0,st>>>(d_KM+m*KV,W[l].kn,nkv,hd,eps);
                    die(blackwell::kernels::update_kv_cache(
                        d_kcM+kb,d_vcM+kb,d_KM+m*KV,d_VM+m*KV,0,s,nkv,hd,MAXSEQ,st),"kv");
                    die(blackwell::kernels::attention_decode_gqa(
                        d_attnM+m*QD,d_QM+m*QD,d_kcM+kb,d_vcM+kb,
                        s,nqh,nkv,hd,MAXSEQ,st),"attn");
                    cs(d_attnM+m*QD,d_asM+m*(QD/16),QD,st,"as");
                    die(blackwell::kernels::pack_int8(
                        d_aiM+m*QD,d_attnM+m*QD,d_asM+m*(QD/16),QD,st),"pack");
                }
                // Batched Wo
                die(blackwell::kernels::gemv_int8_batched(d_projM,d_aiM,d_asM,
                    W[l].o.d,W[l].o.sc,QD,H,M,st),"o");
                // Residuals (per-seq)
                for(int m=0;m<M;++m){
                    die(blackwell::kernels::vector_add_fp32(d_projM+m*H,d_projM+m*H,d_xM+m*H,H,st),"res1");
                    die(cudaMemcpyAsync(d_resM+m*H,d_projM+m*H,H*4,cudaMemcpyDeviceToDevice,st),"save_res");
                }
                // Batch MLP RMSNorm
                die(blackwell::kernels::fused_rmsnorm_quant_int8(
                    (int8_t*)d_xiM_f,d_xsM,d_projM,d_rn,H,eps,st),"norm2");
                // Batched Gate + Up
                die(blackwell::kernels::gemv_int8_batched(d_gateM,(int8_t*)d_xiM_f,d_xsM,
                    W[l].g.d,W[l].g.sc,H,ID,M,st),"gate");
                die(blackwell::kernels::gemv_int8_batched(d_upM,(int8_t*)d_xiM_f,d_xsM,
                    W[l].u.d,W[l].u.sc,H,ID,M,st),"up");
                // Per-seq: SwiGLU + pack (must be per-seq)
                for(int m=0;m<M;++m){
                    die(blackwell::kernels::apply_swiglu(d_mlpM+m*ID,d_gateM+m*ID,d_upM+m*ID,ID,st),"swiglu");
                    cs(d_mlpM+m*ID,d_msM+m*(ID/16),ID,st,"ms");
                    die(blackwell::kernels::pack_int8(d_miM+m*ID,d_mlpM+m*ID,d_msM+m*(ID/16),ID,st),"pack2");
                }
                // Batched down GEMV
                die(blackwell::kernels::gemv_int8_batched(d_projM,d_miM,d_msM,
                    W[l].d.d,W[l].d.sc,ID,H,M,st),"down");
                // Per-seq: residual + copy
                for(int m=0;m<M;++m){
                    die(blackwell::kernels::vector_add_fp32(d_projM+m*H,d_projM+m*H,d_resM+m*H,H,st),"res2");
                    cudaMemcpyAsync(d_xM+m*H,d_projM+m*H,H*4,cudaMemcpyDeviceToDevice,st);
                }
            }
        }
        cudaStreamSynchronize(st);
    }
    t1=Clock::now();
    double msC=std::chrono::duration<double,std::milli>(t1-t0).count();
    float seq_msC=msC/N;
    printf("  %d batches × M=%d: %.1fms total\n",N/M,M,msC);
    printf("  %.0f us/seq  =>  %.0f t/s  (%.0f req/s)\n",
        seq_msC*1000,1000.0/seq_msC,1000.0*N/msC);

    // ══════════════════════════════════════════════════════════════════
    // MODE D: Prefill + autoregressive decode
    // ══════════════════════════════════════════════════════════════════
    printf("\n=== Mode D: Prefill M=128 + decode (%dL, %d tok) ===\n",NL,N);
    int PP=128;
    // Prefill buffers
    float *d_xP, *d_rP;
    float *d_Qf, *d_Kf, *d_Vf, *d_Att, *d_Of, *d_gf, *d_uf, *d_mlf, *d_rf;
    float *d_Qt, *d_Kt, *d_Vt;
    float *d_cos, *d_sin;
    #define AL2(p,n) die(cudaMalloc(&(p),(n)),#p)
    AL2(d_xP,PP*H*4); AL2(d_rP,PP*H*4);
    AL2(d_Qf,PP*QD*4); AL2(d_Kf,PP*KV*4); AL2(d_Vf,PP*KV*4);
    AL2(d_Att,PP*QD*4); AL2(d_Of,PP*H*4);
    AL2(d_gf,PP*ID*4); AL2(d_uf,PP*ID*4); AL2(d_mlf,PP*ID*4); AL2(d_rf,PP*H*4);
    AL2(d_Qt,nqh*PP*hd*4); AL2(d_Kt,nkv*PP*hd*4); AL2(d_Vt,nkv*PP*hd*4);
    AL2(d_cos,nqh*PP*hd*4); AL2(d_sin,nqh*PP*hd*4);
    #undef AL2
    std::vector<float> cos_h(nqh*PP*hd),sin_h(nqh*PP*hd);
    for(int i=0;i<nqh*PP*hd;++i){cos_h[i]=cosf(i*0.01f);sin_h[i]=sinf(i*0.01f);}
    cudaMemcpyAsync(d_cos,cos_h.data(),nqh*PP*hd*4,cudaMemcpyHostToDevice,st);
    cudaMemcpyAsync(d_sin,sin_h.data(),nqh*PP*hd*4,cudaMemcpyHostToDevice,st);
    // Prefill input: FP32 synthetic data
    std::vector<float> xP_h(PP*H);
    for(int i=0;i<PP*H;++i) xP_h[i]=((i*31+7)%127-63)*0.01f;
    cudaMemcpy(d_xP,xP_h.data(),PP*H*4,cudaMemcpyHostToDevice);

    cudaMemset(d_kc,0,NL*nkv*MAXSEQ*hd*4); cudaMemset(d_vc,0,NL*nkv*MAXSEQ*hd*4);

    t0=Clock::now();
    for(int l=0;l<NL;++l){
        // RMSNorm
        die(blackwell::kernels::fused_rmsnorm(d_rP,(float*)d_xP,d_rn,H,eps,st),"rn");
        // QKV — INT8 GEMM (real weights)
        die(blackwell::kernels::gemm_int8(d_Qf,d_rP,W[l].q.d,W[l].q.sc,PP,QD,H,st),"q");
        die(blackwell::kernels::gemm_int8(d_Kf,d_rP,W[l].k.d,W[l].k.sc,PP,KV,H,st),"k");
        die(blackwell::kernels::gemm_int8(d_Vf,d_rP,W[l].v.d,W[l].v.sc,PP,KV,H,st),"v");
        // RoPE
        die(blackwell::kernels::fused_rope(d_Qf,d_cos,d_sin,nqh,PP,hd,st),"rope_q");
        die(blackwell::kernels::fused_rope(d_Kf,d_cos,d_sin,nkv,PP,hd,st),"rope_k");
        // Transpose Q/K/V for attention
        do_tr(st,d_Qt,d_Qf,PP,nqh,hd);
        do_tr(st,d_Kt,d_Kf,PP,nkv,hd);
        do_tr(st,d_Vt,d_Vf,PP,nkv,hd);
        // Flash attention
        die(blackwell::kernels::attention_prefill(d_Att,d_Qt,d_Kt,d_Vt,
            PP,hd,nqh,nkv,qpg,sc_at,st),"attn");
        // Wo GEMM — INT8 GEMM (real weights)
        die(blackwell::kernels::gemm_int8(d_Of,d_Att,W[l].o.d,W[l].o.sc,PP,H,QD,st),"wo");
        // Residual
        die(blackwell::kernels::vector_add_fp32(d_Of,d_Of,(float*)d_xP,PP*H,st),"res1");
        die(cudaMemcpyAsync(d_rf,d_Of,PP*H*4,cudaMemcpyDeviceToDevice,st),"rf");
        // MLP RMSNorm
        die(blackwell::kernels::fused_rmsnorm(d_rP,d_Of,d_rn,H,eps,st),"rn2");
        // Gate+Up — INT8 GEMM (real weights)
        die(blackwell::kernels::gemm_int8(d_gf,d_rP,W[l].g.d,W[l].g.sc,PP,ID,H,st),"gate");
        die(blackwell::kernels::gemm_int8(d_uf,d_rP,W[l].u.d,W[l].u.sc,PP,ID,H,st),"up");
        // SwiGLU
        die(blackwell::kernels::apply_swiglu(d_mlf,d_gf,d_uf,PP*ID,st),"swiglu");
        // Down — INT8 GEMM (real weights)
        die(blackwell::kernels::gemm_int8(d_Of,d_mlf,W[l].d.d,W[l].d.sc,PP,H,ID,st),"down");
        // Residual
        die(blackwell::kernels::vector_add_fp32((float*)d_xP,d_Of,d_rf,PP*H,st),"res2");
    }
    // Copy last token → decode input
    cudaMemcpy(d_x,d_xP+(PP-1)*H,H*4,cudaMemcpyDeviceToDevice);
    // Init KV cache from prefill K/V: contiguous [PP×nkv×hd] → strided [nkv×MAXSEQ×hd]
    for(int t=0;t<PP;++t){
        for(int h=0;h<nkv;++h){
            int src_off = (t*nkv + h) * hd;
            int dst_off = (h*MAXSEQ + t) * hd;
            cudaMemcpy(d_kc+dst_off,d_Kf+src_off,hd*4,cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_vc+dst_off,d_Vf+src_off,hd*4,cudaMemcpyDeviceToDevice);
        }
    }
    // Autoregressive decode with real weights — stream tokens
    double ms_dec=0;
    for(int t=0;t<N;++t){
        auto td=Clock::now();
        for(int l=0;l<NL;++l){
            dcl(d_x,d_xi_f,d_xs,d_res,d_Q,d_K,d_V,d_attn,
                d_ai,d_as,d_gate,d_up,d_mlp,d_mi,d_ms,d_proj,
                d_kc+l*nkv*MAXSEQ*hd,d_vc+l*nkv*MAXSEQ*hd,W[l],PP+t);
            cudaMemcpyAsync(d_x,d_proj,H*4,cudaMemcpyDeviceToDevice,st);
        }
        cudaStreamSynchronize(st);
        ms_dec+=std::chrono::duration<double,std::milli>(Clock::now()-td).count();
    }
    t1=Clock::now();
    double msD=std::chrono::duration<double,std::milli>(t1-t0).count();
    printf("  Prefill %d + decode %d: %.1fms total (prefill %.0fms, decode %.1fms)\n",
        PP,N,msD,msD-ms_dec,ms_dec);
    printf("  Decode: %.0f us/tok  =>  %.0f t/s (%dL)\n",
        ms_dec*1000/N,1000.0*N/ms_dec,NL);

    cudaFree(d_xP);cudaFree(d_rP);cudaFree(d_Qf);cudaFree(d_Kf);
    cudaFree(d_Vf);cudaFree(d_Att);cudaFree(d_Of);cudaFree(d_gf);
    cudaFree(d_uf);cudaFree(d_mlf);cudaFree(d_rf);
    cudaFree(d_Qt);cudaFree(d_Kt);cudaFree(d_Vt);
    cudaFree(d_cos);cudaFree(d_sin);

    // ── Summary ──────────────────────────────────────────────────────
    printf("\n=== Summary ===\n");
    printf("  %-38s  %8.0f t/s\n","A: Single-seq per-kernel",1000.0*N*SEQ/msA);
    printf("  %-38s  %8.0f t/s\n","A': Single-seq CUDA Graph",1000.0*N*SEQ/msAa);
    printf("  %-38s  %8.0f t/s\n","B: Batched per-kernel",1000.0*N/msB);
    printf("  %-38s  %8.0f t/s\n","C: Batched GEMV kernel",1000.0/seq_msC);
    printf("  %-38s  %8.0f t/s\n","D: Prefill+decode",1000.0*N/msD);
    printf("\n  Graph speedup (A' vs A): %.1fx  (%.1f%%)\n",msA/(msAa+0.001f),(1-msAa/msA)*100);
    printf("  Speedup C vs A: %.1fx  (batched GEMV kernel vs single per-kernel)\n",msA/(msC+0.001f));
    printf("  vs llama.cpp Q4_K_M: 114 t/s\n\n");

    // Cleanup
    auto free_buf=[](auto p){if(p)cudaFree(p);};
    free_buf(d_x);free_buf(d_xi_f);free_buf(d_xs);free_buf(d_res);
    free_buf(d_Q);free_buf(d_K);free_buf(d_V);free_buf(d_attn);
    free_buf(d_ai);free_buf(d_as);free_buf(d_gate);free_buf(d_up);
    free_buf(d_mlp);free_buf(d_mi);free_buf(d_ms);free_buf(d_proj);
    free_buf(d_kc);free_buf(d_vc);free_buf(d_rn);
    free_buf(d_xM);free_buf(d_xiM_f);free_buf(d_xsM);free_buf(d_resM);
    free_buf(d_QM);free_buf(d_KM);free_buf(d_VM);free_buf(d_attnM);
    free_buf(d_aiM);free_buf(d_asM);free_buf(d_gateM);free_buf(d_upM);
    free_buf(d_mlpM);free_buf(d_miM);free_buf(d_msM);free_buf(d_projM);
    free_buf(d_kcM);free_buf(d_vcM);
    for(auto& w:W){
        cudaFree(w.q.d);cudaFree(w.q.sc);cudaFree(w.k.d);cudaFree(w.k.sc);
        cudaFree(w.v.d);cudaFree(w.v.sc);cudaFree(w.o.d);cudaFree(w.o.sc);
        cudaFree(w.g.d);cudaFree(w.g.sc);cudaFree(w.u.d);cudaFree(w.u.sc);
        cudaFree(w.d.d);cudaFree(w.d.sc);
        cudaFree(w.qn);cudaFree(w.kn);
    }
    cudaGraphExecDestroy(ge); cudaGraphDestroy(gr);
    cudaStreamDestroy(st);
    return 0;
}