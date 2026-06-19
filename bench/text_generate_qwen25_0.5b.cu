#include "blackwell/int4_weights.h"
using namespace blackwell::weights;
// bench/text_generate_qwen25_0.5b.cu — End-to-end text generation with INT4 Qwen2.5-0.5B
//
// Tokenize prompt → INT4 embedding lookup → 24L INT4 decode
// → final norm → INT4 lm_head GEMV → GPU sampling → print tokens.
//
// Build:
//   cmake --build build --target text_generate_qwen25_0.5b

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cstring>
#include <string>
#include <cstdint>
#include <cmath>
#include "blackwell/kernels.h"
#include "blackwell/bpe_tokenizer.h"
#include "blackwell/bench_kernels.h"

static void die(cudaError_t e, const char* m) {
    if(e!=cudaSuccess){printf("FAIL %s: %s\n",m,cudaGetErrorString(e));exit(1);}
}

using Clock = std::chrono::high_resolution_clock;

// Qwen2.5-0.5B dimensions
const int H=896, Q=896, KV=128, I=4864;
const int nqh=14, nkv=2, hd=64, MAXSEQ=4096;
const float eps=1e-6f;
const int V=151936;
const int NL=24;
const float rope_theta=1000000.0f;

struct LW4 {
    DevW4f16 q,k,v,o,g,u,d;
    float* qnorm; float* knorm; float* rn_in; float* rn_post;
};

int main(int argc, char** argv) {
    const char* prompt = "Once upon a time";
    int max_new = 30;
    const char* wdir = "weights_qwen25_0.5b";
    if(argc>1) prompt=argv[1];
    if(argc>2) max_new=atoi(argv[2]);
    if(argc>3) wdir=argv[3];

    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    fprintf(stderr,"# Text Generation — Qwen2.5-0.5B INT4 (GGUF converted)\n");
    fprintf(stderr,"  Weights: %s\n",wdir);
    fprintf(stderr,"  Device: %s\n",P.name);
    fprintf(stderr,"  Prompt: \"%s\"\n",prompt);
    fprintf(stderr,"  Max new: %d\n",max_new);

    blackwell::BpeTokenizer tok;
    char p[256]; snprintf(p,256,"%s/tokenizer_data.bin",wdir);
    if(tok.load(p)!=0){ fprintf(stderr,"FAIL tokenizer\n"); return 1; }
    auto ids = tok.encode(prompt);
    fprintf(stderr,"  Prompt tokens: %zu\n", ids.size());

    float *d_x32, *d_xi_f, *d_res;
    uint8_t *d_x_i4; float *d_x_i4_sc;
    float *d_Q, *d_K, *d_V, *d_attn;
    uint8_t *d_attn_i4; float *d_attn_i4_sc;
    float *d_proj, *d_gate, *d_up;
    uint8_t *d_mlp_i4; float *d_mlp_i4_sc;
    float *d_fn, *d_kc, *d_vc, *d_logits;
    int *d_next_id;

    #define AL(p,n) die(cudaMalloc(&(p),(n)),"malloc "#p)
    AL(d_x32,H*4); AL(d_xi_f,H*4); AL(d_res,H*4);
    AL(d_x_i4,H/2); AL(d_x_i4_sc,(H/16)*4);
    AL(d_Q,Q*4); AL(d_K,KV*4); AL(d_V,KV*4); AL(d_attn,Q*4);
    AL(d_attn_i4,Q/2); AL(d_attn_i4_sc,(Q/16)*4);
    AL(d_proj,H*4); AL(d_gate,I*4); AL(d_up,I*4);
    AL(d_mlp_i4,I/2); AL(d_mlp_i4_sc,(I/16)*4);
    AL(d_fn,H*4);
    AL(d_kc,(size_t)NL*nkv*MAXSEQ*hd*4);
    AL(d_vc,(size_t)NL*nkv*MAXSEQ*hd*4);
    AL(d_logits,V*4); AL(d_next_id,4);
    #undef AL

    // Init quant scales
    float iv7=1.f/7.f;
    { std::vector<float> tmp(H/16,iv7); cudaMemcpy(d_x_i4_sc,tmp.data(),(H/16)*4,cudaMemcpyHostToDevice); }
    { std::vector<float> tmp(Q/16,iv7); cudaMemcpy(d_attn_i4_sc,tmp.data(),(Q/16)*4,cudaMemcpyHostToDevice); }
    { std::vector<float> tmp(I/16,iv7); cudaMemcpy(d_mlp_i4_sc,tmp.data(),(I/16)*4,cudaMemcpyHostToDevice); }

    fprintf(stderr,"Loading weights...\n");
    std::vector<LW4> W(NL);
    for(int l=0;l<NL;++l){
        snprintf(p,256,"%s/%d_self_attn.q_proj",wdir,l); W[l].q=upload_w4_f16sc(p);
        snprintf(p,256,"%s/%d_self_attn.k_proj",wdir,l); W[l].k=upload_w4_f16sc(p);
        snprintf(p,256,"%s/%d_self_attn.v_proj",wdir,l); W[l].v=upload_w4_f16sc(p);
        snprintf(p,256,"%s/%d_self_attn.o_proj",wdir,l); W[l].o=upload_w4_f16sc(p);
        snprintf(p,256,"%s/%d_mlp.gate_proj",wdir,l); W[l].g=upload_w4_f16sc(p);
        snprintf(p,256,"%s/%d_mlp.up_proj",wdir,l); W[l].u=upload_w4_f16sc(p);
        snprintf(p,256,"%s/%d_mlp.down_proj",wdir,l); W[l].d=upload_w4_f16sc(p);
    }
    // QK norms
    float* qk_h=(float*)malloc(NL*2*hd*4);
    snprintf(p,256,"%s/qk_norms.f32",wdir);
    {FILE*f=fopen(p,"rb");if(!f){printf("FAIL open %s\n",p);exit(1);}size_t nr=fread(qk_h,4,NL*2*hd,f);if(nr!=(size_t)NL*2*hd){printf("FAIL read qk_norms\n");exit(1);}fclose(f);}
    for(int l=0;l<NL;++l){
        cudaMalloc(&W[l].qnorm,hd*4);cudaMemcpy(W[l].qnorm,qk_h+l*2*hd,hd*4,cudaMemcpyHostToDevice);
        cudaMalloc(&W[l].knorm,hd*4);cudaMemcpy(W[l].knorm,qk_h+l*2*hd+hd,hd*4,cudaMemcpyHostToDevice);
    }free(qk_h);
    // Layer norms
    for(int l=0;l<NL;++l){
        float* w=(float*)malloc(H*4);
        snprintf(p,256,"%s/%d_input_layernorm.f32",wdir,l);
        {FILE*f=fopen(p,"rb");if(!f){printf("FAIL open %s\n",p);exit(1);}size_t nr=fread(w,4,H,f);if(nr!=(size_t)H){printf("FAIL read input_norm %d\n",l);exit(1);}fclose(f);}
        cudaMalloc(&W[l].rn_in,H*4);cudaMemcpy(W[l].rn_in,w,H*4,cudaMemcpyHostToDevice);
        snprintf(p,256,"%s/%d_post_attention_layernorm.f32",wdir,l);
        {FILE*f=fopen(p,"rb");if(!f){printf("FAIL open %s\n",p);exit(1);}size_t nr=fread(w,4,H,f);if(nr!=(size_t)H){printf("FAIL read post_norm %d\n",l);exit(1);}fclose(f);}
        cudaMalloc(&W[l].rn_post,H*4);cudaMemcpy(W[l].rn_post,w,H*4,cudaMemcpyHostToDevice);
        free(w);
    }
    // Final norm
    {float*w=(float*)malloc(H*4);
    snprintf(p,256,"%s/final_norm.f32",wdir);
    FILE*f=fopen(p,"rb");if(!f){printf("FAIL open %s\n",p);exit(1);}size_t nr=fread(w,4,H,f);if(nr!=(size_t)H){printf("FAIL read final_norm\n");exit(1);}fclose(f);
    cudaMemcpy(d_fn,w,H*4,cudaMemcpyHostToDevice);free(w);}

    DevW4f16 lm_head_w;
    {char p2[256]; snprintf(p2,256,"%s/lm_head",wdir); lm_head_w=upload_w4_f16sc(p2);}
    uint8_t* host_embed_d=new uint8_t[(size_t)H*V/2];
    float* host_embed_sc=new float[V*(H/16)];
    {char p2[256]; snprintf(p2,256,"%s/embed_tokens",wdir);
    char p3[256]; snprintf(p3,256,"%s.int4_t",p2);
    FILE*f=fopen(p3,"rb");int h[5];fread(h,4,5,f);
     fread(host_embed_d,1,(size_t)h[0]*h[1]/2,f);fclose(f);
     snprintf(p3,256,"%s.scale_t",p2);
     f=fopen(p3,"rb");fread(h,4,5,f);
     {__half* tmp=new __half[(size_t)h[3]*h[4]];fread(tmp,2,(size_t)h[3]*h[4],f);
      for(size_t i=0;i<(size_t)h[3]*h[4];++i) host_embed_sc[i]=__half2float(tmp[i]);
      delete[] tmp;}
     fclose(f);}

    cudaStream_t st; die(cudaStreamCreate(&st),"stream");
    std::vector<float> h_embed(H);

    // Decode loop
    int tid=ids[0];
    dequant_embed_row(h_embed.data(),tid,host_embed_d,host_embed_sc,H);
    die(cudaMemcpyAsync(d_x32,h_embed.data(),H*4,cudaMemcpyHostToDevice,st),"embed_cpy");
    cudaMemset(d_kc,0,(size_t)NL*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vc,0,(size_t)NL*nkv*MAXSEQ*hd*4);

    auto t0=Clock::now();
    for(int step=0;step<max_new;++step){
        for(int l=0;l<NL;++l){
            size_t kv_off=(size_t)l*nkv*MAXSEQ*hd;
            die(cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st),"res_cpy");
            blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_in,H,eps,st);
            blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st);
            blackwell::kernels::gemv_int4_warp_f16wsc(d_Q,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].q.d,W[l].q.sc16,H,Q,st);
            blackwell::kernels::gemv_int4_warp_f16wsc(d_K,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].k.d,W[l].k.sc16,H,KV,st);
            blackwell::kernels::gemv_int4_warp_f16wsc(d_V,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].v.d,W[l].v.sc16,H,KV,st);
            head_norm_kernel<<<nqh,128,0,st>>>(d_Q,W[l].qnorm,nqh,hd,eps);
            head_norm_kernel<<<nkv,128,0,st>>>(d_K,W[l].knorm,nkv,hd,eps);
            apply_rope_kernel<<<nqh,hd/2,0,st>>>(d_Q,nqh,hd,step,rope_theta);
            apply_rope_kernel<<<nkv,hd/2,0,st>>>(d_K,nkv,hd,step,rope_theta);
            blackwell::kernels::update_kv_cache(d_kc+kv_off,d_vc+kv_off,d_K,d_V,0,step,nkv,hd,MAXSEQ,st);
            blackwell::kernels::attention_decode_gqa(d_attn,d_Q,d_kc+kv_off,d_vc+kv_off,step,nqh,nkv,hd,MAXSEQ,st);
            blackwell::kernels::quantize_int4(d_attn_i4,d_attn_i4_sc,d_attn,Q,st);
            blackwell::kernels::gemv_int4_warp_f16wsc(d_proj,(const uint8_t*)d_attn_i4,d_attn_i4_sc,W[l].o.d,W[l].o.sc16,Q,H,st);
            blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,st);
            die(cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st),"res_cpy2");
            blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_post,H,eps,st);
            blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st);
            blackwell::kernels::gemv_int4_warp_f16wsc(d_gate,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].g.d,W[l].g.sc16,H,I,st);
            blackwell::kernels::gemv_int4_warp_f16wsc(d_up,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].u.d,W[l].u.sc16,H,I,st);
            blackwell::kernels::apply_swiglu(d_gate,d_gate,d_up,I,st);
            blackwell::kernels::quantize_int4(d_mlp_i4,d_mlp_i4_sc,d_gate,I,st);
            blackwell::kernels::gemv_int4_warp_f16wsc(d_proj,(const uint8_t*)d_mlp_i4,d_mlp_i4_sc,W[l].d.d,W[l].d.sc16,I,H,st);
            blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,st);
        }
        blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,d_fn,H,eps,st);
        blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st);
        blackwell::kernels::gemv_int4_warp_f16wsc(d_logits,(const uint8_t*)d_x_i4,d_x_i4_sc,lm_head_w.d,lm_head_w.sc16,H,V,st);
        blackwell::kernels::sample_gpu(d_logits,V,0,0,d_next_id,0xdeadbeefLL,step,st);
        die(cudaStreamSynchronize(st),"sync");
        int next_id; cudaMemcpy(&next_id,d_next_id,4,cudaMemcpyDeviceToHost);
        if(step==0) fprintf(stderr,"  first decoded token: %d\n",next_id);
        if(step<max_new-1){
            dequant_embed_row(h_embed.data(),next_id,host_embed_d,host_embed_sc,H);
            die(cudaMemcpyAsync(d_x32,h_embed.data(),H*4,cudaMemcpyHostToDevice,st),"embed_cpy");
        }
        std::string txt=tok.decode(next_id);
        fprintf(stderr,"%s",txt.c_str()); fflush(stderr);
    }
    auto t1=Clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    fprintf(stderr,"\nStats: %d tokens, %.1f ms, %.1f ms/tok = %.0f t/s\n",
        max_new,ms,ms/max_new,1000.0/(ms/max_new));
    return 0;
}
