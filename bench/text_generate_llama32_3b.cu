#include "blackwell/int4_weights.h"
using namespace blackwell::weights;
// bench/text_generate_llama32_3b.cu — Llama 3.2 3B INT4 text generation
// 28 layers, H=3072, I=8192, nqh=24, nkv=8, hd=128, V=128256, rope_theta=500000
//
// Build:
//   CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build && cmake --build build --parallel

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <string>
#include <cstdint>
#include <cmath>
#include <chrono>
#include "blackwell/kernels.h"
#include "blackwell/bpe_tokenizer.h"
using blackwell::BpeTokenizer;

static void die(cudaError_t e, const char* m) {
    if(e!=cudaSuccess){fprintf(stderr,"FAIL %s: %s\n",m,cudaGetErrorString(e));exit(1);}
}

const int H=3072, Q=3072, KV=1024, I=8192;
const int nqh=24, nkv=8, hd=128, MAXSEQ=4096;
const float eps=1e-6f;
const int V=128256;
const int NL=28;
const float rope_theta=500000.0f;

struct LW4 { DevW4f16 q,k,v,o,g,u,d; float* rn_in,*rn_post; };

static void build_rope_cache(float* cos_cache, float* sin_cache, int max_seq, int head_dim) {
    int pairs = head_dim / 2;
    for (int pos = 0; pos < max_seq; pos++)
        for (int d = 0; d < pairs; d++) {
            float theta = (float)pos * powf(rope_theta, -2.0f * (float)d / (float)head_dim);
            cos_cache[pos * pairs + d] = cosf(theta);
            sin_cache[pos * pairs + d] = sinf(theta);
        }
}

int main(int argc, char** argv) {
    const char* PROMPT = (argc > 1) ? argv[1] : "The capital of France is";
    int num_tokens = (argc > 2) ? atoi(argv[2]) : 30;
    const char* WDIR = (argc > 3) ? argv[3] : "weights_llama32_3b";
    
    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    fprintf(stderr, "# Llama 3.2 3B INT4 — %s\n  Weights: %s\n  Prompt: \"%s\"\n  Max new: %d\n",
            P.name, WDIR, PROMPT, num_tokens);

    BpeTokenizer tok;
    char tp[512]; snprintf(tp,512,"%s/tokenizer_data.bin",WDIR);
    if (tok.load(tp)!=0 && tok.load("tokenizer_data.bin")!=0) { fprintf(stderr,"FAIL: no tokenizer\n");return 1;}
    auto ids = tok.encode(PROMPT);
    if (ids.empty()) { fprintf(stderr,"FAIL: empty tokenization\n");return 1;}
    fprintf(stderr,"  Prompt tokens: %zu\n",ids.size());

    float *d_x32,*d_xi_f,*d_res,*d_Q,*d_K,*d_V,*d_attn,*d_proj,*d_gate,*d_up;
    uint8_t *d_x_i4; float *d_x_i4_sc;
    uint8_t *d_attn_i4; float *d_attn_i4_sc;
    uint8_t *d_mlp_i4; float *d_mlp_i4_sc;
    float *d_fn,*d_kc,*d_vc,*d_logits,*d_cos_cache,*d_sin_cache;
    int *d_next_id,*d_seq_pos;

    #define AL(p,n) die(cudaMalloc(&(p),(n)),"malloc")
    AL(d_x32,H*4); AL(d_xi_f,H*4); AL(d_res,H*4);
    AL(d_x_i4,H/2); AL(d_x_i4_sc,(H/16)*4);
    AL(d_Q,Q*4); AL(d_K,KV*4); AL(d_V,KV*4); AL(d_attn,Q*4);
    AL(d_attn_i4,Q/2); AL(d_attn_i4_sc,(Q/16)*4);
    AL(d_proj,H*4); AL(d_gate,I*4); AL(d_up,I*4);
    AL(d_mlp_i4,I/2); AL(d_mlp_i4_sc,(I/16)*4);
    AL(d_fn,H*4);
    AL(d_kc,(size_t)NL*nkv*MAXSEQ*hd*4);
    AL(d_vc,(size_t)NL*nkv*MAXSEQ*hd*4);
    AL(d_logits,V*4); AL(d_next_id,4); AL(d_seq_pos,4);
    int rope_pairs=hd/2;
    AL(d_cos_cache,(size_t)MAXSEQ*rope_pairs*4);
    AL(d_sin_cache,(size_t)MAXSEQ*rope_pairs*4);
    #undef AL

    float iv7=1.f/7.f;
    std::vector<float> tmp;
    tmp.assign(H/16,iv7); cudaMemcpy(d_x_i4_sc,tmp.data(),(H/16)*4,cudaMemcpyHostToDevice);
    tmp.assign(Q/16,iv7); cudaMemcpy(d_attn_i4_sc,tmp.data(),(Q/16)*4,cudaMemcpyHostToDevice);
    tmp.assign(I/16,iv7); cudaMemcpy(d_mlp_i4_sc,tmp.data(),(I/16)*4,cudaMemcpyHostToDevice);

    { // RoPE cache
        std::vector<float> cos_h(MAXSEQ*rope_pairs), sin_h(MAXSEQ*rope_pairs);
        build_rope_cache(cos_h.data(),sin_h.data(),MAXSEQ,hd);
        cudaMemcpy(d_cos_cache,cos_h.data(),(size_t)MAXSEQ*rope_pairs*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_sin_cache,sin_h.data(),(size_t)MAXSEQ*rope_pairs*4,cudaMemcpyHostToDevice);
    }

    fprintf(stderr,"Loading weights...\n");
    std::vector<LW4> W(NL);
    char p[256];
    for(int l=0;l<NL;++l){
        snprintf(p,256,"%s/%d_self_attn.q_proj",WDIR,l); W[l].q=upload_w4_f16sc(p);
        snprintf(p,256,"%s/%d_self_attn.k_proj",WDIR,l); W[l].k=upload_w4_f16sc(p);
        snprintf(p,256,"%s/%d_self_attn.v_proj",WDIR,l); W[l].v=upload_w4_f16sc(p);
        snprintf(p,256,"%s/%d_self_attn.o_proj",WDIR,l); W[l].o=upload_w4_f16sc(p);
        snprintf(p,256,"%s/%d_mlp.gate_proj",WDIR,l); W[l].g=upload_w4_f16sc(p);
        snprintf(p,256,"%s/%d_mlp.up_proj",WDIR,l); W[l].u=upload_w4_f16sc(p);
        snprintf(p,256,"%s/%d_mlp.down_proj",WDIR,l); W[l].d=upload_w4_f16sc(p);
    }

    for(int l=0;l<NL;++l){
        float* w=(float*)malloc(H*4);
        snprintf(p,256,"%s/%d_input_layernorm.f32",WDIR,l);
        {FILE*f=fopen(p,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&W[l].rn_in,H*4);cudaMemcpy(W[l].rn_in,w,H*4,cudaMemcpyHostToDevice);
        snprintf(p,256,"%s/%d_post_attention_layernorm.f32",WDIR,l);
        {FILE*f=fopen(p,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&W[l].rn_post,H*4);cudaMemcpy(W[l].rn_post,w,H*4,cudaMemcpyHostToDevice);
        free(w);
    }
    {float*w=(float*)malloc(H*4);
    snprintf(p,256,"%s/final_norm.f32",WDIR);
    FILE*f=fopen(p,"rb");(void)fread(w,4,H,f);fclose(f);
    cudaMemcpy(d_fn,w,H*4,cudaMemcpyHostToDevice);free(w);}

    // lm_head tied with embed_tokens
    DevW4f16 lm_head_w;
    snprintf(p,256,"%s/embed_tokens",WDIR); lm_head_w=upload_w4_f16sc(p);
    fprintf(stderr,"lm_head loaded from embed_tokens (tied): %dx%d (INT4)\n",lm_head_w.N,lm_head_w.K);
    
    uint8_t* host_embed_d=new uint8_t[(size_t)H*V/2];
    float* host_embed_sc=new float[V*(H/16)];
    snprintf(p,256,"%s/embed_tokens.int4_t",WDIR);
    {FILE*f=fopen(p,"rb");int h[5];fread(h,4,5,f);
     fread(host_embed_d,1,(size_t)h[0]*h[1]/2,f);fclose(f);}
    snprintf(p,256,"%s/embed_tokens.scale_t",WDIR);
    {FILE*f=fopen(p,"rb");int h[5];fread(h,4,5,f);
     __half*tmp=new __half[(size_t)h[3]*h[4]];fread(tmp,2,(size_t)h[3]*h[4],f);fclose(f);
     for(size_t i=0;i<(size_t)h[3]*h[4];++i) host_embed_sc[i]=__half2float(tmp[i]);delete[] tmp;}
    fprintf(stderr,"Embed loaded: %dx%d (INT4)\n",H,V);

    cudaStream_t st; die(cudaStreamCreate(&st),"stream");
    std::vector<float> h_embed(H);
    std::vector<uint32_t> generated;

    cudaMemset(d_kc,0,(size_t)NL*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vc,0,(size_t)NL*nkv*MAXSEQ*hd*4);

    // ── Forward: process one token through all layers ──
    // Templated lambda to avoid code duplication
    auto forward_token = [&](int step) {
        cudaMemcpy(d_seq_pos, &step, 4, cudaMemcpyHostToDevice);
        for(int l=0;l<NL;++l){
            cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st);
            blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_in,H,eps,st);
            blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st);
            blackwell::kernels::gemv_int4_warp_f16wsc(d_Q,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].q.d,W[l].q.sc16,H,Q,st);
            blackwell::kernels::gemv_int4_warp_f16wsc(d_K,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].k.d,W[l].k.sc16,H,KV,st);
            blackwell::kernels::gemv_int4_warp_f16wsc(d_V,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].v.d,W[l].v.sc16,H,KV,st);
            blackwell::kernels::fused_rope_decode(d_Q,d_cos_cache,d_sin_cache,d_seq_pos,nqh,hd,MAXSEQ,st);
            blackwell::kernels::fused_rope_decode(d_K,d_cos_cache,d_sin_cache,d_seq_pos,nkv,hd,MAXSEQ,st);
            size_t kv_off=(size_t)l*nkv*MAXSEQ*hd;
            blackwell::kernels::update_kv_cache(d_kc+kv_off,d_vc+kv_off,d_K,d_V,0,step,nkv,hd,MAXSEQ,st);
            blackwell::kernels::attention_decode_batched_gqa(d_attn,d_Q,d_kc,d_vc,step,nqh,nkv,hd,MAXSEQ,1,
                (size_t)NL*nkv*MAXSEQ*hd,kv_off,st);
            blackwell::kernels::quantize_int4(d_attn_i4,d_attn_i4_sc,d_attn,Q,st);
            blackwell::kernels::gemv_int4_warp_f16wsc(d_proj,(const uint8_t*)d_attn_i4,d_attn_i4_sc,W[l].o.d,W[l].o.sc16,Q,H,st);
            blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,st);
            cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st);
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
    };

    // Prefill: process prompt tokens
    for(int step=0;step<(int)ids.size();++step){
        uint32_t tid=ids[step];
        dequant_embed_row(h_embed.data(),tid,host_embed_d,host_embed_sc,H);
        cudaMemcpyAsync(d_x32,h_embed.data(),H*4,cudaMemcpyHostToDevice,st);
        forward_token(step);
    }

    // Decode loop with repetition penalty
    float rep_pen = 1.5f;
    int *d_recent;
    cudaMalloc(&d_recent, 64 * 4);
    
    uint32_t next_id=ids.back();
    auto t0=std::chrono::high_resolution_clock::now();
    for(int step=ids.size();step<(int)(ids.size()+num_tokens);++step){
        uint32_t tid=next_id;
        dequant_embed_row(h_embed.data(),tid,host_embed_d,host_embed_sc,H);
        cudaMemcpyAsync(d_x32,h_embed.data(),H*4,cudaMemcpyHostToDevice,st);
        forward_token(step);

        // Apply repetition penalty to recent tokens
        if (rep_pen > 1.0f && (int)generated.size() > 0) {
            int num_recent = (int)generated.size();
            if (num_recent > 64) num_recent = 64;
            std::vector<int> h_rec(generated.end() - num_recent, generated.end());
            cudaMemcpy(d_recent, h_rec.data(), num_recent*4, cudaMemcpyHostToDevice);
            blackwell::kernels::apply_repetition_penalty(d_logits, d_recent, num_recent, rep_pen, V, st);
        }

        blackwell::kernels::sample_argmax_gpu(d_logits,V,d_next_id,st);
        cudaStreamSynchronize(st);
        // Debug: print first token logit values
        if (step == (int)ids.size()) {
            float l0, l1, l2;
            cudaMemcpy(&l0, d_logits, 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(&l1, d_logits+1, 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(&l2, d_logits+2, 4, cudaMemcpyDeviceToHost);
            fprintf(stderr, "  decode step0: logits[0]=%.1f logits[1]=%.1f logits[2]=%.1f\n", l0, l1, l2);
        }
        cudaMemcpy(&next_id,d_next_id,4,cudaMemcpyDeviceToHost);
        if (step == (int)ids.size()) fprintf(stderr, "  first decoded token: %d\n", next_id);
        if(next_id==128009||next_id==128001) break;
        generated.push_back(next_id);
    }
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();

    std::string out_text;
    for(auto id:generated) out_text+=tok.decode(id);
    fprintf(stderr,"\n[Output] %s\n",out_text.c_str());
    fprintf(stderr,"\nStats: %zu tokens, %.1f ms, %.1f ms/tok = %.0f t/s\n",
            generated.size(),ms,ms/generated.size(),1000.0/(ms/generated.size()));
    return 0;
}
