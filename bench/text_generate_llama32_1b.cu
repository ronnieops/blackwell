// bench/text_generate_llama32_1b.cu — End-to-end text generation with INT4 Llama 3.2 1B
//
// Tokenize prompt → INT4 embedding lookup → 16L INT4 decode
// → final norm → INT4 lm_head GEMV → GPU sampling → print tokens.
//
// Adapted from text_generate_int4_1.7b.cu (Qwen3-1.7B)
//
// Build:
//   cmake --build build --target text_generate_llama32_1b

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

static void die(cudaError_t e, const char* m) {
    if(e!=cudaSuccess){printf("FAIL %s: %s\n",m,cudaGetErrorString(e));exit(1);}
}

using Clock = std::chrono::high_resolution_clock;

// Llama 3.2 1B dimensions
const int H=2048, Q=2048, KV=512, I=8192;
const int nqh=32, nkv=8, hd=64, MAXSEQ=4096;
const float eps=1e-6f;
const int V=128256;
const int NL=16;
const float rope_theta=500000.0f;

struct DevW4 { int K, N; uint8_t* d; float* sc; };
static DevW4 upload_w4(const char* prefix) {
    char p[256]; snprintf(p,256,"%s.int4_t",prefix);
    FILE* f=fopen(p,"rb"); int h[5]; fread(h,4,5,f);
    DevW4 dw; dw.K=h[0]; dw.N=h[1];
    size_t ds=(size_t)h[0]*h[1]/2;
    uint8_t* td=new uint8_t[ds]; fread(td,1,ds,f); fclose(f);
    cudaMalloc(&dw.d,ds); cudaMemcpy(dw.d,td,ds,cudaMemcpyHostToDevice); delete[] td;
    snprintf(p,256,"%s.scale_t",prefix); f=fopen(p,"rb"); fread(h,4,5,f);
    size_t ss=(size_t)h[3]*h[4];  // Use scale_t header for scale count
    float* ts=new float[ss]; fread(ts,4,ss,f); fclose(f);
    cudaMalloc(&dw.sc,ss*4); cudaMemcpy(dw.sc,ts,ss*4,cudaMemcpyHostToDevice); delete[] ts;
    return dw;
}

// FP32 weight struct (loaded from .fp16 files)
struct DevFP32 { int K, N; float* d; };
static DevFP32 upload_fp32(const char* prefix) {
    char p[256]; snprintf(p,256,"%s.fp16",prefix);
    FILE* f=fopen(p,"rb"); int hdr[2]; fread(hdr,4,2,f);
    int K=hdr[0], N=hdr[1];
    size_t nel=(size_t)N*K;
    std::vector<uint16_t> hp(nel);
    fread(hp.data(),2,nel,f); fclose(f);
    std::vector<float> fp(nel);
    for(size_t i=0;i<nel;i++) {
        uint16_t h=hp[i];
        uint32_t b=(h&0x8000)<<16; int exp=(h>>10)&0x1F;
        uint32_t man=h&0x3FF;
        if(exp==0) { fp[i]=0; }
        else if(exp==31) { fp[i]=(h&0x3FF)?1e18f:1e18f; }
        else { exp-=15; if(exp<-14) exp=-14; fp[i]=ldexpf((1.0f+man/1024.0f),exp); if(h&0x8000) fp[i]=-fp[i]; }
    }
    DevFP32 dw; dw.K=K; dw.N=N;
    cudaMalloc(&dw.d,nel*4); cudaMemcpy(dw.d,fp.data(),nel*4,cudaMemcpyHostToDevice);
    return dw;
}

struct LW4 {
    DevW4 q,k,v,o,g,u,d;
    float* qn; float* kn; float* rn_in; float* rn_post;
};

// FP32 weight layer struct (for --fp16 mode)
struct LF32 {
    DevFP32 q,k,v,o,g,u,d;
    float* qn; float* kn; float* rn_in; float* rn_post;
};

__global__ void head_norm_kernel(float* data, const float* weight, int nh, int hd, float eps) {
    int h=blockIdx.x; if(h>=nh) return;
    float* d=data+h*hd;
    __shared__ float wp[4]; float s=0;
    for(int i=threadIdx.x;i<hd;i+=blockDim.x) s+=d[i]*d[i];
    for(int off=16;off>0;off>>=1) s+=__shfl_xor_sync(0xffffffff,s,off);
    if((threadIdx.x&31)==0) wp[threadIdx.x>>5]=s; __syncthreads();
    if(threadIdx.x<4) s=wp[threadIdx.x]; else s=0;
    for(int off=2;off>0;off>>=1) s+=__shfl_xor_sync(0xffffffff,s,off);
    if(threadIdx.x==0) wp[0]=rsqrtf(s/hd+eps); __syncthreads();
    float is=wp[0];
    for(int i=threadIdx.x;i<hd;i+=blockDim.x) d[i]=d[i]*is*weight[i];
}

__global__ void apply_rope_kernel(float* data, int n_heads, int head_dim, int pos) {
    int h=blockIdx.x; int d=threadIdx.x;
    if(h>=n_heads||d>=head_dim/2) return;
    float* pair=data+h*head_dim+d*2;
    float theta=(float)pos*powf(rope_theta,-2.0f*(float)d/(float)head_dim);
    float c=cosf(theta),s=sinf(theta),x=pair[0],y=pair[1];
    pair[0]=x*c-y*s; pair[1]=x*s+y*c;
}

static void dequant_embed_row(float* out, int token, const uint8_t* host_w,
    const float* host_sc, int K)
{
    int kblocks=K/16;
    for(int b=0;b<kblocks;++b){
        float sc=host_sc[token*kblocks+b];
        for(int i=0;i<16;++i){
            size_t byte_idx=(size_t)token*K/2+(size_t)b*8+i/2;
            uint8_t byte=host_w[byte_idx];
            int nib=(i&1)?((byte>>4)&0x0F):(byte&0x0F);
            int val=nib-8;
            out[b*16+i]=(float)val*sc;
        }
    }
}

int main(int argc, char** argv) {
    const char* prompt = "Once upon a time";
    int max_new = 50;
    bool chat_mode = false;
    float temperature = 0.0f;
    int top_k = 0;
    float rep_pen = 1.5f;
    const char* wdir = "/mnt/data/ai/models/llama32-1b-int4";
    for(int i=1;i<argc;i++){
        if(argv[i][0]=='-'){
            if(strcmp(argv[i],"--chat")==0) chat_mode=true;
            else if(strcmp(argv[i],"-t")==0&&i+1<argc) temperature=atof(argv[++i]);
            else if(strcmp(argv[i],"-k")==0&&i+1<argc) top_k=atoi(argv[++i]);
            else if(strcmp(argv[i],"-r")==0&&i+1<argc) rep_pen=atof(argv[++i]);
            else if(strcmp(argv[i],"-w")==0&&i+1<argc) wdir=argv[++i];
        } else {
            prompt=argv[i];
            if(i+1<argc&&argv[i+1][0]!='-'){ max_new=atoi(argv[++i]); }
        }
    }
    
    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    printf("# Text Generation — Qwen3-1.7B INT4 (GGUF converted)\n");
    printf("  Weights: %s\n", wdir);
    printf("  Device: %s\n", P.name);
    printf("  Prompt: \"%s\"%s\n", prompt, chat_mode?" (chat)":"");
    printf("  Temp: %.1f, Top-K: %d, Rep-pen: %.2f, Max new: %d\n\n", temperature, top_k, rep_pen, max_new);

    blackwell::BpeTokenizer tokenizer;
    char tok_path[512]; snprintf(tok_path,512,"%s/tokenizer_data.bin",wdir);
    if(tokenizer.load(tok_path)!=0){
        fprintf(stderr,"FAIL: no %s\n",tok_path);return 1;
    }

    std::vector<uint32_t> input_ids;
    if(chat_mode){
        input_ids.push_back(151644);
        for(char c:std::string("user\n")) input_ids.push_back((uint32_t)(unsigned char)c);
        auto pt=tokenizer.encode(prompt);
        input_ids.insert(input_ids.end(),pt.begin(),pt.end());
        input_ids.push_back(151645);
        input_ids.push_back(151644);
        for(char c:std::string("assistant\n")) input_ids.push_back((uint32_t)(unsigned char)c);
    }else{
        input_ids=tokenizer.encode(prompt);
    }
    printf("Input: %zu tokens\n", input_ids.size());

    // Device buffers
    float *d_x32, *d_xi_f, *d_res;
    uint8_t *d_x_i4; float *d_x_i4_sc;
    float *d_Q,*d_K,*d_V,*d_attn;
    uint8_t *d_attn_i4; float *d_attn_i4_sc;
    float *d_proj, *d_gate, *d_up;
    uint8_t *d_mlp_i4; float *d_mlp_i4_sc;
    float *d_fn, *d_fn_sc, *d_kc, *d_vc, *d_logits;
    int *d_next_id, *d_recent;

    // const int NL=28; // Qwen3-1.7B — now using global NL=16 for Llama 3.2 1B
    #define AL(p,n){cudaError_t _e=cudaMalloc(&(p),(n));\
        if(_e!=cudaSuccess){printf("FAIL malloc %s: %s\n",#p,cudaGetErrorString(_e));die(_e,#p);}}
    AL(d_x32,H*4);AL(d_xi_f,H*4);AL(d_res,H*4);
    AL(d_x_i4,H/2);AL(d_x_i4_sc,(H/16)*4);
    AL(d_Q,Q*4);AL(d_K,KV*4);AL(d_V,KV*4);AL(d_attn,Q*4);
    AL(d_attn_i4,Q/2);AL(d_attn_i4_sc,(Q/16)*4);
    AL(d_proj,H*4);AL(d_gate,I*4);AL(d_up,I*4);
    AL(d_mlp_i4,I/2);AL(d_mlp_i4_sc,(I/16)*4);
    AL(d_fn,H*4);AL(d_fn_sc,(H/16)*4);
    AL(d_kc,(size_t)NL*nkv*MAXSEQ*hd*4);
    AL(d_vc,(size_t)NL*nkv*MAXSEQ*hd*4);
    AL(d_logits,V*4);AL(d_next_id,4);AL(d_recent,64*4);
    #undef AL

    float iv7=1.f/7.f;
    // Initialize ALL activation scales to 1/7 (not just first element)
    { std::vector<float> tmp(H/16, iv7); cudaMemcpy(d_x_i4_sc, tmp.data(), (H/16)*4, cudaMemcpyHostToDevice); }
    { std::vector<float> tmp(Q/16, iv7); cudaMemcpy(d_attn_i4_sc, tmp.data(), (Q/16)*4, cudaMemcpyHostToDevice); }
    { std::vector<float> tmp(I/16, iv7); cudaMemcpy(d_mlp_i4_sc, tmp.data(), (I/16)*4, cudaMemcpyHostToDevice); }
    { std::vector<float> tmp(H/16, iv7); cudaMemcpy(d_fn_sc, tmp.data(), (H/16)*4, cudaMemcpyHostToDevice); }
    int dummy=0;cudaMemcpy(d_next_id,&dummy,4,cudaMemcpyHostToDevice);

    printf("Loading %d-layer INT4 model...\n",NL);fflush(stdout);
    std::vector<LW4> W(NL); char p_[512];
    for(int l=0;l<NL;++l){
        snprintf(p_,512,"%s/%d_self_attn.q_proj",wdir,l);W[l].q=upload_w4(p_);
        snprintf(p_,512,"%s/%d_self_attn.k_proj",wdir,l);W[l].k=upload_w4(p_);
        snprintf(p_,512,"%s/%d_self_attn.v_proj",wdir,l);W[l].v=upload_w4(p_);
        snprintf(p_,512,"%s/%d_self_attn.o_proj",wdir,l);W[l].o=upload_w4(p_);
        snprintf(p_,512,"%s/%d_mlp.gate_proj",wdir,l);W[l].g=upload_w4(p_);
        snprintf(p_,512,"%s/%d_mlp.up_proj",wdir,l);W[l].u=upload_w4(p_);
        snprintf(p_,512,"%s/%d_mlp.down_proj",wdir,l);W[l].d=upload_w4(p_);
        if((l+1)%7==0||l+1==NL)printf("  layer %d/%d\n",l+1,NL);
    }

    float* qk_h=(float*)malloc(NL*2*hd*4);
    char qkp[512];snprintf(qkp,512,"%s/qk_norms.f32",wdir);
    {FILE*f=fopen(qkp,"rb");(void)fread(qk_h,4,NL*2*hd,f);fclose(f);}
    for(int l=0;l<NL;++l){
        cudaMalloc(&W[l].qn,hd*4);cudaMemcpy(W[l].qn,qk_h+l*2*hd,hd*4,cudaMemcpyHostToDevice);
        cudaMalloc(&W[l].kn,hd*4);cudaMemcpy(W[l].kn,qk_h+l*2*hd+hd,hd*4,cudaMemcpyHostToDevice);
    }free(qk_h);

    for(int l=0;l<NL;++l){
        float* w=(float*)malloc(H*4);
        snprintf(p_,512,"%s/%d_input_layernorm.f32",wdir,l);
        {FILE*f=fopen(p_,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&W[l].rn_in,H*4);cudaMemcpy(W[l].rn_in,w,H*4,cudaMemcpyHostToDevice);
        snprintf(p_,512,"%s/%d_post_attention_layernorm.f32",wdir,l);
        {FILE*f=fopen(p_,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&W[l].rn_post,H*4);cudaMemcpy(W[l].rn_post,w,H*4,cudaMemcpyHostToDevice);
        free(w);
    }

    {float*w=(float*)malloc(H*4);
    char fp[512]; snprintf(fp,512,"%s/final_norm.f32",wdir);
    FILE*f=fopen(fp,"rb");(void)fread(w,4,H,f);fclose(f);
    cudaMemcpy(d_fn,w,H*4,cudaMemcpyHostToDevice);free(w);}

    DevW4 embed=upload_w4((std::string(wdir)+"/embed_tokens").c_str());
    uint8_t* host_embed_d=new uint8_t[(size_t)embed.K*embed.N/2];
    float* host_embed_sc=new float[(size_t)embed.N*(embed.K/16)];
    {char p[512];
    snprintf(p,512,"%s/embed_tokens.int4_t",wdir);
    FILE*f=fopen(p,"rb");int h[5];fread(h,4,5,f);
    size_t ds=(size_t)h[0]*h[1]/2;fread(host_embed_d,1,ds,f);fclose(f);
    snprintf(p,512,"%s/embed_tokens.scale_t",wdir);
    f=fopen(p,"rb");fread(h,4,5,f);size_t ss=(size_t)h[3]*h[4];
    fread(host_embed_sc,4,ss,f);fclose(f);}
    printf("Embed tokens loaded: %d x %d (INT4)\n",embed.K,embed.N);
    // LM head (tied for 1.7B, check if separate file exists)
    char lm_p[512]; snprintf(lm_p, 512, "%s/lm_head", wdir);
    FILE*flm=fopen((std::string(lm_p)+".int4_t").c_str(),"rb");
    DevW4 lm_head_w;
    if(flm){fclose(flm);lm_head_w=upload_w4(lm_p);printf("lm_head loaded: %d x %d (INT4)\n",lm_head_w.K,lm_head_w.N);}
    else{lm_head_w=embed;printf("lm_head: tied to embed (INT4), embed.d=%p, embed.sc=%p\n",embed.d,embed.sc);}
    printf("lm_head loaded: %d x %d (INT4) lm_head_w.d=%p lm_head_w.sc=%p\n", lm_head_w.K, lm_head_w.N, lm_head_w.d, lm_head_w.sc);
    printf("All weights loaded.\n\n");

    cudaStream_t st;die(cudaStreamCreate(&st),"stream");
    srand((unsigned)time(nullptr));

    std::vector<float> h_embed(H);
    // Clear pending CUDA errors before generation
    cudaGetLastError();
    cudaMemset(d_kc,0,(size_t)NL*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vc,0,(size_t)NL*nkv*MAXSEQ*hd*4);

    printf("── Generating ──\n");
    if(chat_mode)printf("[assistant] ");
    else printf("%s",prompt);
    fflush(stdout);

    std::vector<uint32_t> all_ids=input_ids;
    int gen_start=(int)input_ids.size();
    int total=gen_start+max_new;
    auto t_start=Clock::now();

    for(int step=0;step<total;++step){
        uint32_t tid=(step<gen_start)?input_ids[step]:all_ids.back();

        // ── INT4 Embedding: host dequant single row → GPU ──
        dequant_embed_row(h_embed.data(),tid,host_embed_d,host_embed_sc,H);        die(cudaMemcpyAsync(d_x32,h_embed.data(),H*4,cudaMemcpyHostToDevice,st),"embed_cpy");
        // ══ 28-layer decode ══
        for(int l=0;l<NL;++l){

            // Save residual before norm
            die(cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st),"save_res");

            // Pre-attention norm: norm(d_x32, rn_in) → d_xi_f
            die(blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_in,H,eps,st),"rmsnorm_in");

            // Quantize normed input → INT4
            die(blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st),"quant_in");

            // QKV projections (batched M=1)
            cudaError_t _e1, _e2, _e3;
            _e1=blackwell::kernels::gemv_int4_warp(d_Q,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].q.d,W[l].q.sc,H,Q,st);
            cudaStreamSynchronize(st);
            _e2=blackwell::kernels::gemv_int4_warp(d_K,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].k.d,W[l].k.sc,H,KV,st);
            cudaStreamSynchronize(st);
            _e3=blackwell::kernels::gemv_int4_warp(d_V,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].v.d,W[l].v.sc,H,KV,st);
            die(_e1,"q_proj"); die(_e2,"k_proj"); die(_e3,"v_proj");

            // Q/K head norms + RoPE
            head_norm_kernel<<<nqh,128,0,st>>>(d_Q,W[l].qn,nqh,hd,eps);
            die(cudaStreamSynchronize(st),"head_norm_Q");
            head_norm_kernel<<<nkv,128,0,st>>>(d_K,W[l].kn,nkv,hd,eps);
            die(cudaStreamSynchronize(st),"head_norm_K");
            apply_rope_kernel<<<nqh,hd/2,0,st>>>(d_Q,nqh,hd,step);
            die(cudaGetLastError(),"rope_Q");
            apply_rope_kernel<<<nkv,hd/2,0,st>>>(d_K,nkv,hd,step);
            die(cudaGetLastError(),"rope_K");

            // KV cache + attention
            size_t kv_off=(size_t)l*nkv*MAXSEQ*hd;
            die(blackwell::kernels::update_kv_cache(d_kc+kv_off,d_vc+kv_off,d_K,d_V,0,step,nkv,hd,MAXSEQ,st),"kv");
            die(blackwell::kernels::attention_decode_batched_gqa(d_attn,d_Q,d_kc,d_vc,step,nqh,nkv,hd,MAXSEQ,1,
                (size_t)NL*nkv*MAXSEQ*hd,kv_off,st),"attn");

            // Wo projection
            die(blackwell::kernels::quantize_int4(d_attn_i4,d_attn_i4_sc,d_attn,Q,st),"quant_attn");
            die(blackwell::kernels::gemv_int4_warp(d_proj,(const uint8_t*)d_attn_i4,d_attn_i4_sc,W[l].o.d,W[l].o.sc,Q,H,st),"o_proj");

            // Attention residual: d_x32 = d_proj + d_res (original input)
            // Note: use vector_add(out, a, b) = out = a + b
            // We want d_x32 = d_proj + d_res (not d_x32 which is the pre-norm input)
            // But vector_add uses __restrict__ so out=b is UB. Let's use a temp buffer.
            // Actually, d_res is correct — it's the saved pre-norm input.
            // d_proj + d_res → d_x32 using explicit array copy to avoid restrict violation
            die(blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,st),"attn_res");

            // Save pre-MLP state for second residual
            die(cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st),"save_res2");

            // Pre-MLP norm
            die(blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_post,H,eps,st),"rmsnorm_post");
            die(blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st),"quant_mlp_in");

            // MLP gate + up
            die(blackwell::kernels::gemv_int4_warp(d_gate,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].g.d,W[l].g.sc,H,I,st),"gate");
            die(blackwell::kernels::gemv_int4_warp(d_up,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].u.d,W[l].u.sc,H,I,st),"up");

            // Fused SwiGLU + INT4 quant
            blackwell::kernels::apply_swiglu(d_gate, d_gate, d_up, I, st);
            blackwell::kernels::quantize_int4(d_mlp_i4, d_mlp_i4_sc, d_gate, I, st);

            // Down projection
            die(blackwell::kernels::gemv_int4_warp(d_proj,(const uint8_t*)d_mlp_i4,d_mlp_i4_sc,W[l].d.d,W[l].d.sc,I,H,st),"down");

            // MLP residual: d_x32 = d_proj + d_res (pre-MLP state)
            die(blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,st),"mlp_res");
            
        }

        // Final norm + lm_head + GPU sampling
        if(step>=gen_start-1){
            die(blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,d_fn,H,eps,st),"fn");

            die(blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st),"quant_lm");

            die(blackwell::kernels::gemv_int4_warp(d_logits,(const uint8_t*)d_x_i4,d_x_i4_sc,lm_head_w.d,lm_head_w.sc,H,V,st),"lm_head");
            float l0,l264,l37018;
            cudaMemcpy(&l0,d_logits,4,cudaMemcpyDeviceToHost);
            cudaMemcpy(&l264,d_logits+264,4,cudaMemcpyDeviceToHost);
            cudaMemcpy(&l37018,d_logits+37018,4,cudaMemcpyDeviceToHost);
            printf("  logits: tok0=%.1f tok264=%.1f tok37018=%.1f\n",l0,l264,l37018);
            // Repetition penalty: penalize recently generated tokens
            if(rep_pen>1.0f&&(int)all_ids.size()>gen_start){
                int nr=(int)all_ids.size()-gen_start;if(nr>64)nr=64;
                std::vector<int> h_rec(all_ids.end()-nr,all_ids.end());
                cudaMemcpy(d_recent,h_rec.data(),nr*4,cudaMemcpyHostToDevice);
                die(blackwell::kernels::apply_repetition_penalty(d_logits,d_recent,nr,rep_pen,V,st),"rep_pen");
            }
            int next_id;
            die(blackwell::kernels::sample_gpu(d_logits,V,temperature,top_k,d_next_id,0xdeadbeefLL,step,st),"sample");
            die(cudaMemcpy(&next_id,d_next_id,4,cudaMemcpyDeviceToHost),"copy");

            all_ids.push_back(next_id);
            std::string txt=tokenizer.decode(next_id);
            printf("%s",txt.c_str());fflush(stdout);

            if((int)all_ids.size()-gen_start<=3)
                printf(" [tok#%d=%d]",(int)all_ids.size()-gen_start,next_id);

            if(next_id==151643||next_id==151645){printf("\n[EOS]\n");break;}
        }
    }

    auto t_end=Clock::now();
    double ms=std::chrono::duration<double,std::milli>(t_end-t_start).count();
    int gen=(int)all_ids.size()-gen_start;
    printf("\n\n── Stats ──\n");
    printf("  Input: %d  Gen: %d\n",gen_start,gen);
    printf("  Time: %.1f ms  Speed: %.1f ms/tok = %.0f t/s\n",ms,ms/gen,1000.0*gen/ms);

    for(auto&w:W){
        cudaFree(w.q.d);cudaFree(w.q.sc);
        cudaFree(w.k.d);cudaFree(w.k.sc);
        cudaFree(w.v.d);cudaFree(w.v.sc);
        cudaFree(w.o.d);cudaFree(w.o.sc);
        cudaFree(w.g.d);cudaFree(w.g.sc);
        cudaFree(w.u.d);cudaFree(w.u.sc);
        cudaFree(w.d.d);cudaFree(w.d.sc);
        cudaFree(w.qn);cudaFree(w.kn);
        cudaFree(w.rn_in);cudaFree(w.rn_post);
    }
    delete[] host_embed_d;delete[] host_embed_sc;
    return 0;
}