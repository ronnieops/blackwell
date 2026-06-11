// bench/text_generate_llama32_1b_fp16.cu — End-to-end text generation with FP16 Llama 3.2 1B
//
// Tokenize prompt → FP16 embedding lookup → 16L FP16 decode
// → final norm → FP16 lm_head GEMV → GPU sampling → print tokens.
//
// Uses --fp16 GGUF conversion output for lossless quality.
//
// Build:
//   cmake --build build --target text_generate_llama32_1b_fp16

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cstring>
#include <string>
#include <cstdint>
#include <cmath>
#include <algorithm>
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
        int exp=(h>>10)&0x1F;
        uint32_t man=h&0x3FF;
        if(exp==0) { fp[i]=0; }
        else if(exp==31) { fp[i]=(man?1e18f:1e18f); }
        else { exp-=15; if(exp<-14) exp=-14; fp[i]=ldexpf((1.0f+man/1024.0f),exp); if(h&0x8000) fp[i]=-fp[i]; }
    }
    DevFP32 dw; dw.K=K; dw.N=N;
    cudaMalloc(&dw.d,nel*4); cudaMemcpy(dw.d,fp.data(),nel*4,cudaMemcpyHostToDevice);
    return dw;
}

// FP32 weight layer struct
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

// FP16 embedding lookup: dequant FP16 row to FP32
static void dequant_fp16_row(float* out, int token, const float* host_w, int K) {
    const float* row = host_w + (size_t)token * K;
    for(int i=0;i<K;i++) out[i] = row[i];
}

int main(int argc, char** argv) {
    const char* wdir="/mnt/data/ai/models/llama32-1b-int4";
    const char* prompt="Hello";
    int max_new=30; float temp=0.0f; int top_k=0; float rep_pen=1.5f;
    bool chat_mode=false;
    
    for(int i=1;i<argc;i++) {
        if(argv[i][0]=='-' && argv[i][1]=='w' && i+1<argc) wdir=argv[++i];
        else if(argv[i][0]=='-' && argv[i][1]=='p' && i+1<argc) prompt=argv[++i];
        else if(argv[i][0]=='-' && argv[i][1]=='n' && i+1<argc) max_new=atoi(argv[++i]);
        else if(argv[i][0]=='-' && argv[i][1]=='t' && i+1<argc) temp=(float)atof(argv[++i]);
        else if(argv[i][0]=='-' && argv[i][1]=='k' && i+1<argc) top_k=atoi(argv[++i]);
        else if(argv[i][0]=='-' && argv[i][1]=='r' && i+1<argc) rep_pen=(float)atof(argv[++i]);
        else if(argv[i][0]=='-' && argv[i][1]=='c') chat_mode=true;
    }
    
    printf("# Text Generation — Llama 3.2 1B FP16 (GGUF --fp16)\n");
    printf("  Weights: %s\n",wdir);
    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    printf("  Device: %s\n",P.name);
    printf("  Prompt: \"%s\"\n",prompt);
    printf("  Temp: %.1f, Top-K: %d, Rep-pen: %.2f, Max new: %d\n\n",temp,top_k,rep_pen,max_new);
    
    // Tokenize
    std::vector<uint32_t> input_ids;
    blackwell::BpeTokenizer tok;
    char tok_path[512]; snprintf(tok_path,512,"%s/tokenizer_data.bin",wdir);
    if(tok.load(tok_path)!=0){ fprintf(stderr,"FAIL: no %s\n",tok_path); return 1; }
    input_ids = tok.encode(prompt);
    printf("Input: %zu tokens\n",input_ids.size());
    
    // Device buffers
    #define AL(p,n){cudaError_t _e=cudaMalloc(&(p),(n));if(_e!=cudaSuccess){printf("FAIL malloc %s: %s\n",#p,cudaGetErrorString(_e));die(_e,#p);}}
    float *d_x32, *d_Q, *d_K, *d_V, *d_attn, *d_proj;
    float *d_gate, *d_up, *d_mlp_out, *d_res;
    float *d_kc, *d_vc, *d_logits, *d_recent;
    int *d_next_id;
    AL(d_x32,H*4); AL(d_Q,Q*4); AL(d_K,KV*4); AL(d_V,KV*4);
    AL(d_attn,Q*4); AL(d_proj,H*4);
    AL(d_gate,I*4); AL(d_up,I*4); AL(d_mlp_out,H*4); AL(d_res,H*4);
    AL(d_kc,(size_t)NL*nkv*MAXSEQ*hd*4);
    AL(d_vc,(size_t)NL*nkv*MAXSEQ*hd*4);
    AL(d_logits,V*4); AL(d_next_id,sizeof(int)); AL(d_recent,64*4);
    #undef AL
    
    int dummy=0; cudaMemcpy(d_next_id,&dummy,4,cudaMemcpyHostToDevice);
    
    printf("Loading %d-layer FP16 model...\n",NL); fflush(stdout);
    std::vector<LF32> W(NL); char p_[512];
    for(int l=0;l<NL;++l){
        snprintf(p_,512,"%s/%d_self_attn.q_proj",wdir,l);W[l].q=upload_fp32(p_);
        snprintf(p_,512,"%s/%d_self_attn.k_proj",wdir,l);W[l].k=upload_fp32(p_);
        snprintf(p_,512,"%s/%d_self_attn.v_proj",wdir,l);W[l].v=upload_fp32(p_);
        snprintf(p_,512,"%s/%d_self_attn.o_proj",wdir,l);W[l].o=upload_fp32(p_);
        snprintf(p_,512,"%s/%d_mlp.gate_proj",wdir,l);W[l].g=upload_fp32(p_);
        snprintf(p_,512,"%s/%d_mlp.up_proj",wdir,l);W[l].u=upload_fp32(p_);
        snprintf(p_,512,"%s/%d_mlp.down_proj",wdir,l);W[l].d=upload_fp32(p_);
        if((l+1)%7==0||l+1==NL)printf("  layer %d/%d\n",l+1,NL);
    }
    
    // QK norms
    float* qk_h=(float*)malloc(NL*2*hd*4);
    char qkp[512]; snprintf(qkp,512,"%s/qk_norms.f32",wdir);
    {FILE*f=fopen(qkp,"rb");(void)fread(qk_h,4,NL*2*hd,f);fclose(f);}
    for(int l=0;l<NL;++l){
        cudaMalloc(&W[l].qn,hd*4); cudaMemcpy(W[l].qn,qk_h+l*2*hd,hd*4,cudaMemcpyHostToDevice);
        cudaMalloc(&W[l].kn,hd*4); cudaMemcpy(W[l].kn,qk_h+l*2*hd+hd,hd*4,cudaMemcpyHostToDevice);
    }free(qk_h);
    
    // Norm weights
    for(int l=0;l<NL;++l){
        float* w=(float*)malloc(H*4);
        snprintf(p_,512,"%s/%d_input_layernorm.f32",wdir,l);
        {FILE*f=fopen(p_,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&W[l].rn_in,H*4); cudaMemcpy(W[l].rn_in,w,H*4,cudaMemcpyHostToDevice);
        snprintf(p_,512,"%s/%d_post_attention_layernorm.f32",wdir,l);
        {FILE*f=fopen(p_,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&W[l].rn_post,H*4); cudaMemcpy(W[l].rn_post,w,H*4,cudaMemcpyHostToDevice);
        free(w);
    }
    
    // Final norm
    float* fn=(float*)malloc(H*4);
    char fp[512]; snprintf(fp,512,"%s/final_norm.f32",wdir);
    FILE*f=fopen(fp,"rb");(void)fread(fn,4,H,f);fclose(f);
    float* d_fn; cudaMalloc(&d_fn,H*4); cudaMemcpy(d_fn,fn,H*4,cudaMemcpyHostToDevice); free(fn);
    
    // Embeddings (FP16)
    snprintf(p_,512,"%s/embed_tokens.fp16",wdir);
    FILE* femb=fopen(p_,"rb"); int eh[2]; fread(eh,4,2,femb);
    int embed_K=eh[0], embed_N=eh[1];
    size_t embed_nel=(size_t)embed_N*embed_K;
    std::vector<uint16_t> embed_hp(embed_nel);
    fread(embed_hp.data(),2,embed_nel,femb); fclose(femb);
    std::vector<float> embed_fp(embed_nel);
    for(size_t i=0;i<embed_nel;i++) {
        uint16_t h=embed_hp[i];
        int exp=(h>>10)&0x1F;
        uint32_t man=h&0x3FF;
        if(exp==0) embed_fp[i]=0;
        else if(exp==31) embed_fp[i]=(man?1e18f:1e18f);
        else { exp-=15; if(exp<-14) exp=-14; embed_fp[i]=ldexpf((1.0f+man/1024.0f),exp); if(h&0x8000) embed_fp[i]=-embed_fp[i]; }
    }
    float* host_embed=new float[embed_nel];
    memcpy(host_embed,embed_fp.data(),embed_nel*4);
    float* d_embed; cudaMalloc(&d_embed,embed_nel*4); cudaMemcpy(d_embed,host_embed,embed_nel*4,cudaMemcpyHostToDevice);
    DevFP32 embed_fp32; embed_fp32.K=embed_K; embed_fp32.N=embed_N; embed_fp32.d=d_embed;
    printf("Embed tokens loaded: %d x %d (FP16)\n",embed_K,embed_N);

    // LM head (tied to embed)
    printf("lm_head: tied to embed\n");
    
    cudaStream_t st; die(cudaStreamCreate(&st),"stream");
    srand((unsigned)time(nullptr));
    unsigned long long rng_seed=rand();
    
    std::vector<float> h_embed(H);
    cudaMemset(d_kc,0,(size_t)NL*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vc,0,(size_t)NL*nkv*MAXSEQ*hd*4);
    
    printf("\n── Generating ──\n");
    if(chat_mode) printf("[assistant] ");
    else printf("%s",prompt);
    fflush(stdout);
    
    std::vector<uint32_t> all_ids=input_ids;
    int gen_start=(int)input_ids.size();
    int total=gen_start+max_new;
    auto t_start=Clock::now();
    
    for(int step=0;step<total;++step){
        uint32_t tid=(step<gen_start)?input_ids[step]:all_ids.back();
        
        // FP16 Embedding: copy row from host
        dequant_fp16_row(h_embed.data(),tid,host_embed,H);
        die(cudaMemcpyAsync(d_x32,h_embed.data(),H*4,cudaMemcpyHostToDevice,st),"embed_cpy");
        
        // 16-layer decode
        for(int l=0;l<NL;++l){
            // Save residual
            die(cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st),"save_res");
            
            // Pre-attention norm
            die(blackwell::kernels::fused_rmsnorm(d_x32,d_x32,W[l].rn_in,H,eps,st),"rmsnorm_in");
            
            // QKV projections (FP32 GEMV)
            die(blackwell::kernels::gemv_fp32_launch(d_Q,W[l].q.d,d_x32,H,Q,st),"q_proj");
            die(blackwell::kernels::gemv_fp32_launch(d_K,W[l].k.d,d_x32,H,KV,st),"k_proj");
            die(blackwell::kernels::gemv_fp32_launch(d_V,W[l].v.d,d_x32,H,KV,st),"v_proj");
            
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
            die(blackwell::kernels::gemv_fp32_launch(d_proj,W[l].o.d,d_attn,Q,H,st),"o_proj");
            
            // Attention residual
            die(blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,st),"attn_res");
            
            // Save pre-MLP state
            die(cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st),"save_res2");
            
            // Post-attention norm
            die(blackwell::kernels::fused_rmsnorm(d_x32,d_x32,W[l].rn_post,H,eps,st),"rmsnorm_post");
            
            // SwiGLU: gate=gemv(gate_proj,x), up=gemv(up_proj,x), silu(gate)*up
            die(blackwell::kernels::gemv_fp32_launch(d_gate,W[l].g.d,d_x32,H,I,st),"gate_proj");
            die(blackwell::kernels::gemv_fp32_launch(d_up,W[l].u.d,d_x32,H,I,st),"up_proj");
            die(blackwell::kernels::apply_swiglu(d_gate,d_gate,d_up,I,st),"swiglu");
            
            // Down projection
            die(blackwell::kernels::gemv_fp32_launch(d_mlp_out,W[l].d.d,d_gate,I,H,st),"down_proj");
            
            // MLP residual
            die(blackwell::kernels::vector_add_fp32(d_x32,d_mlp_out,d_res,H,st),"mlp_res");
        }
        
        // Final norm + lm_head
        die(blackwell::kernels::fused_rmsnorm(d_logits,d_x32,d_fn,H,eps,st),"final_norm");
        // LM head: tied to embed
        die(blackwell::kernels::gemv_fp32_launch(d_logits,embed_fp32.d,d_logits,H,V,st),"lm_head");
        
        // Repetition penalty
        if(rep_pen!=1.0f){
            std::vector<int> recent(all_ids.end()-std::min((int)all_ids.size(),32),all_ids.end());
            die(blackwell::kernels::apply_repetition_penalty(d_logits,recent.data(),(int)recent.size(),rep_pen,V,st),"rep_pen");
        }
        
        // Sampling
        die(blackwell::kernels::sample_gpu(d_logits,V,temp,top_k,d_next_id,rng_seed,step,st),"sample");
        int next_id; cudaMemcpy(&next_id,d_next_id,4,cudaMemcpyDeviceToHost);
        
        if(step>=gen_start){
            all_ids.push_back((uint32_t)next_id);
            if(next_id==2||next_id==3) break;  // EOS
            
            // Print token
            std::string txt=tok.decode(next_id);
            printf("%s",txt.c_str()); fflush(stdout);
        }
    }
    
    auto t_end=Clock::now();
    auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(t_end-t_start).count();
    int gen=((int)all_ids.size()-gen_start);
    printf("\n\n── Stats ──\n  Input: %zu  Gen: %d\n  Time: %d ms  Speed: %.1f ms/tok = %.0f t/s\n",
        input_ids.size(), gen, (int)ms, (float)ms/gen, 1000.0f*gen/ms);
    
    return 0;
}