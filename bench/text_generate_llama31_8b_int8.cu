#include "blackwell/int4_weights.h"
using namespace blackwell::weights;
// bench/text_generate_llama31_8b_int8.cu — End-to-end text generation with INT8 Llama 3.1 8B
//
// Uses FP32 activations × INT8 weights (gemv_fp32_int8_per_row_warp).
// No activation quantization — natural FP32 residual path for INT8.
//
// Llama 3.1 8B dimensions: NL=32, H=4096, I=14336, nqh=32, nkv=8, hd=128, V=128256
//
// Build:
//   cmake --build build --target text_generate_llama31_8b_int8

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

// Llama 3.1 8B dimensions
const int H=4096, Q=4096, KV=1024, I=14336;
const int nqh=32, nkv=8, hd=128, MAXSEQ=512;
const float eps=1e-6f;
const int V=128256;
const int NL=32;
const float rope_theta=500000.0f;

struct DevW8 { int K, N; int8_t* d; float* sc; };
static DevW8 upload_w8(const char* prefix) {
    char p[256]; snprintf(p,256,"%s.int8_t",prefix);
    FILE* f=fopen(p,"rb"); int h[5]; fread(h,4,5,f);
    DevW8 dw; dw.K=h[0]; dw.N=h[1];
    size_t ds=(size_t)h[0]*h[1];
    int8_t* td=new int8_t[ds]; fread(td,1,ds,f); fclose(f);
    cudaMalloc(&dw.d,ds); cudaMemcpy(dw.d,td,ds,cudaMemcpyHostToDevice); delete[] td;
    snprintf(p,256,"%s.scale_t",prefix); f=fopen(p,"rb"); fread(h,4,5,f);
    size_t ss=(size_t)h[3]*h[4];
    float* ts=new float[ss]; fread(ts,4,ss,f); fclose(f);
    cudaMalloc(&dw.sc,ss*4); cudaMemcpy(dw.sc,ts,ss*4,cudaMemcpyHostToDevice); delete[] ts;
    return dw;
}

struct LW8 {
    DevW8 q,k,v,o,g,u,d;
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

// Host-side embedding dequant for a single token row (INT8 embed table)
    const float* host_sc, int K)
{
    int kblocks=K/16;
    for(int b=0;b<kblocks;++b){
        float sc=host_sc[token*kblocks+b];
        for(int i=0;i<16;++i){
            size_t idx=(size_t)token*K+b*16+i;
            out[b*16+i]=(float)host_w[idx]*sc;
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
    const char* wdir = "/mnt/data/ai/models/llama31-8b-int8-from-safetensors";
    for(int i=1;i<argc;i++){
        if(argv[i][0]=='-'){
            if(strcmp(argv[i],"--chat")==0) chat_mode=true;
            else if(strcmp(argv[i],"-t")==0&&i+1<argc) temperature=atof(argv[++i]);
            else if(strcmp(argv[i],"-k")==0&&i+1<argc) top_k=atoi(argv[++i]);
            else if(strcmp(argv[i],"-r")==0&&i+1<argc) rep_pen=atof(argv[++i]);
            else if(strcmp(argv[i],"-w")==0&&i+1<argc) wdir=argv[++i];
        } else {
            if(i==1) prompt=argv[i];
            else if(i==2) max_new=atoi(argv[i]);
        }
    }

    blackwell::BpeTokenizer tokenizer;
    char tok_path[512]; snprintf(tok_path,512,"%s/tokenizer_data.bin",wdir);
    if(tokenizer.load(tok_path)!=0){printf("FAIL tokenizer\n");return 1;}
    printf("[tokenizer] Loaded\n");

    printf("# Text Generation — Llama 3.1 8B INT8\n");
    printf("  Weights: %s\n", wdir);
    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    printf("  Device: %s\n", P.name);
    printf("  Prompt: \"%s\"\n", prompt);
    printf("  Temp: %.1f, Top-K: %d, Rep-pen: %.2f, Max new: %d\n", temperature, top_k, rep_pen, max_new);

    auto input_ids = tokenizer.encode(prompt);
    if(chat_mode){
        std::string cp="<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"+std::string(prompt)+"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
        input_ids = tokenizer.encode(cp);
    }
    printf("\nInput: %zu tokens\n", input_ids.size());

    printf("Loading %d-layer INT8 model...\n", NL);
    cudaStream_t st; die(cudaStreamCreate(&st),"stream");

    int gen_start=(int)input_ids.size();
    LW8* W=new LW8[NL]();
    char p_[512];
    for(int l=0;l<NL;++l){
        snprintf(p_,512,"%s/%d_self_attn.q_proj",wdir,l);W[l].q=upload_w8(p_);
        snprintf(p_,512,"%s/%d_self_attn.k_proj",wdir,l);W[l].k=upload_w8(p_);
        snprintf(p_,512,"%s/%d_self_attn.v_proj",wdir,l);W[l].v=upload_w8(p_);
        snprintf(p_,512,"%s/%d_self_attn.o_proj",wdir,l);W[l].o=upload_w8(p_);
        snprintf(p_,512,"%s/%d_mlp.gate_proj",wdir,l);W[l].g=upload_w8(p_);
        snprintf(p_,512,"%s/%d_mlp.up_proj",wdir,l);W[l].u=upload_w8(p_);
        snprintf(p_,512,"%s/%d_mlp.down_proj",wdir,l);W[l].d=upload_w8(p_);
        if((l+1)%7==0) printf("  layer %d/%d\n",l+1,NL);
    }
    printf("  layer %d/%d\n",NL,NL);
    fflush(stdout);

    // Load QK norms and layernorms
    float* qk_h=new float[NL*2*hd];
    snprintf(p_,512,"%s/qk_norms.f32",wdir);
    {FILE*f=fopen(p_,"rb");(void)fread(qk_h,4,NL*2*hd,f);fclose(f);}
    for(int l=0;l<NL;++l){
        W[l].qn=qk_h+l*2*hd;
        W[l].kn=qk_h+l*2*hd+hd;
        snprintf(p_,512,"%s/%d_input_layernorm.f32",wdir,l);
        {float* w=new float[H];FILE*f=fopen(p_,"rb");(void)fread(w,4,H,f);fclose(f);
         cudaMalloc(&W[l].rn_in,H*4);cudaMemcpy(W[l].rn_in,w,H*4,cudaMemcpyHostToDevice);delete[] w;}
        snprintf(p_,512,"%s/%d_post_attention_layernorm.f32",wdir,l);
        {float* w=new float[H];FILE*f=fopen(p_,"rb");(void)fread(w,4,H,f);fclose(f);
         cudaMalloc(&W[l].rn_post,H*4);cudaMemcpy(W[l].rn_post,w,H*4,cudaMemcpyHostToDevice);delete[] w;}
    }

    // Final norm
    float* d_fn; {float* w=new float[H];
    snprintf(p_,512,"%s/final_norm.f32",wdir);
    FILE*f=fopen(p_,"rb");(void)fread(w,4,H,f);fclose(f);
    cudaMalloc(&d_fn,H*4);cudaMemcpy(d_fn,w,H*4,cudaMemcpyHostToDevice);delete[] w;}

    // lm_head tied to embed. Upload directly (same file).
    DevW8 lm_head_w=upload_w8((std::string(wdir)+"/embed_tokens").c_str());
    printf("lm_head loaded: %d x %d (INT8)\n", lm_head_w.K, lm_head_w.N);

    // Also need host-side copies for embedding dequant
    snprintf(p_,512,"%s/embed_tokens.int8_t",wdir);
    FILE* fhost=fopen(p_,"rb");int hh[5];fread(hh,4,5,fhost);
    size_t host_ds=(size_t)hh[0]*hh[1];int8_t* host_embed_d=new int8_t[host_ds];
    fread(host_embed_d,1,host_ds,fhost);fclose(fhost);
    snprintf(p_,512,"%s/embed_tokens.scale_t",wdir);
    fhost=fopen(p_,"rb");fread(hh,4,5,fhost);size_t host_ss=(size_t)hh[3]*hh[4];
    float* host_embed_sc=new float[host_ss];fread(host_embed_sc,4,host_ss,fhost);fclose(fhost);

    // Device buffers
    float *d_x32, *d_res, *d_xi_f, *d_Q, *d_K, *d_V, *d_attn, *d_proj, *d_gate, *d_up, *d_logits;
    cudaMalloc(&d_x32,H*4); cudaMalloc(&d_res,H*4); cudaMalloc(&d_xi_f,H*4);
    cudaMalloc(&d_Q,Q*4); cudaMalloc(&d_K,KV*4); cudaMalloc(&d_V,KV*4);
    cudaMalloc(&d_attn,Q*4); cudaMalloc(&d_proj,H*4); cudaMalloc(&d_gate,I*4); cudaMalloc(&d_up,I*4);
    cudaMalloc(&d_logits,V*4);

    // KV cache
    float *d_kc, *d_vc;
    die(cudaMalloc(&d_kc,(size_t)NL*nkv*MAXSEQ*hd*4),"kz_alloc");
    die(cudaMalloc(&d_vc,(size_t)NL*nkv*MAXSEQ*hd*4),"vz_alloc");
    die(cudaMemset(d_kc,0,(size_t)NL*nkv*MAXSEQ*hd*4),"kz");
    die(cudaMemset(d_vc,0,(size_t)NL*nkv*MAXSEQ*hd*4),"vz");

    // Sampling
    int* d_next_id; cudaMalloc(&d_next_id,4);
    // Repetition penalty buffer
    int* d_recent; cudaMalloc(&d_recent,64*4);

    // ── Embedding lookup + full decode ──
    std::vector<uint32_t> all_ids;
    printf("\n\u2500\u2500 Generating \u2500\u2500\n");

    auto t0=Clock::now();
    for(int step=0;step<gen_start+max_new;++step){
        int tid=step<gen_start?input_ids[step]:-1;

        // ── INT8 Embedding: host dequant single row → GPU ──
        if(tid>=0){
            float* h_embed = new float[H];
            dequant_embed_row(h_embed, tid, host_embed_d, host_embed_sc, H);
            die(cudaMemcpyAsync(d_x32,h_embed,H*4,cudaMemcpyHostToDevice,st),"embed_cpy");
            delete[] h_embed;
        }

        // ══ 32-layer decode ══
        for(int l=0;l<NL;++l){
            // Save residual
            die(cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st),"save_res");

            // Pre-attention norm
            die(blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_in,H,eps,st),"rmsnorm_in");

            // QKV: FP32 acts × INT8 per-row weights (no activation quant)
            die(blackwell::kernels::gemv_fp32_int8_per_row_warp(d_Q,d_xi_f,(const int8_t*)W[l].q.d,W[l].q.sc,H,Q,st),"q_proj");
            die(blackwell::kernels::gemv_fp32_int8_per_row_warp(d_K,d_xi_f,(const int8_t*)W[l].k.d,W[l].k.sc,H,KV,st),"k_proj");
            die(blackwell::kernels::gemv_fp32_int8_per_row_warp(d_V,d_xi_f,(const int8_t*)W[l].v.d,W[l].v.sc,H,KV,st),"v_proj");

            // Q/K head norms + RoPE
            head_norm_kernel<<<nqh,128,0,st>>>(d_Q,W[l].qn,nqh,hd,eps);
            head_norm_kernel<<<nkv,128,0,st>>>(d_K,W[l].kn,nkv,hd,eps);
            apply_rope_kernel<<<nqh,hd/2,0,st>>>(d_Q,nqh,hd,step);
            apply_rope_kernel<<<nkv,hd/2,0,st>>>(d_K,nkv,hd,step);

            // KV cache + attention
            size_t kv_off=(size_t)l*nkv*MAXSEQ*hd;
            die(blackwell::kernels::update_kv_cache(d_kc+kv_off,d_vc+kv_off,d_K,d_V,0,step,nkv,hd,MAXSEQ,st),"kv");
            die(blackwell::kernels::attention_decode_batched_gqa(d_attn,d_Q,d_kc,d_vc,step,nqh,nkv,hd,MAXSEQ,1,
                (size_t)NL*nkv*MAXSEQ*hd,kv_off,st),"attn");

            // Wo: FP32 attn × INT8 weights
            die(blackwell::kernels::gemv_fp32_int8_per_row_warp(d_proj,d_attn,(const int8_t*)W[l].o.d,W[l].o.sc,Q,H,st),"o_proj");

            // Attention residual
            die(blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,st),"attn_res");

            // Save pre-MLP state
            die(cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st),"save_res2");

            // Pre-MLP norm
            die(blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_post,H,eps,st),"rmsnorm_post");

            // MLP gate + up: FP32 acts × INT8 weights
            die(blackwell::kernels::gemv_fp32_int8_per_row_warp(d_gate,d_xi_f,(const int8_t*)W[l].g.d,W[l].g.sc,H,I,st),"gate");
            die(blackwell::kernels::gemv_fp32_int8_per_row_warp(d_up,d_xi_f,(const int8_t*)W[l].u.d,W[l].u.sc,H,I,st),"up");

            // SwiGLU in FP32
            blackwell::kernels::apply_swiglu(d_gate,d_gate,d_up,I,st);

            // Down: FP32 SwiGLU out × INT8 weights
            die(blackwell::kernels::gemv_fp32_int8_per_row_warp(d_proj,d_gate,(const int8_t*)W[l].d.d,W[l].d.sc,I,H,st),"down");

            // MLP residual
            die(blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,st),"mlp_res");
        }

        // Final norm + lm_head
        if(step>=gen_start-1){
            die(blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,d_fn,H,eps,st),"fn");

            // lm_head: FP32 acts × INT8 weights
            die(blackwell::kernels::gemv_fp32_int8_per_row_warp(d_logits,d_xi_f,(const int8_t*)lm_head_w.d,lm_head_w.sc,H,V,st),"lm_head");

            float l0,l264,l37018;
            cudaMemcpy(&l0,d_logits,4,cudaMemcpyDeviceToHost);
            cudaMemcpy(&l264,d_logits+264,4,cudaMemcpyDeviceToHost);
            cudaMemcpy(&l37018,d_logits+37018,4,cudaMemcpyDeviceToHost);
            printf("  logits: tok0=%.1f tok264=%.1f tok37018=%.1f\n",l0,l264,l37018);

            // Repetition penalty
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

            if(next_id==128001||next_id==128009){printf("\n[EOS]\n");break;}
        }
        cudaStreamSynchronize(st);
    }
    auto t1=Clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    int gen=(int)all_ids.size();
    printf("\n\n\u2500\u2500 Stats \u2500\u2500\n");
    printf("  Input: %zu  Gen: %d\n",input_ids.size(),gen);
    printf("  Time: %.1f ms  Speed: %.1f ms/tok = %.0f t/s\n",ms,ms/gen,gen*1000.0/ms);

    // Cleanup
    cudaStreamDestroy(st);
    delete[] W;
    delete[] qk_h;
    delete[] host_embed_d;
    delete[] host_embed_sc;
    cudaFree(d_x32); cudaFree(d_res); cudaFree(d_xi_f);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_attn); cudaFree(d_proj); cudaFree(d_gate); cudaFree(d_up);
    cudaFree(d_logits); cudaFree(d_kc); cudaFree(d_vc);
    cudaFree(d_next_id); cudaFree(d_recent); cudaFree(d_fn);
    return 0;
}
