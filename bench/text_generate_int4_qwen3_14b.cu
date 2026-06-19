#include "blackwell/int4_weights.h"
using namespace blackwell::weights;
// bench/text_generate_int4_qwen3_14b.cu — Qwen3-14B INT4 text generation
//
// 40L, H=5120, I=17408, nqh=40, nkv=8, hd=128, V=151936
// FP16 weight scales. Batched GEMV kernels (M=1).
//
// Build:
//   CUDACXX=/usr/local/cuda-13.3/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/text_generate_int4_qwen3_14b.cu build/libblackwell_kernels.a \
//     -o bench/text_generate_int4_qwen3_14b

#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

// FP16 embed table on GPU (pre-loaded, 1.56 GB)
static __half* d_embed_f16 = nullptr;

static void die(cudaError_t e, const char* m) {
    if(e!=cudaSuccess){printf("FAIL %s: %s\n",m,cudaGetErrorString(e));exit(1);}
}

using Clock = std::chrono::high_resolution_clock;

const int NL=40;
const int H=5120, Q=5120, KV=1024, I=17408;
const int nqh=40, nkv=8, hd=128, MAXSEQ=4096;
const float eps=1e-6f;
const int V=151936;

struct LW4 {
    DevW4f16 q,k,v,o,g,u,d;
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
    float theta=(float)pos*powf(1000000.0f,-2.0f*(float)d/(float)head_dim);
    float c=cosf(theta),s=sinf(theta),x=pair[0],y=pair[1];
    pair[0]=x*c-y*s; pair[1]=x*s+y*c;
}

int main(int argc, char** argv) {
    const char* prompt = "Once upon a time";
    int max_new = 50;
    float temperature = 0.0f;
    int top_k = 0;
    const char* wdir = "weights_int4_qwen3_14b_fp16sc";
    // First pass: extract flags
    for(int i=1;i<argc;i++){
        if(strcmp(argv[i],"-w")==0&&i+1<argc) wdir=argv[++i];
        if(strcmp(argv[i],"-t")==0&&i+1<argc) temperature=atof(argv[++i]);
        if(strcmp(argv[i],"-k")==0&&i+1<argc) top_k=atoi(argv[++i]);
    }
    // Non-flag args: first is prompt, second is max_new
    int arg_ix=1;
    for(;arg_ix<argc;arg_ix++){
        if(argv[arg_ix][0]!='-') break;
        if(strcmp(argv[arg_ix],"-w")==0||strcmp(argv[arg_ix],"-t")==0||strcmp(argv[arg_ix],"-k")==0) arg_ix++; // skip value
    }
    if(arg_ix<argc) prompt=argv[arg_ix++];
    if(arg_ix<argc) max_new=atoi(argv[arg_ix]);

    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    printf("# Text Generation — Qwen3-14B INT4\n");
    printf("  Device: %s\n", P.name);
    printf("  Prompt: \"%s\"\n", prompt);
    printf("  Weights: %s\n", wdir);
    printf("  Config: NL=%d H=%d I=%d nqh=%d nkv=%d hd=%d V=%d\n\n", NL, H, I, nqh, nkv, hd, V);

    blackwell::BpeTokenizer tokenizer;
    if(tokenizer.load("tokenizer_data.bin")!=0){
        fprintf(stderr,"FAIL: no tokenizer_data.bin\n");return 1;
    }

    std::vector<uint32_t> input_ids=tokenizer.encode(prompt);
    printf("Input: %zu tokens\n", input_ids.size());

    // Device buffers
    float *d_x32, *d_xi_f, *d_res;
    uint8_t *d_x_i4; float *d_x_i4_sc;
    float *d_Q,*d_K,*d_V,*d_attn;
    uint8_t *d_attn_i4; float *d_attn_i4_sc;
    float *d_proj, *d_gate, *d_up;
    uint8_t *d_mlp_i4; float *d_mlp_i4_sc;
    float *d_fn, *d_fn_sc, *d_kc, *d_vc, *d_logits;
    int *d_next_id;

    // NL defined at file scope
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
    AL(d_logits,V*4);AL(d_next_id,4);
    #undef AL

    float iv7=1.f/7.f;
    { std::vector<float> tmp(H/16, iv7); cudaMemcpy(d_x_i4_sc, tmp.data(), (H/16)*4, cudaMemcpyHostToDevice); }
    { std::vector<float> tmp(Q/16, iv7); cudaMemcpy(d_attn_i4_sc, tmp.data(), (Q/16)*4, cudaMemcpyHostToDevice); }
    { std::vector<float> tmp(I/16, iv7); cudaMemcpy(d_mlp_i4_sc, tmp.data(), (I/16)*4, cudaMemcpyHostToDevice); }
    { std::vector<float> tmp(H/16, iv7); cudaMemcpy(d_fn_sc, tmp.data(), (H/16)*4, cudaMemcpyHostToDevice); }
    int dummy=0;cudaMemcpy(d_next_id,&dummy,4,cudaMemcpyHostToDevice);

    printf("Loading %d-layer INT4 model...\n",NL);fflush(stdout);
    std::vector<LW4> W(NL); char p_[512];
    for(int l=0;l<NL;++l){
        snprintf(p_,512,"%s/%d_q_proj",wdir,l);W[l].q=upload_w4_f16sc(p_);
        snprintf(p_,512,"%s/%d_k_proj",wdir,l);W[l].k=upload_w4_f16sc(p_);
        snprintf(p_,512,"%s/%d_v_proj",wdir,l);W[l].v=upload_w4_f16sc(p_);
        snprintf(p_,512,"%s/%d_o_proj",wdir,l);W[l].o=upload_w4_f16sc(p_);
        snprintf(p_,512,"%s/%d_gate_proj",wdir,l);W[l].g=upload_w4_f16sc(p_);
        snprintf(p_,512,"%s/%d_up_proj",wdir,l);W[l].u=upload_w4_f16sc(p_);
        snprintf(p_,512,"%s/%d_down_proj",wdir,l);W[l].d=upload_w4_f16sc(p_);
        if((l+1)%10==0||l+1==NL)printf("  layer %d/%d\n",l+1,NL);
    }

    float* qk_h=(float*)malloc(NL*2*hd*4);
    {snprintf(p_,512,"%s/qk_norms.f32",wdir);FILE*f=fopen(p_,"rb");if(!f){printf("FAIL open %s\n",p_);exit(1);}size_t nr=fread(qk_h,4,NL*2*hd,f);if(nr!=(size_t)NL*2*hd){printf("FAIL read qk_norms\n");exit(1);}fclose(f);}
    for(int l=0;l<NL;++l){
        cudaMalloc(&W[l].qn,hd*4);cudaMemcpy(W[l].qn,qk_h+l*2*hd,hd*4,cudaMemcpyHostToDevice);
        cudaMalloc(&W[l].kn,hd*4);cudaMemcpy(W[l].kn,qk_h+l*2*hd+hd,hd*4,cudaMemcpyHostToDevice);
    }free(qk_h);

    for(int l=0;l<NL;++l){
        float* w=(float*)malloc(H*4);
        snprintf(p_,512,"%s/%d_ln1.f32",wdir,l);
        {FILE*f=fopen(p_,"rb");if(!f){printf("FAIL open %s\n",p_);exit(1);}size_t nr=fread(w,4,H,f);if(nr!=(size_t)H){printf("FAIL read %s\n",p_);exit(1);}fclose(f);}
        cudaMalloc(&W[l].rn_in,H*4);cudaMemcpy(W[l].rn_in,w,H*4,cudaMemcpyHostToDevice);
        snprintf(p_,512,"%s/%d_ln2.f32",wdir,l);
        {FILE*f=fopen(p_,"rb");if(!f){printf("FAIL open %s\n",p_);exit(1);}size_t nr=fread(w,4,H,f);if(nr!=(size_t)H){printf("FAIL read %s\n",p_);exit(1);}fclose(f);}
        cudaMalloc(&W[l].rn_post,H*4);cudaMemcpy(W[l].rn_post,w,H*4,cudaMemcpyHostToDevice);
        free(w);
    }

    {float*w=(float*)malloc(H*4);
    snprintf(p_,512,"%s/final_norm.f32",wdir);FILE*f=fopen(p_,"rb");if(!f){printf("FAIL open %s\n",p_);exit(1);}size_t nr=fread(w,4,H,f);if(nr!=(size_t)H){printf("FAIL read final_norm\n");exit(1);}fclose(f);
    cudaMemcpy(d_fn,w,H*4,cudaMemcpyHostToDevice);free(w);}

    // Embed loaded as FP16 on GPU (pre-loaded, no dequant per token)
    // embed.fp16 format: [V][H] FP16, row-major
    {char p[512];
    snprintf(p,512,"%s/embed_tokens.fp16",wdir);
    FILE* f=fopen(p,"rb"); if(!f){printf("FAIL open %s\n",p);exit(1);}
    fseek(f,0,SEEK_END); size_t fsize=ftell(f); fseek(f,0,SEEK_SET);
    size_t n_f16 = fsize / 2;
    __half* host_buf = new __half[n_f16];
    size_t nr = fread(host_buf,2,n_f16,f);
    if(nr != n_f16){printf("FAIL read embed.fp16: read %zu expected %zu\n",nr,n_f16);exit(1);}
    fclose(f);
    cudaMalloc(&d_embed_f16, fsize);
    cudaMemcpy(d_embed_f16, host_buf, fsize, cudaMemcpyHostToDevice);
    delete[] host_buf;
    printf("Embed tokens loaded: 151936 x %d (FP16, %.1f MB)\n",H,(float)fsize/1e6);}

    // Load lm_head INT4
    char lm_p[512]; snprintf(lm_p,512,"%s/lm_head",wdir);
    DevW4f16 lm_head_w = upload_w4_f16sc(lm_p);
    printf("lm_head loaded: %d x %d (INT4)\n", lm_head_w.K, lm_head_w.N);
    printf("All weights loaded.\n\n");

    cudaStream_t st;die(cudaStreamCreate(&st),"stream");
    srand((unsigned)time(nullptr));

    std::vector<float> h_embed(H);
    cudaMemset(d_kc,0,(size_t)NL*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vc,0,(size_t)NL*nkv*MAXSEQ*hd*4);

    // Pre-allocate FP16 embed row buffer on GPU
    __half* d_embed_row;
    cudaMalloc(&d_embed_row, (size_t)H * sizeof(__half));

    printf("── Generating ──\n");
    printf("%s",prompt);fflush(stdout);

    std::vector<uint32_t> all_ids=input_ids;
    int gen_start=(int)input_ids.size();
    int total=gen_start+max_new;
    auto t_start=Clock::now();

    for(int step=0;step<total;++step){
        uint32_t tid=(step<gen_start)?input_ids[step]:all_ids.back();

        // FP16 Embedding: D2D copy from pre-loaded GPU table, then FP16→FP32 convert
        die(cudaMemcpyAsync(d_embed_row, d_embed_f16 + (size_t)tid * H,
            (size_t)H * sizeof(__half), cudaMemcpyDeviceToDevice, st), "embed_copy");
        // Convert FP16→FP32: simple launch
        // FP16 Embedding: D2D copy from pre-loaded GPU table, then FP16→FP32 convert
        die(cudaMemcpyAsync(d_embed_row, d_embed_f16 + (size_t)tid * H,
            (size_t)H * sizeof(__half), cudaMemcpyDeviceToDevice, st), "embed_copy");
        blackwell::kernels::convert_fp16_to_fp32(d_x32, d_embed_row, H, st);

        // 40-layer decode
        for(int l=0;l<NL;++l){
            // Save residual before norm
            die(cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st),"save_res");

            die(blackwell::kernels::fused_rmsnorm_quant_int4(d_x_i4,d_x_i4_sc,d_x32,W[l].rn_in,H,eps,st),"rmsnorm_quant_in");

            // Fused QKV projections (3 GEMV → 1 kernel)
            die(blackwell::kernels::fused_qkv_int4_f16wsc(d_Q,d_K,d_V,
                (const uint8_t*)d_x_i4,d_x_i4_sc,
                W[l].q.d,W[l].q.sc16,
                W[l].k.d,W[l].k.sc16,
                W[l].v.d,W[l].v.sc16,
                H,Q,KV,1,st),"fused_qkv");

            // Q/K head norms + RoPE
            head_norm_kernel<<<nqh,128,0,st>>>(d_Q,W[l].qn,nqh,hd,eps);
            die(cudaGetLastError(),"head_norm_Q");
            head_norm_kernel<<<nkv,128,0,st>>>(d_K,W[l].kn,nkv,hd,eps);
            die(cudaGetLastError(),"head_norm_K");
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
            die(blackwell::kernels::gemv_int4_warp_f16wsc(d_proj,(const uint8_t*)d_attn_i4,d_attn_i4_sc,W[l].o.d,W[l].o.sc16,Q,H,st),"o_proj");

            // Attention residual
            die(blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,st),"attn_res");

            // Save pre-MLP state
            die(cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st),"save_res2");

            die(blackwell::kernels::fused_rmsnorm_quant_int4(d_x_i4,d_x_i4_sc,d_x32,W[l].rn_post,H,eps,st),"rmsnorm_quant_post");

            // Fused gate+up projections (2 GEMV → 1 kernel)
            die(blackwell::kernels::fused_gate_up_int4_f16wsc(d_gate,d_up,
                (const uint8_t*)d_x_i4,d_x_i4_sc,
                W[l].g.d,W[l].g.sc16,
                W[l].u.d,W[l].u.sc16,
                H,I,1,st),"fused_gate_up");

            // SwiGLU + INT4 quant (fused)
            die(blackwell::kernels::fused_swiglu_quant_int4(d_mlp_i4,d_mlp_i4_sc,d_gate,d_up,I,st),"swiglu_quant");

            // Down projection
            die(blackwell::kernels::gemv_int4_warp_f16wsc(d_proj,(const uint8_t*)d_mlp_i4,d_mlp_i4_sc,W[l].d.d,W[l].d.sc16,I,H,st),"down");

            // MLP residual
            die(blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,st),"mlp_res");
        }

        // Final norm + lm_head + GPU sampling
        if(step>=gen_start-1){
            die(blackwell::kernels::fused_rmsnorm_quant_int4(d_x_i4,d_x_i4_sc,d_x32,d_fn,H,eps,st),"fn_quant");

            die(blackwell::kernels::gemv_int4_warp_f16wsc(d_logits,(const uint8_t*)d_x_i4,d_x_i4_sc,lm_head_w.d,lm_head_w.sc16,H,V,st),"lm_head");
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
        cudaFree(w.q.d);cudaFree(w.q.sc16);
        cudaFree(w.k.d);cudaFree(w.k.sc16);
        cudaFree(w.v.d);cudaFree(w.v.sc16);
        cudaFree(w.o.d);cudaFree(w.o.sc16);
        cudaFree(w.g.d);cudaFree(w.g.sc16);
        cudaFree(w.u.d);cudaFree(w.u.sc16);
        cudaFree(w.d.d);cudaFree(w.d.sc16);
        cudaFree(w.qn);cudaFree(w.kn);
        cudaFree(w.rn_in);cudaFree(w.rn_post);
    }
    if(d_embed_f16) cudaFree(d_embed_f16);
    return 0;
}
