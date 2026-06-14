// bench/text_generate_int2_qwen3_8b.cu — INT2 8B text generation
//
// INT4 activations × INT2 packed weights (block-16 offset-binary).
// Same decode pipeline as INT4 8B (text_generate_int4_qwen3_8b.cu),
// with gemv_int2_batched replacing gemv_int4_warp on weight GEMV calls.
//
// Estimated 2× throughput (half weight memory bandwidth, GEMV is 92%).
//
// Build:
//   cmake --build build --target text_generate_int2_qwen3_8b

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

// Qwen3-8B dimensions
const int H=4096, Q=4096, KV=1024, I=12288;
const int nqh=32, nkv=8, hd=128, MAXSEQ=4096;
const float eps=1e-6f;
const int V=151936;
const int NL=36;

// INT2 weight: [N][K/4] packed bytes + [N][K/16] FP32 scales
// INT4 weight: [N][K/2] packed bytes + [N][K/16] FP32 scales
// Header: int h[5] = {K, N, block_size, block_count, _}
struct DevW2 { int K, N; uint8_t* d; float* sc; };
static DevW2 upload_w2(const char* prefix) {
    char p[256]; snprintf(p,256,"%s.int2_t",prefix);
    FILE* f=fopen(p,"rb"); int h[5]; fread(h,4,5,f);
    DevW2 dw; dw.K=h[0]; dw.N=h[1];
    size_t ds=(size_t)h[0]*h[1]/4;  // INT2: 4 vals/byte
    uint8_t* td=new uint8_t[ds]; fread(td,1,ds,f); fclose(f);
    cudaMalloc(&dw.d,ds); cudaMemcpy(dw.d,td,ds,cudaMemcpyHostToDevice); delete[] td;
    snprintf(p,256,"%s.scale_t",prefix); f=fopen(p,"rb"); fread(h,4,5,f);
    size_t ss=(size_t)h[3]*h[4];  // Scale count from scale_t header
    float* ts=new float[ss]; fread(ts,4,ss,f); fclose(f);
    cudaMalloc(&dw.sc,ss*4); cudaMemcpy(dw.sc,ts,ss*4,cudaMemcpyHostToDevice); delete[] ts;
    return dw;
}

struct LW2 {
    DevW2 q,k,v,o,g,u,d;
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

// INT4 embedding lookup (same as INT4 path — embedding is INT4)

int main(int argc, char** argv) {
    const char* prompt = "Once upon a time";
    int max_new = 50;
    bool chat_mode = false;
    float temperature = 0.0f;
    int top_k = 0;
    if(argc>1) prompt=argv[1];
    if(argc>2) max_new=atoi(argv[2]);
    for(int i=1;i<argc;i++){
        if(strcmp(argv[i],"--chat")==0) chat_mode=true;
        if(strcmp(argv[i],"-t")==0&&i+1<argc) temperature=atof(argv[++i]);
        if(strcmp(argv[i],"-k")==0&&i+1<argc) top_k=atoi(argv[++i]);
    }

    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    printf("# Text Generation — Qwen3-8B INT2\n");
    printf("  Device: %s\n", P.name);
    printf("  Prompt: \"%s\"%s\n", prompt, chat_mode?" (chat)":"");
    printf("  Temp: %.1f, Top-K: %d, Max new: %d\n\n", temperature, top_k, max_new);

    blackwell::BpeTokenizer tokenizer;
    if(tokenizer.load("tokenizer_data.bin")!=0){
        fprintf(stderr,"FAIL: no tokenizer_data.bin\n");return 1;
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

    // Device buffers (same layout as INT4 — activation quant is INT4)
    float *d_x32, *d_xi_f, *d_res;
    uint8_t *d_x_i4; float *d_x_i4_sc;
    float *d_Q,*d_K,*d_V,*d_attn;
    uint8_t *d_attn_i4; float *d_attn_i4_sc;
    float *d_proj, *d_gate, *d_up;
    uint8_t *d_mlp_i4; float *d_mlp_i4_sc;
    float *d_fn, *d_kc, *d_vc, *d_logits;
    int *d_next_id;

    #define AL(p,n){cudaError_t _e=cudaMalloc(&(p),(n));\
        if(_e!=cudaSuccess){printf("FAIL malloc %s: %s\n",#p,cudaGetErrorString(_e));die(_e,#p);}}
    AL(d_x32,H*4);AL(d_xi_f,H*4);AL(d_res,H*4);
    AL(d_x_i4,H/2);AL(d_x_i4_sc,(H/16)*4);
    AL(d_Q,Q*4);AL(d_K,KV*4);AL(d_V,KV*4);AL(d_attn,Q*4);
    AL(d_attn_i4,Q/2);AL(d_attn_i4_sc,(Q/16)*4);
    AL(d_proj,H*4);AL(d_gate,I*4);AL(d_up,I*4);
    AL(d_mlp_i4,I/2);AL(d_mlp_i4_sc,(I/16)*4);
    AL(d_fn,H*4);
    AL(d_kc,(size_t)NL*nkv*MAXSEQ*hd*4);
    AL(d_vc,(size_t)NL*nkv*MAXSEQ*hd*4);
    AL(d_logits,V*4);AL(d_next_id,4);
    #undef AL

    float iv7=1.f/7.f;
    { std::vector<float> tmp(H/16, iv7); cudaMemcpy(d_x_i4_sc, tmp.data(), (H/16)*4, cudaMemcpyHostToDevice); }
    { std::vector<float> tmp(Q/16, iv7); cudaMemcpy(d_attn_i4_sc, tmp.data(), (Q/16)*4, cudaMemcpyHostToDevice); }
    { std::vector<float> tmp(I/16, iv7); cudaMemcpy(d_mlp_i4_sc, tmp.data(), (I/16)*4, cudaMemcpyHostToDevice); }
    int dummy=0;cudaMemcpy(d_next_id,&dummy,4,cudaMemcpyHostToDevice);

    printf("Loading %d-layer INT2 model...\n",NL);fflush(stdout);
    std::vector<LW2> W(NL); char p_[256];
    for(int l=0;l<NL;++l){
        snprintf(p_,256,"weights_int2_qwen3_8b/%d_self_attn.q_proj",l);W[l].q=upload_w2(p_);
        snprintf(p_,256,"weights_int2_qwen3_8b/%d_self_attn.k_proj",l);W[l].k=upload_w2(p_);
        snprintf(p_,256,"weights_int2_qwen3_8b/%d_self_attn.v_proj",l);W[l].v=upload_w2(p_);
        snprintf(p_,256,"weights_int2_qwen3_8b/%d_self_attn.o_proj",l);W[l].o=upload_w2(p_);
        snprintf(p_,256,"weights_int2_qwen3_8b/%d_mlp.gate_proj",l);W[l].g=upload_w2(p_);
        snprintf(p_,256,"weights_int2_qwen3_8b/%d_mlp.up_proj",l);W[l].u=upload_w2(p_);
        snprintf(p_,256,"weights_int2_qwen3_8b/%d_mlp.down_proj",l);W[l].d=upload_w2(p_);
        if((l+1)%7==0||l+1==NL)printf("  layer %d/%d\n",l+1,NL);
    }

    // Q/K head norms
    float* qk_h=(float*)malloc(NL*2*hd*4);
    {FILE*f=fopen("weights_int2_qwen3_8b/qk_norms.f32","rb");(void)fread(qk_h,4,NL*2*hd,f);fclose(f);}
    for(int l=0;l<NL;++l){
        cudaMalloc(&W[l].qn,hd*4);cudaMemcpy(W[l].qn,qk_h+l*2*hd,hd*4,cudaMemcpyHostToDevice);
        cudaMalloc(&W[l].kn,hd*4);cudaMemcpy(W[l].kn,qk_h+l*2*hd+hd,hd*4,cudaMemcpyHostToDevice);
    }free(qk_h);

    // Layer norms
    for(int l=0;l<NL;++l){
        float* w=(float*)malloc(H*4);
        snprintf(p_,256,"weights_int2_qwen3_8b/%d_input_layernorm.f32",l);
        {FILE*f=fopen(p_,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&W[l].rn_in,H*4);cudaMemcpy(W[l].rn_in,w,H*4,cudaMemcpyHostToDevice);
        snprintf(p_,256,"weights_int2_qwen3_8b/%d_post_attention_layernorm.f32",l);
        {FILE*f=fopen(p_,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&W[l].rn_post,H*4);cudaMemcpy(W[l].rn_post,w,H*4,cudaMemcpyHostToDevice);
        free(w);
    }

    // Final norm
    {float*w=(float*)malloc(H*4);
    FILE*f=fopen("weights_int2_qwen3_8b/final_norm.f32","rb");(void)fread(w,4,H,f);fclose(f);
    cudaMemcpy(d_fn,w,H*4,cudaMemcpyHostToDevice);free(w);}

    // Embedding table (INT4 — same as INT4 path, embeddings are not INT2)
    DevW2 embed=upload_w2("weights_int2_qwen3_8b/embed_tokens");
    uint8_t* host_embed_d=new uint8_t[(size_t)embed.K*embed.N/4];
    float* host_embed_sc=new float[(size_t)embed.N*(embed.K/16)];
    {char p[256];
    snprintf(p,256,"weights_int2_qwen3_8b/embed_tokens.int2_t");
    FILE*f=fopen(p,"rb");int h[5];fread(h,4,5,f);
    size_t ds=(size_t)h[0]*h[1]/4;fread(host_embed_d,1,ds,f);fclose(f);
    snprintf(p,256,"weights_int2_qwen3_8b/embed_tokens.scale_t");
    f=fopen(p,"rb");fread(h,4,5,f);size_t ss=(size_t)h[3]*h[4];
    fread(host_embed_sc,4,ss,f);fclose(f);}
    printf("Embed tokens loaded: %d x %d (INT2)\n",embed.K,embed.N);

    // lm_head (INT2)
    DevW2 lm_head_w = upload_w2("weights_int2_qwen3_8b/lm_head");
    printf("lm_head loaded: %d x %d (INT2)\n", lm_head_w.K, lm_head_w.N);
    printf("All weights loaded.\n\n");

    cudaStream_t st;die(cudaStreamCreate(&st),"stream");
    srand((unsigned)time(nullptr));

    std::vector<float> h_embed(H);
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

        // ── Embedding: host dequant single INT2 row → GPU ──
                // Dequant INT2 embedding: 4 vals/byte, offset-binary (val-2)
        {
            int kblocks = H / 16;
            int K = H;
            for(int b=0;b<kblocks;++b){
                float sc = host_embed_sc[tid * kblocks + b];
                for(int i=0;i<16;++i){
                    size_t byte_idx = (size_t)tid * K / 4 + (size_t)b * 4 + i / 4;
                    uint8_t byte = host_embed_d[byte_idx];
                    int shift = (i % 4) * 2;
                    int val = ((byte >> shift) & 0x3) - 2;  // offset-binary: 0→-2, 1→-1, 2→0, 3→1
                    h_embed[b*16+i] = (float)val * sc;
                }
            }
        }
        die(cudaMemcpyAsync(d_x32,h_embed.data(),H*4,cudaMemcpyHostToDevice,st),"embed_cpy");

        // ══ 36-layer decode ══
        for(int l=0;l<NL;++l){
            // Save residual
            die(cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st),"save_res");

            // Pre-attention norm
            die(blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_in,H,eps,st),"rmsnorm_in");

            // Quantize → INT4 activations
            die(blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st),"quant_in");

            // QKV projections: INT4 activations × INT2 weights
            die(blackwell::kernels::gemv_int2_batched(d_Q,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].q.d,W[l].q.sc,H,Q,1,st),"q_proj");
            die(blackwell::kernels::gemv_int2_batched(d_K,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].k.d,W[l].k.sc,H,KV,1,st),"k_proj");
            die(blackwell::kernels::gemv_int2_batched(d_V,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].v.d,W[l].v.sc,H,KV,1,st),"v_proj");

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

            // Wo projection: INT4 attn × INT2 weights
            die(blackwell::kernels::quantize_int4(d_attn_i4,d_attn_i4_sc,d_attn,Q,st),"quant_attn");
            die(blackwell::kernels::gemv_int2_batched(d_proj,(const uint8_t*)d_attn_i4,d_attn_i4_sc,W[l].o.d,W[l].o.sc,Q,H,1,st),"o_proj");

            // Attention residual
            die(blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,st),"attn_res");

            // Save pre-MLP state
            die(cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st),"save_res2");

            // Pre-MLP norm
            die(blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_post,H,eps,st),"rmsnorm_post");
            die(blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st),"quant_mlp_in");

            // MLP gate + up: INT4 activations × INT2 weights
            die(blackwell::kernels::gemv_int2_batched(d_gate,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].g.d,W[l].g.sc,H,I,1,st),"gate");
            die(blackwell::kernels::gemv_int2_batched(d_up,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].u.d,W[l].u.sc,H,I,1,st),"up");

            // Fused SwiGLU + INT4 quant (activations remain INT4 for down-proj)
            blackwell::kernels::apply_swiglu(d_gate, d_gate, d_up, I, st);
            blackwell::kernels::quantize_int4(d_mlp_i4, d_mlp_i4_sc, d_gate, I, st);

            // Down projection: INT4 activations × INT2 weights
            die(blackwell::kernels::gemv_int2_batched(d_proj,(const uint8_t*)d_mlp_i4,d_mlp_i4_sc,W[l].d.d,W[l].d.sc,I,H,1,st),"down");

            // MLP residual
            die(blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,st),"mlp_res");
        }

        // Final norm + lm_head + GPU sampling
        if(step>=gen_start-1){
            die(blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,d_fn,H,eps,st),"fn");

            // Quantize → INT4 for lm_head
            die(blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st),"quant_lm");

            // lm_head: INT4 activations × INT2 weights
            die(blackwell::kernels::gemv_int2_batched(d_logits,(const uint8_t*)d_x_i4,d_x_i4_sc,lm_head_w.d,lm_head_w.sc,H,V,1,st),"lm_head");

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