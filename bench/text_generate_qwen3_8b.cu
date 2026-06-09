// bench/text_generate_qwen3_8b.cu — End-to-end text generation with INT8 Qwen3-8B
//
// Tokenize prompt → embedding lookup → 36L INT8 decode → lm_head GEMV → GPU sampling
// Uses real INT8 weights from weights_int8_qwen3_8b/ and BPE tokenizer.
//
// Build:
//   CUDACXX=/usr/local/cuda-13.3/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/text_generate_qwen3_8b.cu build/libblackwell_kernels.a \
//     -o bench/text_generate_qwen3_8b
//
// Run: ./bench/text_generate_qwen3_8b "Once upon a time" [max_new_tokens=50]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cstring>
#include <string>
#include "blackwell/kernels.h"
#include "blackwell/bpe_tokenizer.h"

static void die(cudaError_t e, const char* m) {
    if(e!=cudaSuccess){printf("FAIL %s: %s\n",m,cudaGetErrorString(e));exit(1);}
}

// Compute 16-element block absmax scales for pack_int8
__global__ void absmax_scales_kernel(const float* in, float* sc, int n) {
    int blk=blockIdx.x; int lane=threadIdx.x; float amax=0;
    for(int i=lane;i<16&&blk*16+i<n;i+=32) amax=fmaxf(amax,fabsf(in[blk*16+i]));
    for(int off=16;off>0;off>>=1) amax=fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,off));
    if(lane==0) sc[blk]=fmaxf(amax/127.0f,1e-9f);
}

// Per-head RMSNorm for Q/K norms (head_dim=128)
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

// RoPE kernel: apply rotary position embeddings
__global__ void apply_rope_kernel(float* data, int n_heads, int head_dim, int pos) {
    int h=blockIdx.x, d=threadIdx.x;
    if(h>=n_heads||d>=head_dim/2) return;
    float* pair=data+(size_t)h*head_dim+d*2;
    float theta=(float)pos*powf(1000000.0f,-2.0f*(float)d/(float)head_dim);
    float c=cosf(theta),s=sinf(theta),x=pair[0],y=pair[1];
    pair[0]=x*c-y*s; pair[1]=x*s+y*c;
}

using Clock = std::chrono::high_resolution_clock;

// Model constants — Qwen3-8B
const int H=4096, Q=4096, KV=1024, I=12288;
const int nqh=32, nkv=8, hd=128, MAXSEQ=4096, NL=36;
const float eps=1e-6f;
const int V=151936;

struct LW { int K,N; std::vector<int8_t> d; std::vector<float> sc; };
struct DW { int8_t* d; float* sc; };

static LW load_w(const char* p) {
    char x[256]; snprintf(x,256,"%s.int8_t",p);
    FILE* f=fopen(x,"rb"); if(!f){printf("FAIL open %s\n",x);exit(1);}
    int h[5]; (void)fread(h,4,5,f); LW w;
    w.K=h[0]; w.N=h[1]; w.d.resize(h[0]*h[1]); (void)fread(w.d.data(),1,w.d.size(),f); fclose(f);
    snprintf(x,256,"%s.scale_t",p); f=fopen(x,"rb"); (void)fread(h,4,5,f);
    w.sc.resize(h[3]*h[4]); (void)fread(w.sc.data(),4,w.sc.size(),f); fclose(f);
    return w;
}
static DW upload(const LW& w) {
    DW d;
    cudaMalloc(&d.d,w.d.size()); cudaMemcpy(d.d,w.d.data(),w.d.size(),cudaMemcpyHostToDevice);
    cudaMalloc(&d.sc,w.sc.size()*4); cudaMemcpy(d.sc,w.sc.data(),w.sc.size()*4,cudaMemcpyHostToDevice);
    return d;
}

struct L { DW q,k,v,o,g,u,d; float *qn,*kn,*rn_in,*rn_post; };

// Host-side argmax
static int argmax_host(const float* logits, int n) {
    int best=0; float bv=logits[0];
    for(int i=1;i<n;i++) if(logits[i]>bv){bv=logits[i];best=i;}
    return best;
}
static int sample(const float* logits, int n, float temp, int top_k) {
    if(temp<0.01f) return argmax_host(logits,n);
    std::vector<float> probs(n);
    float mx=logits[0]; for(int i=1;i<n;i++) if(logits[i]>mx) mx=logits[i];
    float sum=0;
    for(int i=0;i<n;i++){probs[i]=expf((logits[i]-mx)/temp);sum+=probs[i];}
    if(top_k>0&&top_k<n){
        float thresh=-1e38f;
        std::vector<int> idx(n); for(int i=0;i<n;i++) idx[i]=i;
        std::partial_sort(idx.begin(),idx.begin()+top_k,idx.end(),[&](int a,int b){return probs[a]>probs[b];});
        thresh=probs[idx[top_k-1]];
        for(int i=0;i<n;i++) if(probs[i]<thresh) probs[i]=0;
        sum=0; for(int i=0;i<n;i++) sum+=probs[i];
    }
    float r=(float)rand()/(float)RAND_MAX*sum;
    float cs=0;
    for(int i=0;i<n;i++){cs+=probs[i];if(cs>=r)return i;}
    return n-1;
}

int main(int argc, char** argv) {
    const char* prompt="The capital of France is";
    int max_new=50;
    float temperature=0.0f;
    int top_k=0;
    if(argc>1) prompt=argv[1];
    if(argc>2) max_new=atoi(argv[2]);
    for(int i=1;i<argc;i++){
        if(strcmp(argv[i],"-t")==0&&i+1<argc) temperature=atof(argv[++i]);
        if(strcmp(argv[i],"-k")==0&&i+1<argc) top_k=atoi(argv[++i]);
    }

    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    printf("# Text Generation — Qwen3-8B INT8\n");
    printf("  Device: %s\n", P.name);
    printf("  Prompt: \"%s\"\n", prompt);
    printf("  Temp: %.1f, Top-K: %d, Max new: %d\n\n", temperature, top_k, max_new);

    blackwell::BpeTokenizer tokenizer;
    if(tokenizer.load("tokenizer_data.bin")!=0){
        fprintf(stderr,"FAIL: no tokenizer_data.bin\n");return 1;
    }
    auto input_ids=tokenizer.encode(prompt);
    printf("Input: %zu tokens\n\n", input_ids.size());

    // ── Buffers ──
    float *d_x32,*d_xi_f,*d_res;
    int8_t *d_x_i8; float *d_x_s;
    float *d_Q,*d_K,*d_V,*d_attn;
    int8_t *d_attn_i8; float *d_attn_s;
    float *d_proj,*d_gate,*d_up;
    int8_t *d_mlp_i8; float *d_mlp_s;
    float *d_fn,*d_kc,*d_vc,*d_logits;
    int *d_next_id;

    #define AL(p,n){cudaError_t _e=cudaMalloc(&(p),(n));\
        if(_e!=cudaSuccess){printf("FAIL malloc %s: %s\n",#p,cudaGetErrorString(_e));die(_e,#p);}}
    AL(d_x32,H*4);AL(d_xi_f,H*4);AL(d_res,H*4);
    AL(d_x_i8,H);AL(d_x_s,(H/16)*4);
    AL(d_Q,Q*4);AL(d_K,KV*4);AL(d_V,KV*4);AL(d_attn,Q*4);
    AL(d_attn_i8,Q);AL(d_attn_s,(Q/16)*4);
    AL(d_proj,H*4);AL(d_gate,I*4);AL(d_up,I*4);
    AL(d_mlp_i8,I);AL(d_mlp_s,(I/16)*4);
    AL(d_fn,H*4);
    AL(d_kc,(size_t)NL*nkv*MAXSEQ*hd*4);
    AL(d_vc,(size_t)NL*nkv*MAXSEQ*hd*4);
    AL(d_logits,V*4);AL(d_next_id,4);
    #undef AL

    int dummy=0; cudaMemcpy(d_next_id,&dummy,4,cudaMemcpyHostToDevice);

    // ── Load weights ──
    printf("Loading %d-layer INT8 model...\n",NL); fflush(stdout);
    std::vector<L> W(NL); char p[256];
    for(int l=0;l<NL;++l){
        snprintf(p,256,"weights_int8_qwen3_8b/%d_self_attn.q_proj",l);W[l].q=upload(load_w(p));
        snprintf(p,256,"weights_int8_qwen3_8b/%d_self_attn.k_proj",l);W[l].k=upload(load_w(p));
        snprintf(p,256,"weights_int8_qwen3_8b/%d_self_attn.v_proj",l);W[l].v=upload(load_w(p));
        snprintf(p,256,"weights_int8_qwen3_8b/%d_self_attn.o_proj",l);W[l].o=upload(load_w(p));
        snprintf(p,256,"weights_int8_qwen3_8b/%d_mlp.gate_proj",l);W[l].g=upload(load_w(p));
        snprintf(p,256,"weights_int8_qwen3_8b/%d_mlp.up_proj",l);W[l].u=upload(load_w(p));
        snprintf(p,256,"weights_int8_qwen3_8b/%d_mlp.down_proj",l);W[l].d=upload(load_w(p));
        if((l+1)%9==0||l+1==NL)printf("  layer %d/%d\n",l+1,NL);
    }

    // QK norms
    float* qk_h=(float*)malloc(NL*2*hd*4);
    {FILE*f=fopen("weights_int8_qwen3_8b/qk_norms.f32","rb");(void)fread(qk_h,4,NL*2*hd,f);fclose(f);}
    for(int l=0;l<NL;++l){
        cudaMalloc(&W[l].qn,hd*4);cudaMemcpy(W[l].qn,qk_h+l*2*hd,hd*4,cudaMemcpyHostToDevice);
        cudaMalloc(&W[l].kn,hd*4);cudaMemcpy(W[l].kn,qk_h+l*2*hd+hd,hd*4,cudaMemcpyHostToDevice);
    }free(qk_h);

    // RMSNorm weights
    for(int l=0;l<NL;++l){
        float* w=(float*)malloc(H*4);
        snprintf(p,256,"weights_int8_qwen3_8b/%d_input_layernorm.f32",l);
        {FILE*f=fopen(p,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&W[l].rn_in,H*4);cudaMemcpy(W[l].rn_in,w,H*4,cudaMemcpyHostToDevice);
        snprintf(p,256,"weights_int8_qwen3_8b/%d_post_attention_layernorm.f32",l);
        {FILE*f=fopen(p,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&W[l].rn_post,H*4);cudaMemcpy(W[l].rn_post,w,H*4,cudaMemcpyHostToDevice);
        free(w);
    }

    // Final norm
    {float*w=(float*)malloc(H*4);
    FILE*f=fopen("weights_int8_qwen3_8b/final_norm.f32","rb");(void)fread(w,4,H,f);fclose(f);
    cudaMemcpy(d_fn,w,H*4,cudaMemcpyHostToDevice);free(w);}

    // Embed tokens
    auto emb_lw=load_w("weights_int8_qwen3_8b/embed_tokens");
    DW embed=upload(emb_lw);
    printf("Embed tokens: %d x %d\n",emb_lw.K,emb_lw.N);

    // lm_head (separate — not tied)
    auto lm_lw=load_w("weights_int8_qwen3_8b/lm_head");
    DW lm=upload(lm_lw);
    printf("lm_head: %d x %d\n",lm_lw.K,lm_lw.N);

    // Host copy of embed for lookup
    int8_t* host_embed=emb_lw.d.data();
    float* host_embed_sc=emb_lw.sc.data();
    int E_K=H, E_N=V;

    printf("All weights loaded.\n\n");
    cudaStream_t st; die(cudaStreamCreate(&st),"stream");
    srand((unsigned)time(nullptr));

    std::vector<float> h_embed(H);
    cudaMemset(d_kc,0,(size_t)NL*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vc,0,(size_t)NL*nkv*MAXSEQ*hd*4);

    printf("── Generating ──\n%s",prompt);
    fflush(stdout);

    std::vector<uint32_t> all_ids=input_ids;
    int gen_start=(int)input_ids.size();
    int total=gen_start+max_new;
    auto t_start=Clock::now();

    for(int step=0;step<total;++step){
        uint32_t tid=(step<gen_start)?input_ids[step]:all_ids.back();

        // ── Embedding: dequant single row from host INT8 → GPU FP32 ──
        {
            int kblocks=H/16;
            for(int b=0;b<kblocks;++b){
                float sc=host_embed_sc[(size_t)tid*kblocks+b];
                int off=(size_t)tid*H;
                for(int i=0;i<16;++i){
                    h_embed[b*16+i]=(float)host_embed[off+b*16+i]*sc;
                }
            }
        }
        die(cudaMemcpyAsync(d_x32,h_embed.data(),H*4,cudaMemcpyHostToDevice,st),"embed");

        // ══ 36-layer decode ══
        for(int l=0;l<NL;++l){
            die(cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st),"save_res");
            cudaError_t _prerr=cudaGetLastError(); if(_prerr!=cudaSuccess) { printf("PRE-ERROR: %s\n", cudaGetErrorString(_prerr)); } die(blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_in,H,eps,st),"rmsnorm_in");

            // Quantize normed input
            die(blackwell::kernels::quantize_int8(d_x_i8,d_x_s,d_xi_f,H,st),"quant_in");

            // QKV projections
            die(blackwell::kernels::gemv_int8_warp(d_Q,d_x_i8,d_x_s,W[l].q.d,W[l].q.sc,H,Q,st),"q");
            die(blackwell::kernels::gemv_int8_warp(d_K,d_x_i8,d_x_s,W[l].k.d,W[l].k.sc,H,KV,st),"k");
            die(blackwell::kernels::gemv_int8_warp(d_V,d_x_i8,d_x_s,W[l].v.d,W[l].v.sc,H,KV,st),"v");

            // Head norms + RoPE
            head_norm_kernel<<<nqh,128,0,st>>>(d_Q,W[l].qn,nqh,hd,eps);
            die(cudaGetLastError(),"qn");
            head_norm_kernel<<<nkv,128,0,st>>>(d_K,W[l].kn,nkv,hd,eps);
            die(cudaGetLastError(),"kn");
            apply_rope_kernel<<<nqh,hd/2,0,st>>>(d_Q,nqh,hd,step);
            die(cudaGetLastError(),"rq");
            apply_rope_kernel<<<nkv,hd/2,0,st>>>(d_K,nkv,hd,step);
            die(cudaGetLastError(),"rk");

            // KV cache + attention
            size_t kv_off=(size_t)l*nkv*MAXSEQ*hd;
            die(blackwell::kernels::update_kv_cache(d_kc+kv_off,d_vc+kv_off,d_K,d_V,0,step,nkv,hd,MAXSEQ,st),"kv");
            die(blackwell::kernels::attention_decode_batched_gqa(d_attn,d_Q,d_kc,d_vc,step,nqh,nkv,hd,MAXSEQ,1,
                (size_t)NL*nkv*MAXSEQ*hd,kv_off,st),"attn");

            // Wo projection
            die(blackwell::kernels::quantize_int8(d_attn_i8,d_attn_s,d_attn,Q,st),"quant_attn");
            die(blackwell::kernels::gemv_int8_warp(d_proj,d_attn_i8,d_attn_s,W[l].o.d,W[l].o.sc,Q,H,st),"o");

            // Residual + norm for next
            die(blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,st),"attn_res");

            die(cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st),"save_res2");
            die(blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_post,H,eps,st),"rmsnorm_post");
            die(blackwell::kernels::quantize_int8(d_x_i8,d_x_s,d_xi_f,H,st),"quant_mlp_in");

            // MLP gate + up
            die(blackwell::kernels::gemv_int8_warp(d_gate,d_x_i8,d_x_s,W[l].g.d,W[l].g.sc,H,I,st),"gate");
            die(blackwell::kernels::gemv_int8_warp(d_up,d_x_i8,d_x_s,W[l].u.d,W[l].u.sc,H,I,st),"up");

            // SwiGLU + quant (separate to ensure all scales are computed)
            blackwell::kernels::apply_swiglu(d_gate, d_gate, d_up, I, st);
            die(blackwell::kernels::quantize_int8(d_mlp_i8, d_mlp_s, d_gate, I, st),"quant_mlp");

            // Down projection
            die(blackwell::kernels::gemv_int8_warp(d_proj,d_mlp_i8,d_mlp_s,W[l].d.d,W[l].d.sc,I,H,st),"down");


            // MLP residual
            die(blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,st),"mlp_res");
        }

        // ── Final norm + lm_head + sampling ──
        if(step>=gen_start-1){
            die(blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,d_fn,H,eps,st),"fn");
            die(blackwell::kernels::quantize_int8(d_x_i8,d_x_s,d_xi_f,H,st),"quant_lm");
            // lm_head (separate weight, not tied to embed)
            die(blackwell::kernels::gemv_int8_warp(d_logits,d_x_i8,d_x_s,lm.d,lm.sc,H,V,st),"lm_head");

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
        cudaFree(w.q.d);cudaFree(w.q.sc);cudaFree(w.k.d);cudaFree(w.k.sc);
        cudaFree(w.v.d);cudaFree(w.v.sc);cudaFree(w.o.d);cudaFree(w.o.sc);
        cudaFree(w.g.d);cudaFree(w.g.sc);cudaFree(w.u.d);cudaFree(w.u.sc);
        cudaFree(w.d.d);cudaFree(w.d.sc);
        cudaFree(w.qn);cudaFree(w.kn);cudaFree(w.rn_in);cudaFree(w.rn_post);
    }
    return 0;
}