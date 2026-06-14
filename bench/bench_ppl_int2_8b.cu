// bench/bench_ppl_int2_8b.cu — PPL benchmark for INT2 8B Qwen3
//
// INT4 activations × INT2 packed weights. Same pipeline as INT4 PPL,
// with gemv_int2_batched replacing gemv_int4_warp.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <string>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <chrono>
#include "blackwell/kernels.h"
#include "blackwell/bpe_tokenizer.h"

static void die(cudaError_t e, const char* m) {
    if(e!=cudaSuccess){fprintf(stderr,"FAIL %s: %s\n",m,cudaGetErrorString(e));exit(1);}
}

const int H=4096, Q=4096, KV=1024, I=12288;
const int nqh=32, nkv=8, hd=128, MAXSEQ=4096;
const float eps=1e-6f;
const int V=151936;
const int NL=36;

static const char* TEST_CORPUS = 
    "The capital of Austria is Vienna . The official language is German . "
    "France is a country in Western Europe . Paris is the capital of France . "
    "The weather today is sunny and warm . The city has many museums and parks . "
    "The university is located in the downtown area . Students study hard for exams . "
    "The restaurant serves delicious food and drinks . Service is excellent and fast . "
    "The book is interesting and well written . The story takes place in ancient times . "
    "Music plays a vital role in human culture . People gather to enjoy concerts together .";

struct DevW2 { int K, N; uint8_t* d; float* sc; };
static DevW2 upload_w2(const char* prefix) {
    char p[256]; snprintf(p,256,"%s.int2_t",prefix);
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"FAIL open %s\n",p);exit(1);}
    int h[5]; fread(h,4,5,f);
    DevW2 dw; dw.K=h[0]; dw.N=h[1];
    size_t ds=(size_t)h[0]*h[1]/4;
    uint8_t* td=new uint8_t[ds]; fread(td,1,ds,f); fclose(f);
    cudaMalloc(&dw.d,ds); cudaMemcpy(dw.d,td,ds,cudaMemcpyHostToDevice); delete[] td;
    snprintf(p,256,"%s.scale_t",prefix); f=fopen(p,"rb"); fread(h,4,5,f);
    size_t ss=(size_t)h[3]*h[4];
    float* ts=new float[ss]; fread(ts,4,ss,f); fclose(f);
    cudaMalloc(&dw.sc,ss*4); cudaMemcpy(dw.sc,ts,ss*4,cudaMemcpyHostToDevice); delete[] ts;
    return dw;
}

struct LW2 { DevW2 q,k,v,o,g,u,d; float* qn,*kn,*rn_in,*rn_post; };

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

// INT2 embed dequant: 4 vals/byte, offset-binary (val-2)
static void dequant_embed_int2(float* out, int token, const uint8_t* host_w, const float* host_sc, int K) {
    int kblocks=K/16;
    for(int b=0;b<kblocks;++b){
        float sc=host_sc[token*kblocks+b];
        for(int i=0;i<16;++i){
            size_t byte_idx=(size_t)token*K/4+(size_t)b*4+i/4;
            uint8_t byte=host_w[byte_idx];
            int shift=(i%4)*2;
            int val=((byte>>shift)&0x3)-2;
            out[b*16+i]=(float)val*sc;
        }
    }
}

// Logprob kernel (same as INT4 PPL)
__global__ void logprob_kernel(const float* logits, int V, int correct_id, float* out_logp) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    float lmax = -1e30f;
    for (int i = tid; i < V; i += blockDim.x) lmax = fmaxf(lmax, logits[i]);
    smem[tid] = lmax; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid+s]);
        __syncthreads();
    }
    float gmax = smem[0];
    float sum = 0.0f;
    for (int i = tid; i < V; i += blockDim.x) {
        float v = logits[i] - gmax;
        if (v > -20.0f) sum += expf(v);
    }
    smem[tid] = sum; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid+s];
        __syncthreads();
    }
    float total = smem[0];
    if (tid == 0) {
        *out_logp = logits[correct_id] - gmax - logf(fmaxf(total, 1e-20f));
    }
}

int main(int argc, char** argv) {
    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    fprintf(stderr,"# INT2 8B PPL — %s\n", P.name);

    blackwell::BpeTokenizer tok;
    if(tok.load("tokenizer_data.bin")!=0){ fprintf(stderr,"FAIL: no tokenizer_data.bin\n"); return 1; }

    std::vector<uint32_t> ids;
    auto toks = tok.encode(TEST_CORPUS);
    ids.insert(ids.end(), toks.begin(), toks.end());
    fprintf(stderr,"Corpus: %zu tokens\n", ids.size());

    float *d_x32,*d_xi_f,*d_res,*d_Q,*d_K,*d_V,*d_attn,*d_proj,*d_gate,*d_up;
    uint8_t *d_x_i4; float *d_x_i4_sc;
    uint8_t *d_attn_i4; float *d_attn_i4_sc;
    uint8_t *d_mlp_i4; float *d_mlp_i4_sc;
    float *d_fn,*d_kc,*d_vc,*d_logits,*d_logp;

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
    AL(d_logits,V*4); AL(d_logp,4);
    #undef AL

    float iv7=1.f/7.f;
    { std::vector<float> tmp(H/16,iv7); cudaMemcpy(d_x_i4_sc,tmp.data(),(H/16)*4,cudaMemcpyHostToDevice); }
    { std::vector<float> tmp(Q/16,iv7); cudaMemcpy(d_attn_i4_sc,tmp.data(),(Q/16)*4,cudaMemcpyHostToDevice); }
    { std::vector<float> tmp(I/16,iv7); cudaMemcpy(d_mlp_i4_sc,tmp.data(),(I/16)*4,cudaMemcpyHostToDevice); }

    fprintf(stderr,"Loading INT2 weights...\n");
    std::vector<LW2> W(NL);
    char p[256];
    for(int l=0;l<NL;++l){
        snprintf(p,256,"weights_int2_qwen3_8b/%d_self_attn.q_proj",l); W[l].q=upload_w2(p);
        snprintf(p,256,"weights_int2_qwen3_8b/%d_self_attn.k_proj",l); W[l].k=upload_w2(p);
        snprintf(p,256,"weights_int2_qwen3_8b/%d_self_attn.v_proj",l); W[l].v=upload_w2(p);
        snprintf(p,256,"weights_int2_qwen3_8b/%d_self_attn.o_proj",l); W[l].o=upload_w2(p);
        snprintf(p,256,"weights_int2_qwen3_8b/%d_mlp.gate_proj",l); W[l].g=upload_w2(p);
        snprintf(p,256,"weights_int2_qwen3_8b/%d_mlp.up_proj",l); W[l].u=upload_w2(p);
        snprintf(p,256,"weights_int2_qwen3_8b/%d_mlp.down_proj",l); W[l].d=upload_w2(p);
    }
    float* qk_h=(float*)malloc(NL*2*hd*4);
    {FILE*f=fopen("weights_int2_qwen3_8b/qk_norms.f32","rb");(void)fread(qk_h,4,NL*2*hd,f);fclose(f);}
    for(int l=0;l<NL;++l){
        cudaMalloc(&W[l].qn,hd*4);cudaMemcpy(W[l].qn,qk_h+l*2*hd,hd*4,cudaMemcpyHostToDevice);
        cudaMalloc(&W[l].kn,hd*4);cudaMemcpy(W[l].kn,qk_h+l*2*hd+hd,hd*4,cudaMemcpyHostToDevice);
    }free(qk_h);
    for(int l=0;l<NL;++l){
        float* w=(float*)malloc(H*4);
        snprintf(p,256,"weights_int2_qwen3_8b/%d_input_layernorm.f32",l);
        {FILE*f=fopen(p,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&W[l].rn_in,H*4);cudaMemcpy(W[l].rn_in,w,H*4,cudaMemcpyHostToDevice);
        snprintf(p,256,"weights_int2_qwen3_8b/%d_post_attention_layernorm.f32",l);
        {FILE*f=fopen(p,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&W[l].rn_post,H*4);cudaMemcpy(W[l].rn_post,w,H*4,cudaMemcpyHostToDevice);
        free(w);
    }
    {float*w=(float*)malloc(H*4);
    FILE*f=fopen("weights_int2_qwen3_8b/final_norm.f32","rb");(void)fread(w,4,H,f);fclose(f);
    cudaMemcpy(d_fn,w,H*4,cudaMemcpyHostToDevice);free(w);}

    DevW2 lm_head_w=upload_w2("weights_int2_qwen3_8b/lm_head");
    uint8_t* host_embed_d=new uint8_t[(size_t)H*V/4];
    float* host_embed_sc=new float[V*(H/16)];
    {FILE*f=fopen("weights_int2_qwen3_8b/embed_tokens.int2_t","rb");int h[5];fread(h,4,5,f);
     fread(host_embed_d,1,(size_t)h[0]*h[1]/4,f);fclose(f);
     f=fopen("weights_int2_qwen3_8b/embed_tokens.scale_t","rb");fread(h,4,5,f);
     fread(host_embed_sc,4,(size_t)h[3]*h[4],f);fclose(f);}

    cudaStream_t st; die(cudaStreamCreate(&st),"stream");
    std::vector<float> h_embed(H);

    cudaMemset(d_kc,0,(size_t)NL*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vc,0,(size_t)NL*nkv*MAXSEQ*hd*4);

    double total_logp=0.0; int valid=0;
    auto t0=std::chrono::high_resolution_clock::now();

    for(int step=0;step<(int)ids.size()-1;++step){
        uint32_t tid=ids[step];
        dequant_embed_int2(h_embed.data(),tid,host_embed_d,host_embed_sc,H);
        die(cudaMemcpyAsync(d_x32,h_embed.data(),H*4,cudaMemcpyHostToDevice,st),"embed");

        for(int l=0;l<NL;++l){
            die(cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st),"save_res");
            die(blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_in,H,eps,st),"rn_in");
            die(blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st),"q_in");
            die(blackwell::kernels::gemv_int2_batched(d_Q,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].q.d,W[l].q.sc,H,Q,1,st),"q_proj");
            die(blackwell::kernels::gemv_int2_batched(d_K,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].k.d,W[l].k.sc,H,KV,1,st),"k_proj");
            die(blackwell::kernels::gemv_int2_batched(d_V,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].v.d,W[l].v.sc,H,KV,1,st),"v_proj");
            head_norm_kernel<<<nqh,128,0,st>>>(d_Q,W[l].qn,nqh,hd,eps); die(cudaGetLastError(),"hn_q");
            head_norm_kernel<<<nkv,128,0,st>>>(d_K,W[l].kn,nkv,hd,eps); die(cudaGetLastError(),"hn_k");
            apply_rope_kernel<<<nqh,hd/2,0,st>>>(d_Q,nqh,hd,step); die(cudaGetLastError(),"rp_q");
            apply_rope_kernel<<<nkv,hd/2,0,st>>>(d_K,nkv,hd,step); die(cudaGetLastError(),"rp_k");
            size_t kv_off=(size_t)l*nkv*MAXSEQ*hd;
            die(blackwell::kernels::update_kv_cache(d_kc+kv_off,d_vc+kv_off,d_K,d_V,0,step,nkv,hd,MAXSEQ,st),"kv");
            die(blackwell::kernels::attention_decode_batched_gqa(d_attn,d_Q,d_kc,d_vc,step,nqh,nkv,hd,MAXSEQ,1,
                (size_t)NL*nkv*MAXSEQ*hd,kv_off,st),"attn");
            die(blackwell::kernels::quantize_int4(d_attn_i4,d_attn_i4_sc,d_attn,Q,st),"q_attn");
            die(blackwell::kernels::gemv_int2_batched(d_proj,(const uint8_t*)d_attn_i4,d_attn_i4_sc,W[l].o.d,W[l].o.sc,Q,H,1,st),"o_proj");
            die(blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,st),"res1");
            die(cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st),"save_res2");
            die(blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_post,H,eps,st),"rn_post");
            die(blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st),"q_mlp");
            die(blackwell::kernels::gemv_int2_batched(d_gate,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].g.d,W[l].g.sc,H,I,1,st),"gate");
            die(blackwell::kernels::gemv_int2_batched(d_up,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].u.d,W[l].u.sc,H,I,1,st),"up");
            blackwell::kernels::apply_swiglu(d_gate,d_gate,d_up,I,st);
            die(blackwell::kernels::quantize_int4(d_mlp_i4,d_mlp_i4_sc,d_gate,I,st),"q_mlp2");
            die(blackwell::kernels::gemv_int2_batched(d_proj,(const uint8_t*)d_mlp_i4,d_mlp_i4_sc,W[l].d.d,W[l].d.sc,I,H,1,st),"down");
            die(blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,st),"res2");
        }

        die(blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,d_fn,H,eps,st),"fn");
        die(blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st),"q_fn");
        die(blackwell::kernels::gemv_int2_batched(d_logits,(const uint8_t*)d_x_i4,d_x_i4_sc,lm_head_w.d,lm_head_w.sc,H,V,1,st),"lm");

        int correct = ids[step+1];
        if(step<2) {
            int dids[3] = {correct, 0, 264};
            float dvs[3];
            for(int di=0;di<3;di++) cudaMemcpy(&dvs[di], d_logits+dids[di], 4, cudaMemcpyDeviceToHost);
            fprintf(stderr,"  step%d logits[0]=%.1f tok264=%.1f correct=%d logits[correct]=%.1f\n",step,dvs[1],dvs[2],correct,dvs[0]);
        }
        logprob_kernel<<<1,256,256*4,st>>>(d_logits,V,correct,d_logp);
        die(cudaGetLastError(),"logprob");
        float h_logp; die(cudaMemcpy(&h_logp,d_logp,4,cudaMemcpyDeviceToHost),"logp_cp");
        total_logp += h_logp; valid++;

        if((step+1)%20==0) fprintf(stderr,"  step %d/%d avg_logp=%.3f\r",step+1,(int)ids.size()-1,(float)total_logp/valid);
    }

    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    double ppl=exp(-total_logp/valid);

    fprintf(stderr,"\n── PPL Results ──\n");
    fprintf(stderr,"  Corpus: %d tokens\n",valid);
    fprintf(stderr,"  Log P sum: %.4f\n",total_logp);
    fprintf(stderr,"  PPL: %.2f\n",ppl);
    fprintf(stderr,"  Time: %.0f ms (%.1f ms/token)\n",ms,ms/valid);

    return 0;
}