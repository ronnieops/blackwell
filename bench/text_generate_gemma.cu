// bench/text_generate_gemma.cu — Gemma 4 12B INT4 decode
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
    if(e!=cudaSuccess){fprintf(stderr,"FAIL %s: %s\n",m,cudaGetErrorString(e));exit(1);}
}

using Clock = std::chrono::high_resolution_clock;

const int H=3840, Q=3840, KV=512, I=15360;
const int nqh=16, nkv=8, hd=512, MAXSEQ=2048;
const float eps=1e-6f;
const int V=262144, NL=48;

struct DevW4 { int K,N; uint8_t* d; float* sc; };
struct LW4 { DevW4 q,k,v,o,g,u,d; float *rn_in,*rn_post; };


__global__ void rope_kernel(float* data, int n_heads, int head_dim, int pos, float theta) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_heads * (head_dim / 2)) return;
    int hh = h / (head_dim / 2);
    int dm = h % (head_dim / 2);
    float th = powf(theta, -2.0f * dm / head_dim) * pos;
    float co = cosf(th), si = sinf(th);
    float* x = data + hh * head_dim;
    float x0 = x[dm], x1 = x[dm + head_dim/2];
    x[dm] = x0 * co - x1 * si;
    x[dm + head_dim/2] = x0 * si + x1 * co;
}

__global__ void apply_gelu_gate(float* gate_out, const float* gate, const float* up, int I) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= I) return;
    float x = gate[i];
    float gelu = 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
    gate_out[i] = gelu * up[i];
}

static DevW4 upload_w4(const char* prefix) {
    char p[512]; snprintf(p,512,"%s.int4_t",prefix);
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"FAIL open %s\n",p);exit(1);}
    int h[5]; fread(h,4,5,f);
    DevW4 dw; dw.K=h[0]; dw.N=h[1];
    size_t ds=(size_t)h[0]*h[1]/2;
    uint8_t* td=new uint8_t[ds]; fread(td,1,ds,f); fclose(f);
    cudaMalloc(&dw.d,ds); cudaMemcpy(dw.d,td,ds,cudaMemcpyHostToDevice); delete[]td;
    snprintf(p,512,"%s.scale_t",prefix); f=fopen(p,"rb"); fread(h,4,5,f);
    size_t ss=(size_t)h[3]*h[4];
    float* ts=new float[ss]; fread(ts,4,ss,f); fclose(f);
    cudaMalloc(&dw.sc,ss*4); cudaMemcpy(dw.sc,ts,ss*4,cudaMemcpyHostToDevice); delete[]ts;
    return dw;
}

static bool file_exists(const char* path) {
    FILE* f=fopen(path,"rb"); if(f){fclose(f);return true;} return false;
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
    const char* prompt = argc>1?argv[1]:"Once upon a time";
    int max_new = argc>2?atoi(argv[2]):50;
    const char* wdir = argc>3?argv[3]:"/tmp/gemma_test";

    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    printf("# Text Generation — Gemma 4 12B INT4\n");
    printf("  Device: %s\n", P.name);
    printf("  Prompt: \"%s\"\n", prompt);

    blackwell::BpeTokenizer tokenizer;
    char tp[512]; snprintf(tp,512,"%s/tokenizer_data.bin",wdir);
    if(tokenizer.load(tp)!=0){fprintf(stderr,"FAIL: no tokenizer\n");return 1;}

    // Accept --tokens id1,id2,... for pre-tokenized input
    std::vector<uint32_t> ids;
    for(int i=4;i<argc;i++){
        if(strcmp(argv[i],"--tokens")==0 && i+1<argc){
            const char* tstr=argv[++i];
            const char* p=tstr;
            while(*p){
                while(*p && (*p<'0'||*p>'9')&&*p!='-')p++;
                if(!*p)break;
                long v=strtol(p,(char**)&p,10);
                ids.push_back((uint32_t)v);
            }
        }
    }
    if(ids.empty()) ids=tokenizer.encode(prompt);

    cudaStream_t st; cudaStreamCreate(&st);
    #define AL(p,n) die(cudaMalloc(&(p),(n)),"malloc")

    // Load weights
    printf("Loading %d-layer INT4 model...\n",NL); fflush(stdout);
    std::vector<LW4> W(NL); char p_[512];
    for(int l=0;l<NL;++l){
        snprintf(p_,512,"%s/%d_self_attn.q_proj",wdir,l); W[l].q=upload_w4(p_);
        snprintf(p_,512,"%s/%d_self_attn.k_proj",wdir,l); W[l].k=upload_w4(p_);
        snprintf(p_,512,"%s/%d_self_attn.v_proj",wdir,l);
        if(file_exists(p_)) W[l].v=upload_w4(p_); else W[l].v={0,0,nullptr,nullptr};
        snprintf(p_,512,"%s/%d_self_attn.o_proj",wdir,l); W[l].o=upload_w4(p_);
        snprintf(p_,512,"%s/%d_mlp.gate_proj",wdir,l); W[l].g=upload_w4(p_);
        snprintf(p_,512,"%s/%d_mlp.up_proj",wdir,l);   W[l].u=upload_w4(p_);
        snprintf(p_,512,"%s/%d_mlp.down_proj",wdir,l); W[l].d=upload_w4(p_);
        if((l+1)%8==0) printf("  layer %d/%d\n",l+1,NL);
    }
    printf("  layer %d/%d\n",NL,NL);

    // Norms (no QK head norms for Gemma)
    for(int l=0;l<NL;++l){
        float* w=(float*)malloc(H*4);
        snprintf(p_,512,"%s/%d_input_layernorm.f32",wdir,l);
        {FILE*f=fopen(p_,"rb");if(!f){for(int i=0;i<H;i++)w[i]=1.0f;}else{fread(w,4,H,f);fclose(f);}}
        AL(W[l].rn_in,H*4); cudaMemcpy(W[l].rn_in,w,H*4,cudaMemcpyHostToDevice);
        snprintf(p_,512,"%s/%d_post_attention_layernorm.f32",wdir,l);
        {FILE*f=fopen(p_,"rb");if(!f){for(int i=0;i<H;i++)w[i]=1.0f;}else{fread(w,4,H,f);fclose(f);}}
        AL(W[l].rn_post,H*4); cudaMemcpy(W[l].rn_post,w,H*4,cudaMemcpyHostToDevice);
        free(w);
    }

    // Final norm
    float* d_fn;
    {float*w=new float[H];snprintf(p_,512,"%s/final_norm.f32",wdir);
     FILE*f=fopen(p_,"rb");fread(w,4,H,f);fclose(f);
     AL(d_fn,H*4);cudaMemcpy(d_fn,w,H*4,cudaMemcpyHostToDevice);delete[]w;}

    // Embed + lm_head
    DevW4 embed_w, lm_head_w;
    {snprintf(p_,512,"%s/embed_tokens",wdir);embed_w=upload_w4(p_);
     snprintf(p_,512,"%s/lm_head",wdir);lm_head_w=upload_w4(p_);}
    printf("Embed: %dx%d, lm_head: %dx%d\n",embed_w.K,embed_w.N,lm_head_w.K,lm_head_w.N);

    // Host embed
    uint8_t* host_ed=new uint8_t[(size_t)embed_w.K*embed_w.N/2];
    float* host_es=new float[embed_w.N*(embed_w.K/16)];
    {int h[5];snprintf(p_,512,"%s/embed_tokens.int4_t",wdir);
     FILE*f=fopen(p_,"rb");fread(h,4,5,f);fclose(f);
     size_t ds=(size_t)h[0]*h[1]/2;FILE*f2=fopen(p_,"rb");fseek(f2,20,SEEK_SET);fread(host_ed,1,ds,f2);fclose(f2);
     snprintf(p_,512,"%s/embed_tokens.scale_t",wdir);
     f2=fopen(p_,"rb");fread(h,4,5,f2);fclose(f2);size_t ss=h[3]*h[4];
     FILE*f3=fopen(p_,"rb");fseek(f3,20,SEEK_SET);fread(host_es,4,ss,f3);fclose(f3);}

    // Buffers
    float *d_x32,*d_xi_f,*d_res,*d_Q,*d_K,*d_V,*d_attn,*d_proj,*d_gate,*d_up,*d_kc,*d_vc,*d_logits;
    uint8_t *d_x_i4,*d_attn_i4,*d_mlp_i4; float *d_x_i4_sc,*d_attn_i4_sc,*d_mlp_i4_sc;
    int* d_next_id;
    float* d_v_pair;
    AL(d_x32,H*4);AL(d_xi_f,H*4);AL(d_res,H*4);
    AL(d_x_i4,H/2);AL(d_x_i4_sc,(H/16)*4);
    AL(d_Q,Q*4);AL(d_K,KV*4);AL(d_V,KV*4);AL(d_attn,Q*4);
    AL(d_attn_i4,Q/2);AL(d_attn_i4_sc,(Q/16)*4);
    AL(d_proj,std::max({H,Q,I})*4);AL(d_gate,I*4);AL(d_up,I*4);
    AL(d_mlp_i4,I/2);AL(d_mlp_i4_sc,(I/16)*4);
    AL(d_kc,(size_t)NL*nkv*MAXSEQ*hd*4);
    AL(d_vc,(size_t)NL*nkv*MAXSEQ*hd*4);
    AL(d_logits,V*4);AL(d_next_id,4);
    AL(d_v_pair,(size_t)8*KV*4);
    {
        float iv7=1.f/7.f;
        std::vector<float> t(H/16,iv7);cudaMemcpy(d_x_i4_sc,t.data(),(H/16)*4,cudaMemcpyHostToDevice);
        std::vector<float> t2(Q/16,iv7);cudaMemcpy(d_attn_i4_sc,t2.data(),(Q/16)*4,cudaMemcpyHostToDevice);
        std::vector<float> t3(I/16,iv7);cudaMemcpy(d_mlp_i4_sc,t3.data(),(I/16)*4,cudaMemcpyHostToDevice);
    }

    cudaMemset(d_kc,0,(size_t)NL*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vc,0,(size_t)NL*nkv*MAXSEQ*hd*4);

    // ── Tokenize ──
    int gen_start=(int)ids.size();
    printf("Input: %d tokens\n",gen_start);fflush(stdout);
    printf("── Generating ──\n");fflush(stdout);

    // ── Prefill + Decode ──
    std::vector<uint32_t> all_ids=ids;
    auto t_start=Clock::now();
    int total=gen_start+max_new;

    for(int step=0;step<total;++step){
        int tid=(step<gen_start)?ids[step]:(int)all_ids.back();
        std::vector<float> h_embed(H);
        dequant_embed_row(h_embed.data(),tid,host_ed,host_es,H);
        die(cudaMemcpyAsync(d_x32,h_embed.data(),H*4,cudaMemcpyHostToDevice,st),"embed");

        for(int l=0;l<NL;++l){
            die(cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st),"save_res");
            die(blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_in,H,eps,st),"rmsnorm_in");
            die(blackwell::kernels::quantize_int4_batched(d_x_i4,d_x_i4_sc,d_xi_f,H,1,st),"quant_in");
            die(blackwell::kernels::gemv_int4_batched(d_Q,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].q.d,W[l].q.sc,H,Q,1,st),"q");
            die(blackwell::kernels::gemv_int4_batched(d_K,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].k.d,W[l].k.sc,H,KV,1,st),"k");
            if(W[l].v.d){
                die(blackwell::kernels::gemv_int4_batched(d_V,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].v.d,W[l].v.sc,H,KV,1,st),"v");
                int pair_idx = -1;
                for (int pi : {0, 6, 12, 18, 24, 30, 36, 42}) {
                    if (l == pi) { pair_idx = pi / 6; break; }
                }
                if (pair_idx >= 0) {
                    cudaMemcpyAsync(d_v_pair + pair_idx * KV, d_V, KV*4, cudaMemcpyDeviceToDevice, st);
                }
            } else {
                int pair_idx = l / 6;
                cudaMemcpyAsync(d_V, d_v_pair + pair_idx * KV, KV*4, cudaMemcpyDeviceToDevice, st);
            }
            // RoPE: SWA (l%%6==5) uses 1e4, FA uses 1e6
            float rtheta = (l % 6 == 5) ? 10000.0f : 1000000.0f;
            rope_kernel<<<nqh,hd/2,0,st>>>(d_Q,nqh,hd,step,rtheta);
            rope_kernel<<<nkv,hd/2,0,st>>>(d_K,nkv,hd,step,rtheta);
            size_t kv_off=(size_t)l*nkv*MAXSEQ*hd;
            die(blackwell::kernels::update_kv_cache(d_kc+kv_off,d_vc+kv_off,d_K,d_V,0,step,nkv,hd,MAXSEQ,st),"kv");
            die(blackwell::kernels::attention_decode_batched_gqa(d_attn,d_Q,d_kc,d_vc,step,nqh,nkv,hd,MAXSEQ,1,
                (size_t)NL*nkv*MAXSEQ*hd,kv_off,st),"attn");
            die(blackwell::kernels::quantize_int4_batched(d_attn_i4,d_attn_i4_sc,d_attn,Q,1,st),"quant_attn");
            die(blackwell::kernels::gemv_int4_batched(d_proj,(const uint8_t*)d_attn_i4,d_attn_i4_sc,W[l].o.d,W[l].o.sc,Q,H,1,st),"o");
            die(blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,st),"attn_res");
            die(cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st),"save_res2");
            die(blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_post,H,eps,st),"rmsnorm_post");
            die(blackwell::kernels::quantize_int4_batched(d_x_i4,d_x_i4_sc,d_xi_f,H,1,st),"quant_mlp_in");
            die(blackwell::kernels::gemv_int4_batched(d_gate,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].g.d,W[l].g.sc,H,I,1,st),"gate");
            die(blackwell::kernels::gemv_int4_batched(d_up,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].u.d,W[l].u.sc,H,I,1,st),"up");
            // GeGLU instead of SwiGLU
            int gt=256,gb=(I+255)/256;
            apply_gelu_gate<<<gb,gt,0,st>>>(d_gate,d_gate,d_up,I);
            die(blackwell::kernels::quantize_int4_batched(d_mlp_i4,d_mlp_i4_sc,d_gate,I,1,st),"quant_mlp");
            die(blackwell::kernels::gemv_int4_batched(d_proj,(const uint8_t*)d_mlp_i4,d_mlp_i4_sc,W[l].d.d,W[l].d.sc,I,H,1,st),"down");
            die(blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,st),"mlp_res");
        }

        if(step>=gen_start){
            die(blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,d_fn,H,eps,st),"fn");
            die(blackwell::kernels::quantize_int4_batched(d_x_i4,d_x_i4_sc,d_xi_f,H,1,st),"quant_lm");
            die(blackwell::kernels::gemv_int4_batched(d_logits,(const uint8_t*)d_x_i4,d_x_i4_sc,lm_head_w.d,lm_head_w.sc,H,V,1,st),"lm_head");
            die(blackwell::kernels::sample_gpu(d_logits,V,0.0f,0,d_next_id,0xdeadbeefLL,step,st),"sample");
            int next_id; die(cudaMemcpy(&next_id,d_next_id,4,cudaMemcpyDeviceToHost),"copy");
            all_ids.push_back(next_id);
            printf("%d ",next_id);fflush(stdout);
            if(next_id==1||next_id==2) break;  // Gemma eos=1, bos=2
        }
    }

    auto t_end=Clock::now();
    double ms=std::chrono::duration<double,std::milli>(t_end-t_start).count();
    int gen=(int)all_ids.size()-gen_start;
    printf("\n\n── Stats ──\n");
    printf("  Input: %d  Gen: %d\n",gen_start,gen);
    printf("  Time: %.1f ms  Speed: %.1f ms/tok = %.0f t/s\n",ms,ms/gen,1000.0*gen/ms);

    // Cleanup
    for(auto& w:W){
        cudaFree(w.q.d);cudaFree(w.q.sc);cudaFree(w.k.d);cudaFree(w.k.sc);
        if(w.v.d){cudaFree(w.v.d);cudaFree(w.v.sc);}
        cudaFree(w.o.d);cudaFree(w.o.sc);
        cudaFree(w.g.d);cudaFree(w.g.sc);
        cudaFree(w.u.d);cudaFree(w.u.sc);
        cudaFree(w.d.d);cudaFree(w.d.sc);
        cudaFree(w.rn_in);cudaFree(w.rn_post);
    }
    delete[]host_ed;delete[]host_es;
    return 0;
}
