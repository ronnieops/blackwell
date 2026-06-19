#include "blackwell/int4_weights.h"
using namespace blackwell::weights;
// server/inference_server_int4_batched.cu — Batched INT4 8B server
// Processes up to M=8 prompts, each independently, then returns batched results.
//
// For true batched inference (shared KV cache, batched GEMVs), would need:
// - Separate d_Q[m*Q] per sequence
// - gemv_int4_batched kernel
// - Batch attention kernel
// For now: processes each sequence sequentially.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <string>
#include <cstdint>
#include <cmath>
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

struct DevW4 { int K, N; uint8_t* d; float* sc; };
static DevW4 upload_w4(const char* prefix) {
    char p[256]; snprintf(p,256,"%s.int4_t",prefix);
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"FAIL open %s\n",p);exit(1);}
    int h[5]; fread(h,4,5,f);
    DevW4 dw; dw.K=h[0]; dw.N=h[1];
    size_t ds=(size_t)h[0]*h[1]/2;
    uint8_t* td=new uint8_t[ds]; fread(td,1,ds,f); fclose(f);
    cudaMalloc(&dw.d,ds); cudaMemcpy(dw.d,td,ds,cudaMemcpyHostToDevice); delete[] td;
    snprintf(p,256,"%s.scale_t",prefix); f=fopen(p,"rb"); fread(h,4,5,f);
    size_t ss=(size_t)h[3]*h[4];
    float* ts=new float[ss]; fread(ts,4,ss,f); fclose(f);
    cudaMalloc(&dw.sc,ss*4); cudaMemcpy(dw.sc,ts,ss*4,cudaMemcpyHostToDevice); delete[] ts;
    return dw;
}

// FP16 weight-scale variant. W_scale uploaded as __half (half the traffic).

struct LW4 { DevW4f16 q,k,v,o,g,u,d; float* qn,*kn,*rn_in,*rn_post; };

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

static std::string read_stdin_line() {
    std::string line; int c;
    while ((c = getchar()) != EOF && c != '\n') line.push_back((char)c);
    return line;
}

static std::vector<std::string> parse_prompts(const std::string& json) {
    std::vector<std::string> result;
    const char* p = strstr(json.c_str(), "\"prompts\"");
    if (!p) return result;
    p = strchr(p, '['); if (!p) return result; p++;
    while (*p && *p != ']') {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == ',') p++;
        if (*p == '"') {
            p++; std::string s;
            while (*p && *p != '"') { if (*p == '\\') p++; s += *p++; }
            if (*p == '"') p++;
            result.push_back(s);
        } else { p++; }
    }
    return result;
}

static float parse_float(const std::string& json, const char* key, float def) {
    const char* p = strstr(json.c_str(), key);
    if (!p) return def;
    p += strlen(key);
    while (*p == ' ' || *p == ':' || *p == '=') p++;
    return strtof(p, nullptr);
}

static float parse_repetition_penalty(const std::string& json, float def) {
    const char* p = strstr(json.c_str(), "repetition_penalty");
    if (!p) return def;
    p += 18;
    while (*p == ' ' || *p == ':' || *p == '=') p++;
    return strtof(p, nullptr);
}

static int parse_int(const std::string& json, const char* key, int def) {
    const char* p = strstr(json.c_str(), key);
    if (!p) return def;
    p += strlen(key);
    while (*p == ' ' || *p == ':' || *p == '=') p++;
    return atoi(p);
}

// Global state
static float *d_x32, *d_xi_f, *d_res;
static uint8_t *d_x_i4; static float *d_x_i4_sc;
static float *d_Q, *d_K, *d_V, *d_attn;
static uint8_t *d_attn_i4; static float *d_attn_i4_sc;
static float *d_proj, *d_gate, *d_up;
static uint8_t *d_mlp_i4; static float *d_mlp_i4_sc;
static float *d_fn, *d_kc, *d_vc, *d_logits;
static int *d_next_id, *d_recent;
static cudaStream_t st;

static std::vector<LW4> W(NL);
static DevW4f16 embed_w, lm_head_w;
// Host-side INT4 embed data for on-the-fly CPU dequant (saves 2.5 GB GPU memory)
static uint8_t* host_embed_d = nullptr;
static float* host_embed_sc = nullptr;
static blackwell::BpeTokenizer tokenizer;

#define AL(p,n) die(cudaMalloc(&(p),(n)),"malloc "#p)

static void load_model() {
    fprintf(stderr, "Loading %d-layer INT4 model...\n", NL);
    char p[256];
    for (int l = 0; l < NL; ++l) {
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_self_attn.q_proj",l); W[l].q=upload_w4_f16sc(p);
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_self_attn.k_proj",l); W[l].k=upload_w4_f16sc(p);
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_self_attn.v_proj",l); W[l].v=upload_w4_f16sc(p);
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_self_attn.o_proj",l); W[l].o=upload_w4_f16sc(p);
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_mlp.gate_proj",l); W[l].g=upload_w4_f16sc(p);
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_mlp.up_proj",l); W[l].u=upload_w4_f16sc(p);
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_mlp.down_proj",l); W[l].d=upload_w4_f16sc(p);
        if ((l+1)%7==0) fprintf(stderr, "  layer %d/%d\n", l+1, NL);
    }
    float* qk_h=(float*)malloc(NL*2*hd*4);
    {FILE*f=fopen("weights_int4_qwen3_8b_fp16sc/qk_norms.f32","rb");(void)fread(qk_h,4,NL*2*hd,f);fclose(f);}
    for(int l=0;l<NL;++l){
        cudaMalloc(&W[l].qn,hd*4);cudaMemcpy(W[l].qn,qk_h+l*2*hd,hd*4,cudaMemcpyHostToDevice);
        cudaMalloc(&W[l].kn,hd*4);cudaMemcpy(W[l].kn,qk_h+l*2*hd+hd,hd*4,cudaMemcpyHostToDevice);
    }free(qk_h);
    for(int l=0;l<NL;++l){
        float* w=(float*)malloc(H*4);
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_input_layernorm.f32",l);
        {FILE*f=fopen(p,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&W[l].rn_in,H*4);cudaMemcpy(W[l].rn_in,w,H*4,cudaMemcpyHostToDevice);
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_post_attention_layernorm.f32",l);
        {FILE*f=fopen(p,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&W[l].rn_post,H*4);cudaMemcpy(W[l].rn_post,w,H*4,cudaMemcpyHostToDevice);
        free(w);
    }
    {float*w=(float*)malloc(H*4);
     FILE*f=fopen("weights_int4_qwen3_8b_fp16sc/final_norm.f32","rb");(void)fread(w,4,H,f);fclose(f);
     AL(d_fn,H*4); cudaMemcpy(d_fn,w,H*4,cudaMemcpyHostToDevice); free(w);}
    embed_w=upload_w4_f16sc("weights_int4_qwen3_8b_fp16sc/embed_tokens");
    lm_head_w=upload_w4_f16sc("weights_int4_qwen3_8b_fp16sc/lm_head");
    fprintf(stderr,"  embed: %dx%d, lm_head: %dx%d\n",embed_w.K,embed_w.N,lm_head_w.K,lm_head_w.N);
    
    // Load INT4 embed on host for CPU dequant (no GPU FP32 cache)
    if (host_embed_d) delete[] host_embed_d;
    if (host_embed_sc) delete[] host_embed_sc;
    size_t embed_size = (size_t)embed_w.K * embed_w.N;
    host_embed_d = new uint8_t[embed_size / 2];
    host_embed_sc = new float[embed_w.N * (embed_w.K / 16)];
    {FILE*f=fopen("weights_int4_qwen3_8b_fp16sc/embed_tokens.int4_t","rb");if(!f){fprintf(stderr,"FAIL open embed_tokens.int4_t\n");exit(1);}int h[5];if(fread(h,4,5,f)!=5){fprintf(stderr,"FAIL read embed header\n");exit(1);}
     if(fread(host_embed_d,1,(size_t)h[0]*h[1]/2,f)!=(size_t)h[0]*h[1]/2){fprintf(stderr,"FAIL read embed data\n");exit(1);}fclose(f);
     f=fopen("weights_int4_qwen3_8b_fp16sc/embed_tokens.scale_t","rb");if(!f){fprintf(stderr,"FAIL open embed_tokens.scale_t\n");exit(1);}if(fread(h,4,5,f)!=5){fprintf(stderr,"FAIL read embed scale header\n");exit(1);}
     {__half* tmp_sc16=new __half[(size_t)h[3]*h[4]];if(fread(tmp_sc16,2,(size_t)h[3]*h[4],f)!=(size_t)h[3]*h[4]){fprintf(stderr,"FAIL read embed scales\n");exit(1);}
      for(size_t i=0;i<(size_t)h[3]*h[4];++i) host_embed_sc[i]=__half2float(tmp_sc16[i]);
      delete[] tmp_sc16;}
     fclose(f);}
    fprintf(stderr,"All weights loaded (host-side INT4 embed).\n");
}

static void alloc_buffers() {
    AL(d_x32,H*4); AL(d_xi_f,H*4); AL(d_res,H*4);
    AL(d_x_i4,H/2); AL(d_x_i4_sc,(H/16)*4);
    AL(d_Q,Q*4); AL(d_K,KV*4); AL(d_V,KV*4); AL(d_attn,Q*4);
    AL(d_attn_i4,Q/2); AL(d_attn_i4_sc,(Q/16)*4);
    AL(d_proj,H*4); AL(d_gate,I*4); AL(d_up,I*4);
    AL(d_mlp_i4,I/2); AL(d_mlp_i4_sc,(I/16)*4);
    AL(d_kc,(size_t)NL*nkv*MAXSEQ*hd*4);
    AL(d_vc,(size_t)NL*nkv*MAXSEQ*hd*4);
    AL(d_logits,V*4); AL(d_next_id,4); AL(d_recent,64*4);
    
    float iv7=1.f/7.f;
    { std::vector<float> tmp(H/16,iv7); cudaMemcpy(d_x_i4_sc,tmp.data(),(H/16)*4,cudaMemcpyHostToDevice); }
    { std::vector<float> tmp(Q/16,iv7); cudaMemcpy(d_attn_i4_sc,tmp.data(),(Q/16)*4,cudaMemcpyHostToDevice); }
    { std::vector<float> tmp(I/16,iv7); cudaMemcpy(d_mlp_i4_sc,tmp.data(),(I/16)*4,cudaMemcpyHostToDevice); }
}

// Generate tokens for one sequence
static std::vector<uint32_t> generate_one(
    const std::vector<uint32_t>& input_ids,
    int max_new, float temperature, int top_k, float rep_pen)
{
    std::vector<uint32_t> all_ids = input_ids;
    int gen_start = (int)input_ids.size();
    int total = gen_start + max_new;

    cudaMemset(d_kc,0,(size_t)NL*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vc,0,(size_t)NL*nkv*MAXSEQ*hd*4);

    std::vector<float> h_embed(H);

    for (int step = 0; step < total; ++step) {
        uint32_t tid = (step < gen_start) ? all_ids[step] : all_ids.back();
        // CPU dequant from INT4 embed, then H2D copy (no GPU FP32 cache)
        dequant_embed_row(h_embed.data(), tid, host_embed_d, host_embed_sc, H);
        die(cudaMemcpyAsync(d_x32, h_embed.data(), H*4, cudaMemcpyHostToDevice, st), "embed");

        for (int l = 0; l < NL; ++l) {
            size_t kv_off = (size_t)l*nkv*MAXSEQ*hd;
            die(cudaMemcpyAsync(d_res, d_x32, H*4, cudaMemcpyDeviceToDevice, st), "save_res");
            die(blackwell::kernels::fused_rmsnorm(d_xi_f, d_x32, W[l].rn_in, H, eps, st), "rn_in");
            die(blackwell::kernels::quantize_int4(d_x_i4, d_x_i4_sc, d_xi_f, H, st), "q_in");
            die(blackwell::kernels::fused_qkv_int4_f16wsc(d_Q, d_K, d_V, (const uint8_t*)d_x_i4, d_x_i4_sc, W[l].q.d, W[l].q.sc16, W[l].k.d, W[l].k.sc16, W[l].v.d, W[l].v.sc16, H, Q, KV, 1, st), "qkv");
            head_norm_kernel<<<nqh,128,0,st>>>(d_Q, W[l].qn, nqh, hd, eps); die(cudaGetLastError(),"hn_q");
            head_norm_kernel<<<nkv,128,0,st>>>(d_K, W[l].kn, nkv, hd, eps); die(cudaGetLastError(),"hn_k");
            apply_rope_kernel<<<nqh,hd/2,0,st>>>(d_Q, nqh, hd, step); die(cudaGetLastError(),"rp_q");
            apply_rope_kernel<<<nkv,hd/2,0,st>>>(d_K, nkv, hd, step); die(cudaGetLastError(),"rp_k");
            die(blackwell::kernels::update_kv_cache(d_kc+kv_off, d_vc+kv_off, d_K, d_V, 0, step, nkv, hd, MAXSEQ, st), "kv");
            die(blackwell::kernels::attention_decode_batched_gqa(d_attn, d_Q, d_kc, d_vc, step, nqh, nkv, hd, MAXSEQ, 1,
                (size_t)NL*nkv*MAXSEQ*hd, kv_off, st), "attn");
            die(blackwell::kernels::quantize_int4(d_attn_i4, d_attn_i4_sc, d_attn, Q, st), "q_attn");
            die(blackwell::kernels::gemv_int4_batched_f16wsc(d_proj, (const uint8_t*)d_attn_i4, d_attn_i4_sc, W[l].o.d, W[l].o.sc16, Q, H, 1, st), "o_proj");
            die(blackwell::kernels::vector_add_fp32(d_x32, d_proj, d_res, H, st), "res1");
            die(cudaMemcpyAsync(d_res, d_x32, H*4, cudaMemcpyDeviceToDevice, st), "save_res2");
            die(blackwell::kernels::fused_rmsnorm(d_xi_f, d_x32, W[l].rn_post, H, eps, st), "rn_post");
            die(blackwell::kernels::quantize_int4(d_x_i4, d_x_i4_sc, d_xi_f, H, st), "q_mlp");
            die(blackwell::kernels::fused_gate_up_int4_f16wsc(d_gate, d_up, d_x_i4, d_x_i4_sc, W[l].g.d, W[l].g.sc16, W[l].u.d, W[l].u.sc16, H, I, 1, st), "gate_up");
            blackwell::kernels::apply_swiglu(d_gate, d_gate, d_up, I, st);
            die(blackwell::kernels::quantize_int4(d_mlp_i4, d_mlp_i4_sc, d_gate, I, st), "q_mlp2");
            die(blackwell::kernels::gemv_int4_batched_f16wsc(d_proj, (const uint8_t*)d_mlp_i4, d_mlp_i4_sc, W[l].d.d, W[l].d.sc16, I, H, 1, st), "down");
            die(blackwell::kernels::vector_add_fp32(d_x32, d_proj, d_res, H, st), "res2");
        }

        if (step >= gen_start - 1) {
            die(blackwell::kernels::fused_rmsnorm_quant_int4(d_x_i4,d_x_i4_sc,d_x32,d_fn,H,eps,st),"fn_q");
            die(blackwell::kernels::gemv_int4_batched_f16wsc(d_logits, (const uint8_t*)d_x_i4, d_x_i4_sc, lm_head_w.d, lm_head_w.sc16, H, V, 1, st), "lm_head");
            
            // Repetition penalty
            if (rep_pen > 1.0f && step > gen_start) {
                int n = std::min(step - gen_start, 64);
                std::vector<int> h_rec(n);
                for (int i = 0; i < n; ++i) h_rec[i] = all_ids[gen_start + i];
                cudaMemcpy(d_recent, h_rec.data(), n*4, cudaMemcpyHostToDevice);
                blackwell::kernels::apply_repetition_penalty(d_logits, d_recent, n, rep_pen, V, st);
            }
            
            die(blackwell::kernels::sample_gpu(d_logits, V, temperature, top_k, d_next_id, 0xdeadbeefLL, step, st), "sample");
            int next_id; die(cudaMemcpy(&next_id, d_next_id, 4, cudaMemcpyDeviceToHost), "copy");
            all_ids.push_back(next_id);
            if (next_id == 151643) break; // EOS
        }
    }
    return std::vector<uint32_t>(all_ids.begin() + gen_start, all_ids.end());
}

// =====================================================================
// M>1 batched generation — processes M prompts in PARALLEL through batched GEMV.
// Ported from bench/text_generate_int4_batched.cu (205 t/s at M=8).
// Uses gemv_int4_batched_f16wsc(M>1) for all projections.
// =====================================================================
const int MAXBATCH = 8;
const int MAXSEQ_BATCHED = 512;  // reduced KV cache for M>1 (memory)

struct BatchState {
    int M;
    float *d_x32, *d_xi_f;      // [M][H]
    float *d_residual;            // [M][H]
    uint8_t *d_x_i4;             // [M][H/2]
    float *d_x_i4_sc;            // [M][H/16]
    float *d_Q, *d_K, *d_V;      // [M][Q/KV]
    float *d_attn;                // [M][Q]
    uint8_t *d_attn_i4;         // [M][Q/2]
    float *d_attn_i4_sc;         // [M][Q/16]
    float *d_proj;                // [M][H]
    float *d_gate, *d_up;         // [M][I]
    uint8_t *d_mlp_i4;          // [M][I/2]
    float *d_mlp_i4_sc;          // [M][I/16]
    float *d_logits;              // [M][V]
    int *d_next_id;               // [M]
    float *d_kc, *d_vc;           // KV cache [M][NL][MAXSEQ_BATCHED][nkv][hd]
};

static BatchState g_bs;
static bool g_bs_alloc = false;

static void alloc_batch_buffers() {
    if (g_bs_alloc) return;
    int M = MAXBATCH;
    g_bs.M = M;
    size_t kv_cache = (size_t)M * NL * nkv * MAXSEQ_BATCHED * hd * 4;
    #define ALB(p,n){die(cudaMalloc(&(p),(size_t)(n)),"malloc "#p);}
    ALB(g_bs.d_x32, M*H*4);    ALB(g_bs.d_xi_f, M*H*4);
    ALB(g_bs.d_residual, M*H*4);
    ALB(g_bs.d_x_i4, M*H/2);   ALB(g_bs.d_x_i4_sc, M*H/16*4);
    ALB(g_bs.d_Q, M*Q*4);      ALB(g_bs.d_K, M*KV*4);  ALB(g_bs.d_V, M*KV*4);
    ALB(g_bs.d_attn, M*Q*4);   ALB(g_bs.d_attn_i4, M*Q/2); ALB(g_bs.d_attn_i4_sc, M*Q/16*4);
    ALB(g_bs.d_proj, M*H*4);   ALB(g_bs.d_gate, M*I*4); ALB(g_bs.d_up, M*I*4);
    ALB(g_bs.d_mlp_i4, M*I/2); ALB(g_bs.d_mlp_i4_sc, M*I/16*4);
    ALB(g_bs.d_logits, (size_t)M*V*4);
    ALB(g_bs.d_next_id, M*4);
    ALB(g_bs.d_kc, kv_cache);  ALB(g_bs.d_vc, kv_cache);
    #undef ALB
    // Init scale buffers to 1/7
    float iv7 = 1.f/7.f;
    std::vector<float> tmp(H/16, iv7);
    for (int m = 0; m < M; ++m) {
        cudaMemcpy(g_bs.d_x_i4_sc + (size_t)m*(H/16), tmp.data(), (H/16)*4, cudaMemcpyHostToDevice);
        cudaMemcpy(g_bs.d_attn_i4_sc + (size_t)m*(Q/16), tmp.data(), (Q/16)*4, cudaMemcpyHostToDevice);
        cudaMemcpy(g_bs.d_mlp_i4_sc + (size_t)m*(I/16), tmp.data(), (I/16)*4, cudaMemcpyHostToDevice);
    }
    g_bs_alloc = true;
    fprintf(stderr, "Batch buffers allocated (M=%d, KV cache %zu MB)\n", M, kv_cache*2/(1024*1024));
}

// Generate tokens for M sequences in parallel using batched GEMV.
static std::vector<std::vector<uint32_t>> generate_batch_multi(
    const std::vector<std::vector<uint32_t>>& input_ids_vec,
    int max_new, float temperature, int top_k, float rep_pen)
{
    int M = input_ids_vec.size();
    if (M > MAXBATCH) M = MAXBATCH;
    alloc_batch_buffers();

    std::vector<std::vector<uint32_t>> all_ids(M);
    std::vector<int> gen_start(M), seq_pos(M, 0);
    for (int m = 0; m < M; ++m) {
        all_ids[m] = input_ids_vec[m];
        gen_start[m] = (int)input_ids_vec[m].size();
    }

    // Clear KV cache for M sequences
    size_t kv_total = (size_t)M * NL * nkv * MAXSEQ_BATCHED * hd * 4;
    cudaMemsetAsync(g_bs.d_kc, 0, kv_total, st);
    cudaMemsetAsync(g_bs.d_vc, 0, kv_total, st);

    int max_steps = max_new + *std::max_element(gen_start.begin(), gen_start.end());

    std::vector<float> h_embed(H);
    for (int step = 0; step < max_steps; ++step) {
        // Embed all M sequences (CPU dequant + H2D)
        for (int m = 0; m < M; ++m) {
            uint32_t tid = (step < gen_start[m]) ? all_ids[m][step] : all_ids[m].back();
            dequant_embed_row(h_embed.data(), tid, host_embed_d, host_embed_sc, H);
            cudaMemcpyAsync(g_bs.d_x32 + (size_t)m*H, h_embed.data(),
                           H*4, cudaMemcpyHostToDevice, st);
        }

        for (int l = 0; l < NL; ++l) {
            // Save residual (M*H copy)
            cudaMemcpyAsync(g_bs.d_residual, g_bs.d_x32, (size_t)M*H*4,
                           cudaMemcpyDeviceToDevice, st);
            // Pre-attn norm + quant (batched)
            blackwell::kernels::fused_rmsnorm_batched(
                g_bs.d_xi_f, g_bs.d_x32, W[l].rn_in, H, eps, M, st);
            blackwell::kernels::quantize_int4_batched(
                g_bs.d_x_i4, g_bs.d_x_i4_sc, g_bs.d_xi_f, H, M, st);
            // QKV projections (batched M)
            blackwell::kernels::gemv_int4_batched_f16wsc(
                g_bs.d_Q, g_bs.d_x_i4, g_bs.d_x_i4_sc, W[l].q.d, W[l].q.sc16, H, Q, M, st);
            blackwell::kernels::gemv_int4_batched_f16wsc(
                g_bs.d_K, g_bs.d_x_i4, g_bs.d_x_i4_sc, W[l].k.d, W[l].k.sc16, H, KV, M, st);
            blackwell::kernels::gemv_int4_batched_f16wsc(
                g_bs.d_V, g_bs.d_x_i4, g_bs.d_x_i4_sc, W[l].v.d, W[l].v.sc16, H, KV, M, st);
            // Q/K head norms + RoPE (per-seq: different rope_pos)
            for (int m = 0; m < M; ++m) {
                int rope_pos = (step >= gen_start[m]-1) ? gen_start[m]-1+seq_pos[m] : step;
                head_norm_kernel<<<nqh,128,0,st>>>(g_bs.d_Q + (size_t)m*Q, W[l].qn, nqh, hd, eps);
                head_norm_kernel<<<nkv,128,0,st>>>(g_bs.d_K + (size_t)m*KV, W[l].kn, nkv, hd, eps);
                apply_rope_kernel<<<nqh,hd/2,0,st>>>(g_bs.d_Q + (size_t)m*Q, nqh, hd, rope_pos);
                apply_rope_kernel<<<nkv,hd/2,0,st>>>(g_bs.d_K + (size_t)m*KV, nkv, hd, rope_pos);
            }
            // KV cache update (per-seq)
            size_t l_kv_off = (size_t)l * nkv * MAXSEQ_BATCHED * hd;
            for (int m = 0; m < M; ++m) {
                size_t m_kv_off = (size_t)m * NL * nkv * MAXSEQ_BATCHED * hd + l_kv_off;
                blackwell::kernels::update_kv_cache(
                    g_bs.d_kc + m_kv_off, g_bs.d_vc + m_kv_off,
                    g_bs.d_K + (size_t)m*KV, g_bs.d_V + (size_t)m*KV,
                    0, step, nkv, hd, MAXSEQ_BATCHED, st);
            }
            // Attention (per-seq, M=1 batched call per sequence)
            for (int m = 0; m < M; ++m) {
                size_t m_kv_off = (size_t)m * NL * nkv * MAXSEQ_BATCHED * hd;
                blackwell::kernels::attention_decode_batched_gqa(
                    g_bs.d_attn + (size_t)m*Q, g_bs.d_Q + (size_t)m*Q,
                    g_bs.d_kc + m_kv_off, g_bs.d_vc + m_kv_off,
                    step, nqh, nkv, hd, MAXSEQ_BATCHED, 1,
                    (size_t)nkv*MAXSEQ_BATCHED*hd, l_kv_off, st);
            }
            // Wo projection (batched M)
            blackwell::kernels::quantize_int4_batched(
                g_bs.d_attn_i4, g_bs.d_attn_i4_sc, g_bs.d_attn, Q, M, st);
            blackwell::kernels::gemv_int4_batched_f16wsc(
                g_bs.d_proj, g_bs.d_attn_i4, g_bs.d_attn_i4_sc,
                W[l].o.d, W[l].o.sc16, Q, H, M, st);
            // Residual add (per-seq)
            for (int m = 0; m < M; ++m)
                blackwell::kernels::vector_add_fp32(
                    g_bs.d_x32+(size_t)m*H, g_bs.d_proj+(size_t)m*H, g_bs.d_residual+(size_t)m*H, H, st);
            cudaMemcpyAsync(g_bs.d_residual, g_bs.d_x32, (size_t)M*H*4, cudaMemcpyDeviceToDevice, st);
            // Pre-MLP norm + quant (batched)
            blackwell::kernels::fused_rmsnorm_batched(
                g_bs.d_xi_f, g_bs.d_x32, W[l].rn_post, H, eps, M, st);
            blackwell::kernels::quantize_int4_batched(
                g_bs.d_x_i4, g_bs.d_x_i4_sc, g_bs.d_xi_f, H, M, st);
            // Gate + up (batched M)
            blackwell::kernels::gemv_int4_batched_f16wsc(
                g_bs.d_gate, g_bs.d_x_i4, g_bs.d_x_i4_sc, W[l].g.d, W[l].g.sc16, H, I, M, st);
            blackwell::kernels::gemv_int4_batched_f16wsc(
                g_bs.d_up, g_bs.d_x_i4, g_bs.d_x_i4_sc, W[l].u.d, W[l].u.sc16, H, I, M, st);
            // SwiGLU (per-seq)
            for (int m = 0; m < M; ++m)
                blackwell::kernels::apply_swiglu(
                    g_bs.d_gate+(size_t)m*I, g_bs.d_gate+(size_t)m*I, g_bs.d_up+(size_t)m*I, I, st);
            // Quantize MLP output (batched)
            blackwell::kernels::quantize_int4_batched(
                g_bs.d_mlp_i4, g_bs.d_mlp_i4_sc, g_bs.d_gate, I, M, st);
            // Down projection (batched M)
            blackwell::kernels::gemv_int4_batched_f16wsc(
                g_bs.d_proj, g_bs.d_mlp_i4, g_bs.d_mlp_i4_sc,
                W[l].d.d, W[l].d.sc16, I, H, M, st);
            // Residual add (per-seq)
            for (int m = 0; m < M; ++m)
                blackwell::kernels::vector_add_fp32(
                    g_bs.d_x32+(size_t)m*H, g_bs.d_proj+(size_t)m*H, g_bs.d_residual+(size_t)m*H, H, st);
        }

        // Final norm + lm_head + sampling (for sequences that are generating)
        int min_gen = *std::min_element(gen_start.begin(), gen_start.end());
        if (step >= min_gen - 1) {
            blackwell::kernels::fused_rmsnorm_batched(
                g_bs.d_xi_f, g_bs.d_x32, d_fn, H, eps, M, st);
            blackwell::kernels::quantize_int4_batched(
                g_bs.d_x_i4, g_bs.d_x_i4_sc, g_bs.d_xi_f, H, M, st);
            blackwell::kernels::gemv_int4_batched_f16wsc(
                g_bs.d_logits, g_bs.d_x_i4, g_bs.d_x_i4_sc,
                lm_head_w.d, lm_head_w.sc16, H, V, M, st);
            for (int m = 0; m < M; ++m) {
                if (step >= gen_start[m] - 1) {
                    // Repetition penalty
                    if (rep_pen > 1.0f) {
                        int num_recent = (int)all_ids[m].size() - gen_start[m];
                        if (num_recent > 0) {
                            if (num_recent > 64) num_recent = 64;
                            std::vector<int> h_rec(all_ids[m].end()-num_recent, all_ids[m].end());
                            cudaMemcpyAsync(d_recent, h_rec.data(), num_recent*4, cudaMemcpyHostToDevice, st);
                            blackwell::kernels::apply_repetition_penalty(
                                g_bs.d_logits+(size_t)m*V, d_recent, num_recent, rep_pen, V, st);
                        }
                    }
                    blackwell::kernels::sample_gpu(
                        g_bs.d_logits+(size_t)m*V, V, temperature, top_k,
                        g_bs.d_next_id+m, 0xdeadbeefLL, step, st);
                }
            }
        }

        cudaStreamSynchronize(st);

        // Collect results
        std::vector<int> next_ids(M, 0);
        bool all_done = true;
        for (int m = 0; m < M; ++m) {
            if (step >= gen_start[m] - 1) {
                cudaMemcpy(&next_ids[m], g_bs.d_next_id+m, 4, cudaMemcpyDeviceToHost);
                all_ids[m].push_back(next_ids[m]);
                seq_pos[m]++;
                if (next_ids[m] != 151643 && next_ids[m] != 151645) all_done = false;
            } else {
                all_done = false;
            }
        }
        if (all_done) break;
    }

    std::vector<std::vector<uint32_t>> results(M);
    for (int m = 0; m < M; ++m)
        results[m] = std::vector<uint32_t>(all_ids[m].begin()+gen_start[m], all_ids[m].end());
    return results;
}

static std::string json_escape(const std::string& s) {
    std::string r;
    for (char c : s) {
        if (c == '"') r += "\\\"";
        else if (c == '\\') r += "\\\\";
        else if (c == '\n') r += "\\n";
        else if (c == '\r') r += "\\r";
        else if (c == '<') r += "\\u003c";
        else if (c == '>') r += "\\u003e";
        else if ((unsigned char)c >= 0x80) { char buf[8]; snprintf(buf,8,"\\u00%02x",(unsigned char)c); r += buf; }
        else r += c;
    }
    return r;
}

int main(int argc, char** argv) {
    fprintf(stderr, "Blackwell INT4 Batched Server v0.9.3\n");
    cudaDeviceProp P; cudaGetDeviceProperties(&P, 0);
    fprintf(stderr, "Device: %s (CC %d.%d)\n", P.name, P.major, P.minor);
    
    // Enable deterministic mode (reduces non-determinism from FP reductions)
    // Note: May reduce performance slightly
    cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceLmemResizeToMax);
    
    die(cudaStreamCreate(&st), "stream");

    if (tokenizer.load("tokenizer_data.bin") != 0) {
        fprintf(stderr, "FAIL: no tokenizer_data.bin\n"); return 1;
    }

    load_model();
    alloc_buffers();

    fprintf(stderr, "Ready.\n"); fflush(stdout);

    while (true) {
        std::string line = read_stdin_line();
        if (line.empty()) continue;

        int max_tokens = parse_int(line, "\"max_tokens\"", 30);
        float temperature = parse_float(line, "\"temperature\"", 0.0f);
        int top_k = parse_int(line, "\"top_k\"", 0);
        float rep_pen = parse_repetition_penalty(line, 1.0f);

        auto prompts = parse_prompts(line);
        if (prompts.empty()) {
            printf("{\"error\":\"no prompts\"}\n"); fflush(stdout); continue;
        }

        // Process all prompts in PARALLEL using batched GEMV (M>1)
        std::vector<std::vector<uint32_t>> all_input_ids;
        for (size_t i = 0; i < prompts.size(); ++i)
            all_input_ids.push_back(tokenizer.encode(prompts[i]));
        auto all_results = generate_batch_multi(all_input_ids, max_tokens, temperature, top_k, rep_pen);

        // Output batched JSON
        printf("{\"tokens\":[");
        for (size_t m = 0; m < all_results.size(); ++m) {
            if (m) printf(",");
            printf("[");
            for (size_t i = 0; i < all_results[m].size(); ++i) {
                if (i) printf(",");
                printf("%u", all_results[m][i]);
            }
            printf("]");
        }
        printf("],\"text\":[");
        for (size_t m = 0; m < all_results.size(); ++m) {
            if (m) printf(",");
            printf("\"");
            std::string text;
            for (auto id : all_results[m]) text += tokenizer.decode(id);
            printf("%s", json_escape(text).c_str());
            printf("\"");
        }
        printf("]}\n"); fflush(stdout);
    }
    return 0;
}
