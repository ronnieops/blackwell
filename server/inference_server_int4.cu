// server/inference_server_int4.cu — INT4 Qwen3-8B inference server (JSON stdio)
// Uses the EXACT decode loop from bench/text_generate_int4_qwen3_8b.cu
// which produces correct coherent output at 57 t/s.
//
// Protocol: reads JSON from stdin, writes JSON to stdout.
// Input:  {"prompts":["str1","str2"],"max_tokens":N,"temperature":T,"top_k":K,"repetition_penalty":P}
// Output: {"tokens":[[id1,...],[...]],"text":["text1","text2"]}
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_120a server/inference_server_int4.cu \
//     build/libblackwell_kernels.a -I include -lcudart -lpthread -lz \
//     -o server/inference_server_int4

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
const int nqh=32, nkv=8, hd=128, MAXSEQ=2048;
const float eps=1e-6f;
const int V=151936;
const int NL=36;

// ── INT4 weight struct + loader ──
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
struct DevW4f16 { int K, N; uint8_t* d; __half* sc16; };
static DevW4f16 upload_w4_f16sc(const char* prefix) {
    char p[256]; snprintf(p,256,"%s.int4_t",prefix);
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"FAIL open %s\n",p);exit(1);}
    int h[5]; fread(h,4,5,f);
    DevW4f16 dw; dw.K=h[0]; dw.N=h[1];
    size_t ds=(size_t)h[0]*h[1]/2;
    uint8_t* td=new uint8_t[ds]; fread(td,1,ds,f); fclose(f);
    cudaMalloc(&dw.d,ds); cudaMemcpy(dw.d,td,ds,cudaMemcpyHostToDevice); delete[] td;
    snprintf(p,256,"%s.scale_t",prefix); f=fopen(p,"rb"); fread(h,4,5,f);
    size_t ss=(size_t)h[3]*h[4];
    __half* ts=new __half[ss]; fread(ts,2,ss,f); fclose(f);
    cudaMalloc(&dw.sc16,ss*2); cudaMemcpy(dw.sc16,ts,ss*2,cudaMemcpyHostToDevice); delete[] ts;
    return dw;
}

struct LW4 { DevW4f16 q,k,v,o,g,u,d; float* qn,*kn,*rn_in,*rn_post; };

// ── CUDA kernels (same as benchmark) ──
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

static void dequant_embed_row(float* out, int token, const uint8_t* host_w,
    const float* host_sc, int K) {
    int kblocks=K/16;
    for(int b=0;b<kblocks;++b){
        float sc=host_sc[token*kblocks+b];
        for(int i=0;i<16;++i){
            size_t byte_idx=(size_t)token*K/2+(size_t)b*8+i/2;
            uint8_t byte=host_w[byte_idx];
            int nib=(i&1)?((byte>>4)&0x0F):(byte&0x0F);
            out[b*16+i]=(float)(nib-8)*sc;
        }
    }
}

static void build_rope_cache(float* cos_cache, float* sin_cache, int max_seq, int head_dim) {
    int pairs = head_dim / 2;
    float rope_theta = 1000000.0f;
    for (int pos = 0; pos < max_seq; ++pos) {
        for (int d = 0; d < pairs; ++d) {
            float theta = (float)pos * powf(rope_theta, -2.0f * (float)d / (float)head_dim);
            cos_cache[pos * pairs + d] = cosf(theta);
            sin_cache[pos * pairs + d] = sinf(theta);
        }
    }
}

// ── JSON helpers ──
static std::string json_escape(const std::string& s);

static std::string read_stdin_line() {
    std::string line; int c;
    while ((c = getchar()) != EOF && c != '\n') line.push_back((char)c);
    return line;
}

static std::vector<std::string> parse_string_prompts(const std::string& json) {
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

static int parse_int(const std::string& json, const char* key, int def) {
    const char* p = strstr(json.c_str(), key);
    if (!p) return def;
    p += strlen(key);
    while (*p == ' ' || *p == ':' || *p == '=') p++;
    return atoi(p);
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
    p += 18; // strlen("repetition_penalty")
    while (*p == ' ' || *p == ':' || *p == '=') p++;
    return strtof(p, nullptr);
}

// ── Global state (loaded once) ──
static const int MAX_BATCH = 16;
static float *d_x32, *d_xi_f, *d_res;
static uint8_t *d_x_i4; static float *d_x_i4_sc;
static float *d_Q,*d_K,*d_V,*d_attn;
static uint8_t *d_attn_i4; static float *d_attn_i4_sc;
static float *d_proj, *d_gate, *d_up;
static uint8_t *d_mlp_i4; static float *d_mlp_i4_sc;
static float *d_fn, *d_kc, *d_vc, *d_logits;
static int *d_next_id, *d_recent;
// CUDA Graph support
static int* d_seq_pos;
static int* h_seq_pos_pinned;
static float* d_cos_cache;
static float* d_sin_cache;
static cudaGraph_t graph_decode;
static cudaGraphExec_t graph_exec_decode;
static bool graph_captured = false;
// Batched decode buffers (M=1-8)
static float *d_x32_batch;   // [MAX_BATCH][H]
static float *d_xi_f_batch;   // [MAX_BATCH][H]
static float *d_Q_batch, *d_K_batch, *d_V_batch; // [MAX_BATCH][..]
static float *d_attn_batch;  // [MAX_BATCH][Q]
static float *d_proj_batch;  // [MAX_BATCH][H]
static float *d_gate_batch, *d_up_batch; // [MAX_BATCH][I]
static uint8_t *d_x_i4_batch, *d_attn_i4_batch, *d_mlp_i4_batch;
static float *d_x_i4_sc_batch, *d_attn_i4_sc_batch, *d_mlp_i4_sc_batch;
static float *d_res_batch;   // [MAX_BATCH][H]
// Prefill buffers (M tokens in flight)
static float *d_Q_batch_prefill, *d_K_batch_prefill, *d_V_batch_prefill;
static float *d_attn_batch_prefill;
static cudaStream_t st;

static std::vector<LW4> W(NL);
static DevW4f16 embed_w, lm_head_w;
static uint8_t* host_embed_d;
static float* host_embed_sc;
static blackwell::BpeTokenizer tokenizer;

#define AL(p,n) die(cudaMalloc(&(p),(n)),"malloc " #p)

static void load_model() {
    fprintf(stderr, "Loading %d-layer INT4 model...\n", NL);
    char p[256];
    for (int l = 0; l < NL; ++l) {
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_self_attn.q_proj",l); W[l].q=upload_w4_f16sc(p);
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_self_attn.k_proj",l); W[l].k=upload_w4_f16sc(p);
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_self_attn.v_proj",l); W[l].v=upload_w4_f16sc(p);
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_self_attn.o_proj",l); W[l].o=upload_w4_f16sc(p);
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_mlp.gate_proj",l); W[l].g=upload_w4_f16sc(p);
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_mlp.up_proj",l);   W[l].u=upload_w4_f16sc(p);
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_mlp.down_proj",l); W[l].d=upload_w4_f16sc(p);
        if ((l+1)%7==0) fprintf(stderr, "  layer %d/%d\n", l+1, NL);
    }
    // QK norms
    float* qk_h=(float*)malloc(NL*2*hd*4);
    {FILE*f=fopen("weights_int4_qwen3_8b_fp16sc/qk_norms.f32","rb");size_t nr=fread(qk_h,4,NL*2*hd,f);if(nr!=(size_t)NL*2*hd){fprintf(stderr, "Truncated qk_norms\n");exit(1);}fclose(f);}
    for(int l=0;l<NL;++l){
        cudaMalloc(&W[l].qn,hd*4);cudaMemcpy(W[l].qn,qk_h+l*2*hd,hd*4,cudaMemcpyHostToDevice);
        cudaMalloc(&W[l].kn,hd*4);cudaMemcpy(W[l].kn,qk_h+l*2*hd+hd,hd*4,cudaMemcpyHostToDevice);
    } free(qk_h);
    // Per-layer RMSNorm
    for(int l=0;l<NL;++l){
        float* w=(float*)malloc(H*4);
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_input_layernorm.f32",l);
        {FILE*f=fopen(p,"rb");size_t nr=fread(w,4,H,f);if(nr!=(size_t)H){fprintf(stderr,"Truncated layernorm %d\n",l);exit(1);}fclose(f);}
        cudaMalloc(&W[l].rn_in,H*4);cudaMemcpy(W[l].rn_in,w,H*4,cudaMemcpyHostToDevice);
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_post_attention_layernorm.f32",l);
        {FILE*f=fopen(p,"rb");size_t nr=fread(w,4,H,f);if(nr!=(size_t)H){fprintf(stderr,"Truncated post layernorm %d\n",l);exit(1);}fclose(f);}
        cudaMalloc(&W[l].rn_post,H*4);cudaMemcpy(W[l].rn_post,w,H*4,cudaMemcpyHostToDevice);
        free(w);
    }
    // Final norm
    {float*w=(float*)malloc(H*4);
     FILE*f=fopen("weights_int4_qwen3_8b_fp16sc/final_norm.f32","rb");size_t nr=fread(w,4,H,f);if(nr!=(size_t)H){fprintf(stderr,"Truncated final_norm\n");exit(1);}fclose(f);
     AL(d_fn,H*4); cudaMemcpy(d_fn,w,H*4,cudaMemcpyHostToDevice); free(w);}
    // Embed + lm_head
    embed_w=upload_w4_f16sc("weights_int4_qwen3_8b_fp16sc/embed_tokens");
    lm_head_w=upload_w4_f16sc("weights_int4_qwen3_8b_fp16sc/lm_head");
    fprintf(stderr,"  embed: %dx%d, lm_head: %dx%d\n",embed_w.K,embed_w.N,lm_head_w.K,lm_head_w.N);
    // Host embed for CPU dequant
    host_embed_d=new uint8_t[(size_t)embed_w.K*embed_w.N/2];
    host_embed_sc=new float[embed_w.N*(embed_w.K/16)];
    {FILE*f=fopen("weights_int4_qwen3_8b_fp16sc/embed_tokens.int4_t","rb");int h[5];fread(h,4,5,f);
     size_t ds=(size_t)h[0]*h[1]/2;size_t nr=fread(host_embed_d,1,ds,f);if(nr!=ds){fprintf(stderr,"Truncated embed_tokens\n");exit(1);}fclose(f);
     f=fopen("weights_int4_qwen3_8b_fp16sc/embed_tokens.scale_t","rb");fread(h,4,5,f);
     size_t ss=(size_t)h[3]*h[4];{__half* tmp=new __half[ss];size_t nr=fread(tmp,2,ss,f);if(nr!=ss){fprintf(stderr,"Truncated embed_tokens scales\n");exit(1);}
      for(size_t i=0;i<ss;++i) host_embed_sc[i]=__half2float(tmp[i]);delete[] tmp;}fclose(f);}
    fprintf(stderr,"All weights loaded.\n");
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
    // Batched decode buffers
    AL(d_x32_batch, (size_t)MAX_BATCH * H * 4);
    AL(d_xi_f_batch, (size_t)MAX_BATCH * H * 4);
    AL(d_Q_batch,   (size_t)MAX_BATCH * Q * 4);
    AL(d_K_batch,   (size_t)MAX_BATCH * KV * 4);
    AL(d_V_batch,   (size_t)MAX_BATCH * KV * 4);
    AL(d_attn_batch,(size_t)MAX_BATCH * Q * 4);
    AL(d_proj_batch,(size_t)MAX_BATCH * H * 4);
    AL(d_gate_batch,(size_t)MAX_BATCH * I * 4);
    AL(d_up_batch,  (size_t)MAX_BATCH * I * 4);
    AL(d_x_i4_batch,(size_t)MAX_BATCH * H / 2);
    AL(d_x_i4_sc_batch, (size_t)MAX_BATCH * (H/16) * 4);
    AL(d_attn_i4_batch, (size_t)MAX_BATCH * Q / 2);
    AL(d_attn_i4_sc_batch,(size_t)MAX_BATCH * (Q/16) * 4);
    AL(d_mlp_i4_batch, (size_t)MAX_BATCH * I / 2);
    AL(d_mlp_i4_sc_batch, (size_t)MAX_BATCH * (I/16) * 4);
    AL(d_res_batch, (size_t)MAX_BATCH * H * 4);
    // Prefill Q/K/V buffers (M tokens × Q/KV)
    AL(d_Q_batch_prefill, (size_t)MAX_BATCH * Q * 4);
    AL(d_K_batch_prefill, (size_t)MAX_BATCH * KV * 4);
    AL(d_V_batch_prefill, (size_t)MAX_BATCH * KV * 4);
    // Attention output buffer for prefill (M × Q)
    AL(d_attn_batch_prefill, (size_t)MAX_BATCH * Q * 4);
    // CUDA Graph buffers
    AL(d_seq_pos, sizeof(int));
    cudaHostAlloc(&h_seq_pos_pinned, sizeof(int), cudaHostAllocDefault);
    {
        int rope_pairs = hd / 2;
        AL(d_cos_cache, (size_t)MAXSEQ * rope_pairs * 4);
        AL(d_sin_cache, (size_t)MAXSEQ * rope_pairs * 4);
        std::vector<float> cos_h(MAXSEQ * rope_pairs);
        std::vector<float> sin_h(MAXSEQ * rope_pairs);
        build_rope_cache(cos_h.data(), sin_h.data(), MAXSEQ, hd);
        cudaMemcpy(d_cos_cache, cos_h.data(), (size_t)MAXSEQ * rope_pairs * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_sin_cache, sin_h.data(), (size_t)MAXSEQ * rope_pairs * 4, cudaMemcpyHostToDevice);
    }
    graph_captured = false;
    float iv7=1.f/7.f;
    { std::vector<float> tmp(H/16,iv7); cudaMemcpy(d_x_i4_sc,tmp.data(),(H/16)*4,cudaMemcpyHostToDevice); }
    { std::vector<float> tmp(Q/16,iv7); cudaMemcpy(d_attn_i4_sc,tmp.data(),(Q/16)*4,cudaMemcpyHostToDevice); }
    { std::vector<float> tmp(I/16,iv7); cudaMemcpy(d_mlp_i4_sc,tmp.data(),(I/16)*4,cudaMemcpyHostToDevice); }
}

// ── Generate: exact decode loop from benchmark ──
// Run one token through all decoder layers (no lm_head/sampling).
// Returns hidden state in d_x32.
static void decode_one_token(uint32_t token_id, int step) {
    std::vector<float> h_embed(H);
    dequant_embed_row(h_embed.data(), token_id, host_embed_d, host_embed_sc, H);
    die(cudaMemcpyAsync(d_x32, h_embed.data(), H*4, cudaMemcpyHostToDevice, st), "embed");

    if (graph_captured) {
        // ── CUDA Graph replay ──
        *h_seq_pos_pinned = step;
        cudaMemcpyAsync(d_seq_pos, h_seq_pos_pinned, sizeof(int), cudaMemcpyHostToDevice, st);
        cudaGraphLaunch(graph_exec_decode, st);
    } else {
        // ── Per-kernel path ──
        for (int l = 0; l < NL; ++l) {
            die(cudaMemcpyAsync(d_res, d_x32, H*4, cudaMemcpyDeviceToDevice, st), "save_res");
            die(blackwell::kernels::fused_rmsnorm(d_xi_f, d_x32, W[l].rn_in, H, eps, st), "rmsnorm_in");
            die(blackwell::kernels::quantize_int4_batched(d_x_i4, d_x_i4_sc, d_xi_f, H, 1, st), "quant_in");
            die(blackwell::kernels::fused_qkv_int4_f16wsc(d_Q, d_K, d_V, (const uint8_t*)d_x_i4, d_x_i4_sc, W[l].q.d, W[l].q.sc16, W[l].k.d, W[l].k.sc16, W[l].v.d, W[l].v.sc16, H, Q, KV, 1, st), "qkv");
            head_norm_kernel<<<nqh,128,0,st>>>(d_Q, W[l].qn, nqh, hd, eps);
            head_norm_kernel<<<nkv,128,0,st>>>(d_K, W[l].kn, nkv, hd, eps);
            apply_rope_kernel<<<nqh,hd/2,0,st>>>(d_Q, nqh, hd, step);
            apply_rope_kernel<<<nkv,hd/2,0,st>>>(d_K, nkv, hd, step);
            size_t kv_off = (size_t)l * nkv * MAXSEQ * hd;
            die(blackwell::kernels::update_kv_cache(d_kc+kv_off, d_vc+kv_off, d_K, d_V, 0, step, nkv, hd, MAXSEQ, st), "kv");
            die(blackwell::kernels::attention_decode_batched_gqa(d_attn, d_Q, d_kc, d_vc, step, nqh, nkv, hd, MAXSEQ, 1,
                (size_t)NL*nkv*MAXSEQ*hd, kv_off, st), "attn");
            die(blackwell::kernels::quantize_int4_batched(d_attn_i4, d_attn_i4_sc, d_attn, Q, 1, st), "quant_attn");
            die(blackwell::kernels::gemv_int4_batched_f16wsc(d_proj, (const uint8_t*)d_attn_i4, d_attn_i4_sc, W[l].o.d, W[l].o.sc16, Q, H, 1, st), "o_proj");
            die(blackwell::kernels::vector_add_fp32(d_x32, d_proj, d_res, H, st), "attn_res");
            die(cudaMemcpyAsync(d_res, d_x32, H*4, cudaMemcpyDeviceToDevice, st), "save_res2");
            die(blackwell::kernels::fused_rmsnorm(d_xi_f, d_x32, W[l].rn_post, H, eps, st), "rmsnorm_post");
            die(blackwell::kernels::quantize_int4_batched(d_x_i4, d_x_i4_sc, d_xi_f, H, 1, st), "quant_mlp_in");
            die(blackwell::kernels::fused_gate_up_int4_f16wsc(d_gate, d_up, d_x_i4, d_x_i4_sc, W[l].g.d, W[l].g.sc16, W[l].u.d, W[l].u.sc16, H, I, 1, st), "gate_up");
            blackwell::kernels::apply_swiglu(d_gate, d_gate, d_up, I, st);
            die(blackwell::kernels::quantize_int4_batched(d_mlp_i4, d_mlp_i4_sc, d_gate, I, 1, st), "quant_mlp");
            die(blackwell::kernels::gemv_int4_batched_f16wsc(d_proj, (const uint8_t*)d_mlp_i4, d_mlp_i4_sc, W[l].d.d, W[l].d.sc16, I, H, 1, st), "down");
            die(blackwell::kernels::vector_add_fp32(d_x32, d_proj, d_res, H, st), "mlp_res");
        }
    }
}

// Capture CUDA Graph for decode loop (one token through all layers)
// Must be called after prefill (KV cache populated). Graph captures the full
// 36-layer decode with device-side seq_pos. Between replays, update seq_pos
// via pinned host memory.
static void capture_decode_graph() {
    cudaStream_t gs;
    cudaStreamCreate(&gs);

    int capture_pos = 0;
    cudaMemcpy(d_seq_pos, &capture_pos, sizeof(int), cudaMemcpyHostToDevice);

    cudaStreamBeginCapture(gs, cudaStreamCaptureModeGlobal);

    for (int l = 0; l < NL; ++l) {
        size_t kv_off = (size_t)l * nkv * MAXSEQ * hd;

        // All kernels use the same global buffers (d_x32, d_Q, etc.)
        cudaMemcpyAsync(d_res, d_x32, H*4, cudaMemcpyDeviceToDevice, gs);
        blackwell::kernels::fused_rmsnorm(d_xi_f, d_x32, W[l].rn_in, H, eps, gs);
        blackwell::kernels::quantize_int4_batched(d_x_i4, d_x_i4_sc, d_xi_f, H, 1, gs);
        blackwell::kernels::fused_qkv_int4_f16wsc(d_Q, d_K, d_V, (const uint8_t*)d_x_i4, d_x_i4_sc, W[l].q.d, W[l].q.sc16, W[l].k.d, W[l].k.sc16, W[l].v.d, W[l].v.sc16, H, Q, KV, 1, gs);
        head_norm_kernel<<<nqh,128,0,gs>>>(d_Q, W[l].qn, nqh, hd, eps);
        head_norm_kernel<<<nkv,128,0,gs>>>(d_K, W[l].kn, nkv, hd, eps);
        blackwell::kernels::fused_rope_decode(d_Q, d_cos_cache, d_sin_cache, d_seq_pos, nqh, hd, MAXSEQ, gs);
        blackwell::kernels::fused_rope_decode(d_K, d_cos_cache, d_sin_cache, d_seq_pos, nkv, hd, MAXSEQ, gs);
        blackwell::kernels::update_kv_cache_device(d_kc+kv_off, d_vc+kv_off, d_K, d_V, 0, d_seq_pos, nkv, hd, MAXSEQ, gs);
        blackwell::kernels::attention_decode_batched_gqa_device(d_attn, d_Q, d_kc, d_vc, d_seq_pos, nqh, nkv, hd, MAXSEQ, 1,
            (size_t)NL*nkv*MAXSEQ*hd, kv_off, gs);
        blackwell::kernels::quantize_int4_batched(d_attn_i4, d_attn_i4_sc, d_attn, Q, 1, gs);
        blackwell::kernels::gemv_int4_batched_f16wsc(d_proj, (const uint8_t*)d_attn_i4, d_attn_i4_sc, W[l].o.d, W[l].o.sc16, Q, H, 1, gs);
        blackwell::kernels::vector_add_fp32(d_x32, d_proj, d_res, H, gs);
        cudaMemcpyAsync(d_res, d_x32, H*4, cudaMemcpyDeviceToDevice, gs);
        blackwell::kernels::fused_rmsnorm(d_xi_f, d_x32, W[l].rn_post, H, eps, gs);
        blackwell::kernels::quantize_int4_batched(d_x_i4, d_x_i4_sc, d_xi_f, H, 1, gs);
        blackwell::kernels::fused_gate_up_int4_f16wsc(d_gate, d_up, d_x_i4, d_x_i4_sc, W[l].g.d, W[l].g.sc16, W[l].u.d, W[l].u.sc16, H, I, 1, gs);
        blackwell::kernels::apply_swiglu(d_gate, d_gate, d_up, I, gs);
        blackwell::kernels::quantize_int4_batched(d_mlp_i4, d_mlp_i4_sc, d_gate, I, 1, gs);
        blackwell::kernels::gemv_int4_batched_f16wsc(d_proj, (const uint8_t*)d_mlp_i4, d_mlp_i4_sc, W[l].d.d, W[l].d.sc16, I, H, 1, gs);
        blackwell::kernels::vector_add_fp32(d_x32, d_proj, d_res, H, gs);
    }

    cudaError_t cerr = cudaStreamEndCapture(gs, &graph_decode);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "FAIL graph capture: %s\n", cudaGetErrorString(cerr));
        cudaStreamDestroy(gs);
        return;
    }

    size_t num_nodes = 0;
    cudaGraphGetNodes(graph_decode, NULL, &num_nodes);
    fprintf(stderr, "CUDA Graph: %zu nodes captured\n", num_nodes);

    cerr = cudaGraphInstantiate(&graph_exec_decode, graph_decode, NULL, NULL, 0);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "FAIL graph instantiate: %s\n", cudaGetErrorString(cerr));
        cudaGraphDestroy(graph_decode);
        cudaStreamDestroy(gs);
        return;
    }

    graph_captured = true;
    cudaStreamDestroy(gs);
}

// Prefill M prompt tokens through all layers via batched QKV + per-token attention.
// Processes in chunks of MAX_BATCH. Each chunk's attention reads from the
// decode cache (which has all prior positions from previous chunks).
// KV cache is filled for all M positions. After prefill, d_x32 has
// the hidden state of the last token (M-1).
static void prefill_tokens_batched(const std::vector<uint32_t>& token_ids, int offset, int M) {
    int last_chunk = M;
    while (M > 0) {
        int chunk = M;
        if (chunk > MAX_BATCH) chunk = MAX_BATCH;
        last_chunk = chunk;


        // Embed tokens in this chunk
        for (int m = 0; m < chunk; ++m) {
            std::vector<float> h_embed(H);
            dequant_embed_row(h_embed.data(), token_ids[offset + m], host_embed_d, host_embed_sc, H);
            die(cudaMemcpyAsync(d_x32_batch + m * H, h_embed.data(), H*4, cudaMemcpyHostToDevice, st), "embed_m");
        }

        for (int l = 0; l < NL; ++l) {
            // Save residual
            die(cudaMemcpyAsync(d_res_batch, d_x32_batch, (size_t)chunk * H * 4, cudaMemcpyDeviceToDevice, st), "save_res");

            // Pre-attention norm + quantize
            die(blackwell::kernels::fused_rmsnorm_batched(d_xi_f_batch, d_x32_batch, W[l].rn_in, H, eps, chunk, st), "rmsnorm_in");
            die(blackwell::kernels::quantize_int4_batched(d_x_i4_batch, d_x_i4_sc_batch, d_xi_f_batch, H, chunk, st), "quant_in");

            // QKV projections (batched)
            die(blackwell::kernels::fused_qkv_int4_f16wsc(d_Q_batch_prefill, d_K_batch_prefill, d_V_batch_prefill, (const uint8_t*)d_x_i4_batch, d_x_i4_sc_batch, W[l].q.d, W[l].q.sc16, W[l].k.d, W[l].k.sc16, W[l].v.d, W[l].v.sc16, H, Q, KV, chunk, st), "qkv");

            // Q/K head norms + RoPE per position
            for (int m = 0; m < chunk; ++m) {
                int pos = offset + m;
                head_norm_kernel<<<nqh,128,0,st>>>(d_Q_batch_prefill + m * Q, W[l].qn, nqh, hd, eps);
                head_norm_kernel<<<nkv,128,0,st>>>(d_K_batch_prefill + m * KV, W[l].kn, nkv, hd, eps);
                apply_rope_kernel<<<nqh,hd/2,0,st>>>(d_Q_batch_prefill + m * Q, nqh, hd, pos);
                apply_rope_kernel<<<nkv,hd/2,0,st>>>(d_K_batch_prefill + m * KV, nkv, hd, pos);
            }

            // Write KV cache for this chunk's positions (use _pos variant to avoid
            // pinned-buffer race in tight loop)
            size_t l_kv_off = (size_t)l * nkv * MAXSEQ * hd;
            for (int m = 0; m < chunk; ++m) {
                int pos = offset + m;
                die(blackwell::kernels::update_kv_cache_pos(d_kc + l_kv_off, d_vc + l_kv_off,
                    d_K_batch_prefill + m * KV, d_V_batch_prefill + m * KV,
                    pos, nkv, hd, MAXSEQ, st), "kv");
            }


            // Per-token attention: each token attends to all prior positions in decode cache
            for (int m = 0; m < chunk; ++m) {
                int pos = offset + m;
                die(blackwell::kernels::attention_decode_batched_gqa_pos(
                    d_attn_batch_prefill + m * Q,
                    d_Q_batch_prefill + m * Q,
                    d_kc, d_vc, pos, nqh, nkv, hd, MAXSEQ, 1,
                    (size_t)NL * nkv * MAXSEQ * hd, l_kv_off, st), "attn");
            }

            // Wo projection (batched)
            die(blackwell::kernels::quantize_int4_batched(d_attn_i4_batch, d_attn_i4_sc_batch, d_attn_batch_prefill, Q, chunk, st), "quant_attn");
            die(blackwell::kernels::gemv_int4_batched_f16wsc(d_proj_batch, (const uint8_t*)d_attn_i4_batch, d_attn_i4_sc_batch, W[l].o.d, W[l].o.sc16, Q, H, chunk, st), "o_proj");

            // Residual add
            for (int m = 0; m < chunk; ++m) {
                die(blackwell::kernels::vector_add_fp32(d_x32_batch + m * H, d_proj_batch + m * H, d_res_batch + m * H, H, st), "attn_res");
            }

            // Save for MLP residual
            die(cudaMemcpyAsync(d_res_batch, d_x32_batch, (size_t)chunk * H * 4, cudaMemcpyDeviceToDevice, st), "save_res2");

            // Pre-MLP norm + quantize
            die(blackwell::kernels::fused_rmsnorm_batched(d_xi_f_batch, d_x32_batch, W[l].rn_post, H, eps, chunk, st), "rmsnorm_post");
            die(blackwell::kernels::quantize_int4_batched(d_x_i4_batch, d_x_i4_sc_batch, d_xi_f_batch, H, chunk, st), "quant_mlp_in");

            // MLP gate + up (batched)
            die(blackwell::kernels::fused_gate_up_int4_f16wsc(d_gate_batch, d_up_batch, d_x_i4_batch, d_x_i4_sc_batch, W[l].g.d, W[l].g.sc16, W[l].u.d, W[l].u.sc16, H, I, chunk, st), "gate_up");

            // SwiGLU (per-token)
            for (int m = 0; m < chunk; ++m) {
                blackwell::kernels::apply_swiglu(d_gate_batch + m * I, d_gate_batch + m * I, d_up_batch + m * I, I, st);
            }
            die(blackwell::kernels::quantize_int4_batched(d_mlp_i4_batch, d_mlp_i4_sc_batch, d_gate_batch, I, chunk, st), "quant_mlp");

            // Down projection (batched)
            die(blackwell::kernels::gemv_int4_batched_f16wsc(d_proj_batch, (const uint8_t*)d_mlp_i4_batch, d_mlp_i4_sc_batch, W[l].d.d, W[l].d.sc16, I, H, chunk, st), "down");

            // MLP residual
            for (int m = 0; m < chunk; ++m) {
                die(blackwell::kernels::vector_add_fp32(d_x32_batch + m * H, d_proj_batch + m * H, d_res_batch + m * H, H, st), "mlp_res");
            }

        }

        offset += chunk;
        M -= chunk;
    }

    // Copy last token's hidden state to d_x32 for lm_head
    // last_chunk tokens are at indices 0..last_chunk-1 in d_x32_batch
    die(cudaMemcpyAsync(d_x32, d_x32_batch + (last_chunk - 1) * H, H * 4, cudaMemcpyDeviceToDevice, st), "last_hidden");
    die(cudaStreamSynchronize(st), "sync");

}

// Batched decode: process M tokens through all layers in one pass.
// M=1 must produce identical output to decode_one_token.
// Outputs M hidden states in d_x32_batch[m*H..(m+1)*H].
static void decode_one_token_batched(const uint32_t* token_ids, int step, int M) {
    // Embed M tokens (H2D copy per token)
    for (int m = 0; m < M; ++m) {
        std::vector<float> h_embed(H);
        dequant_embed_row(h_embed.data(), token_ids[m], host_embed_d, host_embed_sc, H);
        die(cudaMemcpyAsync(d_x32_batch + m * H, h_embed.data(), H*4, cudaMemcpyHostToDevice, st), "embed_b");
    }

    for (int l = 0; l < NL; ++l) {
        // Save residual for all M sequences (single large copy)
        die(cudaMemcpyAsync(d_res_batch, d_x32_batch, (size_t)M * H * 4, cudaMemcpyDeviceToDevice, st), "save_res_b");

        // Pre-attention norm + quantize (batched)
        die(blackwell::kernels::fused_rmsnorm_batched(d_xi_f_batch, d_x32_batch, W[l].rn_in, H, eps, M, st), "rmsnorm_in_b");
        die(blackwell::kernels::quantize_int4_batched(d_x_i4_batch, d_x_i4_sc_batch, d_xi_f_batch, H, M, st), "quant_in_b");

        // QKV projections (batched GEMV)
        die(blackwell::kernels::fused_qkv_int4_f16wsc(d_Q_batch, d_K_batch, d_V_batch, (const uint8_t*)d_x_i4_batch, d_x_i4_sc_batch, W[l].q.d, W[l].q.sc16, W[l].k.d, W[l].k.sc16, W[l].v.d, W[l].v.sc16, H, Q, KV, M, st), "qkv_b");

        // Q/K head norms + RoPE (per-sequence — same step for all in speculative verify)
        for (int m = 0; m < M; ++m) {
            head_norm_kernel<<<nqh,128,0,st>>>(d_Q_batch + m * Q, W[l].qn, nqh, hd, eps);
            head_norm_kernel<<<nkv,128,0,st>>>(d_K_batch + m * KV, W[l].kn, nkv, hd, eps);
            apply_rope_kernel<<<nqh,hd/2,0,st>>>(d_Q_batch + m * Q, nqh, hd, step);
            apply_rope_kernel<<<nkv,hd/2,0,st>>>(d_K_batch + m * KV, nkv, hd, step);
        }

        // KV cache: each sequence writes to its own cache slot
        size_t l_kv_off = (size_t)l * nkv * MAXSEQ * hd;
        for (int m = 0; m < M; ++m) {
            size_t kv_off = (size_t)m * NL * nkv * MAXSEQ * hd + l_kv_off;
            die(blackwell::kernels::update_kv_cache(d_kc + kv_off, d_vc + kv_off, d_K_batch + m * KV, d_V_batch + m * KV, 0, step, nkv, hd, MAXSEQ, st), "kv_b");
        }

        // Attention (per-sequence — batched attention with M>1 may be non-deterministic)
        for (int m = 0; m < M; ++m) {
            size_t m_kv_off = (size_t)m * NL * nkv * MAXSEQ * hd;
            die(blackwell::kernels::attention_decode_batched_gqa(d_attn_batch + m * Q, d_Q_batch + m * Q,
                d_kc + m_kv_off, d_vc + m_kv_off, step, nqh, nkv, hd, MAXSEQ, 1,
                (size_t)nkv * MAXSEQ * hd, l_kv_off, st), "attn_b");
        }

        // Wo projection (batched)
        die(blackwell::kernels::quantize_int4_batched(d_attn_i4_batch, d_attn_i4_sc_batch, d_attn_batch, Q, M, st), "quant_attn_b");
        die(blackwell::kernels::gemv_int4_batched_f16wsc(d_proj_batch, (const uint8_t*)d_attn_i4_batch, d_attn_i4_sc_batch, W[l].o.d, W[l].o.sc16, Q, H, M, st), "o_proj_b");

        // Residual add (per-sequence)
        for (int m = 0; m < M; ++m) {
            die(blackwell::kernels::vector_add_fp32(d_x32_batch + m * H, d_proj_batch + m * H, d_res_batch + m * H, H, st), "attn_res_b");
        }

        // Save residual for MLP
        die(cudaMemcpyAsync(d_res_batch, d_x32_batch, (size_t)M * H * 4, cudaMemcpyDeviceToDevice, st), "save_res2_b");

        // Pre-MLP norm + quantize (batched)
        die(blackwell::kernels::fused_rmsnorm_batched(d_xi_f_batch, d_x32_batch, W[l].rn_post, H, eps, M, st), "rmsnorm_post_b");
        die(blackwell::kernels::quantize_int4_batched(d_x_i4_batch, d_x_i4_sc_batch, d_xi_f_batch, H, M, st), "quant_mlp_in_b");

        // MLP gate + up (batched)
        die(blackwell::kernels::fused_gate_up_int4_f16wsc(d_gate_batch, d_up_batch, d_x_i4_batch, d_x_i4_sc_batch, W[l].g.d, W[l].g.sc16, W[l].u.d, W[l].u.sc16, H, I, M, st), "gate_up_b");

        // SwiGLU (per-sequence)
        for (int m = 0; m < M; ++m) {
            blackwell::kernels::apply_swiglu(d_gate_batch + m * I, d_gate_batch + m * I, d_up_batch + m * I, I, st);
        }

        // Quantize MLP output + down projection (batched)
        die(blackwell::kernels::quantize_int4_batched(d_mlp_i4_batch, d_mlp_i4_sc_batch, d_gate_batch, I, M, st), "quant_mlp_b");
        die(blackwell::kernels::gemv_int4_batched_f16wsc(d_proj_batch, (const uint8_t*)d_mlp_i4_batch, d_mlp_i4_sc_batch, W[l].d.d, W[l].d.sc16, I, H, M, st), "down_b");

        // Final residual add (per-sequence)
        for (int m = 0; m < M; ++m) {
            die(blackwell::kernels::vector_add_fp32(d_x32_batch + m * H, d_proj_batch + m * H, d_res_batch + m * H, H, st), "mlp_res_b");
        }
    }
}

static std::vector<uint32_t> generate(const std::vector<uint32_t>& input_ids,
                                       int max_new, float temperature, int top_k, float rep_pen,
                                       bool streaming = false) {
    std::vector<uint32_t> all_ids = input_ids;
    int gen_start = (int)input_ids.size();
    int total = gen_start + max_new;

    cudaMemset(d_kc,0,(size_t)NL*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vc,0,(size_t)NL*nkv*MAXSEQ*hd*4);

    // ── Phase 1: Prefill — process prompt tokens ──
    // Single-chunk batched prefill for short prompts (≤ MAX_BATCH).
    // Long prompts use original per-token decode.
    if (gen_start > 0) {
        prefill_tokens_batched(input_ids, 0, gen_start);
    }

    // Capture CUDA Graph for decode loop once (after prefill to set KV cache)
    // Skip if already captured at startup
    if (!graph_captured) {
        capture_decode_graph();
    }

    // ── Phase 2: Decode — generate new tokens one at a time ──
    for (int step = gen_start; step < total; ++step) {
        uint32_t tid = all_ids.back();
        decode_one_token(tid, step);

        // lm_head + sampling
        die(blackwell::kernels::fused_rmsnorm(d_xi_f, d_x32, d_fn, H, eps, st), "fn");
        die(blackwell::kernels::quantize_int4_batched(d_x_i4, d_x_i4_sc, d_xi_f, H, 1, st), "quant_lm");
        die(blackwell::kernels::gemv_int4_batched_f16wsc(d_logits, (const uint8_t*)d_x_i4, d_x_i4_sc, lm_head_w.d, lm_head_w.sc16, H, V, 1, st), "lm_head");

        // Repetition penalty
        if (rep_pen > 1.0f && (int)all_ids.size() > gen_start) {
            int num_recent = (int)all_ids.size() - gen_start;
            if (num_recent > 64) num_recent = 64;
            std::vector<int> h_recent(all_ids.end() - num_recent, all_ids.end());
            die(cudaMemcpyAsync(d_recent, h_recent.data(), num_recent * sizeof(int), cudaMemcpyHostToDevice, st), "cpy_recent");
            die(blackwell::kernels::apply_repetition_penalty(d_logits, d_recent, num_recent, rep_pen, V, st), "rep_pen");
        }

        int next_id;
        die(blackwell::kernels::sample_gpu(d_logits, V, temperature, top_k, d_next_id, 0xdeadbeefLL, step, st), "sample");
        die(cudaMemcpy(&next_id, d_next_id, 4, cudaMemcpyDeviceToHost), "copy");
        all_ids.push_back(next_id);

        // SSE streaming
        if (streaming) {
            std::string tok_text = tokenizer.decode(next_id);
            std::string escaped = json_escape(tok_text);
            printf("data: {\"token\":%u,\"text\":\"%s\"}\n\n", next_id, escaped.c_str());
            fflush(stdout);
        }

        if (next_id == 151643) break;
    }

    return std::vector<uint32_t>(all_ids.begin() + gen_start, all_ids.end());
}

// ── JSON escaping ──
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
    const char* model = argc > 1 ? argv[1] : "8b";

    fprintf(stderr, "Blackwell INT4 Server v0.9.1\n");
    cudaDeviceProp P; cudaGetDeviceProperties(&P, 0);
    fprintf(stderr, "Device: %s (CC %d.%d)\n", P.name, P.major, P.minor);

    die(cudaStreamCreate(&st), "stream");

    if (tokenizer.load("tokenizer_data.bin") != 0) {
        fprintf(stderr, "FAIL: no tokenizer_data.bin\n"); return 1;
    }

    load_model();
    alloc_buffers();

    // Warm up
    fprintf(stderr, "[WARMUP] Running...\n"); fflush(stdout);
    {
        std::vector<uint32_t> warmup = tokenizer.encode("Hello");
        generate(warmup, 1, 0.0f, 0, 1.0f);
        cudaDeviceSynchronize();
    }
    fprintf(stderr, "[WARMUP] Done\n"); fflush(stdout);

    fprintf(stderr, "Ready.\n"); fflush(stdout);

    // ── Main request loop ──
    while (true) {
        std::string line = read_stdin_line();
        if (line.empty()) continue;
        fprintf(stderr, "[REQ] %s\n", line.c_str());

        int max_tokens = parse_int(line, "\"max_tokens\"", 30);
        float temperature = parse_float(line, "\"temperature\"", 0.0f);
        int top_k = parse_int(line, "\"top_k\"", 0);
        int stream_flag = parse_int(line, "\"stream\"", 0);

        auto str_prompts = parse_string_prompts(line);
        if (str_prompts.empty()) {
            printf("{\"error\":\"no prompts\"}\n"); fflush(stdout); continue;
        }

        // Process all prompts sequentially
        float rep_pen = parse_repetition_penalty(line, 1.5f);
        std::vector<std::vector<uint32_t>> all_gen_tokens;
        std::vector<std::string> all_texts;
        
        for (size_t pi = 0; pi < str_prompts.size(); pi++) {
            auto input_ids = tokenizer.encode(str_prompts[pi]);
            std::string text;

            if (stream_flag) {
                auto gen_tokens = generate(input_ids, max_tokens, temperature, top_k, rep_pen, true);
                for (auto id : gen_tokens) text += tokenizer.decode(id);
                all_gen_tokens.push_back(std::move(gen_tokens));
            } else {
                auto gen_tokens = generate(input_ids, max_tokens, temperature, top_k, rep_pen);
                for (auto id : gen_tokens) text += tokenizer.decode(id);
                all_gen_tokens.push_back(std::move(gen_tokens));
            }

            all_texts.push_back(std::move(text));
        }

        if (stream_flag) {
            // Streaming: emit [DONE] after all sequences
            printf("data: [DONE]\n\n"); fflush(stdout);
        } else {
            // Output JSON
            printf("{\"tokens\":[");
            for (size_t pi = 0; pi < all_gen_tokens.size(); pi++) {
                if (pi) printf(",");
                printf("[");
                for (size_t i = 0; i < all_gen_tokens[pi].size(); i++) {
                    if (i) printf(",");
                    printf("%u", all_gen_tokens[pi][i]);
                }
                printf("]");
            }
            printf("],\"text\":[");
            for (size_t pi = 0; pi < all_texts.size(); pi++) {
                if (pi) printf(",");
                printf("\"%s\"", json_escape(all_texts[pi]).c_str());
            }
            printf("]}\n"); fflush(stdout);
        }
    }
    return 0;
}
