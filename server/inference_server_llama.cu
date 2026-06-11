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

// Model configuration
struct ModelConfig {
    const char* wdir;
    int NL, H, Q, KV, I, nqh, nkv, hd, V, MAXSEQ, eos_id;
    float eps, rope_theta;
};

static const ModelConfig MODELS[] = {
    // Llama 3.2 1B (from safetensors)
    {"/mnt/data/ai/models/llama32-1b-int4-from-safetensors", 16, 2048, 2048, 512, 8192, 32, 8, 64, 128256, 4096, 128001, 1e-6f, 500000.0f},
    // Llama 3.1 8B (from safetensors, MAXSEQ=2048 for 16GB GPU)
    {"/mnt/data/ai/models/llama31-8b-int4-from-safetensors", 32, 4096, 4096, 1024, 14336, 32, 8, 128, 128256, 2048, 128001, 1e-6f, 500000.0f},
    {NULL, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0f, 0.0f}
};

static const ModelConfig* get_config(const char* model) {
    // Try full path match first
    for (int i = 0; MODELS[i].wdir != NULL; i++) {
        if (strcmp(model, MODELS[i].wdir) == 0) return &MODELS[i];
    }
    // Try short name as substring of full path
    for (int i = 0; MODELS[i].wdir != NULL; i++) {
        if (strstr(MODELS[i].wdir, model)) return &MODELS[i];
    }
    fprintf(stderr, "Unknown model: %s\n", model);
    return NULL;
}

// Global config (set at startup)
static const ModelConfig* cfg = NULL;

// Accessors for config (avoid macro conflicts with CUDA kernel params)
inline int CFG_NL() { return cfg->NL; }
inline int CFG_H() { return cfg->H; }
inline int CFG_Q() { return cfg->Q; }
inline int CFG_KV() { return cfg->KV; }
inline int CFG_I() { return cfg->I; }
inline int CFG_nqh() { return cfg->nqh; }
inline int CFG_nkv() { return cfg->nkv; }
inline int CFG_hd() { return cfg->hd; }
inline int CFG_V() { return cfg->V; }
inline int CFG_MAXSEQ() { return cfg->MAXSEQ; }
inline int CFG_EOS_ID() { return cfg->eos_id; }
inline float CFG_eps() { return cfg->eps; }
inline float CFG_rope_theta() { return cfg->rope_theta; }
inline const char* CFG_wdir() { return cfg->wdir; }

#define NL   CFG_NL()
#define H    CFG_H()
#define Q    CFG_Q()
#define KV   CFG_KV()
#define I    CFG_I()
#define nqh  CFG_nqh()
#define nkv  CFG_nkv()
#define hd   CFG_hd()
#define V    CFG_V()
#define MAXSEQ CFG_MAXSEQ()
#define EOS_ID CFG_EOS_ID()
#define eps  CFG_eps()
#define rope_theta CFG_rope_theta()
#define wdir CFG_wdir()

// INT4 weight struct + loader
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

struct LW4 { DevW4 q,k,v,o,g,u,d; float* qn,*kn,*rn_in,*rn_post; };

// ── CUDA kernels (same as benchmark) ──
__global__ void head_norm_kernel(float* data, const float* weight, int nh, int head_dim, float epsilon) {
    int h=blockIdx.x; if(h>=nh) return;
    float* d=data+h*head_dim;
    __shared__ float wp[4]; float s=0;
    for(int i=threadIdx.x;i<head_dim;i+=blockDim.x) s+=d[i]*d[i];
    for(int off=16;off>0;off>>=1) s+=__shfl_xor_sync(0xffffffff,s,off);
    if((threadIdx.x&31)==0) wp[threadIdx.x>>5]=s; __syncthreads();
    if(threadIdx.x<4) s=wp[threadIdx.x]; else s=0;
    for(int off=2;off>0;off>>=1) s+=__shfl_xor_sync(0xffffffff,s,off);
    if(threadIdx.x==0) wp[0]=rsqrtf(s/head_dim+epsilon); __syncthreads();
    float is=wp[0];
    for(int i=threadIdx.x;i<head_dim;i+=blockDim.x) d[i]=d[i]*is*weight[i];
}

__global__ void apply_rope_kernel(float* data, int n_heads, int head_dim, int pos, float rope_theta_val) {
    int h=blockIdx.x; int d=threadIdx.x;
    if(h>=n_heads||d>=head_dim/2) return;
    float* pair=data+h*head_dim+d*2;
    float theta=(float)pos*powf(rope_theta_val,-2.0f*(float)d/(float)head_dim);
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

// ── JSON helpers ──
static std::string json_escape_str(const std::string& s) {
    // Inline version of json_escape for use in generate()
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
static float *d_x32, *d_xi_f, *d_res;
static uint8_t *d_x_i4; static float *d_x_i4_sc;
static float *d_Q,*d_K,*d_V,*d_attn;
static uint8_t *d_attn_i4; static float *d_attn_i4_sc;
static float *d_proj, *d_gate, *d_up;
static uint8_t *d_mlp_i4; static float *d_mlp_i4_sc;
static float *d_fn, *d_kc, *d_vc, *d_logits;
static int *d_next_id, *d_recent;
static cudaStream_t st;

static std::vector<LW4> W;
static DevW4 embed_w, lm_head_w;
static uint8_t* host_embed_d;
static float* host_embed_sc;
static blackwell::BpeTokenizer tokenizer;

#define AL(p,n) die(cudaMalloc(&(p),(n)),"malloc " #p)

static void load_model() {
    fprintf(stderr, "Loading %d-layer INT4 model...\n", NL);
    W.resize(NL);
    char p[256];
    for (int l = 0; l < NL; ++l) {
        snprintf(p,256,"%s/%d_self_attn.q_proj",wdir,l); W[l].q=upload_w4(p);
        snprintf(p,256,"%s/%d_self_attn.k_proj",wdir,l); W[l].k=upload_w4(p);
        snprintf(p,256,"%s/%d_self_attn.v_proj",wdir,l); W[l].v=upload_w4(p);
        snprintf(p,256,"%s/%d_self_attn.o_proj",wdir,l); W[l].o=upload_w4(p);
        snprintf(p,256,"%s/%d_mlp.gate_proj",wdir,l); W[l].g=upload_w4(p);
        snprintf(p,256,"%s/%d_mlp.up_proj",wdir,l);   W[l].u=upload_w4(p);
        snprintf(p,256,"%s/%d_mlp.down_proj",wdir,l); W[l].d=upload_w4(p);
        if ((l+1)%7==0) fprintf(stderr, "  layer %d/%d\n", l+1, NL);
    }
    // QK norms
    float* qk_h=(float*)malloc(NL*2*hd*4);
    {char qp[256]; snprintf(qp,256,"%s/qk_norms.f32",wdir); FILE*f=fopen(qp,"rb");(void)fread(qk_h,4,NL*2*hd,f);fclose(f);}
    for(int l=0;l<NL;++l){
        cudaMalloc(&W[l].qn,hd*4);cudaMemcpy(W[l].qn,qk_h+l*2*hd,hd*4,cudaMemcpyHostToDevice);
        cudaMalloc(&W[l].kn,hd*4);cudaMemcpy(W[l].kn,qk_h+l*2*hd+hd,hd*4,cudaMemcpyHostToDevice);
    } free(qk_h);
    // Per-layer RMSNorm
    for(int l=0;l<NL;++l){
        float* w=(float*)malloc(H*4);
        snprintf(p,256,"%s/%d_input_layernorm.f32",wdir,l);
        {FILE*f=fopen(p,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&W[l].rn_in,H*4);cudaMemcpy(W[l].rn_in,w,H*4,cudaMemcpyHostToDevice);
        snprintf(p,256,"%s/%d_post_attention_layernorm.f32",wdir,l);
        {FILE*f=fopen(p,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&W[l].rn_post,H*4);cudaMemcpy(W[l].rn_post,w,H*4,cudaMemcpyHostToDevice);
        free(w);
    }
    // Final norm
    {float*w=(float*)malloc(H*4);
     char fn[256]; snprintf(fn,256,"%s/final_norm.f32",wdir);
     FILE*f=fopen(fn,"rb");(void)fread(w,4,H,f);fclose(f);
     AL(d_fn,H*4); cudaMemcpy(d_fn,w,H*4,cudaMemcpyHostToDevice); free(w);}
    // Embed + lm_head
    {char ep[256]; snprintf(ep,256,"%s/embed_tokens",wdir); embed_w=upload_w4(ep);}
    {char lp[256]; snprintf(lp,256,"%s/lm_head",wdir); FILE* flm=fopen(lp,"rb"); if(flm){fclose(flm); lm_head_w=upload_w4(lp);} else { lm_head_w=embed_w; fprintf(stderr,"lm_head tied to embed\n");}}
    fprintf(stderr,"  embed: %dx%d, lm_head: %dx%d\n",embed_w.K,embed_w.N,lm_head_w.K,lm_head_w.N);
    // Host embed for CPU dequant
    host_embed_d=new uint8_t[(size_t)embed_w.K*embed_w.N/2];
    host_embed_sc=new float[embed_w.N*(embed_w.K/16)];
    {char ep_i4[256], ep_sc[256]; snprintf(ep_i4,256,"%s/embed_tokens.int4_t",wdir); snprintf(ep_sc,256,"%s/embed_tokens.scale_t",wdir);
     FILE*f=fopen(ep_i4,"rb");int h[5];fread(h,4,5,f);
     size_t ds=(size_t)h[0]*h[1]/2;fread(host_embed_d,1,ds,f);fclose(f);
     f=fopen(ep_sc,"rb");fread(h,4,5,f);
     size_t ss=(size_t)h[3]*h[4];fread(host_embed_sc,4,ss,f);fclose(f);}
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
    float iv7=1.f/7.f;
    { std::vector<float> tmp(H/16,iv7); cudaMemcpy(d_x_i4_sc,tmp.data(),(H/16)*4,cudaMemcpyHostToDevice); }
    { std::vector<float> tmp(Q/16,iv7); cudaMemcpy(d_attn_i4_sc,tmp.data(),(Q/16)*4,cudaMemcpyHostToDevice); }
    { std::vector<float> tmp(I/16,iv7); cudaMemcpy(d_mlp_i4_sc,tmp.data(),(I/16)*4,cudaMemcpyHostToDevice); }
}

// ── Generate: exact decode loop from benchmark ──
static std::vector<uint32_t> generate(const std::vector<uint32_t>& input_ids,
                                       int max_new, float temperature, int top_k, float rep_pen,
                                       bool streaming = false, int seq_idx = 0, int nseq = 1) {
    std::vector<uint32_t> all_ids = input_ids;
    int gen_start = (int)input_ids.size();
    int total = gen_start + max_new;
    std::vector<float> h_embed(H);

    cudaMemset(d_kc,0,(size_t)NL*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vc,0,(size_t)NL*nkv*MAXSEQ*hd*4);

    for (int step = 0; step < total; ++step) {
        uint32_t tid = (step < gen_start) ? input_ids[step] : all_ids.back();
        dequant_embed_row(h_embed.data(), tid, host_embed_d, host_embed_sc, H);
        die(cudaMemcpyAsync(d_x32, h_embed.data(), H*4, cudaMemcpyHostToDevice, st), "embed");

        for (int l = 0; l < NL; ++l) {
            die(cudaMemcpyAsync(d_res, d_x32, H*4, cudaMemcpyDeviceToDevice, st), "save_res");
            die(blackwell::kernels::fused_rmsnorm(d_xi_f, d_x32, W[l].rn_in, H, eps, st), "rmsnorm_in");
            die(blackwell::kernels::quantize_int4_batched(d_x_i4, d_x_i4_sc, d_xi_f, H, 1, st), "quant_in");
            die(blackwell::kernels::gemv_int4_batched(d_Q, (const uint8_t*)d_x_i4, d_x_i4_sc, W[l].q.d, W[l].q.sc, H, Q, 1, st), "q_proj");
            die(blackwell::kernels::gemv_int4_batched(d_K, (const uint8_t*)d_x_i4, d_x_i4_sc, W[l].k.d, W[l].k.sc, H, KV, 1, st), "k_proj");
            die(blackwell::kernels::gemv_int4_batched(d_V, (const uint8_t*)d_x_i4, d_x_i4_sc, W[l].v.d, W[l].v.sc, H, KV, 1, st), "v_proj");
            head_norm_kernel<<<nqh,128,0,st>>>(d_Q, W[l].qn, nqh, hd, eps);
            die(cudaGetLastError(), "head_norm_Q");
            head_norm_kernel<<<nkv,128,0,st>>>(d_K, W[l].kn, nkv, hd, eps);
            die(cudaGetLastError(), "head_norm_K");
            apply_rope_kernel<<<nqh,hd/2,0,st>>>(d_Q, nqh, hd, step, rope_theta);
            die(cudaGetLastError(), "rope_Q");
            apply_rope_kernel<<<nkv,hd/2,0,st>>>(d_K, nkv, hd, step, rope_theta);
            die(cudaGetLastError(), "rope_K");
            size_t kv_off = (size_t)l * nkv * MAXSEQ * hd;
            die(blackwell::kernels::update_kv_cache(d_kc+kv_off, d_vc+kv_off, d_K, d_V, 0, step, nkv, hd, MAXSEQ, st), "kv");
            die(blackwell::kernels::attention_decode_batched_gqa(d_attn, d_Q, d_kc, d_vc, step, nqh, nkv, hd, MAXSEQ, 1,
                (size_t)NL*nkv*MAXSEQ*hd, kv_off, st), "attn");
            die(blackwell::kernels::quantize_int4_batched(d_attn_i4, d_attn_i4_sc, d_attn, Q, 1, st), "quant_attn");
            die(blackwell::kernels::gemv_int4_batched(d_proj, (const uint8_t*)d_attn_i4, d_attn_i4_sc, W[l].o.d, W[l].o.sc, Q, H, 1, st), "o_proj");
            die(blackwell::kernels::vector_add_fp32(d_x32, d_proj, d_res, H, st), "attn_res");
            die(cudaMemcpyAsync(d_res, d_x32, H*4, cudaMemcpyDeviceToDevice, st), "save_res2");
            die(blackwell::kernels::fused_rmsnorm(d_xi_f, d_x32, W[l].rn_post, H, eps, st), "rmsnorm_post");
            die(blackwell::kernels::quantize_int4_batched(d_x_i4, d_x_i4_sc, d_xi_f, H, 1, st), "quant_mlp_in");
            die(blackwell::kernels::gemv_int4_batched(d_gate, (const uint8_t*)d_x_i4, d_x_i4_sc, W[l].g.d, W[l].g.sc, H, I, 1, st), "gate");
            die(blackwell::kernels::gemv_int4_batched(d_up,   (const uint8_t*)d_x_i4, d_x_i4_sc, W[l].u.d, W[l].u.sc, H, I, 1, st), "up");
            blackwell::kernels::apply_swiglu(d_gate, d_gate, d_up, I, st);
            die(blackwell::kernels::quantize_int4_batched(d_mlp_i4, d_mlp_i4_sc, d_gate, I, 1, st), "quant_mlp");
            die(blackwell::kernels::gemv_int4_batched(d_proj, (const uint8_t*)d_mlp_i4, d_mlp_i4_sc, W[l].d.d, W[l].d.sc, I, H, 1, st), "down");
            die(blackwell::kernels::vector_add_fp32(d_x32, d_proj, d_res, H, st), "mlp_res");
        }

        if (step >= gen_start - 1) {
            die(blackwell::kernels::fused_rmsnorm(d_xi_f, d_x32, d_fn, H, eps, st), "fn");
            die(blackwell::kernels::quantize_int4_batched(d_x_i4, d_x_i4_sc, d_xi_f, H, 1, st), "quant_lm");
            die(blackwell::kernels::gemv_int4_batched(d_logits, (const uint8_t*)d_x_i4, d_x_i4_sc, lm_head_w.d, lm_head_w.sc, H, V, 1, st), "lm_head");

            // Repetition penalty: penalize recently generated tokens
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

            // SSE streaming: emit per-token JSON
            if (streaming) {
                std::string tok_text = tokenizer.decode(next_id);
                std::string escaped = json_escape_str(tok_text);
                printf("data: {\"token\":%u,\"text\":\"%s\"}\n\n", next_id, escaped.c_str());
                fflush(stdout);
            }

            if (next_id == CFG_EOS_ID()) break;
        }
    }
    // Return only generated tokens
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
    const char* model = argc > 1 ? argv[1] : "llama32-1b";

    // Set up config
    cfg = get_config(model);
    if (!cfg) { fprintf(stderr, "Unknown model: %s\n", model); return 1; }

    fprintf(stderr, "Blackwell INT4 Llama Server\n");
    fprintf(stderr, "  Model: %s\n", model);
    fprintf(stderr, "  Config: NL=%d H=%d I=%d nqh=%d nkv=%d hd=%d V=%d rope=%.0f\n",
            NL, H, I, nqh, nkv, hd, V, rope_theta);
    cudaDeviceProp P; cudaGetDeviceProperties(&P, 0);
    fprintf(stderr, "Device: %s (CC %d.%d)\n", P.name, P.major, P.minor);

    die(cudaStreamCreate(&st), "stream");

    // Load tokenizer from weight directory
    char tok_path[256];
    snprintf(tok_path, sizeof(tok_path), "%s/tokenizer_data.bin", wdir);
    if (tokenizer.load(tok_path) != 0) {
        fprintf(stderr, "FAIL: no %s\n", tok_path); return 1;
    }

    load_model();
    alloc_buffers();

    // Warm up: run dummy inference to JIT-compile all CUDA kernels
    // This ensures first real request has consistent timing and output
    fprintf(stderr, "[WARMUP] Running dummy inference...\n"); fflush(stdout);
    {
        std::vector<uint32_t> dummy(1, 0); // Valid token (0) as dummy input
        auto warmup = generate(dummy, 1, 0.0f, 0, 1.0f);

    }
    fprintf(stderr, "[WARMUP] Done.\n"); fflush(stdout);

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
                // Streaming mode: generate emits SSE per token
                auto gen_tokens = generate(input_ids, max_tokens, temperature, top_k, rep_pen, true, (int)pi, (int)str_prompts.size());
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
            // Non-streaming: output JSON
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
