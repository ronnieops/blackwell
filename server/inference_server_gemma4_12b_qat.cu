#include "blackwell/int4_weights.h"
using namespace blackwell::weights;
// server/inference_server_gemma4_12b_qat.cu — Gemma 4 12B QAT INT4 inference server (JSON stdio)
// 48 layers, H=3840, I=15360, nqh=16, nkv=8, hd=256, V=262144, rope_theta=10000.0
// GeGLU activation. 8 full-attention layers (5,11,17,23,29,35,41,47) + 40 SWA layers.
// SWA window=1024. FA layers: double Q heads (32), K=V (k_eq_v, 2 heads), no v_proj.
// FP16 weight scales (f16sc) upload path.
//
// Protocol: reads JSON from stdin, writes JSON to stdout.
// Input:  {"prompts":["str1","str2"],"max_tokens":N,"temperature":T,"top_k":K,"repetition_penalty":P}
// Output: {"tokens":[[id1,...],[...]],"text":["text1","text2"]}
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_120a server/inference_server_gemma4_12b_qat.cu \
//     build/libblackwell_kernels.a -I include -lcudart -lpthread -lz \
//     -o server/inference_server_gemma4_12b_qat

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

static void die(cudaError_t e, const char* m) {
    if(e!=cudaSuccess){fprintf(stderr,"FAIL %s: %s\n",m,cudaGetErrorString(e));exit(1);}
}

const int H=3840, Q=4096, KV=2048, I=15360;
const int nqh=16, nkv=8, hd=256, MAXSEQ=2048;
const int SWA_WINDOW=1024;
const float eps=1e-6f;
const int V=262144;
const int NL=48;
const float rope_theta=10000.0f;
// Full-attention layers (non-SWA) at indexes 5,11,17,23,29,35,41,47
// SWA layers have 1, full-attention have 0
const bool SWA_LAYERS[NL] = {
    1,1,1,1,1,0, 1,1,1,1,1,0,
    1,1,1,1,1,0, 1,1,1,1,1,0,
    1,1,1,1,1,0, 1,1,1,1,1,0,
    1,1,1,1,1,0, 1,1,1,1,1,0};

struct LW4 { DevW4f16 q,k,v,o,g,u,d; float *qn,*kn,*rn_in,*rn_post; bool swa; };
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
    for (int pos = 0; pos < max_seq; ++pos)
        for (int d = 0; d < pairs; ++d) {
            float theta = (float)pos * powf(rope_theta, -2.0f * (float)d / (float)head_dim);
            cos_cache[pos * pairs + d] = cosf(theta);
            sin_cache[pos * pairs + d] = sinf(theta);
        }
}

// ── JSON helpers ──
static std::string json_escape(const std::string& s);

static std::string read_stdin_line() {
    std::string line; int c;
    while ((c = getchar()) != EOF && c != '\n') line.push_back((char)c);
    return line;
}

static std::vector<uint32_t> parse_token_ids(const std::string& json) {
    std::vector<uint32_t> ids;
    const char* p = json.c_str();
    // Look for "tokens" array
    p = strstr(json.c_str(), "\"tokens\"");
    if (!p) return ids;
    p = strchr(p, '['); if (!p) return ids; p++;
    while (*p && *p != ']') {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == ',') p++;
        if (*p >= '0' && *p <= '9') {
            ids.push_back((uint32_t)strtoul(p, (char**)&p, 10));
        } else break;
    }
    return ids;
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
    p += 18;
    while (*p == ' ' || *p == ':' || *p == '=') p++;
    return strtof(p, nullptr);
}

// ── Global state ──
static float *d_x32, *d_xi_f, *d_res;
static uint8_t *d_x_i4; static float *d_x_i4_sc;
static float *d_Q,*d_K,*d_V,*d_attn;
static uint8_t *d_attn_i4; static float *d_attn_i4_sc;
static float *d_proj, *d_gate, *d_up;
static uint8_t *d_mlp_i4; static float *d_mlp_i4_sc;
static float *d_fn, *d_kc, *d_vc, *d_logits;
static float *d_cos_cache, *d_sin_cache;
static int *d_next_id, *d_seq_pos;
static std::vector<LW4> W;
static DevW4f16 embed_tokens_w, lm_head_w;
static uint8_t* host_embed_d;
static float* host_embed_sc;
static cudaStream_t st;

static void allocate_buffers() {
    #define AL(p,n) die(cudaMalloc(&(p),(n)),"malloc")
    AL(d_x32,H*4); AL(d_xi_f,H*4); AL(d_res,H*4);
    AL(d_x_i4,H/2); AL(d_x_i4_sc,(H/16)*4);
    AL(d_Q,Q*4); AL(d_K,KV*4); AL(d_V,KV*4); AL(d_attn,Q*4);
    AL(d_attn_i4,Q/2); AL(d_attn_i4_sc,(Q/16)*4);
    AL(d_proj,H*4); AL(d_gate,I*4); AL(d_up,I*4);
    AL(d_mlp_i4,I/2); AL(d_mlp_i4_sc,(I/16)*4);
    AL(d_fn,H*4);
    AL(d_kc,(size_t)NL*nkv*MAXSEQ*hd*4);
    AL(d_vc,(size_t)NL*nkv*MAXSEQ*hd*4);
    AL(d_logits,V*4); AL(d_next_id,4); AL(d_seq_pos,4);
    int rope_pairs=hd/2;
    AL(d_cos_cache,(size_t)MAXSEQ*rope_pairs*4);
    AL(d_sin_cache,(size_t)MAXSEQ*rope_pairs*4);
    #undef AL

    float iv7=1.f/7.f;
    std::vector<float> tmp;
    tmp.assign(H/16,iv7); cudaMemcpy(d_x_i4_sc,tmp.data(),(H/16)*4,cudaMemcpyHostToDevice);
    tmp.assign(Q/16,iv7); cudaMemcpy(d_attn_i4_sc,tmp.data(),(Q/16)*4,cudaMemcpyHostToDevice);
    tmp.assign(I/16,iv7); cudaMemcpy(d_mlp_i4_sc,tmp.data(),(I/16)*4,cudaMemcpyHostToDevice);

    { // RoPE cache
        std::vector<float> cos_h(MAXSEQ*rope_pairs), sin_h(MAXSEQ*rope_pairs);
        build_rope_cache(cos_h.data(),sin_h.data(),MAXSEQ,hd);
        cudaMemcpy(d_cos_cache,cos_h.data(),(size_t)MAXSEQ*rope_pairs*4,cudaMemcpyHostToDevice);
        cudaMemcpy(d_sin_cache,sin_h.data(),(size_t)MAXSEQ*rope_pairs*4,cudaMemcpyHostToDevice);
    }
    die(cudaStreamCreate(&st), "stream");
}

static void load_weights(const char* wdir) {
    fprintf(stderr,"Loading Gemma 4 12B QAT weights from %s...\n",wdir);
    W.resize(NL);
    char p[256];
    for(int l=0;l<NL;++l){
        W[l].swa = SWA_LAYERS[l];
        snprintf(p,256,"%s/%d_self_attn.q_proj",wdir,l); W[l].q=upload_w4_f16sc(p);
        snprintf(p,256,"%s/%d_self_attn.k_proj",wdir,l); W[l].k=upload_w4_f16sc(p);
        snprintf(p,256,"%s/%d_self_attn.v_proj",wdir,l); W[l].v=upload_w4_f16sc(p);
        snprintf(p,256,"%s/%d_self_attn.o_proj",wdir,l); W[l].o=upload_w4_f16sc(p);
        snprintf(p,256,"%s/%d_mlp.gate_proj",wdir,l); W[l].g=upload_w4_f16sc(p);
        snprintf(p,256,"%s/%d_mlp.up_proj",wdir,l); W[l].u=upload_w4_f16sc(p);
        snprintf(p,256,"%s/%d_mlp.down_proj",wdir,l); W[l].d=upload_w4_f16sc(p);
    }

    // QK head norms: non-SWA layers load from file, SWA layers get identity
    for(int l=0;l<NL;++l){
        float* w=(float*)malloc(hd*4);
        if (W[l].swa) {
            for(int i=0;i<hd;i++) w[i]=1.0f;
            cudaMalloc(&W[l].qn,hd*4);cudaMemcpy(W[l].qn,w,hd*4,cudaMemcpyHostToDevice);
            cudaMalloc(&W[l].kn,hd*4);cudaMemcpy(W[l].kn,w,hd*4,cudaMemcpyHostToDevice);
        } else {
            snprintf(p,256,"%s/qk_norms.f32",wdir);
            FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"FAIL open %s (QK norms)\n",p);exit(1);}
            fseek(f, (long)(l*2*hd)*4, SEEK_SET);
            fread(w,4,hd,f);
            cudaMalloc(&W[l].qn,hd*4);cudaMemcpy(W[l].qn,w,hd*4,cudaMemcpyHostToDevice);
            fread(w,4,hd,f);
            cudaMalloc(&W[l].kn,hd*4);cudaMemcpy(W[l].kn,w,hd*4,cudaMemcpyHostToDevice);
            fclose(f);
        }
        free(w);
    }

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
    {float*w=(float*)malloc(H*4);
    snprintf(p,256,"%s/final_norm.f32",wdir);
    FILE*f=fopen(p,"rb");(void)fread(w,4,H,f);fclose(f);
    cudaMemcpy(d_fn,w,H*4,cudaMemcpyHostToDevice);free(w);}

    snprintf(p,256,"%s/lm_head",wdir); lm_head_w=upload_w4_f16sc(p);
    fprintf(stderr,"  lm_head: %dx%d (INT4)\n",lm_head_w.N,lm_head_w.K);

    snprintf(p,256,"%s/embed_tokens",wdir); embed_tokens_w=upload_w4_f16sc(p);
    fprintf(stderr,"  embed_tokens: %dx%d (INT4)\n",embed_tokens_w.N,embed_tokens_w.K);
}

static void load_host_embed(const char* wdir) {
    host_embed_d=new uint8_t[(size_t)H*V/2];
    host_embed_sc=new float[V*(H/16)];
    char p[256];
    snprintf(p,256,"%s/embed_tokens.int4_t",wdir);
    {FILE*f=fopen(p,"rb");int h[5];fread(h,4,5,f);
     fread(host_embed_d,1,(size_t)h[0]*h[1]/2,f);fclose(f);}
    snprintf(p,256,"%s/embed_tokens.scale_t",wdir);
    {FILE*f=fopen(p,"rb");int h[5];fread(h,4,5,f);
     __half*tmp=new __half[(size_t)h[3]*h[4]];fread(tmp,2,(size_t)h[3]*h[4],f);fclose(f);
     for(size_t i=0;i<(size_t)h[3]*h[4];++i) host_embed_sc[i]=__half2float(tmp[i]);delete[] tmp;}
    fprintf(stderr,"  embed table loaded: %dx%d\n",H,V);
}

// ── Forward: process one token through all layers ──
static void forward_token(int step, int* seq_pos) {
    cudaMemcpyAsync(d_seq_pos, seq_pos, 4, cudaMemcpyHostToDevice, st);
    for(int l=0;l<NL;++l){
        cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st);
        blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_in,H,eps,st);
        blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st);
        blackwell::kernels::gemv_int4_warp_f16wsc(d_Q,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].q.d,W[l].q.sc16,H,Q,st);
        blackwell::kernels::gemv_int4_warp_f16wsc(d_K,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].k.d,W[l].k.sc16,H,KV,st);
        blackwell::kernels::gemv_int4_warp_f16wsc(d_V,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].v.d,W[l].v.sc16,H,KV,st);

        // QK head norms
        blackwell::kernels::head_norm(d_Q,W[l].qn,nqh,hd,eps,st);
        blackwell::kernels::head_norm(d_K,W[l].kn,nkv,hd,eps,st);

        // RoPE
        blackwell::kernels::fused_rope_decode(d_Q,d_cos_cache,d_sin_cache,d_seq_pos,nqh,hd,MAXSEQ,st);
        blackwell::kernels::fused_rope_decode(d_K,d_cos_cache,d_sin_cache,d_seq_pos,nkv,hd,MAXSEQ,st);

        // SWA layers cap effective attention window
        int attn_step = step;
        if (W[l].swa && attn_step > SWA_WINDOW) attn_step = SWA_WINDOW;

        size_t kv_off=(size_t)l*nkv*MAXSEQ*hd;
        blackwell::kernels::update_kv_cache(d_kc+kv_off,d_vc+kv_off,d_K,d_V,0,step,nkv,hd,MAXSEQ,st);
        blackwell::kernels::attention_decode_batched_gqa(d_attn,d_Q,d_kc,d_vc,attn_step,nqh,nkv,hd,MAXSEQ,1,
            (size_t)NL*nkv*MAXSEQ*hd,kv_off,st);

        blackwell::kernels::quantize_int4(d_attn_i4,d_attn_i4_sc,d_attn,Q,st);
        blackwell::kernels::gemv_int4_warp_f16wsc(d_proj,(const uint8_t*)d_attn_i4,d_attn_i4_sc,W[l].o.d,W[l].o.sc16,Q,H,st);
        blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,st);
        cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st);
        blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_post,H,eps,st);
        blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st);
        blackwell::kernels::gemv_int4_warp_f16wsc(d_gate,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].g.d,W[l].g.sc16,H,I,st);
        blackwell::kernels::gemv_int4_warp_f16wsc(d_up,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].u.d,W[l].u.sc16,H,I,st);
        // GeGLU (not SwiGLU)
        blackwell::kernels::apply_geglu(d_gate,d_gate,d_up,I,st);
        blackwell::kernels::quantize_int4(d_mlp_i4,d_mlp_i4_sc,d_gate,I,st);
        blackwell::kernels::gemv_int4_warp_f16wsc(d_proj,(const uint8_t*)d_mlp_i4,d_mlp_i4_sc,W[l].d.d,W[l].d.sc16,I,H,st);
        blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,st);
    }
    blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,d_fn,H,eps,st);
    blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st);
    blackwell::kernels::gemv_int4_warp_f16wsc(d_logits,(const uint8_t*)d_x_i4,d_x_i4_sc,lm_head_w.d,lm_head_w.sc16,H,V,st);
}

// ── Generate: given token IDs, run decode ──
static std::string generate(const std::vector<uint32_t>& ids, int max_tokens,
    float temp, int top_k, float rep_pen)
{
    std::vector<uint32_t> generated;
    float* h_embed = new float[H];

    cudaMemset(d_kc,0,(size_t)NL*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vc,0,(size_t)NL*nkv*MAXSEQ*hd*4);

    // Prefill
    for(int step=0;step<(int)ids.size();++step){
        uint32_t tid=ids[step];
        dequant_embed_row(h_embed,tid,host_embed_d,host_embed_sc,H);
        cudaMemcpyAsync(d_x32,h_embed,H*4,cudaMemcpyHostToDevice,st);
        int seq_pos = step;
        forward_token(step, &seq_pos);
    }

    // Decode
    uint32_t next_id=ids.back();
    bool first = true;
    for(int step=(int)ids.size();step<(int)(ids.size()+max_tokens);++step){
        uint32_t tid=next_id;
        dequant_embed_row(h_embed,tid,host_embed_d,host_embed_sc,H);
        cudaMemcpyAsync(d_x32,h_embed,H*4,cudaMemcpyHostToDevice,st);
        int seq_pos = step;
        forward_token(step, &seq_pos);

        // Repetition penalty
        if (rep_pen > 1.0f && !generated.empty()) {
            int n = (int)generated.size();
            if (n > 64) n = 64;
            std::vector<int> rec(generated.end()-n, generated.end());
            int* d_rec;
            cudaMalloc(&d_rec, n*4);
            cudaMemcpy(d_rec, rec.data(), n*4, cudaMemcpyHostToDevice);
            blackwell::kernels::apply_repetition_penalty(d_logits, d_rec, n, rep_pen, V, st);
            cudaFree(d_rec);
        }

        if (first && temp < 0.01f) {
            blackwell::kernels::sample_argmax_gpu(d_logits,V,d_next_id,st);
        } else if (temp < 0.01f) {
            blackwell::kernels::sample_argmax_gpu(d_logits,V,d_next_id,st);
        }
        cudaStreamSynchronize(st);
        cudaMemcpy(&next_id,d_next_id,4,cudaMemcpyDeviceToHost);
        first = false;

        if(next_id==0||next_id==1||next_id==2) break; // EOS/BOS/UNK
        generated.push_back(next_id);
    }
    delete[] h_embed;

    // Output tokens as JSON array
    std::string result = "\"tokens\":[";
    for(size_t i=0;i<generated.size();++i){
        if(i>0) result+=",";
        result+=std::to_string(generated[i]);
    }
    result+="]";
    // Text field (raw token IDs, decode on client)
    result += ",\"text\":[\"\"],\"note\":\"Use gemma_wrapper.py to decode token IDs\"";
    return result;
}

// ── JSON escape helper ──
static std::string json_escape(const std::string& s) {
    std::string r;
    for (size_t i = 0; i < s.size(); i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"') { r += "\\\""; }
        else if (c == '\\') { r += "\\\\"; }
        else if (c == '\n') { r += "\\n"; }
        else if (c == '\r') { r += "\\r"; }
        else if (c == '\t') { r += "\\t"; }
        else if (c == '<') { r += "\\u003c"; }
        else if (c == '>') { r += "\\u003e"; }
        else if (c < 0x20) { char buf[8]; snprintf(buf,8,"\\u%04x",c); r += buf; }
        else { r += s[i]; }
    }
    return r;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr,"Usage: %s weight_dir\n",argv[0]); return 1; }
    const char* wdir = argv[1];

    cudaSetDevice(0);
    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    fprintf(stderr,"# Gemma 4 12B QAT INT4 Server — %s\n  Weights: %s\n", P.name, wdir);

    allocate_buffers();
    load_weights(wdir);
    load_host_embed(wdir);

    // Main loop: read JSON, generate, write JSON
    while (true) {
        std::string line = read_stdin_line();
        if (line.empty()) { fprintf(stderr,"stdin closed, exiting\n"); break; }

        std::vector<uint32_t> ids = parse_token_ids(line);
        if (ids.empty()) { // fallback: accept string prompts, tokenize externally
            fprintf(stderr,"No token IDs in request — use --tokens via gemma_wrapper.py\n");
            printf("{\"error\":\"No token IDs\",\"note\":\"Use gemma_wrapper.py to tokenize first\"}\n");
            fflush(stdout);
            continue;
        }

        int max_tokens = parse_int(line, "max_tokens", 30);
        if (max_tokens < 1 || max_tokens > 2048) max_tokens = 2048;
        float temp = parse_float(line, "temperature", 0.0f);
        int top_k = parse_int(line, "top_k", 1);
        float rep_pen = parse_repetition_penalty(line, 1.0f);

        std::string result = generate(ids, max_tokens, temp, top_k, rep_pen);

        printf("{%s}\n", result.c_str());
        fflush(stdout);
    }
    return 0;
}
