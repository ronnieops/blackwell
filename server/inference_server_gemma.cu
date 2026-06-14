// server/inference_server_gemma.cu — Gemma 4 12B INT4 inference server (JSON stdio)
//
// Protocol: reads JSON from stdin, writes JSON to stdout.
// Input:  {"prompts":["str1","str2"],"max_tokens":N,"temperature":T,"top_k":K,"repetition_penalty":P}
// Output: {"tokens":[[id1,...],[...]],"text":["text1","text2"]}

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

const int H=3840, Q=3840, KV=512, I=15360;
const int nqh=16, nkv=8, hd=512, MAXSEQ=2048;
const float eps=1e-6f;
const int V=262144;
const int NL=48;

// ── INT4 weight struct + loader ──
struct DevW4 { int K, N; uint8_t* d; float* sc; };

static DevW4 upload_w4(const char* prefix) {
    char p[256]; snprintf(p,256,"%s.int4_t",prefix);
    FILE* f=fopen(p,"rb"); if(!f){DevW4 empty={0,0,nullptr,nullptr};return empty;}
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

struct LW4 { DevW4 q,k,v,o,g,u,d; float *rn_in,*rn_post; };

// ── CUDA kernels (same as benchmark) ──
// RoPE with configurable theta (SWA layers use 1e4, full attention uses 1e6)
__global__ void apply_rope_kernel(float* data, int n_heads, int head_dim, int pos, float theta) {
    int h=blockIdx.x; int d=threadIdx.x;
    if(h>=n_heads||d>=head_dim/2) return;
    float* pair=data+h*head_dim+d*2;
    float t=(float)pos*powf(theta,-2.0f*(float)d/(float)head_dim);
    float c=cosf(t),s=sinf(t),x=pair[0],y=pair[1];
    pair[0]=x*c-y*s; pair[1]=x*s+y*c;
}

// GeGLU: gelu(gate) * up
__global__ void apply_gelu_gate(float* gate_out, const float* gate, const float* up, int I) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= I) return;
    float x = gate[i];
    float gelu = 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
    gate_out[i] = gelu * up[i];
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

// ── Global state (loaded once) ──
static const int MAX_BATCH = 8;
static float *d_x32, *d_xi_f, *d_res;
static uint8_t *d_x_i4; static float *d_x_i4_sc;
static float *d_Q,*d_K,*d_V,*d_attn;
static uint8_t *d_attn_i4; static float *d_attn_i4_sc;
static float *d_proj, *d_gate, *d_up;
static uint8_t *d_mlp_i4; static float *d_mlp_i4_sc;
static float *d_fn, *d_kc, *d_vc, *d_logits, *d_v_pair;
static int *d_next_id, *d_recent;
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
static cudaStream_t st;

static std::vector<LW4> W(NL);
static DevW4 embed_w, lm_head_w;
static uint8_t* host_embed_d;
static float* host_embed_sc;
static blackwell::BpeTokenizer tokenizer;

#define AL(p,n) die(cudaMalloc(&(p),(n)),"malloc " #p)

static void load_model() {
    fprintf(stderr, "Loading %d-layer INT4 model...\n", NL);
    char p[256];
    for (int l = 0; l < NL; ++l) {
        snprintf(p,256,"weights_gemma/%d_self_attn.q_proj",l); W[l].q=upload_w4(p);
        snprintf(p,256,"weights_gemma/%d_self_attn.k_proj",l); W[l].k=upload_w4(p);
        snprintf(p,256,"weights_gemma/%d_self_attn.v_proj",l);
        // Sliding window layers (5, 11, 17, 23, 29, 35, 41, 47) lack v_proj
        bool has_v = true;
        for (int swl : {5, 11, 17, 23, 29, 35, 41, 47}) {
            if (l == swl) { has_v = false; break; }
        }
        if (has_v) {
            W[l].v=upload_w4(p);
        } else {
            W[l].v = {0,0,nullptr,nullptr};
        }
        snprintf(p,256,"weights_gemma/%d_self_attn.o_proj",l); W[l].o=upload_w4(p);
        snprintf(p,256,"weights_gemma/%d_mlp.gate_proj",l); W[l].g=upload_w4(p);
        snprintf(p,256,"weights_gemma/%d_mlp.up_proj",l);   W[l].u=upload_w4(p);
        snprintf(p,256,"weights_gemma/%d_mlp.down_proj",l); W[l].d=upload_w4(p);
        if ((l+1)%7==0) fprintf(stderr, "  layer %d/%d\n", l+1, NL);
    }
    // Per-layer RMSNorm
    for(int l=0;l<NL;++l){
        float* w=(float*)malloc(H*4);
        snprintf(p,256,"weights_gemma/%d_input_layernorm.f32",l);
        {FILE*f=fopen(p,"rb");size_t nr=fread(w,4,H,f);if(nr!=(size_t)H){fprintf(stderr,"Truncated layernorm %d\n",l);exit(1);}fclose(f);}
        cudaMalloc(&W[l].rn_in,H*4);cudaMemcpy(W[l].rn_in,w,H*4,cudaMemcpyHostToDevice);
        snprintf(p,256,"weights_gemma/%d_post_attention_layernorm.f32",l);
        {FILE*f=fopen(p,"rb");size_t nr=fread(w,4,H,f);if(nr!=(size_t)H){fprintf(stderr,"Truncated post layernorm %d\n",l);exit(1);}fclose(f);}
        cudaMalloc(&W[l].rn_post,H*4);cudaMemcpy(W[l].rn_post,w,H*4,cudaMemcpyHostToDevice);
        free(w);
    }
    // Final norm
    {float*w=(float*)malloc(H*4);
     FILE*f=fopen("weights_gemma/final_norm.f32","rb");size_t nr=fread(w,4,H,f);if(nr!=(size_t)H){fprintf(stderr,"Truncated final_norm\n");exit(1);}fclose(f);
     AL(d_fn,H*4); cudaMemcpy(d_fn,w,H*4,cudaMemcpyHostToDevice); free(w);}
    // Embed + lm_head
    embed_w=upload_w4("weights_gemma/embed_tokens");
    lm_head_w=upload_w4("weights_gemma/lm_head");
    fprintf(stderr,"  embed: %dx%d, lm_head: %dx%d\n",embed_w.K,embed_w.N,lm_head_w.K,lm_head_w.N);
    // Host embed for CPU dequant
    host_embed_d=new uint8_t[(size_t)embed_w.K*embed_w.N/2];
    host_embed_sc=new float[embed_w.N*(embed_w.K/16)];
    {FILE*f=fopen("weights_gemma/embed_tokens.int4_t","rb");int h[5];fread(h,4,5,f);
     size_t ds=(size_t)h[0]*h[1]/2;size_t nr=fread(host_embed_d,1,ds,f);if(nr!=ds){fprintf(stderr,"Truncated embed_tokens\n");exit(1);}fclose(f);
     f=fopen("weights_gemma/embed_tokens.scale_t","rb");fread(h,4,5,f);
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
    AL(d_v_pair,(size_t)8*KV*4); die(cudaGetLastError(),"d_v_pair");
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

    for (int l = 0; l < NL; ++l) {
        die(cudaMemcpyAsync(d_res, d_x32, H*4, cudaMemcpyDeviceToDevice, st), "save_res");
        die(blackwell::kernels::fused_rmsnorm(d_xi_f, d_x32, W[l].rn_in, H, eps, st), "rmsnorm_in");
        die(blackwell::kernels::quantize_int4_batched(d_x_i4, d_x_i4_sc, d_xi_f, H, 1, st), "quant_in");
        die(blackwell::kernels::gemv_int4_batched(d_Q, (const uint8_t*)d_x_i4, d_x_i4_sc, W[l].q.d, W[l].q.sc, H, Q, 1, st), "q_proj");
        die(blackwell::kernels::gemv_int4_batched(d_K, (const uint8_t*)d_x_i4, d_x_i4_sc, W[l].k.d, W[l].k.sc, H, KV, 1, st), "k_proj");
        if(W[l].v.d) {
            die(blackwell::kernels::gemv_int4_batched(d_V, (const uint8_t*)d_x_i4, d_x_i4_sc, W[l].v.d, W[l].v.sc, H, KV, 1, st), "v_proj");
            // Save V for SWA layers that share this V. Gemma pattern:
            // SWA at layer l shares V with FA layer l-5 (the last FA before the group of 5)
            // SWA indexes: 5,11,17,23,29,35,41,47 → FA pairs: 0,6,12,18,24,30,36,42
            int pair_idx = -1;
            for (int pi : {0, 6, 12, 18, 24, 30, 36, 42}) {
                if (l == pi) { pair_idx = pi / 6; break; }
            }
            if (pair_idx >= 0) {
                die(cudaMemcpyAsync(d_v_pair + pair_idx * KV, d_V, KV*4, cudaMemcpyDeviceToDevice, st), "save_v_pair");
            }
        } else {
            // SWA layer: reuse V from FA layer l-5
            int pair_idx = l / 6;  // 5/6=0, 11/6=1, 17/6=2, etc.
            die(cudaMemcpyAsync(d_V, d_v_pair + pair_idx * KV, KV*4, cudaMemcpyDeviceToDevice, st), "reuse_v_pair");
        }

        // RoPE: SWA layers (5,11,17,23,29,35,41,47) use theta=1e4, others use 1e6
        float rope_theta = (l % 6 == 5) ? 10000.0f : 1000000.0f;
        apply_rope_kernel<<<nqh,hd/2,0,st>>>(d_Q, nqh, hd, step, rope_theta);
        apply_rope_kernel<<<nkv,hd/2,0,st>>>(d_K, nkv, hd, step, rope_theta);
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
        apply_gelu_gate<<<(I+255)/256,256,0,st>>>(d_gate, d_gate, d_up, I);
        die(blackwell::kernels::quantize_int4_batched(d_mlp_i4, d_mlp_i4_sc, d_gate, I, 1, st), "quant_mlp");
        die(blackwell::kernels::gemv_int4_batched(d_proj, (const uint8_t*)d_mlp_i4, d_mlp_i4_sc, W[l].d.d, W[l].d.sc, I, H, 1, st), "down");
        die(blackwell::kernels::vector_add_fp32(d_x32, d_proj, d_res, H, st), "mlp_res");
    }
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
        die(blackwell::kernels::gemv_int4_batched(d_Q_batch, (const uint8_t*)d_x_i4_batch, d_x_i4_sc_batch, W[l].q.d, W[l].q.sc, H, Q, M, st), "q_proj_b");
        die(blackwell::kernels::gemv_int4_batched(d_K_batch, (const uint8_t*)d_x_i4_batch, d_x_i4_sc_batch, W[l].k.d, W[l].k.sc, H, KV, M, st), "k_proj_b");
        for (int m = 0; m < M; ++m) {
            if(W[l].v.d) {
                die(blackwell::kernels::gemv_int4_batched(d_V_batch + m * KV, (const uint8_t*)d_x_i4_batch, d_x_i4_sc_batch, W[l].v.d, W[l].v.sc, H, KV, 1, st), "v_proj_b");
                // Save V for SWA layers at paired FA layer
                int pair_idx = -1;
                for (int pi : {0, 6, 12, 18, 24, 30, 36, 42}) {
                    if (l == pi) { pair_idx = pi / 6; break; }
                }
                if (pair_idx >= 0 && m == 0) {
                    die(cudaMemcpyAsync(d_v_pair + pair_idx * KV, d_V_batch, KV*4, cudaMemcpyDeviceToDevice, st), "save_v_pair_b");
                }
            } else {
                // SWA layer: reuse V from FA layer l-5
                int pair_idx = l / 6;
                die(cudaMemcpyAsync(d_V_batch + m * KV, d_v_pair + pair_idx * KV, KV*4, cudaMemcpyDeviceToDevice, st), "reuse_v_pair_b");
            }
        }  // end V per-sequence loop

        // Q/K head norms + RoPE (per-sequence — same step for all in speculative verify)
        for (int m = 0; m < M; ++m) {

            float rtheta = (l % 6 == 5) ? 10000.0f : 1000000.0f;
            apply_rope_kernel<<<nqh,hd/2,0,st>>>(d_Q_batch + m * Q, nqh, hd, step, rtheta);
            apply_rope_kernel<<<nkv,hd/2,0,st>>>(d_K_batch + m * KV, nkv, hd, step, rtheta);
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
        die(blackwell::kernels::gemv_int4_batched(d_proj_batch, (const uint8_t*)d_attn_i4_batch, d_attn_i4_sc_batch, W[l].o.d, W[l].o.sc, Q, H, M, st), "o_proj_b");

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
        die(blackwell::kernels::gemv_int4_batched(d_gate_batch, (const uint8_t*)d_x_i4_batch, d_x_i4_sc_batch, W[l].g.d, W[l].g.sc, H, I, M, st), "gate_b");
        die(blackwell::kernels::gemv_int4_batched(d_up_batch, (const uint8_t*)d_x_i4_batch, d_x_i4_sc_batch, W[l].u.d, W[l].u.sc, H, I, M, st), "up_b");

        // GeGLU (per-sequence)
        for (int m = 0; m < M; ++m) {
            apply_gelu_gate<<<(I+255)/256,256,0,st>>>(d_gate_batch + m * I, d_gate_batch + m * I, d_up_batch + m * I, I);
        }

        // Quantize MLP output + down projection (batched)
        die(blackwell::kernels::quantize_int4_batched(d_mlp_i4_batch, d_mlp_i4_sc_batch, d_gate_batch, I, M, st), "quant_mlp_b");
        die(blackwell::kernels::gemv_int4_batched(d_proj_batch, (const uint8_t*)d_mlp_i4_batch, d_mlp_i4_sc_batch, W[l].d.d, W[l].d.sc, I, H, M, st), "down_b");

        // Final residual add (per-sequence)
        for (int m = 0; m < M; ++m) {
            die(blackwell::kernels::vector_add_fp32(d_x32_batch + m * H, d_proj_batch + m * H, d_res_batch + m * H, H, st), "mlp_res_b");
        }
    }
}

using TokVec = std::vector<uint32_t>;

__host__ static TokVec generate(const TokVec& input_ids,
                                       int max_new, float temperature, int top_k, float rep_pen,
                                       bool streaming = false) {
    std::vector<uint32_t> all_ids = input_ids;
    int gen_start = (int)input_ids.size();
    int total = gen_start + max_new;

    cudaMemset(d_kc,0,(size_t)NL*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vc,0,(size_t)NL*nkv*MAXSEQ*hd*4);

    // ── Phase 1: Prefill — process all prompt tokens (no lm_head/sampling) ──
    for (int step = 0; step < gen_start; ++step) {
        decode_one_token(input_ids[step], step);
    }

    // ── Phase 2: Decode — generate new tokens one at a time ──
    for (int step = gen_start; step < total; ++step) {
        uint32_t tid = all_ids.back();
        decode_one_token(tid, step);

        // lm_head + sampling
        die(blackwell::kernels::fused_rmsnorm(d_xi_f, d_x32, d_fn, H, eps, st), "fn");
        die(blackwell::kernels::quantize_int4_batched(d_x_i4, d_x_i4_sc, d_xi_f, H, 1, st), "quant_lm");
        die(blackwell::kernels::gemv_int4_batched(d_logits, (const uint8_t*)d_x_i4, d_x_i4_sc, lm_head_w.d, lm_head_w.sc, H, V, 1, st), "lm_head");

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

        if (next_id == 1 || next_id == 106) break;
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
        float rep_pen = parse_float(line, "\"repetition_penalty\"", 1.5f);
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
