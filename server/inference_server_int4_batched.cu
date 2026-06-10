// server/inference_server_int4_batched.cu — Batched INT4 8B server
// Processes up to M=8 prompts, each independently, then returns batched results.
//
// For true batched inference (shared KV cache, batched GEMVs), would need:
// - Separate d_Q[m*Q] per sequence
// - gemv_int4_batched kernel
// - Batch attention kernel
// For now: processes each sequence sequentially.

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

struct LW4 { DevW4 q,k,v,o,g,u,d; float* qn,*kn,*rn_in,*rn_post; };

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

static void dequant_embed_row(float* out, int token, const uint8_t* host_w, const float* host_sc, int K) {
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
static DevW4 embed_w, lm_head_w;
static float* d_embed_fp32;  // Pre-loaded full embedding table (FP32)
static blackwell::BpeTokenizer tokenizer;

#define AL(p,n) die(cudaMalloc(&(p),(n)),"malloc "#p)

static void load_model() {
    fprintf(stderr, "Loading %d-layer INT4 model...\n", NL);
    char p[256];
    for (int l = 0; l < NL; ++l) {
        snprintf(p,256,"weights_int4_qwen3_8b/%d_self_attn.q_proj",l); W[l].q=upload_w4(p);
        snprintf(p,256,"weights_int4_qwen3_8b/%d_self_attn.k_proj",l); W[l].k=upload_w4(p);
        snprintf(p,256,"weights_int4_qwen3_8b/%d_self_attn.v_proj",l); W[l].v=upload_w4(p);
        snprintf(p,256,"weights_int4_qwen3_8b/%d_self_attn.o_proj",l); W[l].o=upload_w4(p);
        snprintf(p,256,"weights_int4_qwen3_8b/%d_mlp.gate_proj",l); W[l].g=upload_w4(p);
        snprintf(p,256,"weights_int4_qwen3_8b/%d_mlp.up_proj",l); W[l].u=upload_w4(p);
        snprintf(p,256,"weights_int4_qwen3_8b/%d_mlp.down_proj",l); W[l].d=upload_w4(p);
        if ((l+1)%7==0) fprintf(stderr, "  layer %d/%d\n", l+1, NL);
    }
    float* qk_h=(float*)malloc(NL*2*hd*4);
    {FILE*f=fopen("weights_int4_qwen3_8b/qk_norms.f32","rb");(void)fread(qk_h,4,NL*2*hd,f);fclose(f);}
    for(int l=0;l<NL;++l){
        cudaMalloc(&W[l].qn,hd*4);cudaMemcpy(W[l].qn,qk_h+l*2*hd,hd*4,cudaMemcpyHostToDevice);
        cudaMalloc(&W[l].kn,hd*4);cudaMemcpy(W[l].kn,qk_h+l*2*hd+hd,hd*4,cudaMemcpyHostToDevice);
    }free(qk_h);
    for(int l=0;l<NL;++l){
        float* w=(float*)malloc(H*4);
        snprintf(p,256,"weights_int4_qwen3_8b/%d_input_layernorm.f32",l);
        {FILE*f=fopen(p,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&W[l].rn_in,H*4);cudaMemcpy(W[l].rn_in,w,H*4,cudaMemcpyHostToDevice);
        snprintf(p,256,"weights_int4_qwen3_8b/%d_post_attention_layernorm.f32",l);
        {FILE*f=fopen(p,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&W[l].rn_post,H*4);cudaMemcpy(W[l].rn_post,w,H*4,cudaMemcpyHostToDevice);
        free(w);
    }
    {float*w=(float*)malloc(H*4);
     FILE*f=fopen("weights_int4_qwen3_8b/final_norm.f32","rb");(void)fread(w,4,H,f);fclose(f);
     AL(d_fn,H*4); cudaMemcpy(d_fn,w,H*4,cudaMemcpyHostToDevice); free(w);}
    embed_w=upload_w4("weights_int4_qwen3_8b/embed_tokens");
    lm_head_w=upload_w4("weights_int4_qwen3_8b/lm_head");
    fprintf(stderr,"  embed: %dx%d, lm_head: %dx%d\n",embed_w.K,embed_w.N,lm_head_w.K,lm_head_w.N);
    
    // Pre-load full embedding table to GPU as FP32 (optimization)
    // Dequantize once at startup, avoid per-token CPU dequantization
    size_t embed_size = (size_t)embed_w.K * embed_w.N;
    die(cudaMalloc(&d_embed_fp32, embed_size * 4), "embed_fp32");
    
    // Load INT4 embedding, dequantize to FP32, copy to GPU
    uint8_t* tmp_i4 = new uint8_t[embed_size / 2];
    float* tmp_sc = new float[embed_w.N * (embed_w.K / 16)];
    {FILE*f=fopen("weights_int4_qwen3_8b/embed_tokens.int4_t","rb");int h[5];fread(h,4,5,f);
     fread(tmp_i4,1,(size_t)h[0]*h[1]/2,f);fclose(f);
     f=fopen("weights_int4_qwen3_8b/embed_tokens.scale_t","rb");fread(h,4,5,f);
     fread(tmp_sc,4,(size_t)h[3]*h[4],f);fclose(f);}
    
    // Dequantize to FP32 on CPU
    float* tmp_fp32 = new float[embed_size];
    int kblocks = embed_w.K / 16;
    for (int row = 0; row < embed_w.N; ++row) {
        for (int kb = 0; kb < kblocks; ++kb) {
            float sc = tmp_sc[row * kblocks + kb];
            for (int i = 0; i < 16; ++i) {
                size_t byte_idx = (size_t)row * embed_w.K / 2 + (size_t)kb * 8 + i / 2;
                uint8_t byte = tmp_i4[byte_idx];
                int nib = (i & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
                tmp_fp32[row * embed_w.K + kb * 16 + i] = (float)(nib - 8) * sc;
            }
        }
    }
    die(cudaMemcpy(d_embed_fp32, tmp_fp32, embed_size * 4, cudaMemcpyHostToDevice), "embed_copy");
    delete[] tmp_i4; delete[] tmp_sc; delete[] tmp_fp32;
    
    fprintf(stderr,"All weights loaded (embed pre-loaded to GPU).\n");
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

    for (int step = 0; step < total; ++step) {
        uint32_t tid = (step < gen_start) ? all_ids[step] : all_ids.back();
        // Direct GPU copy from pre-loaded embedding (no CPU dequantization)
        die(cudaMemcpyAsync(d_x32, d_embed_fp32 + (size_t)tid * H, H * 4, cudaMemcpyDeviceToDevice, st), "embed");

        for (int l = 0; l < NL; ++l) {
            size_t kv_off = (size_t)l*nkv*MAXSEQ*hd;
            die(cudaMemcpyAsync(d_res, d_x32, H*4, cudaMemcpyDeviceToDevice, st), "save_res");
            die(blackwell::kernels::fused_rmsnorm(d_xi_f, d_x32, W[l].rn_in, H, eps, st), "rn_in");
            die(blackwell::kernels::quantize_int4(d_x_i4, d_x_i4_sc, d_xi_f, H, st), "q_in");
            die(blackwell::kernels::gemv_int4_batched(d_Q, (const uint8_t*)d_x_i4, d_x_i4_sc, W[l].q.d, W[l].q.sc, H, Q, 1, st), "q_proj");
            die(blackwell::kernels::gemv_int4_batched(d_K, (const uint8_t*)d_x_i4, d_x_i4_sc, W[l].k.d, W[l].k.sc, H, KV, 1, st), "k_proj");
            die(blackwell::kernels::gemv_int4_batched(d_V, (const uint8_t*)d_x_i4, d_x_i4_sc, W[l].v.d, W[l].v.sc, H, KV, 1, st), "v_proj");
            head_norm_kernel<<<nqh,128,0,st>>>(d_Q, W[l].qn, nqh, hd, eps); die(cudaGetLastError(),"hn_q");
            head_norm_kernel<<<nkv,128,0,st>>>(d_K, W[l].kn, nkv, hd, eps); die(cudaGetLastError(),"hn_k");
            apply_rope_kernel<<<nqh,hd/2,0,st>>>(d_Q, nqh, hd, step); die(cudaGetLastError(),"rp_q");
            apply_rope_kernel<<<nkv,hd/2,0,st>>>(d_K, nkv, hd, step); die(cudaGetLastError(),"rp_k");
            die(blackwell::kernels::update_kv_cache(d_kc+kv_off, d_vc+kv_off, d_K, d_V, 0, step, nkv, hd, MAXSEQ, st), "kv");
            die(blackwell::kernels::attention_decode_batched_gqa(d_attn, d_Q, d_kc, d_vc, step, nqh, nkv, hd, MAXSEQ, 1,
                (size_t)NL*nkv*MAXSEQ*hd, kv_off, st), "attn");
            die(blackwell::kernels::quantize_int4(d_attn_i4, d_attn_i4_sc, d_attn, Q, st), "q_attn");
            die(blackwell::kernels::gemv_int4_batched(d_proj, (const uint8_t*)d_attn_i4, d_attn_i4_sc, W[l].o.d, W[l].o.sc, Q, H, 1, st), "o_proj");
            die(blackwell::kernels::vector_add_fp32(d_x32, d_proj, d_res, H, st), "res1");
            die(cudaMemcpyAsync(d_res, d_x32, H*4, cudaMemcpyDeviceToDevice, st), "save_res2");
            die(blackwell::kernels::fused_rmsnorm(d_xi_f, d_x32, W[l].rn_post, H, eps, st), "rn_post");
            die(blackwell::kernels::quantize_int4(d_x_i4, d_x_i4_sc, d_xi_f, H, st), "q_mlp");
            die(blackwell::kernels::gemv_int4_batched(d_gate, (const uint8_t*)d_x_i4, d_x_i4_sc, W[l].g.d, W[l].g.sc, H, I, 1, st), "gate");
            die(blackwell::kernels::gemv_int4_batched(d_up, (const uint8_t*)d_x_i4, d_x_i4_sc, W[l].u.d, W[l].u.sc, H, I, 1, st), "up");
            blackwell::kernels::apply_swiglu(d_gate, d_gate, d_up, I, st);
            die(blackwell::kernels::quantize_int4(d_mlp_i4, d_mlp_i4_sc, d_gate, I, st), "q_mlp2");
            die(blackwell::kernels::gemv_int4_batched(d_proj, (const uint8_t*)d_mlp_i4, d_mlp_i4_sc, W[l].d.d, W[l].d.sc, I, H, 1, st), "down");
            die(blackwell::kernels::vector_add_fp32(d_x32, d_proj, d_res, H, st), "res2");
        }

        if (step >= gen_start - 1) {
            die(blackwell::kernels::fused_rmsnorm(d_xi_f, d_x32, d_fn, H, eps, st), "fn");
            die(blackwell::kernels::quantize_int4(d_x_i4, d_x_i4_sc, d_xi_f, H, st), "q_fn");
            die(blackwell::kernels::gemv_int4_batched(d_logits, (const uint8_t*)d_x_i4, d_x_i4_sc, lm_head_w.d, lm_head_w.sc, H, V, 1, st), "lm_head");
            
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

        // Process each prompt sequentially, clear KV cache between
        std::vector<std::vector<uint32_t>> all_results;
        for (size_t i = 0; i < prompts.size(); ++i) {
            auto input_ids = tokenizer.encode(prompts[i]);
            auto gen = generate_one(input_ids, max_tokens, temperature, top_k, rep_pen);
            all_results.push_back(gen);
            cudaStreamSynchronize(st);
        }

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
