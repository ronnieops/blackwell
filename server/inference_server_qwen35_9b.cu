// server/inference_server_qwen35_9b.cu — Blackwell INT8 GatedDeltaNet inference server
//
// 32 layers: 24 linear_attention (GatedDeltaNet SSM) + 8 full_attention (GQA)
// Pattern: lin,lin,lin,full (x8)
// Protocol: JSON lines on stdin/stdout
// Model: Qwen3.5-9B GatedDeltaNet, V=248320
//
// Build:
//   CUDACXX=/usr/local/cuda-13.3/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include server/inference_server_qwen35_9b.cu build/libblackwell_kernels.a \
//     -o server/inference_server_9b -lcudart -lpthread -lz

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include "blackwell/kernels.h"
#include "blackwell/bpe_tokenizer.h"

static void die(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) { fprintf(stderr, "FAIL %s: %s\n", msg, cudaGetErrorString(e)); exit(1); }
}

// ── Constants ────────────────────────────────────────────────────────
static const int NL=32, H=4096, I=12288, V=248320;
static const int NK=16, NV=32, HD=128;         // GatedDeltaNet dims
static const int CD = NK*HD*2 + NV*HD;          // 12288
static const int CK = 4;                         // conv kernel size
static const int NQ=16, NKV=4, HDA=256;         // full attention dims
static const int PD = HDA/4;                     // 64 RoPE pair-dim
static const int MS = 2048;                      // max seq len for KV cache

static bool is_lin(int l) { return (l % 4) != 3; }
static int full_idx(int l) { return l / 4; }

// ── Kernels ──────────────────────────────────────────────────────────

__global__ void head_norm_k(float* d, const float* w, int nh, int hd, float eps) {
    int h = blockIdx.x; if (h >= nh) return;
    float* p = d + h*hd; __shared__ float ws[4];
    float s = 0; int tid = threadIdx.x;
    for (int i = tid; i < hd; i += blockDim.x) s += p[i]*p[i];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff,s,o);
    if ((tid&31)==0) ws[tid>>5]=s; __syncthreads();
    if (tid<32){float v=(tid<4)?ws[tid]:0;
        for(int o=2;o>0;o>>=1)v+=__shfl_xor_sync(0xffffffff,v,o);
        if(tid==0)ws[0]=rsqrtf(v/hd+eps);}
    __syncthreads(); float is=ws[0];
    for (int i=tid;i<hd;i+=blockDim.x) p[i]=p[i]*is*w[i];
}

__global__ void rope_k(float* d, int nh, int hd, int pos, int pd) {
    int h = blockIdx.x, t = threadIdx.x; if (h >= nh || t >= pd/2) return;
    int i2=t*2; float* pair = d + h*hd + i2;
    float th = (float)pos * powf(10000000.f, -2.f*(float)t/(float)pd);
    float c = cosf(th), s = sinf(th), x = pair[0], y = pair[1];
    pair[0] = x*c - y*s; pair[1] = x*s + y*c;
}

// Apply sigmoid gate to attention output (Qwen3.5 attn_output_gate)
__global__ void attn_gate_k(float* out, const float* gate, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = out[i] * (1.0f / (1.0f + expf(-gate[i])));
}

__global__ void compute_g_k(float* g, const float* a, const float* al, const float* dt, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x; if (i >= n) return;
    float sp = logf(1.f+expf(a[i]+dt[i]));
    g[i] = -expf(al[i])*sp;
}

// ── Weight loading ───────────────────────────────────────────────────

struct DevW { int K, N; int8_t* d; float* sc; };

static DevW load_int8_w(const char* prefix) {
    char p[512]; snprintf(p,512,"%s.int8_t",prefix);
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"FAIL open %s\n",p);exit(1);}
    int h[5]; (void)fread(h,4,5,f);
    std::vector<int8_t> tmp((size_t)h[0]*h[1]); (void)fread(tmp.data(),1,tmp.size(),f);fclose(f);
    DevW dw{h[0],h[1],nullptr,nullptr};
    cudaMalloc(&dw.d,(size_t)h[0]*h[1]); cudaMemcpy(dw.d,tmp.data(),dw.K*dw.N,cudaMemcpyHostToDevice);
    snprintf(p,512,"%s.scale_t",prefix); f=fopen(p,"rb");if(!f){fprintf(stderr,"FAIL open %s\n",p);exit(1);}
    (void)fread(h,4,5,f); size_t ns=(size_t)h[3]*h[4]; std::vector<float> ts(ns); (void)fread(ts.data(),4,ns,f);fclose(f);
    cudaMalloc(&dw.sc,ns*4); cudaMemcpy(dw.sc,ts.data(),ns*4,cudaMemcpyHostToDevice);
    return dw;
}

static float* load_f32(const char* prefix, int n) {
    char p[512]; snprintf(p,512,"%s.f32",prefix);
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"FAIL open %s\n",p);exit(1);}
    std::vector<float> tmp(n); (void)fread(tmp.data(),4,n,f);fclose(f);
    float* d; cudaMalloc(&d,n*4); cudaMemcpy(d,tmp.data(),n*4,cudaMemcpyHostToDevice);
    return d;
}

static float* load_bf16(const char* prefix, int n) {
    char p[512]; snprintf(p,512,"%s.f16",prefix);
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"FAIL open %s\n",p);exit(1);}
    uint16_t* h16=(uint16_t*)malloc(n*2); (void)fread(h16,2,n,f);fclose(f);
    float* h32=(float*)malloc(n*4);
    for(int i=0;i<n;i++){uint32_t u=(uint32_t)h16[i]<<16; memcpy(&h32[i],&u,4);}
    float* d; cudaMalloc(&d,n*4); cudaMemcpy(d,h32,n*4,cudaMemcpyHostToDevice); free(h16);free(h32);
    return d;
}

// ── Server struct ────────────────────────────────────────────────────

struct ServerState {
    int M; float eps;

    // Layer weights
    struct GdnLinW {
        DevW qkv, a, b, z, out;
        float *conv_w, *A_log, *dt_bias, *norm_w;
    };
    struct GdnFullW {
        DevW q, k, v, o;
        float *qn, *kn;
    };
    struct DevMlp { DevW gate, up, down; };

    std::vector<GdnLinW> linW;
    std::vector<GdnFullW> fullW;
    std::vector<DevMlp> mlp;
    std::vector<float*> d_rn_in;
    std::vector<float*> d_rn_post;
    float* d_fn;
    DevW emb, lm_head;

    int8_t* h_emb_int8;
    float* h_emb_scale;

    // GDN state
    float *d_cs, *d_rs;  // [NL][CD*(CK-1)] + [NL][NV*HD*HD]
    float *d_qkvc, *d_g, *d_beta, *d_ao, *d_z;
    float *d_q_cnt, *d_k_cnt, *d_v_cnt;
    float *d_q_bc, *d_k_bc;

    // Full attention
    float *dQ, *dK, *dV, *d_kc, *d_vc; // KV: [8 full][NKV][MS][HDA]

    // Shared buffers
    float *d_x, *d_xn, *d_proj;
    int8_t* d_ai; float* d_as;
    float* d_mlp_res; int8_t* d_mlp_ai; float* d_mlp_as;
    float* d_logits; int* d_next_id;
    float* d_residual;

    cudaStream_t st;
};

// ── Tokenizer helpers ────────────────────────────────────────────────

static std::vector<uint32_t> encode_special(const std::string& s, blackwell::BpeTokenizer& tok) {
    std::vector<uint32_t> r; size_t pos=0;
    while(pos<s.size()){
        if(s.compare(pos,12,"<|im_start|>")==0){r.push_back(151644);pos+=12;}
        else if(s.compare(pos,10,"<|im_end|>")==0){r.push_back(151645);pos+=10;}
        else{
            size_t np=s.find("<|im_",pos);
            if(np==std::string::npos)np=s.size();
            auto ids=tok.encode(s.substr(pos,np-pos));
            r.insert(r.end(),ids.begin(),ids.end()); pos=np;
        }
    }
    return r;
}

static std::string read_line() {
    std::string l; int c;
    while((c=getchar())!=EOF&&c!='\n')l.push_back((char)c);
    return l;
}

static int find_int(const std::string& j, const char* k, int d=0){
    const char* p=strstr(j.c_str(),k);if(!p)return d;
    p=strchr(p,':');if(!p)return d;p++;
    while(*p==' '||*p=='\t')p++;
    return (int)strtol(p,nullptr,10);
}

static float find_float(const std::string& j, const char* k, float d=0){
    const char* p=strstr(j.c_str(),k);if(!p)return d;
    p=strchr(p,':');if(!p)return d;p++;
    while(*p==' '||*p=='\t')p++;
    return (float)atof(p);
}

static std::vector<uint32_t> parse_prompt(const std::string& j, blackwell::BpeTokenizer& tok){
    // Prefer "prompts":["..."] array (http_subprocess format), fallback to "prompt":"..."
    const char* pp=strstr(j.c_str(),"\"prompts\"");
    if(pp&&pp[9]==':'){
        const char* p=strchr(pp,':');if(!p)return{};p++;
        while(*p==' '||*p=='\t')p++;
        while(*p&&*p!='[')p++;
        if(*p!='[')return{};p++;
        std::vector<uint32_t> res;
        while(*p&&*p!=']'){
            while(*p&&*p!='\"')p++;
            if(*p!='\"')break;p++;
            std::string s;
            while(*p&&*p!='\"'){
                if(*p=='\\'&&*(p+1)=='n'){s+='\n';p+=2;}
                else if(*p=='\\'&&*(p+1)=='\"'){s+='\"';p+=2;}
                else if(*p=='\\'&&*(p+1)=='\\'){s+='\\';p+=2;}
                else{s+=*p;p++;}
            }
            if(*p=='\"')p++;
            auto ids=encode_special(s,tok);
            res.insert(res.end(),ids.begin(),ids.end());
            while(*p&&*p!='\"'&&*p!=']')p++;
        }
        if(!res.empty())return res;
    }
    // Fallback: "prompt":"..."
    pp=strstr(j.c_str(),"\"prompt\"");
    if(pp&&pp[8]==':'){
        const char* p=strchr(pp,':');if(!p)return{};p++;
        while(*p==' '||*p=='\t')p++;
        if(*p!='\"')return{};p++;
        std::string s;
        while(*p&&*p!='\"'){
            if(*p=='\\'&&*(p+1)=='n'){s+='\n';p+=2;}
            else if(*p=='\\'&&*(p+1)=='\"'){s+='\"';p+=2;}
            else if(*p=='\\'&&*(p+1)=='\\'){s+='\\';p+=2;}
            else{s+=*p;p++;}
        }
        return encode_special(s,tok);
    }
    return{};
}

static void write_results(const std::vector<uint32_t>& gen, blackwell::BpeTokenizer& tok){
    printf("{\"tokens\":[");
    for(size_t t=0;t<gen.size();t++){if(t>0)printf(",");printf("%u",gen[t]);}
    printf("],\"text\":\"");
    std::string txt;
    for(size_t t=0;t<gen.size();t++)txt+=tok.decode(gen[t]);
    for(size_t i=0;i<txt.size();i++){
        char c=txt[i];
        if(c=='\"')printf("\\\"");
        else if(c=='\\')printf("\\\\");
        else if(c=='\n')printf("\\n");
        else printf("%c",c);
    }
    printf("\"}\n");fflush(stdout);
}

// ── Embed ────────────────────────────────────────────────────────────

static void embed(ServerState& S, uint32_t tid) {
    std::vector<float> hh(H);
    for(int d=0;d<H;d++)
        hh[d]=(float)S.h_emb_int8[tid*H+d]*S.h_emb_scale[tid*(H/16)+d/16];
    cudaMemcpy(S.d_residual,hh.data(),H*4,cudaMemcpyHostToDevice);
}

// ── Decode step ──────────────────────────────────────────────────────

static void decode_step(ServerState& S, int seq_pos) {
    for(int l=0;l<NL;l++){
        bool il=is_lin(l);

        // Input RMSNorm
        die(blackwell::kernels::fused_rmsnorm(S.d_xn,S.d_residual,S.d_rn_in[l],H,S.eps,S.st),"rn_in");
        die(blackwell::kernels::quantize_int8(S.d_ai,S.d_as,S.d_xn,H,S.st),"q_in");

        if(il){
            // ── Linear attention (GatedDeltaNet) ──
            die(blackwell::kernels::gemv_int8_warp(S.d_xn,S.d_ai,S.d_as,
                S.linW[l].qkv.d,S.linW[l].qkv.sc,H,CD,S.st),"lqkv");
            die(blackwell::kernels::gated_delta_conv1d_update(
                S.d_cs+l*CD*(CK-1),S.d_xn,S.linW[l].conv_w,S.d_qkvc,S.st),"lcv");
            // Gate projections
            die(blackwell::kernels::gemv_int8_warp(S.d_g,S.d_ai,S.d_as,
                S.linW[l].a.d,S.linW[l].a.sc,H,NV,S.st),"la");
            die(blackwell::kernels::gemv_int8_warp(S.d_beta,S.d_ai,S.d_as,
                S.linW[l].b.d,S.linW[l].b.sc,H,NV,S.st),"lb");
            // g = -exp(A_log)*softplus(a+dt_bias)
            compute_g_k<<<(NV+255)/256,256,0,S.st>>>(S.d_g,S.d_g,S.linW[l].A_log,S.linW[l].dt_bias,NV);
            die(blackwell::kernels::gemv_int8_warp(S.d_z,S.d_ai,S.d_as,
                S.linW[l].z.d,S.linW[l].z.sc,H,H,S.st),"lz");
            // Recurrent step: copy Q/K/V to contiguous buffers, then step
            cudaMemcpyAsync(S.d_q_cnt,S.d_qkvc,NK*HD*4,cudaMemcpyDeviceToDevice,S.st);
            cudaMemcpyAsync(S.d_k_cnt,S.d_qkvc+NK*HD,NK*HD*4,cudaMemcpyDeviceToDevice,S.st);
            cudaMemcpyAsync(S.d_v_cnt,S.d_qkvc+NK*HD*2,NV*HD*4,cudaMemcpyDeviceToDevice,S.st);
            die(blackwell::kernels::gated_delta_recurrent_step(
                S.d_q_cnt,S.d_k_cnt,S.d_v_cnt,S.d_g,S.d_beta,
                S.d_q_bc,S.d_k_bc,S.d_rs+l*NV*HD*HD,S.d_ao,1,S.st),"lrs");
            // RMSNorm gated
            die(blackwell::kernels::gated_delta_rmsnorm_gated(
                S.d_proj,S.d_ao,S.d_z,S.linW[l].norm_w,1,S.eps,S.st),"lrng");
            // Quantize + out_proj
            die(blackwell::kernels::quantize_int8(S.d_ai,S.d_as,S.d_proj,H,S.st),"q_lo");
            die(blackwell::kernels::gemv_int8_warp(S.d_proj,S.d_ai,S.d_as,
                S.linW[l].out.d,S.linW[l].out.sc,H,H,S.st),"lout");
        } else {
            // ── Full attention (GQA) ──
            int fi=full_idx(l);
            die(blackwell::kernels::gemv_int8_warp(S.dQ,S.d_ai,S.d_as,
                S.fullW[l].q.d,S.fullW[l].q.sc,H,2*NQ*HDA,S.st),"fq");
            die(blackwell::kernels::gemv_int8_warp(S.dK,S.d_ai,S.d_as,
                S.fullW[l].k.d,S.fullW[l].k.sc,H,NKV*HDA,S.st),"fk");
            die(blackwell::kernels::gemv_int8_warp(S.dV,S.d_ai,S.d_as,
                S.fullW[l].v.d,S.fullW[l].v.sc,H,NKV*HDA,S.st),"fv");
            head_norm_k<<<NQ,128,0,S.st>>>(S.dQ,S.fullW[l].qn,NQ,HDA,S.eps);
            head_norm_k<<<NKV,128,0,S.st>>>(S.dK,S.fullW[l].kn,NKV,HDA,S.eps);
            rope_k<<<NQ,PD/2,0,S.st>>>(S.dQ,NQ,HDA,seq_pos,PD);
            rope_k<<<NKV,PD/2,0,S.st>>>(S.dK,NKV,HDA,seq_pos,PD);
            size_t ko=(size_t)fi*NKV*MS*HDA;
            die(blackwell::kernels::update_kv_cache(
                S.d_kc+ko,S.d_vc+ko,S.dK,S.dV,0,seq_pos,NKV,HDA,MS,S.st),"fkvc");
            die(blackwell::kernels::attention_decode_gqa(
                S.d_proj,S.dQ,S.d_kc+ko,S.d_vc+ko,seq_pos,NQ,NKV,HDA,MS,S.st),"fatt");
            // Apply attn_output_gate: attn_out *= sigmoid(gate)
            {
                int gn = NQ * HDA;
                attn_gate_k<<<(gn + 255) / 256, 256, 0, S.st>>>(S.d_proj, S.dQ + NQ * HDA, gn);
                die(cudaGetLastError(), "gate");
            }
            // Quantize + out_proj
            die(blackwell::kernels::quantize_int8(S.d_ai,S.d_as,S.d_proj,H,S.st),"q_fo");
            die(blackwell::kernels::gemv_int8_warp(S.d_proj,S.d_ai,S.d_as,
                S.fullW[l].o.d,S.fullW[l].o.sc,H,H,S.st),"fout");
        }

        // ── Residual 1 ──
        die(blackwell::kernels::vector_add_fp32(S.d_xn,S.d_proj,S.d_residual,H,S.st),"res1");
        cudaMemcpyAsync(S.d_residual,S.d_xn,H*4,cudaMemcpyDeviceToDevice,S.st);

        // ── Post-attention RMSNorm ──
        die(blackwell::kernels::fused_rmsnorm(S.d_xn,S.d_xn,S.d_rn_post[l],H,S.eps,S.st),"rn_post");
        die(blackwell::kernels::quantize_int8(S.d_ai,S.d_as,S.d_xn,H,S.st),"q_post");

        // ── MLP ──
        die(blackwell::kernels::gemv_int8_warp(S.d_mlp_res,S.d_ai,S.d_as,
            S.mlp[l].gate.d,S.mlp[l].gate.sc,H,I,S.st),"mg");
        die(blackwell::kernels::gemv_int8_warp(S.d_mlp_res+I,S.d_ai,S.d_as,
            S.mlp[l].up.d,S.mlp[l].up.sc,H,I,S.st),"mu");
        die(blackwell::kernels::apply_swiglu(S.d_mlp_res,S.d_mlp_res,S.d_mlp_res+I,I,S.st),"ms");
        die(blackwell::kernels::quantize_int8(S.d_mlp_ai,S.d_mlp_as,S.d_mlp_res,I,S.st),"q_mlp");
        die(blackwell::kernels::gemv_int8_warp(S.d_xn,S.d_mlp_ai,S.d_mlp_as,
            S.mlp[l].down.d,S.mlp[l].down.sc,I,H,S.st),"md");

        // ── Residual 2 ──
        die(blackwell::kernels::vector_add_fp32(S.d_xn,S.d_xn,S.d_residual,H,S.st),"res2");
        cudaMemcpyAsync(S.d_residual,S.d_xn,H*4,cudaMemcpyDeviceToDevice,S.st);

    }
}

// ── Main ──────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    fprintf(stderr,"Blackwell INT8 GatedDeltaNet Server v0.7.0 (Qwen3.5-9B)\n");
    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    fprintf(stderr,"Device: %s (CC %d.%d)\n",p.name,p.major,p.minor);

    ServerState S;
    S.M=1; S.eps=1e-6f;
    die(cudaStreamCreate(&S.st),"stream");

    blackwell::BpeTokenizer tok;
    if(tok.load("tokenizer_data_9b.bin")!=0){
        fprintf(stderr,"FAIL: tokenizer_data_9b.bin\n"); return 1;
    }

    const char* WDIR="weights_int8_qwen35_9b";
    fprintf(stderr,"Loading weights...\n");
    S.linW.resize(NL); S.fullW.resize(NL); S.mlp.resize(NL);
    S.d_rn_in.resize(NL); S.d_rn_post.resize(NL);

    for(int l=0;l<NL;l++){
        bool il=is_lin(l); char p[256];
        snprintf(p,256,"%s/%d_input_layernorm",WDIR,l);
        S.d_rn_in[l]=load_f32(p,H);
        snprintf(p,256,"%s/%d_post_attention_layernorm",WDIR,l);
        S.d_rn_post[l]=load_f32(p,H);

        if(il){
            snprintf(p,256,"%s/%d_linear_attn.in_proj_qkv",WDIR,l); S.linW[l].qkv=load_int8_w(p);
            snprintf(p,256,"%s/%d_linear_attn.in_proj_a",WDIR,l);   S.linW[l].a=load_int8_w(p);
            snprintf(p,256,"%s/%d_linear_attn.in_proj_b",WDIR,l);   S.linW[l].b=load_int8_w(p);
            snprintf(p,256,"%s/%d_linear_attn.in_proj_z",WDIR,l);   S.linW[l].z=load_int8_w(p);
            snprintf(p,256,"%s/%d_linear_attn.out_proj",WDIR,l);    S.linW[l].out=load_int8_w(p);
            snprintf(p,256,"%s/%d_linear_attn.A_log",WDIR,l);      S.linW[l].A_log=load_f32(p,NV);
            snprintf(p,256,"%s/%d_linear_attn.dt_bias",WDIR,l);    S.linW[l].dt_bias=load_f32(p,NV);
            snprintf(p,256,"%s/%d_linear_attn.norm",WDIR,l);       S.linW[l].norm_w=load_f32(p,HD);
            snprintf(p,256,"%s/%d_linear_attn.conv1d.weight",WDIR,l);
            S.linW[l].conv_w=load_bf16(p,CD*CK);
            S.fullW[l].q.d=nullptr; S.fullW[l].qn=nullptr;
        } else {
            snprintf(p,256,"%s/%d_self_attn.q_proj",WDIR,l); S.fullW[l].q=load_int8_w(p);
            snprintf(p,256,"%s/%d_self_attn.k_proj",WDIR,l); S.fullW[l].k=load_int8_w(p);
            snprintf(p,256,"%s/%d_self_attn.v_proj",WDIR,l); S.fullW[l].v=load_int8_w(p);
            snprintf(p,256,"%s/%d_self_attn.o_proj",WDIR,l); S.fullW[l].o=load_int8_w(p);
            snprintf(p,256,"%s/%d_self_attn.q_norm",WDIR,l); S.fullW[l].qn=load_f32(p,HDA);
            snprintf(p,256,"%s/%d_self_attn.k_norm",WDIR,l); S.fullW[l].kn=load_f32(p,HDA);
            S.linW[l].qkv.d=nullptr; S.linW[l].out.d=nullptr;
        }
        snprintf(p,256,"%s/%d_mlp.gate_proj",WDIR,l); S.mlp[l].gate=load_int8_w(p);
        snprintf(p,256,"%s/%d_mlp.up_proj",WDIR,l);   S.mlp[l].up=load_int8_w(p);
        snprintf(p,256,"%s/%d_mlp.down_proj",WDIR,l); S.mlp[l].down=load_int8_w(p);
        if(l%8==0) fprintf(stderr,"  layer %d/%d\n",l,NL);
    }

    S.d_fn=load_f32((std::string(WDIR)+"/final_norm").c_str(),H);
    S.emb=load_int8_w((std::string(WDIR)+"/embed_tokens").c_str());

    { char p[256]; snprintf(p,256,"%s/lm_head.int8_t",WDIR);
      FILE* f=fopen(p,"rb"); if(f){fclose(f);S.lm_head=load_int8_w((std::string(WDIR)+"/lm_head").c_str());
        fprintf(stderr,"  lm_head: separate (INT8)\n");} else S.lm_head.d=nullptr; }

    { char p[256]; snprintf(p,256,"%s/embed_tokens.int8_t",WDIR);
      FILE* f=fopen(p,"rb"); int h[5]; (void)fread(h,4,5,f);
      size_t num=(size_t)h[0]*h[1]; S.h_emb_int8=(int8_t*)malloc(num); (void)fread(S.h_emb_int8,1,num,f);fclose(f);
      snprintf(p,256,"%s/embed_tokens.scale_t",WDIR);
      f=fopen(p,"rb"); (void)fread(h,4,5,f); size_t ns=(size_t)h[3]*h[4];
      S.h_emb_scale=(float*)malloc(ns*4); (void)fread(S.h_emb_scale,4,ns,f);fclose(f); }
    fprintf(stderr,"  done\n");

    // ── Allocate buffers ──
    cudaMalloc(&S.d_residual,H*4);
    // d_xn sized CD for qkv GEMV (CD=12288 > H=4096)
    cudaMalloc(&S.d_x,H*4); cudaMalloc(&S.d_xn,CD*4); cudaMalloc(&S.d_proj,H*4);
    cudaMalloc(&S.d_ai,H); cudaMalloc(&S.d_as,(H/16)*4);

    // GDN
    cudaMalloc(&S.d_qkvc,CD*4);
    cudaMalloc(&S.d_g,NV*4); cudaMalloc(&S.d_beta,NV*4);
    cudaMalloc(&S.d_ao,NV*HD*4); cudaMalloc(&S.d_z,H*4);
    cudaMalloc(&S.d_q_cnt,NK*HD*4); cudaMalloc(&S.d_k_cnt,NK*HD*4); cudaMalloc(&S.d_v_cnt,NV*HD*4);
    cudaMalloc(&S.d_q_bc,NV*HD*4); cudaMalloc(&S.d_k_bc,NV*HD*4);
    cudaMalloc(&S.d_cs,NL*CD*(CK-1)*4); cudaMemset(S.d_cs,0,NL*CD*(CK-1)*4);
    cudaMalloc(&S.d_rs,NL*NV*HD*HD*4); cudaMemset(S.d_rs,0,NL*NV*HD*HD*4);

    // Full attention
    size_t kvs=8*NKV*MS*HDA*4;
    cudaMalloc(&S.dQ,2*NQ*HDA*4); cudaMalloc(&S.dK,NKV*HDA*4); cudaMalloc(&S.dV,NKV*HDA*4);
    cudaMalloc(&S.d_kc,kvs); cudaMalloc(&S.d_vc,kvs);
    cudaMemset(S.d_kc,0,kvs); cudaMemset(S.d_vc,0,kvs);

    // MLP
    cudaMalloc(&S.d_mlp_res,I*2*4); cudaMalloc(&S.d_mlp_ai,I); cudaMalloc(&S.d_mlp_as,(I/16)*4);
    // Logits
    cudaMalloc(&S.d_logits,V*4); cudaMalloc(&S.d_next_id,sizeof(int));

    fprintf(stderr,"Ready.\n");

    // ── Main loop ──
    while(true){
        std::string line=read_line();
        if(line.empty())break;

        auto prompt=parse_prompt(line,tok);
        if(prompt.empty()) continue;

        int max_tokens=find_int(line,"max_tokens",30);
        float temp=find_float(line,"temperature",0);
        int top_k=find_int(line,"top_k",0);
        bool stream=find_int(line,"stream",0)==1;

        cudaStreamSynchronize(S.st);

        // Prompt tokens through full decode (no prefill)
        for(size_t s=0;s<prompt.size();s++){
            embed(S,prompt[s]);
            cudaStreamSynchronize(S.st);
            decode_step(S,(int)s);
        }

        std::vector<uint32_t> output;
        output.insert(output.end(),prompt.begin(),prompt.end());

        for(int s=0;s<max_tokens;s++){
            int seq_pos=(int)output.size()-1;

            // Final norm → lm_head
            die(blackwell::kernels::fused_rmsnorm(S.d_xn,S.d_residual,S.d_fn,H,S.eps,S.st),"frn");
            die(blackwell::kernels::quantize_int8(S.d_ai,S.d_as,S.d_xn,H,S.st),"fq");
            die(blackwell::kernels::gemv_int8_warp(S.d_logits,S.d_ai,S.d_as,
                (S.lm_head.d?S.lm_head.d:S.emb.d),
                (S.lm_head.d?S.lm_head.sc:S.emb.sc),H,V,S.st),"lm");

            die(blackwell::kernels::sample_gpu(S.d_logits,V,temp,top_k,
                S.d_next_id,0xdeadbeefLL,seq_pos,S.st),"sample");
            uint32_t next_id;
            cudaMemcpy(&next_id,S.d_next_id,sizeof(int),cudaMemcpyDeviceToHost);
            output.push_back(next_id);

            // Streaming: emit SSE line after each generated token
            if(stream){
                std::string tok_txt=tok.decode(next_id);
                printf("data: {\"token\":%u,\"text\":\"",next_id);fflush(stdout);
                for(size_t i=0;i<tok_txt.size();i++){
                    char c=tok_txt[i];
                    if(c=='"')printf("\\\"");
                    else if(c=='\\')printf("\\\\");
                    else if(c=='\n')printf("\\n");
                    else if(c=='\r')printf("\\r");
                    else if(c=='\t')printf("\\t");
                    else if((unsigned char)c<0x20)printf("\\u%04x",(unsigned char)c);
                    else printf("%c",c);
                }
                printf("\"}\n\n");fflush(stdout);
            }

            if(next_id==151643)break;

            embed(S,next_id);
            cudaStreamSynchronize(S.st);
            decode_step(S,seq_pos+1);
        }

        std::vector<uint32_t> gen(output.begin()+(int)prompt.size(),output.end());
        if(!stream){
            write_results(gen,tok);
        } else {
            printf("data: [DONE]\n\n");fflush(stdout);
        }

        // Reset state
        cudaMemset(S.d_cs,0,NL*CD*(CK-1)*4);
        cudaMemset(S.d_rs,0,NL*NV*HD*HD*4);
        cudaMemset(S.d_kc,0,kvs); cudaMemset(S.d_vc,0,kvs);
        cudaStreamSynchronize(S.st);
    }
    return 0;
}
