#include "blackwell/int4_weights.h"
using namespace blackwell::weights;
// bench/text_generate_gemma4_12b_qat.cu — Gemma 4 12B QAT INT4 text generation
// 48 layers, H=3840, I=15360, nqh=16, nkv=8, V=262144. 8 FA + 40 SWA.
// SWA: hd=256, nkv=8, rope_theta=10000. FA: hd=512, nkv=1, K=V, rope_theta=1e6.
//
// Usage: ./bench/text_generate_gemma4_12b_qat [token_id_file] [num_tokens] [weight_dir]
//   token_id_file: text file with comma-separated token IDs (from gemma_wrapper.py)
//   num_tokens: tokens to generate (default 30)
//   weight_dir: default weights_gemma4_12b_qat
//
// Build:
//   CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build && cmake --build build --parallel

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <string>
#include <cstdint>
#include <cmath>
#include <chrono>
#include "blackwell/kernels.h"

static void die(cudaError_t e, const char* m) {
    if(e!=cudaSuccess){fprintf(stderr,"FAIL %s: %s\n",m,cudaGetErrorString(e));exit(1);}
}

// Gemma 4 12B QAT: 48 layers, H=3840, I=15360, nqh=16, V=262144
// SWA layers (40): hd_swa=256, nkv_swa=8, Q=4096, KV=2048, roof_theta=10000
// FA layers (8 at 5,11,17,23,29,35,41,47): hd_fa=512, nkv_fa=1, K=V (k_eq_v), rope_theta=1000000

// Metadata from GGUF: head_count=16, head_count_kv=[8,8,8,8,8,1,...8,1]
// key_length=512 (global), key_length_swa=256, shared_kv_layers=0
const int H=3840, Q=4096, KV=2048, I=15360;
const int nqh=16, nkv_swa=8, hd_swa=256;
const int nkv_fa=1, hd_fa=512;
const int MAXSEQ=2048;
const int SWA_WINDOW=1024;
const float eps=1e-6f;
const float FINAL_LOGIT_SOFTCAP = 30.0f;
const int V=262144;
const int NL=48;
const float rope_theta_swa=10000.0f;
const float rope_theta_fa=1000000.0f;
// Full-attention layers (non-SWA) at indexes 5,11,17,23,29,35,41,47
// SWA layers have 1, full-attention have 0
const bool SWA_LAYERS[NL] = {
    1,1,1,1,1,0, 1,1,1,1,1,0,
    1,1,1,1,1,0, 1,1,1,1,1,0,
    1,1,1,1,1,0, 1,1,1,1,1,0,
    1,1,1,1,1,0, 1,1,1,1,1,0};

struct LW4 {
    DevW4f16 q,k,v,o,g,u,d;
    float *qn,*kn,*rn_in,*rn_post;
    bool swa; bool fa;
    int l_nqh;   // Q heads
    int l_nkv;   // K/V heads
    int l_hd;    // head dim (512 for FA, 256 for SWA)
    int l_q_dim; // Q projection dim
    int l_k_dim; // K projection dim
    int l_v_dim; // V projection dim
    int l_o_dim; // O projection input dim
    float l_rope_theta;
    float* l_cos_cache; // RoPE cos cache for this layer's head_dim
    float* l_sin_cache;
};

static void build_rope_cache(float* cos_cache, float* sin_cache, int max_seq, int head_dim, float base_theta) {
    int pairs = head_dim / 2;
    for (int pos = 0; pos < max_seq; pos++)
        for (int d = 0; d < pairs; d++) {
            float theta = (float)pos * powf(base_theta, -2.0f * (float)d / (float)head_dim);
            cos_cache[pos * pairs + d] = cosf(theta);
            sin_cache[pos * pairs + d] = sinf(theta);
        }
}

int main(int argc, char** argv) {
    const char* TOKEN_FILE = (argc > 1) ? argv[1] : "prompt_ids.txt";
    int num_tokens = (argc > 2) ? atoi(argv[2]) : 30;
    const char* WDIR = (argc > 3) ? argv[3] : "weights_gemma4_12b_qat";
    
    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    fprintf(stderr, "# Gemma 4 12B QAT INT4 — %s\n  Weights: %s\n  Token file: %s\n  Max new: %d\n",
            P.name, WDIR, TOKEN_FILE, num_tokens);

    // Read prompt token IDs from file
    std::vector<uint32_t> ids;
    {FILE* f=fopen(TOKEN_FILE,"r"); if(!f){fprintf(stderr,"FAIL: cannot open %s\n",TOKEN_FILE);return 1;}
     int id; while(fscanf(f,"%d,",&id)==1) ids.push_back((uint32_t)id); fclose(f);}
    if(ids.empty()){fprintf(stderr,"FAIL: empty prompt token file\n");return 1;}
    fprintf(stderr,"  Prompt tokens: %zu\n",ids.size());

    float *d_x32,*d_xi_f,*d_res,*d_Q,*d_K,*d_V,*d_attn,*d_proj,*d_gate,*d_up;
    uint8_t *d_x_i4; float *d_x_i4_sc;
    uint8_t *d_attn_i4; float *d_attn_i4_sc;
    uint8_t *d_mlp_i4; float *d_mlp_i4_sc;
    float *d_fn,*d_kc,*d_vc,*d_logits;
    int *d_next_id,*d_seq_pos;
    int *d_swa_pos; // SWA-capped seq_pos

    // FA layers need larger buffers for double Q heads (8192 vs 4096)
    const int MAX_Q_DIM = 8192;
    #define AL(p,n) die(cudaMalloc(&(p),(n)),"malloc")
    AL(d_x32,H*4); AL(d_xi_f,H*4); AL(d_res,H*4);
    AL(d_x_i4,H/2); AL(d_x_i4_sc,(H/16)*4);
    AL(d_Q,MAX_Q_DIM*4); AL(d_K,KV*4); AL(d_V,KV*4); AL(d_attn,MAX_Q_DIM*4);
    AL(d_attn_i4,MAX_Q_DIM/2); AL(d_attn_i4_sc,(MAX_Q_DIM/16)*4);
    AL(d_proj,H*4); AL(d_gate,I*4); AL(d_up,I*4);
    AL(d_mlp_i4,I/2); AL(d_mlp_i4_sc,(I/16)*4);
    AL(d_fn,H*4);
    // KV cache: 40 SWA layers × 8 heads × hd_swa=256 + 8 FA layers × 1 head × hd_fa=512
    const size_t KV_SLOT_SWA = (size_t)nkv_swa * MAXSEQ * hd_swa; // 8*2048*256
    const size_t KV_SLOT_FA  = (size_t)nkv_fa  * MAXSEQ * hd_fa;  // 1*2048*512
    // Calculate per-layer offsets: SWA layers at their natural position, FA layers at theirs
    // Total = sum of per-layer slots
    size_t total_kv = 0;
    size_t kv_offsets[NL];
    for (int l = 0; l < NL; l++) {
        kv_offsets[l] = total_kv;
        total_kv += (SWA_LAYERS[l] ? KV_SLOT_SWA : KV_SLOT_FA);
    }
    AL(d_kc, total_kv * 4);
    AL(d_vc, total_kv * 4);
    AL(d_logits,V*4); AL(d_next_id,4); AL(d_seq_pos,4); AL(d_swa_pos,4);
    #undef AL

    float iv7=1.f/7.f;
    std::vector<float> tmp;
    tmp.assign(H/16,iv7); cudaMemcpy(d_x_i4_sc,tmp.data(),(H/16)*4,cudaMemcpyHostToDevice);
    tmp.assign(MAX_Q_DIM/16,iv7); cudaMemcpy(d_attn_i4_sc,tmp.data(),(MAX_Q_DIM/16)*4,cudaMemcpyHostToDevice);
    tmp.assign(I/16,iv7); cudaMemcpy(d_mlp_i4_sc,tmp.data(),(I/16)*4,cudaMemcpyHostToDevice);

    // RoPE caches moved to per-layer init (after weight loading, below)

    fprintf(stderr,"Loading weights...\n");
    std::vector<LW4> W(NL);
    char p[256];
    // Separate RoPE caches for SWA (hd=256) and FA (hd=512)
    int rope_pairs_swa = hd_swa / 2;
    int rope_pairs_fa = hd_fa / 2;
    float* d_cos_swa; float* d_sin_swa;
    float* d_cos_fa;  float* d_sin_fa;
    #define AL(p,n) die(cudaMalloc(&(p),(n)),"malloc")
    AL(d_cos_swa, MAXSEQ * rope_pairs_swa * 4);
    AL(d_sin_swa, MAXSEQ * rope_pairs_swa * 4);
    AL(d_cos_fa,  MAXSEQ * rope_pairs_fa  * 4);
    AL(d_sin_fa,  MAXSEQ * rope_pairs_fa  * 4);
    {
        std::vector<float> c(MAXSEQ * rope_pairs_swa), s(MAXSEQ * rope_pairs_swa);
        build_rope_cache(c.data(), s.data(), MAXSEQ, hd_swa, rope_theta_swa);
        cudaMemcpy(d_cos_swa, c.data(), MAXSEQ * rope_pairs_swa * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_sin_swa, s.data(), MAXSEQ * rope_pairs_swa * 4, cudaMemcpyHostToDevice);
    }
    {
        std::vector<float> c(MAXSEQ * rope_pairs_fa), s(MAXSEQ * rope_pairs_fa);
        build_rope_cache(c.data(), s.data(), MAXSEQ, hd_fa, rope_theta_fa);
        cudaMemcpy(d_cos_fa, c.data(), MAXSEQ * rope_pairs_fa * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_sin_fa, s.data(), MAXSEQ * rope_pairs_fa * 4, cudaMemcpyHostToDevice);
    }

    for(int l=0;l<NL;++l){
        W[l].swa = SWA_LAYERS[l];
        W[l].fa = !SWA_LAYERS[l];
        if (W[l].fa) {
            // Full-attention: hd=512, nkv=1, nqh=16, K=V (k_eq_v)
            // Q = 16*512=8192, K = 1*512=512, V = 1*512=512
            W[l].l_hd = hd_fa;
            W[l].l_nqh = 16;
            W[l].l_nkv = 1;
            W[l].l_q_dim = 8192;
            W[l].l_k_dim = 512;
            W[l].l_v_dim = 512; // K == V
            W[l].l_o_dim = 8192;
            W[l].l_rope_theta = rope_theta_fa;
            W[l].l_cos_cache = d_cos_fa;
            W[l].l_sin_cache = d_sin_fa;
        } else {
            // Sliding window: hd=256, nkv=8, nqh=16
            // Q = 16*256=4096, K = 8*256=2048, V = 8*256=2048
            W[l].l_hd = hd_swa;
            W[l].l_nqh = 16;
            W[l].l_nkv = 8;
            W[l].l_q_dim = Q;
            W[l].l_k_dim = KV;
            W[l].l_v_dim = KV;
            W[l].l_o_dim = Q;
            W[l].l_rope_theta = rope_theta_swa;
            W[l].l_cos_cache = d_cos_swa;
            W[l].l_sin_cache = d_sin_swa;
        }
        // Layer 47 has no q_proj — share from layer 46
        if (l == 47) {
            fprintf(stderr,"  Layer 47: sharing q_proj from layer 46\n");
            W[l].q = W[46].q;
        } else {
            snprintf(p,256,"%s/%d_self_attn.q_proj",WDIR,l); W[l].q=upload_w4_f16sc(p);
        }
        snprintf(p,256,"%s/%d_self_attn.k_proj",WDIR,l); W[l].k=upload_w4_f16sc(p);
        // FA layers have no v_proj — K = V (k_eq_v in Gemma 4)
        if (W[l].fa) {
            fprintf(stderr,"  Layer %d (FA): K == V (k_eq_v, no separate v_proj)\n",l);
            W[l].v = W[l].k; // K and V share the same weight matrix
        } else {
            snprintf(p,256,"%s/%d_self_attn.v_proj",WDIR,l); W[l].v=upload_w4_f16sc(p);
        }
        snprintf(p,256,"%s/%d_self_attn.o_proj",WDIR,l); W[l].o=upload_w4_f16sc(p);
        snprintf(p,256,"%s/%d_mlp.gate_proj",WDIR,l); W[l].g=upload_w4_f16sc(p);
        snprintf(p,256,"%s/%d_mlp.up_proj",WDIR,l); W[l].u=upload_w4_f16sc(p);
        snprintf(p,256,"%s/%d_mlp.down_proj",WDIR,l); W[l].d=upload_w4_f16sc(p);
    }

    // QK head norms: per-layer files ({l}_attn_q_norm.f32 / {l}_attn_k_norm.f32)
    // Files are native per-layer dim: 256 floats for SWA, 512 for FA
    // (converter writes actual tensor element counts). Read l_hd directly.
    for(int l=0;l<NL;++l){
        int l_hd = W[l].l_hd;
        float w512[512];
        snprintf(p,256,"%s/%d_attn_q_norm.f32",WDIR,l);
        FILE* f=fopen(p,"rb");
        int nq = 0;
        if (f) { nq = (int)fread(w512,4,l_hd,f); fclose(f); }
        // Safety: zero-fill if file short (shouldn't happen for native dims)
        for (int i = nq; i < l_hd; i++) w512[i] = 0.0f;
        cudaMalloc(&W[l].qn,l_hd*4);cudaMemcpy(W[l].qn,w512,l_hd*4,cudaMemcpyHostToDevice);

        snprintf(p,256,"%s/%d_attn_k_norm.f32",WDIR,l);
        f=fopen(p,"rb");
        int nk = 0;
        if (f) { nk = (int)fread(w512,4,l_hd,f); fclose(f); }
        for (int i = nk; i < l_hd; i++) w512[i] = 0.0f;
        cudaMalloc(&W[l].kn,l_hd*4);cudaMemcpy(W[l].kn,w512,l_hd*4,cudaMemcpyHostToDevice);
    }

    // Gemma 4 has 4 RMSNorms per layer:
    //   attn_norm -> input_layernorm (pre-attention)
    //   post_attention_norm -> post_attn_norm (post-attention residual)
    //   ffn_norm -> post_attention_layernorm (pre-FFN)
    //   post_ffw_norm -> post_ffn_norm (post-FFN residual)
    // Our LW4 struct uses rn_in (pre-attention) and rn_post (pre-FFN)
    // We also need rn_post_attn (post-attention) and rn_post_ffn (post-FFN)
    struct LW4_extra { float* rn_post_attn; float* rn_post_ffn; };
    std::vector<LW4_extra> Wx(NL);

    for(int l=0;l<NL;++l){
        float* w=(float*)malloc(H*4);
        // Pre-attention RMSNorm
        snprintf(p,256,"%s/%d_input_layernorm.f32",WDIR,l);
        {FILE*f=fopen(p,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&W[l].rn_in,H*4);cudaMemcpy(W[l].rn_in,w,H*4,cudaMemcpyHostToDevice);
        // Post-attention RMSNorm
        snprintf(p,256,"%s/%d_post_attn_norm.f32",WDIR,l);
        {FILE*f=fopen(p,"rb");(void)fread(w,4,H,f); fclose(f);}
        cudaMalloc(&Wx[l].rn_post_attn,H*4); cudaMemcpy(Wx[l].rn_post_attn,w,H*4,cudaMemcpyHostToDevice);
        // Pre-FFN RMSNorm (maps to post_attention_layernorm in our naming)
        snprintf(p,256,"%s/%d_post_attention_layernorm.f32",WDIR,l);
        {FILE*f=fopen(p,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&W[l].rn_post,H*4);cudaMemcpy(W[l].rn_post,w,H*4,cudaMemcpyHostToDevice);
        // Post-FFN RMSNorm
        snprintf(p,256,"%s/%d_post_ffn_norm.f32",WDIR,l);
        {FILE*f=fopen(p,"rb");(void)fread(w,4,H,f); fclose(f);}
        cudaMalloc(&Wx[l].rn_post_ffn,H*4); cudaMemcpy(Wx[l].rn_post_ffn,w,H*4,cudaMemcpyHostToDevice);
        free(w);
    }
    {float*w=(float*)malloc(H*4);
    snprintf(p,256,"%s/final_norm.f32",WDIR);
    FILE*f=fopen(p,"rb");(void)fread(w,4,H,f);fclose(f);
    cudaMemcpy(d_fn,w,H*4,cudaMemcpyHostToDevice);free(w);}

    // lm_head: Gemma 4 QAT uses tied embeddings (embed_tokens == lm_head)
    DevW4f16 lm_head_w;
    char lm_path[256];
    snprintf(lm_path,256,"%s/lm_head.int4_t",WDIR);
    FILE* lm_f=fopen(lm_path,"rb");
    if (lm_f) {
        fclose(lm_f);
        snprintf(p,256,"%s/lm_head",WDIR);
        lm_head_w=upload_w4_f16sc(p);
        fprintf(stderr,"lm_head loaded: %dx%d (INT4)\n",lm_head_w.N,lm_head_w.K);
    } else {
        // Tied embeddings: use embed_tokens as lm_head
        fprintf(stderr,"lm_head not found — using tied embed_tokens as lm_head\n");
        snprintf(p,256,"%s/embed_tokens",WDIR);
        lm_head_w=upload_w4_f16sc(p);
        fprintf(stderr,"lm_head (tied): %dx%d (INT4)\n",lm_head_w.N,lm_head_w.K);
    }
    
    // Embed token table
    uint8_t* host_embed_d=new uint8_t[(size_t)H*V/2];
    float* host_embed_sc=new float[V*(H/16)];
    snprintf(p,256,"%s/embed_tokens.int4_t",WDIR);
    {FILE*f=fopen(p,"rb");int h[5];fread(h,4,5,f);
     fread(host_embed_d,1,(size_t)h[0]*h[1]/2,f);fclose(f);}
    snprintf(p,256,"%s/embed_tokens.scale_t",WDIR);
    {FILE*f=fopen(p,"rb");int h[5];fread(h,4,5,f);
     __half*tmp=new __half[(size_t)h[3]*h[4]];fread(tmp,2,(size_t)h[3]*h[4],f);fclose(f);
     for(size_t i=0;i<(size_t)h[3]*h[4];++i) host_embed_sc[i]=__half2float(tmp[i]);delete[] tmp;}
    fprintf(stderr,"Embed loaded: %dx%d (INT4)\n",H,V);

    cudaStream_t st; die(cudaStreamCreate(&st),"stream");
    std::vector<float> h_embed(H);
    std::vector<uint32_t> generated;

    cudaMemset(d_kc, 0, total_kv * 4);
    cudaMemset(d_vc, 0, total_kv * 4);

    // ── Forward: process one token through all layers ──
    auto forward_token = [&](int step) {
        cudaMemcpy(d_seq_pos, &step, 4, cudaMemcpyHostToDevice);
        for(int l=0;l<NL;++l){
            int l_hd   = W[l].l_hd;
            int l_nqh  = W[l].l_nqh;
            int l_nkv  = W[l].l_nkv;
            int l_q_dim= W[l].l_q_dim;
            int l_k_dim= W[l].l_k_dim;
            int l_v_dim= W[l].l_v_dim;
            int l_o_dim= W[l].l_o_dim;

            // Per-layer KV slot size in the variable-stride cache
            size_t kv_slot = (size_t)l_nkv * MAXSEQ * l_hd;
            size_t kv_off = kv_offsets[l];

            // Save residual
            cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st);
            // Pre-attention RMSNorm
            blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_in,H,eps,st);
            blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st);
            blackwell::kernels::gemv_int4_warp_f16wsc(d_Q,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].q.d,W[l].q.sc16,H,l_q_dim,st);
            blackwell::kernels::gemv_int4_warp_f16wsc(d_K,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].k.d,W[l].k.sc16,H,l_k_dim,st);
            blackwell::kernels::gemv_int4_warp_f16wsc(d_V,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].v.d,W[l].v.sc16,H,l_v_dim,st);

            // QK head norms
            blackwell::kernels::head_norm(d_Q,W[l].qn,l_nqh,l_hd,eps,st);
            blackwell::kernels::head_norm(d_K,W[l].kn,l_nkv,l_hd,eps,st);

            // RoPE (per-layer head_dim and theta)
            blackwell::kernels::fused_rope_decode(d_Q,W[l].l_cos_cache,W[l].l_sin_cache,d_seq_pos,l_nqh,l_hd,MAXSEQ,st);
            blackwell::kernels::fused_rope_decode(d_K,W[l].l_cos_cache,W[l].l_sin_cache,d_seq_pos,l_nkv,l_hd,MAXSEQ,st);

            // Attention with sliding window cap for SWA layers
            int attn_step = step;
            if (W[l].swa && attn_step > SWA_WINDOW) attn_step = SWA_WINDOW;

            // KV cache write: use update_kv_cache for SWA, D2D for FA (K==V means same src for both)
            if (W[l].fa) {
                int n_heads_k = l_k_dim / l_hd; // 1
                for (int h = 0; h < n_heads_k; h++) {
                    cudaMemcpyAsync(d_kc+kv_off + (size_t)h*MAXSEQ*l_hd + step*l_hd,
                                    d_K + (size_t)h*l_hd, l_hd*4, cudaMemcpyDeviceToDevice, st);
                    cudaMemcpyAsync(d_vc+kv_off + (size_t)h*MAXSEQ*l_hd + step*l_hd,
                                    d_V + (size_t)h*l_hd, l_hd*4, cudaMemcpyDeviceToDevice, st);
                }
            } else {
                blackwell::kernels::update_kv_cache(d_kc+kv_off,d_vc+kv_off,d_K,d_V,0,step,l_nkv,l_hd,MAXSEQ,st);
            }

            // Attention: use per-layer head count and head_dim. FA has 16 Q heads × 512 → 1 KV head × 512.
            // kv_batch_elems = total_kv (floats between M sequences). kv_layer_elems = 0 (single-sequence).
            blackwell::kernels::attention_decode_batched_gqa(d_attn,d_Q,d_kc,d_vc,attn_step,l_nqh,l_nkv,l_hd,MAXSEQ,1,
                total_kv,kv_off,st);

            // Quantize attention output for o_proj
            blackwell::kernels::quantize_int4(d_attn_i4,d_attn_i4_sc,d_attn,l_o_dim,st);
            blackwell::kernels::gemv_int4_warp_f16wsc(d_proj,(const uint8_t*)d_attn_i4,d_attn_i4_sc,W[l].o.d,W[l].o.sc16,l_o_dim,H,st);
            // Post-attention RMSNorm + residual (4-norm Gemma 4 pattern)
            blackwell::kernels::fused_rmsnorm(d_xi_f,d_proj,Wx[l].rn_post_attn,H,eps,st);
            blackwell::kernels::vector_add_fp32(d_x32,d_xi_f,d_res,H,st);

            // FFN path
            cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st);
            blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_post,H,eps,st);
            blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st);
            blackwell::kernels::gemv_int4_warp_f16wsc(d_gate,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].g.d,W[l].g.sc16,H,I,st);
            blackwell::kernels::gemv_int4_warp_f16wsc(d_up,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].u.d,W[l].u.sc16,H,I,st);
            blackwell::kernels::apply_geglu(d_gate,d_gate,d_up,I,st);
            blackwell::kernels::quantize_int4(d_mlp_i4,d_mlp_i4_sc,d_gate,I,st);
            blackwell::kernels::gemv_int4_warp_f16wsc(d_proj,(const uint8_t*)d_mlp_i4,d_mlp_i4_sc,W[l].d.d,W[l].d.sc16,I,H,st);
            blackwell::kernels::fused_rmsnorm(d_xi_f,d_proj,Wx[l].rn_post_ffn,H,eps,st);
            blackwell::kernels::vector_add_fp32(d_x32,d_xi_f,d_res,H,st);
        }
        blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,d_fn,H,eps,st);
        blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st);
        blackwell::kernels::gemv_int4_warp_f16wsc(d_logits,(const uint8_t*)d_x_i4,d_x_i4_sc,lm_head_w.d,lm_head_w.sc16,H,V,st);
        // NOTE: final_logit_softcapping (30.0) exists but saturates all logits
        // to ±30 with tanh, producing uniform distribution → EOS token.
        // INT4 quantization noise pushes raw logits beyond softcap range.
        // Skip softcap until quality improves.
        // blackwell::kernels::apply_logit_softcap(d_logits, V, FINAL_LOGIT_SOFTCAP, st);
    };

    // Debug: print pre-decode hidden state
    {
        std::vector<float> hh(H);
        cudaMemcpy(hh.data(), d_x32, H*4, cudaMemcpyDeviceToHost);
        float maxv=0, minv=0, sum=0;
        for(int i=0;i<H;i++){if(i==0||hh[i]>maxv)maxv=hh[i];if(i==0||hh[i]<minv)minv=hh[i];sum+=fabsf(hh[i]);}
        fprintf(stderr,"  pre-decode hidden state: min=%.4f max=%.4f avg_abs=%.4f\n",minv,maxv,sum/H);
    }

    // Prefill: process prompt tokens
    for(int step=0;step<(int)ids.size();++step){
        uint32_t tid=ids[step];
        dequant_embed_row(h_embed.data(),tid,host_embed_d,host_embed_sc,H);
        cudaMemcpyAsync(d_x32,h_embed.data(),H*4,cudaMemcpyHostToDevice,st);
        forward_token(step);
    }

    uint32_t next_id=ids.back();
    auto t0=std::chrono::high_resolution_clock::now();
    for(int step=ids.size();step<(int)(ids.size()+num_tokens);++step){
        uint32_t tid=next_id;
        dequant_embed_row(h_embed.data(),tid,host_embed_d,host_embed_sc,H);
        cudaMemcpyAsync(d_x32,h_embed.data(),H*4,cudaMemcpyHostToDevice,st);
        forward_token(step);

        blackwell::kernels::sample_argmax_gpu(d_logits,V,d_next_id,st);
        cudaStreamSynchronize(st);
        cudaMemcpy(&next_id,d_next_id,4,cudaMemcpyDeviceToHost);
        if (step == (int)ids.size()) {
            float l0, l1, l2;
            cudaMemcpy(&l0, d_logits, 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(&l1, d_logits+1, 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(&l2, d_logits+2, 4, cudaMemcpyDeviceToHost);
            fprintf(stderr, "  decode step0: logits[0]=%.1f logits[1]=%.1f logits[2]=%.1f\n", l0, l1, l2);
            fprintf(stderr, "  first decoded token: %d\n", next_id);
        }
        if(next_id==0||next_id==1) break; // EOS only (Gemma 4: 1=EOS, 2=BOS is valid output)
        generated.push_back(next_id);
    }
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();

    fprintf(stderr,"\n[Output tokens]");
    for(auto id:generated) fprintf(stderr," %u",id);
    fprintf(stderr,"\n[Output text] Use gemma_wrapper.py to decode token IDs\n");
    fprintf(stderr,"\nStats: %zu tokens, %.1f ms, %.1f ms/tok = %.0f t/s\n",
            generated.size(),ms,ms/generated.size(),1000.0/(ms/generated.size()));
    return 0;
}
