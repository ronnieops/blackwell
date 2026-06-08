// bench_ppl.cu — Perplexity benchmark for Blackwell INT8 models
//
// Usage: ./bench/bench_ppl <model> <num_tokens>
//   model: 1.7b (default), 8b
//
// For 9b GDN: use bench_ppl_9b.cu instead

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include "blackwell/kernels.h"
#include "blackwell/bpe_tokenizer.h"

#define AL(e) do{cudaError_t _e=(e);if(_e!=cudaSuccess){\
    fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));exit(1);}}while(0)

// ── Test corpus ───────────────────────────────────────────────────────
static const char* TEST_TEXT =
    "The Republic of Austria is a federal republic in Central Europe . "
    "It is bordered by Germany to the northwest , the Czech Republic to the north , "
    "Slovakia to the northeast , Hungary to the east , Slovenia and Italy to the south , "
    "Switzerland and Liechtenstein to the west . "
    "The capital of Austria is Vienna . "
    "The official language is German .";

// ── Logprob kernel: log P(correct | logits) via logsumexp trick ──────
// Simple max kernel (no shared mem issues)
// Logprob kernel: log P(correct | logits) via logsumexp trick
__global__ void logprob_kernel(const float* logits, int V, int correct_id, float* out) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int n = blockDim.x;

    float mx = -INFINITY;
    for (int i = tid; i < V; i += n) { float v = logits[i]; if (v > mx) mx = v; }
    smem[tid] = mx; __syncthreads();
    for (int o = n/2; o > 0; o >>= 1) { if (tid < o && smem[tid+o] > smem[tid]) smem[tid] = smem[tid+o]; __syncthreads(); }
    float maxv = smem[0]; __syncthreads();

    float sum_exp = 0.0f;
    for (int i = tid; i < V; i += n) sum_exp += expf(logits[i] - maxv);
    smem[tid] = sum_exp; __syncthreads();
    for (int o = n/2; o > 0; o >>= 1) { if (tid < o) smem[tid] += smem[tid+o]; __syncthreads(); }

    float logp = (correct_id >= 0 && correct_id < V) ? logits[correct_id] - (maxv + logf(smem[0])) : 0.0f;
    if (tid == 0) out[0] = logp;
}

// ── Weight loading ────────────────────────────────────────────────────
struct Int8W { int K, N; int8_t* d; float* sc; };
struct Fp16W { int K, N; __half* d; };

static Fp16W upload_fp16(const char* path) {
    FILE* f=fopen(path,"rb"); if(!f){fprintf(stderr,"FAIL open %s\n",path);exit(1);}
    int h[2]; (void)fread(h,4,2,f);
    size_t n=(size_t)h[0]*h[1];
    std::vector<uint16_t> tmp(n); (void)fread(tmp.data(),2,n,f); fclose(f);
    __half* d; AL(cudaMalloc(&d,n*2)); AL(cudaMemcpy(d,tmp.data(),n*2,cudaMemcpyHostToDevice));
    return {h[0],h[1],d};
}

static cudaError_t gemv_dispatch(float* y, bool is_fp16, __half* fp16_w,
    const int8_t* x_i8, const float* x_sc, int K, int N,
    const int8_t* w_i8, const float* w_sc, cudaStream_t st) {
    if (is_fp16 && fp16_w) {
        blackwell::kernels::gemv_fp16_warp_launch(y, fp16_w, x_i8, x_sc, K, N, st);
        return cudaGetLastError();
    } else {
        return blackwell::kernels::gemv_int8_warp(y, x_i8, x_sc, w_i8, w_sc, K, N, st);
    }
}
static Int8W load_int8_w(const char* prefix) {
    char p[512]; snprintf(p,512,"%s.int8_t",prefix);
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"FAIL open %s\n",p);exit(1);}
    int h[5]; (void)fread(h,4,5,f);
    size_t sz=(size_t)h[0]*h[1];
    std::vector<int8_t> tmp(sz); (void)fread(tmp.data(),1,sz,f); fclose(f);
    Int8W w{h[0],h[1],nullptr,nullptr};
    AL(cudaMalloc(&w.d,sz)); AL(cudaMemcpy(w.d,tmp.data(),sz,cudaMemcpyHostToDevice));
    snprintf(p,512,"%s.scale_t",prefix); f=fopen(p,"rb"); if(!f){fprintf(stderr,"FAIL open %s\n",p);exit(1);}
    (void)fread(h,4,5,f); size_t ns=(size_t)h[3]*h[4];
    std::vector<float> ts(ns); (void)fread(ts.data(),4,ns,f); fclose(f);
    AL(cudaMalloc(&w.sc,ns*4)); AL(cudaMemcpy(w.sc,ts.data(),ns*4,cudaMemcpyHostToDevice));
    return w;
}
static float* load_f32(const char* pfx, int n) {
    char p[512]; snprintf(p,512,"%s.f32",pfx);
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"FAIL open %s\n",p);exit(1);}
    std::vector<float> tmp(n); (void)fread(tmp.data(),4,n,f); fclose(f);
    float* d; AL(cudaMalloc(&d,n*4)); AL(cudaMemcpy(d,tmp.data(),n*4,cudaMemcpyHostToDevice));
    return d;
}

// ── Head norm + RoPE (inline kernels) ────────────────────────────────
__global__ void hn_kernel(float* d, const float* w, int nh, int hd, float eps) {
    int h = blockIdx.x; if (h >= nh) return;
    float* p = d + h * hd; __shared__ float ws[4];
    float s = 0; int tid = threadIdx.x;
    for (int i = tid; i < hd; i += blockDim.x) s += p[i] * p[i];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if ((tid & 31) == 0) ws[tid >> 5] = s; __syncthreads();
    if (tid < 32) { float v = (tid < 4) ? ws[tid] : 0; for (int o = 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o); if (tid == 0) ws[0] = rsqrtf(v / hd + eps); }
    __syncthreads(); float is = ws[0];
    for (int i = tid; i < hd; i += blockDim.x) p[i] = p[i] * is * w[i];
}
__global__ void rope_kernel(float* d, int nh, int hd, int pos) {
    int h = blockIdx.x, t = threadIdx.x; if (h >= nh || t >= hd/2) return;
    float* pair = d + h * hd + t * 2;
    float th = (float)pos * powf(1000000.0f, -2.0f * (float)t / (float)hd);
    float c = cosf(th), s = sinf(th), x = pair[0], y = pair[1];
    pair[0] = x * c - y * s; pair[1] = x * s + y * c;
}

// ── Decode step for transformer models (1.7B/8B) ─────────────────────
struct LayerW {
    Int8W q, k, v, o, gate, up, down;
    __half *fp16_q, *fp16_k, *fp16_v, *fp16_o;
    __half *fp16_gate, *fp16_up, *fp16_down;
    bool is_fp16;
    float *rn_in, *rn_post, *qk_n;
};

static void decode_step(float* d_residual, int seq_pos, int l,
    int NL, int H, int Q, int KV, int ID, int V,
    int nqh, int nkv, int hd,
    int8_t* d_ai, float* d_as,
    float* d_Q, float* d_K, float* d_V,
    float* d_attn,
    int8_t* d_attn_i8, float* d_attn_i8s,
    float* d_proj,
    float* d_gate, float* d_up, float* d_mlp,
    int8_t* d_mlp_i8, float* d_mlp_i8s,
    const LayerW& L,
    float* d_kc, float* d_vc, cudaStream_t st)
{
    size_t kv_off = (size_t)l * nkv * hd * KV;

    // Input layernorm + quant
    AL(blackwell::kernels::fused_rmsnorm_quant_int8(d_ai, d_as, d_residual, L.rn_in, H, 1e-6f, st));
    // QKV
    AL(gemv_dispatch(d_Q, L.is_fp16, L.fp16_q, d_ai, d_as, H, Q, L.q.d, L.q.sc, st));
    AL(gemv_dispatch(d_K, L.is_fp16, L.fp16_k, d_ai, d_as, H, KV, L.k.d, L.k.sc, st));
    AL(gemv_dispatch(d_V, L.is_fp16, L.fp16_v, d_ai, d_as, H, KV, L.v.d, L.v.sc, st));
    // Head norm + RoPE (qk_n layout: [nqh*hd] q_norms + [nkv*hd] k_norms)
    hn_kernel<<<nqh, 128, 0, st>>>(d_Q, L.qk_n, nqh, hd, 1e-6f);
    hn_kernel<<<nkv, 128, 0, st>>>(d_K, L.qk_n + nqh*hd, nkv, hd, 1e-6f);
    rope_kernel<<<nqh, hd/2, 0, st>>>(d_Q, nqh, hd, seq_pos);
    rope_kernel<<<nkv, hd/2, 0, st>>>(d_K, nkv, hd, seq_pos);
    AL(cudaGetLastError());
    // KV cache
    AL(blackwell::kernels::update_kv_cache(d_kc + kv_off, d_vc + kv_off, d_K, d_V, 0, seq_pos, nkv, hd, KV, st));
    // Attention
    AL(blackwell::kernels::attention_decode_gqa(d_attn, d_Q, d_kc + kv_off, d_vc + kv_off, seq_pos, nqh, nkv, hd, KV, st));
    // Out proj + residual 1
    AL(blackwell::kernels::quantize_int8(d_attn_i8, d_attn_i8s, d_attn, Q, st));
    AL(gemv_dispatch(d_proj, L.is_fp16, L.fp16_o, d_attn_i8, d_attn_i8s, Q, H, L.o.d, L.o.sc, st));
    AL(blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_residual, H, st));
    AL(cudaMemcpyAsync(d_residual, d_proj, H*4, cudaMemcpyDeviceToDevice, st));
    // Post-attention layernorm
    AL(blackwell::kernels::fused_rmsnorm_quant_int8(d_ai, d_as, d_proj, L.rn_post, H, 1e-6f, st));
    // MLP
    AL(gemv_dispatch(d_gate, L.is_fp16, L.fp16_gate, d_ai, d_as, H, ID, L.gate.d, L.gate.sc, st));
    AL(gemv_dispatch(d_up, L.is_fp16, L.fp16_up, d_ai, d_as, H, ID, L.up.d, L.up.sc, st));
    AL(blackwell::kernels::apply_swiglu(d_mlp, d_gate, d_up, ID, st));
    AL(blackwell::kernels::quantize_int8(d_mlp_i8, d_mlp_i8s, d_mlp, ID, st));
    AL(gemv_dispatch(d_proj, L.is_fp16, L.fp16_down, d_mlp_i8, d_mlp_i8s, ID, H, L.down.d, L.down.sc, st));
    // Residual 2
    AL(blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_residual, H, st));
    AL(cudaMemcpyAsync(d_residual, d_proj, H*4, cudaMemcpyDeviceToDevice, st));
}

// ── Main ──────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    const char* model_name = (argc > 1) ? argv[1] : "1.7b";
    int max_tokens = (argc > 2) ? atoi(argv[2]) : 100;

    const char* wdir;
    int NL, H, Q, KV, ID, V, nqh, nkv, hd;
    if (strstr(model_name, "8b")) {
        NL=36; H=4096; Q=4096; KV=1024; ID=12288; V=151936; nqh=32; nkv=8; hd=128;
        wdir = "weights_int8_qwen3_8b";
    } else {
        NL=28; H=2048; Q=2048; KV=1024; ID=6144; V=151936; nqh=16; nkv=8; hd=128;
        wdir = "weights_int8_bf16";
    }

    printf("PPL Benchmark: model=%s NL=%d H=%d V=%d\n", model_name, NL, H, V);

    // ── Load tokenizer ──
    blackwell::BpeTokenizer tokenizer;
    if (tokenizer.load("tokenizer_data.bin") != 0) { fprintf(stderr, "FAIL tokenizer\n"); return 1; }
    auto tids = tokenizer.encode(std::string(TEST_TEXT));
    printf("Test text: %zu tokens\n", tids.size());
    int N = max_tokens < (int)tids.size() ? max_tokens : (int)tids.size();
    printf("Evaluating: %d tokens\n", N);

    // ── Load weights ──
    printf("Loading weights...\n");
    cudaStream_t st; cudaStreamCreate(&st);

    // Embed
    Int8W emb = load_int8_w((std::string(wdir)+"/embed_tokens").c_str());
    std::vector<int8_t> h_emb_i8(V * H);
    std::vector<float> h_emb_sc(V * (H / 16));
    {
        char p[512]; snprintf(p,512,"%s/embed_tokens.int8_t",wdir);
        FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"FAIL open %s\n",p);return 1;}
        int hh[5]; fread(hh,4,5,f);
        size_t emb_sz = (size_t)hh[0] * hh[1];
        h_emb_i8.resize(emb_sz); fread(h_emb_i8.data(),1,emb_sz,f); fclose(f);
        snprintf(p,512,"%s/embed_tokens.scale_t",wdir);
        f=fopen(p,"rb"); if(!f){fprintf(stderr,"FAIL open %s\n",p);return 1;}
        fread(hh,4,5,f);
        h_emb_sc.resize((size_t)hh[3]*hh[4]); fread(h_emb_sc.data(),4,(size_t)hh[3]*hh[4],f); fclose(f);
    }
    float* d_fn = load_f32((std::string(wdir)+"/final_norm").c_str(), H);

    // Layer weights
    std::vector<LayerW> layers(NL);
    for (int l = 0; l < NL; l++) {
        char p[256]; auto& L = layers[l];
        // Check for FP16
        snprintf(p,256,"%s/%d_self_attn.q_proj.fp16",wdir,l);
        bool fp16 = (fopen(p,"rb") != nullptr);
        L.is_fp16 = fp16;
        if (fp16) {
            snprintf(p,256,"%s/%d_self_attn.q_proj.fp16",wdir,l); auto w=upload_fp16(p); L.q={w.K,w.N,nullptr,nullptr}; L.fp16_q=w.d;
            snprintf(p,256,"%s/%d_self_attn.k_proj.fp16",wdir,l); w=upload_fp16(p); L.k={w.K,w.N,nullptr,nullptr}; L.fp16_k=w.d;
            snprintf(p,256,"%s/%d_self_attn.v_proj.fp16",wdir,l); w=upload_fp16(p); L.v={w.K,w.N,nullptr,nullptr}; L.fp16_v=w.d;
            snprintf(p,256,"%s/%d_self_attn.o_proj.fp16",wdir,l); w=upload_fp16(p); L.o={w.K,w.N,nullptr,nullptr}; L.fp16_o=w.d;
            snprintf(p,256,"%s/%d_mlp.gate_proj.fp16",wdir,l); w=upload_fp16(p); L.gate={w.K,w.N,nullptr,nullptr}; L.fp16_gate=w.d;
            snprintf(p,256,"%s/%d_mlp.up_proj.fp16",wdir,l); w=upload_fp16(p); L.up={w.K,w.N,nullptr,nullptr}; L.fp16_up=w.d;
            snprintf(p,256,"%s/%d_mlp.down_proj.fp16",wdir,l); w=upload_fp16(p); L.down={w.K,w.N,nullptr,nullptr}; L.fp16_down=w.d;
        } else {
            L.fp16_q=L.fp16_k=L.fp16_v=L.fp16_o=nullptr;
            L.fp16_gate=L.fp16_up=L.fp16_down=nullptr;
            snprintf(p,256,"%s/%d_self_attn.q_proj",wdir,l); L.q = load_int8_w(p);
            snprintf(p,256,"%s/%d_self_attn.k_proj",wdir,l); L.k = load_int8_w(p);
            snprintf(p,256,"%s/%d_self_attn.v_proj",wdir,l); L.v = load_int8_w(p);
            snprintf(p,256,"%s/%d_self_attn.o_proj",wdir,l); L.o = load_int8_w(p);
            snprintf(p,256,"%s/%d_mlp.gate_proj",wdir,l); L.gate = load_int8_w(p);
            snprintf(p,256,"%s/%d_mlp.up_proj",wdir,l);   L.up = load_int8_w(p);
            snprintf(p,256,"%s/%d_mlp.down_proj",wdir,l); L.down = load_int8_w(p);
        }
        snprintf(p,256,"%s/%d_input_layernorm",wdir,l); L.rn_in = load_f32(p, H);
        snprintf(p,256,"%s/%d_post_attention_layernorm",wdir,l); L.rn_post = load_f32(p, H);
        snprintf(p,256,"%s/qk_norms.f32",wdir);
        // qk_norms.f32: NL*2*hd floats, 2*hd per layer (q_norm + k_norm)
        // Expand per-head: q_norm[hd] × nqh heads + k_norm[hd] × nkv heads
        {
            FILE* qf = fopen(p,"rb"); if(!qf){fprintf(stderr,"FAIL open %s\n",p);exit(1);}
            fseek(qf,(long)l*2*hd*4,SEEK_SET);
            std::vector<float> qk_buf(2*hd);
            fread(qk_buf.data(),4,2*hd,qf); fclose(qf);
            int total_qk = nqh*hd + nkv*hd;
            std::vector<float> expanded(total_qk);
            for(int h=0;h<nqh;h++) memcpy(&expanded[h*hd],&qk_buf[0],hd*4);
            for(int h=0;h<nkv;h++) memcpy(&expanded[nqh*hd+h*hd],&qk_buf[hd],hd*4);
            AL(cudaMalloc(&L.qk_n,total_qk*4));
            AL(cudaMemcpy(L.qk_n,expanded.data(),total_qk*4,cudaMemcpyHostToDevice));
        }
    }

    // Optional lm_head
    Int8W lm_head{0,0,nullptr,nullptr};
    char p[256]; snprintf(p,256,"%s/lm_head.int8_t",wdir);
    FILE* f = fopen(p,"rb");
    if (f) { fclose(f); lm_head = load_int8_w((std::string(wdir)+"/lm_head").c_str()); printf("  lm_head: separate\n"); }
    else { printf("  lm_head: tied\n"); }
    printf("  done\n");

    // ── Allocate GPU buffers ──
    float *d_residual, *d_xn, *d_logits, *d_logp;
    int8_t *d_ai; float *d_as;
    float *d_Q, *d_K, *d_V, *d_attn, *d_proj, *d_gate, *d_up, *d_mlp;
    int8_t *d_attn_i8, *d_mlp_i8; float *d_attn_i8s, *d_mlp_i8s;
    float *d_kc, *d_vc;

    AL(cudaMalloc(&d_residual, H*4));
    AL(cudaMalloc(&d_xn, H*4));
    AL(cudaMalloc(&d_logits, V*4));
    AL(cudaMalloc(&d_logp, 4));
    AL(cudaMalloc(&d_ai, H));
    AL(cudaMalloc(&d_as, (H/16)*4));
    AL(cudaMalloc(&d_Q, Q*4));
    AL(cudaMalloc(&d_K, KV*4));
    AL(cudaMalloc(&d_V, KV*4));
    AL(cudaMalloc(&d_attn, Q*4));
    AL(cudaMalloc(&d_proj, H*4));
    AL(cudaMalloc(&d_gate, ID*4));
    AL(cudaMalloc(&d_up, ID*4));
    AL(cudaMalloc(&d_mlp, ID*4));
    AL(cudaMalloc(&d_attn_i8, Q));
    AL(cudaMalloc(&d_attn_i8s, (Q/16)*4));
    AL(cudaMalloc(&d_mlp_i8, ID));
    AL(cudaMalloc(&d_mlp_i8s, (ID/16)*4));

    size_t kv_sz = (size_t)NL * nkv * hd * KV * 4;
    AL(cudaMalloc(&d_kc, kv_sz)); AL(cudaMemset(d_kc, 0, kv_sz));
    AL(cudaMalloc(&d_vc, kv_sz)); AL(cudaMemset(d_vc, 0, kv_sz));

    // ── Run benchmark ──────────────────────────────────────────────
    double total_logp = 0.0;
    int valid_tokens = 0;
    std::vector<float> h_hidden(H);
    float host_logp;

    auto t_start = std::chrono::high_resolution_clock::now();

    // Step 0: embed first token, decode through model, predict token[1]
    {
        uint32_t tok0 = tids[0];
        for (int d = 0; d < H; d++)
            h_hidden[d] = (float)h_emb_i8[tok0 * H + d] * h_emb_sc[tok0 * (H/16) + d/16];
        AL(cudaMemcpy(d_residual, h_hidden.data(), H*4, cudaMemcpyHostToDevice));
        AL(cudaStreamSynchronize(st));
        for (int l = 0; l < NL; l++) {
            decode_step(d_residual, 0, l,
                NL, H, Q, KV, ID, V, nqh, nkv, hd,
                d_ai, d_as,
                d_Q, d_K, d_V, d_attn,
                d_attn_i8, d_attn_i8s, d_proj,
                d_gate, d_up, d_mlp,
                d_mlp_i8, d_mlp_i8s,
                layers[l], d_kc, d_vc, st);
        }
        AL(cudaStreamSynchronize(st));
        // Predict log P(tids[1] | tids[0])
        AL(blackwell::kernels::fused_rmsnorm(d_xn, d_residual, d_fn, H, 1e-6f, st));
        AL(blackwell::kernels::quantize_int8(d_ai, d_as, d_xn, H, st));
        AL(blackwell::kernels::gemv_int8_warp(d_logits, d_ai, d_as,
            (lm_head.d ? lm_head.d : emb.d), (lm_head.sc ? lm_head.sc : emb.sc), H, V, st));
        AL(cudaGetLastError());
        {
            int shmem = sizeof(float) * 256;
            logprob_kernel<<<1, 256, shmem, st>>>(d_logits, V, (int)tids[1], d_logp);
            AL(cudaMemcpy(&host_logp, d_logp, 4, cudaMemcpyDeviceToHost));
            total_logp += (double)host_logp;
            valid_tokens++;
        }
    }

    // Steps 1..N-1: embed token_i, decode, predict token_{i+1}
    for (int step = 1; step + 1 < N; step++) {
        // Embed current token (which the model needs to process)
        uint32_t tok_id = tids[step];
        for (int d = 0; d < H; d++)
            h_hidden[d] = (float)h_emb_i8[tok_id * H + d] * h_emb_sc[tok_id * (H/16) + d/16];
        AL(cudaMemcpy(d_residual, h_hidden.data(), H*4, cudaMemcpyHostToDevice));

        // Run all layers at position = step
        AL(cudaStreamSynchronize(st));
        for (int l = 0; l < NL; l++) {
            decode_step(d_residual, step, l,
                NL, H, Q, KV, ID, V, nqh, nkv, hd,
                d_ai, d_as,
                d_Q, d_K, d_V, d_attn,
                d_attn_i8, d_attn_i8s, d_proj,
                d_gate, d_up, d_mlp,
                d_mlp_i8, d_mlp_i8s,
                layers[l], d_kc, d_vc, st);
        }

        // Predict log P(tids[step+1] | tids[0..step])
        AL(blackwell::kernels::fused_rmsnorm(d_xn, d_residual, d_fn, H, 1e-6f, st));
        AL(blackwell::kernels::quantize_int8(d_ai, d_as, d_xn, H, st));
        AL(blackwell::kernels::gemv_int8_warp(d_logits, d_ai, d_as,
            (lm_head.d ? lm_head.d : emb.d), (lm_head.sc ? lm_head.sc : emb.sc), H, V, st));
        AL(cudaGetLastError());
        {
            int shmem = sizeof(float) * 256;
            logprob_kernel<<<1, 256, shmem, st>>>(d_logits, V, (int)tids[step + 1], d_logp);
            AL(cudaMemcpy(&host_logp, d_logp, 4, cudaMemcpyDeviceToHost));
            total_logp += (double)host_logp;
            valid_tokens++;
        }
    }

    AL(cudaStreamSynchronize(st));
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
    double tps = (double)N / elapsed_s;
    double ppl = exp(-total_logp / (double)valid_tokens);

    printf("\n=== Results ===\n");
    printf("  Tokens:    %d\n", N);
    printf("  Time:      %.3f s\n", elapsed_s);
    printf("  Throughput: %.0f t/s (%.2f ms/tok)\n", tps, 1000.0 / tps);
    printf("  Log P sum: %.4f (over %d tokens)\n", total_logp, valid_tokens);
    printf("  Perplexity: %.2f\n", ppl);
    printf("===============\n");
    return 0;
}
