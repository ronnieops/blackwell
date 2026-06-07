// bench_ppl_int8_fp32act.cu — INT8 weights + FP32 activations PPL benchmark
// Tests: does removing activation quantization fix quality?
// Loads INT8 weights → dequant to FP32 → FP32 GEMV (no activation quantization)

#include <cuda_runtime.h>
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

static const char* TEST_TEXT =
    "The Republic of Austria is a federal republic in Central Europe . "
    "It is bordered by Germany to the northwest , the Czech Republic to the north , "
    "Slovakia to the northeast , Hungary to the east , Slovenia and Italy to the south , "
    "Switzerland and Liechtenstein to the west . "
    "The capital of Austria is Vienna . "
    "The official language is German .";

__global__ void logprob_kernel(const float* logits, int V, int correct_id, float* out) {
    extern __shared__ float smem[];
    int tid = threadIdx.x; int n = blockDim.x;
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

__global__ void gemv_fp32_kernel(float* y, const float* x, const float* W, int K, int N) {
    __shared__ float smem[4];
    int row = blockIdx.x; if (row >= N) return;
    const float* w_row = W + (size_t)row * K;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) sum += w_row[i] * x[i];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
    int warp_id = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) smem[warp_id] = sum; __syncthreads();
    if (warp_id == 0) {
        sum = (lane < 4) ? smem[lane] : 0.0f;
        for (int o = 2; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
        if (lane == 0) y[row] = sum;
    }
}
static void gemv_fp32(float* y, const float* x, const float* W, int K, int N, cudaStream_t st) {
    gemv_fp32_kernel<<<N, 128, 0, st>>>(y, x, W, K, N);
}

struct Fp32W { int K, N; float* d; };

static Fp32W load_int8_as_fp32(const char* prefix) {
    char p[512]; snprintf(p, 512, "%s.int8_t", prefix);
    FILE* f = fopen(p, "rb"); if (!f) { fprintf(stderr, "FAIL open %s\n", p); exit(1); }
    int h[5]; fread(h, 4, 5, f);
    int K = h[0], N = h[1]; size_t sz = (size_t)K * N;
    std::vector<int8_t> i8_data(sz); fread(i8_data.data(), 1, sz, f); fclose(f);

    // Load scales
    snprintf(p, 512, "%s.scale_t", prefix);
    f = fopen(p, "rb"); if (!f) { fprintf(stderr, "FAIL open %s\n", p); exit(1); }
    int hs[5]; fread(hs, 4, 5, f);
    int block = hs[2], nblks = hs[3];
    size_t nscales = (size_t)nblks * hs[4];
    std::vector<float> scales(nscales); fread(scales.data(), 4, nscales, f); fclose(f);

    // Dequant INT8 → FP32 with block scales
    std::vector<float> fp32(sz);
    for (int row = 0; row < N; row++) {
        for (int b = 0; b < nblks; b++) {
            float sc = scales[row * nblks + b];
            for (int j = 0; j < block; j++) {
                fp32[row * K + b * block + j] = (float)i8_data[row * K + b * block + j] * sc;
            }
        }
    }

    Fp32W w{K, N, nullptr};
    AL(cudaMalloc(&w.d, sz * 4)); AL(cudaMemcpy(w.d, fp32.data(), sz * 4, cudaMemcpyHostToDevice));
    return w;
}

static float* load_f32(const char* pfx, int n) {
    char p[512]; snprintf(p, 512, "%s.f32", pfx);
    FILE* f = fopen(p, "rb"); if (!f) { fprintf(stderr, "FAIL open %s\n", p); exit(1); }
    std::vector<float> tmp(n); fread(tmp.data(), 4, n, f); fclose(f);
    float* d; AL(cudaMalloc(&d, n * 4)); AL(cudaMemcpy(d, tmp.data(), n * 4, cudaMemcpyHostToDevice));
    return d;
}

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

struct LayerW {
    Fp32W q, k, v, o, gate, up, down;
    float *rn_in, *rn_post, *qk_n;
};

static void decode_step_fp32(float* d_residual, int seq_pos, int l,
    int H, int Q, int KV, int ID, int V, int nqh, int nkv, int hd,
    float* d_x_in, float* d_Q, float* d_K, float* d_V,
    float* d_attn, float* d_proj, float* d_gate, float* d_up, float* d_mlp,
    const LayerW& L, float* d_kc, float* d_vc, cudaStream_t st) {
    size_t kv_off = (size_t)l * nkv * hd * KV;
    AL(blackwell::kernels::fused_rmsnorm(d_x_in, d_residual, L.rn_in, H, 1e-6f, st));
    gemv_fp32(d_Q, d_x_in, L.q.d, H, Q, st);
    gemv_fp32(d_K, d_x_in, L.k.d, H, KV, st);
    gemv_fp32(d_V, d_x_in, L.v.d, H, KV, st);
    hn_kernel<<<nqh, 128, 0, st>>>(d_Q, L.qk_n, nqh, hd, 1e-6f);
    hn_kernel<<<nkv, 128, 0, st>>>(d_K, L.qk_n + nqh * hd, nkv, hd, 1e-6f);
    rope_kernel<<<nqh, hd/2, 0, st>>>(d_Q, nqh, hd, seq_pos);
    rope_kernel<<<nkv, hd/2, 0, st>>>(d_K, nkv, hd, seq_pos);
    AL(cudaGetLastError());
    AL(blackwell::kernels::update_kv_cache(d_kc + kv_off, d_vc + kv_off, d_K, d_V, 0, seq_pos, nkv, hd, KV, st));
    AL(blackwell::kernels::attention_decode_gqa(d_attn, d_Q, d_kc + kv_off, d_vc + kv_off, seq_pos, nqh, nkv, hd, KV, st));
    gemv_fp32(d_proj, d_attn, L.o.d, Q, H, st);
    AL(blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_residual, H, st));
    AL(cudaMemcpyAsync(d_residual, d_proj, H * 4, cudaMemcpyDeviceToDevice, st));
    AL(blackwell::kernels::fused_rmsnorm(d_x_in, d_proj, L.rn_post, H, 1e-6f, st));
    gemv_fp32(d_gate, d_x_in, L.gate.d, H, ID, st);
    gemv_fp32(d_up, d_x_in, L.up.d, H, ID, st);
    AL(blackwell::kernels::apply_swiglu(d_mlp, d_gate, d_up, ID, st));
    gemv_fp32(d_proj, d_mlp, L.down.d, ID, H, st);
    AL(blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_residual, H, st));
    AL(cudaMemcpyAsync(d_residual, d_proj, H * 4, cudaMemcpyDeviceToDevice, st));
}

int main(int argc, char** argv) {
    int max_tokens = (argc > 1) ? atoi(argv[1]) : 20;
    const char* wdir = "weights_int8_bf16";
    int NL=28, H=2048, Q=2048, KV=1024, ID=6144, V=151936, nqh=16, nkv=8, hd=128;
    printf("INT8 weight-only + FP32 activation PPL Benchmark\n");

    blackwell::BpeTokenizer tok;
    if (tok.load("tokenizer_data.bin") != 0) { fprintf(stderr, "FAIL tokenizer\n"); return 1; }
    auto tids = tok.encode(std::string(TEST_TEXT));
    int N = max_tokens < (int)tids.size() ? max_tokens : (int)tids.size();
    printf("Tokens: %d\n", N);

    printf("Loading INT8 weights (dequant to FP32)...\n");
    cudaStream_t st; cudaStreamCreate(&st);

    // Load embed on host (matching bench_ppl.cu format: tok*H + d)
    std::vector<int8_t> h_emb_i8((size_t)V * H);
    std::vector<float> h_emb_sc((size_t)V * (H / 16));
    {
        char p[512]; snprintf(p,512,"%s/embed_tokens.int8_t",wdir);
        FILE* f=fopen(p,"rb"); int hh[5]; fread(hh,4,5,f);
        size_t emb_sz=(size_t)hh[0]*hh[1]; h_emb_i8.resize(emb_sz);
        fread(h_emb_i8.data(),1,emb_sz,f); fclose(f);
        snprintf(p,512,"%s/embed_tokens.scale_t",wdir); f=fopen(p,"rb");
        fread(hh,4,5,f); size_t ns=(size_t)hh[3]*hh[4]; h_emb_sc.resize(ns);
        fread(h_emb_sc.data(),4,ns,f); fclose(f);
    }
    // Pre-compute FP32 embed for fast host lookup
    std::vector<float> h_emb((size_t)V * H);
    for (int tok=0; tok<V; tok++)
        for (int d=0; d<H; d++)
            h_emb[tok*H+d] = (float)h_emb_i8[tok*H+d] * h_emb_sc[tok*(H/16)+d/16];
    h_emb_i8.clear(); h_emb_sc.clear();
    // GPU embed for lm_head (tied)
    Fp32W emb; emb.K=H; emb.N=V; AL(cudaMalloc(&emb.d,(size_t)V*H*4));
    AL(cudaMemcpy(emb.d,h_emb.data(),(size_t)V*H*4,cudaMemcpyHostToDevice));
    float* d_fn = load_f32((std::string(wdir)+"/final_norm").c_str(), H);

    std::vector<LayerW> layers(NL);
    for (int l=0;l<NL;l++) {
        char p[256]; auto& L=layers[l];
        snprintf(p,256,"%s/%d_self_attn.q_proj",wdir,l); L.q=load_int8_as_fp32(p);
        snprintf(p,256,"%s/%d_self_attn.k_proj",wdir,l); L.k=load_int8_as_fp32(p);
        snprintf(p,256,"%s/%d_self_attn.v_proj",wdir,l); L.v=load_int8_as_fp32(p);
        snprintf(p,256,"%s/%d_self_attn.o_proj",wdir,l); L.o=load_int8_as_fp32(p);
        snprintf(p,256,"%s/%d_mlp.gate_proj",wdir,l); L.gate=load_int8_as_fp32(p);
        snprintf(p,256,"%s/%d_mlp.up_proj",wdir,l);   L.up=load_int8_as_fp32(p);
        snprintf(p,256,"%s/%d_mlp.down_proj",wdir,l); L.down=load_int8_as_fp32(p);
        snprintf(p,256,"%s/%d_input_layernorm",wdir,l); L.rn_in=load_f32(p,H);
        snprintf(p,256,"%s/%d_post_attention_layernorm",wdir,l); L.rn_post=load_f32(p,H);
        snprintf(p,256,"%s/qk_norms.f32",wdir);
        { FILE* qf=fopen(p,"rb"); fseek(qf,(long)l*2*hd*4,SEEK_SET);
          std::vector<float> qk(2*hd); fread(qk.data(),4,2*hd,qf); fclose(qf);
          int tqk=nqh*hd+nkv*hd; std::vector<float> ex(tqk);
          for(int h=0;h<nqh;h++) memcpy(&ex[h*hd],&qk[0],hd*4);
          for(int h=0;h<nkv;h++) memcpy(&ex[nqh*hd+h*hd],&qk[hd],hd*4);
          AL(cudaMalloc(&L.qk_n,tqk*4)); AL(cudaMemcpy(L.qk_n,ex.data(),tqk*4,cudaMemcpyHostToDevice)); }
    }
    printf("  done\n");

    float *d_res,*d_xin,*d_logits,*d_logp,*d_Q,*d_K,*d_V,*d_attn,*d_proj,*d_gate,*d_up,*d_mlp,*d_kc,*d_vc;
    AL(cudaMalloc(&d_res,H*4)); AL(cudaMalloc(&d_xin,H*4)); AL(cudaMalloc(&d_logits,V*4)); AL(cudaMalloc(&d_logp,4));
    AL(cudaMalloc(&d_Q,Q*4)); AL(cudaMalloc(&d_K,KV*4)); AL(cudaMalloc(&d_V,KV*4)); AL(cudaMalloc(&d_attn,Q*4));
    AL(cudaMalloc(&d_proj,H*4)); AL(cudaMalloc(&d_gate,ID*4)); AL(cudaMalloc(&d_up,ID*4)); AL(cudaMalloc(&d_mlp,ID*4));
    size_t kvs=(size_t)NL*nkv*hd*KV*4; AL(cudaMalloc(&d_kc,kvs)); AL(cudaMemset(d_kc,0,kvs));
    AL(cudaMalloc(&d_vc,kvs)); AL(cudaMemset(d_vc,0,kvs));

    double total_logp=0; int vt=0; float hlp;
    std::vector<float> hh(H);
    auto t0=std::chrono::high_resolution_clock::now();
    for(int s=0;s+1<N;s++){
        memcpy(hh.data(),&h_emb[tids[s]*H],H*4);
        AL(cudaMemcpy(d_res,hh.data(),H*4,cudaMemcpyHostToDevice));
        AL(cudaStreamSynchronize(st));
        for(int l=0;l<NL;l++) decode_step_fp32(d_res,s,l,H,Q,KV,ID,V,nqh,nkv,hd,d_xin,d_Q,d_K,d_V,d_attn,d_proj,d_gate,d_up,d_mlp,layers[l],d_kc,d_vc,st);
        AL(blackwell::kernels::fused_rmsnorm(d_xin,d_res,d_fn,H,1e-6f,st));
        gemv_fp32(d_logits,d_xin,emb.d,H,V,st); AL(cudaGetLastError());
        logprob_kernel<<<1,256,sizeof(float)*256,st>>>(d_logits,V,(int)tids[s+1],d_logp);
        AL(cudaMemcpy(&hlp,d_logp,4,cudaMemcpyDeviceToHost));
        total_logp+=(double)hlp; vt++;
    }
    AL(cudaStreamSynchronize(st));
    auto t1=std::chrono::high_resolution_clock::now();
    double el=std::chrono::duration<double>(t1-t0).count();
    printf("\n=== INT8 weight-only + FP32 activation ===\n");
    printf("  Tokens: %d  Time: %.3fs  %.0f t/s\n", vt, el, vt/el);
    printf("  Log P sum: %.4f\n", total_logp);
    printf("  Perplexity: %.2f\n", exp(-total_logp/vt));
    printf("  (INT8 weight+act: PPL=7,351,868)\n");
    printf("  (BF16 reference: PPL=12.4)\n");
    return 0;
}
