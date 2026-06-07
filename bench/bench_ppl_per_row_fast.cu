// Single-token PPL test for per-row — quick debug
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include "blackwell/kernels.h"
#define AL(e) do{if((e)!=cudaSuccess){fprintf(stderr,"FAIL %s:%d\n",__FILE__,__LINE__);exit(1);}}while(0)

struct Int8W { int K, N; int8_t* d; float* sc; };
static Int8W load_int8_per_row(const char* prefix) {
    char p[512]; snprintf(p,512,"%s.int8_t",prefix);
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"FAIL open %s\n",p);exit(1);}
    int h[5]; fread(h,4,5,f);
    int K=h[0], N=h[1]; size_t sz=(size_t)K*N;
    std::vector<int8_t> i8(sz); fread(i8.data(),1,sz,f); fclose(f);
    snprintf(p,512,"%s.scale_t",prefix); f=fopen(p,"rb"); if(!f){fprintf(stderr,"FAIL open %s\n",p);exit(1);}
    int hs[5]; fread(hs,4,5,f);
    int nscales=hs[4];
    std::vector<float> sc(nscales); fread(sc.data(),4,nscales,f); fclose(f);
    Int8W w{K,N,nullptr,nullptr};
    AL(cudaMalloc(&w.d,sz)); AL(cudaMemcpy(w.d,i8.data(),sz,cudaMemcpyHostToDevice));
    AL(cudaMalloc(&w.sc,nscales*4)); AL(cudaMemcpy(w.sc,sc.data(),nscales*4,cudaMemcpyHostToDevice));
    return w;
}
static float* load_f32(const char* pfx, int n) {
    char p[512]; snprintf(p,512,"%s.f32",pfx);
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"FAIL open %s\n",p);exit(1);}
    std::vector<float> tmp(n); fread(tmp.data(),4,n,f);fclose(f);
    float* d; AL(cudaMalloc(&d,n*4)); AL(cudaMemcpy(d,tmp.data(),n*4,cudaMemcpyHostToDevice));
    return d;
}

struct LayerW { Int8W q,k,v,o,gate,up,down; float *rn_in,*rn_post,*qk_n; };

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
    int i2=t*2; float* pair = d + h*hd + i2;
    float theta = (float)pos * powf(1000000.0f, -2.0f*(float)t/(float)hd);
    float c = cosf(theta), s = sinf(theta), x = pair[0], y = pair[1];
    pair[0] = x*c - y*s; pair[1] = x*s + y*c;
}
__global__ void logprob_kernel(const float* logits, int V, int correct_id, float* out) {
    extern __shared__ float smem[]; int tid = threadIdx.x, n = blockDim.x;
    float mx = -1e30; for (int i = tid; i < V; i += n) { float v = logits[i]; if (v > mx) mx = v; }
    for (int o = blockDim.x/2; o > 0; o >>= 1) { float v = __shfl_xor_sync(0xffffffff, mx, o); if (v > mx) mx = v; }
    if ((tid & 31) == 0) smem[tid>>5] = mx; __syncthreads();
    if (tid < 32) { mx = (tid<4) ? smem[tid] : -1e30f; for (int o = 2; o > 0; o >>= 1) { float v = __shfl_xor_sync(0xffffffff, mx, o); if (v > mx) mx = v; } if (tid==0) smem[0]=mx; } __syncthreads();
    float maxv = smem[0];
    float sum_exp = 0; for (int i = tid; i < V; i += n) sum_exp += expf(logits[i] - maxv);
    for (int o = blockDim.x/2; o > 0; o >>= 1) sum_exp += __shfl_xor_sync(0xffffffff, sum_exp, o);
    if ((tid & 31) == 0) smem[tid>>5] = sum_exp; __syncthreads();
    if (tid < 32) { float v = (tid<4)?smem[tid]:0; for (int o = 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o); if (tid==0) smem[0]=v; } __syncthreads();
    float logp = (correct_id >= 0 && correct_id < V) ? logits[correct_id] - (maxv + logf(smem[0])) : 0.0f;
    if (tid == 0) out[0] = logp;
}
__global__ void gemv_fp32_k(float* y, const float* x, const float* W, int K, int N) {
    __shared__ float sm[4];
    int row = blockIdx.x; if (row >= N) return;
    const float* wr = W + (size_t)row * K;
    float sum = 0; for (int i = threadIdx.x; i < K; i += blockDim.x) sum += wr[i] * x[i];
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
    int w = threadIdx.x / 32, l = threadIdx.x % 32;
    if (l == 0) sm[w] = sum; __syncthreads();
    if (w == 0) { sum = (l<4)?sm[l]:0; for (int o=2;o>0;o>>=1)sum+=__shfl_xor_sync(0xffffffff,sum,o); if(l==0) y[row]=sum; }
}

int main() {
    const char* wdir = "/mnt/data/dev/projects/blackwell/weights_int8_per_row";
    int NL=28, H=2048, Q=2048, KV=1024, ID=6144, V=151936, nqh=16, nkv=8, hd=128;
    
    // Load embed tokens
    printf("Loading embed...\n"); fflush(stdout);
    char p[512]; snprintf(p,512,"%s/embed_tokens.int8_t",wdir);
    FILE* f=fopen(p,"rb"); int hh[5]; fread(hh,4,5,f);
    size_t emb_sz=(size_t)hh[0]*hh[1]; 
    std::vector<int8_t> h_emb_i8(emb_sz); fread(h_emb_i8.data(),1,emb_sz,f); fclose(f);
    snprintf(p,512,"%s/embed_tokens.scale_t",wdir); f=fopen(p,"rb");
    fread(hh,4,5,f); size_t ns=(size_t)hh[4];
    std::vector<float> h_emb_sc(ns); fread(h_emb_sc.data(),4,ns,f); fclose(f);
    std::vector<float> h_emb((size_t)V*H);
    for (int tok=0; tok<V; tok++)
        for (int d=0; d<H; d++)
            h_emb[tok*H+d] = (float)h_emb_i8[tok*H+d] * h_emb_sc[tok];
    float *d_emb; AL(cudaMalloc(&d_emb,(size_t)V*H*4));
    AL(cudaMemcpy(d_emb,h_emb.data(),(size_t)V*H*4,cudaMemcpyHostToDevice));
    printf("  done\n"); fflush(stdout);
    
    // Load layers
    printf("Loading layers...\n"); fflush(stdout);
    std::vector<LayerW> layers(NL);
    for (int l=0; l<NL; l++) {
        snprintf(p,256,"%s/%d_self_attn.q_proj",wdir,l); layers[l].q=load_int8_per_row(p);
        snprintf(p,256,"%s/%d_self_attn.k_proj",wdir,l); layers[l].k=load_int8_per_row(p);
        snprintf(p,256,"%s/%d_self_attn.v_proj",wdir,l); layers[l].v=load_int8_per_row(p);
        snprintf(p,256,"%s/%d_self_attn.o_proj",wdir,l); layers[l].o=load_int8_per_row(p);
        snprintf(p,256,"%s/%d_mlp.gate_proj",wdir,l); layers[l].gate=load_int8_per_row(p);
        snprintf(p,256,"%s/%d_mlp.up_proj",wdir,l);   layers[l].up=load_int8_per_row(p);
        snprintf(p,256,"%s/%d_mlp.down_proj",wdir,l); layers[l].down=load_int8_per_row(p);
        snprintf(p,256,"%s/%d_input_layernorm",wdir,l); layers[l].rn_in=load_f32(p,H);
        snprintf(p,256,"%s/%d_post_attention_layernorm",wdir,l); layers[l].rn_post=load_f32(p,H);
        snprintf(p,256,"%s/qk_norms.f32",wdir);
        { FILE* qf=fopen(p,"rb"); fseek(qf,(long)l*2*hd*4,SEEK_SET);
          std::vector<float> qk(2*hd); fread(qk.data(),4,2*hd,qf); fclose(qf);
          int tqk=nqh*hd+nkv*hd; std::vector<float> ex(tqk);
          for(int h=0;h<nqh;h++) memcpy(&ex[h*hd],&qk[0],hd*4);
          for(int h=0;h<nkv;h++) memcpy(&ex[nqh*hd+h*hd],&qk[hd],hd*4);
          AL(cudaMalloc(&layers[l].qk_n,tqk*4)); AL(cudaMemcpy(layers[l].qk_n,ex.data(),tqk*4,cudaMemcpyHostToDevice)); }
    }
    float* d_fn = load_f32((std::string(wdir)+"/final_norm").c_str(), H);
    printf("  done\n"); fflush(stdout);
    
    cudaStream_t st; cudaStreamCreate(&st);
    
    // Buffers
    float *d_res,*d_xin,*d_logits,*d_logp,*d_Q,*d_K,*d_V,*d_attn,*d_proj,*d_gate,*d_up,*d_mlp,*d_kc,*d_vc;
    AL(cudaMalloc(&d_res,H*4)); AL(cudaMalloc(&d_xin,H*4)); AL(cudaMalloc(&d_logits,V*4)); AL(cudaMalloc(&d_logp,4));
    AL(cudaMalloc(&d_Q,Q*4)); AL(cudaMalloc(&d_K,KV*4)); AL(cudaMalloc(&d_V,KV*4)); AL(cudaMalloc(&d_attn,Q*4));
    AL(cudaMalloc(&d_proj,H*4)); AL(cudaMalloc(&d_gate,ID*4)); AL(cudaMalloc(&d_up,ID*4)); AL(cudaMalloc(&d_mlp,ID*4));
    size_t kvs=(size_t)NL*nkv*hd*KV*4; AL(cudaMalloc(&d_kc,kvs)); AL(cudaMemset(d_kc,0,kvs));
    AL(cudaMalloc(&d_vc,kvs)); AL(cudaMemset(d_vc,0,kvs));
    
    // Tokens from WikiText-2 (same as original benchmark)
    std::vector<uint32_t> tids = {198, 4820, 18315, 19, 323, 538, 151, 1079, 19, 18, 301, 3403, 2697, 10, 374, 10430, 11, 306, 25229, 10};
    int T = (int)tids.size() - 1;
    
    double total_logp = 0;
    for (int s = 0; s < T; s++) {
        // Embed
        AL(cudaMemcpy(d_res, d_emb + (size_t)tids[s] * H, H * 4, cudaMemcpyDeviceToDevice));
        
        // Decode
        for (int l = 0; l < NL; l++) {
            size_t kv_off = (size_t)l * nkv * hd * KV;
            AL(blackwell::kernels::fused_rmsnorm(d_xin, d_res, layers[l].rn_in, H, 1e-6f, st));
            AL(blackwell::kernels::gemv_int8_per_row(d_Q, d_xin, layers[l].q.d, layers[l].q.sc, H, Q, st));
            AL(blackwell::kernels::gemv_int8_per_row(d_K, d_xin, layers[l].k.d, layers[l].k.sc, H, KV, st));
            AL(blackwell::kernels::gemv_int8_per_row(d_V, d_xin, layers[l].v.d, layers[l].v.sc, H, KV, st));
            hn_kernel<<<nqh,128,0,st>>>(d_Q, layers[l].qk_n, nqh, hd, 1e-6f);
            hn_kernel<<<nkv,128,0,st>>>(d_K, layers[l].qk_n + nqh * hd, nkv, hd, 1e-6f);
            rope_kernel<<<nqh,hd/2,0,st>>>(d_Q, nqh, hd, s);
            rope_kernel<<<nkv,hd/2,0,st>>>(d_K, nkv, hd, s);
            AL(cudaGetLastError());
            AL(blackwell::kernels::update_kv_cache(d_kc+kv_off, d_vc+kv_off, d_K, d_V, 0, s, nkv, hd, KV, st));
            AL(blackwell::kernels::attention_decode_gqa(d_attn, d_Q, d_kc+kv_off, d_vc+kv_off, s, nqh, nkv, hd, KV, st));
            AL(blackwell::kernels::gemv_int8_per_row(d_proj, d_attn, layers[l].o.d, layers[l].o.sc, Q, H, st));
            AL(blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_res, H, st));
            AL(cudaMemcpyAsync(d_res, d_proj, H*4, cudaMemcpyDeviceToDevice, st));
            AL(blackwell::kernels::fused_rmsnorm(d_xin, d_proj, layers[l].rn_post, H, 1e-6f, st));
            AL(blackwell::kernels::gemv_int8_per_row(d_gate, d_xin, layers[l].gate.d, layers[l].gate.sc, H, ID, st));
            AL(blackwell::kernels::gemv_int8_per_row(d_up, d_xin, layers[l].up.d, layers[l].up.sc, H, ID, st));
            AL(blackwell::kernels::apply_swiglu(d_mlp, d_gate, d_up, ID, st));
            AL(blackwell::kernels::gemv_int8_per_row(d_proj, d_mlp, layers[l].down.d, layers[l].down.sc, ID, H, st));
            AL(blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_res, H, st));
            AL(cudaMemcpyAsync(d_res, d_proj, H*4, cudaMemcpyDeviceToDevice, st));
        }
        
        // Final norm + lm_head
        AL(blackwell::kernels::fused_rmsnorm(d_xin, d_res, d_fn, H, 1e-6f, st));
        gemv_fp32_k<<<V,128,0,st>>>(d_logits, d_xin, d_emb, H, V);
        logprob_kernel<<<1,256,sizeof(float)*256,st>>>(d_logits, V, (int)tids[s+1], d_logp);
        AL(cudaMemcpy(&total_logp, d_logp, 4, cudaMemcpyDeviceToHost));
        
        // Debug: print logits for first token
        if (s == 0) {
            float dbg[10]; cudaMemcpy(dbg, d_logits, 40, cudaMemcpyDeviceToHost);
            fprintf(stderr, "Token %d logits[0..9]: ", s);
            for (int i = 0; i < 10; i++) fprintf(stderr, "%f ", dbg[i]);
            float mn=1e10,mx=-1e10;
            for (int i=0; i<10; i++){if(dbg[i]<mn)mn=dbg[i];if(dbg[i]>mx)mx=dbg[i];}
            fprintf(stderr, "mn=%f mx=%f\n", mn, mx);
            fprintf(stderr, "Correct token: %d, logp=%f\n", tids[s+1], total_logp);
        }
    }
    AL(cudaStreamSynchronize(st));
    double ppl = exp(-total_logp/T);
    printf("PPL: %.2f (logP=%.4f, tokens=%d)\n", ppl, total_logp, T);
    printf("vs BF16 baseline: %.2f\n", 12.4);
    printf("vs INT8 block-16: %.2f\n", 18.82);
    return 0;
}
