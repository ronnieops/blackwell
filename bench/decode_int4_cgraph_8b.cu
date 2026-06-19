#include "blackwell/int4_weights.h"
using namespace blackwell::weights;
// bench/decode_int4_cgraph_8b.cu — CUDA Graph for INT4 8B
// Captures full 36-layer decode loop with graph-safe KV cache + attention + RoPE.
// All seq_pos reads from device memory (no H2D memcpy in capture).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <algorithm>
#include <chrono>
#include "blackwell/kernels.h"
#include "blackwell/bpe_tokenizer.h"

static void die(cudaError_t e, const char* m) {
    if(e!=cudaSuccess){fprintf(stderr,"FAIL %s: %s\n",m,cudaGetErrorString(e));exit(1);}
}

const int H=4096, Q=4096, KV=1024, I=12288;
const int nqh=32, nkv=8, hd=128, MAXSEQ=512;
const float eps=1e-6f;
const int V=151936;
const int NL=36;
const float rope_theta=1000000.0f;

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

struct LW4 { DevW4f16 q,k,v,o,g,u,d; float* qn,*kn,*rn_in,*rn_post; };

// Head RMSNorm kernel (graph-safe, no host params that change per-step)
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

// Pre-compute cos/sin RoPE cache for all positions up to MAXSEQ
static void build_rope_cache(float* cos_cache, float* sin_cache, int max_seq, int head_dim) {
    int pairs = head_dim / 2;
    for (int pos = 0; pos < max_seq; ++pos) {
        for (int d = 0; d < pairs; ++d) {
            float theta = (float)pos * powf(rope_theta, -2.0f * (float)d / (float)head_dim);
            cos_cache[pos * pairs + d] = cosf(theta);
            sin_cache[pos * pairs + d] = sinf(theta);
        }
    }
}

int main(int argc, char** argv) {
    int num_tokens = argc > 1 ? atoi(argv[1]) : 100;

    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    fprintf(stderr,"# INT4 8B CUDA Graph (fully graph-safe) — %s\n", P.name);

    blackwell::BpeTokenizer tok;
    if(tok.load("tokenizer_data.bin")!=0){ fprintf(stderr,"FAIL tokenizer\n"); return 1; }
    auto ids = tok.encode("The capital of France is");
    fprintf(stderr,"Prompt: %zu tokens\n", ids.size());

    float *d_x32, *d_xi_f, *d_res;
    uint8_t *d_x_i4; float *d_x_i4_sc;
    float *d_Q, *d_K, *d_V, *d_attn;
    uint8_t *d_attn_i4; float *d_attn_i4_sc;
    float *d_proj, *d_gate, *d_up;
    uint8_t *d_mlp_i4; float *d_mlp_i4_sc;
    float *d_fn, *d_kc, *d_vc, *d_logits;
    int *d_next_id;

    #define AL(p,n) die(cudaMalloc(&(p),(n)),"malloc "#p)
    AL(d_x32,H*4); AL(d_xi_f,H*4); AL(d_res,H*4);
    AL(d_x_i4,H/2); AL(d_x_i4_sc,(H/16)*4);
    AL(d_Q,Q*4); AL(d_K,KV*4); AL(d_V,KV*4); AL(d_attn,Q*4);
    AL(d_attn_i4,Q/2); AL(d_attn_i4_sc,(Q/16)*4);
    AL(d_proj,H*4); AL(d_gate,I*4); AL(d_up,I*4);
    AL(d_mlp_i4,I/2); AL(d_mlp_i4_sc,(I/16)*4);
    AL(d_fn,H*4);
    AL(d_kc,(size_t)NL*nkv*MAXSEQ*hd*4);
    AL(d_vc,(size_t)NL*nkv*MAXSEQ*hd*4);
    AL(d_logits,V*4); AL(d_next_id,4);

    // Device-side seq_pos for CUDA Graph (updated via cudaMemcpyAsync from pinned)
    int* d_seq_pos;
    int* h_seq_pos_pinned;
    cudaMalloc(&d_seq_pos, sizeof(int));
    cudaHostAlloc(&h_seq_pos_pinned, sizeof(int), cudaHostAllocDefault);

    // RoPE cos/sin caches (pre-computed on host, uploaded to device)
    int rope_pairs = hd / 2;
    float* d_cos_cache;
    float* d_sin_cache;
    AL(d_cos_cache, (size_t)MAXSEQ * rope_pairs * 4);
    AL(d_sin_cache, (size_t)MAXSEQ * rope_pairs * 4);
    #undef AL
    {
        std::vector<float> cos_h(MAXSEQ * rope_pairs);
        std::vector<float> sin_h(MAXSEQ * rope_pairs);
        build_rope_cache(cos_h.data(), sin_h.data(), MAXSEQ, hd);
        cudaMemcpy(d_cos_cache, cos_h.data(), (size_t)MAXSEQ * rope_pairs * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_sin_cache, sin_h.data(), (size_t)MAXSEQ * rope_pairs * 4, cudaMemcpyHostToDevice);
    }

    // Initialize quantize scales (block-16, all values ~1)
    float iv7=1.f/7.f;
    { std::vector<float> tmp(H/16,iv7); cudaMemcpy(d_x_i4_sc,tmp.data(),(H/16)*4,cudaMemcpyHostToDevice); }
    { std::vector<float> tmp(Q/16,iv7); cudaMemcpy(d_attn_i4_sc,tmp.data(),(Q/16)*4,cudaMemcpyHostToDevice); }
    { std::vector<float> tmp(I/16,iv7); cudaMemcpy(d_mlp_i4_sc,tmp.data(),(I/16)*4,cudaMemcpyHostToDevice); }

    fprintf(stderr,"Loading weights...\n");
    std::vector<LW4> W(NL);
    char p[256];
    for(int l=0;l<NL;++l){
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_self_attn.q_proj",l); W[l].q=upload_w4_f16sc(p);
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_self_attn.k_proj",l); W[l].k=upload_w4_f16sc(p);
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_self_attn.v_proj",l); W[l].v=upload_w4_f16sc(p);
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_self_attn.o_proj",l); W[l].o=upload_w4_f16sc(p);
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_mlp.gate_proj",l); W[l].g=upload_w4_f16sc(p);
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_mlp.up_proj",l); W[l].u=upload_w4_f16sc(p);
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_mlp.down_proj",l); W[l].d=upload_w4_f16sc(p);
    }
    float* qk_h=(float*)malloc(NL*2*hd*4);
    {FILE*f=fopen("weights_int4_qwen3_8b_fp16sc/qk_norms.f32","rb");(void)fread(qk_h,4,NL*2*hd,f);fclose(f);}
    for(int l=0;l<NL;++l){
        cudaMalloc(&W[l].qn,hd*4);cudaMemcpy(W[l].qn,qk_h+l*2*hd,hd*4,cudaMemcpyHostToDevice);
        cudaMalloc(&W[l].kn,hd*4);cudaMemcpy(W[l].kn,qk_h+l*2*hd+hd,hd*4,cudaMemcpyHostToDevice);
    }free(qk_h);
    for(int l=0;l<NL;++l){
        float* w=(float*)malloc(H*4);
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_input_layernorm.f32",l);
        {FILE*f=fopen(p,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&W[l].rn_in,H*4);cudaMemcpy(W[l].rn_in,w,H*4,cudaMemcpyHostToDevice);
        snprintf(p,256,"weights_int4_qwen3_8b_fp16sc/%d_post_attention_layernorm.f32",l);
        {FILE*f=fopen(p,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&W[l].rn_post,H*4);cudaMemcpy(W[l].rn_post,w,H*4,cudaMemcpyHostToDevice);
        free(w);
    }
    {float*w=(float*)malloc(H*4);
    FILE*f=fopen("weights_int4_qwen3_8b_fp16sc/final_norm.f32","rb");(void)fread(w,4,H,f);fclose(f);
    cudaMemcpy(d_fn,w,H*4,cudaMemcpyHostToDevice);free(w);}

    DevW4f16 lm_head_w=upload_w4_f16sc("weights_int4_qwen3_8b_fp16sc/lm_head");
    uint8_t* host_embed_d=new uint8_t[(size_t)H*V/2];
    float* host_embed_sc=new float[V*(H/16)];
    {FILE*f=fopen("weights_int4_qwen3_8b_fp16sc/embed_tokens.int4_t","rb");int h[5];fread(h,4,5,f);
     fread(host_embed_d,1,(size_t)h[0]*h[1]/2,f);fclose(f);
     f=fopen("weights_int4_qwen3_8b_fp16sc/embed_tokens.scale_t","rb");fread(h,4,5,f);
     {__half* tmp=new __half[(size_t)h[3]*h[4]];fread(tmp,2,(size_t)h[3]*h[4],f);
      for(size_t i=0;i<(size_t)h[3]*h[4];++i) host_embed_sc[i]=__half2float(tmp[i]);
      delete[] tmp;}
     fclose(f);}

    cudaStream_t st; die(cudaStreamCreate(&st),"stream");
    std::vector<float> h_embed(H);

    // ── Per-kernel baseline (with H2D memcpy for seq_pos) ──────────────
    fprintf(stderr, "\n── Per-kernel baseline (%d tokens) ──\n", num_tokens);
    
    // Reset state
    dequant_embed_row(h_embed.data(), ids[0], host_embed_d, host_embed_sc, H);
    cudaMemcpyAsync(d_x32, h_embed.data(), H*4, cudaMemcpyHostToDevice, st);
    cudaMemset(d_kc, 0, (size_t)NL*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vc, 0, (size_t)NL*nkv*MAXSEQ*hd*4);

    auto t0 = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < num_tokens; ++step) {
        for(int l=0;l<NL;++l){
            size_t kv_off = (size_t)l*nkv*MAXSEQ*hd;
            cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st);
            blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_in,H,eps,st);
            blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st);
            blackwell::kernels::gemv_int4_warp_f16wsc(d_Q,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].q.d,W[l].q.sc16,H,Q,st);
            blackwell::kernels::gemv_int4_warp_f16wsc(d_K,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].k.d,W[l].k.sc16,H,KV,st);
            blackwell::kernels::gemv_int4_warp_f16wsc(d_V,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].v.d,W[l].v.sc16,H,KV,st);
            head_norm_kernel<<<nqh,128,0,st>>>(d_Q,W[l].qn,nqh,hd,eps);
            head_norm_kernel<<<nkv,128,0,st>>>(d_K,W[l].kn,nkv,hd,eps);
            // Use fused_rope_decode with device seq_pos (graph-safe kernel, but we do H2D update here)
            *h_seq_pos_pinned = step;
            cudaMemcpyAsync(d_seq_pos, h_seq_pos_pinned, sizeof(int), cudaMemcpyHostToDevice, st);
            blackwell::kernels::fused_rope_decode(d_Q, d_cos_cache, d_sin_cache, d_seq_pos, nqh, hd, MAXSEQ, st);
            blackwell::kernels::fused_rope_decode(d_K, d_cos_cache, d_sin_cache, d_seq_pos, nkv, hd, MAXSEQ, st);
            blackwell::kernels::update_kv_cache_device(d_kc+kv_off,d_vc+kv_off,d_K,d_V,0,d_seq_pos,nkv,hd,MAXSEQ,st);
            blackwell::kernels::attention_decode_batched_gqa_device(d_attn,d_Q,d_kc,d_vc,d_seq_pos,nqh,nkv,hd,MAXSEQ,1,
                (size_t)NL*nkv*MAXSEQ*hd,kv_off,st);
            blackwell::kernels::quantize_int4(d_attn_i4,d_attn_i4_sc,d_attn,Q,st);
            blackwell::kernels::gemv_int4_warp_f16wsc(d_proj,(const uint8_t*)d_attn_i4,d_attn_i4_sc,W[l].o.d,W[l].o.sc16,Q,H,st);
            blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,st);
            cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st);
            blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_post,H,eps,st);
            blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st);
            blackwell::kernels::gemv_int4_warp_f16wsc(d_gate,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].g.d,W[l].g.sc16,H,I,st);
            blackwell::kernels::gemv_int4_warp_f16wsc(d_up,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].u.d,W[l].u.sc16,H,I,st);
            blackwell::kernels::apply_swiglu(d_gate,d_gate,d_up,I,st);
            blackwell::kernels::quantize_int4(d_mlp_i4,d_mlp_i4_sc,d_gate,I,st);
            blackwell::kernels::gemv_int4_warp_f16wsc(d_proj,(const uint8_t*)d_mlp_i4,d_mlp_i4_sc,W[l].d.d,W[l].d.sc16,I,H,st);
            blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,st);
        }
        blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,d_fn,H,eps,st);
        blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st);
        blackwell::kernels::gemv_int4_warp_f16wsc(d_logits,(const uint8_t*)d_x_i4,d_x_i4_sc,lm_head_w.d,lm_head_w.sc16,H,V,st);
        blackwell::kernels::sample_gpu(d_logits,V,0,0,d_next_id,0xdeadbeefLL,step,st);
        cudaStreamSynchronize(st);
        
        // Embed next token
        int next_id;
        cudaMemcpy(&next_id, d_next_id, 4, cudaMemcpyDeviceToHost);
        if (step == 0) fprintf(stderr, "  step0 token: %d\n", next_id);
        if (step < num_tokens - 1) {
            dequant_embed_row(h_embed.data(), next_id, host_embed_d, host_embed_sc, H);
            cudaMemcpyAsync(d_x32, h_embed.data(), H*4, cudaMemcpyHostToDevice, st);
        }
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_per = std::chrono::duration<double,std::milli>(t1-t0).count();
    fprintf(stderr, "  Per-kernel: %.1f ms/token = %.0f t/s\n", ms_per/num_tokens, 1000.0/(ms_per/num_tokens));

    // ── CUDA Graph capture (FULLY graph-safe) ───────────────────────────
    // All operations use device-side seq_pos. No H2D memcpy inside capture.
    // Between replays: write pinned seq_pos, cudaMemcpyAsync to d_seq_pos.
    fprintf(stderr, "\n── CUDA Graph capture (%d layers) ──\n", NL);
    
    // Reset state
    dequant_embed_row(h_embed.data(), ids[0], host_embed_d, host_embed_sc, H);
    cudaMemcpyAsync(d_x32, h_embed.data(), H*4, cudaMemcpyHostToDevice, st);
    cudaMemset(d_kc, 0, (size_t)NL*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vc, 0, (size_t)NL*nkv*MAXSEQ*hd*4);
    cudaStreamSynchronize(st);

    cudaStream_t graph_stream;
    die(cudaStreamCreate(&graph_stream), "graph_stream");

    // Set initial seq_pos for capture
    int capture_pos = 0;
    cudaMemcpy(d_seq_pos, &capture_pos, sizeof(int), cudaMemcpyHostToDevice);

    cudaStreamBeginCapture(graph_stream, cudaStreamCaptureModeGlobal);
    
    for (int l = 0; l < NL; ++l) {
        size_t kv_off = (size_t)l * nkv * MAXSEQ * hd;
        
        // Residual copy (D2D, graph-safe)
        cudaMemcpyAsync(d_res, d_x32, H*4, cudaMemcpyDeviceToDevice, graph_stream);
        // Input layernorm + quantize + QKV
        blackwell::kernels::fused_rmsnorm(d_xi_f, d_x32, W[l].rn_in, H, eps, graph_stream);
        blackwell::kernels::quantize_int4(d_x_i4, d_x_i4_sc, d_xi_f, H, graph_stream);
        blackwell::kernels::gemv_int4_warp_f16wsc(d_Q, (const uint8_t*)d_x_i4, d_x_i4_sc, W[l].q.d, W[l].q.sc16, H, Q, graph_stream);
        blackwell::kernels::gemv_int4_warp_f16wsc(d_K, (const uint8_t*)d_x_i4, d_x_i4_sc, W[l].k.d, W[l].k.sc16, H, KV, graph_stream);
        blackwell::kernels::gemv_int4_warp_f16wsc(d_V, (const uint8_t*)d_x_i4, d_x_i4_sc, W[l].v.d, W[l].v.sc16, H, KV, graph_stream);
        // Head norms (pure GPU, graph-safe)
        head_norm_kernel<<<nqh, 128, 0, graph_stream>>>(d_Q, W[l].qn, nqh, hd, eps);
        head_norm_kernel<<<nkv, 128, 0, graph_stream>>>(d_K, W[l].kn, nkv, hd, eps);
        // RoPE via fused_rope_decode (reads d_seq_pos from device, graph-safe)
        blackwell::kernels::fused_rope_decode(d_Q, d_cos_cache, d_sin_cache, d_seq_pos, nqh, hd, MAXSEQ, graph_stream);
        blackwell::kernels::fused_rope_decode(d_K, d_cos_cache, d_sin_cache, d_seq_pos, nkv, hd, MAXSEQ, graph_stream);
        // KV cache write (graph-safe, uses device-side seq_pos)
        blackwell::kernels::update_kv_cache_device(
            d_kc + kv_off, d_vc + kv_off, d_K, d_V, 0, d_seq_pos,
            nkv, hd, MAXSEQ, graph_stream);
        // Attention (graph-safe, uses device-side seq_pos)
        blackwell::kernels::attention_decode_batched_gqa_device(
            d_attn, d_Q, d_kc, d_vc, d_seq_pos, nqh, nkv, hd, MAXSEQ, 1,
            (size_t)NL * nkv * MAXSEQ * hd, kv_off, graph_stream);
        // O projection + residual
        blackwell::kernels::quantize_int4(d_attn_i4, d_attn_i4_sc, d_attn, Q, graph_stream);
        blackwell::kernels::gemv_int4_warp_f16wsc(d_proj, (const uint8_t*)d_attn_i4, d_attn_i4_sc, W[l].o.d, W[l].o.sc16, Q, H, graph_stream);
        blackwell::kernels::vector_add_fp32(d_x32, d_proj, d_res, H, graph_stream);
        // MLP: residual + layernorm + quantize + gate/up/swiglu/down
        cudaMemcpyAsync(d_res, d_x32, H*4, cudaMemcpyDeviceToDevice, graph_stream);
        blackwell::kernels::fused_rmsnorm(d_xi_f, d_x32, W[l].rn_post, H, eps, graph_stream);
        blackwell::kernels::quantize_int4(d_x_i4, d_x_i4_sc, d_xi_f, H, graph_stream);
        blackwell::kernels::gemv_int4_warp_f16wsc(d_gate, (const uint8_t*)d_x_i4, d_x_i4_sc, W[l].g.d, W[l].g.sc16, H, I, graph_stream);
        blackwell::kernels::gemv_int4_warp_f16wsc(d_up, (const uint8_t*)d_x_i4, d_x_i4_sc, W[l].u.d, W[l].u.sc16, H, I, graph_stream);
        blackwell::kernels::apply_swiglu(d_gate, d_gate, d_up, I, graph_stream);
        blackwell::kernels::quantize_int4(d_mlp_i4, d_mlp_i4_sc, d_gate, I, graph_stream);
        blackwell::kernels::gemv_int4_warp_f16wsc(d_proj, (const uint8_t*)d_mlp_i4, d_mlp_i4_sc, W[l].d.d, W[l].d.sc16, I, H, graph_stream);
        blackwell::kernels::vector_add_fp32(d_x32, d_proj, d_res, H, graph_stream);
    }
    // Final norm + lm_head (also captured in graph)
    blackwell::kernels::fused_rmsnorm(d_xi_f, d_x32, d_fn, H, eps, graph_stream);
    blackwell::kernels::quantize_int4(d_x_i4, d_x_i4_sc, d_xi_f, H, graph_stream);
    blackwell::kernels::gemv_int4_warp_f16wsc(d_logits, (const uint8_t*)d_x_i4, d_x_i4_sc, lm_head_w.d, lm_head_w.sc16, H, V, graph_stream);
    
    cudaGraph_t graph;
    cudaError_t cerr = cudaStreamEndCapture(graph_stream, &graph);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "FAIL capture: %s\n", cudaGetErrorString(cerr));
        // Try to get more info
        cudaGetLastError();
        return 1;
    }
    
    // Count nodes
    size_t num_nodes = 0;
    cudaGraphGetNodes(graph, NULL, &num_nodes);
    fprintf(stderr, "  Captured %zu nodes\n", num_nodes);
    
    cudaGraphExec_t graph_exec;
    cerr = cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "FAIL instantiate: %s\n", cudaGetErrorString(cerr));
        cudaGraphDestroy(graph);
        return 1;
    }
    fprintf(stderr, "  Instantiated OK\n");
    
    // Warmup launches
    for (int i = 0; i < 3; ++i) {
        *h_seq_pos_pinned = 0;
        cudaMemcpyAsync(d_seq_pos, h_seq_pos_pinned, sizeof(int), cudaMemcpyHostToDevice, st);
        dequant_embed_row(h_embed.data(), ids[0], host_embed_d, host_embed_sc, H);
        cudaMemcpyAsync(d_x32, h_embed.data(), H*4, cudaMemcpyHostToDevice, st);
        cudaStreamSynchronize(st);
        cudaGraphLaunch(graph_exec, st);
        cudaStreamSynchronize(st);
    }
    
    // ── Graph benchmark ─────────────────────────────────────────────────
    fprintf(stderr, "\n── Graph benchmark (%d tokens) ──\n", num_tokens);
    
    // Reset for benchmark
    dequant_embed_row(h_embed.data(), ids[0], host_embed_d, host_embed_sc, H);
    cudaMemcpyAsync(d_x32, h_embed.data(), H*4, cudaMemcpyHostToDevice, st);
    cudaMemset(d_kc, 0, (size_t)NL*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vc, 0, (size_t)NL*nkv*MAXSEQ*hd*4);
    cudaStreamSynchronize(st);
    
    auto t2 = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < num_tokens; ++step) {
        // Update seq_pos: write to pinned memory, async copy to device
        *h_seq_pos_pinned = step;
        cudaMemcpyAsync(d_seq_pos, h_seq_pos_pinned, sizeof(int), cudaMemcpyHostToDevice, st);
        
        // Launch graph (full decode: QKV → head_norm → RoPE → KV cache → attention → MLP → lm_head)
        cudaGraphLaunch(graph_exec, st);
        cudaStreamSynchronize(st);
        
        // Sample
        int next_id;
        blackwell::kernels::sample_gpu(d_logits, V, 0, 0, d_next_id, 0xdeadbeefLL, step, st);
        cudaMemcpy(&next_id, d_next_id, 4, cudaMemcpyDeviceToHost);
        
        if (step < 5) {
            fprintf(stderr, "  step%d token: %d\n", step, next_id);
        }
        
        // Embed next token
        if (step < num_tokens - 1) {
            dequant_embed_row(h_embed.data(), next_id, host_embed_d, host_embed_sc, H);
            cudaMemcpyAsync(d_x32, h_embed.data(), H*4, cudaMemcpyHostToDevice, st);
        }
    }
    
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms_graph = std::chrono::duration<double,std::milli>(t3-t2).count();
    
    fprintf(stderr, "\n══ Results ══\n");
    fprintf(stderr, "  Per-kernel: %.2f ms/token = %.1f t/s\n", ms_per/num_tokens, 1000.0/(ms_per/num_tokens));
    fprintf(stderr, "  Graph:      %.2f ms/token = %.1f t/s\n", ms_graph/num_tokens, 1000.0/(ms_graph/num_tokens));
    if (ms_per > ms_graph)
        fprintf(stderr, "  Speedup:    %.1f%%\n", 100.0*(ms_per-ms_graph)/ms_per);
    else
        fprintf(stderr, "  Slowdown:   %.1f%%\n", 100.0*(ms_graph-ms_per)/ms_per);

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    return 0;
}
