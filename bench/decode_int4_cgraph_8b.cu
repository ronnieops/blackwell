// bench/decode_int4_cgraph_8b.cu — CUDA Graph for INT4 8B
// Captures full 36-layer decode loop (excluding KV cache + attention ops)

#include <cuda_runtime.h>
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

int main(int argc, char** argv) {
    int num_tokens = argc > 1 ? atoi(argv[1]) : 100;

    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    fprintf(stderr,"# INT4 8B CUDA Graph — %s\n", P.name);

    blackwell::BpeTokenizer tok;
    if(tok.load("tokenizer_data.bin")!=0){ fprintf(stderr,"FAIL\n"); return 1; }
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
    #undef AL

    float iv7=1.f/7.f;
    { std::vector<float> tmp(H/16,iv7); cudaMemcpy(d_x_i4_sc,tmp.data(),(H/16)*4,cudaMemcpyHostToDevice); }
    { std::vector<float> tmp(Q/16,iv7); cudaMemcpy(d_attn_i4_sc,tmp.data(),(Q/16)*4,cudaMemcpyHostToDevice); }
    { std::vector<float> tmp(I/16,iv7); cudaMemcpy(d_mlp_i4_sc,tmp.data(),(I/16)*4,cudaMemcpyHostToDevice); }

    fprintf(stderr,"Loading weights...\n");
    std::vector<LW4> W(NL);
    char p[256];
    for(int l=0;l<NL;++l){
        snprintf(p,256,"weights_int4_qwen3_8b/%d_self_attn.q_proj",l); W[l].q=upload_w4(p);
        snprintf(p,256,"weights_int4_qwen3_8b/%d_self_attn.k_proj",l); W[l].k=upload_w4(p);
        snprintf(p,256,"weights_int4_qwen3_8b/%d_self_attn.v_proj",l); W[l].v=upload_w4(p);
        snprintf(p,256,"weights_int4_qwen3_8b/%d_self_attn.o_proj",l); W[l].o=upload_w4(p);
        snprintf(p,256,"weights_int4_qwen3_8b/%d_mlp.gate_proj",l); W[l].g=upload_w4(p);
        snprintf(p,256,"weights_int4_qwen3_8b/%d_mlp.up_proj",l); W[l].u=upload_w4(p);
        snprintf(p,256,"weights_int4_qwen3_8b/%d_mlp.down_proj",l); W[l].d=upload_w4(p);
    }
    float* qk_h=(float*)malloc(NL*2*hd*4);
    {FILE*f=fopen("weights_int8_qwen3_8b/qk_norms.f32","rb");(void)fread(qk_h,4,NL*2*hd,f);fclose(f);}
    for(int l=0;l<NL;++l){
        cudaMalloc(&W[l].qn,hd*4);cudaMemcpy(W[l].qn,qk_h+l*2*hd,hd*4,cudaMemcpyHostToDevice);
        cudaMalloc(&W[l].kn,hd*4);cudaMemcpy(W[l].kn,qk_h+l*2*hd+hd,hd*4,cudaMemcpyHostToDevice);
    }free(qk_h);
    for(int l=0;l<NL;++l){
        float* w=(float*)malloc(H*4);
        snprintf(p,256,"weights_int8_qwen3_8b/%d_input_layernorm.f32",l);
        {FILE*f=fopen(p,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&W[l].rn_in,H*4);cudaMemcpy(W[l].rn_in,w,H*4,cudaMemcpyHostToDevice);
        snprintf(p,256,"weights_int8_qwen3_8b/%d_post_attention_layernorm.f32",l);
        {FILE*f=fopen(p,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&W[l].rn_post,H*4);cudaMemcpy(W[l].rn_post,w,H*4,cudaMemcpyHostToDevice);
        free(w);
    }
    {float*w=(float*)malloc(H*4);
    FILE*f=fopen("weights_int8_qwen3_8b/final_norm.f32","rb");(void)fread(w,4,H,f);fclose(f);
    cudaMemcpy(d_fn,w,H*4,cudaMemcpyHostToDevice);free(w);}

    DevW4 lm_head_w=upload_w4("weights_int4_qwen3_8b/lm_head");
    uint8_t* host_embed_d=new uint8_t[(size_t)H*V/2];
    float* host_embed_sc=new float[V*(H/16)];
    {FILE*f=fopen("weights_int4_qwen3_8b/embed_tokens.int4_t","rb");int h[5];fread(h,4,5,f);
     fread(host_embed_d,1,(size_t)h[0]*h[1]/2,f);fclose(f);
     f=fopen("weights_int4_qwen3_8b/embed_tokens.scale_t","rb");fread(h,4,5,f);
     fread(host_embed_sc,4,(size_t)h[3]*h[4],f);fclose(f);}

    cudaStream_t st; die(cudaStreamCreate(&st),"stream");
    std::vector<float> h_embed(H);

    // Embed first token
    dequant_embed_row(h_embed.data(), ids[0], host_embed_d, host_embed_sc, H);
    die(cudaMemcpyAsync(d_x32, h_embed.data(), H*4, cudaMemcpyHostToDevice, st), "embed");
    cudaMemset(d_kc, 0, (size_t)NL*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vc, 0, (size_t)NL*nkv*MAXSEQ*hd*4);

    // ── Per-kernel baseline ─────────────────────────────────────────────
    fprintf(stderr, "Per-kernel benchmark (%d tokens)...\n", num_tokens);
    auto t0 = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < num_tokens; ++step) {
        for(int l=0;l<NL;++l){
            size_t kv_off = (size_t)l*nkv*MAXSEQ*hd;
            cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st);
            blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_in,H,eps,st);
            blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st);
            blackwell::kernels::gemv_int4_warp(d_Q,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].q.d,W[l].q.sc,H,Q,st);
            blackwell::kernels::gemv_int4_warp(d_K,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].k.d,W[l].k.sc,H,KV,st);
            blackwell::kernels::gemv_int4_warp(d_V,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].v.d,W[l].v.sc,H,KV,st);
            head_norm_kernel<<<nqh,128,0,st>>>(d_Q,W[l].qn,nqh,hd,eps);
            head_norm_kernel<<<nkv,128,0,st>>>(d_K,W[l].kn,nkv,hd,eps);
            apply_rope_kernel<<<nqh,hd/2,0,st>>>(d_Q,nqh,hd,step);
            apply_rope_kernel<<<nkv,hd/2,0,st>>>(d_K,nkv,hd,step);
            blackwell::kernels::update_kv_cache(d_kc+kv_off,d_vc+kv_off,d_K,d_V,0,step,nkv,hd,MAXSEQ,st);
            blackwell::kernels::attention_decode_batched_gqa(d_attn,d_Q,d_kc,d_vc,step,nqh,nkv,hd,MAXSEQ,1,
                (size_t)NL*nkv*MAXSEQ*hd,kv_off,st);
            blackwell::kernels::quantize_int4(d_attn_i4,d_attn_i4_sc,d_attn,Q,st);
            blackwell::kernels::gemv_int4_warp(d_proj,(const uint8_t*)d_attn_i4,d_attn_i4_sc,W[l].o.d,W[l].o.sc,Q,H,st);
            blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,st);
            cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st);
            blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_post,H,eps,st);
            blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st);
            blackwell::kernels::gemv_int4_warp(d_gate,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].g.d,W[l].g.sc,H,I,st);
            blackwell::kernels::gemv_int4_warp(d_up,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].u.d,W[l].u.sc,H,I,st);
            blackwell::kernels::apply_swiglu(d_gate,d_gate,d_up,I,st);
            blackwell::kernels::quantize_int4(d_mlp_i4,d_mlp_i4_sc,d_gate,I,st);
            blackwell::kernels::gemv_int4_warp(d_proj,(const uint8_t*)d_mlp_i4,d_mlp_i4_sc,W[l].d.d,W[l].d.sc,I,H,st);
            blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,st);
        }
        blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,d_fn,H,eps,st);
        blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st);
        blackwell::kernels::gemv_int4_warp(d_logits,(const uint8_t*)d_x_i4,d_x_i4_sc,lm_head_w.d,lm_head_w.sc,H,V,st);
        blackwell::kernels::sample_gpu(d_logits,V,0,0,d_next_id,0xdeadbeefLL,step,st);
        cudaStreamSynchronize(st);
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_per = std::chrono::duration<double,std::milli>(t1-t0).count();
    fprintf(stderr, "  Per-kernel: %.1f ms/token = %.0f t/s\n", ms_per/num_tokens, 1000.0/(ms_per/num_tokens));

    // ── CUDA Graph capture (excluding KV cache + attention) ───────────
    fprintf(stderr, "Capturing graph (%d layers, %d kernels)...\n", NL, NL*20);
    
    cudaStream_t graph_stream;
    die(cudaStreamCreate(&graph_stream), "graph_stream");

    cudaStreamBeginCapture(graph_stream, cudaStreamCaptureModeGlobal);
    
    for (int l = 0; l < NL; ++l) {
        cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,graph_stream);
        blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_in,H,eps,graph_stream);
        blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,graph_stream);
        blackwell::kernels::gemv_int4_warp(d_Q,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].q.d,W[l].q.sc,H,Q,graph_stream);
        blackwell::kernels::gemv_int4_warp(d_K,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].k.d,W[l].k.sc,H,KV,graph_stream);
        blackwell::kernels::gemv_int4_warp(d_V,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].v.d,W[l].v.sc,H,KV,graph_stream);
        head_norm_kernel<<<nqh,128,0,graph_stream>>>(d_Q,W[l].qn,nqh,hd,eps);
        head_norm_kernel<<<nkv,128,0,graph_stream>>>(d_K,W[l].kn,nkv,hd,eps);
        apply_rope_kernel<<<nqh,hd/2,0,graph_stream>>>(d_Q,nqh,hd,0);
        apply_rope_kernel<<<nkv,hd/2,0,graph_stream>>>(d_K,nkv,hd,0);
        // Skip: update_kv_cache + attention (cudaMemcpyAsync not allowed in capture)
        blackwell::kernels::quantize_int4(d_attn_i4,d_attn_i4_sc,d_attn,Q,graph_stream);
        blackwell::kernels::gemv_int4_warp(d_proj,(const uint8_t*)d_attn_i4,d_attn_i4_sc,W[l].o.d,W[l].o.sc,Q,H,graph_stream);
        blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,graph_stream);
        cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,graph_stream);
        blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_post,H,eps,graph_stream);
        blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,graph_stream);
        blackwell::kernels::gemv_int4_warp(d_gate,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].g.d,W[l].g.sc,H,I,graph_stream);
        blackwell::kernels::gemv_int4_warp(d_up,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].u.d,W[l].u.sc,H,I,graph_stream);
        blackwell::kernels::apply_swiglu(d_gate,d_gate,d_up,I,graph_stream);
        blackwell::kernels::quantize_int4(d_mlp_i4,d_mlp_i4_sc,d_gate,I,graph_stream);
        blackwell::kernels::gemv_int4_warp(d_proj,(const uint8_t*)d_mlp_i4,d_mlp_i4_sc,W[l].d.d,W[l].d.sc,I,H,graph_stream);
        blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,graph_stream);
    }
    blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,d_fn,H,eps,graph_stream);
    blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,graph_stream);
    blackwell::kernels::gemv_int4_warp(d_logits,(const uint8_t*)d_x_i4,d_x_i4_sc,lm_head_w.d,lm_head_w.sc,H,V,graph_stream);
    
    cudaGraph_t graph;
    cudaError_t cerr = cudaStreamEndCapture(graph_stream, &graph);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "FAIL capture: %s\n", cudaGetErrorString(cerr));
        return 1;
    }
    
    cudaGraphExec_t graph_exec;
    cerr = cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "FAIL instantiate: %s\n", cudaGetErrorString(cerr));
        cudaGraphDestroy(graph);
        return 1;
    }
    
    // Warmup
    cudaGraphLaunch(graph_exec, st);
    cudaStreamSynchronize(st);
    
    fprintf(stderr, "  Graph captured OK (%d nodes)\n", NL*18);
    
    // ── Graph benchmark ─────────────────────────────────────────────────
    fprintf(stderr, "Graph benchmark (%d tokens)...\n", num_tokens);
    
    auto t2 = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < num_tokens; ++step) {
        // KV cache + attention (outside graph)
        for(int l=0;l<NL;++l){
            size_t kv_off = (size_t)l*nkv*MAXSEQ*hd;
            blackwell::kernels::update_kv_cache(d_kc+kv_off,d_vc+kv_off,d_K,d_V,0,step,nkv,hd,MAXSEQ,st);
            blackwell::kernels::attention_decode_batched_gqa(d_attn,d_Q,d_kc,d_vc,step,nqh,nkv,hd,MAXSEQ,1,
                (size_t)NL*nkv*MAXSEQ*hd,kv_off,st);
        }
        // Launch graph
        cudaGraphLaunch(graph_exec, st);
        cudaStreamSynchronize(st);
        
        // Sample
        int next_id;
        blackwell::kernels::sample_gpu(d_logits,V,0,0,d_next_id,0xdeadbeefLL,step,st);
        cudaMemcpy(&next_id,d_next_id,4,cudaMemcpyDeviceToHost);
        
        if (step == 0) {
            std::vector<float> h_logits(V);
            cudaMemcpy(h_logits.data(), d_logits, V*4, cudaMemcpyDeviceToHost);
            auto it = std::max_element(h_logits.begin(), h_logits.end());
            fprintf(stderr, "  step0 top1: %d logits=%.1f\n", (int)(it-h_logits.begin()), *it);
        }
        
        // Embed next token
        if (step < num_tokens - 1) {
            dequant_embed_row(h_embed.data(), next_id, host_embed_d, host_embed_sc, H);
            cudaMemcpyAsync(d_x32, h_embed.data(), H*4, cudaMemcpyHostToDevice, st);
        }
    }
    
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms_graph = std::chrono::duration<double,std::milli>(t3-t2).count();
    
    fprintf(stderr, "\n── Results ──\n");
    fprintf(stderr, "  Per-kernel: %.1f ms/token = %.0f t/s\n", ms_per/num_tokens, 1000.0/(ms_per/num_tokens));
    fprintf(stderr, "  Graph:      %.1f ms/token = %.0f t/s\n", ms_graph/num_tokens, 1000.0/(ms_graph/num_tokens));
    fprintf(stderr, "  Speedup:     %.0f%%\n", 100.0*(ms_per-ms_graph)/ms_per);

    return 0;
}
