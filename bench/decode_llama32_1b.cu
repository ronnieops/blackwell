// bench/decode_llama32_1b.cu — INT4 decode for Llama 3.2 1B
// Config: 16 layers, H=2048, I=8192, nqh=32, nkv=8, hd=64, V=128256
// Build: cmake --build build --target decode_llama32_1b

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include "blackwell/kernels.h"

#define die(e,m) do{if(e!=cudaSuccess){printf("FAIL %s: %s\n",m,cudaGetErrorString(e));exit(1);}}while(0)

using Clock = std::chrono::high_resolution_clock;

const int NL=16, H=2048, I=8192, nqh=32, nkv=8, hd=64, V=128256;
const int MAXSEQ=512;
const float rope_theta=500000.0f, eps=1e-6f;

struct W4 { int K,N; uint8_t* d; float* sc; };

__global__ static void head_norm_kernel(float* out, const float* in, const float* w, int nh, int hd, float eps) {
    int h = blockIdx.x;
    if (h >= nh) return;
    float mean=0, var=0;
    for (int i=0;i<hd;i++) mean+=in[h*hd+i];
    mean/=hd;
    for (int i=0;i<hd;i++){ float d=in[h*hd+i]-mean; var+=d*d; }
    var = rsqrtf(var/hd + eps);
    for (int i=0;i<hd;i++) out[h*hd+i]=(in[h*hd+i]-mean)*var*w[h*hd+i];
}

__global__ static void rope_kernel(float* data, int n_heads, int head_dim, int pos, float rope_theta) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (h >= n_heads || tid >= head_dim/2) return;
    int offset = h * head_dim;
    float c = cosf(pos * powf(rope_theta, -2.0f * (2*tid) / head_dim));
    float s = sinf(pos * powf(rope_theta, -2.0f * (2*tid) / head_dim));
    float x0 = data[offset + 2*tid];
    float x1 = data[offset + 2*tid + 1];
    data[offset + 2*tid] = x0 * c - x1 * s;
    data[offset + 2*tid + 1] = x0 * s + x1 * c;
}

static W4 load_w4(const char* p) {
    char fp[512]; snprintf(fp,512,"%s.int4_t",p);
    FILE* f=fopen(fp,"rb");
    if(!f){ printf("FAIL: cannot open %s\n", fp); exit(1); }
    int h[5]; fread(h,4,5,f);
    W4 w; w.K=h[0]; w.N=h[1];
    size_t nb=(size_t)h[0]*h[1]/2;
    uint8_t* t=new uint8_t[nb]; fread(t,1,nb,f); fclose(f);
    printf("  Allocating %zu bytes... ", nb); fflush(stdout);
    cudaError_t e = cudaMalloc(&w.d,nb);
    if(e!=cudaSuccess){ printf("cudaMalloc failed: %s\n", cudaGetErrorString(e)); exit(1); }
    printf("copying... "); fflush(stdout);
    e = cudaMemcpy(w.d,t,nb,cudaMemcpyHostToDevice);
    if(e!=cudaSuccess){ printf("cudaMemcpy failed: %s\n", cudaGetErrorString(e)); exit(1); }
    delete[] t;
    snprintf(fp,512,"%s.scale_t",p); f=fopen(fp,"rb");
    if(!f){ printf("FAIL: cannot open %s\n", fp); exit(1); }
    fread(h,4,5,f);
    size_t ns=(size_t)h[3]*h[4];
    float* ts=new float[ns]; fread(ts,4,ns,f); fclose(f);
    e = cudaMalloc(&w.sc,ns*4);
    if(e!=cudaSuccess){ printf("cudaMalloc scale failed: %s\n", cudaGetErrorString(e)); exit(1); }
    e = cudaMemcpy(w.sc,ts,ns*4,cudaMemcpyHostToDevice);
    if(e!=cudaSuccess){ printf("cudaMemcpy scale failed: %s\n", cudaGetErrorString(e)); exit(1); }
    delete[] ts;
    printf("OK\n"); fflush(stdout);
    return w;
}

int main(int argc, char** argv) {
    const char* dir = (argc>1) ? argv[1] : "/mnt/data/ai/models/llama32-1b-int4";
    int steps = (argc>2) ? atoi(argv[2]) : 50;
    
    printf("Initializing..."); fflush(stdout);
    cudaSetDevice(0);
    printf(" CUDA set"); fflush(stdout);
    cudaStream_t st; cudaStreamCreate(&st);
    
    // Load embedding (INT4, transposed [V][H])
    // GGUF Llama mapper uses "embed_tokens" not "token_embd"
    W4 emb_w = load_w4((std::string(dir)+"/embed_tokens").c_str());
    printf("Embedding: %d x %d INT4\n", emb_w.K, emb_w.N);
    
    // Pre-load full FP32 embedding table
    float* h_emb = new float[V * H];
    uint8_t* hi4 = new uint8_t[V * H/2];
    FILE* f = fopen((std::string(dir)+"/embed_tokens.int4_t").c_str(),"rb");
    int eh[5]; fread(eh,4,5,f); fread(hi4,1,(size_t)V*H/2,f); fclose(f);
    FILE* fs = fopen((std::string(dir)+"/embed_tokens.scale_t").c_str(),"rb");
    int sh[5]; fread(sh,4,5,fs);
    float* hsc = new float[sh[3]*sh[4]]; fread(hsc,4,sh[3]*sh[4],fs); fclose(fs);
    // Dequantize INT4 block-16 to FP32
    for(int v=0;v<V;v++) for(int i=0;i<H;i++){
        int idx=v*H+i, bidx=idx/2, nib=(idx%2)?(hi4[bidx]>>4):(hi4[bidx]&0xF);
        int bl = idx / 16;  // block-16
        h_emb[idx]=(nib-8)*hsc[bl];
    }
    delete[] hi4; delete[] hsc;
    float* d_emb; cudaMalloc(&d_emb,V*H*4); cudaMemcpy(d_emb,h_emb,V*H*4,cudaMemcpyHostToDevice);
    delete[] h_emb; die(cudaGetLastError(),"emb");
    
    // KV cache
    float* d_kv; cudaMalloc(&d_kv,(size_t)NL*MAXSEQ*nkv*hd*4);
    cudaMemset(d_kv,0,(size_t)NL*MAXSEQ*nkv*hd*4);
    
    // Load layer weights
    W4 qw[NL],kw[NL],vw[NL],ow[NL],gw[NL],uw[NL],dw[NL];
    float *ln1[NL],*ln2[NL], *d_ln1[NL],*d_ln2[NL];
    for(int l=0;l<NL;l++){
        char p[256];
        snprintf(p,256,"%s/%d_self_attn.q_proj",dir,l); qw[l]=load_w4(p);
        printf("Loaded q_proj layer %d\n", l); fflush(stdout);
        snprintf(p,256,"%s/%d_self_attn.k_proj",dir,l); kw[l]=load_w4(p);
        printf("Loaded k_proj layer %d\n", l); fflush(stdout);
        snprintf(p,256,"%s/%d_self_attn.v_proj",dir,l); vw[l]=load_w4(p);
        printf("Loaded v_proj layer %d\n", l); fflush(stdout);
        snprintf(p,256,"%s/%d_self_attn.o_proj",dir,l); ow[l]=load_w4(p);
        snprintf(p,256,"%s/%d_mlp.gate_proj",dir,l); gw[l]=load_w4(p);
        snprintf(p,256,"%s/%d_mlp.up_proj",dir,l); uw[l]=load_w4(p);
        snprintf(p,256,"%s/%d_mlp.down_proj",dir,l); dw[l]=load_w4(p);
        printf("Loaded mlp layer %d\n", l); fflush(stdout);
        ln1[l]=(float*)malloc(H*4); ln2[l]=(float*)malloc(H*4);
        snprintf(p,256,"%s/%d_input_layernorm.f32",dir,l);
        f=fopen(p,"rb"); fread(ln1[l],4,H,f); fclose(f);
        snprintf(p,256,"%s/%d_post_attention_layernorm.f32",dir,l);
        f=fopen(p,"rb"); fread(ln2[l],4,H,f); fclose(f);
        cudaMalloc(&d_ln1[l],H*4); cudaMemcpy(d_ln1[l],ln1[l],H*4,cudaMemcpyHostToDevice);
        cudaMalloc(&d_ln2[l],H*4); cudaMemcpy(d_ln2[l],ln2[l],H*4,cudaMemcpyHostToDevice);
    }
    
    float* fn=(float*)malloc(H*4); f=fopen((std::string(dir)+"/final_norm.f32").c_str(),"rb");
    fread(fn,4,H,f); fclose(f);
    float* d_fn; cudaMalloc(&d_fn,H*4); cudaMemcpy(d_fn,fn,H*4,cudaMemcpyHostToDevice);
    
    float* qkn=(float*)malloc(nqh*hd*4);
    f=fopen((std::string(dir)+"/qk_norms.f32").c_str(),"rb"); fread(qkn,4,nqh*hd,f); fclose(f);
    float* d_qkn; cudaMalloc(&d_qkn,nqh*hd*4); cudaMemcpy(d_qkn,qkn,nqh*hd*4,cudaMemcpyHostToDevice);
    
    // Allocate activations
    float *dd_h, *dd_q, *dd_k, *dd_v, *dd_qn, *dd_kn, *dd_attn, *dd_o,
          *dd_gate, *dd_up, *dd_mlp, *dd_res, *dd_fn_h;
    uint8_t* dd_x_i4; float* dd_x_sc;
    cudaMalloc(&dd_h,H*4); cudaMalloc(&dd_q,nqh*hd*4); cudaMalloc(&dd_k,nkv*hd*4);
    cudaMalloc(&dd_v,nkv*hd*4); cudaMalloc(&dd_qn,nqh*hd*4); cudaMalloc(&dd_kn,nkv*hd*4);
    cudaMalloc(&dd_attn,nqh*hd*4); cudaMalloc(&dd_o,H*4);
    cudaMalloc(&dd_gate,I*4); cudaMalloc(&dd_up,I*4); cudaMalloc(&dd_mlp,H*4);
    cudaMalloc(&dd_res,H*4); cudaMalloc(&dd_fn_h,H*4);
    int max_dim = H > I ? H : I;
    cudaMalloc(&dd_x_i4,max_dim/2); cudaMalloc(&dd_x_sc,(max_dim+15)/16*4);
    
    float rope_cfg[2]={rope_theta,(float)hd};
    float* d_rope; cudaMalloc(&d_rope,8); cudaMemcpy(d_rope,rope_cfg,8,cudaMemcpyHostToDevice);
    int seq_pos=0; int* d_sp; cudaMalloc(&d_sp,4); cudaMemcpy(d_sp,&seq_pos,4,cudaMemcpyHostToDevice);
    
    printf("Config: NL=%d H=%d I=%d nqh=%d nkv=%d hd=%d V=%d rope=%.0f\n",
           NL,H,I,nqh,nkv,hd,V,rope_theta);
    
    int input_token = 13860;
    
    // Warmup
    printf("Starting warmup..."); fflush(stdout);
    blackwell::kernels::fused_rmsnorm(d_emb+input_token*H,d_ln1[0],dd_h,H,eps,st);
    printf(" rmsnorm done"); fflush(stdout);
    blackwell::kernels::quantize_int4(dd_x_i4,dd_x_sc,dd_h,H,st);
    printf(" quant done"); fflush(stdout);
    blackwell::kernels::gemv_int4_warp(dd_q,(uint8_t*)dd_x_i4,dd_x_sc,qw[0].d,qw[0].sc,H,nqh*hd,st);
    printf(" q GEMV done"); fflush(stdout);
    blackwell::kernels::gemv_int4_warp(dd_k,(uint8_t*)dd_x_i4,dd_x_sc,kw[0].d,kw[0].sc,H,nkv*hd,st);
    printf(" k GEMV done"); fflush(stdout);
    blackwell::kernels::gemv_int4_warp(dd_v,(uint8_t*)dd_x_i4,dd_x_sc,vw[0].d,vw[0].sc,H,nkv*hd,st);
    printf(" v GEMV done"); fflush(stdout);
    cudaStreamSynchronize(st);
    fprintf(stderr, "AFTER SYNC\n"); fflush(stderr);
    cudaError_t e = cudaPeekAtLastError();
    fprintf(stderr, "CUDA error check: %s\n", cudaGetErrorString(e)); fflush(stderr);
    fprintf(stderr, " warmup done\n"); fflush(stderr);
    
    printf(" about to set last_id..."); fflush(stdout);
    int last_id = input_token;
    printf(" last_id=%d done\n", last_id); fflush(stdout);
    auto start = Clock::now();
    printf(" Starting decode loop...\n"); fflush(stdout);
    
    for(int step=0;step<steps;step++){
        if(step==0) { printf(" step0..."); fflush(stdout); }
        cudaMemcpy(d_sp,&seq_pos,4,cudaMemcpyHostToDevice);
        
        blackwell::kernels::fused_rmsnorm(d_emb+last_id*H,d_ln1[0],dd_h,H,eps,st);
        if(step==0) { printf(" rmsnorm"); fflush(stdout); }
        blackwell::kernels::quantize_int4(dd_x_i4,dd_x_sc,dd_h,H,st);
        if(step==0) { printf(" quant"); fflush(stdout); }
        
        for(int l=0;l<NL;l++){
            if(step==0 && l==0) { printf(" layer0"); fflush(stdout); }
            // QKV
            blackwell::kernels::gemv_int4_warp(dd_q,(uint8_t*)dd_x_i4,dd_x_sc,qw[l].d,qw[l].sc,H,nqh*hd,st);
            blackwell::kernels::gemv_int4_warp(dd_k,(uint8_t*)dd_x_i4,dd_x_sc,kw[l].d,kw[l].sc,H,nkv*hd,st);
            blackwell::kernels::gemv_int4_warp(dd_v,(uint8_t*)dd_x_i4,dd_x_sc,vw[l].d,vw[l].sc,H,nkv*hd,st);
            if(step==0 && l==0) { printf(" qkv"); fflush(stdout); }
            
            // Head norm (skip for Llama - qk_norms are identity weights)
            cudaMemcpy(dd_qn,dd_q,nqh*hd*4,cudaMemcpyDeviceToDevice);
            cudaMemcpy(dd_kn,dd_k,nkv*hd*4,cudaMemcpyDeviceToDevice);
            
            // RoPE (inline)
            rope_kernel<<<nqh,hd/2,0,st>>>(dd_qn,nqh,hd,seq_pos,rope_theta);
            rope_kernel<<<nkv,hd/2,0,st>>>(dd_kn,nkv,hd,seq_pos,rope_theta);
            if(step==0 && l==0) { printf(" rope"); fflush(stdout); }
            
            // Save residual
            cudaMemcpy(dd_res,dd_h,H*4,cudaMemcpyDeviceToDevice);
            
            // Attention (layer l, KV cache at d_kv + layer_off)
            size_t layer_off = (size_t)l * nkv * hd * MAXSEQ;
            if(step==0 && l==0) { printf(" attn start"); fflush(stdout); }
            blackwell::kernels::attention_decode_gqa(dd_qn, d_kv + layer_off, dd_kn, d_kv + layer_off,
                seq_pos, nqh, nkv, hd, MAXSEQ, st);
            if(step==0 && l==0) { printf(" attn done"); fflush(stdout); }
            
            // O projection (quantize attn output)
            if(step==0 && l==0) { printf(" quant_attn"); fflush(stdout); }
            blackwell::kernels::quantize_int4(dd_x_i4,dd_x_sc,dd_attn,nqh*hd,st);
            if(step==0 && l==0) { printf(" o_gemv"); fflush(stdout); }
            blackwell::kernels::gemv_int4_warp(dd_o,(uint8_t*)dd_x_i4,dd_x_sc,ow[l].d,ow[l].sc,nqh*hd,H,st);
            if(step==0 && l==0) { printf(" resid1"); fflush(stdout); }
            

            // Residual 1
            blackwell::kernels::vector_add_fp32(dd_res,dd_o,dd_h,H,st);
            
            // Save residual 2
            cudaMemcpy(dd_res,dd_h,H*4,cudaMemcpyDeviceToDevice);
            
            // MLP layernorm
            blackwell::kernels::fused_rmsnorm(dd_h,d_ln2[l],dd_h,H,eps,st);
            
            // Gate + Up
            blackwell::kernels::quantize_int4(dd_x_i4,dd_x_sc,dd_h,H,st);
            blackwell::kernels::gemv_int4_warp(dd_gate,(uint8_t*)dd_x_i4,dd_x_sc,gw[l].d,gw[l].sc,H,I,st);
            blackwell::kernels::gemv_int4_warp(dd_up,(uint8_t*)dd_x_i4,dd_x_sc,uw[l].d,uw[l].sc,H,I,st);
            blackwell::kernels::apply_swiglu(dd_gate,dd_up,dd_gate,I,st);
            
            // Down
            blackwell::kernels::quantize_int4(dd_x_i4,dd_x_sc,dd_gate,I,st);
            blackwell::kernels::gemv_int4_warp(dd_h,(uint8_t*)dd_x_i4,dd_x_sc,dw[l].d,dw[l].sc,I,H,st);
            
            // Residual 2
            blackwell::kernels::vector_add_fp32(dd_res,dd_h,dd_h,H,st);
        }
        
        // Final norm
        blackwell::kernels::fused_rmsnorm(dd_h,d_fn,dd_fn_h,H,eps,st);
        cudaStreamSynchronize(st);
        
        // Sample (argmax)
        float* h_fn_h = (float*)malloc(H*4);
        cudaMemcpy(h_fn_h,dd_fn_h,H*4,cudaMemcpyDeviceToHost);
        float max_dot=-1e9; int max_id=0;
        for(int v=0;v<V;v++){
            float dot=0; float* e=d_emb+v*H;
            for(int i=0;i<H;i++) dot+=h_fn_h[i]*e[i];
            if(dot>max_dot){max_dot=dot;max_id=v;}
        }
        free(h_fn_h);
        
        if(step<5) printf("[%d] token=%d (%.1f)\n", step, max_id, max_dot);
        last_id=max_id;
        seq_pos++;
    }
    
    auto end = Clock::now();
    double ms = std::chrono::duration<double,std::milli>(end-start).count();
    printf("\n%d steps: %.1f ms, %.1f t/s\n", steps, ms, steps*1000.0/ms);
    
    return 0;
}