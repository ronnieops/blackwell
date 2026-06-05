// bench/verify_int4_asym_full.cu — Full 28-layer asymmetric INT4 vs INT8 SNR
//
// Loads asymmetric INT4 weights, runs 28 layers, compares per-layer vs INT8.
//
// Build:
//   /usr/local/cuda-13.3/bin/nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/verify_int4_asym_full.cu build/libblackwell_kernels.a \
//     -o bench/verify_int4_asym_full
//
// Run: ./bench/verify_int4_asym_full [token_id]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include "blackwell/kernels.h"

static void die(cudaError_t e, const char* m) {
    if(e!=cudaSuccess){printf("FAIL %s: %s\n",m,cudaGetErrorString(e));exit(1);}
}

const int H=2048, Q=2048, KV=1024, I=6144;
const int nqh=16, nkv=8, hd=128, MAXSEQ=4096;
const float eps=1e-6f;
const int V=151936;
const int NL=28;

struct DW8 { int8_t* d; float* sc; };
static DW8 upload_w8(const char* prefix) {
    char p[256]; snprintf(p,256,"%s.int8_t",prefix);
    FILE* f=fopen(p,"rb"); int h[5]; fread(h,4,5,f); DW8 dw;
    size_t ds=(size_t)h[0]*h[1];
    int8_t* td=new int8_t[ds]; fread(td,1,ds,f); fclose(f);
    cudaMalloc(&dw.d,ds); cudaMemcpy(dw.d,td,ds,cudaMemcpyHostToDevice); delete[] td;
    snprintf(p,256,"%s.scale_t",prefix); f=fopen(p,"rb"); fread(h,4,5,f);
    size_t ss=(size_t)h[3]*h[4];
    float* ts=new float[ss]; fread(ts,4,ss,f); fclose(f);
    cudaMalloc(&dw.sc,ss*4); cudaMemcpy(dw.sc,ts,ss*4,cudaMemcpyHostToDevice); delete[] ts;
    return dw;
}

struct DW4 { uint8_t* d; float* sc; };
static DW4 upload_w4_asym(const char* prefix) {
    char p[256]; snprintf(p,256,"%s.int4_t",prefix);
    FILE* f=fopen(p,"rb"); int h[5]; fread(h,4,5,f); DW4 dw;
    size_t ds=(size_t)h[0]*h[1]/2;
    uint8_t* td=new uint8_t[ds]; fread(td,1,ds,f); fclose(f);
    cudaMalloc(&dw.d,ds); cudaMemcpy(dw.d,td,ds,cudaMemcpyHostToDevice); delete[] td;
    snprintf(p,256,"%s.sc_zero_t",prefix); f=fopen(p,"rb"); fread(h,4,5,f);
    size_t ss=(size_t)h[3]*h[4]*2;
    float* ts=new float[ss]; fread(ts,4,ss,f); fclose(f);
    cudaMalloc(&dw.sc,ss*4); cudaMemcpy(dw.sc,ts,ss*4,cudaMemcpyHostToDevice); delete[] ts;
    return dw;
}

__global__ void head_norm_kernel(float* data, const float* weight, int nh, int hd, float eps) {
    int h=blockIdx.x; if(h>=nh) return;
    float* d=data+h*hd; __shared__ float wp[4]; float s=0;
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

int main(int argc, char** argv) {
    int token_id = 42;
    if(argc>1) token_id=atoi(argv[1]);

    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    printf("# Asymmetric INT4 Full 28-Layer SNR — Qwen3-1.7B\n");
    printf("  Device: %s\n", P.name);
    printf("  Token: %d\n\n", token_id);

    // ── Buffers (shared) ──
    float *d_x, *d_xi_f, *d_res;
    float *d_Q, *d_K, *d_V, *d_attn;
    uint8_t *d_x4; float *d_x4_sz;
    int8_t *d_xi8, *d_ai8, *d_mi8;
    float *d_proj, *d_gate, *d_up;
    uint8_t *d_m4; float *d_m4_sz;
    uint8_t *d_a4; float *d_a4_sz;
    float *d_fn, *d_kc, *d_vc;

    #define AL(p,n){cudaError_t _e=cudaMalloc(&(p),(n));\
        if(_e!=cudaSuccess){printf("FAIL malloc %s: %s\n",#p,cudaGetErrorString(_e));die(_e,#p);}}
    AL(d_x,H*4); AL(d_xi_f,H*4); AL(d_res,H*4);
    AL(d_Q,Q*4); AL(d_K,KV*4); AL(d_V,KV*4); AL(d_attn,Q*4);
    AL(d_x4,H/2); AL(d_x4_sz,2*(H/16)*4);
    AL(d_xi8,H); AL(d_ai8,Q); AL(d_mi8,I);
    AL(d_proj,H*4); AL(d_gate,I*4); AL(d_up,I*4);
    AL(d_m4,I/2); AL(d_m4_sz,2*(I/16)*4);
    AL(d_a4,Q/2); AL(d_a4_sz,2*(Q/16)*4);
    AL(d_fn,H*4);
    AL(d_kc,(size_t)NL*nkv*MAXSEQ*hd*4);
    AL(d_vc,(size_t)NL*nkv*MAXSEQ*hd*4);
    #undef AL

    cudaStream_t st; die(cudaStreamCreate(&st),"stream");

    printf("Loading INT8 weights...\n"); fflush(stdout);
    std::vector<DW8> W8_q(NL),W8_k(NL),W8_v(NL),W8_o(NL);
    std::vector<DW8> W8_g(NL),W8_u(NL),W8_d(NL);
    char p8[256];
    for(int l=0;l<NL;++l){
        snprintf(p8,256,"weights_int8_bf16/%d_self_attn.q_proj",l); W8_q[l]=upload_w8(p8);
        snprintf(p8,256,"weights_int8_bf16/%d_self_attn.k_proj",l); W8_k[l]=upload_w8(p8);
        snprintf(p8,256,"weights_int8_bf16/%d_self_attn.v_proj",l); W8_v[l]=upload_w8(p8);
        snprintf(p8,256,"weights_int8_bf16/%d_self_attn.o_proj",l); W8_o[l]=upload_w8(p8);
        snprintf(p8,256,"weights_int8_bf16/%d_mlp.gate_proj",l);   W8_g[l]=upload_w8(p8);
        snprintf(p8,256,"weights_int8_bf16/%d_mlp.up_proj",l);     W8_u[l]=upload_w8(p8);
        snprintf(p8,256,"weights_int8_bf16/%d_mlp.down_proj",l);  W8_d[l]=upload_w8(p8);
        if((l+1)%7==0) printf("  INT8 layer %d/%d\n",l+1,NL);
    }

    printf("Loading asymmetric INT4 weights...\n"); fflush(stdout);
    std::vector<DW4> W4_q(NL),W4_k(NL),W4_v(NL),W4_o(NL);
    std::vector<DW4> W4_g(NL),W4_u(NL),W4_d(NL);
    char p4[256];
    for(int l=0;l<NL;++l){
        snprintf(p4,256,"weights_int4_qwen3_1.7b_asym/%d_self_attn.q_proj",l); W4_q[l]=upload_w4_asym(p4);
        snprintf(p4,256,"weights_int4_qwen3_1.7b_asym/%d_self_attn.k_proj",l); W4_k[l]=upload_w4_asym(p4);
        snprintf(p4,256,"weights_int4_qwen3_1.7b_asym/%d_self_attn.v_proj",l); W4_v[l]=upload_w4_asym(p4);
        snprintf(p4,256,"weights_int4_qwen3_1.7b_asym/%d_self_attn.o_proj",l); W4_o[l]=upload_w4_asym(p4);
        snprintf(p4,256,"weights_int4_qwen3_1.7b_asym/%d_mlp.gate_proj",l);   W4_g[l]=upload_w4_asym(p4);
        snprintf(p4,256,"weights_int4_qwen3_1.7b_asym/%d_mlp.up_proj",l);     W4_u[l]=upload_w4_asym(p4);
        snprintf(p4,256,"weights_int4_qwen3_1.7b_asym/%d_mlp.down_proj",l);  W4_d[l]=upload_w4_asym(p4);
        if((l+1)%7==0) printf("  INT4 layer %d/%d\n",l+1,NL);
    }

    // Norm weights (shared between INT8 and INT4)
    float* qk_h=(float*)malloc(NL*2*hd*4);
    {FILE*f=fopen("weights_int8_bf16/qk_norms.f32","rb");
    (void)fread(qk_h,4,NL*2*hd,f);fclose(f);}
    std::vector<float*> d_qn(NL),d_kn(NL);
    for(int l=0;l<NL;++l){
        cudaMalloc(&d_qn[l],hd*4); cudaMemcpy(d_qn[l],qk_h+l*2*hd,hd*4,cudaMemcpyHostToDevice);
        cudaMalloc(&d_kn[l],hd*4); cudaMemcpy(d_kn[l],qk_h+l*2*hd+hd,hd*4,cudaMemcpyHostToDevice);
    }
    free(qk_h);

    std::vector<float*> d_rn_in(NL),d_rn_post(NL);
    for(int l=0;l<NL;++l){
        float* w=(float*)malloc(H*4);
        snprintf(p8,256,"weights_int8_bf16/%d_input_layernorm.f32",l);
        {FILE*f=fopen(p8,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&d_rn_in[l],H*4); cudaMemcpy(d_rn_in[l],w,H*4,cudaMemcpyHostToDevice);
        snprintf(p8,256,"weights_int8_bf16/%d_post_attention_layernorm.f32",l);
        {FILE*f=fopen(p8,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&d_rn_post[l],H*4); cudaMemcpy(d_rn_post[l],w,H*4,cudaMemcpyHostToDevice);
        free(w);
    }

    {float*w=(float*)malloc(H*4);
    FILE*f=fopen("weights_int8_bf16/final_norm.f32","rb");(void)fread(w,4,H,f);fclose(f);
    cudaMemcpy(d_fn,w,H*4,cudaMemcpyHostToDevice);free(w);}

    // Embeddings (host)
    int8_t* host_e8=new int8_t[(size_t)V*H];
    float* host_e8_sc=new float[(size_t)V*(H/16)];
    {FILE*f=fopen("weights_int8_bf16/embed_tokens.int8_t","rb");
    int h[5]; fread(h,4,5,f); fread(host_e8,1,(size_t)h[0]*h[1],f); fclose(f);
    f=fopen("weights_int8_bf16/embed_tokens.scale_t","rb");
    fread(h,4,5,f); fread(host_e8_sc,4,(size_t)h[3]*h[4],f); fclose(f);}

    uint8_t* host_e4=new uint8_t[(size_t)V*H/2];
    float* host_e4_sz=new float[(size_t)V*2*(H/16)];
    {FILE*f=fopen("weights_int4_qwen3_1.7b_asym/embed_tokens.int4_t","rb");
    int h[5]; fread(h,4,5,f); fread(host_e4,1,(size_t)h[0]*h[1]/2,f); fclose(f);
    f=fopen("weights_int4_qwen3_1.7b_asym/embed_tokens.sc_zero_t","rb");
    fread(h,4,5,f); fread(host_e4_sz,4,(size_t)h[3]*h[4]*2,f); fclose(f);}

    printf("All weights loaded.\n\n");

    // Host state storage
    std::vector<std::vector<float>> s8(NL, std::vector<float>(H));
    std::vector<std::vector<float>> s4(NL, std::vector<float>(H));
    std::vector<float> hb(H);

    // ────────────────── INT8 (28 layers) ──────────────────
    printf("── INT8 pipeline ──\n");
    for(int i=0;i<H;i++) hb[i]=(float)host_e8[token_id*H+i]*host_e8_sc[token_id*(H/16)+i/16];
    die(cudaMemcpyAsync(d_x,hb.data(),H*4,cudaMemcpyHostToDevice,st),"e8");
    for(int l=0;l<NL;++l){
        die(cudaMemcpyAsync(d_res,d_x,H*4,cudaMemcpyDeviceToDevice,st),"sr8");
        die(blackwell::kernels::fused_rmsnorm(d_xi_f,d_x,d_rn_in[l],H,eps,st),"ri8");
        die(blackwell::kernels::quantize_int8(d_xi8,d_x4_sz,d_xi_f,H,st),"qi8");
        die(blackwell::kernels::gemv_int8_warp(d_Q,d_xi8,d_x4_sz,W8_q[l].d,W8_q[l].sc,H,Q,st),"q8");
        die(blackwell::kernels::gemv_int8_warp(d_K,d_xi8,d_x4_sz,W8_k[l].d,W8_k[l].sc,H,KV,st),"k8");
        die(blackwell::kernels::gemv_int8_warp(d_V,d_xi8,d_x4_sz,W8_v[l].d,W8_v[l].sc,H,KV,st),"v8");
        head_norm_kernel<<<nqh,128,0,st>>>(d_Q,d_qn[l],nqh,hd,eps);die(cudaGetLastError(),"hq8");
        head_norm_kernel<<<nkv,128,0,st>>>(d_K,d_kn[l],nkv,hd,eps);die(cudaGetLastError(),"hk8");
        apply_rope_kernel<<<nqh,hd/2,0,st>>>(d_Q,nqh,hd,0);die(cudaGetLastError(),"rq8");
        apply_rope_kernel<<<nkv,hd/2,0,st>>>(d_K,nkv,hd,0);die(cudaGetLastError(),"rk8");
        int kb=l*nkv*MAXSEQ*hd;
        die(blackwell::kernels::update_kv_cache(d_kc+kb,d_vc+kb,d_K,d_V,0,0,nkv,hd,MAXSEQ,st),"kv8");
        die(blackwell::kernels::attention_decode_gqa(d_attn,d_Q,d_kc+kb,d_vc+kb,0,nqh,nkv,hd,MAXSEQ,st),"a8");
        die(blackwell::kernels::quantize_int8(d_ai8,d_a4_sz,d_attn,Q,st),"qa8");
        die(blackwell::kernels::gemv_int8_warp(d_proj,d_ai8,d_a4_sz,W8_o[l].d,W8_o[l].sc,Q,H,st),"o8");
        die(blackwell::kernels::vector_add_fp32(d_x,d_proj,d_res,H,st),"ra8");
        die(cudaMemcpyAsync(d_res,d_x,H*4,cudaMemcpyDeviceToDevice,st),"sr2_8");
        die(blackwell::kernels::fused_rmsnorm(d_xi_f,d_x,d_rn_post[l],H,eps,st),"rp8");
        die(blackwell::kernels::quantize_int8(d_mi8,d_m4_sz,d_xi_f,H,st),"qm8");
        die(blackwell::kernels::gemv_int8_warp(d_gate,d_mi8,d_m4_sz,W8_g[l].d,W8_g[l].sc,H,I,st),"g8");
        die(blackwell::kernels::gemv_int8_warp(d_up,d_mi8,d_m4_sz,W8_u[l].d,W8_u[l].sc,H,I,st),"u8");
        die(blackwell::kernels::apply_swiglu(d_gate,d_gate,d_up,I,st),"sw8");
        die(blackwell::kernels::quantize_int8(d_mi8,d_m4_sz,d_gate,I,st),"qd8");
        die(blackwell::kernels::gemv_int8_warp(d_x,d_mi8,d_m4_sz,W8_d[l].d,W8_d[l].sc,I,H,st),"d8");
        die(blackwell::kernels::vector_add_fp32(d_x,d_x,d_res,H,st),"rm8");
        die(cudaMemcpy(s8[l].data(),d_x,H*4,cudaMemcpyDeviceToHost),"cp8");
        if((l+1)%7==0||l+1==NL)printf("  l %d/%d\n",l+1,NL);
    }

    // ────────────────── Asymmetric INT4 (28 layers) ──────────────────
    printf("\n── Asymmetric INT4 pipeline ──\n");
    cudaMemset(d_kc,0,(size_t)NL*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vc,0,(size_t)NL*nkv*MAXSEQ*hd*4);

    // INT4 asymmetric embedding dequant
    {
        int kblks=H/16;
        for(int b=0;b<kblks;++b){
            float sc=host_e4_sz[(size_t)token_id*2*kblks+b*2];
            float zf=host_e4_sz[(size_t)token_id*2*kblks+b*2+1];
            int zero=(int)zf;
            for(int i=0;i<16;++i){
                size_t bi=(size_t)token_id*H/2+(size_t)b*8+i/2;
                uint8_t by=host_e4[bi];
                int nb=(i&1)?((by>>4)&0x0F):(by&0x0F);
                hb[b*16+i]=(float)(nb-zero)*sc;
            }
        }
    }
    die(cudaMemcpyAsync(d_x,hb.data(),H*4,cudaMemcpyHostToDevice,st),"e4");

    for(int l=0;l<NL;++l){
        // Attention residual
        die(cudaMemcpyAsync(d_res,d_x,H*4,cudaMemcpyDeviceToDevice,st),"sr4");
        // Pre-attention norm + asym quant (non-fused for correctness)
        die(blackwell::kernels::fused_rmsnorm(d_xi_f,d_x,d_rn_in[l],H,eps,st),"ri4");
        die(blackwell::kernels::quantize_int4_asym(d_x4,d_x4_sz,d_xi_f,H,st),"qi4");
        // QKV
        die(blackwell::kernels::gemv_int4_asym_batched(d_Q,d_x4,d_x4_sz,W4_q[l].d,W4_q[l].sc,H,Q,1,st),"q4");
        die(blackwell::kernels::gemv_int4_asym_batched(d_K,d_x4,d_x4_sz,W4_k[l].d,W4_k[l].sc,H,KV,1,st),"k4");
        die(blackwell::kernels::gemv_int4_asym_batched(d_V,d_x4,d_x4_sz,W4_v[l].d,W4_v[l].sc,H,KV,1,st),"v4");
        head_norm_kernel<<<nqh,128,0,st>>>(d_Q,d_qn[l],nqh,hd,eps);die(cudaGetLastError(),"hq4");
        head_norm_kernel<<<nkv,128,0,st>>>(d_K,d_kn[l],nkv,hd,eps);die(cudaGetLastError(),"hk4");
        apply_rope_kernel<<<nqh,hd/2,0,st>>>(d_Q,nqh,hd,0);die(cudaGetLastError(),"rq4");
        apply_rope_kernel<<<nkv,hd/2,0,st>>>(d_K,nkv,hd,0);die(cudaGetLastError(),"rk4");
        int kb=l*nkv*MAXSEQ*hd;
        die(blackwell::kernels::update_kv_cache(d_kc+kb,d_vc+kb,d_K,d_V,0,0,nkv,hd,MAXSEQ,st),"kv4");
        die(blackwell::kernels::attention_decode_batched_gqa(
            d_attn,d_Q,d_kc,d_vc,0,nqh,nkv,hd,MAXSEQ,1,
            (size_t)NL*nkv*MAXSEQ*hd,kb,st),"a4");
        // Wo
        die(blackwell::kernels::quantize_int4_asym(d_a4,d_a4_sz,d_attn,Q,st),"qa4");
        die(blackwell::kernels::gemv_int4_asym_batched(d_proj,d_a4,d_a4_sz,W4_o[l].d,W4_o[l].sc,Q,H,1,st),"o4");
        // Attn residual
        die(blackwell::kernels::vector_add_fp32(d_x,d_proj,d_res,H,st),"ra4");
        // Save MLP residual
        die(cudaMemcpyAsync(d_res,d_x,H*4,cudaMemcpyDeviceToDevice,st),"sr2_4");
        // Post-attn norm + asym quant
        die(blackwell::kernels::fused_rmsnorm(d_xi_f,d_x,d_rn_post[l],H,eps,st),"rp4");
        die(blackwell::kernels::quantize_int4_asym(d_x4,d_x4_sz,d_xi_f,H,st),"qm4");
        // Gate + Up
        die(blackwell::kernels::gemv_int4_asym_batched(d_gate,d_x4,d_x4_sz,W4_g[l].d,W4_g[l].sc,H,I,1,st),"g4");
        die(blackwell::kernels::gemv_int4_asym_batched(d_up,d_x4,d_x4_sz,W4_u[l].d,W4_u[l].sc,H,I,1,st),"u4");
        // SwiGLU + asym quant
        die(blackwell::kernels::fused_swiglu_quant_int4_asym(d_m4,d_m4_sz,d_gate,d_up,I,st),"sw4");
        // Down
        die(blackwell::kernels::gemv_int4_asym_batched(d_x,d_m4,d_m4_sz,W4_d[l].d,W4_d[l].sc,I,H,1,st),"d4");
        // MLP residual
        die(blackwell::kernels::vector_add_fp32(d_x,d_x,d_res,H,st),"rm4");
        die(cudaMemcpy(s4[l].data(),d_x,H*4,cudaMemcpyDeviceToHost),"cp4");
        if((l+1)%7==0||l+1==NL)printf("  l %d/%d\n",l+1,NL);
    }

    // ────────────────── Per-layer SNR ──────────────────
    printf("\n── Per-Layer SNR (Asymmetric INT4 vs INT8) ──\n");
    printf("  %-6s  %12s  %12s  %8s  %8s  %8s\n","Layer","MSE","RMSE","MaxErr","PSNR(dB)","Corr");
    printf("  %s\n",std::string(70,'-').c_str());

    double cum_mse=0;
    double worst_psnr=999; int worst_l=-1;

    for(int l=0;l<NL;++l){
        double mse=0,max_err=0;
        double sx=0,sy=0,sxx=0,syy=0,sxy=0;
        float max_ref=0;
        for(int i=0;i<H;++i){
            float d=s4[l][i]-s8[l][i];
            mse+=d*d;
            if(fabsf(d)>max_err)max_err=fabsf(d);
            if(fabsf(s8[l][i])>max_ref)max_ref=fabsf(s8[l][i]);
            sx+=s8[l][i];sy+=s4[l][i];
            sxx+=(double)s8[l][i]*s8[l][i];
            syy+=(double)s4[l][i]*s4[l][i];
            sxy+=(double)s4[l][i]*s8[l][i];
        }
        mse/=H; double rmse=sqrt(mse);
        double psnr=(mse>1e-20)?10*log10(max_ref*max_ref/mse):999;
        cum_mse+=mse;
        double n=H,corr=(n*sxy-sx*sy)/sqrt((n*sxx-sx*sx)*(n*syy-sy*sy));
        if(psnr<worst_psnr){worst_psnr=psnr;worst_l=l;}
        printf("  %-6d  %12.6e  %12.6e  %8.4f  %8.2f  %8.6f\n",
               l+1,mse,rmse,max_err,psnr,corr);
    }
    printf("\n── Summary ──\n");
    printf("  Worst PSNR: layer %d (%.2f dB)\n",worst_l+1,worst_psnr);
    printf("  Final cumulative MSE: %.6e\n",cum_mse);
    printf("  Final RMSE: %.6e\n",sqrt(cum_mse));

    // Top-5 final state diff
    printf("\n  Top-5 final layer diffs:\n");
    std::vector<std::pair<float,int>> sd;
    for(int i=0;i<H;i++)sd.push_back({fabsf(s4[NL-1][i]-s8[NL-1][i]),i});
    std::sort(sd.begin(),sd.end(),std::greater<>());
    for(int t=0;t<5;t++){
        int i=sd[t].second;
        printf("    idx %4d: INT8=%8.4f  INT4=%8.4f  diff=%8.4f\n",i,s8[NL-1][i],s4[NL-1][i],s4[NL-1][i]-s8[NL-1][i]);
    }

    // Cleanup
    for(int l=0;l<NL;l++){for(auto w:{&W8_q[l],&W8_k[l],&W8_v[l],&W8_o[l],&W8_g[l],&W8_u[l],&W8_d[l]}){cudaFree(w->d);cudaFree(w->sc);}}
    for(int l=0;l<NL;l++){for(auto w:{&W4_q[l],&W4_k[l],&W4_v[l],&W4_o[l],&W4_g[l],&W4_u[l],&W4_d[l]}){cudaFree(w->d);cudaFree(w->sc);}}
    for(int l=0;l<NL;l++){cudaFree(d_qn[l]);cudaFree(d_kn[l]);cudaFree(d_rn_in[l]);cudaFree(d_rn_post[l]);}
    delete[] host_e8; delete[] host_e8_sc; delete[] host_e4; delete[] host_e4_sz;
    return 0;
}