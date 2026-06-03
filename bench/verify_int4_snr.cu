// bench/verify_int4_snr.cu — Per-layer INT4 vs INT8 SNR comparison
//
// Runs both INT8 and INT4 pipelines for a single token, captures hidden
// state at each layer boundary, computes per-element error metrics.
//
// Build:
//   CUDACXX=/usr/local/cuda-13.3/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/verify_int4_snr.cu build/libblackwell_kernels.a \
//     -o bench/verify_int4_snr
//
// Run: ./bench/verify_int4_snr [token_id]

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

// ── Model constants (Qwen3-1.7B) ──
const int H=2048, Q=2048, KV=1024, I=6144;
const int nqh=16, nkv=8, hd=128, MAXSEQ=4096;
const float eps=1e-6f;
const int V=151936;
const int NL=28;

// ── Helpers: INT8 upload ──
struct DW8 { int8_t* d; float* sc; };
static DW8 upload_w8(const char* prefix) {
    char p[256]; snprintf(p,256,"%s.int8_t",prefix);
    FILE* f=fopen(p,"rb"); int h[5]; fread(h,4,5,f);
    DW8 dw;
    size_t ds=(size_t)h[0]*h[1];
    int8_t* td=new int8_t[ds]; fread(td,1,ds,f); fclose(f);
    cudaMalloc(&dw.d,ds); cudaMemcpy(dw.d,td,ds,cudaMemcpyHostToDevice); delete[] td;
    snprintf(p,256,"%s.scale_t",prefix); f=fopen(p,"rb"); fread(h,4,5,f);
    size_t ss=(size_t)h[3]*h[4];
    float* ts=new float[ss]; fread(ts,4,ss,f); fclose(f);
    cudaMalloc(&dw.sc,ss*4); cudaMemcpy(dw.sc,ts,ss*4,cudaMemcpyHostToDevice); delete[] ts;
    return dw;
}

// ── Helpers: INT4 upload ──
struct DW4 { uint8_t* d; float* sc; };
static DW4 upload_w4_sc_zero(const char* prefix) {
    char p[256]; snprintf(p,256,"%s.int4_t",prefix);
    FILE* f=fopen(p,"rb"); int h[5]; fread(h,4,5,f);
    DW4 dw;
    size_t ds=(size_t)h[0]*h[1]/2;
    uint8_t* td=new uint8_t[ds]; fread(td,1,ds,f); fclose(f);
    cudaMalloc(&dw.d,ds); cudaMemcpy(dw.d,td,ds,cudaMemcpyHostToDevice); delete[] td;
    snprintf(p,256,"%s.sc_zero_t",prefix); f=fopen(p,"rb"); fread(h,4,5,f);
    size_t ss=(size_t)h[3]*h[4]*2;  // 2 floats per block (scale + zero)
    float* ts=new float[ss]; fread(ts,4,ss,f); fclose(f);
    cudaMalloc(&dw.sc,ss*4); cudaMemcpy(dw.sc,ts,ss*4,cudaMemcpyHostToDevice); delete[] ts;
    return dw;
}

// ── Head norm + RoPE kernels (exact match with text_generate_int4) ──
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

// ── INT4 embedding helper ──
static void dequant_embed_row_int4(float* out, int token,
    const uint8_t* host_w, const float* host_sc, int K)
{
    int kblocks=K/16;
    for(int b=0;b<kblocks;++b){
        float sc=host_sc[token*kblocks+b];
        for(int i=0;i<16;++i){
            size_t byte_idx=(size_t)token*K/2+(size_t)b*8+i/2;
            uint8_t byte=host_w[byte_idx];
            int nib=(i&1)?((byte>>4)&0x0F):(byte&0x0F);
            int val=nib-8;
            out[b*16+i]=(float)val*sc;
        }
    }
}

// ── INT8 embedding helper ──
static void dequant_embed_row_int8(float* out, int token,
    const int8_t* host_w, const float* host_sc, int K)
{
    for(int i=0;i<K;i++){
        out[i]=(float)host_w[token*K+i]*host_sc[token*(K/16)+i/16];
    }
}

int main(int argc, char** argv) {
    int token_id = 42;  // default test token
    if(argc>1) token_id=atoi(argv[1]);

    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    printf("# INT4 vs INT8 Per-Layer SNR — Qwen3-1.7B\n");
    printf("  Device: %s\n", P.name);
    printf("  Token: %d\n\n", token_id);

    // ── Allocate device buffers (shared between INT8 + INT4) ──
    float *d_x, *d_xi_f, *d_res;
    float *d_Q, *d_K, *d_V, *d_attn;
    uint8_t *d_x_i4; float *d_x_i4_sc;
    int8_t *d_x_i8, *d_attn_i8, *d_mlp_i8;
    float *d_proj, *d_gate, *d_up;
    uint8_t *d_mlp_i4; float *d_mlp_i4_sc;
    uint8_t *d_attn_i4; float *d_attn_i4_sc;
    float *d_fn, *d_kc, *d_vc;

    #define AL(p,n){cudaError_t _e=cudaMalloc(&(p),(n));\
        if(_e!=cudaSuccess){printf("FAIL malloc %s: %s\n",#p,cudaGetErrorString(_e));die(_e,#p);}}
    AL(d_x,H*4); AL(d_xi_f,H*4); AL(d_res,H*4);
    AL(d_Q,Q*4); AL(d_K,KV*4); AL(d_V,KV*4); AL(d_attn,Q*4);
    AL(d_x_i4,H/2); AL(d_x_i4_sc,(H/16)*4);
    AL(d_x_i8,H); AL(d_attn_i8,Q); AL(d_mlp_i8,I);
    AL(d_proj,H*4); AL(d_gate,I*4); AL(d_up,I*4);
    AL(d_mlp_i4,I/2); AL(d_mlp_i4_sc,(I/16)*4);
    AL(d_attn_i4,Q/2); AL(d_attn_i4_sc,(Q/16)*4);
    AL(d_fn,H*4);
    AL(d_kc,(size_t)NL*nkv*MAXSEQ*hd*4);
    AL(d_vc,(size_t)NL*nkv*MAXSEQ*hd*4);
    #undef AL

    // Init scale buffers to 1/7 (INT4) and 1/127 (INT8 will use per-block scales)
    float iv7=1.f/7.f;
    cudaMemcpy(d_x_i4_sc,&iv7,4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_attn_i4_sc,&iv7,4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_mlp_i4_sc,&iv7,4,cudaMemcpyHostToDevice);

    cudaStream_t st; die(cudaStreamCreate(&st),"stream");

    // ── Load INT8 weights ──
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
    }

    // ── Load INT4 weights ──
    printf("Loading INT4 weights...\n"); fflush(stdout);
    std::vector<DW4> W4_q(NL),W4_k(NL),W4_v(NL),W4_o(NL);
    std::vector<DW4> W4_g(NL),W4_u(NL),W4_d(NL);
    char p4[256];
    for(int l=0;l<NL;++l){
        snprintf(p4,256,"weights_int4_qwen3_1.7b/%d_self_attn.q_proj",l); W4_q[l]=upload_w4(p4);
        snprintf(p4,256,"weights_int4_qwen3_1.7b/%d_self_attn.k_proj",l); W4_k[l]=upload_w4(p4);
        snprintf(p4,256,"weights_int4_qwen3_1.7b/%d_self_attn.v_proj",l); W4_v[l]=upload_w4(p4);
        snprintf(p4,256,"weights_int4_qwen3_1.7b/%d_self_attn.o_proj",l); W4_o[l]=upload_w4(p4);
        snprintf(p4,256,"weights_int4_qwen3_1.7b/%d_mlp.gate_proj",l);   W4_g[l]=upload_w4(p4);
        snprintf(p4,256,"weights_int4_qwen3_1.7b/%d_mlp.up_proj",l);     W4_u[l]=upload_w4(p4);
        snprintf(p4,256,"weights_int4_qwen3_1.7b/%d_mlp.down_proj",l);  W4_d[l]=upload_w4(p4);
    }

    // ── Norm weights (shared between INT8 and INT4) ──
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

    // ── Embeddings ──
    // Load INT8 embeddings directly to host (for host-side dequant)
    int8_t* host_emb8_d=new int8_t[(size_t)V*H];
    float* host_emb8_sc=new float[(size_t)V*(H/16)];
    {
        FILE*f=fopen("weights_int8_bf16/embed_tokens.int8_t","rb");
        int h[5]; fread(h,4,5,f);
        fread(host_emb8_d,1,(size_t)h[0]*h[1],f); fclose(f);
        f=fopen("weights_int8_bf16/embed_tokens.scale_t","rb");
        fread(h,4,5,f); fread(host_emb8_sc,4,(size_t)h[3]*h[4],f); fclose(f);
    }

    // INT4 embedding (host side)
    uint8_t* host_emb4_d=new uint8_t[(size_t)V*H/2];
    float* host_emb4_sc=new float[(size_t)V*(H/16)];
    {
        FILE*f=fopen("weights_int4_qwen3_1.7b/embed_tokens.int4_t","rb");
        int h[5]; fread(h,4,5,f);
        fread(host_emb4_d,1,(size_t)h[0]*h[1]/2,f); fclose(f);
        f=fopen("weights_int4_qwen3_1.7b/embed_tokens.scale_t","rb");
        fread(h,4,5,f); fread(host_emb4_sc,4,(size_t)h[3]*h[4],f); fclose(f);
    }

    printf("All weights loaded.\n\n");

    // ── Host buffers for layer-by-layer hidden states ──
    std::vector<std::vector<float>> int8_states(NL, std::vector<float>(H));
    std::vector<std::vector<float>> int4_states(NL, std::vector<float>(H));
    std::vector<float> hbuf(H);

    // ────────────────────────────────────────────────────────────────
    // RUN INT8 PIPELINE — save hidden state after each layer
    // ────────────────────────────────────────────────────────────────
    printf("── INT8 pipeline ──\n"); fflush(stdout);

    // Embedding (INT8)
    dequant_embed_row_int8(hbuf.data(), token_id, host_emb8_d, host_emb8_sc, H);
    die(cudaMemcpyAsync(d_x, hbuf.data(), H*4, cudaMemcpyHostToDevice, st), "emb8_cpy");

    for(int l=0; l<NL; ++l) {
        // Save residual
        die(cudaMemcpyAsync(d_res, d_x, H*4, cudaMemcpyDeviceToDevice, st), "save_res8");

        // RMSNorm input
        die(blackwell::kernels::fused_rmsnorm(d_xi_f, d_x, d_rn_in[l], H, eps, st), "rmsnorm_in8");

        // Quantize → INT8, then QKV
        die(blackwell::kernels::quantize_int8(d_x_i8, d_x_i4_sc, d_xi_f, H, st), "quant_in8");
        die(blackwell::kernels::gemv_int8_warp(d_Q, d_x_i8, d_x_i4_sc, W8_q[l].d, W8_q[l].sc, H, Q, st), "q8");
        die(blackwell::kernels::gemv_int8_warp(d_K, d_x_i8, d_x_i4_sc, W8_k[l].d, W8_k[l].sc, H, KV, st), "k8");
        die(blackwell::kernels::gemv_int8_warp(d_V, d_x_i8, d_x_i4_sc, W8_v[l].d, W8_v[l].sc, H, KV, st), "v8");

        // Head norms + RoPE
        head_norm_kernel<<<nqh,128,0,st>>>(d_Q,d_qn[l],nqh,hd,eps);
        die(cudaGetLastError(),"hn_q8");
        head_norm_kernel<<<nkv,128,0,st>>>(d_K,d_kn[l],nkv,hd,eps);
        die(cudaGetLastError(),"hn_k8");
        apply_rope_kernel<<<nqh,hd/2,0,st>>>(d_Q,nqh,hd,0);
        die(cudaGetLastError(),"rope_q8");
        apply_rope_kernel<<<nkv,hd/2,0,st>>>(d_K,nkv,hd,0);
        die(cudaGetLastError(),"rope_k8");

        // KV cache + attention
        int kb=l*nkv*MAXSEQ*hd;
        die(blackwell::kernels::update_kv_cache(d_kc+kb,d_vc+kb,d_K,d_V,0,0,nkv,hd,MAXSEQ,st),"kv8");
        die(blackwell::kernels::attention_decode_gqa(d_attn,d_Q,d_kc+kb,d_vc+kb,0,nqh,nkv,hd,MAXSEQ,st),"attn8");

        // Wo projection + residual 1
        die(blackwell::kernels::quantize_int8(d_attn_i8, d_attn_i4_sc, d_attn, Q, st), "quant_attn8");
        die(blackwell::kernels::gemv_int8_warp(d_proj, d_attn_i8, d_attn_i4_sc, W8_o[l].d, W8_o[l].sc, Q, H, st), "o8");
        die(blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_res, H, st), "res1_8");

        // Save post-attention state for MLP residual
        die(cudaMemcpyAsync(d_res, d_proj, H*4, cudaMemcpyDeviceToDevice, st), "save_res2_8");

        // Post-attention RMSNorm
        die(blackwell::kernels::fused_rmsnorm(d_xi_f, d_proj, d_rn_post[l], H, eps, st), "rmsnorm_post8");

        // Gate + Up + SwiGLU
        die(blackwell::kernels::quantize_int8(d_mlp_i8, d_mlp_i4_sc, d_xi_f, H, st), "quant_mlp8");
        die(blackwell::kernels::gemv_int8_warp(d_gate, d_mlp_i8, d_mlp_i4_sc, W8_g[l].d, W8_g[l].sc, H, I, st), "gate8");
        die(blackwell::kernels::gemv_int8_warp(d_up, d_mlp_i8, d_mlp_i4_sc, W8_u[l].d, W8_u[l].sc, H, I, st), "up8");
        die(blackwell::kernels::apply_swiglu(d_gate, d_gate, d_up, I, st), "swiglu8");

        // Down projection + residual 2
        die(blackwell::kernels::quantize_int8(d_mlp_i8, d_mlp_i4_sc, d_gate, I, st), "quant_down8");
        die(blackwell::kernels::gemv_int8_warp(d_x, d_mlp_i8, d_mlp_i4_sc, W8_d[l].d, W8_d[l].sc, I, H, st), "down8");
        die(blackwell::kernels::vector_add_fp32(d_x, d_x, d_res, H, st), "res2_8");

        // Copy final hidden state to host
        die(cudaMemcpy(int8_states[l].data(), d_x, H*4, cudaMemcpyDeviceToHost), "cpy8");
        if((l+1)%7==0||l+1==NL) printf("  layer %d/%d ✅\n",l+1,NL);
    }

    // ────────────────────────────────────────────────────────────────
    // RUN INT4 PIPELINE — save hidden state after each layer
    // ────────────────────────────────────────────────────────────────
    printf("\n── INT4 pipeline ──\n"); fflush(stdout);

    // Clear KV cache
    cudaMemset(d_kc,0,(size_t)NL*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vc,0,(size_t)NL*nkv*MAXSEQ*hd*4);

    // Embedding (INT4)
    dequant_embed_row_int4(hbuf.data(), token_id, host_emb4_d, host_emb4_sc, H);
    die(cudaMemcpyAsync(d_x, hbuf.data(), H*4, cudaMemcpyHostToDevice, st), "emb4_cpy");

    for(int l=0; l<NL; ++l) {
        die(cudaMemcpyAsync(d_res, d_x, H*4, cudaMemcpyDeviceToDevice, st), "save_res4");

        die(blackwell::kernels::fused_rmsnorm(d_xi_f, d_x, d_rn_in[l], H, eps, st), "rmsnorm_in4");

        die(blackwell::kernels::quantize_int4(d_x_i4, d_x_i4_sc, d_xi_f, H, st), "quant_in4");
        die(blackwell::kernels::gemv_int4_batched(d_Q, (const uint8_t*)d_x_i4, d_x_i4_sc, W4_q[l].d, W4_q[l].sc, H, Q, 1, st), "q4");
        die(blackwell::kernels::gemv_int4_batched(d_K, (const uint8_t*)d_x_i4, d_x_i4_sc, W4_k[l].d, W4_k[l].sc, H, KV, 1, st), "k4");
        die(blackwell::kernels::gemv_int4_batched(d_V, (const uint8_t*)d_x_i4, d_x_i4_sc, W4_v[l].d, W4_v[l].sc, H, KV, 1, st), "v4");

        head_norm_kernel<<<nqh,128,0,st>>>(d_Q,d_qn[l],nqh,hd,eps);
        die(cudaGetLastError(),"hn_q4");
        head_norm_kernel<<<nkv,128,0,st>>>(d_K,d_kn[l],nkv,hd,eps);
        die(cudaGetLastError(),"hn_k4");
        apply_rope_kernel<<<nqh,hd/2,0,st>>>(d_Q,nqh,hd,0);
        die(cudaGetLastError(),"rope_q4");
        apply_rope_kernel<<<nkv,hd/2,0,st>>>(d_K,nkv,hd,0);
        die(cudaGetLastError(),"rope_k4");

        int kb=l*nkv*MAXSEQ*hd;
        die(blackwell::kernels::update_kv_cache(d_kc+kb,d_vc+kb,d_K,d_V,0,0,nkv,hd,MAXSEQ,st),"kv4");
        die(blackwell::kernels::attention_decode_batched_gqa(
            d_attn,d_Q,d_kc,d_vc,0,nqh,nkv,hd,MAXSEQ,1,
            (size_t)NL*nkv*MAXSEQ*hd,kb,st),"attn4");

        die(blackwell::kernels::quantize_int4(d_attn_i4, d_attn_i4_sc, d_attn, Q, st), "quant_attn4");
        die(blackwell::kernels::gemv_int4_batched(d_proj, (const uint8_t*)d_attn_i4, d_attn_i4_sc, W4_o[l].d, W4_o[l].sc, Q, H, 1, st), "o4");
        die(blackwell::kernels::vector_add_fp32(d_x, d_proj, d_res, H, st), "res1_4");

        die(cudaMemcpyAsync(d_res, d_x, H*4, cudaMemcpyDeviceToDevice, st), "save_res2_4");
        die(blackwell::kernels::fused_rmsnorm(d_xi_f, d_x, d_rn_post[l], H, eps, st), "rmsnorm_post4");
        die(blackwell::kernels::quantize_int4(d_x_i4, d_x_i4_sc, d_xi_f, H, st), "quant_mlp4");
        die(blackwell::kernels::gemv_int4_batched(d_gate, (const uint8_t*)d_x_i4, d_x_i4_sc, W4_g[l].d, W4_g[l].sc, H, I, 1, st), "gate4");
        die(blackwell::kernels::gemv_int4_batched(d_up, (const uint8_t*)d_x_i4, d_x_i4_sc, W4_u[l].d, W4_u[l].sc, H, I, 1, st), "up4");
        die(blackwell::kernels::fused_swiglu_quant_int4(d_mlp_i4, d_mlp_i4_sc, d_gate, d_up, I, st), "swiglu4");
        die(blackwell::kernels::gemv_int4_batched(d_proj, (const uint8_t*)d_mlp_i4, d_mlp_i4_sc, W4_d[l].d, W4_d[l].sc, I, H, 1, st), "down4");
        die(blackwell::kernels::vector_add_fp32(d_x, d_proj, d_res, H, st), "res2_4");

        die(cudaMemcpy(int4_states[l].data(), d_x, H*4, cudaMemcpyDeviceToHost), "cpy4");
        if((l+1)%7==0||l+1==NL) printf("  layer %d/%d ✅\n",l+1,NL);
    }

    // ────────────────────────────────────────────────────────────────
    // COMPUTE per-layer SNR
    // ────────────────────────────────────────────────────────────────
    printf("\n── Per-Layer SNR (INT4 vs INT8) ──\n");
    printf("  %-6s  %12s  %12s  %8s  %8s  %8s  %s\n",
           "Layer", "MSE", "RMSE", "MaxErr", "PSNR(dB)", "Corr", "CumulMSE");
    printf("  %s\n", std::string(75,'-').c_str());

    double cumulative_mse = 0.0;
    double worst_psnr = 999.0;
    int worst_layer = -1;

    for(int l=0; l<NL; ++l) {
        double mse = 0.0;
        double max_err = 0.0;
        double s_x = 0.0, s_y = 0.0, s_xx = 0.0, s_yy = 0.0, s_xy = 0.0;
        float max_ref = 0.0f;

        for(int i=0; i<H; ++i) {
            float diff = int4_states[l][i] - int8_states[l][i];
            mse += (double)diff * diff;
            if(fabsf(diff) > max_err) max_err = fabsf(diff);
            if(fabsf(int8_states[l][i]) > max_ref) max_ref = fabsf(int8_states[l][i]);
            s_x += int8_states[l][i];
            s_y += int4_states[l][i];
            s_xx += (double)int8_states[l][i] * int8_states[l][i];
            s_yy += (double)int4_states[l][i] * int4_states[l][i];
            s_xy += (double)int8_states[l][i] * int4_states[l][i];
        }

        mse /= H;
        double rmse = sqrt(mse);
        double psnr = (mse > 1e-20) ? 10.0 * log10((double)max_ref * max_ref / mse) : 999.0;
        cumulative_mse += mse;

        // Pearson correlation
        double n = H;
        double denom = sqrt((n*s_xx - s_x*s_x) * (n*s_yy - s_y*s_y));
        double corr = (denom > 1e-20) ? (n*s_xy - s_x*s_y) / denom : 0.0;

        if(psnr < worst_psnr) { worst_psnr = psnr; worst_layer = l; }

        printf("  %-6d  %12.6e  %12.6e  %8.4f  %8.2f  %8.6f  %12.6e\n",
               l+1, mse, rmse, max_err, psnr, corr, cumulative_mse);
    }

    printf("\n── Summary ──\n");
    printf("  Worst PSNR: layer %d (%.2f dB)\n", worst_layer+1, worst_psnr);
    printf("  Final cumulative MSE (28 layers): %.6e\n", cumulative_mse);
    printf("  Final RMSE: %.6e\n", sqrt(cumulative_mse));

    // Token-by-token analysis of final layer (lm_head would produce)
    printf("\n  Top-5 logit diff indicators (last layer):\n");
    printf("  INT8 top 5 magnitude positions:\n");
    // Find top 5 largest values in INT8 final state
    std::vector<std::pair<float,int>> sorted;
    for(int i=0;i<H;i++) sorted.push_back({fabsf(int8_states[NL-1][i]), i});
    std::sort(sorted.begin(), sorted.end(), std::greater<>());
    for(int t=0;t<5;t++){
        int idx=sorted[t].second;
        float i8=int8_states[NL-1][idx], i4=int4_states[NL-1][idx];
        printf("    idx %5d: INT8=%8.4f  INT4=%8.4f  diff=%8.4f\n", idx, i8, i4, i4-i8);
    }

    // Cleanup
    for(auto& w : W8_q) { cudaFree(w.d); cudaFree(w.sc); }
    for(auto& w : W8_k) { cudaFree(w.d); cudaFree(w.sc); }
    for(auto& w : W8_v) { cudaFree(w.d); cudaFree(w.sc); }
    for(auto& w : W8_o) { cudaFree(w.d); cudaFree(w.sc); }
    for(auto& w : W8_g) { cudaFree(w.d); cudaFree(w.sc); }
    for(auto& w : W8_u) { cudaFree(w.d); cudaFree(w.sc); }
    for(auto& w : W8_d) { cudaFree(w.d); cudaFree(w.sc); }
    for(auto& w : W4_q) { cudaFree(w.d); cudaFree(w.sc); }
    for(auto& w : W4_k) { cudaFree(w.d); cudaFree(w.sc); }
    for(auto& w : W4_v) { cudaFree(w.d); cudaFree(w.sc); }
    for(auto& w : W4_o) { cudaFree(w.d); cudaFree(w.sc); }
    for(auto& w : W4_g) { cudaFree(w.d); cudaFree(w.sc); }
    for(auto& w : W4_u) { cudaFree(w.d); cudaFree(w.sc); }
    for(auto& w : W4_d) { cudaFree(w.d); cudaFree(w.sc); }
    for(int l=0;l<NL;l++){cudaFree(d_qn[l]);cudaFree(d_kn[l]);cudaFree(d_rn_in[l]);cudaFree(d_rn_post[l]);}
    delete[] host_emb8_d; delete[] host_emb8_sc;
    delete[] host_emb4_d; delete[] host_emb4_sc;

    return 0;
}