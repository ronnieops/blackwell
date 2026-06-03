// bench/test_asym_int4_gemv.cu — End-to-end asymmetric INT4 GEMV correctness
//
// Single layer, non-fused path:
//   quantize_int4_asym → gemv_int4_asym_batched → vector_add → rmsnorm → quantize_int4_asym
//
// Compares final output vs INT8 reference for same inputs.
// Uses INT4 asymmetric weights from weights_int4_qwen3_1.7b_asym/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <numeric>
#include "blackwell/kernels.h"

static void die(cudaError_t e, const char* m) {
    if(e!=cudaSuccess){printf("FAIL %s: %s\n",m,cudaGetErrorString(e));exit(1);}
}

const int H=2048, Q=2048, KV=1024, I=6144;
const int nqh=16, nkv=8, hd=128, MAXSEQ=4096;
const float eps=1e-6f;

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
    size_t ss=(size_t)h[3]*h[4]*2; // scale+zero pairs
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

int main() {
    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    printf("# Asymmetric INT4 GEMV Correctness — Qwen3-1.7B\n");
    printf("  Device: %s\n\n", P.name);

    const int L = 0; // test layer 0

    float *d_x8, *d_xi_f, *d_res;
    float *d_Q, *d_K, *d_V, *d_attn;
    uint8_t *d_x4; float *d_x4_sz;
    int8_t *d_xi8, *d_ai8, *d_mi8;
    float *d_proj, *d_gate, *d_up;
    uint8_t *d_m4; float *d_m4_sz;
    uint8_t *d_a4; float *d_a4_sz;
    float *d_kc, *d_vc;

    #define AL(p,n){cudaError_t _e=cudaMalloc(&(p),(n));\
        if(_e!=cudaSuccess){printf("FAIL malloc %s: %s\n",#p,cudaGetErrorString(_e));die(_e,#p);}}
    AL(d_x8,H*4); AL(d_xi_f,H*4); AL(d_res,H*4);
    AL(d_Q,Q*4); AL(d_K,KV*4); AL(d_V,KV*4); AL(d_attn,Q*4);
    AL(d_x4,H/2); AL(d_x4_sz,2*(H/16)*4);
    AL(d_xi8,H); AL(d_ai8,Q); AL(d_mi8,I);
    AL(d_proj,H*4); AL(d_gate,I*4); AL(d_up,I*4);
    AL(d_m4,I/2); AL(d_m4_sz,2*(I/16)*4);
    AL(d_a4,Q/2); AL(d_a4_sz,2*(Q/16)*4);
    AL(d_kc,(size_t)1*nkv*MAXSEQ*hd*4);  // just 1 layer
    AL(d_vc,(size_t)1*nkv*MAXSEQ*hd*4);
    #undef AL

    cudaStream_t st; die(cudaStreamCreate(&st),"stream");

    // Load INT8 weights (layer 0 only)
    DW8 w8_q = upload_w8("weights_int8_bf16/0_self_attn.q_proj");
    DW8 w8_k = upload_w8("weights_int8_bf16/0_self_attn.k_proj");
    DW8 w8_v = upload_w8("weights_int8_bf16/0_self_attn.v_proj");
    DW8 w8_o = upload_w8("weights_int8_bf16/0_self_attn.o_proj");
    DW8 w8_g = upload_w8("weights_int8_bf16/0_mlp.gate_proj");
    DW8 w8_u = upload_w8("weights_int8_bf16/0_mlp.up_proj");
    DW8 w8_d = upload_w8("weights_int8_bf16/0_mlp.down_proj");

    // Load asymmetric INT4 weights (layer 0)
    DW4 w4_q = upload_w4_asym("weights_int4_qwen3_1.7b_asym/0_self_attn.q_proj");
    DW4 w4_k = upload_w4_asym("weights_int4_qwen3_1.7b_asym/0_self_attn.k_proj");
    DW4 w4_v = upload_w4_asym("weights_int4_qwen3_1.7b_asym/0_self_attn.v_proj");
    DW4 w4_o = upload_w4_asym("weights_int4_qwen3_1.7b_asym/0_self_attn.o_proj");
    DW4 w4_g = upload_w4_asym("weights_int4_qwen3_1.7b_asym/0_mlp.gate_proj");
    DW4 w4_u = upload_w4_asym("weights_int4_qwen3_1.7b_asym/0_mlp.up_proj");
    DW4 w4_d = upload_w4_asym("weights_int4_qwen3_1.7b_asym/0_mlp.down_proj");

    // Norm weights
    float* qn_h=new float[2*hd];
    {FILE*f=fopen("weights_int8_bf16/qk_norms.f32","rb");
    (void)fread(qn_h,4,2*hd,f);fclose(f);}
    float *d_qn, *d_kn;
    cudaMalloc(&d_qn,hd*4); cudaMemcpy(d_qn,qn_h,hd*4,cudaMemcpyHostToDevice);
    cudaMalloc(&d_kn,hd*4); cudaMemcpy(d_kn,qn_h+hd,hd*4,cudaMemcpyHostToDevice);
    delete[] qn_h;

    float* d_rn_in, *d_rn_post;
    {
    float* w=(float*)malloc(H*4);
    FILE*f=fopen("weights_int8_bf16/0_input_layernorm.f32","rb");
    (void)fread(w,4,H,f);fclose(f);
    cudaMalloc(&d_rn_in,H*4);cudaMemcpy(d_rn_in,w,H*4,cudaMemcpyHostToDevice);
    f=fopen("weights_int8_bf16/0_post_attention_layernorm.f32","rb");
    (void)fread(w,4,H,f);fclose(f);
    cudaMalloc(&d_rn_post,H*4);cudaMemcpy(d_rn_post,w,H*4,cudaMemcpyHostToDevice);
    free(w);
    }

    // 1. Create random test input
    float* h_x = new float[H];
    double x_sum = 0, x_ss = 0;
    for(int i=0;i<H;i++) {
        h_x[i] = ((float)rand()/RAND_MAX - 0.5f) * 3.0f;
        x_sum += h_x[i];
        x_ss += h_x[i]*h_x[i];
    }
    printf("Input x: mean=%.4f std=%.4f\n\n", x_sum/H, sqrt(x_ss/H - x_sum*x_sum/(H*H)));

    // 2. Run INT8 reference pipeline (1 layer)
    die(cudaMemcpyAsync(d_x8, h_x, H*4, cudaMemcpyHostToDevice, st), "cpy_in");

    // Save residual
    die(cudaMemcpyAsync(d_res, d_x8, H*4, cudaMemcpyDeviceToDevice, st), "save_res");

    // RMSNorm
    die(blackwell::kernels::fused_rmsnorm(d_xi_f, d_x8, d_rn_in, H, eps, st), "rn_in");
    die(blackwell::kernels::quantize_int8(d_xi8, d_x4_sz, d_xi_f, H, st), "q_in");
    die(blackwell::kernels::gemv_int8_warp(d_Q, d_xi8, d_x4_sz, w8_q.d, w8_q.sc, H, Q, st), "q");
    die(blackwell::kernels::gemv_int8_warp(d_K, d_xi8, d_x4_sz, w8_k.d, w8_k.sc, H, KV, st), "k");
    die(blackwell::kernels::gemv_int8_warp(d_V, d_xi8, d_x4_sz, w8_v.d, w8_v.sc, H, KV, st), "v");

    head_norm_kernel<<<nqh,128,0,st>>>(d_Q,d_qn,nqh,hd,eps); die(cudaGetLastError(),"hn_q");
    head_norm_kernel<<<nkv,128,0,st>>>(d_K,d_kn,nkv,hd,eps); die(cudaGetLastError(),"hn_k");
    apply_rope_kernel<<<nqh,hd/2,0,st>>>(d_Q,nqh,hd,0); die(cudaGetLastError(),"rope_q");
    apply_rope_kernel<<<nkv,hd/2,0,st>>>(d_K,nkv,hd,0); die(cudaGetLastError(),"rope_k");

    die(blackwell::kernels::update_kv_cache(d_kc,d_vc,d_K,d_V,0,0,nkv,hd,MAXSEQ,st),"kv");
    die(blackwell::kernels::attention_decode_gqa(d_attn,d_Q,d_kc,d_vc,0,nqh,nkv,hd,MAXSEQ,st),"attn");

    die(blackwell::kernels::quantize_int8(d_ai8, d_a4_sz, d_attn, Q, st), "q_attn");
    die(blackwell::kernels::gemv_int8_warp(d_proj, d_ai8, d_a4_sz, w8_o.d, w8_o.sc, Q, H, st), "o");
    die(blackwell::kernels::vector_add_fp32(d_x8, d_proj, d_res, H, st), "res1");

    // Save pre-MLP state
    die(cudaMemcpyAsync(d_res, d_x8, H*4, cudaMemcpyDeviceToDevice, st), "save_res2");

    die(blackwell::kernels::fused_rmsnorm(d_xi_f, d_x8, d_rn_post, H, eps, st), "rn_post");
    die(blackwell::kernels::quantize_int8(d_mi8, d_m4_sz, d_xi_f, H, st), "q_mlp");
    die(blackwell::kernels::gemv_int8_warp(d_gate, d_mi8, d_m4_sz, w8_g.d, w8_g.sc, H, I, st), "gate");
    die(blackwell::kernels::gemv_int8_warp(d_up, d_mi8, d_m4_sz, w8_u.d, w8_u.sc, H, I, st), "up");
    die(blackwell::kernels::apply_swiglu(d_gate, d_gate, d_up, I, st), "swiglu");
    die(blackwell::kernels::quantize_int8(d_mi8, d_m4_sz, d_gate, I, st), "q_down");
    die(blackwell::kernels::gemv_int8_warp(d_x8, d_mi8, d_m4_sz, w8_d.d, w8_d.sc, I, H, st), "down");
    die(blackwell::kernels::vector_add_fp32(d_x8, d_x8, d_res, H, st), "res2");

    float* h_ref = new float[H];
    die(cudaMemcpy(h_ref, d_x8, H*4, cudaMemcpyDeviceToHost), "cpy_ref");
    double ref_sum=0, ref_ss=0;
    for(int i=0;i<H;i++){ref_sum+=h_ref[i];ref_ss+=h_ref[i]*h_ref[i];}
    printf("INT8 reference output: mean=%.4f std=%.4f\n\n", 
           ref_sum/H, sqrt(ref_ss/H - ref_sum*ref_sum/(H*H)));

    // 3. Run asymmetric INT4 pipeline (same input, non-fused)
    cudaMemset(d_kc,0,(size_t)1*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vc,0,(size_t)1*nkv*MAXSEQ*hd*4);

    die(cudaMemcpyAsync(d_x8, h_x, H*4, cudaMemcpyHostToDevice, st), "cpy_in2");
    die(cudaMemcpyAsync(d_res, d_x8, H*4, cudaMemcpyDeviceToDevice, st), "save_res_i4");

    // RMSNorm (FP32) + quantize_int4_asym
    die(blackwell::kernels::fused_rmsnorm(d_xi_f, d_x8, d_rn_in, H, eps, st), "rn_in_i4");
    die(blackwell::kernels::quantize_int4_asym(d_x4, d_x4_sz, d_xi_f, H, st), "q_in_i4");

    // Debug: read first scale+zero back
    float h_sz[2];
    cudaMemcpy(h_sz, d_x4_sz, 8, cudaMemcpyDeviceToHost);
    printf("  x scales[0]: scale=%.6f zero=%d\n", h_sz[0], (int)h_sz[1]);

    // QKV asymmetric GEMV
    die(blackwell::kernels::gemv_int4_asym_batched(d_Q, d_x4, d_x4_sz, w4_q.d, w4_q.sc, H, Q, 1, st), "q_i4");
    die(blackwell::kernels::gemv_int4_asym_batched(d_K, d_x4, d_x4_sz, w4_k.d, w4_k.sc, H, KV, 1, st), "k_i4");
    die(blackwell::kernels::gemv_int4_asym_batched(d_V, d_x4, d_x4_sz, w4_v.d, w4_v.sc, H, KV, 1, st), "v_i4");

    head_norm_kernel<<<nqh,128,0,st>>>(d_Q,d_qn,nqh,hd,eps); die(cudaGetLastError(),"hn_q4");
    head_norm_kernel<<<nkv,128,0,st>>>(d_K,d_kn,nkv,hd,eps); die(cudaGetLastError(),"hn_k4");
    apply_rope_kernel<<<nqh,hd/2,0,st>>>(d_Q,nqh,hd,0); die(cudaGetLastError(),"rope_q4");
    apply_rope_kernel<<<nkv,hd/2,0,st>>>(d_K,nkv,hd,0); die(cudaGetLastError(),"rope_k4");

    die(blackwell::kernels::update_kv_cache(d_kc,d_vc,d_K,d_V,0,0,nkv,hd,MAXSEQ,st),"kv4");
    die(blackwell::kernels::attention_decode_batched_gqa(
        d_attn,d_Q,d_kc,d_vc,0,nqh,nkv,hd,MAXSEQ,1,
        (size_t)1*nkv*MAXSEQ*hd,0,st),"attn4");

    die(blackwell::kernels::quantize_int4_asym(d_a4, d_a4_sz, d_attn, Q, st), "q_attn_i4");
    die(blackwell::kernels::gemv_int4_asym_batched(d_proj, d_a4, d_a4_sz, w4_o.d, w4_o.sc, Q, H, 1, st), "o_i4");
    die(blackwell::kernels::vector_add_fp32(d_x8, d_proj, d_res, H, st), "res1_i4");

    die(cudaMemcpyAsync(d_res, d_x8, H*4, cudaMemcpyDeviceToDevice, st), "save_res2_i4");
    die(blackwell::kernels::fused_rmsnorm(d_xi_f, d_x8, d_rn_post, H, eps, st), "rn_post_i4");
    die(blackwell::kernels::quantize_int4_asym(d_x4, d_x4_sz, d_xi_f, H, st), "q_mlp_i4");
    die(blackwell::kernels::gemv_int4_asym_batched(d_gate, d_x4, d_x4_sz, w4_g.d, w4_g.sc, H, I, 1, st), "gate_i4");
    die(blackwell::kernels::gemv_int4_asym_batched(d_up, d_x4, d_x4_sz, w4_u.d, w4_u.sc, H, I, 1, st), "up_i4");
    die(blackwell::kernels::fused_swiglu_quant_int4_asym(d_m4, d_m4_sz, d_gate, d_up, I, st), "swiglu_i4");
    die(blackwell::kernels::gemv_int4_asym_batched(d_x8, d_m4, d_m4_sz, w4_d.d, w4_d.sc, I, H, 1, st), "down_i4");
    die(blackwell::kernels::vector_add_fp32(d_x8, d_x8, d_res, H, st), "res2_i4");

    float* h_i4 = new float[H];
    die(cudaMemcpy(h_i4, d_x8, H*4, cudaMemcpyDeviceToHost), "cpy_i4");

    // 4. Compare
    double mse=0, max_err=0;
    for(int i=0;i<H;i++) {
        double d = h_i4[i] - h_ref[i];
        mse += d*d;
        if(fabsf(d) > max_err) max_err = fabsf(d);
    }
    mse /= H;
    double psnr = (mse>1e-20) ? 10*log10(10.0*10.0/mse) : 999;

    printf("── Single layer (L=0) comparison ──\n");
    printf("  MSE: %.6e  RMSE: %.6e  MaxErr: %.4f  PSNR: %.2f dB\n\n", mse, sqrt(mse), max_err, psnr);

    // Correlation
    double sx=0,sy=0,sxx=0,syy=0,sxy=0;
    for(int i=0;i<H;i++){sx+=h_ref[i];sy+=h_i4[i];sxx+=h_ref[i]*h_ref[i];syy+=h_i4[i]*h_i4[i];sxy+=h_i4[i]*h_ref[i];}
    double n=H,corr=(n*sxy-sx*sy)/sqrt((n*sxx-sx*sx)*(n*syy-sy*sy));
    printf("  Correlation: %.6f\n\n", corr);

    // Top differences
    printf("  Top-5 differences:\n");
    std::vector<std::pair<double,int>> diffs;
    for(int i=0;i<H;i++) diffs.push_back({fabsf(h_i4[i]-h_ref[i]),i});
    std::sort(diffs.begin(),diffs.end(),std::greater<>());
    for(int t=0;t<5;t++){
        int i=diffs[t].second;
        printf("    idx %4d: INT8=%8.4f  INT4=%8.4f  diff=%8.4f\n",i,h_ref[i],h_i4[i],h_i4[i]-h_ref[i]);
    }

    // Cleanup
    delete[] h_x; delete[] h_ref; delete[] h_i4;
    cudaFree(d_qn); cudaFree(d_kn); cudaFree(d_rn_in); cudaFree(d_rn_post);
    for(auto w : {&w8_q,&w8_k,&w8_v,&w8_o,&w8_g,&w8_u,&w8_d})
        {cudaFree(w->d);cudaFree(w->sc);}
    for(auto w : {&w4_q,&w4_k,&w4_v,&w4_o,&w4_g,&w4_u,&w4_d})
        {cudaFree(w->d);cudaFree(w->sc);}
    return 0;
}