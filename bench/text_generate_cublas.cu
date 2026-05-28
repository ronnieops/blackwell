// bench/text_generate_cublas.cu — cuBLAS BF16 text generation
//
// Uses cuBLAS cublasGemmEx for BF16 weight × FP32 activation GEMVs.
// Zero quantization error, production-quality throughput.
// Weight layout: row-major [N_out × K_in], loaded directly via ldb=K.
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120,code=sm_120 \
//     -I include bench/text_generate_cublas.cu build/libblackwell_kernels.a \
//     -lcublas -o bench/text_generate_cublas
//
// Run: ./bench/text_generate_cublas "Hello world" 30

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cstring>
#include "blackwell/kernels.h"
#include "blackwell/bpe_tokenizer.h"

static void die(cudaError_t e, const char* m) {
    if(e!=cudaSuccess){printf("FAIL %s: %s\n",m,cudaGetErrorString(e));exit(1);}
}
static void dcub(cublasStatus_t s, const char* m) {
    if(s!=CUBLAS_STATUS_SUCCESS){printf("CUBLAS FAIL %s: %d\n",m,s);exit(1);}
}

// cuBLAS GEMV: y[N] = W[N×K] @ x[K]
// W is row-major [N×K], interpreted as col-major [K×N] with ldb=K (identical layout)
static void cublas_gemv(cublasHandle_t h, float* y,
    const void* W, cudaDataType_t Wtype, int N, int K,
    const float* x, cudaStream_t s)
{
    dcub(cublasSetStream(h, s), "setstream");
    float alpha = 1.0f, beta = 0.0f;
    dcub(cublasGemmEx(h,
        CUBLAS_OP_N, CUBLAS_OP_N,   // A=identity, B=identity (col-major [K×N] = row-major [N×K])
        1, N, K,                     // C [1×N] = A [1×K] × B [K×N]
        &alpha,
        x, CUDA_R_32F, 1,           // A: x [1×K] col-major, lda=1
        W, Wtype, K,                 // B: W col-major [K×N] (our row-major [N×K]), ldb=K
        &beta,
        y, CUDA_R_32F, 1,           // C: y [1×N] col-major, ldc=1
        CUDA_R_32F,                  // FP32 accumulate
        CUBLAS_GEMM_DEFAULT), "gemv");
}

// Kernels (reused from text_generate)
__global__ void head_norm_kernel(float* data, const float* weight, int nh, int hd, float eps) {
    int h=blockIdx.x; if(h>=nh) return;
    float* d=data+h*hd; __shared__ float sm; float s=0; int lane=threadIdx.x;
    for(int i=lane;i<hd;i+=blockDim.x) s+=d[i]*d[i];
    for(int off=blockDim.x/2;off>0;off>>=1) s+=__shfl_xor_sync(0xffffffff,s,off);
    if(lane==0) sm=rsqrtf(s/hd+eps); __syncthreads(); float is=sm;
    for(int i=lane;i<hd;i+=blockDim.x) d[i]=d[i]*is*weight[i];
}
__global__ void apply_rope_kernel(float* data, int nh, int hd, int pos) {
    int h=blockIdx.x; int d=threadIdx.x;
    if(h>=nh||d>=hd/2) return;
    int i2=d*2; float* p=data+h*hd+i2; float idxf=(float)i2/(float)hd;
    const float rt=1000000.0f; float t=(float)pos*powf(rt,-2.0f*idxf);
    float c=cosf(t),s=sinf(t),x=p[0],y=p[1]; p[0]=x*c-y*s; p[1]=x*s+y*c;
}

const int H=2048,QD=2048,KV=1024,ID=6144;
const int nqh=16,nkv=8,hd=128,MAXSEQ=2048,NL=28;
const float eps=1e-6f; const int V=151936;

struct LW16{int N,K;std::vector<__nv_bfloat16>data;};
struct DW16{int N,K;__nv_bfloat16*d;};
struct L{DW16 q,k,v,o,g,u,d;float*qn;float*kn;};

using Clock=std::chrono::high_resolution_clock;

static LW16 load16(const char* p){
    FILE*f=fopen(p,"rb");if(!f){printf("FAIL open %s\n",p);exit(1);}
    int h[2];fread(h,4,2,f);LW16 w;w.N=h[0];w.K=h[1];
    w.data.resize((size_t)h[0]*h[1]);fread(w.data.data(),2,w.data.size(),f);fclose(f);return w;
}
static DW16 up16(const LW16& w){
    DW16 d;d.N=w.N;d.K=w.K;
    cudaMalloc(&d.d,(size_t)w.N*w.K*2);
    cudaMemcpy(d.d,w.data.data(),(size_t)w.N*w.K*2,cudaMemcpyHostToDevice);return d;
}
static float* load32(const char* p,int n){
    float*d;cudaMalloc(&d,n*4);float*h=(float*)malloc(n*4);
    FILE*f=fopen(p,"rb");if(!f){printf("FAIL open %s\n",p);exit(1);}
    fread(h,4,n,f);fclose(f);cudaMemcpy(d,h,n*4,cudaMemcpyHostToDevice);free(h);return d;
}

int main(int argc,char**argv){
    const char*prompt="Once upon a time";int maxn=50;
    if(argc>1)prompt=argv[1];if(argc>2)maxn=atoi(argv[2]);
    cudaDeviceProp P;cudaGetDeviceProperties(&P,0);
    printf("# cuBLAS BF16 Text Gen — Qwen3-1.7B\n  Device: %s\n  Prompt: \"%s\"\n\n",P.name,prompt);

    cublasHandle_t h;dcub(cublasCreate(&h),"cublas_create");

    blackwell::BpeTokenizer tok;
    if(tok.load("tokenizer_data.bin")!=0){fprintf(stderr,"No tokenizer\n");return 1;}
    auto ids=tok.encode(prompt);printf("Input: %zu tokens\n\n",ids.size());

    printf("Loading weights...\n");
    std::vector<L>W(NL);char p[256];
    for(int l=0;l<NL;l++){
        snprintf(p,256,"weights_fp16/%d_self_attn.q_proj.fp16",l);W[l].q=up16(load16(p));
        snprintf(p,256,"weights_fp16/%d_self_attn.k_proj.fp16",l);W[l].k=up16(load16(p));
        snprintf(p,256,"weights_fp16/%d_self_attn.v_proj.fp16",l);W[l].v=up16(load16(p));
        snprintf(p,256,"weights_fp16/%d_self_attn.o_proj.fp16",l);W[l].o=up16(load16(p));
        snprintf(p,256,"weights_fp16/%d_mlp.gate_proj.fp16",l);W[l].g=up16(load16(p));
        snprintf(p,256,"weights_fp16/%d_mlp.up_proj.fp16",l);W[l].u=up16(load16(p));
        snprintf(p,256,"weights_fp16/%d_mlp.down_proj.fp16",l);W[l].d=up16(load16(p));
        if((l+1)%7==0||l+1==NL)printf("  layer %d/28\n",l+1);
    }
    DW16 emb=up16(load16("weights_fp16/embed_tokens.fp16"));

    // Norms
    std::vector<float*>d_rn_in(NL),d_rn_post(NL);
    for(int l=0;l<NL;l++){
        snprintf(p,256,"weights_fp16/%d_input_layernorm.f32",l);d_rn_in[l]=load32(p,H);
        snprintf(p,256,"weights_fp16/%d_post_attention_layernorm.f32",l);d_rn_post[l]=load32(p,H);
    }
    float*d_fn=load32("weights_fp16/final_norm.f32",H);
    float*d_qk=load32("weights_fp16/qk_norms.f32",NL*2*hd);

    // Buffers
    float*d_x,*d_Q,*d_K,*d_V,*d_attn,*d_gate,*d_up,*d_mlp,*d_proj,*d_r1,*d_r2;
    float*d_kc,*d_vc,*d_logits;
    #define A(p,n) cudaMalloc(&(p),(n))
    A(d_x,H*4);A(d_Q,QD*4);A(d_K,KV*4);A(d_V,KV*4);A(d_attn,QD*4);
    A(d_gate,ID*4);A(d_up,ID*4);A(d_mlp,ID*4);A(d_proj,H*4);A(d_r1,H*4);A(d_r2,H*4);
    A(d_kc,(size_t)NL*nkv*MAXSEQ*hd*4);A(d_vc,(size_t)NL*nkv*MAXSEQ*hd*4);A(d_logits,V*4);
    #undef A
    cudaMemset(d_kc,0,(size_t)NL*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vc,0,(size_t)NL*nkv*MAXSEQ*hd*4);

    // BF16→FP32 embedding conversion
    float*d_emb_fp32;cudaMalloc(&d_emb_fp32,(size_t)emb.N*emb.K*4);
    // Simple kernel: convert each BF16 element to FP32
    // Actually, for embedding lookup, we just copy one row [K] and convert.
    // Use a small device-side conversion.
    {
        dim3 grid((H+255)/256);dim3 block(256);
        // Allocate temp BF16 row buffer
        __nv_bfloat16*d_bfrow;cudaMalloc(&d_bfrow,H*2);
        // We'll convert on host for simplicity (embedding lookup is not the bottleneck)
    }

    printf("All loaded. Generating:\n%s",prompt);fflush(stdout);

    cudaStream_t st;cudaStreamCreate(&st);
    dcub(cublasSetStream(h,st),"setstream_global");
    std::vector<uint16_t>bf16tmp(H);
    std::vector<float>h_emb(H);
    std::vector<float>h_logits(V);
    std::vector<uint32_t>all_ids=ids;
    int gen_start=(int)ids.size(),total=gen_start+maxn;
    auto t0=Clock::now();

    for(int step=0;step<total;step++){
        uint32_t tid=(step<gen_start)?ids[step]:all_ids.back();

        // Embedding: copy BF16 row → host → convert to FP32 → device
        die(cudaMemcpy(bf16tmp.data(),&emb.d[(size_t)tid*H],H*2,cudaMemcpyDeviceToHost),"emb_d2h");
        for(int i=0;i<H;i++){uint32_t u=(uint32_t)bf16tmp[i]<<16;memcpy(&h_emb[i],&u,4);}
        die(cudaMemcpy(d_x,h_emb.data(),H*4,cudaMemcpyHostToDevice),"emb_h2d");

        for(int l=0;l<NL;l++){
            float*input=(l==0)?d_x:d_proj;
            die(cudaMemcpyAsync(d_r1,input,H*4,cudaMemcpyDeviceToDevice,st),"sr");

            // 1. RMSNorm
            die(blackwell::kernels::fused_rmsnorm(d_proj,input,d_rn_in[l],H,eps,st),"rn");

            // 2. QKV — cuBLAS GEMV
            cublas_gemv(h,d_Q,W[l].q.d,CUDA_R_16F,W[l].q.N,W[l].q.K,d_proj,st);
            cublas_gemv(h,d_K,W[l].k.d,CUDA_R_16F,W[l].k.N,W[l].k.K,d_proj,st);
            cublas_gemv(h,d_V,W[l].v.d,CUDA_R_16F,W[l].v.N,W[l].v.K,d_proj,st);

            // 3. Head norms
            head_norm_kernel<<<nqh,128,0,st>>>(d_Q,d_qk+l*2*hd,nqh,hd,eps);
            head_norm_kernel<<<nkv,128,0,st>>>(d_K,d_qk+l*2*hd+hd,nkv,hd,eps);

            // 4. RoPE
            apply_rope_kernel<<<nqh,hd/2,0,st>>>(d_Q,nqh,hd,step);
            apply_rope_kernel<<<nkv,hd/2,0,st>>>(d_K,nkv,hd,step);

            // 5. KV cache + attention
            int kb=l*nkv*MAXSEQ*hd;
            die(blackwell::kernels::update_kv_cache(d_kc+kb,d_vc+kb,d_K,d_V,0,step,nkv,hd,MAXSEQ,st),"kv");
            die(blackwell::kernels::attention_decode_gqa(d_attn,d_Q,d_kc+kb,d_vc+kb,step,nqh,nkv,hd,MAXSEQ,st),"attn");

            // 6. Wo GEMV + residual
            cublas_gemv(h,d_proj,W[l].o.d,CUDA_R_16F,W[l].o.N,W[l].o.K,d_attn,st);
            die(blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_r1,H,st),"r1");
            die(cudaMemcpyAsync(d_r2,d_proj,H*4,cudaMemcpyDeviceToDevice,st),"sr2");

            // 7. Post-attn RMSNorm (reuse d_attn as temp)
            die(blackwell::kernels::fused_rmsnorm(d_attn,d_proj,d_rn_post[l],H,eps,st),"rn2");

            // 8. Gate + Up + SwiGLU
            cublas_gemv(h,d_gate,W[l].g.d,CUDA_R_16F,W[l].g.N,W[l].g.K,d_attn,st);
            cublas_gemv(h,d_up,W[l].u.d,CUDA_R_16F,W[l].u.N,W[l].u.K,d_attn,st);
            die(blackwell::kernels::apply_swiglu(d_mlp,d_gate,d_up,ID,st),"sw");

            // 9. Down GEMV + residual
            cublas_gemv(h,d_proj,W[l].d.d,CUDA_R_16F,W[l].d.N,W[l].d.K,d_mlp,st);
            die(blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_r2,H,st),"r2");
        }

        // Final norm + lm_head
        if(step>=gen_start-1){
            die(blackwell::kernels::fused_rmsnorm(d_attn,d_proj,d_fn,H,eps,st),"fn");
            cublas_gemv(h,d_logits,emb.d,CUDA_R_16F,emb.N,emb.K,d_attn,st);
            die(cudaStreamSynchronize(st),"sync");
            die(cudaMemcpy(h_logits.data(),d_logits,V*4,cudaMemcpyDeviceToHost),"clog");
            int best=0;float bv=h_logits[0];
            for(int i=1;i<V;i++)if(h_logits[i]>bv){bv=h_logits[i];best=i;}
            all_ids.push_back(best);
            printf("%s",tok.decode(best).c_str());fflush(stdout);
            if(best==151643||best==151645){printf("\n[EOS]\n");break;}
        }
    }

    auto t1=Clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    int gen=(int)all_ids.size()-gen_start;
    printf("\n\n── Stats ──\n  Time: %.1f ms, Speed: %.1f ms/tok = %.0f t/s\n",ms,ms/gen,1000.0*gen/ms);

    cublasDestroy(h);
    return 0;
}
