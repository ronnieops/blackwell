// bench/text_generate_fp32.cu — cuBLAS FP32 text generation
// Uses cublasSgemv for all GEMVs. No quantization. Correct text output.
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120,code=sm_120 \
//     -I include bench/text_generate_fp32.cu build/libblackwell_kernels.a \
//     -lcublas -o bench/text_generate_fp32
//
// Run: ./bench/text_generate_fp32 "Hello world" 30

#include <cuda_runtime.h>
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

// cuBLAS SGEMV: y[N] = W[N×K] @ x[K]
// W stored row-major [N×K] = col-major [K×N] with ldb=K
static void sgemv(cublasHandle_t h, float* y, const float* W,
    int N, int K, const float* x, cudaStream_t s)
{
    dcub(cublasSetStream(h, s), "setstream");
    float alpha=1.0f, beta=0.0f;
    dcub(cublasSgemv(h, CUBLAS_OP_T, K, N, &alpha,
        W, K,      // W [K×N] col-major (our row-major [N×K]), lda=K
        x, 1,      // x [K×1]
        &beta,
        y, 1),     // y [N×1]
    "sgemv");
}

// Kernels
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
// BF16→FP32 dequant kernel (for embedding lookup)
__global__ void bf16_to_fp32(float* out, const __nv_bfloat16* in, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    out[i] = __bfloat162float(in[i]);
}

const int H=2048,QD=2048,KV=1024,ID=6144;
const int nqh=16,nkv=8,hd=128,MAXSEQ=2048,NL=28;
const float eps=1e-6f; const int V=151936;

struct LW{int N,K;std::vector<__nv_bfloat16>data;};
struct DW{int N,K;__nv_bfloat16*d; float*d_fp32;};
struct L{DW q,k,v,o,g,u,d;float*qn;float*kn;};

using Clock=std::chrono::high_resolution_clock;

static LW load_bf16(const char* p){
    FILE*f=fopen(p,"rb");if(!f){printf("FAIL open %s\n",p);exit(1);}
    int h[2];fread(h,4,2,f);LW w;w.N=h[0];w.K=h[1];
    w.data.resize((size_t)h[0]*h[1]);fread(w.data.data(),2,w.data.size(),f);fclose(f);return w;
}
static DW upload_bf16(const LW& w){
    DW d;d.N=w.N;d.K=w.K;
    cudaMalloc(&d.d,(size_t)w.N*w.K*2);
    cudaMemcpy(d.d,w.data.data(),(size_t)w.N*w.K*2,cudaMemcpyHostToDevice);
    // Also convert to FP32 for cuBLAS
    cudaMalloc(&d.d_fp32,(size_t)w.N*w.K*4);
    int total=w.N*w.K;
    bf16_to_fp32<<<(total+255)/256,256>>>(d.d_fp32,d.d,total);
    return d;
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
    printf("# cuBLAS FP32 Text Gen — Qwen3-1.7B\n  Device: %s\n  Prompt: \"%s\"\n\n",P.name,prompt);

    cublasHandle_t h;dcub(cublasCreate(&h),"cublas_create");

    blackwell::BpeTokenizer tok;
    if(tok.load("tokenizer_data.bin")!=0){fprintf(stderr,"No tokenizer\n");return 1;}
    auto ids=tok.encode(prompt);printf("Input: %zu tokens\n\n",ids.size());

    printf("Loading weights (FP32 via cuBLAS)...\n");
    std::vector<L>W(NL);char p[256];
    for(int l=0;l<NL;l++){
        snprintf(p,256,"weights_bf16/%d_self_attn.q_proj.bf16",l);W[l].q=upload_bf16(load_bf16(p));
        snprintf(p,256,"weights_bf16/%d_self_attn.k_proj.bf16",l);W[l].k=upload_bf16(load_bf16(p));
        snprintf(p,256,"weights_bf16/%d_self_attn.v_proj.bf16",l);W[l].v=upload_bf16(load_bf16(p));
        snprintf(p,256,"weights_bf16/%d_self_attn.o_proj.bf16",l);W[l].o=upload_bf16(load_bf16(p));
        snprintf(p,256,"weights_bf16/%d_mlp.gate_proj.bf16",l);W[l].g=upload_bf16(load_bf16(p));
        snprintf(p,256,"weights_bf16/%d_mlp.up_proj.bf16",l);W[l].u=upload_bf16(load_bf16(p));
        snprintf(p,256,"weights_bf16/%d_mlp.down_proj.bf16",l);W[l].d=upload_bf16(load_bf16(p));
        if((l+1)%7==0||l+1==NL)printf("  layer %d/28\n",l+1);
    }
    DW emb=upload_bf16(load_bf16("weights_bf16/embed_tokens.bf16"));

    std::vector<float*>d_rn_in(NL),d_rn_post(NL);
    for(int l=0;l<NL;l++){
        snprintf(p,256,"weights_bf16/%d_input_layernorm.f32",l);d_rn_in[l]=load32(p,H);
        snprintf(p,256,"weights_bf16/%d_post_attention_layernorm.f32",l);d_rn_post[l]=load32(p,H);
    }
    float*d_fn=load32("weights_bf16/final_norm.f32",H);
    float*d_qk=load32("weights_bf16/qk_norms.f32",NL*2*hd);

    float*d_x,*d_Q,*d_K,*d_V,*d_attn,*d_gate,*d_up,*d_mlp,*d_proj,*d_r1,*d_r2;
    float*d_kc,*d_vc,*d_logits;
    #define A(p,n) cudaMalloc(&(p),(n))
    A(d_x,H*4);A(d_Q,QD*4);A(d_K,KV*4);A(d_V,KV*4);A(d_attn,QD*4);
    A(d_gate,ID*4);A(d_up,ID*4);A(d_mlp,ID*4);A(d_proj,H*4);A(d_r1,H*4);A(d_r2,H*4);
    A(d_kc,(size_t)NL*nkv*MAXSEQ*hd*4);A(d_vc,(size_t)NL*nkv*MAXSEQ*hd*4);A(d_logits,V*4);
    #undef A
    cudaMemset(d_kc,0,(size_t)NL*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vc,0,(size_t)NL*nkv*MAXSEQ*hd*4);

    printf("All loaded. Generating:\n%s",prompt);fflush(stdout);

    cudaStream_t st;cudaStreamCreate(&st);
    dcub(cublasSetStream(h,st),"setstream_global");
    std::vector<float>h_emb(H);
    std::vector<float>h_logits(V);
    std::vector<uint32_t>all_ids=ids;
    int gen_start=(int)ids.size(),total=gen_start+maxn;
    auto t0=Clock::now();

    for(int step=0;step<total;step++){
        uint32_t tid=(step<gen_start)?ids[step]:all_ids.back();

        // Embedding lookup: dequantize BF16 row on GPU → FP32
        bf16_to_fp32<<<(H+255)/256,256,0,st>>>(d_x,&emb.d[(size_t)tid*H],H);

        for(int l=0;l<NL;l++){
            float*input=(l==0)?d_x:d_proj;
            die(cudaMemcpyAsync(d_r1,input,H*4,cudaMemcpyDeviceToDevice,st),"sr");

            die(blackwell::kernels::fused_rmsnorm(d_proj,input,d_rn_in[l],H,eps,st),"rn");

            // QKV
            sgemv(h,d_Q,W[l].q.d_fp32,W[l].q.N,W[l].q.K,d_proj,st);
            sgemv(h,d_K,W[l].k.d_fp32,W[l].k.N,W[l].k.K,d_proj,st);
            sgemv(h,d_V,W[l].v.d_fp32,W[l].v.N,W[l].v.K,d_proj,st);

            head_norm_kernel<<<nqh,128,0,st>>>(d_Q,d_qk+l*2*hd,nqh,hd,eps);
            head_norm_kernel<<<nkv,128,0,st>>>(d_K,d_qk+l*2*hd+hd,nkv,hd,eps);

            apply_rope_kernel<<<nqh,hd/2,0,st>>>(d_Q,nqh,hd,step);
            apply_rope_kernel<<<nkv,hd/2,0,st>>>(d_K,nkv,hd,step);

            int kb=l*nkv*MAXSEQ*hd;
            die(blackwell::kernels::update_kv_cache(d_kc+kb,d_vc+kb,d_K,d_V,0,step,nkv,hd,MAXSEQ,st),"kv");
            die(blackwell::kernels::attention_decode_gqa(d_attn,d_Q,d_kc+kb,d_vc+kb,step,nqh,nkv,hd,MAXSEQ,st),"attn");

            sgemv(h,d_proj,W[l].o.d_fp32,W[l].o.N,W[l].o.K,d_attn,st);
            die(blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_r1,H,st),"r1");
            die(cudaMemcpyAsync(d_r2,d_proj,H*4,cudaMemcpyDeviceToDevice,st),"sr2");

            die(blackwell::kernels::fused_rmsnorm(d_attn,d_proj,d_rn_post[l],H,eps,st),"rn2");

            sgemv(h,d_gate,W[l].g.d_fp32,W[l].g.N,W[l].g.K,d_attn,st);
            sgemv(h,d_up,W[l].u.d_fp32,W[l].u.N,W[l].u.K,d_attn,st);
            die(blackwell::kernels::apply_swiglu(d_mlp,d_gate,d_up,ID,st),"sw");

            sgemv(h,d_proj,W[l].d.d_fp32,W[l].d.N,W[l].d.K,d_mlp,st);
            die(blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_r2,H,st),"r2");
        }

        if(step>=gen_start-1){
            die(blackwell::kernels::fused_rmsnorm(d_attn,d_proj,d_fn,H,eps,st),"fn");
            sgemv(h,d_logits,emb.d_fp32,emb.N,emb.K,d_attn,st);
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
