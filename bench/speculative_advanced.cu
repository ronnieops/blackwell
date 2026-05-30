// bench/speculative_advanced.cu — Speculative decode with batched attention + CUDA Graph
//
// Uses latest pipeline: batched attention, fused unpack+pack, direct batch buffer writes.
// Measures speculation throughput vs autoregressive (M=1) baseline.
//
// Build:
//   CUDACXX=/usr/local/cuda-13.3/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/speculative_advanced.cu build/libblackwell_kernels.a \
//     -o bench/speculative_advanced

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>
#include <cmath>
#include "blackwell/kernels.h"

struct GpuTimer { cudaEvent_t s,e; GpuTimer(){cudaEventCreate(&s);cudaEventCreate(&e);} ~GpuTimer(){cudaEventDestroy(s);cudaEventDestroy(e);}
    void start(cudaStream_t st=0){cudaEventRecord(s,st);}
    float stop(cudaStream_t st=0){cudaEventRecord(e,st);cudaEventSynchronize(e);float m=0;cudaEventElapsedTime(&m,s,e);return m;}
};
struct IW { int K,N; int8_t*d; float*ds; };
static IW load_iw(const char* d, const char* n){
    char p[256]; snprintf(p,256,"%s/%s.int8_t",d,n); FILE*f=fopen(p,"rb");
    if(!f){printf("Cannot open %s\n",p);exit(1);}
    int h[5]; fread(h,4,5,f);
    IW w{h[0],h[1],nullptr,nullptr};
    std::vector<int8_t> tmp((size_t)w.K*w.N); fread(tmp.data(),1,(size_t)w.K*w.N,f); fclose(f);
    cudaMalloc(&w.d,(size_t)w.K*w.N); cudaMemcpy(w.d,tmp.data(),(size_t)w.K*w.N,cudaMemcpyHostToDevice);
    snprintf(p,256,"%s/%s.scale_t",d,n); f=fopen(p,"rb"); fread(h,4,5,f);
    size_t ns=(size_t)h[3]*h[4]; std::vector<float> tmp_s(ns); fread(tmp_s.data(),4,ns,f); fclose(f);
    cudaMalloc(&w.ds,ns*4); cudaMemcpy(w.ds,tmp_s.data(),ns*4,cudaMemcpyHostToDevice);
    return w;
}
struct LW { IW q,k,v,o,g,u,d;
    void load(const char*dir,int l){
        char b[128];
        snprintf(b,128,"%d_self_attn.q_proj",l); q=load_iw(dir,b);
        snprintf(b,128,"%d_self_attn.k_proj",l); k=load_iw(dir,b);
        snprintf(b,128,"%d_self_attn.v_proj",l); v=load_iw(dir,b);
        snprintf(b,128,"%d_self_attn.o_proj",l); o=load_iw(dir,b);
        snprintf(b,128,"%d_mlp.gate_proj",l);  g=load_iw(dir,b);
        snprintf(b,128,"%d_mlp.up_proj",l);    u=load_iw(dir,b);
        snprintf(b,128,"%d_mlp.down_proj",l);  d=load_iw(dir,b);
    }
    void free_all(){
        cudaFree(q.d);cudaFree(q.ds);cudaFree(k.d);cudaFree(k.ds);
        cudaFree(v.d);cudaFree(v.ds);cudaFree(o.d);cudaFree(o.ds);
        cudaFree(g.d);cudaFree(g.ds);cudaFree(u.d);cudaFree(u.ds);
        cudaFree(d.d);cudaFree(d.ds);
    }
};

int main(int argc, char** argv) {
    int NL=4, M=4, steps=20;
    if(argc>1)NL=atoi(argv[1]);
    if(argc>2)M=atoi(argv[2]);
    if(argc>3)steps=atoi(argv[3]);
    if(NL>28)NL=28; if(M<1)M=1; if(M>8)M=8;

    cudaDeviceProp p;cudaGetDeviceProperties(&p,0);
    printf("# Speculative Decode (Advanced) — Qwen3-1.7B\n");
    printf("Device: %s (CC %d.%d)\n",p.name,p.major,p.minor);
    printf("Layers: %d, Drafts M: %d, Steps: %d\n",NL,M,steps);

    const int H=2048,Q=2048,KV=1024,I=6144,nqh=16,nkv=8,hd=128,ms=2048;
    const float s13=1.f/3.f;
    const char* dir="weights_int8_bf16";

    printf("Loading weights...\n");
    std::vector<LW> L(NL);
    for(int i=0;i<NL;++i) L[i].load(dir,i);

    // ── Buffers ─────────────────────────────────────────────────────────────
    // Single-token buffers
    float *d_rn; cudaMalloc(&d_rn,H*4);
    std::vector<float>rn_h(H,1.f); cudaMemcpy(d_rn,rn_h.data(),H*4,cudaMemcpyHostToDevice);

    // Per-seq scratch
    float *d_res_s; int8_t *d_xi8_s; float *d_xi8s_s;
    cudaMalloc(&d_res_s,I*4); cudaMalloc(&d_xi8_s,I); cudaMalloc(&d_xi8s_s,(I/16)*4);

    // Single-token state (1 sequence)
    void *d_xfp4; float *d_xs;
    cudaMalloc(&d_xfp4,H); cudaMalloc(&d_xs,(H/16)*4);

    float *d_Q_s,*d_K_s,*d_V_s,*d_attn_s,*d_proj_s;
    float *d_gate_s,*d_up_s,*d_mlp_s;
    int8_t *d_ai8_s,*d_mi8_s; float *d_ai8s_s,*d_mi8s_s;
    cudaMalloc(&d_Q_s,Q*4); cudaMalloc(&d_K_s,KV*4); cudaMalloc(&d_V_s,KV*4);
    cudaMalloc(&d_attn_s,Q*4); cudaMalloc(&d_proj_s,H*4);
    cudaMalloc(&d_gate_s,I*4); cudaMalloc(&d_up_s,I*4); cudaMalloc(&d_mlp_s,I*4);
    cudaMalloc(&d_ai8_s,Q); cudaMalloc(&d_ai8s_s,(Q/16)*4);
    cudaMalloc(&d_mi8_s,I); cudaMalloc(&d_mi8s_s,(I/16)*4);

    // M-token batched buffers
    void** d_xfp4M = new void*[M]; float** d_xsM = new float*[M];
    for(int m=0;m<M;++m){cudaMalloc(&d_xfp4M[m],H);cudaMalloc(&d_xsM[m],(H/16)*4);}

    int8_t *d_xM; float *d_xMs;
    cudaMalloc(&d_xM,M*H); cudaMalloc(&d_xMs,M*(H/16)*4);
    float *d_QM,*d_KM,*d_VM,*d_attnM,*d_projM;
    float *d_gateM,*d_upM,*d_mlpM;
    int8_t *d_ai8M,*d_mi8M; float *d_ai8sM,*d_mi8sM;
    cudaMalloc(&d_QM,M*Q*4); cudaMalloc(&d_KM,M*KV*4); cudaMalloc(&d_VM,M*KV*4);
    cudaMalloc(&d_attnM,M*Q*4); cudaMalloc(&d_projM,M*H*4);
    cudaMalloc(&d_gateM,M*I*4); cudaMalloc(&d_upM,M*I*4); cudaMalloc(&d_mlpM,M*I*4);
    cudaMalloc(&d_ai8M,M*Q); cudaMalloc(&d_ai8sM,M*(Q/16)*4);
    cudaMalloc(&d_mi8M,M*I); cudaMalloc(&d_mi8sM,M*(I/16)*4);

    // Init scales
    float ixv=1.f/127.f;
    std::vector<float> i8s(H/16,ixv), ai8s(Q/16,ixv), mi8s(I/16,ixv);
    cudaMemcpy(d_xi8s_s,i8s.data(),(H/16)*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_ai8s_s,ai8s.data(),(Q/16)*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_mi8s_s,mi8s.data(),(I/16)*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_xMs,i8s.data(),(H/16)*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_ai8sM,ai8s.data(),(Q/16)*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_mi8sM,mi8s.data(),(I/16)*4,cudaMemcpyHostToDevice);

    // Init state
    std::vector<float> xh(H,1.f), xsh(H/16,s13);
    float *d_x32; cudaMalloc(&d_x32,H*4);
    cudaMemcpy(d_x32,xh.data(),H*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_xs,xsh.data(),(H/16)*4,cudaMemcpyHostToDevice);
    blackwell::kernels::pack_fp4(d_xfp4,d_x32,d_xs,H,0);
    for(int m=0;m<M;++m){
        cudaMemcpy(d_xsM[m],xsh.data(),(H/16)*4,cudaMemcpyHostToDevice);
        blackwell::kernels::pack_fp4(d_xfp4M[m],d_x32,d_xsM[m],H,0);
    }

    // KV cache
    float *d_kc,*d_vc;
    size_t kv_sz = (size_t)NL * nkv * ms * hd * 4;  // single-token
    size_t kv_szM = (size_t)M * NL * nkv * ms * hd * 4;  // M-token
    cudaMalloc(&d_kc,kv_sz); cudaMalloc(&d_vc,kv_sz);
    cudaMalloc(&d_kc,kv_szM); cudaMalloc(&d_vc,kv_szM);
    cudaMemset(d_kc,0,kv_szM); cudaMemset(d_vc,0,kv_szM);

    size_t kv_seq_stride = (size_t)NL * nkv * ms * hd;

    // Fill KV cache
    printf("Filling KV cache (seq=0..128)... ");
    fflush(stdout);
    int sq=128;
    auto kv_off = [&](int m, int l){ return (size_t)m * kv_seq_stride + (size_t)l * nkv * ms * hd; };
    for(int s=0;s<=sq;++s){
        for(int m=0;m<M;++m){
            for(int l=0;l<NL;++l){
                size_t ko=kv_off(m,l);
                blackwell::kernels::unpack_fp4(d_res_s,d_xfp4M[m],d_xsM[m],H,0);
                blackwell::kernels::pack_int8(d_xM+m*H,d_res_s,d_xMs+m*(H/16),H,0);
                blackwell::kernels::gemv_int8_warp(d_QM+m*Q,d_xM+m*H,d_xMs+m*(H/16),L[l].q.d,L[l].q.ds,H,Q,0);
                blackwell::kernels::gemv_int8_warp(d_KM+m*KV,d_xM+m*H,d_xMs+m*(H/16),L[l].k.d,L[l].k.ds,H,KV,0);
                blackwell::kernels::gemv_int8_warp(d_VM+m*KV,d_xM+m*H,d_xMs+m*(H/16),L[l].v.d,L[l].v.ds,H,KV,0);
                blackwell::kernels::update_kv_cache(d_kc+ko,d_vc+ko,d_KM+m*KV,d_VM+m*KV,0,s,nkv,hd,ms,0);
                blackwell::kernels::attention_decode_gqa(d_attnM+m*Q,d_QM+m*Q,d_kc+ko,d_vc+ko,s,nqh,nkv,hd,ms,0);
                blackwell::kernels::pack_int8(d_ai8M+m*Q,d_attnM+m*Q,d_ai8sM+m*(Q/16),Q,0);
                blackwell::kernels::gemv_int8_warp(d_projM+m*H,d_ai8M+m*Q,d_ai8sM+m*(Q/16),L[l].o.d,L[l].o.ds,Q,H,0);
                blackwell::kernels::unpack_fp4(d_res_s,d_xfp4M[m],d_xsM[m],H,0);
                blackwell::kernels::vector_add_fp32(d_projM+m*H,d_projM+m*H,d_res_s,H,0);
                blackwell::kernels::fused_rmsnorm_quant_int8(d_xM+m*H,d_xMs+m*(H/16),d_projM+m*H,d_rn,H,1e-6f,0);
                blackwell::kernels::fused_rmsnorm_pack(d_xfp4M[m],d_xsM[m],d_projM+m*H,d_rn,H,1e-6f,0);
                // MLP
                blackwell::kernels::unpack_fp4(d_res_s,d_xfp4M[m],d_xsM[m],H,0);
                blackwell::kernels::pack_int8(d_xM+m*H,d_res_s,d_xMs+m*(H/16),H,0);
                blackwell::kernels::gemv_int8_warp(d_gateM+m*I,d_xM+m*H,d_xMs+m*(H/16),L[l].g.d,L[l].g.ds,H,I,0);
                blackwell::kernels::gemv_int8_warp(d_upM+m*I,d_xM+m*H,d_xMs+m*(H/16),L[l].u.d,L[l].u.ds,H,I,0);
                blackwell::kernels::apply_swiglu(d_mlpM+m*I,d_gateM+m*I,d_upM+m*I,I,0);
                blackwell::kernels::pack_int8(d_mi8M+m*I,d_mlpM+m*I,d_mi8sM+m*(I/16),I,0);
                blackwell::kernels::gemv_int8_warp(d_projM+m*H,d_mi8M+m*I,d_mi8sM+m*(I/16),L[l].d.d,L[l].d.ds,I,H,0);
                blackwell::kernels::vector_add_fp32(d_projM+m*H,d_projM+m*H,d_res_s,H,0);
                blackwell::kernels::fused_rmsnorm_quant_int8(d_xM+m*H,d_xMs+m*(H/16),d_projM+m*H,d_rn,H,1e-6f,0);
                blackwell::kernels::fused_rmsnorm_pack(d_xfp4M[m],d_xsM[m],d_projM+m*H,d_rn,H,1e-6f,0);
            }
        }
    }
    printf("done\n");

    // Copy single-token state from seq 0
    cudaMemcpy(d_xfp4,d_xfp4M[0],H,cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_xs,d_xsM[0],(H/16)*4,cudaMemcpyDeviceToDevice);

    // Save initial state
    void** d_xfp4_init=new void*[M]; float** d_xs_init=new float*[M];
    for(int m=0;m<M;++m){cudaMalloc(&d_xfp4_init[m],H);cudaMalloc(&d_xs_init[m],(H/16)*4);
        cudaMemcpy(d_xfp4_init[m],d_xfp4M[m],H,cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_xs_init[m],d_xsM[m],(H/16)*4,cudaMemcpyDeviceToDevice);}
    void* d_xfp4_s_init; float* d_xs_s_init;
    cudaMalloc(&d_xfp4_s_init,H); cudaMalloc(&d_xs_s_init,(H/16)*4);
    cudaMemcpy(d_xfp4_s_init,d_xfp4,H,cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_xs_s_init,d_xs,(H/16)*4,cudaMemcpyDeviceToDevice);
    float *d_kc_save,*d_vc_save;
    cudaMalloc(&d_kc_save,kv_szM); cudaMalloc(&d_vc_save,kv_szM);
    cudaMemcpy(d_kc_save,d_kc,kv_szM,cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_vc_save,d_vc,kv_szM,cudaMemcpyDeviceToDevice);

    int warm=5, bench=steps;
    // Use bench param for benchmark iterations
    bench = steps;

    // ── Mode A: Autoregressive M=1 (single token, per-kernel, no graph) ────
    printf("\n=== Mode A: Autoregressive (M=1) ===\n");
    for(int w=0;w<warm;++w){
        for(int l=0;l<NL;++l){
            size_t ko=(size_t)l*nkv*ms*hd;
            blackwell::kernels::unpack_fp4(d_res_s,d_xfp4,d_xs,H,0);
            blackwell::kernels::pack_int8(d_xi8_s,d_res_s,d_xi8s_s,H,0);
            blackwell::kernels::gemv_int8_warp(d_Q_s,d_xi8_s,d_xi8s_s,L[l].q.d,L[l].q.ds,H,Q,0);
            blackwell::kernels::gemv_int8_warp(d_K_s,d_xi8_s,d_xi8s_s,L[l].k.d,L[l].k.ds,H,KV,0);
            blackwell::kernels::gemv_int8_warp(d_V_s,d_xi8_s,d_xi8s_s,L[l].v.d,L[l].v.ds,H,KV,0);
            blackwell::kernels::update_kv_cache(d_kc+ko,d_vc+ko,d_K_s,d_V_s,0,sq,nkv,hd,ms,0);
            blackwell::kernels::attention_decode_gqa(d_attn_s,d_Q_s,d_kc+ko,d_vc+ko,sq,nqh,nkv,hd,ms,0);
            blackwell::kernels::pack_int8(d_ai8_s,d_attn_s,d_ai8s_s,Q,0);
            blackwell::kernels::gemv_int8_warp(d_proj_s,d_ai8_s,d_ai8s_s,L[l].o.d,L[l].o.ds,Q,H,0);
            blackwell::kernels::vector_add_fp32(d_proj_s,d_proj_s,d_res_s,H,0);
            blackwell::kernels::fused_rmsnorm_quant_int8(d_xi8_s,d_xi8s_s,d_proj_s,d_rn,H,1e-6f,0);
            blackwell::kernels::fused_rmsnorm_pack(d_xfp4,d_xs,d_proj_s,d_rn,H,1e-6f,0);
            // MLP
            blackwell::kernels::unpack_fp4(d_res_s,d_xfp4,d_xs,H,0);
            blackwell::kernels::pack_int8(d_xi8_s,d_res_s,d_xi8s_s,H,0);
            blackwell::kernels::gemv_int8_warp(d_gate_s,d_xi8_s,d_xi8s_s,L[l].g.d,L[l].g.ds,H,I,0);
            blackwell::kernels::gemv_int8_warp(d_up_s,d_xi8_s,d_xi8s_s,L[l].u.d,L[l].u.ds,H,I,0);
            blackwell::kernels::apply_swiglu(d_mlp_s,d_gate_s,d_up_s,I,0);
            blackwell::kernels::pack_int8(d_mi8_s,d_mlp_s,d_mi8s_s,I,0);
            blackwell::kernels::gemv_int8_warp(d_proj_s,d_mi8_s,d_mi8s_s,L[l].d.d,L[l].d.ds,I,H,0);
            blackwell::kernels::vector_add_fp32(d_proj_s,d_proj_s,d_res_s,H,0);
            blackwell::kernels::fused_rmsnorm_quant_int8(d_xi8_s,d_xi8s_s,d_proj_s,d_rn,H,1e-6f,0);
            blackwell::kernels::fused_rmsnorm_pack(d_xfp4,d_xs,d_proj_s,d_rn,H,1e-6f,0);
        }
    }
    cudaDeviceSynchronize();
    GpuTimer tA; tA.start();
    for(int i=0;i<bench;++i){
        for(int l=0;l<NL;++l){
            size_t ko=(size_t)l*nkv*ms*hd;
            blackwell::kernels::unpack_fp4(d_res_s,d_xfp4,d_xs,H,0);
            blackwell::kernels::pack_int8(d_xi8_s,d_res_s,d_xi8s_s,H,0);
            blackwell::kernels::gemv_int8_warp(d_Q_s,d_xi8_s,d_xi8s_s,L[l].q.d,L[l].q.ds,H,Q,0);
            blackwell::kernels::gemv_int8_warp(d_K_s,d_xi8_s,d_xi8s_s,L[l].k.d,L[l].k.ds,H,KV,0);
            blackwell::kernels::gemv_int8_warp(d_V_s,d_xi8_s,d_xi8s_s,L[l].v.d,L[l].v.ds,H,KV,0);
            blackwell::kernels::update_kv_cache(d_kc+ko,d_vc+ko,d_K_s,d_V_s,0,sq,nkv,hd,ms,0);
            blackwell::kernels::attention_decode_gqa(d_attn_s,d_Q_s,d_kc+ko,d_vc+ko,sq,nqh,nkv,hd,ms,0);
            blackwell::kernels::pack_int8(d_ai8_s,d_attn_s,d_ai8s_s,Q,0);
            blackwell::kernels::gemv_int8_warp(d_proj_s,d_ai8_s,d_ai8s_s,L[l].o.d,L[l].o.ds,Q,H,0);
            blackwell::kernels::vector_add_fp32(d_proj_s,d_proj_s,d_res_s,H,0);
            blackwell::kernels::fused_rmsnorm_quant_int8(d_xi8_s,d_xi8s_s,d_proj_s,d_rn,H,1e-6f,0);
            blackwell::kernels::fused_rmsnorm_pack(d_xfp4,d_xs,d_proj_s,d_rn,H,1e-6f,0);
            blackwell::kernels::unpack_fp4(d_res_s,d_xfp4,d_xs,H,0);
            blackwell::kernels::pack_int8(d_xi8_s,d_res_s,d_xi8s_s,H,0);
            blackwell::kernels::gemv_int8_warp(d_gate_s,d_xi8_s,d_xi8s_s,L[l].g.d,L[l].g.ds,H,I,0);
            blackwell::kernels::gemv_int8_warp(d_up_s,d_xi8_s,d_xi8s_s,L[l].u.d,L[l].u.ds,H,I,0);
            blackwell::kernels::apply_swiglu(d_mlp_s,d_gate_s,d_up_s,I,0);
            blackwell::kernels::pack_int8(d_mi8_s,d_mlp_s,d_mi8s_s,I,0);
            blackwell::kernels::gemv_int8_warp(d_proj_s,d_mi8_s,d_mi8s_s,L[l].d.d,L[l].d.ds,I,H,0);
            blackwell::kernels::vector_add_fp32(d_proj_s,d_proj_s,d_res_s,H,0);
            blackwell::kernels::fused_rmsnorm_quant_int8(d_xi8_s,d_xi8s_s,d_proj_s,d_rn,H,1e-6f,0);
            blackwell::kernels::fused_rmsnorm_pack(d_xfp4,d_xs,d_proj_s,d_rn,H,1e-6f,0);
        }
    }
    float ms_a=tA.stop();

    // ── Mode B: M-token batched (verification pass, per-kernel) ────────────
    // Restore single-token state from init
    cudaMemcpy(d_xfp4,d_xfp4_s_init,H,cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_xs,d_xs_s_init,(H/16)*4,cudaMemcpyDeviceToDevice);
    for(int m=0;m<M;++m){
        cudaMemcpy(d_xfp4M[m],d_xfp4_init[m],H,cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_xsM[m],d_xs_init[m],(H/16)*4,cudaMemcpyDeviceToDevice);
    }
    cudaMemcpy(d_kc,d_kc_save,kv_szM,cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_vc,d_vc_save,kv_szM,cudaMemcpyDeviceToDevice);

    printf("\n=== Mode B: M=%d batched verification (per-kernel) ===\n",M);
    for(int w=0;w<warm;++w){
        for(int l=0;l<NL;++l){
            size_t klo=(size_t)l*nkv*ms*hd;
            // Fused unpack+pack all M
            for(int m=0;m<M;++m){
                blackwell::kernels::unpack_fp4_pack_int8(
                    d_xM+m*H,d_xMs+m*(H/16),d_xfp4M[m],d_xsM[m],d_xMs+m*(H/16),H,0);
            }
            // Q/K/V GEMV + KV cache (per-seq)
            for(int m=0;m<M;++m){
                size_t ko=m*kv_seq_stride+klo;
                blackwell::kernels::gemv_int8_warp(d_QM+m*Q,d_xM+m*H,d_xMs+m*(H/16),L[l].q.d,L[l].q.ds,H,Q,0);
                blackwell::kernels::gemv_int8_warp(d_KM+m*KV,d_xM+m*H,d_xMs+m*(H/16),L[l].k.d,L[l].k.ds,H,KV,0);
                blackwell::kernels::gemv_int8_warp(d_VM+m*KV,d_xM+m*H,d_xMs+m*(H/16),L[l].v.d,L[l].v.ds,H,KV,0);
                blackwell::kernels::update_kv_cache(d_kc+ko,d_vc+ko,d_KM+m*KV,d_VM+m*KV,0,sq,nkv,hd,ms,0);
            }
            // ONE batched attention
            blackwell::kernels::attention_decode_batched_gqa(d_attnM,d_QM,d_kc,d_vc,
                sq,nqh,nkv,hd,ms,M,kv_seq_stride,klo,0);
            // Wo + residual + rmsnorm
            for(int m=0;m<M;++m){
                blackwell::kernels::pack_int8(d_ai8M+m*Q,d_attnM+m*Q,d_ai8sM+m*(Q/16),Q,0);
                blackwell::kernels::gemv_int8_warp(d_projM+m*H,d_ai8M+m*Q,d_ai8sM+m*(Q/16),L[l].o.d,L[l].o.ds,Q,H,0);
                blackwell::kernels::unpack_fp4(d_res_s,d_xfp4M[m],d_xsM[m],H,0);
                blackwell::kernels::vector_add_fp32(d_projM+m*H,d_projM+m*H,d_res_s,H,0);
                blackwell::kernels::fused_rmsnorm_quant_int8(d_xM+m*H,d_xMs+m*(H/16),d_projM+m*H,d_rn,H,1e-6f,0);
                blackwell::kernels::fused_rmsnorm_pack(d_xfp4M[m],d_xsM[m],d_projM+m*H,d_rn,H,1e-6f,0);
            }
            // Batched MLP
            blackwell::kernels::gemv_int8_batched(d_gateM,d_xM,d_xMs,L[l].g.d,L[l].g.ds,H,I,M,0);
            blackwell::kernels::gemv_int8_batched(d_upM,d_xM,d_xMs,L[l].u.d,L[l].u.ds,H,I,M,0);
            for(int m=0;m<M;++m) blackwell::kernels::apply_swiglu(d_mlpM+m*I,d_gateM+m*I,d_upM+m*I,I,0);
            for(int m=0;m<M;++m) blackwell::kernels::pack_int8(d_mi8M+m*I,d_mlpM+m*I,d_mi8sM+m*(I/16),I,0);
            blackwell::kernels::gemv_int8_batched(d_projM,d_mi8M,d_mi8sM,L[l].d.d,L[l].d.ds,I,H,M,0);
            for(int m=0;m<M;++m){
                blackwell::kernels::unpack_fp4(d_res_s,d_xfp4M[m],d_xsM[m],H,0);
                blackwell::kernels::vector_add_fp32(d_projM+m*H,d_projM+m*H,d_res_s,H,0);
                blackwell::kernels::fused_rmsnorm_quant_int8(d_xM+m*H,d_xMs+m*(H/16),d_projM+m*H,d_rn,H,1e-6f,0);
                blackwell::kernels::fused_rmsnorm_pack(d_xfp4M[m],d_xsM[m],d_projM+m*H,d_rn,H,1e-6f,0);
            }
        }
    }
    cudaDeviceSynchronize();
    GpuTimer tB; tB.start();
    for(int i=0;i<bench;++i){
        for(int l=0;l<NL;++l){
            size_t klo=(size_t)l*nkv*ms*hd;
            for(int m=0;m<M;++m){
                blackwell::kernels::unpack_fp4_pack_int8(
                    d_xM+m*H,d_xMs+m*(H/16),d_xfp4M[m],d_xsM[m],d_xMs+m*(H/16),H,0);
            }
            for(int m=0;m<M;++m){
                size_t ko=m*kv_seq_stride+klo;
                blackwell::kernels::gemv_int8_warp(d_QM+m*Q,d_xM+m*H,d_xMs+m*(H/16),L[l].q.d,L[l].q.ds,H,Q,0);
                blackwell::kernels::gemv_int8_warp(d_KM+m*KV,d_xM+m*H,d_xMs+m*(H/16),L[l].k.d,L[l].k.ds,H,KV,0);
                blackwell::kernels::gemv_int8_warp(d_VM+m*KV,d_xM+m*H,d_xMs+m*(H/16),L[l].v.d,L[l].v.ds,H,KV,0);
                blackwell::kernels::update_kv_cache(d_kc+ko,d_vc+ko,d_KM+m*KV,d_VM+m*KV,0,sq,nkv,hd,ms,0);
            }
            blackwell::kernels::attention_decode_batched_gqa(d_attnM,d_QM,d_kc,d_vc,
                sq,nqh,nkv,hd,ms,M,kv_seq_stride,klo,0);
            for(int m=0;m<M;++m){
                blackwell::kernels::pack_int8(d_ai8M+m*Q,d_attnM+m*Q,d_ai8sM+m*(Q/16),Q,0);
                blackwell::kernels::gemv_int8_warp(d_projM+m*H,d_ai8M+m*Q,d_ai8sM+m*(Q/16),L[l].o.d,L[l].o.ds,Q,H,0);
                blackwell::kernels::unpack_fp4(d_res_s,d_xfp4M[m],d_xsM[m],H,0);
                blackwell::kernels::vector_add_fp32(d_projM+m*H,d_projM+m*H,d_res_s,H,0);
                blackwell::kernels::fused_rmsnorm_quant_int8(d_xM+m*H,d_xMs+m*(H/16),d_projM+m*H,d_rn,H,1e-6f,0);
                blackwell::kernels::fused_rmsnorm_pack(d_xfp4M[m],d_xsM[m],d_projM+m*H,d_rn,H,1e-6f,0);
            }
            blackwell::kernels::gemv_int8_batched(d_gateM,d_xM,d_xMs,L[l].g.d,L[l].g.ds,H,I,M,0);
            blackwell::kernels::gemv_int8_batched(d_upM,d_xM,d_xMs,L[l].u.d,L[l].u.ds,H,I,M,0);
            for(int m=0;m<M;++m) blackwell::kernels::apply_swiglu(d_mlpM+m*I,d_gateM+m*I,d_upM+m*I,I,0);
            for(int m=0;m<M;++m) blackwell::kernels::pack_int8(d_mi8M+m*I,d_mlpM+m*I,d_mi8sM+m*(I/16),I,0);
            blackwell::kernels::gemv_int8_batched(d_projM,d_mi8M,d_mi8sM,L[l].d.d,L[l].d.ds,I,H,M,0);
            for(int m=0;m<M;++m){
                blackwell::kernels::unpack_fp4(d_res_s,d_xfp4M[m],d_xsM[m],H,0);
                blackwell::kernels::vector_add_fp32(d_projM+m*H,d_projM+m*H,d_res_s,H,0);
                blackwell::kernels::fused_rmsnorm_quant_int8(d_xM+m*H,d_xMs+m*(H/16),d_projM+m*H,d_rn,H,1e-6f,0);
                blackwell::kernels::fused_rmsnorm_pack(d_xfp4M[m],d_xsM[m],d_projM+m*H,d_rn,H,1e-6f,0);
            }
        }
    }
    float ms_b=tB.stop();

    // ── Mode C: M-token batched + CUDA Graph ────────────────────────────────
    printf("\n=== Mode C: CUDA Graph batched M=%d ===\n",M);
    cudaDeviceSynchronize();
    cudaGetLastError();

    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,8*1024*1024);
    cudaStream_t gs; cudaStreamCreate(&gs);

    cudaAccessPolicyWindow np;
    np.base_ptr=(void*)d_rn; np.num_bytes=H*4; np.hitRatio=1.0f;
    np.hitProp=cudaAccessPropertyPersisting; np.missProp=cudaAccessPropertyStreaming;
    cudaStreamAttrValue na; na.accessPolicyWindow=np;
    cudaStreamSetAttribute(gs,cudaStreamAttributeAccessPolicyWindow,&na);

    // Pre-trigger batched attention
    blackwell::kernels::attention_decode_batched_gqa(d_attnM,d_QM,d_kc,d_vc,sq,nqh,nkv,hd,ms,M,kv_seq_stride,0,gs);
    cudaStreamSynchronize(gs);

    printf("  Capturing %d layers × %d seqs... ",NL,M);
    fflush(stdout);

    cudaStreamBeginCapture(gs,cudaStreamCaptureModeGlobal);
    for(int l=0;l<NL;++l){
        size_t klo=(size_t)l*nkv*ms*hd;
        for(int m=0;m<M;++m){
            blackwell::kernels::unpack_fp4_pack_int8(
                d_xM+m*H,d_xMs+m*(H/16),d_xfp4M[m],d_xsM[m],d_xMs+m*(H/16),H,gs);
        }
        for(int m=0;m<M;++m){
            size_t ko=m*kv_seq_stride+klo;
            blackwell::kernels::gemv_int8_warp(d_QM+m*Q,d_xM+m*H,d_xMs+m*(H/16),L[l].q.d,L[l].q.ds,H,Q,gs);
            blackwell::kernels::gemv_int8_warp(d_KM+m*KV,d_xM+m*H,d_xMs+m*(H/16),L[l].k.d,L[l].k.ds,H,KV,gs);
            blackwell::kernels::gemv_int8_warp(d_VM+m*KV,d_xM+m*H,d_xMs+m*(H/16),L[l].v.d,L[l].v.ds,H,KV,gs);
            blackwell::kernels::update_kv_cache(d_kc+ko,d_vc+ko,d_KM+m*KV,d_VM+m*KV,0,sq,nkv,hd,ms,gs);
        }
        blackwell::kernels::attention_decode_batched_gqa(d_attnM,d_QM,d_kc,d_vc,
            sq,nqh,nkv,hd,ms,M,kv_seq_stride,klo,gs);
        for(int m=0;m<M;++m){
            blackwell::kernels::pack_int8(d_ai8M+m*Q,d_attnM+m*Q,d_ai8sM+m*(Q/16),Q,gs);
            blackwell::kernels::gemv_int8_warp(d_projM+m*H,d_ai8M+m*Q,d_ai8sM+m*(Q/16),L[l].o.d,L[l].o.ds,Q,H,gs);
            blackwell::kernels::unpack_fp4(d_res_s,d_xfp4M[m],d_xsM[m],H,gs);
            blackwell::kernels::vector_add_fp32(d_projM+m*H,d_projM+m*H,d_res_s,H,gs);
            blackwell::kernels::fused_rmsnorm_quant_int8(d_xM+m*H,d_xMs+m*(H/16),d_projM+m*H,d_rn,H,1e-6f,gs);
            blackwell::kernels::fused_rmsnorm_pack(d_xfp4M[m],d_xsM[m],d_projM+m*H,d_rn,H,1e-6f,gs);
        }
        blackwell::kernels::gemv_int8_batched(d_gateM,d_xM,d_xMs,L[l].g.d,L[l].g.ds,H,I,M,gs);
        blackwell::kernels::gemv_int8_batched(d_upM,d_xM,d_xMs,L[l].u.d,L[l].u.ds,H,I,M,gs);
        for(int m=0;m<M;++m) blackwell::kernels::apply_swiglu(d_mlpM+m*I,d_gateM+m*I,d_upM+m*I,I,gs);
        for(int m=0;m<M;++m) blackwell::kernels::pack_int8(d_mi8M+m*I,d_mlpM+m*I,d_mi8sM+m*(I/16),I,gs);
        blackwell::kernels::gemv_int8_batched(d_projM,d_mi8M,d_mi8sM,L[l].d.d,L[l].d.ds,I,H,M,gs);
        for(int m=0;m<M;++m){
            blackwell::kernels::unpack_fp4(d_res_s,d_xfp4M[m],d_xsM[m],H,gs);
            blackwell::kernels::vector_add_fp32(d_projM+m*H,d_projM+m*H,d_res_s,H,gs);
            blackwell::kernels::fused_rmsnorm_quant_int8(d_xM+m*H,d_xMs+m*(H/16),d_projM+m*H,d_rn,H,1e-6f,gs);
            blackwell::kernels::fused_rmsnorm_pack(d_xfp4M[m],d_xsM[m],d_projM+m*H,d_rn,H,1e-6f,gs);
        }
    }
    cudaGraph_t graph; cudaStreamEndCapture(gs,&graph);
    cudaGraphExec_t gexec; cudaGraphInstantiate(&gexec,graph,NULL,NULL,0);
    printf("OK\n");

    printf("  Graph warmup...\n");
    for(int i=0;i<warm;++i) cudaGraphLaunch(gexec,gs);
    cudaStreamSynchronize(gs);

    printf("  Graph benchmark (%d iters)...\n",bench);
    GpuTimer tC; tC.start(gs);
    for(int i=0;i<bench;++i) cudaGraphLaunch(gexec,gs);
    cudaStreamSynchronize(gs);
    float ms_c=tC.stop();

    // ── Results ──────────────────────────────────────────────────────────────
    float pt_a=ms_a/bench, pt_b=ms_b/bench, pt_c=ms_c/bench;
    float tps_single=1e3/pt_a;
    float tps_spec=((float)M+1)*1e3/pt_b;                    // M+1 tokens per spec step
    float tps_spec_cg=((float)M+1)*1e3/pt_c;
    float s28_single=1e3/(pt_a*28.f/NL);
    float s28_spec=1e3/(pt_b*28.f/NL);
    float s28_spec_cg=1e3/(pt_c*28.f/NL);

    printf("\n=== Results (%d layers, M=%d drafts, %d steps) ===\n",NL,M,bench);
    printf("  %-30s  %10s  %10s  %10s\n","Method","Per-step","t/s","Scaled28");
    printf("  %-30s  %7.3fms   %7.1f    %7.1f\n","Autoregressive (M=1)",
        pt_a,tps_single,s28_single);
    char m4s[64]; snprintf(m4s,64,"Speculative M=%d (per-kernel)",M);
    char m4g[64]; snprintf(m4g,64,"Speculative M=%d + CUDA Graph",M);
    printf("  %-30s  %7.3fms   %7.1f    %7.1f\n",m4s,pt_b,tps_spec,s28_spec);
    printf("  %-30s  %7.3fms   %7.1f    %7.1f\n",m4g,pt_c,tps_spec_cg,s28_spec_cg);
    printf("  Spec speedup (per-kernel): %.2fx (%.1f%%)\n",
        tps_spec/tps_single, (tps_spec/tps_single-1)*100.f);
    printf("  Spec speedup (CUDA Graph): %.2fx (%.1f%%)\n",
        tps_spec_cg/tps_single, (tps_spec_cg/tps_single-1)*100.f);
    printf("  Note: assumes 100%% draft acceptance (simplified benchmark)\n");
    printf("  Target: llama.cpp 276.0 t/s\n");

    // Cleanup
    cudaGraphExecDestroy(gexec); cudaGraphDestroy(graph); cudaStreamDestroy(gs);
    for(int m=0;m<M;++m){cudaFree(d_xfp4M[m]);cudaFree(d_xsM[m]);cudaFree(d_xfp4_init[m]);cudaFree(d_xs_init[m]);}
    delete[] d_xfp4M; delete[] d_xsM; delete[] d_xfp4_init; delete[] d_xs_init;
    cudaFree(d_xfp4_s_init); cudaFree(d_xs_s_init);
    cudaFree(d_kc_save); cudaFree(d_vc_save);
    for(auto&l:L)l.free_all();
    cudaFree(d_rn); cudaFree(d_res_s); cudaFree(d_xi8_s); cudaFree(d_xi8s_s);
    cudaFree(d_xfp4); cudaFree(d_xs);
    cudaFree(d_Q_s); cudaFree(d_K_s); cudaFree(d_V_s); cudaFree(d_attn_s); cudaFree(d_proj_s);
    cudaFree(d_gate_s); cudaFree(d_up_s); cudaFree(d_mlp_s);
    cudaFree(d_ai8_s); cudaFree(d_ai8s_s); cudaFree(d_mi8_s); cudaFree(d_mi8s_s);
    cudaFree(d_xM); cudaFree(d_xMs);
    cudaFree(d_QM); cudaFree(d_KM); cudaFree(d_VM); cudaFree(d_attnM); cudaFree(d_projM);
    cudaFree(d_gateM); cudaFree(d_upM); cudaFree(d_mlpM);
    cudaFree(d_ai8M); cudaFree(d_ai8sM); cudaFree(d_mi8M); cudaFree(d_mi8sM);
    cudaFree(d_kc); cudaFree(d_vc); cudaFree(d_x32);
    return 0;
}
