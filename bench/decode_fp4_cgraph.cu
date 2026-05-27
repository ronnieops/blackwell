// bench/decode_fp4_cgraph.cu — FP4 CUDA Graph decode benchmark
//
// Compare FP4 MMA decode throughput vs INT8 dp4a CUDA Graph.
// Uses FP4 weights from weights/ (28 layers).
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/decode_fp4_cgraph.cu build/libblackwell_kernels.a \
//     -o bench/decode_fp4_cgraph

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>
#include "blackwell/kernels.h"

struct GpuTimer { cudaEvent_t s,e; GpuTimer(){cudaEventCreate(&s);cudaEventCreate(&e);} ~GpuTimer(){cudaEventDestroy(s);cudaEventDestroy(e);} void start(cudaStream_t st=0){cudaEventRecord(s,st);} float stop(cudaStream_t st=0){cudaEventRecord(e,st);cudaEventSynchronize(e);float m=0;cudaEventElapsedTime(&m,s,e);return m;} };
static void chk(cudaError_t e, const char* m){if(e!=cudaSuccess){printf("FAIL %s: %s\n",m,cudaGetErrorString(e));exit(1);}}

struct Fp4W { int K, N; std::vector<uint8_t> d; std::vector<float> s; };
static Fp4W load_fp4(const char* p){Fp4W w;FILE*f=fopen(p,"rb");int h[5];fread(h,4,5,f);w.K=h[0];w.N=h[1];w.d.resize((size_t)w.K*w.N);fread(w.d.data(),1,w.d.size(),f);size_t ns=(size_t)h[3]*h[4];w.s.resize(ns);fread(w.s.data(),4,ns,f);fclose(f);return w;}
static void up_fp4(const Fp4W& w, void*& d_d, float*& d_s){cudaMalloc(&d_d,(size_t)w.K*w.N);cudaMemcpy(d_d,w.d.data(),(size_t)w.K*w.N,cudaMemcpyHostToDevice);cudaMalloc(&d_s,w.s.size()*4);cudaMemcpy(d_s,w.s.data(),w.s.size()*4,cudaMemcpyHostToDevice);}

struct LW { void*q,*k,*v,*o,*gate,*up,*down; float*qs,*ks,*vs,*os,*gs,*us,*ds; };
static LW load_layer(int l) {
    char p[256];LW w;
    snprintf(p,256,"weights/%d_self_attn.q_proj.fp4",l);auto wq=load_fp4(p);up_fp4(wq,w.q,w.qs);
    snprintf(p,256,"weights/%d_self_attn.k_proj.fp4",l);auto wk=load_fp4(p);up_fp4(wk,w.k,w.ks);
    snprintf(p,256,"weights/%d_self_attn.v_proj.fp4",l);auto wv=load_fp4(p);up_fp4(wv,w.v,w.vs);
    snprintf(p,256,"weights/%d_self_attn.o_proj.fp4",l);auto wo=load_fp4(p);up_fp4(wo,w.o,w.os);
    snprintf(p,256,"weights/%d_mlp.gate_proj.fp4",l);auto wg=load_fp4(p);up_fp4(wg,w.gate,w.gs);
    snprintf(p,256,"weights/%d_mlp.up_proj.fp4",l);auto wu=load_fp4(p);up_fp4(wu,w.up,w.us);
    snprintf(p,256,"weights/%d_mlp.down_proj.fp4",l);auto wd=load_fp4(p);up_fp4(wd,w.down,w.ds);
    return w;
}
static void free_layer(LW& w){cudaFree(w.q);cudaFree(w.qs);cudaFree(w.k);cudaFree(w.ks);cudaFree(w.v);cudaFree(w.vs);cudaFree(w.o);cudaFree(w.os);cudaFree(w.gate);cudaFree(w.gs);cudaFree(w.up);cudaFree(w.us);cudaFree(w.down);cudaFree(w.ds);}

int main(int argc, char** argv) {
    int NL=4; if(argc>1)NL=atoi(argv[1]);
    cudaDeviceProp p;cudaGetDeviceProperties(&p,0);
    printf("# FP4 CUDA Graph Decode — Qwen3-1.7B\nDevice: %s (CC %d.%d)\nLayers: %d\n",p.name,p.major,p.minor,NL);

    const int H=2048,Q=2048,KV=1024,I=6144,nqh=16,nkv=8,hd=128,ms=2048;
    const float s13=1.f/3.f;

    printf("Loading FP4 weights...\n");
    std::vector<LW> lw(NL);
    for(int l=0;l<NL;++l){lw[l]=load_layer(l);printf("  L%d\n",l);}

    // Buffers
    float*d_x32;void*d_xfp4;float*d_xs;
    cudaMalloc(&d_x32,H*4);cudaMalloc(&d_xfp4,H);cudaMalloc(&d_xs,(H/16)*4);
    float*d_rn;cudaMalloc(&d_rn,H*4);
    std::vector<float>ones(H,1.f);cudaMemcpy(d_rn,ones.data(),H*4,cudaMemcpyHostToDevice);

    float*d_Q,*d_K,*d_V,*d_attn,*d_proj,*d_gate,*d_up,*d_mlp,*d_res;
    void*d_xfp4_a,*d_xfp4_mlp,*d_attn_fp4,*d_mlp_fp4;
    float*d_attn_s,*d_mlp_s;
    cudaMalloc(&d_Q,Q*4);cudaMalloc(&d_K,KV*4);cudaMalloc(&d_V,KV*4);
    cudaMalloc(&d_attn,Q*4);cudaMalloc(&d_proj,H*4);
    cudaMalloc(&d_gate,I*4);cudaMalloc(&d_up,I*4);cudaMalloc(&d_mlp,I*4);
    cudaMalloc(&d_res,I*4);
    cudaMalloc(&d_xfp4_a,H);cudaMalloc(&d_xfp4_mlp,I);
    cudaMalloc(&d_attn_fp4,Q);cudaMalloc(&d_mlp_fp4,I);
    cudaMalloc(&d_attn_s,(Q/16)*4);cudaMalloc(&d_mlp_s,(I/16)*4);

    // Init
    std::vector<float>x_init(H,1.f),xs_init(H/16,s13);
    cudaMemcpy(d_x32,x_init.data(),H*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_xs,xs_init.data(),(H/16)*4,cudaMemcpyHostToDevice);
    blackwell::kernels::pack_fp4(d_xfp4,d_x32,d_xs,H,0);
    std::vector<float>as_init(Q/16,s13),ms_init(I/16,s13);
    cudaMemcpy(d_attn_s,as_init.data(),(Q/16)*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_mlp_s,ms_init.data(),(I/16)*4,cudaMemcpyHostToDevice);

    // KV cache
    float*d_kc,*d_vc;
    size_t kv_sz=(size_t)NL*nkv*ms*hd*4;
    cudaMalloc(&d_kc,kv_sz);cudaMalloc(&d_vc,kv_sz);
    cudaMemset(d_kc,0,kv_sz);cudaMemset(d_vc,0,kv_sz);

    int sq=128;
    printf("Filling KV cache (seq=0..%d)...\n",sq);
    for(int s=0;s<=sq;++s){
        for(int l=0;l<NL;++l){
            int kb=l*nkv*ms*hd;
            blackwell::kernels::gemv_fp4_v2(d_Q,d_xfp4,d_xs,lw[l].q,lw[l].qs,H,Q,0);
            blackwell::kernels::gemv_fp4_v2(d_K,d_xfp4,d_xs,lw[l].k,lw[l].ks,H,KV,0);
            blackwell::kernels::gemv_fp4_v2(d_V,d_xfp4,d_xs,lw[l].v,lw[l].vs,H,KV,0);
            blackwell::kernels::update_kv_cache(d_kc+kb,d_vc+kb,d_K,d_V,0,s,nkv,hd,ms,0);
            blackwell::kernels::attention_decode_gqa(d_attn,d_Q,d_kc+kb,d_vc+kb,s,nqh,nkv,hd,ms,0);
            blackwell::kernels::pack_fp4(d_attn_fp4,d_attn,d_attn_s,Q,0);
            blackwell::kernels::gemv_fp4_v2(d_proj,d_attn_fp4,d_attn_s,lw[l].o,lw[l].os,Q,H,0);
            blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_res,H,0);
            blackwell::kernels::fused_rmsnorm_pack(d_xfp4,d_xs,d_proj,d_rn,H,1e-5f,0);
            // MLP
            blackwell::kernels::unpack_fp4(d_res,d_xfp4,d_xs,H,0);
            blackwell::kernels::fused_gate_up_gemv(d_gate,d_up,d_xfp4,d_xs,lw[l].gate,lw[l].gs,lw[l].up,lw[l].us,H,I,0);
            blackwell::kernels::apply_swiglu(d_mlp,d_gate,d_up,I,0);
            blackwell::kernels::pack_fp4(d_mlp_fp4,d_mlp,d_mlp_s,I,0);
            
            blackwell::kernels::gemv_fp4_v2(d_proj,d_mlp_fp4,d_mlp_s,lw[l].down,lw[l].ds,I,H,0);
            blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_res,H,0);
            blackwell::kernels::fused_rmsnorm_pack(d_xfp4,d_xs,d_proj,d_rn,H,1e-5f,0);
        }
    }
    cudaDeviceSynchronize();
    printf("done\n");

    // CUDA Graph capture
    cudaStream_t gs;cudaStreamCreate(&gs);
    blackwell::kernels::attention_decode_gqa(d_attn,d_Q,d_kc,d_vc,sq,nqh,nkv,hd,ms,gs);
    cudaStreamSynchronize(gs);

    printf("Capturing %d layers...\n",NL);
    cudaStreamBeginCapture(gs,cudaStreamCaptureModeGlobal);
    for(int l=0;l<NL;++l){
        int kb=l*nkv*ms*hd;
        blackwell::kernels::unpack_fp4(d_res,d_xfp4,d_xs,H,gs);
        blackwell::kernels::gemv_fp4_v2(d_Q,d_xfp4,d_xs,lw[l].q,lw[l].qs,H,Q,gs);
        blackwell::kernels::gemv_fp4_v2(d_K,d_xfp4,d_xs,lw[l].k,lw[l].ks,H,KV,gs);
        blackwell::kernels::gemv_fp4_v2(d_V,d_xfp4,d_xs,lw[l].v,lw[l].vs,H,KV,gs);
        blackwell::kernels::update_kv_cache(d_kc+kb,d_vc+kb,d_K,d_V,0,sq,nkv,hd,ms,gs);
        blackwell::kernels::attention_decode_gqa(d_attn,d_Q,d_kc+kb,d_vc+kb,sq,nqh,nkv,hd,ms,gs);
        blackwell::kernels::pack_fp4(d_attn_fp4,d_attn,d_attn_s,Q,gs);
        blackwell::kernels::gemv_fp4_v2(d_proj,d_attn_fp4,d_attn_s,lw[l].o,lw[l].os,Q,H,gs);
        blackwell::kernels::fused_rmsnorm(d_proj,d_proj,d_rn,H,1e-5f,gs);
        blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,gs);
        blackwell::kernels::fused_rmsnorm_pack(d_xfp4,d_xs,d_x32,d_rn,H,1e-5f,gs);
        blackwell::kernels::unpack_fp4(d_res,d_xfp4,d_xs,H,gs);
        blackwell::kernels::fused_gate_up_gemv(d_gate,d_up,d_xfp4,d_xs,lw[l].gate,lw[l].gs,lw[l].up,lw[l].us,H,I,gs);
        blackwell::kernels::apply_swiglu(d_mlp,d_gate,d_up,I,gs);
        blackwell::kernels::pack_fp4(d_mlp_fp4,d_mlp,d_mlp_s,I,gs);
        
        blackwell::kernels::gemv_fp4_v2(d_proj,d_mlp_fp4,d_mlp_s,lw[l].down,lw[l].ds,I,H,gs);
        blackwell::kernels::fused_rmsnorm(d_proj,d_proj,d_rn,H,1e-5f,gs);
        blackwell::kernels::vector_add_fp32(d_mlp,d_proj,d_res,I,gs);
        blackwell::kernels::fused_rmsnorm_pack(d_xfp4,d_xs,d_mlp,d_rn,H,1e-5f,gs);
    }
    cudaGraph_t graph;cudaError_t ce=cudaStreamEndCapture(gs,&graph);
    if(ce!=cudaSuccess){printf("Capture FAIL: %s\n",cudaGetErrorString(ce));return 1;}
    cudaGraphExec_t gexec;cudaGraphInstantiate(&gexec,graph,NULL,NULL,0);
    printf("OK\n");

    // Per-kernel baseline
    int bench=20;
    printf("Per-kernel benchmark...\n");
    GpuTimer t0;t0.start();
    for(int i=0;i<bench;++i){
        for(int l=0;l<NL;++l){
            int kb=l*nkv*ms*hd;
            blackwell::kernels::unpack_fp4(d_res,d_xfp4,d_xs,H,0);
            blackwell::kernels::gemv_fp4_v2(d_Q,d_xfp4,d_xs,lw[l].q,lw[l].qs,H,Q,0);
            blackwell::kernels::gemv_fp4_v2(d_K,d_xfp4,d_xs,lw[l].k,lw[l].ks,H,KV,0);
            blackwell::kernels::gemv_fp4_v2(d_V,d_xfp4,d_xs,lw[l].v,lw[l].vs,H,KV,0);
            blackwell::kernels::update_kv_cache(d_kc+kb,d_vc+kb,d_K,d_V,0,sq,nkv,hd,ms,0);
            blackwell::kernels::attention_decode_gqa(d_attn,d_Q,d_kc+kb,d_vc+kb,sq,nqh,nkv,hd,ms,0);
            blackwell::kernels::pack_fp4(d_attn_fp4,d_attn,d_attn_s,Q,0);
            blackwell::kernels::gemv_fp4_v2(d_proj,d_attn_fp4,d_attn_s,lw[l].o,lw[l].os,Q,H,0);
            blackwell::kernels::fused_rmsnorm(d_proj,d_proj,d_rn,H,1e-5f,0);
            blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,0);
            blackwell::kernels::fused_rmsnorm_pack(d_xfp4,d_xs,d_x32,d_rn,H,1e-5f,0);
            blackwell::kernels::unpack_fp4(d_res,d_xfp4,d_xs,H,0);
            blackwell::kernels::fused_gate_up_gemv(d_gate,d_up,d_xfp4,d_xs,lw[l].gate,lw[l].gs,lw[l].up,lw[l].us,H,I,0);
            blackwell::kernels::apply_swiglu(d_mlp,d_gate,d_up,I,0);
            blackwell::kernels::pack_fp4(d_mlp_fp4,d_mlp,d_mlp_s,I,0);
            
            blackwell::kernels::gemv_fp4_v2(d_proj,d_mlp_fp4,d_mlp_s,lw[l].down,lw[l].ds,I,H,0);
            blackwell::kernels::fused_rmsnorm(d_proj,d_proj,d_rn,H,1e-5f,0);
            blackwell::kernels::vector_add_fp32(d_mlp,d_proj,d_res,I,0);
            blackwell::kernels::fused_rmsnorm_pack(d_xfp4,d_xs,d_mlp,d_rn,H,1e-5f,0);
        }
    }
    float b_ms=t0.stop();float b_pt=b_ms/bench;
    float b_s28=1000.f/(b_pt*28.f/NL);

    // Graph benchmark
    printf("Graph benchmark...\n");
    GpuTimer tg;tg.start(gs);
    for(int i=0;i<bench;++i)cudaGraphLaunch(gexec,gs);
    cudaStreamSynchronize(gs);
    float g_ms=tg.stop();float g_pt=g_ms/bench;
    float g_s28=1000.f/(g_pt*28.f/NL);

    printf("\n=== FP4 Results (%d layers) ===\n",NL);
    printf("  %-20s  %8s  %8s  %8s\n","Method","Per-tok","t/s","Scaled28");
    printf("  %-20s  %7.3fms  %7.1f   %7.1f\n","Per-kernel",b_pt,1000.f/b_pt,b_s28);
    printf("  %-20s  %7.3fms  %7.1f   %7.1f\n","CUDA Graph",g_pt,1000.f/g_pt,g_s28);
    printf("  Speedup: %.2fx (%.1f%%)\n",b_pt/g_pt,(1-b_pt/g_pt)*100.f);
    printf("  INT8 CUDA Graph: 122.7 t/s (reference)\n");

    cudaGraphExecDestroy(gexec);cudaGraphDestroy(graph);cudaStreamDestroy(gs);
    for(auto&l:lw)free_layer(l);
    cudaFree(d_x32);cudaFree(d_xfp4);cudaFree(d_xs);cudaFree(d_rn);
    cudaFree(d_Q);cudaFree(d_K);cudaFree(d_V);cudaFree(d_attn);cudaFree(d_proj);
    cudaFree(d_gate);cudaFree(d_up);cudaFree(d_mlp);cudaFree(d_res);
    cudaFree(d_xfp4_a);cudaFree(d_xfp4_mlp);cudaFree(d_attn_fp4);cudaFree(d_mlp_fp4);
    cudaFree(d_attn_s);cudaFree(d_mlp_s);cudaFree(d_kc);cudaFree(d_vc);
    return 0;
}