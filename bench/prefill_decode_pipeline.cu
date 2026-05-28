// bench/prefill_decode_pipeline.cu — 1-layer prefill + autoregressive decode
//
// Prefill: measure individual kernel timings (synthetic INT8 data, like existing benchmarks)
// Decode:  full connected pipeline (INT8, all kernels feed each other correctly)
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/prefill_decode_pipeline.cu build/libblackwell_kernels.a \
//     -o bench/prefill_decode_pipeline
//
// Run: ./bench/prefill_decode_pipeline [M=128] [DT=50] [IT=20]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include "blackwell/kernels.h"

static void die(cudaError_t e, const char* m) {
    if(e!=cudaSuccess){printf("FAIL %s: %s\n",m,cudaGetErrorString(e));exit(1);}}

struct GpuTimer{
    cudaEvent_t s,e;
    GpuTimer(){cudaEventCreate(&s);cudaEventCreate(&e);}
    ~GpuTimer(){cudaEventDestroy(s);cudaEventDestroy(e);}
    void start(){cudaEventRecord(s);}
    float stop(){cudaEventRecord(e);cudaEventSynchronize(e);float ms=0;cudaEventElapsedTime(&ms,s,e);return ms;}
};

// Qwen3-1.7B dims
const int H=2048, Q=2048, KV=1024, I=6144, nqh=12, nkv=1, hd=64, qpg=nqh/nkv;
const float eps=1e-6f, sc_at=1.f/sqrtf((float)hd);

static void fill_i8(std::vector<int8_t>& v){
    for(size_t i=0;i<v.size();++i)v[i]=((i*17+13)%127)-64;}
static void fill_sc(std::vector<float>& v){
    for(auto& s:v)s=1.f/127.f;}

// ── INT8/FP4 weight upload ─────────────────────────────────────────────
// For dispatch_matmul (Prefill): weight in [K×N] raw bytes (treated as FP4)
// For gemv_int8 (Decode): weight transposed to [N×K]
struct DevW{int8_t*d;float*sc;int K,N;};
static DevW mk_w(int K,int N,cudaStream_t st){
    std::vector<int8_t>w(K*N);fill_i8(w);
    int sr=(K+15)/16,scn=(N+15)/16;
    std::vector<float>s(sr*scn);fill_sc(s);
    DevW dw;dw.K=K;dw.N=N;
    cudaMalloc(&dw.d,K*N);cudaMemcpyAsync(dw.d,w.data(),K*N,cudaMemcpyHostToDevice,st);
    cudaMalloc(&dw.sc,s.size()*4);cudaMemcpyAsync(dw.sc,s.data(),s.size()*4,cudaMemcpyHostToDevice,st);
    return dw;
}
static DevW mk_wt(int K,int N,cudaStream_t st){
    std::vector<int8_t>w(K*N);fill_i8(w);
    std::vector<int8_t>wt(N*K);
    for(int k=0;k<K;++k)for(int n=0;n<N;++n)wt[n*K+k]=w[k*N+n];
    int sr=(K+15)/16,scn=(N+15)/16;
    std::vector<float>so(sr*scn);fill_sc(so);
    std::vector<float>sv(scn*sr);
    for(int r=0;r<sr;++r)for(int c=0;c<scn;++c)sv[c*sr+r]=so[r*scn+c];
    DevW dw;dw.K=K;dw.N=N;
    cudaMalloc(&dw.d,N*K);cudaMemcpyAsync(dw.d,wt.data(),N*K,cudaMemcpyHostToDevice,st);
    cudaMalloc(&dw.sc,sv.size()*4);cudaMemcpyAsync(dw.sc,sv.data(),sv.size()*4,cudaMemcpyHostToDevice,st);
    return dw;
}

// ── Transpose Q/K/V for attention: [M, QH*HD] → [QH, M, HD] ────────────
__global__ void transpose_attn(float* dst, const float* src, int M, int Hd, int HD) {
    int i=threadIdx.x+blockIdx.x*blockDim.x;
    int N=M*Hd*HD; if(i>=N)return;
    int m=i/(Hd*HD); int h=(i/HD)%Hd; int d=i%HD;
    dst[h*M*HD+m*HD+d]=src[m*Hd*HD+h*HD+d];
}
static void do_transpose(float* dst, const float* src, int M, int Hd, int HD, cudaStream_t st) {
    int N=M*Hd*HD; int T=256;
    transpose_attn<<<(N+T-1)/T,T,0,st>>>(dst,src,M,Hd,HD);
}

// ── Pinned memory timer ────────────────────────────────────────────────
static double bench_n(cudaStream_t st, int IT, auto&& fn) {
    GpuTimer tm;
    fn(); // warmup
    tm.start();
    for(int i=0;i<IT;++i)fn();
    return tm.stop()/IT;
}

int main(int argc, char** argv) {
    int M=128, DT=50, IT=20;
    if(argc>1)M=atoi(argv[1]);
    if(argc>2)DT=atoi(argv[2]);
    if(argc>3)IT=atoi(argv[3]);

    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("# Prefill+Decode Pipeline — %s  SM:%d\n",p.name,p.multiProcessorCount);
    printf("Prefill: M=%d  Decode: %d tokens  IT=%d\n\n",M,DT,IT);

    cudaStream_t st; die(cudaStreamCreate(&st),"stream");

    // ── Weights ────────────────────────────────────────────────────────
    printf("Allocating weights...\n");
    struct W{DevW q,k,v,o,g,u,d, qt,kt,vt,ot,gt,ut,dt;}w;
    w.q=mk_w(H,Q,st); w.qt=mk_wt(H,Q,st);
    w.k=mk_w(H,KV,st); w.kt=mk_wt(H,KV,st);
    w.v=mk_w(H,KV,st); w.vt=mk_wt(H,KV,st);
    w.o=mk_w(Q,H,st); w.ot=mk_wt(Q,H,st);
    w.g=mk_w(H,I,st); w.gt=mk_wt(H,I,st);
    w.u=mk_w(H,I,st); w.ut=mk_wt(H,I,st);
    w.d=mk_w(I,H,st); w.dt=mk_wt(I,H,st);
    cudaStreamSynchronize(st);

    // ── Prefill buffers ────────────────────────────────────────────────
    float *d_x, *d_rn, *d_Qf, *d_Kf, *d_Vf, *d_attn, *d_proj, *d_res;
    float *d_gate, *d_up, *d_mlp;
    int8_t *d_xi; float *d_xs;
    float *d_Qt, *d_Kt, *d_Vt;

    die(cudaMalloc(&d_x,    M*H*4),"x"); die(cudaMalloc(&d_rn,H*4),"rn");
    die(cudaMalloc(&d_Qf,   M*Q*4),"Qf"); die(cudaMalloc(&d_Kf,M*KV*4),"Kf");
    die(cudaMalloc(&d_Vf,   M*KV*4),"Vf"); die(cudaMalloc(&d_attn,M*Q*4),"attn");
    die(cudaMalloc(&d_proj, M*H*4),"proj"); die(cudaMalloc(&d_res,M*H*4),"res");
    die(cudaMalloc(&d_gate, M*I*4),"gate"); die(cudaMalloc(&d_up,M*I*4),"up");
    die(cudaMalloc(&d_mlp,  M*I*4),"mlp");
    die(cudaMalloc(&d_xi,   M*H),"xi"); die(cudaMalloc(&d_xs,M*(H/16)*4),"xs");
    die(cudaMalloc(&d_Qt,   nqh*M*hd*4),"Qt");
    die(cudaMalloc(&d_Kt,   nkv*M*hd*4),"Kt");
    die(cudaMalloc(&d_Vt,   nkv*M*hd*4),"Vt");

    // ── Decode buffers ─────────────────────────────────────────────────
    float *d_xd, *d_Qd, *d_Kd, *d_Vd, *d_ad, *d_pd, *d_gd, *d_ud, *d_md, *d_rd;
    int8_t *d_xid, *d_ai, *d_mi;
    float *d_xsd, *d_as, *d_ms;
    float *d_kc, *d_vc;
    int max_seq=M+DT;

    die(cudaMalloc(&d_xd, H*4),"xd"); die(cudaMalloc(&d_Qd,Q*4),"Qd");
    die(cudaMalloc(&d_Kd, KV*4),"Kd"); die(cudaMalloc(&d_Vd,KV*4),"Vd");
    die(cudaMalloc(&d_ad, Q*4),"ad"); die(cudaMalloc(&d_pd,H*4),"pd");
    die(cudaMalloc(&d_gd, I*4),"gd"); die(cudaMalloc(&d_ud,I*4),"ud");
    die(cudaMalloc(&d_md, I*4),"md"); die(cudaMalloc(&d_rd,H*4),"rd");
    die(cudaMalloc(&d_xid, H),"xid"); die(cudaMalloc(&d_ai,Q),"ai");
    die(cudaMalloc(&d_mi, I),"mi");
    die(cudaMalloc(&d_xsd,(H/16)*4),"xsd"); die(cudaMalloc(&d_as,(Q/16)*4),"as");
    die(cudaMalloc(&d_ms, (I/16)*4),"ms");
    die(cudaMalloc(&d_kc, nkv*max_seq*hd*4),"kc");
    die(cudaMalloc(&d_vc, nkv*max_seq*hd*4),"vc");
    die(cudaMemset(d_kc,0,nkv*max_seq*hd*4),"kc");
    die(cudaMemset(d_vc,0,nkv*max_seq*hd*4),"vc");

    // ── Constants ──────────────────────────────────────────────────────
    std::vector<float> rn_h(H,1.f);
    cudaMemcpyAsync(d_rn, rn_h.data(), H*4, cudaMemcpyHostToDevice, st);

    float *d_cos,*d_sin;
    cudaMalloc(&d_cos,nqh*M*hd*4); cudaMalloc(&d_sin,nqh*M*hd*4);
    std::vector<float> cos_h(nqh*M*hd), sin_h(nqh*M*hd);
    for(int i=0;i<nqh*M*hd;++i){cos_h[i]=cosf(i*0.01f);sin_h[i]=sinf(i*0.01f);}
    cudaMemcpyAsync(d_cos,cos_h.data(),nqh*M*hd*4,cudaMemcpyHostToDevice,st);
    cudaMemcpyAsync(d_sin,sin_h.data(),nqh*M*hd*4,cudaMemcpyHostToDevice,st);

    // Input x (random FP32)
    std::vector<float> x_h(M*H);
    for(int i=0;i<M*H;++i) x_h[i]=((i*31+7)%127-63)*0.01f;
    cudaMemcpyAsync(d_x, x_h.data(), M*H*4, cudaMemcpyHostToDevice, st);
    cudaStreamSynchronize(st);

    // ==================================================================
    // PREFILL PHASE (individual kernel timings, synthetic data per kernel)
    // ==================================================================
    printf("\n=== PREFILL PHASE (M=%d, per-kernel synthetic data) ===\n\n",M);

    // Prefill GEMMs need FP4-like input (random INT8 works as FP4 bytes)
    // Allocate once per test (like prefill_benchmark.cu)
    auto mk_gemm_data = [&](int mm, int kk, int nn) {
        std::vector<int8_t> aa(mm*kk); fill_i8(aa);
        int asr=(mm+15)/16, asc=(kk+15)/16;
        std::vector<float> asv(asr*asc); fill_sc(asv);
        std::vector<int8_t> bb(kk*nn); fill_i8(bb);
        int bsr=(kk+15)/16, bsc=(nn+15)/16;
        std::vector<float> bsv(bsr*bsc); fill_sc(bsv);
        int8_t *da,*db; float *das,*dbs; float *dc;
        cudaMalloc(&da,mm*kk); cudaMalloc(&das,asv.size()*4);
        cudaMalloc(&db,kk*nn); cudaMalloc(&dbs,bsv.size()*4); cudaMalloc(&dc,mm*nn*4);
        cudaMemcpyAsync(da,aa.data(),mm*kk,cudaMemcpyHostToDevice,st);
        cudaMemcpyAsync(das,asv.data(),asv.size()*4,cudaMemcpyHostToDevice,st);
        cudaMemcpyAsync(db,bb.data(),kk*nn,cudaMemcpyHostToDevice,st);
        cudaMemcpyAsync(dbs,bsv.data(),bsv.size()*4,cudaMemcpyHostToDevice,st);
        cudaStreamSynchronize(st);
        return std::make_tuple(da,das,db,dbs,dc,mm,nn,kk);
    };
    auto free_gemm = [&](auto t){
        auto [da,das,db,dbs,dc,mm,nn,kk]=t;
        cudaFree(da);cudaFree(das);cudaFree(db);cudaFree(dbs);cudaFree(dc);
    };

    double t_prefill=0;

    // RMSNorm
    auto t_rn = bench_n(st,IT,[&]{
        blackwell::kernels::fused_rmsnorm(d_proj,d_x,d_rn,H,eps,st);});
    printf("  RMSNorm (M=%d):          %7.3f ms\n",M,t_rn);
    t_prefill+=t_rn;

    // QKV GEMMs
    auto gq=mk_gemm_data(M,H,Q);
    auto rq=[&]{blackwell::kernels::dispatch_matmul(
        std::get<4>(gq),std::get<0>(gq),std::get<2>(gq),
        std::get<1>(gq),std::get<3>(gq),M,Q,H,
        blackwell::kernels::KernelMode::Prefill,st);};
    auto t_q=bench_n(st,IT,rq);
    printf("  Q GEMM (M=%d,K=%d,N=%d):  %7.3f ms\n",M,H,Q,t_q);
    free_gemm(gq);

    auto gk=mk_gemm_data(M,H,KV);
    auto rk=[&]{blackwell::kernels::dispatch_matmul(
        std::get<4>(gk),std::get<0>(gk),std::get<2>(gk),
        std::get<1>(gk),std::get<3>(gk),M,KV,H,
        blackwell::kernels::KernelMode::Prefill,st);};
    auto t_k=bench_n(st,IT,rk);
    printf("  K GEMM (M=%d,K=%d,N=%d):  %7.3f ms\n",M,H,KV,t_k);
    free_gemm(gk);

    auto gv=mk_gemm_data(M,H,KV);
    auto rv=[&]{blackwell::kernels::dispatch_matmul(
        std::get<4>(gv),std::get<0>(gv),std::get<2>(gv),
        std::get<1>(gv),std::get<3>(gv),M,KV,H,
        blackwell::kernels::KernelMode::Prefill,st);};
    auto t_v=bench_n(st,IT,rv);
    printf("  V GEMM (M=%d,K=%d,N=%d):  %7.3f ms\n",M,H,KV,t_v);
    free_gemm(gv);
    t_prefill+=t_q+t_k+t_v;

    // RoPE (needs M tokens, random Q data)
    auto t_rq=bench_n(st,IT,[&]{
        blackwell::kernels::fused_rope(d_Qf,d_cos,d_sin,nqh,M,hd,st);});
    auto t_rk=bench_n(st,IT,[&]{
        blackwell::kernels::fused_rope(d_Kf,d_cos,d_sin,nkv,M,hd,st);});
    printf("  RoPE:                     %7.3f ms\n",t_rq+t_rk);
    t_prefill+=t_rq+t_rk;

    // Transpose (Q: [M×Q]→[nqh×M×hd], K/V: [M×KV]→[nkv×M×hd])
    auto t_tx=bench_n(st,IT,[&]{
        do_transpose(d_Qt,d_Qf,M,nqh,hd,st);
        do_transpose(d_Kt,d_Kf,M,nkv,hd,st);
        do_transpose(d_Vt,d_Vf,M,nkv,hd,st);});
    printf("  Transpose Q/K/V:          %7.3f ms\n",t_tx);
    t_prefill+=t_tx;

    // Flash attention (needs FP32 Q/K/V in correct layout)
    // Generate random Q/K/V for attention (like bench_flash_attention.cu)
    float *d_atQ,*d_atK,*d_atV,*d_atO;
    std::vector<float> atQ_h(nqh*M*hd),atK_h(nkv*M*hd),atV_h(nkv*M*hd);
    for(int i=0;i<(int)atQ_h.size();++i)atQ_h[i]=((i*17+13)%127-63)*0.01f;
    for(int i=0;i<(int)atK_h.size();++i)atK_h[i]=((i*23+7)%127-63)*0.01f;
    for(int i=0;i<(int)atV_h.size();++i)atV_h[i]=((i*31+11)%127-63)*0.01f;
    cudaMalloc(&d_atQ,nqh*M*hd*4);cudaMalloc(&d_atK,nkv*M*hd*4);
    cudaMalloc(&d_atV,nkv*M*hd*4);cudaMalloc(&d_atO,nqh*M*hd*4);
    cudaMemcpyAsync(d_atQ,atQ_h.data(),nqh*M*hd*4,cudaMemcpyHostToDevice,st);
    cudaMemcpyAsync(d_atK,atK_h.data(),nkv*M*hd*4,cudaMemcpyHostToDevice,st);
    cudaMemcpyAsync(d_atV,atV_h.data(),nkv*M*hd*4,cudaMemcpyHostToDevice,st);
    cudaStreamSynchronize(st);
    auto t_at=bench_n(st,IT,[&]{
        blackwell::kernels::attention_prefill(d_atO,d_atQ,d_atK,d_atV,
            M,hd,nqh,nkv,qpg,sc_at,st);});
    printf("  Flash attention:           %7.3f ms\n",t_at);
    t_prefill+=t_at;
    cudaFree(d_atQ);cudaFree(d_atK);cudaFree(d_atV);cudaFree(d_atO);

    // Wo GEMM (Q→H)
    auto go=mk_gemm_data(M,Q,H);
    auto ro=[&]{blackwell::kernels::dispatch_matmul(
        std::get<4>(go),std::get<0>(go),std::get<2>(go),
        std::get<1>(go),std::get<3>(go),M,H,Q,
        blackwell::kernels::KernelMode::Prefill,st);};
    auto t_wo=bench_n(st,IT,ro);
    printf("  Wo GEMM (M=%d,K=%d,N=%d): %7.3f ms\n",M,Q,H,t_wo);
    free_gemm(go);
    t_prefill+=t_wo;

    // Residual add (elementwise)
    auto t_add=bench_n(st,IT,[&]{
        blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_x,M*H,st);});
    printf("  Vector add (residual):     %7.3f ms\n",t_add);
    t_prefill+=t_add;

    // MLP: Gate/Up GEMMs (H→I)
    auto gg=mk_gemm_data(M,H,I);
    auto rg=[&]{blackwell::kernels::dispatch_matmul(
        std::get<4>(gg),std::get<0>(gg),std::get<2>(gg),
        std::get<1>(gg),std::get<3>(gg),M,I,H,
        blackwell::kernels::KernelMode::Prefill,st);};
    auto t_gu=bench_n(st,IT,rg);
    auto gu=mk_gemm_data(M,H,I);
    auto ru=[&]{blackwell::kernels::dispatch_matmul(
        std::get<4>(gu),std::get<0>(gu),std::get<2>(gu),
        std::get<1>(gu),std::get<3>(gu),M,I,H,
        blackwell::kernels::KernelMode::Prefill,st);};
    auto t_up=bench_n(st,IT,ru);
    printf("  Gate+Up GEMMs (H→I):      %7.3f ms\n",t_gu+t_up);
    free_gemm(gg);free_gemm(gu);
    t_prefill+=t_gu+t_up;

    // SwiGLU
    auto t_sg=bench_n(st,IT,[&]{
        blackwell::kernels::apply_swiglu(d_mlp,d_gate,d_up,M*I,st);});
    printf("  SwiGLU:                   %7.3f ms\n",t_sg);
    t_prefill+=t_sg;

    // Down GEMM (I→H)
    auto gd=mk_gemm_data(M,I,H);
    auto rd=[&]{blackwell::kernels::dispatch_matmul(
        std::get<4>(gd),std::get<0>(gd),std::get<2>(gd),
        std::get<1>(gd),std::get<3>(gd),M,H,I,
        blackwell::kernels::KernelMode::Prefill,st);};
    auto t_dn=bench_n(st,IT,rd);
    printf("  Down GEMM (I→H):          %7.3f ms\n",t_dn);
    free_gemm(gd);
    t_prefill+=t_dn;

    // Second residual
    auto t_add2=bench_n(st,IT,[&]{
        blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_res,M*H,st);});
    printf("  Vector add (residual2):    %7.3f ms\n",t_add2);
    t_prefill+=t_add2;

    printf("  ──────────────────────────────────\n");
    printf("  TOTAL per layer:          %7.3f ms\n",t_prefill);
    printf("  28L total:                %7.3f ms\n",t_prefill*28);

    // ==================================================================
    // DECODE PHASE (full connected pipeline, INT8)
    // ==================================================================
    printf("\n=== DECODE PHASE (%d tokens, INT8 connected pipeline) ===\n\n",DT);

    // Seed decode input
    cudaMemcpyAsync(d_xd, x_h.data()+(M-1)*H, H*4, cudaMemcpyHostToDevice, st);
    cudaStreamSynchronize(st);

    // Zero KV cache (re-init)
    cudaMemsetAsync(d_kc,0,nkv*max_seq*hd*4,st);
    cudaMemsetAsync(d_vc,0,nkv*max_seq*hd*4,st);

    // Fill KV cache with prefill tokens (use synthetic K/V)
    std::vector<float> kv_h(M*KV);
    for(int i=0;i<M*KV;++i)kv_h[i]=((i*23+7)%127-63)*0.01f;
    for(int t=0;t<M;++t){
        cudaMemcpyAsync(d_Kd,kv_h.data()+t*KV,KV*4,cudaMemcpyHostToDevice,st);
        cudaMemcpyAsync(d_Vd,kv_h.data()+t*KV,KV*4,cudaMemcpyHostToDevice,st);
        blackwell::kernels::update_kv_cache(d_kc,d_vc,d_Kd,d_Vd,0,t,nkv,hd,max_seq,st);
    }
    cudaStreamSynchronize(st);

    double t_dec=0;
    int seq_pos=M;
    GpuTimer tm_dec;

    for(int t=0;t<DT;++t){
        tm_dec.start();
        // 1. RMSNorm+quant
        die(blackwell::kernels::fused_rmsnorm_quant_int8(d_xid,d_xsd,d_xd,d_rn,H,eps,st),"rmsnorm");
        // 2-4. QKV GEMV
        die(blackwell::kernels::gemv_int8(d_Qd,d_xid,d_xsd,w.qt.d,w.qt.sc,H,Q,st),"q");
        die(blackwell::kernels::gemv_int8(d_Kd,d_xid,d_xsd,w.kt.d,w.kt.sc,H,KV,st),"k");
        die(blackwell::kernels::gemv_int8(d_Vd,d_xid,d_xsd,w.vt.d,w.vt.sc,H,KV,st),"v");
        // 5. KV cache
        die(blackwell::kernels::update_kv_cache(d_kc,d_vc,d_Kd,d_Vd,0,seq_pos,nkv,hd,max_seq,st),"kv");
        // 6. Decode attention
        die(blackwell::kernels::attention_decode_gqa(d_ad,d_Qd,d_kc,d_vc,seq_pos,nqh,nkv,hd,max_seq,st),"attn");
        // 7. Wo GEMV (pack→gemv)
        die(blackwell::kernels::pack_int8(d_ai,d_ad,d_as,Q,st),"pack");
        die(blackwell::kernels::gemv_int8(d_pd,d_ai,d_as,w.ot.d,w.ot.sc,Q,H,st),"wo");
        // 8. Residual
        die(blackwell::kernels::vector_add_fp32(d_pd,d_pd,d_xd,H,st),"res");
        // 9. RMSNorm+quant for MLP
        die(blackwell::kernels::fused_rmsnorm_quant_int8(d_xid,d_xsd,d_pd,d_rn,H,eps,st),"rn2");
        // 10. Gate+Up GEMV
        die(blackwell::kernels::gemv_int8(d_gd,d_xid,d_xsd,w.gt.d,w.gt.sc,H,I,st),"gate");
        die(blackwell::kernels::gemv_int8(d_ud,d_xid,d_xsd,w.ut.d,w.ut.sc,H,I,st),"up");
        // 11. SwiGLU
        die(blackwell::kernels::apply_swiglu(d_md,d_gd,d_ud,I,st),"swiglu");
        // 12. Down GEMV
        die(blackwell::kernels::pack_int8(d_mi,d_md,d_ms,I,st),"pack2");
        die(blackwell::kernels::gemv_int8(d_pd,d_mi,d_ms,w.dt.d,w.dt.sc,I,H,st),"down");
        // 13. Residual
        die(blackwell::kernels::vector_add_fp32(d_xd,d_pd,d_xd,H,st),"res2");
        ++seq_pos;
        t_dec+=tm_dec.stop();
    }

    double tps=1000.0/(t_dec/DT);
    printf("  Per token:  %8.1f us  (%.2f ms / %d)\n", t_dec/DT*1000, t_dec, DT);
    printf("  Throughput: %8.0f t/s\n", tps);

    // ── SUMMARY ────────────────────────────────────────────────────────
    printf("\n=== SUMMARY (1 layer) ===\n");
    printf("  Prefill M=%d:  %.3f ms  (%.0f t/s)\n", M, t_prefill, M/(t_prefill/1000));
    printf("  Decode %d:    %.3f ms  (%.0f t/s)\n", DT, t_dec, tps);
    printf("  28L prefill:  %.3f ms\n", t_prefill*28);
    printf("  28L decode:   %.3f ms  (%.0f t/s)\n", t_dec*28, tps);
    printf("  Total (1L):   %.3f ms\n", t_prefill+t_dec);

    // Cleanup
    cudaFree(d_x);cudaFree(d_rn);cudaFree(d_Qf);cudaFree(d_Kf);cudaFree(d_Vf);
    cudaFree(d_attn);cudaFree(d_proj);cudaFree(d_res);
    cudaFree(d_gate);cudaFree(d_up);cudaFree(d_mlp);
    cudaFree(d_xi);cudaFree(d_xs);
    cudaFree(d_Qt);cudaFree(d_Kt);cudaFree(d_Vt);
    cudaFree(d_xd);cudaFree(d_Qd);cudaFree(d_Kd);cudaFree(d_Vd);
    cudaFree(d_ad);cudaFree(d_pd);cudaFree(d_gd);cudaFree(d_ud);cudaFree(d_md);cudaFree(d_rd);
    cudaFree(d_xid);cudaFree(d_ai);cudaFree(d_mi);
    cudaFree(d_xsd);cudaFree(d_as);cudaFree(d_ms);
    cudaFree(d_kc);cudaFree(d_vc);
    cudaFree(d_cos);cudaFree(d_sin);
    cudaStreamDestroy(st);
    return 0;
}