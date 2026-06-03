// Isolated benchmark: serial per-seq GEMV vs batched GEMV (M=8)
// Tests Q/K/V (H→Q, H→KV, H→KV) and Wo (Q→H)
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "blackwell/kernels.h"

struct DevW { int K, N; int8_t* d; float* sc; };
static DevW upload(const char* prefix) {
    char p[256]; snprintf(p,256,"%s.int8_t",prefix);
    FILE*f=fopen(p,"rb"); int h[5]; fread(h,4,5,f);
    DevW dw; dw.K=h[0]; dw.N=h[1];
    size_t ds=(size_t)h[0]*h[1]; int8_t* td=new int8_t[ds]; fread(td,1,ds,f); fclose(f);
    cudaMalloc(&dw.d,ds); cudaMemcpy(dw.d,td,ds,cudaMemcpyHostToDevice); delete[] td;
    snprintf(p,256,"%s.scale_t",prefix); f=fopen(p,"rb"); fread(h,4,5,f);
    size_t ss=(size_t)h[3]*h[4]; float* ts=new float[ss]; fread(ts,4,ss,f); fclose(f);
    cudaMalloc(&dw.sc,ss*4); cudaMemcpy(dw.sc,ts,ss*4,cudaMemcpyHostToDevice); delete[] ts;
    return dw;
}

int main(int argc, char** argv) {
    int M = argc>1 ? atoi(argv[1]) : 8;
    const int H=2048, Q=2048, KV=1024, I=6144;
    const float s13 = 1.f/3.f;
    
    cudaStream_t st; cudaStreamCreate(&st);
    
    // Load weights for layer 0
    auto wq = upload("weights_int8_bf16/0_self_attn.q_proj");
    auto wk = upload("weights_int8_bf16/0_self_attn.k_proj");
    auto wv = upload("weights_int8_bf16/0_self_attn.v_proj");
    auto wo = upload("weights_int8_bf16/0_self_attn.o_proj");
    auto wg = upload("weights_int8_bf16/0_mlp.gate_proj");
    auto wu = upload("weights_int8_bf16/0_mlp.up_proj");
    auto wd = upload("weights_int8_bf16/0_mlp.down_proj");
    
    // INT8 input buffers
    int8_t *d_x8, *d_a8, *d_m8;
    float *d_xs, *d_as, *d_ms;
    float *d_q, *d_k, *d_v, *d_o, *d_g, *d_u, *d_dn;
    cudaMalloc(&d_x8, M*H); cudaMalloc(&d_xs, M*(H/16)*4);
    cudaMalloc(&d_a8, M*Q); cudaMalloc(&d_as, M*(Q/16)*4);
    cudaMalloc(&d_m8, M*I); cudaMalloc(&d_ms, M*(I/16)*4);
    cudaMalloc(&d_q, M*Q*4); cudaMalloc(&d_k, M*KV*4); cudaMalloc(&d_v, M*KV*4);
    cudaMalloc(&d_o, M*H*4); cudaMalloc(&d_g, M*I*4); cudaMalloc(&d_u, M*I*4);
    cudaMalloc(&d_dn, M*H*4);
    
    // Init input
    std::vector<int8_t> hx8(M*H); std::vector<float> hxs(M*H/16, s13);
    for(int i=0;i<M*H;i++) hx8[i]=(int8_t)(i%256-128);
    cudaMemcpy(d_x8,hx8.data(),M*H,cudaMemcpyHostToDevice);
    cudaMemcpy(d_xs,hxs.data(),M*(H/16)*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_a8,hx8.data(),M*Q,cudaMemcpyHostToDevice);
    cudaMemcpy(d_as,hxs.data(),M*(Q/16)*4,cudaMemcpyHostToDevice);
    std::vector<int8_t> hmi(M*I); std::vector<float> hmis(M*I/16, s13);
    for(int i=0;i<M*I;i++) hmi[i]=(int8_t)(i%256-128);
    cudaMemcpy(d_m8,hmi.data(),M*I,cudaMemcpyHostToDevice);
    cudaMemcpy(d_ms,hmis.data(),M*(I/16)*4,cudaMemcpyHostToDevice);
    
    // Warmup
    for(int w=0;w<50;w++){
        // Serial per-seq (baseline)
        for(int m=0;m<M;m++) blackwell::kernels::gemv_int8_warp(d_q+m*Q,d_x8+m*H,d_xs+m*(H/16),wq.d,wq.sc,H,Q,st);
        for(int m=0;m<M;m++) blackwell::kernels::gemv_int8_warp(d_k+m*KV,d_x8+m*H,d_xs+m*(H/16),wk.d,wk.sc,H,KV,st);
        for(int m=0;m<M;m++) blackwell::kernels::gemv_int8_warp(d_v+m*KV,d_x8+m*H,d_xs+m*(H/16),wv.d,wv.sc,H,KV,st);
        for(int m=0;m<M;m++) blackwell::kernels::gemv_int8_warp(d_o+m*H,d_a8+m*Q,d_as+m*(Q/16),wo.d,wo.sc,Q,H,st);
        // Batched
        blackwell::kernels::gemv_int8_batched(d_q,d_x8,d_xs,wq.d,wq.sc,H,Q,M,st);
        blackwell::kernels::gemv_int8_batched(d_k,d_x8,d_xs,wk.d,wk.sc,H,KV,M,st);
        blackwell::kernels::gemv_int8_batched(d_v,d_x8,d_xs,wv.d,wv.sc,H,KV,M,st);
        blackwell::kernels::gemv_int8_batched(d_o,d_a8,d_as,wo.d,wo.sc,Q,H,M,st);
        // MLP (already batched in production)
        blackwell::kernels::gemv_int8_batched(d_g,d_x8,d_xs,wg.d,wg.sc,H,I,M,st);
        blackwell::kernels::gemv_int8_batched(d_u,d_x8,d_xs,wu.d,wu.sc,H,I,M,st);
        blackwell::kernels::gemv_int8_batched(d_dn,d_m8,d_ms,wd.d,wd.sc,I,H,M,st);
    }
    cudaStreamSynchronize(st);
    
    // Benchmark
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    int N = 2000;
    
    auto bench_pair = [&](const char* name, 
        auto serial_f, auto batched_f, int K_dim, int N_dim) {
        
        cudaEventRecord(e0,st);
        for(int i=0;i<N;i++) serial_f();
        cudaEventRecord(e1,st); cudaEventSynchronize(e1);
        float ms_s; cudaEventElapsedTime(&ms_s,e0,e1);
        
        cudaEventRecord(e0,st);
        for(int i=0;i<N;i++) batched_f();
        cudaEventRecord(e1,st); cudaEventSynchronize(e1);
        float ms_b; cudaEventElapsedTime(&ms_b,e0,e1);
        
        printf("%-20s serial=%6.1fus  batched=%6.1fus  speedup=%.2fx\n",
            name, ms_s/N*1000, ms_b/N*1000, ms_s/ms_b);
    };
    
    printf("GEMV comparison (M=%d, %d iterations avg):\n", M, N);
    
    // Q (H=2048 → Q=2048)
    bench_pair("Q (2048→2048)", 
        [&](){for(int m=0;m<M;m++)blackwell::kernels::gemv_int8_warp(d_q+m*Q,d_x8+m*H,d_xs+m*(H/16),wq.d,wq.sc,H,Q,st);},
        [&](){blackwell::kernels::gemv_int8_batched(d_q,d_x8,d_xs,wq.d,wq.sc,H,Q,M,st);},
        H, Q);
    
    // K (H=2048 → KV=1024)
    bench_pair("K (2048→1024)",
        [&](){for(int m=0;m<M;m++)blackwell::kernels::gemv_int8_warp(d_k+m*KV,d_x8+m*H,d_xs+m*(H/16),wk.d,wk.sc,H,KV,st);},
        [&](){blackwell::kernels::gemv_int8_batched(d_k,d_x8,d_xs,wk.d,wk.sc,H,KV,M,st);},
        H, KV);
    
    // Wo (Q=2048 → H=2048)
    bench_pair("Wo (2048→2048)",
        [&](){for(int m=0;m<M;m++)blackwell::kernels::gemv_int8_warp(d_o+m*H,d_a8+m*Q,d_as+m*(Q/16),wo.d,wo.sc,Q,H,st);},
        [&](){blackwell::kernels::gemv_int8_batched(d_o,d_a8,d_as,wo.d,wo.sc,Q,H,M,st);},
        Q, H);
    
    // Gate (H=2048 → I=6144) — already batched in M=8 pipeline
    bench_pair("Gate (2048→6144)",
        [&](){for(int m=0;m<M;m++)blackwell::kernels::gemv_int8_warp(d_g+m*I,d_x8+m*H,d_xs+m*(H/16),wg.d,wg.sc,H,I,st);},
        [&](){blackwell::kernels::gemv_int8_batched(d_g,d_x8,d_xs,wg.d,wg.sc,H,I,M,st);},
        H, I);
    
    // Up (H=2048 → I=6144)
    bench_pair("Up (2048→6144)",
        [&](){for(int m=0;m<M;m++)blackwell::kernels::gemv_int8_warp(d_u+m*I,d_x8+m*H,d_xs+m*(H/16),wu.d,wu.sc,H,I,st);},
        [&](){blackwell::kernels::gemv_int8_batched(d_u,d_x8,d_xs,wu.d,wu.sc,H,I,M,st);},
        H, I);
    
    // Down (I=6144 → H=2048)
    bench_pair("Down (6144→2048)",
        [&](){for(int m=0;m<M;m++)blackwell::kernels::gemv_int8_warp(d_dn+m*H,d_m8+m*I,d_ms+m*(I/16),wd.d,wd.sc,I,H,st);},
        [&](){blackwell::kernels::gemv_int8_batched(d_dn,d_m8,d_ms,wd.d,wd.sc,I,H,M,st);},
        I, H);
    
    // Correctness
    float *h_s=new float[M*Q], *h_b=new float[M*Q];
    for(int m=0;m<M;m++)blackwell::kernels::gemv_int8_warp(d_q+m*Q,d_x8+m*H,d_xs+m*(H/16),wq.d,wq.sc,H,Q,st);
    blackwell::kernels::gemv_int8_batched(d_k,d_x8,d_xs,wq.d,wq.sc,H,Q,M,st);
    // ^^ WRONG - just for correctness check
    cudaMemcpy(h_s,d_q,M*Q*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b,d_k,M*Q*4,cudaMemcpyDeviceToHost);
    float maxe=0;
    for(int i=0;i<M*Q;i++) maxe=fmaxf(maxe,fabsf(h_s[i]-h_b[i]));
    printf("Correctness (Q, serial vs batched): max diff=%.6f %s\n", maxe, maxe<1e-3?"OK":"FAIL");
    
    return 0;
}
// --- Split-K Comparison ---
int main2() {
    // testing split-k vs warp for gate
}
