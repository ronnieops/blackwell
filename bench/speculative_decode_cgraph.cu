// bench/speculative_decode_cgraph.cu — Speculative decode with CUDA Graph
//
// Optimizes batched speculative path using CUDA Graph for the per-token decode.
// Captures one token's full layer pipeline into a graph, replays M times per step.
// Saves ~15 kernel launches × NL layers per step for the batched path.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>
#include <cmath>
#include "blackwell/kernels.h"

struct GpuTimer { cudaEvent_t s,e; GpuTimer(){cudaEventCreate(&s);cudaEventCreate(&e);} ~GpuTimer(){cudaEventDestroy(s);cudaEventDestroy(e);} void start(){cudaEventRecord(s,0);} float stop(){cudaEventRecord(e,0);cudaEventSynchronize(e);float m=0;cudaEventElapsedTime(&m,s,e);return m;} };
static void die(cudaError_t e, const char* m){if(e!=cudaSuccess){printf("FAIL %s: %s\n",m,cudaGetErrorString(e));exit(1);}}
static void chk(cudaError_t e){if(e){printf("CUDA err %d\n",e);exit(1);}}

struct IW { int K,N; int8_t*d; float*ds; };
static IW load_iw(const char* d, const char* n){
    char p[256]; snprintf(p,256,"%s/%s.int8_t",d,n); FILE*f=fopen(p,"rb");
    if(!f){printf("Cannot open %s\n",p);exit(1);}
    int h[5]; size_t r=fread(h,4,5,f); if(r!=5){printf("Bad header %s\n",p);exit(1);}
    IW w{h[0],h[1],nullptr,nullptr};
    std::vector<int8_t> tmp((size_t)w.K*w.N);
    size_t nr=fread(tmp.data(),1,(size_t)w.K*w.N,f);
    if(nr!=(size_t)w.K*w.N){printf("Bad data %s\n",p);exit(1);}
    fclose(f);
    cudaMalloc(&w.d,(size_t)w.K*w.N);
    cudaMemcpy(w.d,tmp.data(),(size_t)w.K*w.N,cudaMemcpyHostToDevice);
    snprintf(p,256,"%s/%s.scale_t",d,n); f=fopen(p,"rb");
    if(!f){printf("Cannot open %s\n",p);exit(1);}
    r=fread(h,4,5,f); if(r!=5){printf("Bad scale header %s\n",p);exit(1);}
    size_t ns=(size_t)h[3]*h[4];
    std::vector<float> tmp_s(ns);
    nr=fread(tmp_s.data(),4,ns,f);
    if(nr!=ns){printf("Bad scale data %s\n",p);exit(1);}
    fclose(f);
    cudaMalloc(&w.ds,ns*4);
    cudaMemcpy(w.ds,tmp_s.data(),ns*4,cudaMemcpyHostToDevice);
    return w;
}
struct LW { IW q,k,v,o,gate,up,down;
    void load(const char*d,int l){
        char b[128];
        snprintf(b,128,"%d_self_attn.q_proj",l); q=load_iw(d,b);
        snprintf(b,128,"%d_self_attn.k_proj",l); k=load_iw(d,b);
        snprintf(b,128,"%d_self_attn.v_proj",l); v=load_iw(d,b);
        snprintf(b,128,"%d_self_attn.o_proj",l); o=load_iw(d,b);
        snprintf(b,128,"%d_mlp.gate_proj",l);  gate=load_iw(d,b);
        snprintf(b,128,"%d_mlp.up_proj",l);    up=load_iw(d,b);
        snprintf(b,128,"%d_mlp.down_proj",l);  down=load_iw(d,b);
    }
    void free_(){cudaFree(q.d);cudaFree(q.ds);cudaFree(k.d);cudaFree(k.ds);cudaFree(v.d);cudaFree(v.ds);cudaFree(o.d);cudaFree(o.ds);cudaFree(gate.d);cudaFree(gate.ds);cudaFree(up.d);cudaFree(up.ds);cudaFree(down.d);cudaFree(down.ds);}
};

static constexpr int H=2048, QD=2048, KV=1024, ID=6144, B=16;
static constexpr float eps=1e-6f;

static int* d_kv_offset = nullptr;
static int* h_kv_offset = nullptr;  // pinned host memory for graph capture

// ── Build CUDA Graph for single-token per-layer decode ────────────────────
// Captures 1 layer's decode (attention + MLP) with kv_offset in device mem
// Uses cudaMemcpyAsync H2D for kv_offset update (captured in graph)
static void build_token_graph(LW& L, cudaGraphExec_t* ge,
    float* d_Q, float* d_K, float* d_V, float* d_attn, float* d_proj,
    float* d_gate, float* d_up, float* d_mlp, float* d_rn,
    int8_t* d_xi, float* d_xs, int8_t* d_ai, float* d_as, int8_t* d_mi, float* d_ms,
    float* d_res, float* d_kc, float* d_vc, int sq)
{
    cudaStream_t st = 0;
    // Init buffers with non-zero data (for warm-up kernel calls)
    std::vector<int8_t> init_xi(H, 1);
    cudaMemcpy(d_xi, init_xi.data(), H, cudaMemcpyHostToDevice);
    std::vector<float> ones_fp(H, 0.5f);
    cudaMemcpy(d_res, ones_fp.data(), H*4, cudaMemcpyHostToDevice);
    *h_kv_offset = 0; printf("  Warmup...\n"); fflush(stdout);
    cudaMemcpy(d_kv_offset, h_kv_offset, sizeof(int), cudaMemcpyHostToDevice);

    // Warm-up: call all kernels once BEFORE capture (triggers static allocation)
    // This avoids "operation not permitted when stream is capturing" errors
    // from cudaMalloc inside static guards.
    blackwell::kernels::gemv_int8_warp(d_Q,d_xi,d_xs,L.q.d,L.q.ds,H,QD,st);
    blackwell::kernels::gemv_int8_warp(d_K,d_xi,d_xs,L.k.d,L.k.ds,H,KV,st);
    blackwell::kernels::gemv_int8_warp(d_V,d_xi,d_xs,L.v.d,L.v.ds,H,KV,st);
    blackwell::kernels::update_kv_cache(d_kc,d_vc,d_K,d_V,0,sq,8,128,H,st);
    blackwell::kernels::attention_decode_gqa(d_attn,d_Q,d_kc,d_vc,sq,16,8,128,H,st);
    blackwell::kernels::pack_int8(d_ai,d_attn,d_as,QD,st);
    blackwell::kernels::gemv_int8_warp(d_proj,d_ai,d_as,L.o.d,L.o.ds,QD,H,st);
    blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_res,H,st);
    blackwell::kernels::fused_rmsnorm_quant_int8(d_xi,d_xs,d_proj,d_rn,H,eps,st);
    blackwell::kernels::gemv_int8_warp(d_gate,d_xi,d_xs,L.gate.d,L.gate.ds,H,ID,st);
    blackwell::kernels::gemv_int8_warp(d_up,d_xi,d_xs,L.up.d,L.up.ds,H,ID,st);
    blackwell::kernels::apply_swiglu(d_mlp,d_gate,d_up,ID,st);
    blackwell::kernels::pack_int8(d_mi,d_mlp,d_ms,ID,st);
    blackwell::kernels::gemv_int8_warp(d_proj,d_mi,d_ms,L.down.d,L.down.ds,ID,H,st);
    // Full device sync ensures all static allocations (cudaMalloc in decode.cu) complete
    cudaDeviceSynchronize();
    cudaError_t we = cudaGetLastError();
    if (we != cudaSuccess) { printf("Warm-up error: %s\n", cudaGetErrorString(we)); exit(1); }

    cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);

    // Copy kv_offset H2D (captured in graph)
    // NOTE: d_kv_offset[0] is read during capture and baked into kernel args.
    // The H2D copy updates device memory but kernel argument pointers are fixed.
    // Each layer gets its own graph with correct per-layer offset.
    cudaMemcpyAsync(d_kv_offset, h_kv_offset, sizeof(int), cudaMemcpyHostToDevice, st);

    // Attention
    blackwell::kernels::gemv_int8_warp(d_Q,d_xi,d_xs,L.q.d,L.q.ds,H,QD,st);
    blackwell::kernels::gemv_int8_warp(d_K,d_xi,d_xs,L.k.d,L.k.ds,H,KV,st);
    blackwell::kernels::gemv_int8_warp(d_V,d_xi,d_xs,L.v.d,L.v.ds,H,KV,st);
    blackwell::kernels::update_kv_cache(d_kc+d_kv_offset[0],d_vc+d_kv_offset[0],
        d_K,d_V,0,sq,8,128,H,st);
    blackwell::kernels::attention_decode_gqa(d_attn,d_Q,
        d_kc+d_kv_offset[0],d_vc+d_kv_offset[0],sq,16,8,128,H,st);
    blackwell::kernels::pack_int8(d_ai,d_attn,d_as,QD,st);
    blackwell::kernels::gemv_int8_warp(d_proj,d_ai,d_as,L.o.d,L.o.ds,QD,H,st);
    blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_res,H,st);
    cudaMemcpyAsync(d_res,d_proj,H*4,cudaMemcpyDeviceToDevice,st);
    blackwell::kernels::fused_rmsnorm_quant_int8(d_xi,d_xs,d_proj,d_rn,H,eps,st);
    // MLP
    blackwell::kernels::gemv_int8_warp(d_gate,d_xi,d_xs,L.gate.d,L.gate.ds,H,ID,st);
    blackwell::kernels::gemv_int8_warp(d_up,d_xi,d_xs,L.up.d,L.up.ds,H,ID,st);
    blackwell::kernels::apply_swiglu(d_mlp,d_gate,d_up,ID,st);
    blackwell::kernels::pack_int8(d_mi,d_mlp,d_ms,ID,st);
    blackwell::kernels::gemv_int8_warp(d_proj,d_mi,d_ms,L.down.d,L.down.ds,ID,H,st);
    blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_res,H,st);
    cudaMemcpyAsync(d_xi,d_proj,H,cudaMemcpyDeviceToDevice,st);

    cudaGraph_t gr;
    cudaError_t ce = cudaStreamEndCapture(st,&gr);
    if(ce!=cudaSuccess){printf("FAIL capture: %s\n",cudaGetErrorString(ce));exit(1);}
    ce=cudaGraphInstantiate(ge,gr,NULL,NULL,0);
    if(ce!=cudaSuccess){printf("FAIL instantiate: %s\n",cudaGetErrorString(ce));exit(1);}
    cudaGraphDestroy(gr);
}

int main(int argc, char** argv) {
    int NL=4, M=4, steps=20;
    if(argc>1)NL=atoi(argv[1]);
    if(argc>2)M=atoi(argv[2]);
    if(argc>3)steps=atoi(argv[3]);

    cudaDeviceProp p;cudaGetDeviceProperties(&p,0);
    printf("# Speculative Decode CUDA Graph — Qwen3-1.7B\n");
    printf("Device: %s (CC %d.%d)\n",p.name,p.major,p.minor);
    printf("Layers: %d, Drafts M: %d, Steps: %d\n",NL,M,steps);

    const char* dir="weights_int8_bf16";
    printf("Loading weights...\n");
    std::vector<LW> L(NL);
    for(int i=0;i<NL;++i)L[i].load(dir,i);

    // Allocate pinned host + device memory for kv_cache offset (CUDA Graph)
    chk(cudaMalloc(&d_kv_offset, sizeof(int)));
    chk(cudaMallocHost(&h_kv_offset, sizeof(int)));
    *h_kv_offset = 0;

    // Per-token buffers
    float*d_Q,*d_K,*d_V,*d_attn,*d_proj,*d_gate,*d_up,*d_mlp,*d_rn;
    float*d_res;
    int8_t*d_xi,*d_ai,*d_mi;
    float*d_xs,*d_as,*d_ms;
    chk(cudaMalloc(&d_Q,H*4));chk(cudaMalloc(&d_K,KV*4));chk(cudaMalloc(&d_V,KV*4));
    chk(cudaMalloc(&d_attn,QD*4));chk(cudaMalloc(&d_proj,H*4));
    chk(cudaMalloc(&d_gate,ID*4));chk(cudaMalloc(&d_up,ID*4));chk(cudaMalloc(&d_mlp,ID*4));
    chk(cudaMalloc(&d_rn,H*4));chk(cudaMalloc(&d_res,H*4));
    std::vector<float>ones(H,1.f);cudaMemcpy(d_rn,ones.data(),H*4,cudaMemcpyHostToDevice);
    chk(cudaMalloc(&d_xi,H));chk(cudaMalloc(&d_xs,(H/B)*4));
    chk(cudaMalloc(&d_ai,QD));chk(cudaMalloc(&d_as,(QD/B)*4));
    chk(cudaMalloc(&d_mi,ID));chk(cudaMalloc(&d_ms,(ID/B)*4));
    float ixv=1.f/127.f;
    std::vector<float>xs(H/B,ixv),xa(QD/B,ixv),xm(ID/B,ixv);
    cudaMemcpy(d_xs,xs.data(),(H/B)*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_as,xa.data(),(QD/B)*4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_ms,xm.data(),(ID/B)*4,cudaMemcpyHostToDevice);

    // Batched buffers
    int8_t*d_xM;float*d_xMs;
    chk(cudaMalloc(&d_xM,M*H));chk(cudaMalloc(&d_xMs,M*(H/B)*4));
    float*d_projM;chk(cudaMalloc(&d_projM,M*H*4));
    float*d_gateM,*d_upM,*d_mlpM;
    chk(cudaMalloc(&d_gateM,M*ID*4));chk(cudaMalloc(&d_upM,M*ID*4));chk(cudaMalloc(&d_mlpM,M*ID*4));
    int8_t*d_miM;chk(cudaMalloc(&d_miM,M*ID));
    float*d_miMs;chk(cudaMalloc(&d_miMs,M*(ID/B)*4));
    float*d_resM;chk(cudaMalloc(&d_resM,M*H*4));

    // Init residual / x buffers
    chk(cudaMemset(d_res,0,H*4));chk(cudaMemset(d_resM,0,M*H*4));
    chk(cudaMemset(d_xi,0,H));chk(cudaMemset(d_xM,0,M*H));

    // KV cache
    float*d_kc,*d_vc;
    size_t kv_sz=(size_t)NL*M*8*H*128*4;
    chk(cudaMalloc(&d_kc,kv_sz));chk(cudaMalloc(&d_vc,kv_sz));
    chk(cudaMemset(d_kc,0,kv_sz));chk(cudaMemset(d_vc,0,kv_sz));

    int sq=128;

    // ── Build per-layer CUDA Graphs ────────────────────────────────────────
    printf("Building per-layer CUDA Graphs...\n");
    std::vector<cudaGraphExec_t> graphs(NL);
    // Build graph for each layer
    cudaStream_t gst=0;
    for(int l=0;l<NL;++l){
        // Set initial kv_offset = 0 (will be updated before each graph launch)
        *h_kv_offset = l * M * 8 * H * 128;  // base offset for this layer
        cudaStreamSynchronize(0);
        build_token_graph(L[l], &graphs[l],
            d_Q,d_K,d_V,d_attn,d_proj,d_gate,d_up,d_mlp,d_rn,
            d_xi,d_xs,d_ai,d_as,d_mi,d_ms,d_res,d_kc,d_vc,sq);
        printf("  Layer %d graph built.\n",l);
    }

    // ══════════════════════════════════════════════════════════════════
    // Mode A': CUDA Graph autoregressive
    // ══════════════════════════════════════════════════════════════════
    printf("\n=== Mode A': CUDA Graph autoregressive (%dL, %d tok) ===\n",NL,steps);
    GpuTimer ta;ta.start();
    for(int i=0;i<steps;++i){
        for(int l=0;l<NL;++l){
            // Update kv_offset for this layer + seq_pos
            *h_kv_offset = l * M * 8 * H * 128;
            cudaGraphLaunch(graphs[l],0);
        }
    }
    cudaDeviceSynchronize();float ms_a=ta.stop();
    float tp_a=steps*1000.f/ms_a;
    float s28_a=1000.f/(ms_a/steps*28.f/NL);
    printf("  Total: %.2f ms, Per-token: %.3f ms, t/s: %.1f, Scaled28: %.1f\n",ms_a,ms_a/steps,tp_a,s28_a);

    // ══════════════════════════════════════════════════════════════════
    // Mode B: Speculative + CUDA Graph per-token
    // ══════════════════════════════════════════════════════════════════
    printf("\n=== Mode B: Speculative CUDA Graph (M=%d drafts) ===\n",M);
    GpuTimer tb;tb.start();
    for(int i=0;i<steps;++i){
        // Draft tokens (batched MLP path using per-token graph)
        for(int m=0;m<M;++m){
            for(int l=0;l<NL;++l){
                *h_kv_offset = l * M * 8 * H * 128 + m * 8 * H * 128;
                cudaGraphLaunch(graphs[l],0);
            }
        }
        // Target token (1 more pass using same graphs)
        for(int l=0;l<NL;++l){
            *h_kv_offset = l * M * 8 * H * 128 + 0;
            cudaGraphLaunch(graphs[l],0);
        }
    }
    cudaDeviceSynchronize();float ms_b=tb.stop();
    float tokens_per_step=M+1;
    float tp_b=tokens_per_step*steps*1000.f/ms_b;
    float s28_b=1000.f/(ms_b/steps*28.f/NL);
    printf("  Total: %.2f ms, M+1 tokens/step: %.1f, t/s: %.1f, Scaled28: %.1f\n",ms_b,tokens_per_step,tp_b,s28_b);
    printf("  Speedup vs autoregressive: %.2fx (%.1f%%)\n",tp_b/tp_a,(tp_b/tp_a-1)*100.f);

    printf("\n=== Comparison ===\n");
    printf("  %-30s  %8s  %8s  %8s\n","Method","ms/step","t/s","Scaled28");
    printf("  %-30s  %7.3fms  %7.1f   %7.1f\n","Auto CUDA Graph",ms_a/steps,tp_a,s28_a);
    printf("  %-30s  %7.3fms  %7.1f   %7.1f\n","Spec CUDA Graph",ms_b/steps,tp_b,s28_b);
    printf("  Improvement: %.1f%% more tokens/s\n",(tp_b/tp_a-1)*100.f);

    // Cleanup
    for(int l=0;l<NL;++l)cudaGraphExecDestroy(graphs[l]);
    cudaFree(d_kv_offset);
    cudaFreeHost(h_kv_offset);
    for(auto&l:L)l.free_();
    cudaFree(d_Q);cudaFree(d_K);cudaFree(d_V);cudaFree(d_attn);cudaFree(d_proj);
    cudaFree(d_gate);cudaFree(d_up);cudaFree(d_mlp);cudaFree(d_rn);cudaFree(d_res);
    cudaFree(d_xi);cudaFree(d_xs);cudaFree(d_ai);cudaFree(d_as);cudaFree(d_mi);cudaFree(d_ms);
    cudaFree(d_xM);cudaFree(d_xMs);cudaFree(d_projM);
    cudaFree(d_gateM);cudaFree(d_upM);cudaFree(d_mlpM);
    cudaFree(d_miM);cudaFree(d_miMs);cudaFree(d_resM);
    cudaFree(d_kc);cudaFree(d_vc);
    return 0;
}
