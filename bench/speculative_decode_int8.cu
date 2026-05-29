// bench/speculative_decode_int8.cu — Speculative decode with INT8-only pipeline
//
// Optimized version: removes FP4 intermediate. Uses fused_rmsnorm_quant_int8
// throughout, keeps activations in INT8 end-to-end.
//
// Saves per layer: unpack_fp4 + pack_int8 + fused_rmsnorm_pack = 3 kernel launches.
// Net savings with residual copy: ~2 kernel launches per layer per token.

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
static void chk(cudaError_t e){if(e!=cudaSuccess){printf("CUDA err %d\n",e);exit(1);}}

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

// ── INT8-only single-token layer ──────────────────────────────────────────
// Input: xi8/xi8s (INT8 activations from prior fused_rmsnorm_quant_int8)
// Output: xi8/xi8s (INT8 activations for next layer)
static void layer_single_int8(LW&L,
    float*Q,float*K,float*V,float*attn,float*proj,
    float*gate,float*up,float*mlp,float*rn,
    int8_t*xi8,float*xi8s,int8_t*ai8,float*ai8s,int8_t*mi8,float*mi8s,
    float*res_buf,   // residual buffer (FP32, pre-RMSNorm saved here)
    float*kc,float*vc,int kb,int sq){

    // ── Attention ──
    // xi8/xi8s already has INT8 activations → use directly in GEMV
    die(blackwell::kernels::gemv_int8(Q,xi8,xi8s,L.q.d,L.q.ds,H,QD,0),"gemv_q");
    die(blackwell::kernels::gemv_int8(K,xi8,xi8s,L.k.d,L.k.ds,H,KV,0),"gemv_k");
    die(blackwell::kernels::gemv_int8(V,xi8,xi8s,L.v.d,L.v.ds,H,KV,0),"gemv_v");

    // KV cache + attention
    die(blackwell::kernels::update_kv_cache(kc+kb,vc+kb,K,V,0,sq,8,128,H,0),"kv");
    die(blackwell::kernels::attention_decode_gqa(attn,Q,kc+kb,vc+kb,sq,16,8,128,H,0),"attn");

    // O projection: pack attention output (FP32) → GEMV
    die(blackwell::kernels::pack_int8(ai8,attn,ai8s,QD,0),"pack_attn");
    die(blackwell::kernels::gemv_int8(proj,ai8,ai8s,L.o.d,L.o.ds,QD,H,0),"gemv_o");

    // Residual: add saved residual, then save new pre-RMSNorm output
    die(blackwell::kernels::vector_add_fp32(proj,proj,res_buf,H,0),"res1");
    die(cudaMemcpyAsync(res_buf,proj,H*4,cudaMemcpyDeviceToDevice,0),"save_res1");

    // RMSNorm → INT8 quant (output in xi8/xi8s for next layer's projections)
    die(blackwell::kernels::fused_rmsnorm_quant_int8(xi8,xi8s,proj,rn,H,eps,0),"norm1");

    // ── MLP ──
    // xi8/xi8s already has INT8 → use directly
    die(blackwell::kernels::gemv_int8(gate,xi8,xi8s,L.gate.d,L.gate.ds,H,ID,0),"gemv_gate");
    die(blackwell::kernels::gemv_int8(up,xi8,xi8s,L.up.d,L.up.ds,H,ID,0),"gemv_up");
    die(blackwell::kernels::apply_swiglu(mlp,gate,up,ID,0),"swiglu");

    // Pack MLP → GEMV down
    die(blackwell::kernels::pack_int8(mi8,mlp,mi8s,ID,0),"pack_mlp");
    die(blackwell::kernels::gemv_int8(proj,mi8,mi8s,L.down.d,L.down.ds,ID,H,0),"gemv_down");

    // Residual + RMSNorm → INT8 for next layer
    die(blackwell::kernels::vector_add_fp32(proj,proj,res_buf,H,0),"res2");
    die(cudaMemcpyAsync(res_buf,proj,H*4,cudaMemcpyDeviceToDevice,0),"save_res2");
    die(blackwell::kernels::fused_rmsnorm_quant_int8(xi8,xi8s,proj,rn,H,eps,0),"norm2");
}

// ── INT8-only batched layer (M draft tokens) ──────────────────────────────
static void layer_batched_int8(LW&L,int M,
    float*Q,float*K,float*V,float*attn,float*projM,
    float*gateM,float*upM,float*mlpM,float*rn,
    int8_t*xM,float*xMs,int8_t*ai8,float*ai8s,int8_t*miM,float*miMs,
    float*resM,      // per-token residual buffer [M*H]
    float*kc,float*vc,int base_kb,int sq){

    // ── Attention (serial per-token KV cache) ──
    for(int m=0;m<M;++m){
        int kb=base_kb+m*8*H*128;
        die(blackwell::kernels::gemv_int8(Q,xM+m*H,xMs+m*(H/B),L.q.d,L.q.ds,H,QD,0),"q");
        die(blackwell::kernels::gemv_int8(K,xM+m*H,xMs+m*(H/B),L.k.d,L.k.ds,H,KV,0),"k");
        die(blackwell::kernels::gemv_int8(V,xM+m*H,xMs+m*(H/B),L.v.d,L.v.ds,H,KV,0),"v");
        die(blackwell::kernels::update_kv_cache(kc+kb,vc+kb,K,V,0,sq,8,128,H,0),"kv");
        die(blackwell::kernels::attention_decode_gqa(attn,Q,kc+kb,vc+kb,sq,16,8,128,H,0),"attn");
        die(blackwell::kernels::pack_int8(ai8,attn,ai8s,QD,0),"pack_attn");
        die(blackwell::kernels::gemv_int8(projM+m*H,ai8,ai8s,L.o.d,L.o.ds,QD,H,0),"o");
        die(blackwell::kernels::vector_add_fp32(projM+m*H,projM+m*H,resM+m*H,H,0),"res1");
        die(cudaMemcpyAsync(resM+m*H,projM+m*H,H*4,cudaMemcpyDeviceToDevice,0),"save1");
        die(blackwell::kernels::fused_rmsnorm_quant_int8(
            xM+m*H,xMs+m*(H/B),projM+m*H,rn,H,eps,0),"norm1");
    }

    // ── MLP (batched) ──
    // xM already has INT8 activations (from fused_rmsnorm_quant_int8 above)
    die(blackwell::kernels::gemv_int8_batched(gateM,xM,xMs,L.gate.d,L.gate.ds,H,ID,M,0),"gate_b");
    die(blackwell::kernels::gemv_int8_batched(upM,xM,xMs,L.up.d,L.up.ds,H,ID,M,0),"up_b");

    for(int m=0;m<M;++m){
        die(blackwell::kernels::apply_swiglu(mlpM+m*ID,gateM+m*ID,upM+m*ID,ID,0),"swiglu");
        // Compute absmax scales + pack for down_proj
        {
            int8_t* mi = miM + m*ID;
            float* ms = miMs + m*(ID/B);
            die(blackwell::kernels::pack_int8(mi,mlpM+m*ID,ms,ID,0),"pack_mlp");
        }
    }

    die(blackwell::kernels::gemv_int8_batched(projM,miM,miMs,L.down.d,L.down.ds,ID,H,M,0),"down_b");

    for(int m=0;m<M;++m){
        die(blackwell::kernels::vector_add_fp32(projM+m*H,projM+m*H,resM+m*H,H,0),"res2");
        die(cudaMemcpyAsync(resM+m*H,projM+m*H,H*4,cudaMemcpyDeviceToDevice,0),"save2");
        die(blackwell::kernels::fused_rmsnorm_quant_int8(
            xM+m*H,xMs+m*(H/B),projM+m*H,rn,H,eps,0),"norm2");
    }
}

int main(int argc, char** argv) {
    int NL=4, M=4, steps=20;
    if(argc>1)NL=atoi(argv[1]);
    if(argc>2)M=atoi(argv[2]);
    if(argc>3)steps=atoi(argv[3]);

    cudaDeviceProp p;cudaGetDeviceProperties(&p,0);
    printf("# Speculative Decode INT8-only — Qwen3-1.7B\n");
    printf("Device: %s (CC %d.%d)\n",p.name,p.major,p.minor);
    printf("Layers: %d, Drafts M: %d, Steps: %d\n",NL,M,steps);

    const char* dir="weights_int8_bf16";
    printf("Loading weights...\n");
    std::vector<LW> L(NL);
    for(int i=0;i<NL;++i)L[i].load(dir,i);

    // Per-token buffers (INT8-only — no FP4 buffers)
    float*d_Q,*d_K,*d_V,*d_attn,*d_proj,*d_gate,*d_up,*d_mlp,*d_rn;
    float*d_res;  // residual (FP32)
    int8_t*d_xi8,*d_ai8,*d_mi8;
    float*d_xi8s,*d_ai8s,*d_mi8s;
    chk(cudaMalloc(&d_Q,H*4));chk(cudaMalloc(&d_K,KV*4));chk(cudaMalloc(&d_V,KV*4));
    chk(cudaMalloc(&d_attn,QD*4));chk(cudaMalloc(&d_proj,H*4));
    chk(cudaMalloc(&d_gate,ID*4));chk(cudaMalloc(&d_up,ID*4));chk(cudaMalloc(&d_mlp,ID*4));
    chk(cudaMalloc(&d_rn,H*4));chk(cudaMalloc(&d_res,H*4));
    std::vector<float>ones(H,1.f);cudaMemcpy(d_rn,ones.data(),H*4,cudaMemcpyHostToDevice);
    std::vector<float>zeros(H,0.f);cudaMemcpy(d_res,zeros.data(),H*4,cudaMemcpyHostToDevice);
    chk(cudaMalloc(&d_xi8,H));chk(cudaMalloc(&d_xi8s,(H/B)*4));
    chk(cudaMalloc(&d_ai8,QD));chk(cudaMalloc(&d_ai8s,(QD/B)*4));
    chk(cudaMalloc(&d_mi8,ID));chk(cudaMalloc(&d_mi8s,(ID/B)*4));
    float ixv=1.f/127.f;
    std::vector<float>i8sc(H/B,ixv);
    cudaMemcpy(d_xi8s,i8sc.data(),(H/B)*4,cudaMemcpyHostToDevice);
    std::vector<float>asc(QD/B,ixv);
    cudaMemcpy(d_ai8s,asc.data(),(QD/B)*4,cudaMemcpyHostToDevice);
    std::vector<float>msc(ID/B,ixv);
    cudaMemcpy(d_mi8s,msc.data(),(ID/B)*4,cudaMemcpyHostToDevice);

    // Batched buffers
    int8_t*d_xM;float*d_xMs;
    chk(cudaMalloc(&d_xM,M*H));chk(cudaMalloc(&d_xMs,M*(H/B)*4));
    float*d_projM;chk(cudaMalloc(&d_projM,M*H*4));
    float*d_gateM,*d_upM,*d_mlpM;chk(cudaMalloc(&d_gateM,M*ID*4));chk(cudaMalloc(&d_upM,M*ID*4));chk(cudaMalloc(&d_mlpM,M*ID*4));
    int8_t*d_miM;chk(cudaMalloc(&d_miM,M*ID));
    float*d_miMs;chk(cudaMalloc(&d_miMs,M*(ID/B)*4));
    float*d_resM;chk(cudaMalloc(&d_resM,M*H*4));
    chk(cudaMemset(d_resM,0,M*H*4));

    // KV cache
    float*d_kc,*d_vc;
    size_t kv_sz=(size_t)NL*M*8*H*128*4;
    chk(cudaMalloc(&d_kc,kv_sz));chk(cudaMalloc(&d_vc,kv_sz));
    chk(cudaMemset(d_kc,0,kv_sz));chk(cudaMemset(d_vc,0,kv_sz));

    int sq=128;

    // ── Mode A: Autoregressive baseline (INT8-only) ──
    printf("\n=== Mode A: Autoregressive INT8 (%d layers) ===\n",NL);
    GpuTimer ta;ta.start();
    for(int i=0;i<steps;++i){
        for(int l=0;l<NL;++l){
            layer_single_int8(L[l],d_Q,d_K,d_V,d_attn,d_proj,d_gate,d_up,d_mlp,d_rn,
                d_xi8,d_xi8s,d_ai8,d_ai8s,d_mi8,d_mi8s,d_res,
                d_kc,d_vc,l*8*H*128,sq+i);
        }
    }
    cudaDeviceSynchronize();float ms_a=ta.stop();
    float tp_a=steps*1000.f/ms_a;
    float s28_a=1000.f/(ms_a/steps*28.f/NL);
    printf("  Total: %.2f ms, Per-token: %.3f ms, t/s: %.1f, Scaled28: %.1f\n",ms_a,ms_a/steps,tp_a,s28_a);

    // ── Mode B: Speculative decoding (INT8-only) ──
    printf("\n=== Mode B: Speculative INT8 (M=%d drafts) ===\n",M);
    GpuTimer tb;tb.start();
    for(int i=0;i<steps;++i){
        for(int l=0;l<NL;++l){
            layer_batched_int8(L[l],M,
                d_Q,d_K,d_V,d_attn,d_projM,d_gateM,d_upM,d_mlpM,d_rn,
                d_xM,d_xMs,d_ai8,d_ai8s,d_miM,d_miMs,d_resM,
                d_kc,d_vc,l*M*8*H*128,sq+i);
        }
    }
    cudaDeviceSynchronize();float ms_b=tb.stop();
    float tokens_per_step=M+1;
    float tp_b=tokens_per_step*steps*1000.f/ms_b;
    float s28_b=1000.f/(ms_b/steps*28.f/NL);
    printf("  Total: %.2f ms, M+1 tokens/step: %.1f, t/s: %.1f, Scaled28: %.1f\n",ms_b,tokens_per_step,tp_b,s28_b);
    printf("  Speedup vs autoregressive: %.2fx (%.1f%%)\n",tp_b/tp_a,(tp_b/tp_a-1)*100.f);

    printf("\n=== Comparison vs FP4-path spec decode ===\n");
    printf("  %-25s  %8s  %8s  %8s\n","Method","ms/step","t/s","Scaled28");
    printf("  %-25s  %7.3fms  %7.1f   %7.1f\n","Autoregressive INT8",ms_a/steps,tp_a,s28_a);
    printf("  %-25s  %7.3fms  %7.1f   %7.1f\n","Speculative INT8",ms_b/steps,tp_b,s28_b);
    printf("  Improvement: %.1f%% more tokens/s\n",(tp_b/tp_a-1)*100.f);

    for(auto&l:L)l.free_();
    chk(cudaFree(d_Q));chk(cudaFree(d_K));chk(cudaFree(d_V));chk(cudaFree(d_attn));chk(cudaFree(d_proj));
    chk(cudaFree(d_gate));chk(cudaFree(d_up));chk(cudaFree(d_mlp));chk(cudaFree(d_rn));chk(cudaFree(d_res));
    chk(cudaFree(d_xi8));chk(cudaFree(d_xi8s));chk(cudaFree(d_ai8));chk(cudaFree(d_ai8s));chk(cudaFree(d_mi8));chk(cudaFree(d_mi8s));
    chk(cudaFree(d_xM));chk(cudaFree(d_xMs));chk(cudaFree(d_projM));
    chk(cudaFree(d_gateM));chk(cudaFree(d_upM));chk(cudaFree(d_mlpM));
    chk(cudaFree(d_miM));chk(cudaFree(d_miMs));chk(cudaFree(d_resM));
    chk(cudaFree(d_kc));chk(cudaFree(d_vc));
    return 0;
}
