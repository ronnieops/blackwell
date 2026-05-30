// bench/validate_pipeline.cu — End-to-end 1-layer pipeline validation
//
// Runs 1 layer of inference_server Mode A decode with fixed input,
// dumps FP32 logits. Compare with Python reference.
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/validate_pipeline.cu build/libblackwell_kernels.a \
//     -o bench/validate_pipeline
//
// Run: ./bench/validate_pipeline
// Then: python3 bench/validate_pipeline.py (reference)

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>
#include <chrono>
#include <algorithm>
#include "blackwell/kernels.h"

// Compute 16-element block absmax scales for pack_int8
__global__ void absmax_scales_kernel(const float* in, float* sc, int n) {
    int blk=blockIdx.x; int lane=threadIdx.x; float amax=0;
    for(int i=lane;i<16&&blk*16+i<n;i+=32) amax=fmaxf(amax,fabsf(in[blk*16+i]));
    for(int off=16;off>0;off>>=1) amax=fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,off));
    if(lane==0) sc[blk]=fmaxf(amax/127.0f,1e-9f);
}
static void compute_scales(float* in, float* out, int n, cudaStream_t st, const char* nm) {
    absmax_scales_kernel<<<n/16,32,0,st>>>(in,out,n);
    cudaError_t e=cudaPeekAtLastError();
    if(e!=cudaSuccess){printf("FAIL scales %s: %s\n",nm,cudaGetErrorString(e));exit(1);}
}

#define die(e,m) do{auto _e=(e);if(_e!=cudaSuccess){\
    printf("FAIL %s: %s\n",m,cudaGetErrorString(_e));exit(1);}}while(0)

struct LW { std::vector<int8_t> d; std::vector<float> sc; };
struct DW { int8_t* d; float* sc; };

static LW lw(const char* p) {
    char x[256]; snprintf(x,256,"%s.int8_t",p);
    FILE* f=fopen(x,"rb"); if(!f){printf("FAIL open %s\n",x);exit(1);}
    int h[5]; (void)fread(h,4,5,f); LW w;
    w.d.resize(h[0]*h[1]); (void)fread(w.d.data(),1,w.d.size(),f); fclose(f);
    snprintf(x,256,"%s.scale_t",p); f=fopen(x,"rb"); (void)fread(h,4,5,f);
    w.sc.resize(h[3]*h[4]); (void)fread(w.sc.data(),4,w.sc.size(),f); fclose(f);
    return w;
}
static DW dw(const LW& w) {
    DW d;
    cudaMalloc(&d.d,w.d.size()); cudaMemcpy(d.d,w.d.data(),w.d.size(),cudaMemcpyHostToDevice);
    cudaMalloc(&d.sc,w.sc.size()*4); cudaMemcpy(d.sc,w.sc.data(),w.sc.size()*4,cudaMemcpyHostToDevice);
    return d;
}

struct L { DW q,k,v,o,g,u,d; };

int main() {
    const int H=2048, QD=2048, KV=1024, ID=6144, hd=128, nqh=16, nkv=8;
    const float eps=1e-6f;
    
    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("# Pipeline Validation — 1 Layer\n");
    printf("  Device: %s\n\n",p.name);
    
    // Load Layer 0 weights
    L W0;
    {
        char p_[256];
        snprintf(p_,256,"weights_int8_bf16/0_self_attn.q_proj"); W0.q=dw(lw(p_));
        snprintf(p_,256,"weights_int8_bf16/0_self_attn.k_proj"); W0.k=dw(lw(p_));
        snprintf(p_,256,"weights_int8_bf16/0_self_attn.v_proj"); W0.v=dw(lw(p_));
        snprintf(p_,256,"weights_int8_bf16/0_self_attn.o_proj"); W0.o=dw(lw(p_));
        snprintf(p_,256,"weights_int8_bf16/0_mlp.gate_proj");   W0.g=dw(lw(p_));
        snprintf(p_,256,"weights_int8_bf16/0_mlp.up_proj");     W0.u=dw(lw(p_));
        snprintf(p_,256,"weights_int8_bf16/0_mlp.down_proj");   W0.d=dw(lw(p_));
    }
    printf("Loaded Layer 0.\n");
    
    const int MAXSEQ=2048;
    float *d_x, *d_xi, *d_xs, *d_res, *d_rn;
    float *d_Q, *d_K, *d_V, *d_attn, *d_proj;
    float *d_gate, *d_up, *d_mlp;
    int8_t *d_ai, *d_mi;
    float *d_as, *d_ms;
    float *d_kc, *d_vc;
    
    #define AL(p,n) die(cudaMalloc(&(p),(n)),#p)
    AL(d_x,H*4); AL(d_xi,H*4); AL(d_xs,H/16*4); AL(d_res,H*4); AL(d_rn,H*4);
    AL(d_Q,QD*4); AL(d_K,KV*4); AL(d_V,KV*4); AL(d_attn,QD*4); AL(d_proj,H*4);
    AL(d_gate,ID*4); AL(d_up,ID*4); AL(d_mlp,ID*4);
    AL(d_ai,QD); AL(d_mi,ID);
    AL(d_as,QD/16*4); AL(d_ms,ID/16*4);
    AL(d_kc,nkv*MAXSEQ*hd*4); AL(d_vc,nkv*MAXSEQ*hd*4);
    #undef AL
    
    die(cudaMemset(d_kc,0,nkv*MAXSEQ*hd*4),"memset kc");
    die(cudaMemset(d_vc,0,nkv*MAXSEQ*hd*4),"memset vc");
    
    cudaStream_t st; die(cudaStreamCreate(&st),"stream");
    
    // Load RMSNorm weights from model (read with Python and write to file)
    // For now, load from inference_server-compatible binary file
    // Generate: python3 -c "import struct,json,numpy as np;
    //  M='/mnt/data/ai/hf/qwen3-1.7b-base/model.safetensors';
    //  f=open(M,'rb');l=struct.unpack('Q',f.read(8))[0];h=json.loads(f.read(l));
    //  s,e=h['model.layers.0.input_layernorm.weight']['data_offsets'];f.seek(8+l+s);
    //  u=np.frombuffer(f.read(e-s),dtype=np.uint16);
    //  f32=(u.astype(np.uint32)<<16).view(np.float32);f32.tofile('/tmp/rn_weight.bin')"
    std::vector<float> rn(H);
    FILE* rf=fopen("/tmp/rn_weight.bin","rb");
    if(!rf){printf("FAIL: no rn_weight.bin. Generate first.\n");exit(1);}
    fread(rn.data(),4,H,rf); fclose(rf);
    die(cudaMemcpy(d_rn,rn.data(),H*4,cudaMemcpyHostToDevice),"rnmcpy");
    
    // Fixed deterministic input: sawtooth [-0.07, 0.07]
    std::vector<float> inp(H);
    for(int j=0;j<H;++j) inp[j]=(j%17-8)*0.01f;
    die(cudaMemcpy(d_x,inp.data(),H*4,cudaMemcpyHostToDevice),"memcpy");
    
    // Forward: 1 layer, seq_pos=0
    int sp=0;
    {
        auto dump=[&](const char* name, float* d, int n){
            std::vector<float> h(n);
            die(cudaMemcpy(h.data(),d,n*4,cudaMemcpyDeviceToHost),"dump");
            float s=0,sq=0,mn=h[0],mx=h[0];
            for(int i=0;i<n;++i){s+=h[i];sq+=h[i]*h[i];mn=fminf(mn,h[i]);mx=fmaxf(mx,h[i]);}
            printf("  %12s: mean=% .4f std=%.4f min=%.4f max=%.4f zeros=%d/%d\n",
                name,s/n,sqrtf(sq/n-s*s/(n*n)),mn,mx,int(std::count(h.begin(),h.end(),0.0f)),n);
        };
        dump("rn_weights",d_rn,H);
        
        auto kr = blackwell::kernels::fused_rmsnorm_quant_int8(
            (int8_t*)d_xi,d_xs,(float*)d_x,d_rn,H,eps,st);
        printf("  rmsnorm returned: %s\n",cudaGetErrorString(kr));
        cudaError_t lerr = cudaPeekAtLastError();
        printf("  peek: %s\n",cudaGetErrorString(lerr));
        
        dump("x (input)",d_x,H);
        dump("x_scales",d_xs,H/16);
        
        // Also dump d_rn output (stored in d_rn? No, d_rn is weight, not modified)
        
        die(blackwell::kernels::gemv_int8_warp(d_Q,(int8_t*)d_xi,d_xs,W0.q.d,W0.q.sc,H,QD,st),"q");
        die(blackwell::kernels::gemv_int8_warp(d_K,(int8_t*)d_xi,d_xs,W0.k.d,W0.k.sc,H,KV,st),"k");
        die(blackwell::kernels::gemv_int8_warp(d_V,(int8_t*)d_xi,d_xs,W0.v.d,W0.v.sc,H,KV,st),"v");
        
        die(blackwell::kernels::update_kv_cache(d_kc,d_vc,d_K,d_V,0,sp,nkv,hd,MAXSEQ,st),"kv");
        // Dump specific KV cache slots for heads 0 and 1
        {
            auto dump_at=[&](const char* nm, float* d, int base, int n){
                std::vector<float> h(n);
                die(cudaMemcpy(h.data(),d+base,n*4,cudaMemcpyDeviceToHost),"dmp");
                float s=0,sq=0,mn=h[0],mx=h[0]; int z=0;
                for(int i=0;i<n;++i){s+=h[i];sq+=h[i]*h[i];mn=fminf(mn,h[i]);mx=fmaxf(mx,h[i]);z+=(h[i]==0?1:0);}
                printf("  %15s[%d]: mean=% .4f std=%.4f mn=%.4f mx=%.4f z=%d\n",nm,base,s/n,sqrtf(sq/n-s*s/(n*n)),mn,mx,z);
            };
            dump_at("K_cache",d_kc,0*nkv*MAXSEQ*hd+0,hd);  // KV head 0, seq_pos 0
            dump_at("K_cache",d_kc,1*MAXSEQ*hd+0,hd);  // KV head 1, seq_pos 0
            dump_at("K_cache",d_kc,2*MAXSEQ*hd+0,hd);  // KV head 2, seq_pos 0
            dump_at("K_new",d_K,0*hd,hd);  // K projection head 0
            dump_at("K_new",d_K,1*hd,hd);  // K projection head 1
            dump_at("K_new",d_K,2*hd,hd);  // K projection head 2
        }
        
        auto ae=blackwell::kernels::attention_decode_gqa(d_attn,d_Q,d_kc,d_vc,sp,nqh,nkv,hd,MAXSEQ,st);
        printf("  attn returned: %s\n",cudaGetErrorString(ae));
        cudaError_t apeek=cudaPeekAtLastError();
        printf("  attn peek: %s\n",cudaGetErrorString(apeek));
        
        dump("V_cache[2]",d_vc+2*MAXSEQ*hd,hd);
        dump("attn",d_attn,QD);
        // Per-head attn check
        for(int hh=0;hh<min(8,nqh);++hh){
            auto dmp=[&](float* d, int n, const char* nm){
                std::vector<float> h(n);
                die(cudaMemcpy(h.data(),d,n*4,cudaMemcpyDeviceToHost),"dmp");
                float s=0,sq=0,mn=h[0],mx=h[0];
                for(int i=0;i<n;++i){s+=h[i];sq+=h[i]*h[i];mn=fminf(mn,h[i]);mx=fmaxf(mx,h[i]);}
                printf("  %12s h%d: mean=% .4f std=%.4f min=%.4f max=%.4f\n",nm,hh,s/n,sqrtf(sq/n-s*s/(n*n)),mn,mx);
            };
            dmp(d_Q+hh*hd,hd,"Q");
            dmp(d_attn+hh*hd,hd,"attn");
        }
        
        compute_scales(d_attn,d_as,QD,st,"attn");
        die(blackwell::kernels::pack_int8(d_ai,d_attn,d_as,QD,st),"pack");
        die(blackwell::kernels::gemv_int8_warp(d_proj,d_ai,d_as,W0.o.d,W0.o.sc,QD,H,st),"o");
        dump("Wo_out",d_proj,H);
        // Correct residual: hidden = Wo_out + x, save copy, also update d_proj for MLP
        die(blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_x,H,st),"res1");  // proj = Wo_out + x = hidden
        die(cudaMemcpyAsync(d_res,d_proj,H*4,cudaMemcpyDeviceToDevice,st),"save_res"); // save hidden
        
        die(blackwell::kernels::fused_rmsnorm_quant_int8(
            (int8_t*)d_xi,d_xs,d_proj,d_rn,H,eps,st),"rmsnorm2");
        die(blackwell::kernels::gemv_int8_warp(d_gate,(int8_t*)d_xi,d_xs,W0.g.d,W0.g.sc,H,ID,st),"gate");
        die(blackwell::kernels::gemv_int8_warp(d_up,(int8_t*)d_xi,d_xs,W0.u.d,W0.u.sc,H,ID,st),"up");
        die(blackwell::kernels::apply_swiglu(d_mlp,d_gate,d_up,ID,st),"swiglu");
        compute_scales(d_mlp,d_ms,ID,st,"mlp");
        die(blackwell::kernels::pack_int8(d_mi,d_mlp,d_ms,ID,st),"pack2");
        die(blackwell::kernels::gemv_int8_warp(d_proj,d_mi,d_ms,W0.d.d,W0.d.sc,ID,H,st),"down");
        // Correct MLP residual: down_out + saved_res
        die(blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_res,H,st),"res2");
    }
    die(cudaStreamSynchronize(st),"sync");
    
    // Read output
    std::vector<float> out(H);
    die(cudaMemcpy(out.data(),d_proj,H*4,cudaMemcpyDeviceToHost),"read out");
    
    // Print output stats and full array for Python comparison
    float sum=0, sq=0, mn=out[0], mx=out[0];
    for(int j=0;j<H;++j){sum+=out[j];sq+=out[j]*out[j];mn=fminf(mn,out[j]);mx=fmaxf(mx,out[j]);}
    printf("\n  Output: mean=%.6f std=%.6f min=%.6f max=%.6f\n",
        sum/H,sqrtf(sq/H-sum*sum/(H*H)),mn,mx);
    
    // Dump as Python list
    printf("\n  Output logits (first 32):");
    for(int j=0;j<32&&j<H;++j) printf(" %.6f",out[j]);
    printf("\n");
    
    // Full dump to file for Python comparison
    FILE* f=fopen("/tmp/layer0_out.bin","wb");
    fwrite(out.data(),4,H,f); fclose(f);
    printf("\n  Full output written to /tmp/layer0_out.bin (%.1f KB)\n",H*4/1024.0);
    printf("  Compare: python3 bench/validate_pipeline.py\n");
    
    return 0;
}