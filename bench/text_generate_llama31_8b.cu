// bench/text_generate_llama31_8b.cu — Llama 3.1 8B single-seq INT4 generation
// Reads dims from weight dir metadata (rope_config.f32, .int4_t headers).
//
// Build: part of CMake (target: text_generate_llama31_8b)
//
// Usage: ./bench/text_generate_llama31_8b [prompt] [max_tokens] [weight_dir]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cstring>
#include <string>
#include <cmath>
#include "blackwell/kernels.h"
#include "blackwell/bpe_tokenizer.h"
using blackwell::BpeTokenizer;

static void die(cudaError_t e, const char* m) {
    if(e!=cudaSuccess){printf("FAIL %s: %s\n",m,cudaGetErrorString(e));exit(1);}
}

// Read dimensions from weight files
struct ModelDims {
    int H, Q, KV, I, nqh, nkv, hd, V, NL, MAXSEQ;
};

static ModelDims read_dims(const char* wdir) {
    ModelDims d = {};
    // Read rope_config.f32 for hd
    char path[256];
    snprintf(path, 256, "%s/rope_config.f32", wdir);
    FILE* f = fopen(path, "rb");
    if (f) {
        float cfg[2];
        fread(cfg, 4, 2, f);
        fclose(f);
        d.hd = (int)cfg[1];
        d.H = d.hd * 32;  // default nqh=32 for 8B
    } else {
        d.hd = 128;
        d.H = 4096;
    }
    
    // Read any layer-0 q_proj to get actual dims
    snprintf(path, 256, "%s/0_self_attn.q_proj.int4_t", wdir);
    f = fopen(path, "rb");
    if (f) {
        int hdr[5];
        fread(hdr, 4, 5, f);
        fclose(f);
        d.H = hdr[0];  // K = input dim
        // N = output dim = nqh * hd = d.H (for self-attn)
        d.hd = d.H / 32;  // assume nqh=32
    }
    
    // Count layers
    int max_layer = 0;
    for (int l = 0; l < 100; l++) {
        snprintf(path, 256, "%s/%d_self_attn.q_proj.int4_t", wdir, l);
        f = fopen(path, "rb");
        if (f) { fclose(f); max_layer = l + 1; }
        else break;
    }
    d.NL = max_layer;
    
    // Read I from up_proj
    snprintf(path, 256, "%s/0_mlp.gate_proj.int4_t", wdir);
    f = fopen(path, "rb");
    if (f) {
        int hdr[5];
        fread(hdr, 4, 5, f);
        fclose(f);
        d.I = hdr[1];  // N = output dim of gate = I
    } else d.I = 14336;
    
    // nqh/nkv from H / hd
    d.nqh = d.H / d.hd;
    d.nkv = 8;  // Llama 3.1 8B
    d.Q = d.nqh * d.hd;
    d.KV = d.nkv * d.hd;
    d.MAXSEQ = 4096;
    
    // V from embed_tokens shape
    snprintf(path, 256, "%s/embed_tokens.int4_t", wdir);
    f = fopen(path, "rb");
    if (f) {
        int hdr[5];
        fread(hdr, 4, 5, f);
        fclose(f);
        d.V = hdr[1];  // N = vocab size
    } else d.V = 128256;
    
    return d;
}

struct DevW4 { int K, N; uint8_t* d; float* sc; };

static DevW4 upload_w4(const char* prefix) {
    char p[256]; snprintf(p,256,"%s.int4_t",prefix);
    FILE* f=fopen(p,"rb"); if(!f){printf("FAIL open %s\n",p);exit(1);}
    int h[5]; fread(h,4,5,f);
    DevW4 dw; dw.K=h[0]; dw.N=h[1];
    size_t ds=(size_t)h[0]*h[1]/2;
    uint8_t* td=new uint8_t[ds]; fread(td,1,ds,f); fclose(f);
    cudaMalloc(&dw.d,ds); cudaMemcpy(dw.d,td,ds,cudaMemcpyHostToDevice); delete[] td;
    snprintf(p,256,"%s.scale_t",prefix); f=fopen(p,"rb"); fread(h,4,5,f);
    size_t ss=(size_t)h[3]*h[4];
    float* ts=new float[ss]; fread(ts,4,ss,f); fclose(f);
    cudaMalloc(&dw.sc,ss*4); cudaMemcpy(dw.sc,ts,ss*4,cudaMemcpyHostToDevice); delete[] ts;
    return dw;
}

struct LW4 { DevW4 q,k,v,o,g,u,d; float* qn,*kn,*rn_in,*rn_post; };

__global__ void head_norm_kernel(float* data, const float* weight, int nh, int hd, float eps) {
    int h=blockIdx.x; if(h>=nh) return;
    float* d=data+h*hd;
    __shared__ float wp[4]; float s=0;
    for(int i=threadIdx.x;i<hd;i+=blockDim.x) s+=d[i]*d[i];
    for(int off=16;off>0;off>>=1) s+=__shfl_xor_sync(0xffffffff,s,off);
    if((threadIdx.x&31)==0) wp[threadIdx.x>>5]=s; __syncthreads();
    if(threadIdx.x<4) s=wp[threadIdx.x]; else s=0;
    for(int off=2;off>0;off>>=1) s+=__shfl_xor_sync(0xffffffff,s,off);
    if(threadIdx.x==0) wp[0]=rsqrtf(s/hd+eps); __syncthreads();
    float is=wp[0];
    for(int i=threadIdx.x;i<hd;i+=blockDim.x) d[i]=d[i]*is*weight[i];
}

__global__ void apply_rope_kernel(float* data, int n_heads, int head_dim, int pos, float rope_theta) {
    int h=blockIdx.x; int d=threadIdx.x;
    if(h>=n_heads||d>=head_dim/2) return;
    float* pair=data+h*head_dim+d*2;
    float theta=(float)pos*powf(rope_theta,-2.0f*(float)d/(float)head_dim);
    float c=cosf(theta),s=sinf(theta),x=pair[0],y=pair[1];
    pair[0]=x*c-y*s; pair[1]=x*s+y*c;
}

static void dequant_embed_row(float* out, int token, const uint8_t* host_w,
    const float* host_sc, int K)
{
    int kblocks=K/16;
    for(int b=0;b<kblocks;++b){
        float sc=host_sc[token*kblocks+b];
        for(int i=0;i<16;++i){
            size_t byte_idx=(size_t)token*K/2+(size_t)b*8+i/2;
            uint8_t byte=host_w[byte_idx];
            int nib=(i&1)?((byte>>4)&0x0F):(byte&0x0F);
            out[b*16+i]=(float)(nib-8)*sc;
        }
    }
}

int main(int argc, char** argv) {
    const char* prompt = "The capital of France is";
    int max_new = 30;
    const char* wdir = "weights_llama31_int4";
    float temperature = 0.0f;
    int top_k = 0;
    
    if(argc>1) prompt=argv[1];
    if(argc>2) max_new=atoi(argv[2]);
    if(argc>3) wdir=argv[3];
    for(int i=1;i<argc;i++){
        if(strcmp(argv[i],"-t")==0&&i+1<argc) temperature=atof(argv[++i]);
        if(strcmp(argv[i],"-k")==0&&i+1<argc) top_k=atoi(argv[++i]);
    }
    
    auto dims = read_dims(wdir);
    const int H=dims.H, Q=dims.Q, KV=dims.KV, I=dims.I;
    const int nqh=dims.nqh, nkv=dims.nkv, hd=dims.hd, NL=dims.NL, V=dims.V, MAXSEQ=dims.MAXSEQ;
    const float eps=1e-6f;
    
    // Read rope_theta
    float rope_theta = 500000.0f;
    {
        char p[256]; snprintf(p,256,"%s/rope_config.f32",wdir);
        FILE* f=fopen(p,"rb"); if(f){float cfg[2]; fread(cfg,4,2,f); rope_theta=cfg[0]; fclose(f);}
    }
    
    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    printf("# Llama 3.1 8B INT4 — %d layers\n", NL);
    printf("  Weights: %s\n", wdir);
    printf("  Dims: H=%d I=%d nqh=%d nkv=%d hd=%d V=%d NL=%d\n", H,I,nqh,nkv,hd,V,NL);
    printf("  Device: %s\n", P.name);
    printf("  Prompt: \"%s\"\n", prompt);
    printf("  Temp: %.1f, Top-K: %d, Max new: %d\n\n", temperature, top_k, max_new);
    
    BpeTokenizer tokenizer;
    {
        char p[256]; snprintf(p,256,"%s/tokenizer_data.bin",wdir);
        if(tokenizer.load(p)!=0){fprintf(stderr,"FAIL: can't load tokenizer\n");return 1;}
    }
    
    // Encode prompt
    std::vector<uint32_t> input_ids;
    // Llama 3.1: prepend BOS (128000) unless prompt starts with special token
    auto ids = tokenizer.encode(prompt);
    input_ids.push_back(128000);  // <|begin_of_text|>
    input_ids.insert(input_ids.end(), ids.begin(), ids.end());
    
    printf("Input: %zu tokens\n", input_ids.size());
    
    // Device buffers
    float *d_x32, *d_xi_f, *d_res;
    uint8_t *d_x_i4; float *d_x_i4_sc;
    float *d_Q, *d_K, *d_V, *d_attn;
    uint8_t *d_attn_i4; float *d_attn_i4_sc;
    float *d_proj, *d_gate, *d_up;
    uint8_t *d_mlp_i4; float *d_mlp_i4_sc;
    float *d_fn, *d_fn_sc, *d_kc, *d_vc, *d_logits;
    int *d_next_id;
    
    #define AL(p,n){cudaError_t _e=cudaMalloc(&(p),(n));\
        if(_e!=cudaSuccess){printf("FAIL malloc %s: %s\n",#p,cudaGetErrorString(_e));exit(1);}}
    
    AL(d_x32,H*4);AL(d_xi_f,H*4);AL(d_res,H*4);
    AL(d_x_i4,H/2);AL(d_x_i4_sc,H/16*4);
    AL(d_Q,Q*4);AL(d_K,KV*4);AL(d_V,KV*4);
    AL(d_attn,Q*4);AL(d_attn_i4,Q/2);AL(d_attn_i4_sc,Q/16*4);
    AL(d_proj,H*4);AL(d_gate,I*4);AL(d_up,I*4);
    AL(d_mlp_i4,I/2);AL(d_mlp_i4_sc,I/16*4);
    AL(d_fn,H*4);AL(d_fn_sc,(H/16)*4);
    AL(d_kc,(size_t)NL*nkv*MAXSEQ*hd*4);
    AL(d_vc,(size_t)NL*nkv*MAXSEQ*hd*4);
    AL(d_logits,V*4);AL(d_next_id,4);
    
    #undef AL
    
    printf("Loading %d-layer model...\n", NL);
    
    char p[256];
    
    // Load weights
    LW4* W = new LW4[NL]();
    for(int l=0;l<NL;l++){
        snprintf(p,256,"%s/%d_self_attn.q_proj",wdir,l); W[l].q=upload_w4(p);
        snprintf(p,256,"%s/%d_self_attn.k_proj",wdir,l); W[l].k=upload_w4(p);
        snprintf(p,256,"%s/%d_self_attn.v_proj",wdir,l); W[l].v=upload_w4(p);
        snprintf(p,256,"%s/%d_self_attn.o_proj",wdir,l); W[l].o=upload_w4(p);
        snprintf(p,256,"%s/%d_mlp.gate_proj",wdir,l); W[l].g=upload_w4(p);
        snprintf(p,256,"%s/%d_mlp.up_proj",wdir,l); W[l].u=upload_w4(p);
        snprintf(p,256,"%s/%d_mlp.down_proj",wdir,l); W[l].d=upload_w4(p);
        
        // Norms
        {
            snprintf(p,256,"%s/%d_input_layernorm.f32",wdir,l);
            FILE* f=fopen(p,"rb"); float* w=new float[H]; fread(w,4,H,f); fclose(f);
            cudaMalloc(&W[l].rn_in,H*4); cudaMemcpy(W[l].rn_in,w,H*4,cudaMemcpyHostToDevice); delete[] w;
        }
        {
            snprintf(p,256,"%s/%d_post_attention_layernorm.f32",wdir,l);
            FILE* f=fopen(p,"rb"); float* w=new float[H]; fread(w,4,H,f); fclose(f);
            cudaMalloc(&W[l].rn_post,H*4); cudaMemcpy(W[l].rn_post,w,H*4,cudaMemcpyHostToDevice); delete[] w;
        }
        
        if((l+1)%8==0||l==NL-1) printf("  layer %d/%d\n",l+1,NL);
    }
    
    // Q/K head norms
    {
        snprintf(p,256,"%s/qk_norms.f32",wdir);
        FILE* f=fopen(p,"rb");
        if(f){
            float* buf = new float[(size_t)NL*2*hd];
            fread(buf,4,(size_t)NL*2*hd,f); fclose(f);
            for(int l=0;l<NL;l++){
                cudaMalloc(&W[l].qn,hd*4); cudaMemcpy(W[l].qn,buf+(size_t)l*2*hd,hd*4,cudaMemcpyHostToDevice);
                cudaMalloc(&W[l].kn,hd*4); cudaMemcpy(W[l].kn,buf+(size_t)l*2*hd+hd,hd*4,cudaMemcpyHostToDevice);
            }
            delete[] buf;
        }
    }
    
    // Embedding
    snprintf(p,256,"%s/embed_tokens.int4_t",wdir);
    FILE* f_emb=fopen(p,"rb");
    if(!f_emb){fprintf(stderr,"FAIL: no embed_tokens\n");return 1;}
    int eh[5]; fread(eh,4,5,f_emb);
    int V_dim=eh[1]; int embed_K=eh[0];
    size_t eds=(size_t)V_dim*embed_K/2;
    uint8_t* host_embed_d=new uint8_t[eds];
    fread(host_embed_d,1,eds,f_emb); fclose(f_emb);
    
    snprintf(p,256,"%s/embed_tokens.scale_t",wdir);
    f_emb=fopen(p,"rb"); int esc_h[5]; fread(esc_h,4,5,f_emb);
    size_t ess=(size_t)esc_h[3]*esc_h[4];
    float* host_embed_sc=new float[ess];
    fread(host_embed_sc,4,ess,f_emb); fclose(f_emb);
    printf("Embed loaded: %d x %d (INT4)\n", V_dim, embed_K);
    
    // lm_head (may be tied)
    DevW4 lm_head_w;
    snprintf(p,256,"%s/lm_head.int4_t",wdir);
    FILE* flm = fopen(p,"rb");
    if(flm){
        fclose(flm);
        lm_head_w=upload_w4((std::string(wdir)+"/lm_head").c_str());
        printf("lm_head loaded: %d x %d (INT4)\n", lm_head_w.K, lm_head_w.N);
    } else {
        // lm_head = embed (tied)
        lm_head_w.K=embed_K; lm_head_w.N=V_dim;
        // Re-use embed GPU pointers — careful: embed and lm_head share same GPU mem
        // But we allocated embed differently. Need to upload.
        DevW4 embed_gpu;
        cudaMalloc(&embed_gpu.d,eds); cudaMemcpy(embed_gpu.d,host_embed_d,eds,cudaMemcpyHostToDevice);
        cudaMalloc(&embed_gpu.sc,ess*4); cudaMemcpy(embed_gpu.sc,host_embed_sc,ess*4,cudaMemcpyHostToDevice);
        lm_head_w.d=embed_gpu.d; lm_head_w.sc=embed_gpu.sc;
        printf("lm_head: tied to embed\n");
    }
    
    // Final norm
    {
        snprintf(p,256,"%s/final_norm.f32",wdir);
        FILE* f=fopen(p,"rb"); float* w=new float[H]; fread(w,4,H,f); fclose(f);
        cudaMemcpy(d_fn,w,H*4,cudaMemcpyHostToDevice); delete[] w;
    }
    
    printf("All weights loaded.\n\n── Generating ──\n");
    
    auto t_start=std::chrono::high_resolution_clock::now();
    std::vector<uint32_t> all_ids;
    int gen_start=(int)input_ids.size();
    
    for(int step=0;;step++){
        cudaStream_t st=0;
        
        if(step<gen_start){
            // Prompt step
            int tid=input_ids[step];
            float host_emb[H];
            dequant_embed_row(host_emb,tid,host_embed_d,host_embed_sc,H);
            die(cudaMemcpyAsync(d_x32,host_emb,H*4,cudaMemcpyHostToDevice,st),"embed_cpy");
        } else {
            // Generate — sample from last logit
            int next_id;
            die(cudaMemcpy(&next_id,d_next_id,4,cudaMemcpyDeviceToHost),"copy");
            all_ids.push_back(next_id);
            std::string txt=tokenizer.decode(next_id);
            printf("%s",txt.c_str());fflush(stdout);
            
            if(next_id==128001||next_id==128009){printf("\n[EOS]\n");break;}
            if((int)all_ids.size()-gen_start>=max_new) break;
            if(next_id==0) break;
            
            float host_emb[H];
            dequant_embed_row(host_emb,next_id,host_embed_d,host_embed_sc,H);
            die(cudaMemcpyAsync(d_x32,host_emb,H*4,cudaMemcpyHostToDevice,st),"embed_cpy");
        }
        
        // Save pre-norm state for residual connection
        die(cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st),"save_res");
        
        for(int l=0;l<NL;l++){
            // Pre-attention norm + quant
            die(blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_in,H,eps,st),"rmsnorm");
            die(blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st),"quant_qkv");
            
            // QKV projections
            die(blackwell::kernels::gemv_int4_warp(d_Q,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].q.d,W[l].q.sc,H,Q,st),"q_proj");
            die(blackwell::kernels::gemv_int4_warp(d_K,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].k.d,W[l].k.sc,H,KV,st),"k_proj");
            die(blackwell::kernels::gemv_int4_warp(d_V,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].v.d,W[l].v.sc,H,KV,st),"v_proj");
            
            // Head norm (Q, K)
            head_norm_kernel<<<nqh,hd>>>(d_Q,W[l].qn?W[l].qn:W[l].rn_in,nqh,hd,eps);
            head_norm_kernel<<<nkv,hd>>>(d_K,W[l].kn?W[l].kn:W[l].rn_in,nkv,hd,eps);
            
            // RoPE
            int seq_pos = step;
            apply_rope_kernel<<<nqh,hd/2>>>(d_Q,nqh,hd,seq_pos,rope_theta);
            apply_rope_kernel<<<nkv,hd/2>>>(d_K,nkv,hd,seq_pos,rope_theta);
            
            // KV cache update
            die(blackwell::kernels::update_kv_cache(d_kc,d_vc,d_K,d_V,0,seq_pos,
                nkv,hd,MAXSEQ,st),"kv_cache");
            
            // Attention
            die(blackwell::kernels::attention_decode_gqa(d_attn,d_Q,d_kc,d_vc,
                seq_pos,nqh,nkv,hd,MAXSEQ,st),"attn");
            
            // Output projection
            die(blackwell::kernels::quantize_int4(d_attn_i4,d_attn_i4_sc,d_attn,Q,st),"quant_attn");
            die(blackwell::kernels::gemv_int4_warp(d_proj,(const uint8_t*)d_attn_i4,
                d_attn_i4_sc,W[l].o.d,W[l].o.sc,Q,H,st),"o_proj");
            
            // Attention residual: d_x32 = d_proj + d_res
            die(blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,st),"attn_res");
            die(cudaMemcpyAsync(d_res,d_x32,H*4,cudaMemcpyDeviceToDevice,st),"save_res2");
            
            // Pre-MLP norm
            die(blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,W[l].rn_post,H,eps,st),"rmsnorm_post");
            die(blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st),"quant_mlp_in");
            
            // MLP gate + up
            die(blackwell::kernels::gemv_int4_warp(d_gate,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].g.d,W[l].g.sc,H,I,st),"gate");
            die(blackwell::kernels::gemv_int4_warp(d_up,(const uint8_t*)d_x_i4,d_x_i4_sc,W[l].u.d,W[l].u.sc,H,I,st),"up");
            
            // SwiGLU + quant
            die(blackwell::kernels::apply_swiglu(d_gate,d_gate,d_up,I,st),"swiglu");
            die(blackwell::kernels::quantize_int4(d_mlp_i4,d_mlp_i4_sc,d_gate,I,st),"quant_mlp");
            
            // Down projection
            die(blackwell::kernels::gemv_int4_warp(d_proj,(const uint8_t*)d_mlp_i4,
                d_mlp_i4_sc,W[l].d.d,W[l].d.sc,I,H,st),"down");
            
            // MLP residual
            die(blackwell::kernels::vector_add_fp32(d_x32,d_proj,d_res,H,st),"mlp_res");
        }
        
        // Final norm + lm_head + sampling
        if(step>=gen_start-1){
            die(blackwell::kernels::fused_rmsnorm(d_xi_f,d_x32,d_fn,H,eps,st),"fn");
            die(blackwell::kernels::quantize_int4(d_x_i4,d_x_i4_sc,d_xi_f,H,st),"quant_lm");
            die(blackwell::kernels::gemv_int4_warp(d_logits,(const uint8_t*)d_x_i4,
                d_x_i4_sc,lm_head_w.d,lm_head_w.sc,H,V,st),"lm_head");
            
            if(temperature<=0){
                die(blackwell::kernels::sample_argmax_gpu(d_logits,V,d_next_id,st),"sample");
            } else {
                die(blackwell::kernels::sample_gpu(d_logits,V,temperature,top_k,
                    d_next_id,0xdeadbeefLL,step,st),"sample");
            }
        }
        
        cudaStreamSynchronize(st);
    }
    
    auto t_end=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t_end-t_start).count();
    int gen=(int)all_ids.size()-gen_start;
    printf("\n\n── Stats ──\n");
    printf("  Input: %d  Gen: %d\n",gen_start,gen);
    printf("  Time: %.1f ms  Speed: %.1f ms/tok = %.0f t/s\n",ms,ms/gen,1000.0*gen/ms);
    
    // Cleanup
    for(int l=0;l<NL;l++){
        cudaFree(W[l].q.d);cudaFree(W[l].q.sc);
        cudaFree(W[l].k.d);cudaFree(W[l].k.sc);
        cudaFree(W[l].v.d);cudaFree(W[l].v.sc);
        cudaFree(W[l].o.d);cudaFree(W[l].o.sc);
        cudaFree(W[l].g.d);cudaFree(W[l].g.sc);
        cudaFree(W[l].u.d);cudaFree(W[l].u.sc);
        cudaFree(W[l].d.d);cudaFree(W[l].d.sc);
        cudaFree(W[l].qn);cudaFree(W[l].kn);
        cudaFree(W[l].rn_in);cudaFree(W[l].rn_post);
    }
    delete[] W;
    cudaFree(lm_head_w.d);cudaFree(lm_head_w.sc);
    delete[] host_embed_d;delete[] host_embed_sc;
    return 0;
}
