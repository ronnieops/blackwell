// bench/text_generate_speculative.cu — Speculative decoding: 1.7B draft + 8B target
//
// Draft (Qwen3-1.7B, 28L, H=2048) proposes K tokens.
// Target (Qwen3-8B, 36L, H=4096) verifies K tokens one-by-one.
// Accept matching tokens, resample from target on mismatch.
//
// Usage: ./bench/text_generate_speculative "prompt" max_tokens [K]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include "blackwell/kernels.h"
#include "blackwell/bpe_tokenizer.h"

#define CK(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess){fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));exit(1);} } while(0)

// Shared inline kernels
__global__ void head_norm_kernel(float* x, const float* g, int nheads, int hd, float eps) {
    int h=blockIdx.x, tid=threadIdx.x; if(h>=nheads) return;
    float*hx=x+h*hd; const float*gg=g+h*hd;
    float sum=0,sq=0;
    for(int i=tid;i<hd;i+=128){float v=hx[i]*gg[i]; sum+=v; sq+=v*v;}
    __shared__ float s[4]; s[0]=s[1]=0;
    float p=sum; for(int o=16;o>=1;o>>=1) p+=__shfl_xor_sync(0xffffffff,p,o);
    if(tid%32==0) s[0]=p; __syncthreads();
    float p2=sq; for(int o=16;o>=1;o>>=1) p2+=__shfl_xor_sync(0xffffffff,p2,o);
    if(tid%32==0) s[1]=p2; __syncthreads();
    float inv=1.0f/(sqrtf(s[1]/hd-(s[0]/hd)*(s[0]/hd))+eps);
    for(int i=tid;i<hd;i+=128) hx[i]=hx[i]*gg[i]*inv;
}
__global__ void apply_rope_kernel(float* x, int nh, int hd, int pos) {
    int h=blockIdx.x, i=threadIdx.x; if(h>=nh||i>=hd/2) return;
    int d=i*2; float theta=pos*powf(1000000.0f,-2.0f*(float)d/(float)hd);
    float cs=cosf(theta),sn=sinf(theta);
    float x0=x[h*hd+d],x1=x[h*hd+d+1];
    x[h*hd+d]=x0*cs-x1*sn; x[h*hd+d+1]=x0*sn+x1*cs;
}

struct DevW4 { uint8_t* d; float* sc; int K,N; };
struct LW { DevW4 q,k,v,o,g,u,d; float *rn_in,*rn_post,*qn,*kn; };

static DevW4 upload_w4(const char* prefix) {
    DevW4 w={}; char p[512]; size_t sz; int h[5];
    snprintf(p,512,"%s.int4_t",prefix);
    FILE*f=fopen(p,"rb"); if(!f){fprintf(stderr,"FAIL open %s\n",p);exit(1);}
    fread(h,4,5,f); w.N=h[0]; w.K=h[1]; sz=(size_t)w.N*w.K/2;
    uint8_t* buf=new uint8_t[sz]; fread(buf,1,sz,f); fclose(f);
    CK(cudaMalloc(&w.d,sz)); CK(cudaMemcpy(w.d,buf,sz,cudaMemcpyHostToDevice)); delete[]buf;
    snprintf(p,512,"%s.scale_t",prefix);
    f=fopen(p,"rb"); fread(h,4,5,f); size_t ss=h[3]*h[4];
    float*sbuf=new float[ss]; fread(sbuf,4,ss,f); fclose(f);
    CK(cudaMalloc(&w.sc,ss*4)); CK(cudaMemcpy(w.sc,sbuf,ss*4,cudaMemcpyHostToDevice)); delete[]sbuf;
    return w;
}

static void dequant_embed(float*out,int tok,const uint8_t*hd,const float*hs,int H) {
    int nb=H/16; size_t ro=(size_t)tok*H/2;
    for(int b=0;b<nb;b++){float sc=hs[(size_t)tok*nb+b];
        for(int j=0;j<8;j++){uint8_t by=hd[ro+b*8+j];
            out[b*16+j*2]=((int)(by&0xF)-8)*sc; out[b*16+j*2+1]=((int)((by>>4)&0xF)-8)*sc;}}
}

// Model struct: holds all state for one model
struct Model {
    int H,Q,KV,I,NQH,NKV,HD,NL,MAXSEQ; float eps;
    LW* W; DevW4 embed, lm;
    float *fn, *kc, *vc;
    float *x32,*xi_f,*res,*Q_,*K_,*V_,*attn,*proj,*up;
    uint8_t *x_i4,*attn_i4,*mlp_i4;
    float *x_i4_sc,*attn_i4_sc,*mlp_i4_sc;
    uint8_t* host_ed; float* host_es;
    float *logits; int *next_id;
    cudaStream_t st;
};

static void model_load(Model& m, const char* dir, int H,int Q,int KV,int I,int NQH,int NKV,int HD,int NL,int MAXSEQ,float eps) {
    m.H=H;m.Q=Q;m.KV=KV;m.I=I;m.NQH=NQH;m.NKV=NKV;m.HD=HD;m.NL=NL;m.MAXSEQ=MAXSEQ;m.eps=eps;
    m.W=new LW[NL];
    fprintf(stderr,"  Loading %d-layer model from %s...\n",NL,dir);
    for(int l=0;l<NL;l++){char p[256];
        snprintf(p,256,"%s/%d_self_attn.q_proj",dir,l); m.W[l].q=upload_w4(p);
        snprintf(p,256,"%s/%d_self_attn.k_proj",dir,l); m.W[l].k=upload_w4(p);
        snprintf(p,256,"%s/%d_self_attn.v_proj",dir,l); m.W[l].v=upload_w4(p);
        snprintf(p,256,"%s/%d_self_attn.o_proj",dir,l); m.W[l].o=upload_w4(p);
        snprintf(p,256,"%s/%d_mlp.gate_proj",dir,l);    m.W[l].g=upload_w4(p);
        snprintf(p,256,"%s/%d_mlp.up_proj",dir,l);       m.W[l].u=upload_w4(p);
        snprintf(p,256,"%s/%d_mlp.down_proj",dir,l);     m.W[l].d=upload_w4(p);
    }
    {char p[256];snprintf(p,256,"%s/embed_tokens",dir);m.embed=upload_w4(p);
     snprintf(p,256,"%s/lm_head",dir);m.lm=upload_w4(p);}
    // Host embed
    {char p[256];int h[5];snprintf(p,256,"%s/embed_tokens.int4_t",dir);
     FILE*f=fopen(p,"rb");fread(h,4,5,f);size_t ds=(size_t)h[0]*h[1]/2;
     m.host_ed=new uint8_t[ds];fread(m.host_ed,1,ds,f);fclose(f);
     snprintf(p,256,"%s/embed_tokens.scale_t",dir);
     f=fopen(p,"rb");fread(h,4,5,f);size_t ss=h[3]*h[4];
     m.host_es=new float[ss];fread(m.host_es,4,ss,f);fclose(f);}
    // Norms
    float* qk_h=new float[NL*2*HD]; memset(qk_h,0,NL*2*HD*4);
    {char p[256];snprintf(p,256,"%s/qk_norms.f32",dir);
     FILE*f=fopen(p,"rb");if(f){fread(qk_h,4,NL*2*HD,f);fclose(f);}}
    CK(cudaMalloc(&m.fn,H*4));
    {char p[256];snprintf(p,256,"%s/final_norm.f32",dir);
     FILE*f=fopen(p,"rb");float*w=new float[H];fread(w,4,H,f);fclose(f);
     CK(cudaMemcpy(m.fn,w,H*4,cudaMemcpyHostToDevice));delete[]w;}
    for(int l=0;l<NL;l++){
        CK(cudaMalloc(&m.W[l].rn_in,H*4));CK(cudaMalloc(&m.W[l].rn_post,H*4));
        CK(cudaMalloc(&m.W[l].qn,HD*4));CK(cudaMalloc(&m.W[l].kn,HD*4));
        char p[256];float*w=new float[H];
        snprintf(p,256,"%s/%d_input_layernorm.f32",dir,l);
        {FILE*f=fopen(p,"rb");fread(w,4,H,f);fclose(f);CK(cudaMemcpy(m.W[l].rn_in,w,H*4,cudaMemcpyHostToDevice));}
        snprintf(p,256,"%s/%d_post_attention_layernorm.f32",dir,l);
        {FILE*f=fopen(p,"rb");fread(w,4,H,f);fclose(f);CK(cudaMemcpy(m.W[l].rn_post,w,H*4,cudaMemcpyHostToDevice));}
        delete[]w;
        CK(cudaMemcpy(m.W[l].qn,qk_h+l*2*HD,HD*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(m.W[l].kn,qk_h+l*2*HD+HD,HD*4,cudaMemcpyHostToDevice));
    }
    delete[]qk_h;
    // Buffers
    auto AL=[&](float**p,size_t n){CK(cudaMalloc(p,n));};
    AL(&m.x32,H*4);AL(&m.xi_f,H*4);AL(&m.res,H*4);
    AL(&m.Q_,Q*4);AL(&m.K_,KV*4);AL(&m.V_,KV*4);AL(&m.attn,Q*4);
    AL(&m.proj,(size_t)std::max({H,Q,I})*4);AL(&m.up,I*4);
    AL(&m.kc,(size_t)NL*NKV*MAXSEQ*HD*4);AL(&m.vc,(size_t)NL*NKV*MAXSEQ*HD*4);
    AL(&m.logits,(size_t)151936*4);CK(cudaMalloc(&m.next_id,4));
    CK(cudaMalloc(&m.x_i4,H/2));CK(cudaMalloc(&m.x_i4_sc,(H/16)*4));
    CK(cudaMalloc(&m.attn_i4,Q/2));CK(cudaMalloc(&m.attn_i4_sc,(Q/16)*4));
    CK(cudaMalloc(&m.mlp_i4,I/2));CK(cudaMalloc(&m.mlp_i4_sc,(I/16)*4));
    CK(cudaMemset(m.kc,0,(size_t)NL*NKV*MAXSEQ*HD*4));
    CK(cudaMemset(m.vc,0,(size_t)NL*NKV*MAXSEQ*HD*4));
    fprintf(stderr,"  embed %dx%d lm %dx%d\n",m.embed.K,m.embed.N,m.lm.K,m.lm.N);
}

// Forward one token through all layers
static void model_forward(Model& m, int token, int pos) {
    float* h_emb=new float[m.H];
    dequant_embed(h_emb,token,m.host_ed,m.host_es,m.H);
    CK(cudaMemcpyAsync(m.x32,h_emb,m.H*4,cudaMemcpyHostToDevice,m.st));
    delete[]h_emb;

    for(int l=0;l<m.NL;l++){
        CK(cudaMemcpyAsync(m.res,m.x32,m.H*4,cudaMemcpyDeviceToDevice,m.st));
        CK(blackwell::kernels::fused_rmsnorm(m.xi_f,m.x32,m.W[l].rn_in,m.H,m.eps,m.st));
        CK(blackwell::kernels::quantize_int4_batched(m.x_i4,m.x_i4_sc,m.xi_f,m.H,1,m.st));
        CK(blackwell::kernels::gemv_int4_batched(m.Q_,m.x_i4,m.x_i4_sc,m.W[l].q.d,m.W[l].q.sc,m.H,m.Q,1,m.st));
        CK(blackwell::kernels::gemv_int4_batched(m.K_,m.x_i4,m.x_i4_sc,m.W[l].k.d,m.W[l].k.sc,m.H,m.KV,1,m.st));
        CK(blackwell::kernels::gemv_int4_batched(m.V_,m.x_i4,m.x_i4_sc,m.W[l].v.d,m.W[l].v.sc,m.H,m.KV,1,m.st));
        head_norm_kernel<<<m.NQH,128,0,m.st>>>(m.Q_,m.W[l].qn,m.NQH,m.HD,m.eps);
        head_norm_kernel<<<m.NKV,128,0,m.st>>>(m.K_,m.W[l].kn,m.NKV,m.HD,m.eps);
        apply_rope_kernel<<<m.NQH,m.HD/2,0,m.st>>>(m.Q_,m.NQH,m.HD,pos);
        apply_rope_kernel<<<m.NKV,m.HD/2,0,m.st>>>(m.K_,m.NKV,m.HD,pos);
        size_t kvo=(size_t)l*m.NKV*m.MAXSEQ*m.HD;
        CK(blackwell::kernels::update_kv_cache(m.kc+kvo,m.vc+kvo,m.K_,m.V_,0,pos,m.NKV,m.HD,m.MAXSEQ,m.st));
        CK(blackwell::kernels::attention_decode_batched_gqa(m.attn,m.Q_,m.kc,m.vc,pos,m.NQH,m.NKV,m.HD,m.MAXSEQ,1,
            (size_t)m.NL*m.NKV*m.MAXSEQ*m.HD,kvo,m.st));
        CK(blackwell::kernels::quantize_int4_batched(m.attn_i4,m.attn_i4_sc,m.attn,m.Q,1,m.st));
        CK(blackwell::kernels::gemv_int4_batched(m.proj,m.attn_i4,m.attn_i4_sc,m.W[l].o.d,m.W[l].o.sc,m.Q,m.H,1,m.st));
        CK(blackwell::kernels::vector_add_fp32(m.x32,m.proj,m.res,m.H,m.st));
        CK(cudaMemcpyAsync(m.res,m.x32,m.H*4,cudaMemcpyDeviceToDevice,m.st));
        CK(blackwell::kernels::fused_rmsnorm(m.xi_f,m.x32,m.W[l].rn_post,m.H,m.eps,m.st));
        CK(blackwell::kernels::quantize_int4_batched(m.x_i4,m.x_i4_sc,m.xi_f,m.H,1,m.st));
        CK(blackwell::kernels::gemv_int4_batched(m.proj,m.x_i4,m.x_i4_sc,m.W[l].g.d,m.W[l].g.sc,m.H,m.I,1,m.st));
        CK(blackwell::kernels::gemv_int4_batched(m.up,m.x_i4,m.x_i4_sc,m.W[l].u.d,m.W[l].u.sc,m.H,m.I,1,m.st));
        blackwell::kernels::apply_swiglu(m.proj,m.proj,m.up,m.I,m.st);
        CK(blackwell::kernels::quantize_int4_batched(m.mlp_i4,m.mlp_i4_sc,m.proj,m.I,1,m.st));
        CK(blackwell::kernels::gemv_int4_batched(m.proj,m.mlp_i4,m.mlp_i4_sc,m.W[l].d.d,m.W[l].d.sc,m.I,m.H,1,m.st));
        CK(blackwell::kernels::vector_add_fp32(m.x32,m.proj,m.res,m.H,m.st));
    }
}

static int model_sample(Model& m) {
    CK(blackwell::kernels::fused_rmsnorm(m.xi_f,m.x32,m.fn,m.H,m.eps,m.st));
    CK(blackwell::kernels::quantize_int4_batched(m.x_i4,m.x_i4_sc,m.xi_f,m.H,1,m.st));
    CK(blackwell::kernels::gemv_int4_batched(m.logits,m.x_i4,m.x_i4_sc,m.lm.d,m.lm.sc,m.H,151936,1,m.st));
    CK(blackwell::kernels::sample_gpu(m.logits,151936,0.0f,0,m.next_id,0xdeadbeefLL,0,m.st));
    int id; CK(cudaMemcpy(&id,m.next_id,4,cudaMemcpyDeviceToHost)); return id;
}

int main(int argc,char**argv){
    const char* prompt=argc>1?argv[1]:"The capital of France is";
    int max_new=argc>2?atoi(argv[2]):30;
    int K=argc>3?atoi(argv[3]):4;

    printf("── Speculative Decoding: 1.7B draft → 8B target ──\n");
    printf("Prompt: \"%s\"  K=%d  max_new=%d\n\n",prompt,K,max_new);

    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    printf("Device: %s (CC %d.%d)\n",P.name,P.major,P.minor);

    Model draft={}, target={};
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    draft.st = st;
    target.st = st;

    blackwell::BpeTokenizer tok;
    if(tok.load("tokenizer_data.bin")!=0){fprintf(stderr,"FAIL: tokenizer\n");return 1;}

    model_load(draft,"weights_int4_qwen3_1.7b", 2048,2048,1024,6144, 16,8,128, 28,512, 1e-6f);
    CK(cudaGetLastError());
    model_load(target,"weights_int4_qwen3_8b", 4096,4096,1024,12288, 32,8,128, 36,512, 1e-6f);
    CK(cudaGetLastError());

    size_t gf,gt; cudaMemGetInfo(&gf,&gt);
    printf("GPU: %.0f MB used\n\n",(gt-gf)/1024.0/1024);

    auto ids=tok.encode(prompt);
    int gs=(int)ids.size();
    printf("Input: %d tokens. Prefilling...\n",gs);

    // Prefill both
    // Prefill both models
    // Prefill with per-step sync to find the crash
    for(int s=0;s<gs;s++){
        model_forward(draft,ids[s],s);
        model_forward(target,ids[s],s);
    }
    CK(cudaStreamSynchronize(st));

    // ── Speculative decode ──
    printf("Generating (K=%d)...\n",K);
    auto t0=std::chrono::steady_clock::now();
    std::vector<uint32_t> all=ids;
    int pos=gs, accepted=0, drafted=0;

    while((int)all.size()-gs<max_new){
        // Draft proposes K tokens
        std::vector<int> dtoks;
        for(int k=0;k<K;k++){
            model_forward(draft,all.back(),pos+k);
            dtoks.push_back(model_sample(draft));
            all.push_back(dtoks.back());
            drafted++;
        }

        // Target verifies
        int acc=0;
        for(int k=0;k<K;k++){
            model_forward(target,all[pos+k],pos+k);
            int tt=model_sample(target);
            if(tt==dtoks[k]) acc++;
            else { all[pos+k]=tt; all.resize(pos+k+1); break; }
        }
        if(acc==K){
            model_forward(target,all.back(),pos+K);
            all.push_back(model_sample(target));
            acc++;
        }
        accepted+=acc; pos=(int)all.size();
        if(all.back()==151643) break;
    }

    auto t1=std::chrono::steady_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    int gen=(int)all.size()-gs;

    printf("\n── Results ──\n");
    printf("Generated: %d tokens in %.0f ms\n",gen,ms);
    printf("Speed: %.1f ms/tok = %.0f t/s\n",ms/gen,gen/(ms/1000));
    printf("Acceptance: %d/%d (%.1f%%)\n",accepted,drafted,100.0*accepted/drafted);
    printf("Text: ");
    for(int i=gs;i<(int)all.size();i++) printf("%s",tok.decode(all[i]).c_str());
    printf("\n");
    return 0;
}
