// bench/text_generate.cu — End-to-end text generation with INT8 Qwen3-1.7B
//
// Tokenize prompt → embedding lookup → 28L decode → argmax → print tokens.
// Uses real INT8 weights from weights_int8_bf16/ and BPE tokenizer.
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/text_generate.cu build/libblackwell_kernels.a \
//     -o bench/text_generate
//
// Run: ./bench/text_generate "Once upon a time" [max_new_tokens=50]
//   Chat mode: ./bench/text_generate "Your question" 50 --chat

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cstring>
#include <string>
#include "blackwell/kernels.h"
#include "blackwell/bpe_tokenizer.h"

static void die(cudaError_t e, const char* m) {
    if(e!=cudaSuccess){printf("FAIL %s: %s\n",m,cudaGetErrorString(e));exit(1);}
}

// Compute 16-element block absmax scales for pack_int8
__global__ void absmax_scales_kernel(const float* in, float* sc, int n) {
    int blk=blockIdx.x; int lane=threadIdx.x; float amax=0;
    for(int i=lane;i<16&&blk*16+i<n;i+=32) amax=fmaxf(amax,fabsf(in[blk*16+i]));
    for(int off=16;off>0;off>>=1) amax=fmaxf(amax,__shfl_xor_sync(0xffffffff,amax,off));
    if(lane==0) sc[blk]=fmaxf(amax/127.0f,1e-9f);
}

// Per-head RMSNorm for Q/K norms
__global__ void head_norm_kernel(float* data, const float* weight, int nh, int hd, float eps) {
    int h=blockIdx.x; if(h>=nh) return;
    float* d=data+h*hd;
    __shared__ float warp_partial[4];
    float s=0;
    int tid=threadIdx.x;
    for(int i=tid;i<hd;i+=blockDim.x) s+=d[i]*d[i];
    for(int off=16;off>0;off>>=1) s+=__shfl_xor_sync(0xffffffff,s,off);
    if((tid&31)==0) warp_partial[tid>>5]=s;
    __syncthreads();
    if(tid<4) s=warp_partial[tid]; else s=0;
    for(int off=2;off>0;off>>=1) s+=__shfl_xor_sync(0xffffffff,s,off);
    if(tid==0) warp_partial[0]=rsqrtf(s/hd+eps);
    __syncthreads();
    float is=warp_partial[0];
    for(int i=tid;i<hd;i+=blockDim.x) d[i]=d[i]*is*weight[i];
}

// RoPE kernel: apply rotary position embeddings to Q and K
// Qwen3: rotate each pair (2i, 2i+1) by θ_i × pos
__global__ void apply_rope_kernel(float* data, int n_heads, int head_dim, int pos) {
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (h >= n_heads || d >= head_dim/2) return;
    int i2 = d * 2;
    float* pair = data + h * head_dim + i2;
    // Standard RoPE: theta_i = pos * base^(-2*i/d) for pair index i=0..hd/2-1
    // Qwen3-1.7B uses rope_theta=1000000 (config.json), not Llama default 10000
    const float rope_theta = 1000000.0f;
    float theta = (float)pos * powf(rope_theta, -2.0f * (float)d / (float)head_dim);
    float c = cosf(theta), s = sinf(theta);
    float x = pair[0], y = pair[1];
    pair[0] = x * c - y * s;
    pair[1] = x * s + y * c;
}

using Clock = std::chrono::high_resolution_clock;

// Model constants — Qwen3-1.7B
const int H=2048, QD=2048, KV=1024, ID=6144;
const int nqh=16, nkv=8, hd=128, MAXSEQ=4096;
const float eps=1e-6f;
const int V=151936;  // vocab size

// ── Weight structures ────────────────────────────────────────────────
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

struct L { DW q,k,v,o,g,u,d; float* qn; float* kn; };

// Host-side argmax over 151936-element logit buffer
static int argmax_host(const float* logits, int n) {
    int best=0; float bv=logits[0];
    for(int i=1;i<n;i++) if(logits[i]>bv){bv=logits[i];best=i;}
    return best;
}

// Temperature + top-k sampling
static int sample(const float* logits, int n, float temp, int top_k) {
    if (temp < 0.01f) return argmax_host(logits, n);
    float threshold = -1e38f;
    if (top_k > 0 && top_k < n) {
        std::vector<float> tmp(n);
        for (int i = 0; i < n; i++) tmp[i] = logits[i];
        std::nth_element(tmp.begin(), tmp.begin() + n - top_k, tmp.end(),
            [](float a, float b){ return a > b; });
        threshold = tmp[n - top_k];
    }
    std::vector<float> valid_logits;
    for (int i = 0; i < n; i++) {
        if (logits[i] >= threshold) valid_logits.push_back(logits[i]);
    }
    int n_valid = (int)valid_logits.size();
    if (n_valid == 0) return argmax_host(logits, n);
    float max_logit = valid_logits[0];
    for (int i = 1; i < n_valid; i++) if (valid_logits[i] > max_logit) max_logit = valid_logits[i];
    std::vector<float> probs(n_valid);
    float sum = 0.0f;
    for (int i = 0; i < n_valid; i++) {
        probs[i] = expf((valid_logits[i] - max_logit) / temp);
        sum += probs[i];
    }
    float r = (float)rand() / (float)RAND_MAX * sum;
    float cum = 0.0f;
    for (int i = 0; i < n_valid; i++) {
        cum += probs[i];
        if (r <= cum) {
            int found = 0;
            for (int j = 0; j < n; j++) {
                if (logits[j] >= threshold) {
                    if (found == i) return j;
                    found++;
                }
            }
            break;
        }
    }
    return argmax_host(logits, n);
}

int main(int argc, char** argv) {
    const char* prompt = "Once upon a time";
    int max_new = 50;
    bool chat_mode = false;
    float temperature = 1.0f;
    int top_k = 0;
    if (argc > 1) prompt = argv[1];
    if (argc > 2) max_new = atoi(argv[2]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i],"--chat")==0) chat_mode = true;
        if (strcmp(argv[i],"-t")==0 && i+1<argc) temperature = atof(argv[++i]);
        if (strcmp(argv[i],"-k")==0 && i+1<argc) top_k = atoi(argv[++i]);
    }

    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    printf("# Text Generation — Qwen3-1.7B INT8\n");
    printf("  Device: %s\n", P.name);
    printf("  Prompt: \"%s\"%s\n", prompt, chat_mode ? " (chat mode)" : "");
    printf("  Temperature: %.1f%s, Top-K: %d\n", temperature, temperature < 0.01f ? " (greedy)" : "", top_k);
    printf("  Max new tokens: %d\n\n", max_new);

    // Load tokenizer
    blackwell::BpeTokenizer tokenizer;
    if (tokenizer.load("tokenizer_data.bin") != 0) {
        fprintf(stderr, "FAIL: no tokenizer_data.bin (run: python3 scripts/prepare_tokenizer.py)\n"); return 1;
    }

    // Build input sequence
    std::vector<uint32_t> input_ids;
    if (chat_mode) {
        input_ids.push_back(151644);
        for (char c : std::string("user\n")) input_ids.push_back((uint32_t)(unsigned char)c);
        auto prompt_toks = tokenizer.encode(prompt);
        input_ids.insert(input_ids.end(), prompt_toks.begin(), prompt_toks.end());
        input_ids.push_back(151645);
        input_ids.push_back(151644);
        for (char c : std::string("assistant\n")) input_ids.push_back((uint32_t)(unsigned char)c);
    } else {
        input_ids = tokenizer.encode(prompt);
    }

    printf("Input: %zu tokens\n\n", input_ids.size());

    // Allocate device memory
    float *d_x, *d_xi_f, *d_xs;
    float *d_Q, *d_K, *d_V, *d_attn;
    int8_t *d_ai; float *d_as;
    float *d_gate, *d_up, *d_mlp;
    int8_t *d_mi; float *d_ms;
    float *d_proj, *d_res_save, *d_res_save2;
    float *d_kc, *d_vc;
    float *d_fn, *d_fn_sc, *d_logits;
    int8_t *d_emb_d; float *d_emb_sc;

    const int NL=28;

    #define AL(p,n) { cudaError_t _e=cudaMalloc(&(p),(n)); if(_e!=cudaSuccess) printf("FAIL malloc %s: %s\n",#p,cudaGetErrorString(_e)); die(_e,#p); }
    AL(d_x,H*4); AL(d_xi_f,H*4); AL(d_xs,(H/16)*4);
    AL(d_Q,QD*4); AL(d_K,KV*4); AL(d_V,KV*4);
    AL(d_attn,QD*4); AL(d_ai,QD); AL(d_as,(QD/16)*4);
    AL(d_gate,ID*4); AL(d_up,ID*4);
    AL(d_mlp,ID*4); AL(d_mi,ID); AL(d_ms,(ID/16)*4);
    AL(d_proj,H*4); AL(d_res_save,H*4); AL(d_res_save2,H*4);
    AL(d_kc,NL*nkv*MAXSEQ*hd*4); AL(d_vc,NL*nkv*MAXSEQ*hd*4);
    AL(d_fn,H*4); AL(d_fn_sc,(H/16)*4); AL(d_logits,V*4);
    #undef AL

    // Init scale buffers to 1/127
    float iv=1.f/127.f;
    cudaMemcpy(d_xs,&iv,4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_as,&iv,4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_ms,&iv,4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_fn_sc,&iv,4,cudaMemcpyHostToDevice);

    // Load weights
    printf("Loading %d-layer model...\n", NL); fflush(stdout);
    std::vector<L> W(NL); char p_[256];
    for(int l=0;l<NL;l++){
        snprintf(p_,256,"weights_int8_bf16/%d_self_attn.q_proj",l); W[l].q=dw(lw(p_));
        snprintf(p_,256,"weights_int8_bf16/%d_self_attn.k_proj",l); W[l].k=dw(lw(p_));
        snprintf(p_,256,"weights_int8_bf16/%d_self_attn.v_proj",l); W[l].v=dw(lw(p_));
        snprintf(p_,256,"weights_int8_bf16/%d_self_attn.o_proj",l); W[l].o=dw(lw(p_));
        snprintf(p_,256,"weights_int8_bf16/%d_mlp.gate_proj",l);   W[l].g=dw(lw(p_));
        snprintf(p_,256,"weights_int8_bf16/%d_mlp.up_proj",l);     W[l].u=dw(lw(p_));
        snprintf(p_,256,"weights_int8_bf16/%d_mlp.down_proj",l);  W[l].d=dw(lw(p_));
        if((l+1)%7==0||l+1==NL) printf("  layer %d/%d\n",l+1,NL);
    }

    // Q/K norms
    float* qk_h=(float*)malloc(28*2*128*4);
    {FILE* f=fopen("weights_int8_bf16/qk_norms.f32","rb");
    (void)fread(qk_h,4,28*2*128,f);fclose(f);}
    for(int l=0;l<NL;l++){
        cudaMalloc(&W[l].qn,128*4); cudaMemcpy(W[l].qn,qk_h+l*2*128,128*4,cudaMemcpyHostToDevice);
        cudaMalloc(&W[l].kn,128*4); cudaMemcpy(W[l].kn,qk_h+l*2*128+128,128*4,cudaMemcpyHostToDevice);
    }
    free(qk_h);

    // Per-layer RMSNorm weights
    std::vector<float*> d_rn_in(NL), d_rn_post(NL);
    for(int l=0;l<NL;l++){
        float* w=(float*)malloc(H*4);
        snprintf(p_,256,"weights_int8_bf16/%d_input_layernorm.f32",l);
        {FILE* f=fopen(p_,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&d_rn_in[l],H*4); cudaMemcpy(d_rn_in[l],w,H*4,cudaMemcpyHostToDevice);
        snprintf(p_,256,"weights_int8_bf16/%d_post_attention_layernorm.f32",l);
        {FILE* f=fopen(p_,"rb");(void)fread(w,4,H,f);fclose(f);}
        cudaMalloc(&d_rn_post[l],H*4); cudaMemcpy(d_rn_post[l],w,H*4,cudaMemcpyHostToDevice);
        free(w);
    }

    // Final norm
    {float* w=(float*)malloc(H*4);
    FILE* f=fopen("weights_int8_bf16/final_norm.f32","rb");
    (void)fread(w,4,H,f);fclose(f);
    cudaMemcpy(d_fn,w,H*4,cudaMemcpyHostToDevice); free(w);}

    // Embed tokens
    LW emb = lw("weights_int8_bf16/embed_tokens");
    cudaMalloc(&d_emb_d,emb.d.size());  cudaMemcpy(d_emb_d,emb.d.data(),emb.d.size(),cudaMemcpyHostToDevice);
    cudaMalloc(&d_emb_sc,emb.sc.size()*4); cudaMemcpy(d_emb_sc,emb.sc.data(),emb.sc.size()*4,cudaMemcpyHostToDevice);

    printf("All weights loaded.\n\n");

    cudaStream_t st; die(cudaStreamCreate(&st),"stream");
    srand((unsigned)time(nullptr));

    // Host buffers
    std::vector<float> h_embed(H);
    std::vector<float> h_logits(V);

    // Clear KV cache
    cudaMemset(d_kc,0,NL*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vc,0,NL*nkv*MAXSEQ*hd*4);

    // ══════════════════════════════════════════════════════════════════
    printf("── Generating ──\n");
    if (chat_mode) { printf("[assistant] "); } else { printf("%s", prompt); }
    fflush(stdout);

    std::vector<uint32_t> all_ids = input_ids;
    int gen_start = (int)input_ids.size();
    int total = gen_start + max_new;

    auto t_start = Clock::now();

    for(int step=0; step<total; step++) {
        uint32_t tid = (step < gen_start) ? input_ids[step] : all_ids.back();

        // Embedding lookup: dequantize INT8 row → float32 host
        for(int d=0; d<H; d++){
            h_embed[d] = (float)emb.d[tid * H + d] * emb.sc[tid * (H/16) + d/16];
        }
        die(cudaMemcpy(d_x,h_embed.data(),H*4,cudaMemcpyHostToDevice),"embed_cpy");

        // 28 layers — per-kernel INT8 (fast __dp4a path)
        for(int l=0;l<NL;l++) {
            float* input = (l==0) ? d_x : d_proj;
            die(cudaMemcpyAsync(d_res_save,input,H*4,cudaMemcpyDeviceToDevice,st),"save_res");

            // 1. Input RMSNorm (FP32 output)
            {
                cudaError_t e = blackwell::kernels::fused_rmsnorm(
                    d_xi_f,input,d_rn_in[l],H,eps,st);
                if(cudaGetLastError()!=cudaSuccess||e!=cudaSuccess){printf("FAIL rmsnorm_in l=%d: %s\n",l,cudaGetErrorString(e));exit(1);}
            }

            // 2. QKV (FP32 × INT8 per-row)
            {
                cudaError_t e;
                e=blackwell::kernels::gemv_fp32_int8_per_row(d_Q,d_xi_f,W[l].q.d,W[l].q.sc,H,QD,st);
                if(cudaGetLastError()!=cudaSuccess||e!=cudaSuccess){printf("FAIL q l=%d\n",l);exit(1);}
                e=blackwell::kernels::gemv_fp32_int8_per_row(d_K,d_xi_f,W[l].k.d,W[l].k.sc,H,KV,st);
                if(cudaGetLastError()!=cudaSuccess||e!=cudaSuccess){printf("FAIL k l=%d\n",l);exit(1);}
                e=blackwell::kernels::gemv_fp32_int8_per_row(d_V,d_xi_f,W[l].v.d,W[l].v.sc,H,KV,st);
                if(cudaGetLastError()!=cudaSuccess||e!=cudaSuccess){printf("FAIL v l=%d\n",l);exit(1);}
            }

            // 3. Q/K head norms
            head_norm_kernel<<<nqh,128,0,st>>>(d_Q,W[l].qn,nqh,hd,eps);
            if(cudaGetLastError()!=cudaSuccess){printf("FAIL head_norm_Q l=%d\n",l);exit(1);}
            head_norm_kernel<<<nkv,128,0,st>>>(d_K,W[l].kn,nkv,hd,eps);
            if(cudaGetLastError()!=cudaSuccess){printf("FAIL head_norm_K l=%d\n",l);exit(1);}

            // 4. RoPE (fixed theta=1000000 for Qwen3)
            apply_rope_kernel<<<nqh,hd/2,0,st>>>(d_Q,nqh,hd,step);
            if(cudaGetLastError()!=cudaSuccess){printf("FAIL rope_Q l=%d\n",l);exit(1);}
            apply_rope_kernel<<<nkv,hd/2,0,st>>>(d_K,nkv,hd,step);
            if(cudaGetLastError()!=cudaSuccess){printf("FAIL rope_K l=%d\n",l);exit(1);}

            // Per-layer KV cache offset
            int kb = l * nkv * MAXSEQ * hd;

            // 5. KV cache + attention
            {
                cudaError_t e;
                e=blackwell::kernels::update_kv_cache(d_kc+kb,d_vc+kb,d_K,d_V,0,step,nkv,hd,MAXSEQ,st);
                if(cudaGetLastError()!=cudaSuccess||e!=cudaSuccess){printf("FAIL kv l=%d\n",l);exit(1);}
                e=blackwell::kernels::attention_decode_gqa(d_attn,d_Q,d_kc+kb,d_vc+kb,step,nqh,nkv,hd,MAXSEQ,st);
                if(cudaGetLastError()!=cudaSuccess||e!=cudaSuccess){printf("FAIL attn l=%d\n",l);exit(1);}
            }

            // 6. Wo GEMV + residual 1 (FP32 activations)
            {
                cudaError_t e;
                e=blackwell::kernels::gemv_fp32_int8_per_row(d_proj,d_attn,W[l].o.d,W[l].o.sc,QD,H,st);
                if(cudaGetLastError()!=cudaSuccess||e!=cudaSuccess){printf("FAIL o l=%d\n",l);exit(1);}
                e=blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_res_save,H,st);
                if(cudaGetLastError()!=cudaSuccess||e!=cudaSuccess){printf("FAIL res1 l=%d\n",l);exit(1);}
            }
            die(cudaMemcpyAsync(d_res_save2,d_proj,H*4,cudaMemcpyDeviceToDevice,st),"save_res2");

            // 7. Post-attention RMSNorm (FP32)
            {
                cudaError_t e=blackwell::kernels::fused_rmsnorm(
                    d_xi_f,d_proj,d_rn_post[l],H,eps,st);
                if(cudaGetLastError()!=cudaSuccess||e!=cudaSuccess){printf("FAIL rmsnorm_post l=%d\n",l);exit(1);}
            }

            // 8. Gate + Up GEMVs + SwiGLU (FP32 activations)
            {
                cudaError_t e;
                e=blackwell::kernels::gemv_fp32_int8_per_row(d_gate,d_xi_f,W[l].g.d,W[l].g.sc,H,ID,st);
                if(cudaGetLastError()!=cudaSuccess||e!=cudaSuccess){printf("FAIL gate l=%d\n",l);exit(1);}
                e=blackwell::kernels::gemv_fp32_int8_per_row(d_up,d_xi_f,W[l].u.d,W[l].u.sc,H,ID,st);
                if(cudaGetLastError()!=cudaSuccess||e!=cudaSuccess){printf("FAIL up l=%d\n",l);exit(1);}
                e=blackwell::kernels::apply_swiglu(d_mlp,d_gate,d_up,ID,st);
                if(cudaGetLastError()!=cudaSuccess||e!=cudaSuccess){printf("FAIL swiglu l=%d\n",l);exit(1);}
            }

            // 9. Down GEMV + residual 2 (FP32 activations)
            {
                cudaError_t e;
                e=blackwell::kernels::gemv_fp32_int8_per_row(d_proj,d_mlp,W[l].d.d,W[l].d.sc,ID,H,st);
                if(cudaGetLastError()!=cudaSuccess||e!=cudaSuccess){printf("FAIL down l=%d\n",l);exit(1);}
                e=blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_res_save2,H,st);
                if(cudaGetLastError()!=cudaSuccess||e!=cudaSuccess){printf("FAIL res2 l=%d\n",l);exit(1);}
            }
        } // end for(int l=0;l<NL;l++)

        // Final normalize + lm_head
        if(step >= gen_start - 1) {
            {
                cudaError_t e=blackwell::kernels::fused_rmsnorm(
                    d_xi_f,d_proj,d_fn,H,eps,st);
                if(cudaGetLastError()!=cudaSuccess||e!=cudaSuccess){printf("FAIL fn step=%d\n",step);exit(1);}
            }
            {
                cudaError_t e=blackwell::kernels::gemv_fp32_int8_per_row(d_logits,d_xi_f,
                    d_emb_d,d_emb_sc,H,V,st);
                if(cudaGetLastError()!=cudaSuccess||e!=cudaSuccess){printf("FAIL lm_head step=%d\n",step);exit(1);}
            }
            die(cudaStreamSynchronize(st),"sync_final");
            die(cudaMemcpy(h_logits.data(),d_logits,V*4,cudaMemcpyDeviceToHost),"logits_cpy");

            int next_id = sample(h_logits.data(), V, temperature, top_k);

            all_ids.push_back(next_id);
            std::string txt = tokenizer.decode(next_id);
            printf("%s", txt.c_str()); fflush(stdout);

            if (all_ids.size() - gen_start <= 3) {
                printf(" [tok#%d=%d]", (int)all_ids.size() - gen_start, next_id);
            }

            if(next_id == 151643 || next_id == 151645) {
                printf("\n[EOS at step %d]\n", step); break;
            }
        }
    }

    auto t_end = Clock::now();
    double ms = std::chrono::duration<double,std::milli>(t_end-t_start).count();
    int gen = (int)all_ids.size() - gen_start;

    printf("\n\n── Stats ──\n");
    printf("  Input: %d tokens, Generated: %d tokens\n", gen_start, gen);
    printf("  Time: %.1f ms, Speed: %.1f ms/tok = %.0f t/s\n",
           ms, ms/gen, 1000.0*gen/ms);

    return 0;
}
