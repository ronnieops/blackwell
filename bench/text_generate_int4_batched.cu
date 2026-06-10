// bench/text_generate_int4_batched.cu — Batched INT4 8B text generation
// Processes M sequences in parallel for higher throughput.
// M=1: ~56 t/s, M=4: ~100+ t/s, M=8: ~150+ t/s (projected)
//
// Build:
//   CUDACXX=/usr/local/cuda-13.3/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/text_generate_int4_batched.cu build/libblackwell_kernels.a \
//     -o bench/text_generate_int4_batched

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cstring>
#include <string>
#include <algorithm>
#include "blackwell/kernels.h"
#include "blackwell/bpe_tokenizer.h"
using blackwell::BpeTokenizer;

static void die(cudaError_t e, const char* m) {
    if(e!=cudaSuccess){printf("FAIL %s: %s\n",m,cudaGetErrorString(e));exit(1);}
}

const int H=4096, Q=4096, KV=1024, I=12288;
const int nqh=32, nkv=8, hd=128, MAXSEQ=512;  // Reduced for batched memory
const float eps=1e-6f;
const int V=151936;
const int NL=36;

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

__global__ void apply_rope_kernel(float* data, int n_heads, int head_dim, int pos) {
    int h=blockIdx.x; int d=threadIdx.x;
    if(h>=n_heads||d>=head_dim/2) return;
    float* pair=data+h*head_dim+d*2;
    float theta=(float)pos*powf(1000000.0f,-2.0f*(float)d/(float)head_dim);
    float c=cosf(theta),s=sinf(theta),x=pair[0],y=pair[1];
    pair[0]=x*c-y*s; pair[1]=x*s+y*c;
}

static void dequant_embed_row(float* out, int token, const uint8_t* host_w, const float* host_sc, int K) {
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

struct ServerState {
    int M;
    float *d_x32, *d_xi_f;  // [M][H]
    float *d_residual;      // [M][H] save pre-norm
    uint8_t *d_x_i4;        // [M][K/2]
    float *d_x_i4_sc;       // [M][K/16]
    float *d_Q, *d_K, *d_V; // [M][Q/KV]
    float *d_attn;          // [M][Q]
    uint8_t *d_attn_i4;    // [M][Q/2]
    float *d_attn_i4_sc;    // [M][Q/16]
    float *d_proj;          // [M][H]
    float *d_gate, *d_up;   // [M][I]
    uint8_t *d_mlp_i4;     // [M][I/2]
    float *d_mlp_i4_sc;    // [M][I/16]
    float *d_logits;        // [M][V]
    int *d_next_id;         // [M]
    int *d_recent;          // [M][64] recent tokens for rep_pen
    int *h_recent;          // [M][64] host-side recent tokens
    float *d_kc, *d_vc;     // KV cache [M][NL][ms][nkv][hd] - separate per sequence
    float *d_fn;            // final norm [H]
    float *d_fn_sc;         // final norm scales [H/16]
    cudaStream_t st;
};

static void alloc_buffers(ServerState& S, int M) {
    S.M = M;
    size_t kv_cache = (size_t)M * NL * nkv * MAXSEQ * hd * 4;
    
    #define AL(p,n){cudaError_t _e=cudaMalloc(&(p),(n));\
        if(_e!=cudaSuccess){printf("FAIL malloc %s: %s\n",#p,cudaGetErrorString(_e));exit(1);}}
    
    AL(S.d_x32, (size_t)M * H * 4);
    AL(S.d_xi_f, (size_t)M * H * 4);
    AL(S.d_residual, (size_t)M * H * 4);
    AL(S.d_x_i4, (size_t)M * H / 2);
    AL(S.d_x_i4_sc, (size_t)M * H / 16 * 4);
    AL(S.d_Q, (size_t)M * Q * 4);
    AL(S.d_K, (size_t)M * KV * 4);
    AL(S.d_V, (size_t)M * KV * 4);
    AL(S.d_attn, (size_t)M * Q * 4);
    AL(S.d_attn_i4, (size_t)M * Q / 2);
    AL(S.d_attn_i4_sc, (size_t)M * Q / 16 * 4);
    AL(S.d_proj, (size_t)M * H * 4);
    AL(S.d_gate, (size_t)M * I * 4);
    AL(S.d_up, (size_t)M * I * 4);
    AL(S.d_mlp_i4, (size_t)M * I / 2);
    AL(S.d_mlp_i4_sc, (size_t)M * I / 16 * 4);
    AL(S.d_logits, (size_t)M * V * 4);
    AL(S.d_next_id, M * 4);
    AL(S.d_recent, M * 64 * 4);  // 64 recent tokens per sequence
    S.h_recent = new int[M * 64];  // host-side buffer for recent tokens
    AL(S.d_kc, kv_cache);
    AL(S.d_vc, kv_cache);
    AL(S.d_fn, H * 4);
    AL(S.d_fn_sc, H / 16 * 4);
    #undef AL
    
    // Init scales
    float iv7 = 1.f/7.f;
    std::vector<float> tmp(H/16, iv7);
    for (int m = 0; m < M; ++m) {
        cudaMemcpy(S.d_x_i4_sc + m * (H/16), tmp.data(), (H/16)*4, cudaMemcpyHostToDevice);
        cudaMemcpy(S.d_attn_i4_sc + m * (Q/16), tmp.data(), (Q/16)*4, cudaMemcpyHostToDevice);
        cudaMemcpy(S.d_mlp_i4_sc + m * (I/16), tmp.data(), (I/16)*4, cudaMemcpyHostToDevice);
    }
    cudaMemcpy(S.d_fn_sc, tmp.data(), (H/16)*4, cudaMemcpyHostToDevice);
    
    int dummy = 0;
    for (int m = 0; m < M; ++m) {
        cudaMemcpy(S.d_next_id + m, &dummy, 4, cudaMemcpyHostToDevice);
    }
    
    cudaStreamCreate(&S.st);
}

static void free_buffers(ServerState& S) {
    #define FR(p) if(p){cudaFree(p);p=nullptr;}
    FR(S.d_x32); FR(S.d_xi_f); FR(S.d_residual);
    FR(S.d_x_i4); FR(S.d_x_i4_sc);
    FR(S.d_Q); FR(S.d_K); FR(S.d_V);
    FR(S.d_attn); FR(S.d_attn_i4); FR(S.d_attn_i4_sc);
    FR(S.d_proj); FR(S.d_gate); FR(S.d_up);
    FR(S.d_mlp_i4); FR(S.d_mlp_i4_sc);
    FR(S.d_logits); FR(S.d_next_id);
    FR(S.d_recent);
    if (S.h_recent) { delete[] S.h_recent; S.h_recent = nullptr; }
    FR(S.d_kc); FR(S.d_vc);
    FR(S.d_fn); FR(S.d_fn_sc);
    #undef FR
}

static void generate_batch(
    ServerState& S,
    const std::vector<std::vector<uint32_t>>& prompts,
    int max_new,
    float temperature,
    int top_k,
    float rep_pen,
    const std::vector<LW4>& W,
    DevW4& lm_head_w,
    const uint8_t* host_embed_d,
    const float* host_embed_sc,
    BpeTokenizer& tokenizer)
{
    int M = prompts.size();
    std::vector<std::vector<uint32_t>> all_ids(M);
    std::vector<int> gen_start(M), total(M);
    std::vector<int> seq_pos(M, 0);
    
    for (int m = 0; m < M; ++m) {
        all_ids[m] = prompts[m];
        gen_start[m] = prompts[m].size();
        total[m] = gen_start[m] + max_new;
    }
    
    // Clear KV cache (all M sequences)
    cudaMemset(S.d_kc, 0, (size_t)M * NL * nkv * MAXSEQ * hd * 4);
    cudaMemset(S.d_vc, 0, (size_t)M * NL * nkv * MAXSEQ * hd * 4);
    
    auto t_start = std::chrono::high_resolution_clock::now();
    
    // Decode loop - process all sequences step by step
    int max_steps = max_new + *std::max_element(gen_start.begin(), gen_start.end());
    
    for (int step = 0; step < max_steps; ++step) {
        // Embed all sequences
        std::vector<float> h_embed(M * H);
        for (int m = 0; m < M; ++m) {
            uint32_t tid = (step < gen_start[m]) ? prompts[m][step] : all_ids[m].back();
            dequant_embed_row(h_embed.data() + m * H, tid, host_embed_d, host_embed_sc, H);
        }
        
        for (int m = 0; m < M; ++m) {
            cudaMemcpyAsync(S.d_x32 + m * H, h_embed.data() + m * H, H * 4, 
                           cudaMemcpyHostToDevice, S.st);
        }
        
        // 36-layer decode (batched kernels where possible)
        for (int l = 0; l < NL; ++l) {
            // Save residual for all sequences (single large copy)
            cudaMemcpyAsync(S.d_residual, S.d_x32, (size_t)M * H * 4,
                           cudaMemcpyDeviceToDevice, S.st);
            
            // Pre-attention norm + quantize (batched)
            blackwell::kernels::fused_rmsnorm_batched(
                S.d_xi_f, S.d_x32, W[l].rn_in, H, eps, M, S.st);
            blackwell::kernels::quantize_int4_batched(
                S.d_x_i4, S.d_x_i4_sc, S.d_xi_f, H, M, S.st);
            
            // QKV projections (single batched GEMV each)
            blackwell::kernels::gemv_int4_batched(
                S.d_Q, S.d_x_i4, S.d_x_i4_sc,
                W[l].q.d, W[l].q.sc, H, Q, M, S.st);
            blackwell::kernels::gemv_int4_batched(
                S.d_K, S.d_x_i4, S.d_x_i4_sc,
                W[l].k.d, W[l].k.sc, H, KV, M, S.st);
            blackwell::kernels::gemv_int4_batched(
                S.d_V, S.d_x_i4, S.d_x_i4_sc,
                W[l].v.d, W[l].v.sc, H, KV, M, S.st);
            
            // Q/K head norms + RoPE (per-sequence — cheap ops, different rope_pos per seq)
            for (int m = 0; m < M; ++m) {
                int rope_pos = (step >= gen_start[m] - 1) ? gen_start[m] - 1 + seq_pos[m] : step;
                head_norm_kernel<<<nqh, 128, 0, S.st>>>(S.d_Q + m * Q, W[l].qn, nqh, hd, eps);
                head_norm_kernel<<<nkv, 128, 0, S.st>>>(S.d_K + m * KV, W[l].kn, nkv, hd, eps);
                apply_rope_kernel<<<nqh, hd/2, 0, S.st>>>(S.d_Q + m * Q, nqh, hd, rope_pos);
                apply_rope_kernel<<<nkv, hd/2, 0, S.st>>>(S.d_K + m * KV, nkv, hd, rope_pos);
            }
            
            // KV cache update (per-sequence — kernel only supports batch_idx=0)
            size_t l_kv_off = (size_t)l * nkv * MAXSEQ * hd;
            for (int m = 0; m < M; ++m) {
                size_t kv_off = (size_t)m * NL * nkv * MAXSEQ * hd + l_kv_off;
                blackwell::kernels::update_kv_cache(
                    S.d_kc + kv_off, S.d_vc + kv_off,
                    S.d_K + m * KV, S.d_V + m * KV,
                    0, step, nkv, hd, MAXSEQ, S.st);
                }
            
            // Attention — per-sequence (batched M>2 non-deterministic)
            for (int m = 0; m < M; ++m) {
                size_t m_kv_off = (size_t)m * NL * nkv * MAXSEQ * hd;
                blackwell::kernels::attention_decode_batched_gqa(
                    S.d_attn + m * Q, S.d_Q + m * Q,
                    S.d_kc + m_kv_off, S.d_vc + m_kv_off,
                    step, nqh, nkv, hd, MAXSEQ, 1,
                    (size_t)nkv * MAXSEQ * hd, l_kv_off, S.st);
            }
            
            // Wo projection (batched)
            blackwell::kernels::quantize_int4_batched(
                S.d_attn_i4, S.d_attn_i4_sc, S.d_attn, Q, M, S.st);
            blackwell::kernels::gemv_int4_batched(
                S.d_proj, S.d_attn_i4, S.d_attn_i4_sc,
                W[l].o.d, W[l].o.sc, Q, H, M, S.st);
            // Residual add (per-sequence — no batched vector_add)
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::vector_add_fp32(
                    S.d_x32 + m * H, S.d_proj + m * H, S.d_residual + m * H, H, S.st);
            }
            cudaMemcpyAsync(S.d_residual, S.d_x32, (size_t)M * H * 4,
                           cudaMemcpyDeviceToDevice, S.st);
            
            // Pre-MLP norm + quantize (batched)
            blackwell::kernels::fused_rmsnorm_batched(
                S.d_xi_f, S.d_x32, W[l].rn_post, H, eps, M, S.st);
            blackwell::kernels::quantize_int4_batched(
                S.d_x_i4, S.d_x_i4_sc, S.d_xi_f, H, M, S.st);
            
            // MLP gate + up (batched)
            blackwell::kernels::gemv_int4_batched(
                S.d_gate, S.d_x_i4, S.d_x_i4_sc,
                W[l].g.d, W[l].g.sc, H, I, M, S.st);
            blackwell::kernels::gemv_int4_batched(
                S.d_up, S.d_x_i4, S.d_x_i4_sc,
                W[l].u.d, W[l].u.sc, H, I, M, S.st);
            
            // SwiGLU (per-sequence — no batched version)
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::apply_swiglu(S.d_gate + m * I, S.d_gate + m * I, S.d_up + m * I, I, S.st);
            }
            // Quantize MLP output (batched)
            blackwell::kernels::quantize_int4_batched(
                S.d_mlp_i4, S.d_mlp_i4_sc, S.d_gate, I, M, S.st);
            
            // Down projection (batched)
            blackwell::kernels::gemv_int4_batched(
                S.d_proj, S.d_mlp_i4, S.d_mlp_i4_sc,
                W[l].d.d, W[l].d.sc, I, H, M, S.st);
            // Final residual add (per-sequence)
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::vector_add_fp32(
                    S.d_x32 + m * H, S.d_proj + m * H, S.d_residual + m * H, H, S.st);
            }
        }
        
        // Final norm + lm_head + sampling when any sequence is generating
        {
            int min_gen = *std::min_element(gen_start.begin(), gen_start.end());
            if (step >= min_gen - 1) {
                // Batched final norm + quantize + lm_head
                blackwell::kernels::fused_rmsnorm_batched(
                    S.d_xi_f, S.d_x32, S.d_fn, H, eps, M, S.st);
                blackwell::kernels::quantize_int4_batched(
                    S.d_x_i4, S.d_x_i4_sc, S.d_xi_f, H, M, S.st);
                blackwell::kernels::gemv_int4_batched(
                    S.d_logits, S.d_x_i4, S.d_x_i4_sc,
                    lm_head_w.d, lm_head_w.sc, H, V, M, S.st);
                // Per-sequence sampling (only for sequences at generation stage)
                for (int m = 0; m < M; ++m) {
                    if (step >= gen_start[m] - 1) {
                        // Repetition penalty
                        if (rep_pen > 1.0f) {
                            int num_recent = (int)all_ids[m].size() - gen_start[m];
                            if (num_recent > 0) {
                                if (num_recent > 64) num_recent = 64;
                                std::vector<int> h_recent(all_ids[m].end() - num_recent, all_ids[m].end());
                                cudaMemcpyAsync(S.d_recent + m * 64, h_recent.data(), num_recent * 4, cudaMemcpyHostToDevice, S.st);
                                blackwell::kernels::apply_repetition_penalty(
                                    S.d_logits + m * V, S.d_recent + m * 64, num_recent, rep_pen, V, S.st);
                            }
                        }
                        blackwell::kernels::sample_gpu(
                            S.d_logits + m * V, V, temperature, top_k,
                            S.d_next_id + m, 0xdeadbeefLL, step, S.st);
                    }
                }
            }
        }
        
        cudaStreamSynchronize(S.st);
        
        // Copy results and update
        std::vector<int> next_ids(M, 0);
        for (int m = 0; m < M; ++m) {
            if (step >= gen_start[m] - 1) {
                cudaMemcpy(&next_ids[m], S.d_next_id + m, 4, cudaMemcpyDeviceToHost);
                all_ids[m].push_back(next_ids[m]);

            }
        }
        
        // Update seq_pos
        for (int m = 0; m < M; ++m) {
            if (step >= gen_start[m] - 1) {
                seq_pos[m]++;
            }
        }
        
        // Check for EOS (only for sequences that have started generating)
        bool all_done = true;
        for (int m = 0; m < M; ++m) {
            if (step >= gen_start[m] - 1) {
                if (next_ids[m] != 151643 && next_ids[m] != 151645) {
                    all_done = false;
                }
            } else {
                // Sequence hasn't started generating yet, not done
                all_done = false;
            }
        }
        if (all_done) break;
    }
    
    auto t_end = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    
    // Output
    for (int m = 0; m < M; ++m) {
        printf("[Prompt %d] ", m + 1);
        std::vector<uint32_t> gen_tokens(all_ids[m].begin() + gen_start[m], all_ids[m].end());
        std::string txt = tokenizer.decode(gen_tokens);
        printf("%s\n", txt.c_str());
    }
    
    int total_gen = 0;
    for (int m = 0; m < M; ++m) {
        total_gen += all_ids[m].size() - gen_start[m];
    }
    printf("\nStats: %d tokens, %.1f ms, %.1f ms/tok = %.0f t/s\n",
           total_gen, ms, ms / total_gen, 1000.0 * total_gen / ms);
}

int main(int argc, char** argv) {
    const char* prompt = "The capital of France is";
    int M = 4;
    int max_new = 20;
    
    if (argc >= 2) prompt = argv[1];
    if (argc >= 3) M = atoi(argv[2]);
    if (argc >= 4) max_new = atoi(argv[3]);
    const char* wdir = "weights_int4_qwen3_8b";
    if (argc >= 5) wdir = argv[4];
    
    printf("# Batched INT4 8B — %d sequences\n", M);
    printf("  Weights: %s\n", wdir);
    printf("  Prompt: \"%s\"\n", prompt);
    
    // Reset CUDA device to ensure clean state
    cudaDeviceReset();
    printf("  Max new: %d\n\n", max_new);
    
    BpeTokenizer tokenizer;
    if (tokenizer.load("tokenizer_data.bin") != 0) {
        printf("FAIL: no tokenizer_data.bin\n"); return 1;
    }
    
    // Tokenize same prompt M times
    std::vector<std::vector<uint32_t>> prompts(M);
    auto input_ids = tokenizer.encode(prompt);
    for (int m = 0; m < M; ++m) {
        prompts[m] = input_ids;
    }
    
    // Load model weights
    std::vector<LW4> W(NL);
    
    // Load Q/K head norms from combined qk_norms.f32 (once for all layers)
    float* qk_h = (float*)malloc(NL * 2 * hd * 4);
    char wqk[256]; snprintf(wqk, 256, "%s/qk_norms.f32", wdir);
    { FILE* f2 = fopen(wqk, "rb"); fread(qk_h, 4, NL * 2 * hd, f2); fclose(f2); }
    
    for (int l = 0; l < NL; ++l) {
        char p[512];
        snprintf(p, 512, "%s/%d_input_layernorm.f32", wdir, l);
        float* w = (float*)malloc(H * 4);
        FILE* f = fopen(p, "rb"); fread(w, 4, H, f); fclose(f);
        cudaMalloc(&W[l].rn_in, H * 4); cudaMemcpy(W[l].rn_in, w, H * 4, cudaMemcpyHostToDevice); free(w);
        
        snprintf(p, 512, "%s/%d_post_attention_layernorm.f32", wdir, l);
        w = (float*)malloc(H * 4);
        f = fopen(p, "rb"); fread(w, 4, H, f); fclose(f);
        cudaMalloc(&W[l].rn_post, H * 4); cudaMemcpy(W[l].rn_post, w, H * 4, cudaMemcpyHostToDevice); free(w);
        
        snprintf(p, 512, "%s/%d_self_attn.q_proj", wdir, l);
        W[l].q = upload_w4(p);
        snprintf(p, 512, "%s/%d_self_attn.k_proj", wdir, l);
        W[l].k = upload_w4(p);
        snprintf(p, 512, "%s/%d_self_attn.v_proj", wdir, l);
        W[l].v = upload_w4(p);
        snprintf(p, 512, "%s/%d_self_attn.o_proj", wdir, l);
        W[l].o = upload_w4(p);
        snprintf(p, 512, "%s/%d_mlp.gate_proj", wdir, l);
        W[l].g = upload_w4(p);
        snprintf(p, 512, "%s/%d_mlp.up_proj", wdir, l);
        W[l].u = upload_w4(p);
        snprintf(p, 512, "%s/%d_mlp.down_proj", wdir, l);
        W[l].d = upload_w4(p);
        
        // Q/K head norms from qk_norms.f32: [l][2][hd] layout (same as single benchmark)
        // qn at offset l*2*hd, kn at offset l*2*hd + hd
        cudaMalloc(&W[l].qn, hd * 4); cudaMemcpy(W[l].qn, qk_h + l * 2 * hd, hd * 4, cudaMemcpyHostToDevice);
        cudaMalloc(&W[l].kn, hd * 4); cudaMemcpy(W[l].kn, qk_h + l * 2 * hd + hd, hd * 4, cudaMemcpyHostToDevice);
        
        if ((l + 1) % 9 == 0) printf("  layer %d/%d\n", l + 1, NL);
    }
    free(qk_h);  // Free qk_norms buffer
    
    char pw[512];
    DevW4 embed = upload_w4((std::string(wdir) + "/embed_tokens").c_str());
    uint8_t* host_embed_d = new uint8_t[(size_t)embed.K * embed.N / 2];
    float* host_embed_sc = new float[(size_t)embed.N * (embed.K / 16)];
    {
        snprintf(pw, 512, "%s/embed_tokens.int4_t", wdir);
        FILE* f = fopen(pw, "rb"); int h[5]; fread(h, 4, 5, f);
        size_t ds = (size_t)h[0] * h[1] / 2; fread(host_embed_d, 1, ds, f); fclose(f);
        snprintf(pw, 512, "%s/embed_tokens.scale_t", wdir);
        f = fopen(pw, "rb"); fread(h, 4, 5, f); size_t ss = (size_t)h[3] * h[4];
        fread(host_embed_sc, 4, ss, f); fclose(f);
    }
    printf("Embed loaded: %d x %d (INT4)\n", embed.K, embed.N);
    
    DevW4 lm_head_w = upload_w4((std::string(wdir) + "/lm_head").c_str());
    // Check lm_head weights and scales
    float* h_lm_sc = new float[16];
    cudaMemcpy(h_lm_sc, lm_head_w.sc, 64, cudaMemcpyDeviceToHost);
    printf("lm_head loaded: %d x %d (INT4), sc[0-3]=[%.4f, %.4f, %.4f, %.4f]\n", 
           lm_head_w.K, lm_head_w.N, h_lm_sc[0], h_lm_sc[1], h_lm_sc[2], h_lm_sc[3]);
    delete[] h_lm_sc;
    
    float* w = (float*)malloc(H * 4);
    snprintf(pw, 512, "%s/final_norm.f32", wdir);
    FILE* f = fopen(pw, "rb");
    fread(w, 4, H, f); fclose(f);
    float* d_fn; cudaMalloc(&d_fn, H * 4); cudaMemcpy(d_fn, w, H * 4, cudaMemcpyHostToDevice);
    float* d_fn_sc; cudaMalloc(&d_fn_sc, H / 16 * 4);
    free(w);
    
    printf("\nAll weights loaded.\n\n");
    
    ServerState S;
    alloc_buffers(S, M);
    
    // Copy final norm weights to server state
    cudaMemcpy(S.d_fn, d_fn, H * 4, cudaMemcpyDeviceToDevice);
    
    float rep_pen = 1.3f;  // Repetition penalty (1.0=disabled, 1.3=moderate)
    generate_batch(S, prompts, max_new, 0.0f, 0, rep_pen, W, lm_head_w, host_embed_d, host_embed_sc, tokenizer);
    
    free_buffers(S);
    return 0;
}
