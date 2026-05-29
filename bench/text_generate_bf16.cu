// bench/text_generate_bf16.cu — BF16 text generation (no quantization error)
//
// Loads BF16 weights, uses FP32×BF16 GEMV kernel, produces correct text.
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/text_generate_bf16.cu build/libblackwell_kernels.a \
//     -o bench/text_generate_bf16
//
// Run: ./bench/text_generate_bf16 "Hello world" 20

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cstring>
#include "blackwell/kernels.h"
#include "blackwell/bpe_tokenizer.h"

static void die(cudaError_t e, const char* m) {
    if(e!=cudaSuccess){printf("FAIL %s: %s\n",m,cudaGetErrorString(e));exit(1);}
}

// BF16 GEMV kernel: FP32 activations × BF16 weights → FP32 output
// Weight layout: W [N_out × K_in] BF16 row-major (NOT transposed)
constexpr int kBF16Blk = 256;

__launch_bounds__(kBF16Blk, 1)
__global__ void gemv_bf16_fp32_kernel(
    float* __restrict__ y_out,
    const float* __restrict__ x_fp32,
    const __nv_bfloat16* __restrict__ W,  // [N_out × K_in]
    int K, int N)
{
    int n_out = blockIdx.x * kBF16Blk + threadIdx.x;
    if (n_out >= N) return;

    const __nv_bfloat16* w_row = &W[(size_t)n_out * K];
    float acc = 0.0f;

    // Process pairs of BF16 values
    int num_pairs = K / 2;
    for (int p = 0; p < num_pairs; ++p) {
        __nv_bfloat162 w_pair = reinterpret_cast<const __nv_bfloat162*>(w_row)[p];
        float2 x_pair = reinterpret_cast<const float2*>(x_fp32)[p];
        acc += __bfloat162float(w_pair.x) * x_pair.x;
        acc += __bfloat162float(w_pair.y) * x_pair.y;
    }
    // Handle odd K
    if (K & 1) {
        acc += __bfloat162float(w_row[K-1]) * x_fp32[K-1];
    }

    y_out[n_out] = acc;
}

// Weight structures
struct LW16 { int N, K; std::vector<__nv_bfloat16> data; };
struct DW16 { int N, K; __nv_bfloat16* d; };

static LW16 load_bf16(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("FAIL open %s\n", path); exit(1); }
    int h[2]; fread(h, 4, 2, f);
    LW16 w; w.N = h[0]; w.K = h[1];
    w.data.resize((size_t)h[0] * h[1]);
    fread(w.data.data(), 2, w.data.size(), f);
    fclose(f);
    return w;
}

static DW16 upload_bf16(const LW16& w) {
    DW16 d; d.N = w.N; d.K = w.K;
    cudaMalloc(&d.d, (size_t)w.N * w.K * 2);
    cudaMemcpy(d.d, w.data.data(), (size_t)w.N * w.K * 2, cudaMemcpyHostToDevice);
    return d;
}

// FP32 weight loader (for norms)
static float* load_f32(const char* path, int expected) {
    float* d; cudaMalloc(&d, expected * 4);
    float* h = (float*)malloc(expected * 4);
    FILE* f = fopen(path, "rb");
    if (!f) { printf("FAIL open %s\n", path); exit(1); }
    fread(h, 4, expected, f); fclose(f);
    cudaMemcpy(d, h, expected * 4, cudaMemcpyHostToDevice);
    free(h); return d;
}

// Kernels from text_generate (reused)
__global__ void head_norm_kernel(float* data, const float* weight, int nh, int hd, float eps) {
    int h=blockIdx.x; if(h>=nh) return;
    float* d=data+h*hd;
    __shared__ float wp[4];
    float s=0; int tid=threadIdx.x;
    for(int i=tid;i<hd;i+=blockDim.x) s+=d[i]*d[i];
    for(int off=16;off>0;off>>=1) s+=__shfl_xor_sync(0xffffffff,s,off);
    if((tid&31)==0) wp[tid>>5]=s; __syncthreads();
    if(tid<4) s=wp[tid]; else s=0;
    for(int off=2;off>0;off>>=1) s+=__shfl_xor_sync(0xffffffff,s,off);
    if(tid==0) wp[0]=rsqrtf(s/hd+eps); __syncthreads();
    float is=wp[0];
    for(int i=tid;i<hd;i+=blockDim.x) d[i]=d[i]*is*weight[i];
}

__global__ void apply_rope_kernel(float* data, int n_heads, int head_dim, int pos) {
    int h = blockIdx.x; int d = threadIdx.x;
    if (h >= n_heads || d >= head_dim/2) return;
    int i2 = d * 2; float* pair = data + h * head_dim + i2;
    float idxf = (float)d / (float)head_dim;
    const float rope_theta = 1000000.0f;
    float theta = (float)pos * powf(rope_theta, -2.0f * idxf);
    float c = cosf(theta), s = sinf(theta);
    float x = pair[0], y = pair[1];
    pair[0] = x * c - y * s; pair[1] = x * s + y * c;
}

// Model constants
const int H=2048, QD=2048, KV=1024, ID=6144;
const int nqh=16, nkv=8, hd=128, MAXSEQ=2048, NL=28;
const float eps=1e-6f; const int V=151936;

struct L { DW16 q,k,v,o,g,u,d; float* qn; float* kn; };

using Clock = std::chrono::high_resolution_clock;

int main(int argc, char** argv) {
    const char* prompt = "Once upon a time";
    int max_new = 50;
    if (argc > 1) prompt = argv[1];
    if (argc > 2) max_new = atoi(argv[2]);

    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    printf("# BF16 Text Generation — Qwen3-1.7B (NO quantization)\n");
    printf("  Device: %s\n  Prompt: \"%s\"\n  Max tokens: %d\n\n", P.name, prompt, max_new);

    blackwell::BpeTokenizer tokenizer;
    if (tokenizer.load("tokenizer_data.bin") != 0) { fprintf(stderr, "No tokenizer\n"); return 1; }
    auto input_ids = tokenizer.encode(prompt);
    printf("Input: %zu tokens\n\n", input_ids.size());

    // Load BF16 weights
    printf("Loading BF16 weights...\n");
    std::vector<L> W(NL); char p[256];
    for (int l = 0; l < NL; l++) {
        snprintf(p,256,"weights_bf16/%d_self_attn.q_proj.bf16",l); W[l].q = upload_bf16(load_bf16(p));
        snprintf(p,256,"weights_bf16/%d_self_attn.k_proj.bf16",l); W[l].k = upload_bf16(load_bf16(p));
        snprintf(p,256,"weights_bf16/%d_self_attn.v_proj.bf16",l); W[l].v = upload_bf16(load_bf16(p));
        snprintf(p,256,"weights_bf16/%d_self_attn.o_proj.bf16",l); W[l].o = upload_bf16(load_bf16(p));
        snprintf(p,256,"weights_bf16/%d_mlp.gate_proj.bf16",l);   W[l].g = upload_bf16(load_bf16(p));
        snprintf(p,256,"weights_bf16/%d_mlp.up_proj.bf16",l);     W[l].u = upload_bf16(load_bf16(p));
        snprintf(p,256,"weights_bf16/%d_mlp.down_proj.bf16",l);  W[l].d = upload_bf16(load_bf16(p));
        if ((l+1)%7==0||l+1==NL) printf("  layer %d/28\n", l+1);
    }
    DW16 emb = upload_bf16(load_bf16("weights_bf16/embed_tokens.bf16"));

    // RMSNorm weights
    std::vector<float*> d_rn_in(NL), d_rn_post(NL);
    for (int l = 0; l < NL; l++) {
        snprintf(p,256,"weights_bf16/%d_input_layernorm.f32",l);
        d_rn_in[l] = load_f32(p, H);
        snprintf(p,256,"weights_bf16/%d_post_attention_layernorm.f32",l);
        d_rn_post[l] = load_f32(p, H);
    }
    float *d_fn = load_f32("weights_bf16/final_norm.f32", H);

    // Q/K norms
    float* d_qk_norms = load_f32("weights_bf16/qk_norms.f32", NL*2*hd);

    // Allocate GPU buffers
    float *d_x, *d_Q, *d_K, *d_V, *d_attn, *d_gate, *d_up, *d_mlp, *d_proj;
    float *d_res1, *d_res2, *d_kc, *d_vc, *d_logits;
    #define A(p,n) cudaMalloc(&(p),(n))
    A(d_x,H*4); A(d_Q,QD*4); A(d_K,KV*4); A(d_V,KV*4); A(d_attn,QD*4);
    A(d_gate,ID*4); A(d_up,ID*4); A(d_mlp,ID*4); A(d_proj,H*4);
    A(d_res1,H*4); A(d_res2,H*4);
    A(d_kc,(size_t)NL*nkv*MAXSEQ*hd*4); A(d_vc,(size_t)NL*nkv*MAXSEQ*hd*4);
    A(d_logits,V*4);
    #undef A
    cudaMemset(d_kc,0,(size_t)NL*nkv*MAXSEQ*hd*4);
    cudaMemset(d_vc,0,(size_t)NL*nkv*MAXSEQ*hd*4);

    printf("All loaded.\n\n");

    // Helper: BF16 GEMV
    auto bf16_gemv = [&](float* y, float* x, __nv_bfloat16* W, int K, int N, cudaStream_t s) {
        int nb = (N + kBF16Blk - 1) / kBF16Blk;
        gemv_bf16_fp32_kernel<<<nb, kBF16Blk, 0, s>>>(y, x, W, K, N);
    };

    cudaStream_t st; cudaStreamCreate(&st);
    std::vector<float> h_embed(H);
    std::vector<float> h_logits(V);

    printf("── Generating ──\n%s", prompt); fflush(stdout);

    std::vector<uint32_t> all_ids = input_ids;
    int gen_start = (int)input_ids.size();
    int total = gen_start + max_new;

    auto t_start = Clock::now();

    for (int step = 0; step < total; step++) {
        uint32_t tid = (step < gen_start) ? input_ids[step] : all_ids.back();

        // Embedding lookup: row tid from BF16 embed_tokens
        die(cudaMemcpy(d_x, &emb.d[(size_t)tid * H], H * 2, cudaMemcpyDeviceToDevice), "embed");
        // Convert BF16 → FP32 for d_x (we need FP32 for RMSNorm)
        // Actually, let's do embedding dequant on host for simplicity
        // No — better to do it on device. Let me use a simple kernel.

        // Simple BF16→FP32 conversion for embedding row
        // Actually, gemv_bf16_fp32_kernel expects FP32 input. For embedding,
        // we need to dequantize the BF16 row to FP32 first.
        // Quick solution: use a small conversion kernel.
        // For now, let me do it the simplest way: host-side conversion.
        // TODO: optimize with device kernel.
        static thread_local std::vector<uint16_t> bf16_buf(H);
        die(cudaMemcpy(bf16_buf.data(), &emb.d[(size_t)tid * H], H * 2, cudaMemcpyDeviceToHost), "embed_h");
        for (int i = 0; i < H; i++) {
            uint32_t u32 = (uint32_t)bf16_buf[i] << 16;
            memcpy(&h_embed[i], &u32, 4);
        }
        die(cudaMemcpy(d_x, h_embed.data(), H * 4, cudaMemcpyHostToDevice), "embed_cpy");

        for (int l = 0; l < NL; l++) {
            float* input = (l == 0) ? d_x : d_proj;
            die(cudaMemcpyAsync(d_res1, input, H*4, cudaMemcpyDeviceToDevice, st), "save_res");

            // 1. Input RMSNorm
            die(blackwell::kernels::fused_rmsnorm(d_proj, input, d_rn_in[l], H, eps, st), "rmsnorm_in");

            // 2. QKV GEMVs (BF16 weights)
            bf16_gemv(d_Q, d_proj, W[l].q.d, W[l].q.K, W[l].q.N, st);
            bf16_gemv(d_K, d_proj, W[l].k.d, W[l].k.K, W[l].k.N, st);
            bf16_gemv(d_V, d_proj, W[l].v.d, W[l].v.K, W[l].v.N, st);

            // 3. Q/K head norms
            head_norm_kernel<<<nqh,128,0,st>>>(d_Q, d_qk_norms + l*2*hd, nqh, hd, eps);
            head_norm_kernel<<<nkv,128,0,st>>>(d_K, d_qk_norms + l*2*hd + hd, nkv, hd, eps);

            // 4. RoPE
            apply_rope_kernel<<<nqh,hd/2,0,st>>>(d_Q, nqh, hd, step);
            apply_rope_kernel<<<nkv,hd/2,0,st>>>(d_K, nkv, hd, step);

            // 5. KV cache + attention
            int kb = l * nkv * MAXSEQ * hd;
            die(blackwell::kernels::update_kv_cache(d_kc+kb, d_vc+kb, d_K, d_V, 0, step, nkv, hd, MAXSEQ, st), "kv");
            die(blackwell::kernels::attention_decode_gqa(d_attn, d_Q, d_kc+kb, d_vc+kb, step, nqh, nkv, hd, MAXSEQ, st), "attn");

            // 6. Wo GEMV + residual
            bf16_gemv(d_proj, d_attn, W[l].o.d, W[l].o.K, W[l].o.N, st);
            die(blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_res1, H, st), "res1");

            die(cudaMemcpyAsync(d_res2, d_proj, H*4, cudaMemcpyDeviceToDevice, st), "save_res2");

            // 7. Post-attention RMSNorm (reuse d_attn as temp buffer)
            die(blackwell::kernels::fused_rmsnorm(d_attn, d_proj, d_rn_post[l], H, eps, st), "rmsnorm_post");

            // 8. Gate + Up GEMVs + SwiGLU
            bf16_gemv(d_gate, d_attn, W[l].g.d, W[l].g.K, W[l].g.N, st);
            bf16_gemv(d_up, d_attn, W[l].u.d, W[l].u.K, W[l].u.N, st);
            die(blackwell::kernels::apply_swiglu(d_mlp, d_gate, d_up, ID, st), "swiglu");

            // 9. Down GEMV + residual
            bf16_gemv(d_proj, d_mlp, W[l].d.d, W[l].d.K, W[l].d.N, st);
            die(blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_res2, H, st), "res2");
        }

        // Final norm + lm_head
        if (step >= gen_start - 1) {
            die(blackwell::kernels::fused_rmsnorm(d_attn, d_proj, d_fn, H, eps, st), "fn");
            bf16_gemv(d_logits, d_attn, emb.d, emb.K, emb.N, st);
            die(cudaStreamSynchronize(st), "sync");
            die(cudaMemcpy(h_logits.data(), d_logits, V*4, cudaMemcpyDeviceToHost), "logits");

            int next = 0; float bv = h_logits[0];
            for (int i = 1; i < V; i++) if (h_logits[i] > bv) { bv = h_logits[i]; next = i; }
            all_ids.push_back(next);
            printf("%s", tokenizer.decode(next).c_str()); fflush(stdout);
            if (next == 151643 || next == 151645) { printf("\n[EOS]\n"); break; }
        }
    }

    auto t_end = Clock::now();
    double ms = std::chrono::duration<double,std::milli>(t_end-t_start).count();
    int gen = (int)all_ids.size() - gen_start;
    printf("\n\n── Stats ──\n  Time: %.1f ms, Speed: %.1f ms/tok = %.0f t/s\n", ms, ms/gen, 1000.0*gen/ms);

    return 0;
}
