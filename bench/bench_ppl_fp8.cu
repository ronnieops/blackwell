// bench_ppl_fp8.cu — FP8 E4M3 Perplexity benchmark for Blackwell
//
// Quality validation: FP8 weights + per-block scales → FP32 dequant → FP32 GEMV
// No activation quantization — pure FP32 compute path.
//
// Usage: ./bench/bench_ppl_fp8 <num_tokens>
//   Loads from weights_fp8_bf16/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include "blackwell/kernels.h"
#include "blackwell/bpe_tokenizer.h"

#define AL(e) do{cudaError_t _e=(e);if(_e!=cudaSuccess){\
    fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));exit(1);}}while(0)

// ── Test corpus (same as bench_ppl.cu) ────────────────────────────────
static const char* TEST_TEXT =
    "The Republic of Austria is a federal republic in Central Europe . "
    "It is bordered by Germany to the northwest , the Czech Republic to the north , "
    "Slovakia to the northeast , Hungary to the east , Slovenia and Italy to the south , "
    "Switzerland and Liechtenstein to the west . "
    "The capital of Austria is Vienna . "
    "The official language is German .";

// ── FP8 E4M3 dequantization (CPU) ────────────────────────────────────
static void fp8_e4m3_to_fp32(const uint8_t* fp8, float* fp32, int n) {
    for (int i = 0; i < n; i++) {
        uint8_t b = fp8[i];
        int sign = (b >> 7) & 1;
        int exp = (b >> 3) & 0xF;
        int mant = b & 0x7;
        float value;
        if (exp > 0) {
            value = (1.0f + (float)mant / 8.0f) * powf(2.0f, (float)(exp - 7));
        } else {
            value = ((float)mant / 8.0f) * powf(2.0f, -6.0f);
        }
        fp32[i] = sign ? -value : value;
    }
}

// ── Apply per-row scales: dequant_fp8[row] * scale[row] ─────────────
static void apply_row_scales(float* fp32_weights, const float* scales,
                              int N, int K) {
    for (int row = 0; row < N; row++) {
        float sc = scales[row];
        for (int j = 0; j < K; j++) {
            fp32_weights[row * K + j] *= sc;
        }
    }
}

// ── Logprob kernel ────────────────────────────────────────────────────
__global__ void logprob_kernel(const float* logits, int V, int correct_id, float* out) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int n = blockDim.x;

    float mx = -INFINITY;
    for (int i = tid; i < V; i += n) { float v = logits[i]; if (v > mx) mx = v; }
    smem[tid] = mx; __syncthreads();
    for (int o = n/2; o > 0; o >>= 1) { if (tid < o && smem[tid+o] > smem[tid]) smem[tid] = smem[tid+o]; __syncthreads(); }
    float maxv = smem[0]; __syncthreads();

    float sum_exp = 0.0f;
    for (int i = tid; i < V; i += n) sum_exp += expf(logits[i] - maxv);
    smem[tid] = sum_exp; __syncthreads();
    for (int o = n/2; o > 0; o >>= 1) { if (tid < o) smem[tid] += smem[tid+o]; __syncthreads(); }

    float logp = (correct_id >= 0 && correct_id < V) ? logits[correct_id] - (maxv + logf(smem[0])) : 0.0f;
    if (tid == 0) out[0] = logp;
}

// ── FP32 GEMV kernel (quality validation, not speed-optimized) ────────
__global__ void gemv_fp32_kernel(float* y, const float* x, const float* W,
                                  int K, int N) {
    __shared__ float smem[4];
    int row = blockIdx.x;
    if (row >= N) return;

    const float* w_row = W + (size_t)row * K;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        sum += w_row[i] * x[i];
    }

    for (int o = 16; o > 0; o >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, o);

    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) smem[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < 4) ? smem[lane] : 0.0f;
        for (int o = 2; o > 0; o >>= 1)
            sum += __shfl_xor_sync(0xffffffff, sum, o);
        if (lane == 0) y[row] = sum;
    }
}

static void gemv_fp32(float* d_y, const float* d_x, const float* d_W,
                       int K, int N, cudaStream_t st) {
    gemv_fp32_kernel<<<N, 128, 0, st>>>(d_y, d_x, d_W, K, N);
}

// ── Weight loading ────────────────────────────────────────────────────
struct Fp32W { int K, N; float* d; };

static Fp32W load_fp8_as_fp32(const char* prefix) {
    char p[512]; snprintf(p, 512, "%s.fp8_t", prefix);
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "FAIL open %s\n", p); exit(1); }
    int h[5]; fread(h, 4, 5, f);
    int K = h[0], N = h[1];
    size_t sz = (size_t)K * N;
    std::vector<uint8_t> fp8_data(sz);
    fread(fp8_data.data(), 1, sz, f);
    fclose(f);

    // Load per-row scales
    snprintf(p, 512, "%s.scale_t", prefix);
    f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "FAIL open %s\n", p); exit(1); }
    int hs[5]; fread(hs, 4, 5, f);
    // hs[3] = N (num scales = num rows), hs[4] = N
    int nscales = hs[3];
    std::vector<float> scales(nscales);
    fread(scales.data(), 4, nscales, f);
    fclose(f);

    // Dequant FP8 → FP32 on CPU
    std::vector<float> fp32_data(sz);
    fp8_e4m3_to_fp32(fp8_data.data(), fp32_data.data(), (int)sz);

    // Apply per-row scales
    apply_row_scales(fp32_data.data(), scales.data(), N, K);

    Fp32W w{K, N, nullptr};
    AL(cudaMalloc(&w.d, sz * 4));
    AL(cudaMemcpy(w.d, fp32_data.data(), sz * 4, cudaMemcpyHostToDevice));
    return w;
}

// Load FP8 embed to host (for row lookup)
static void load_fp8_embed_host(const char* prefix, float* host_buf, int expected_sz) {
    char p[512]; snprintf(p, 512, "%s.fp8_t", prefix);
    FILE* f = fopen(p, "rb"); if (!f) { fprintf(stderr, "FAIL open %s\n", p); exit(1); }
    int h[5]; fread(h, 4, 5, f);
    int K = h[0], N = h[1];
    size_t sz = (size_t)K * N;
    if ((int)sz != expected_sz) { fprintf(stderr, "embed size mismatch: %zu vs %d\n", sz, expected_sz); exit(1); }
    std::vector<uint8_t> fp8_data(sz);
    fread(fp8_data.data(), 1, sz, f); fclose(f);

    snprintf(p, 512, "%s.scale_t", prefix);
    f = fopen(p, "rb"); if (!f) { fprintf(stderr, "FAIL open %s\n", p); exit(1); }
    int hs[5]; fread(hs, 4, 5, f);
    int nscales = hs[3];
    std::vector<float> scales(nscales);
    fread(scales.data(), 4, nscales, f); fclose(f);

    fp8_e4m3_to_fp32(fp8_data.data(), host_buf, (int)sz);
    apply_row_scales(host_buf, scales.data(), N, K);
}

static float* load_f32(const char* pfx, int n) {
    char p[512]; snprintf(p, 512, "%s.f32", pfx);
    FILE* f = fopen(p, "rb"); if (!f) { fprintf(stderr, "FAIL open %s\n", p); exit(1); }
    std::vector<float> tmp(n); fread(tmp.data(), 4, n, f); fclose(f);
    float* d; AL(cudaMalloc(&d, n * 4)); AL(cudaMemcpy(d, tmp.data(), n * 4, cudaMemcpyHostToDevice));
    return d;
}

// ── Head norm + RoPE ──────────────────────────────────────────────────
__global__ void hn_kernel(float* d, const float* w, int nh, int hd, float eps) {
    int h = blockIdx.x; if (h >= nh) return;
    float* p = d + h * hd; __shared__ float ws[4];
    float s = 0; int tid = threadIdx.x;
    for (int i = tid; i < hd; i += blockDim.x) s += p[i] * p[i];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if ((tid & 31) == 0) ws[tid >> 5] = s; __syncthreads();
    if (tid < 32) { float v = (tid < 4) ? ws[tid] : 0; for (int o = 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o); if (tid == 0) ws[0] = rsqrtf(v / hd + eps); }
    __syncthreads(); float is = ws[0];
    for (int i = tid; i < hd; i += blockDim.x) p[i] = p[i] * is * w[i];
}
__global__ void rope_kernel(float* d, int nh, int hd, int pos) {
    int h = blockIdx.x, t = threadIdx.x; if (h >= nh || t >= hd/2) return;
    float* pair = d + h * hd + t * 2;
    float th = (float)pos * powf(1000000.0f, -2.0f * (float)t / (float)hd);
    float c = cosf(th), s = sinf(th), x = pair[0], y = pair[1];
    pair[0] = x * c - y * s; pair[1] = x * s + y * c;
}

// ── Layer weights ─────────────────────────────────────────────────────
struct LayerW {
    Fp32W q, k, v, o, gate, up, down;
    float *rn_in, *rn_post, *qk_n;
};

// ── FP32 decode step ──────────────────────────────────────────────────
static void decode_step_fp32(float* d_residual, int seq_pos, int l,
    int NL, int H, int Q, int KV, int ID, int V,
    int nqh, int nkv, int hd,
    float* d_x_in,
    float* d_Q, float* d_K, float* d_V,
    float* d_attn,
    float* d_proj,
    float* d_gate, float* d_up, float* d_mlp,
    const LayerW& L,
    float* d_kc, float* d_vc, cudaStream_t st)
{
    size_t kv_off = (size_t)l * nkv * hd * KV;

    // Input layernorm → d_x_in
    AL(blackwell::kernels::fused_rmsnorm(d_x_in, d_residual, L.rn_in, H, 1e-6f, st));

    // QKV projections (FP32 GEMV)
    gemv_fp32(d_Q, d_x_in, L.q.d, H, Q, st);
    gemv_fp32(d_K, d_x_in, L.k.d, H, KV, st);
    gemv_fp32(d_V, d_x_in, L.v.d, H, KV, st);

    // Head norm + RoPE
    hn_kernel<<<nqh, 128, 0, st>>>(d_Q, L.qk_n, nqh, hd, 1e-6f);
    hn_kernel<<<nkv, 128, 0, st>>>(d_K, L.qk_n + nqh * hd, nkv, hd, 1e-6f);
    rope_kernel<<<nqh, hd/2, 0, st>>>(d_Q, nqh, hd, seq_pos);
    rope_kernel<<<nkv, hd/2, 0, st>>>(d_K, nkv, hd, seq_pos);
    AL(cudaGetLastError());

    // KV cache
    AL(blackwell::kernels::update_kv_cache(d_kc + kv_off, d_vc + kv_off, d_K, d_V, 0, seq_pos, nkv, hd, KV, st));

    // Attention
    AL(blackwell::kernels::attention_decode_gqa(d_attn, d_Q, d_kc + kv_off, d_vc + kv_off, seq_pos, nqh, nkv, hd, KV, st));

    // Out projection + residual 1
    gemv_fp32(d_proj, d_attn, L.o.d, Q, H, st);
    AL(blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_residual, H, st));
    AL(cudaMemcpyAsync(d_residual, d_proj, H * 4, cudaMemcpyDeviceToDevice, st));

    // Post-attention layernorm
    AL(blackwell::kernels::fused_rmsnorm(d_x_in, d_proj, L.rn_post, H, 1e-6f, st));

    // MLP
    gemv_fp32(d_gate, d_x_in, L.gate.d, H, ID, st);
    gemv_fp32(d_up, d_x_in, L.up.d, H, ID, st);
    AL(blackwell::kernels::apply_swiglu(d_mlp, d_gate, d_up, ID, st));
    gemv_fp32(d_proj, d_mlp, L.down.d, ID, H, st);

    // Residual 2
    AL(blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_residual, H, st));
    AL(cudaMemcpyAsync(d_residual, d_proj, H * 4, cudaMemcpyDeviceToDevice, st));
}

// ── Main ──────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    int max_tokens = (argc > 1) ? atoi(argv[1]) : 100;

    const char* wdir = "weights_fp8_bf16";
    int NL = 28, H = 2048, Q = 2048, KV = 1024, ID = 6144, V = 151936;
    int nqh = 16, nkv = 8, hd = 128;

    printf("FP8 PPL Benchmark: NL=%d H=%d V=%d\n", NL, H, V);
    printf("Weight dir: %s (FP8 block-scaled → FP32 dequant)\n", wdir);

    // Load tokenizer
    blackwell::BpeTokenizer tokenizer;
    if (tokenizer.load("tokenizer_data.bin") != 0) { fprintf(stderr, "FAIL tokenizer\n"); return 1; }
    auto tids = tokenizer.encode(std::string(TEST_TEXT));
    printf("Test text: %zu tokens\n", tids.size());
    int N = max_tokens < (int)tids.size() ? max_tokens : (int)tids.size();
    printf("Evaluating: %d tokens\n", N);

    // Load weights
    printf("Loading FP8 weights (dequant to FP32)...\n");
    cudaStream_t st; cudaStreamCreate(&st);

    // Embed (host-side for row lookup)
    std::vector<float> h_emb(V * H);
    load_fp8_embed_host((std::string(wdir) + "/embed_tokens").c_str(), h_emb.data(), V * H);

    // Embed (device-side for lm_head if tied)
    Fp32W emb; emb.K = H; emb.N = V;
    AL(cudaMalloc(&emb.d, (size_t)V * H * 4));
    AL(cudaMemcpy(emb.d, h_emb.data(), (size_t)V * H * 4, cudaMemcpyHostToDevice));

    float* d_fn = load_f32((std::string(wdir) + "/final_norm").c_str(), H);

    // Layer weights
    std::vector<LayerW> layers(NL);
    for (int l = 0; l < NL; l++) {
        char p[256]; auto& L = layers[l];
        snprintf(p, 256, "%s/%d_self_attn.q_proj", wdir, l); L.q = load_fp8_as_fp32(p);
        snprintf(p, 256, "%s/%d_self_attn.k_proj", wdir, l); L.k = load_fp8_as_fp32(p);
        snprintf(p, 256, "%s/%d_self_attn.v_proj", wdir, l); L.v = load_fp8_as_fp32(p);
        snprintf(p, 256, "%s/%d_self_attn.o_proj", wdir, l); L.o = load_fp8_as_fp32(p);
        snprintf(p, 256, "%s/%d_mlp.gate_proj", wdir, l); L.gate = load_fp8_as_fp32(p);
        snprintf(p, 256, "%s/%d_mlp.up_proj", wdir, l);   L.up = load_fp8_as_fp32(p);
        snprintf(p, 256, "%s/%d_mlp.down_proj", wdir, l); L.down = load_fp8_as_fp32(p);
        snprintf(p, 256, "%s/%d_input_layernorm", wdir, l); L.rn_in = load_f32(p, H);
        snprintf(p, 256, "%s/%d_post_attention_layernorm", wdir, l); L.rn_post = load_f32(p, H);

        // qk_norms: NL*2*hd floats, 2*hd per layer
        snprintf(p, 256, "%s/qk_norms.f32", wdir);
        {
            FILE* qf = fopen(p, "rb");
            if (!qf) { fprintf(stderr, "FAIL open %s\n", p); exit(1); }
            fseek(qf, (long)l * 2 * hd * 4, SEEK_SET);
            std::vector<float> qk_buf(2 * hd);
            fread(qk_buf.data(), 4, 2 * hd, qf);
            fclose(qf);
            int total_qk = nqh * hd + nkv * hd;
            std::vector<float> expanded(total_qk);
            for (int h = 0; h < nqh; h++)
                memcpy(&expanded[h * hd], &qk_buf[0], hd * 4);
            for (int h = 0; h < nkv; h++)
                memcpy(&expanded[nqh * hd + h * hd], &qk_buf[hd], hd * 4);
            AL(cudaMalloc(&L.qk_n, total_qk * 4));
            AL(cudaMemcpy(L.qk_n, expanded.data(), total_qk * 4, cudaMemcpyHostToDevice));
        }
    }

    // Optional lm_head
    Fp32W lm_head{0, 0, nullptr};
    {
        char p[256]; snprintf(p, 256, "%s/lm_head.fp8_t", wdir);
        FILE* f = fopen(p, "rb");
        if (f) { fclose(f); lm_head = load_fp8_as_fp32((std::string(wdir) + "/lm_head").c_str()); printf("  lm_head: separate\n"); }
        else { printf("  lm_head: tied (embed_tokens)\n"); }
    }
    printf("  done\n");

    // Allocate GPU buffers
    float *d_residual, *d_x_in, *d_logits, *d_logp;
    float *d_Q, *d_K, *d_V, *d_attn, *d_proj, *d_gate, *d_up, *d_mlp;
    float *d_kc, *d_vc;

    AL(cudaMalloc(&d_residual, H * 4));
    AL(cudaMalloc(&d_x_in, H * 4));
    AL(cudaMalloc(&d_logits, V * 4));
    AL(cudaMalloc(&d_logp, 4));
    AL(cudaMalloc(&d_Q, Q * 4));
    AL(cudaMalloc(&d_K, KV * 4));
    AL(cudaMalloc(&d_V, KV * 4));
    AL(cudaMalloc(&d_attn, Q * 4));
    AL(cudaMalloc(&d_proj, H * 4));
    AL(cudaMalloc(&d_gate, ID * 4));
    AL(cudaMalloc(&d_up, ID * 4));
    AL(cudaMalloc(&d_mlp, ID * 4));

    size_t kv_sz = (size_t)NL * nkv * hd * KV * 4;
    AL(cudaMalloc(&d_kc, kv_sz)); AL(cudaMemset(d_kc, 0, kv_sz));
    AL(cudaMalloc(&d_vc, kv_sz)); AL(cudaMemset(d_vc, 0, kv_sz));

    // Run benchmark
    double total_logp = 0.0;
    int valid_tokens = 0;
    std::vector<float> h_hidden(H);
    float host_logp;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step + 1 < N; step++) {
        uint32_t tok_id = tids[step];
        memcpy(h_hidden.data(), &h_emb[tok_id * H], H * 4);
        AL(cudaMemcpy(d_residual, h_hidden.data(), H * 4, cudaMemcpyHostToDevice));
        AL(cudaStreamSynchronize(st));

        for (int l = 0; l < NL; l++) {
            decode_step_fp32(d_residual, step, l,
                NL, H, Q, KV, ID, V, nqh, nkv, hd,
                d_x_in, d_Q, d_K, d_V, d_attn, d_proj,
                d_gate, d_up, d_mlp,
                layers[l], d_kc, d_vc, st);
        }
        AL(cudaStreamSynchronize(st));

        // Predict log P(tids[step+1] | tids[0..step])
        float* d_xn = d_x_in;
        AL(blackwell::kernels::fused_rmsnorm(d_xn, d_residual, d_fn, H, 1e-6f, st));
        gemv_fp32(d_logits, d_xn,
            (lm_head.d ? lm_head.d : emb.d),
            H, V, st);
        AL(cudaGetLastError());

        int shmem = sizeof(float) * 256;
        logprob_kernel<<<1, 256, shmem, st>>>(d_logits, V, (int)tids[step + 1], d_logp);
        AL(cudaMemcpy(&host_logp, d_logp, 4, cudaMemcpyDeviceToHost));
        total_logp += (double)host_logp;
        valid_tokens++;
    }

    AL(cudaStreamSynchronize(st));
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
    double tps = (double)valid_tokens / elapsed_s;
    double ppl = exp(-total_logp / (double)valid_tokens);

    printf("\n=== FP8 E4M3 Block-Scaled Results ===\n");
    printf("  Tokens:     %d\n", valid_tokens);
    printf("  Time:       %.3f s\n", elapsed_s);
    printf("  Throughput: %.0f t/s (%.2f ms/tok)\n", tps, 1000.0 / tps);
    printf("  Log P sum:  %.4f\n", total_logp);
    printf("  Perplexity: %.2f\n", ppl);
    printf("======================================\n");

    // Quality gate
    printf("\n--- Quality Gate ---\n");
    printf("  INT8 PPL:      7,351,868 (baseline)\n");
    if (ppl < 200) {
        printf("  PASS: FP8 PPL=%.2f < 200 → proceed to Phase 2\n", ppl);
    } else if (ppl < 500) {
        printf("  WARN: FP8 PPL=%.2f in [200, 500)\n", ppl);
    } else {
        printf("  FAIL: FP8 PPL=%.2f > 500\n", ppl);
    }

    return 0;
}
