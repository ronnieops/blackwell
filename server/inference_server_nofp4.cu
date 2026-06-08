// server/inference_server_nofp4.cu — Blackwell INT8 nofp4 inference server
//
// FP32 residual state (no FP4 pack/unpack). Persistent C++ daemon for M=8 batched decode.
// Uses CORRECT model architecture: per-layer RMSNorm, Q/K head norms, RoPE.
// Protocol: JSON lines on stdin/stdout.
//
// Build:
//   CUDACXX=/usr/local/cuda-13.3/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include server/inference_server_nofp4.cu build/libblackwell_kernels.a \
//     -o server/inference_server

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include "blackwell/kernels.h"
#include "blackwell/bpe_tokenizer.h"

static void die(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) { fprintf(stderr, "FAIL %s: %s\n", msg, cudaGetErrorString(e)); exit(1); }
}

// ── Inline CUDA kernels (not in library) ──────────────────────────────

// Per-head RMSNorm for Q/K attention heads (from text_generate.cu)
__global__ void head_norm_kernel(float* data, const float* weight, int nh, int hd, float eps) {
    int h = blockIdx.x; if (h >= nh) return;
    float* d = data + h * hd;
    __shared__ float warp_partial[4];
    float s = 0;
    int tid = threadIdx.x;
    for (int i = tid; i < hd; i += blockDim.x) s += d[i] * d[i];
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
    if ((tid & 31) == 0) warp_partial[tid >> 5] = s;
    __syncthreads();
    if (tid < 4) s = warp_partial[tid]; else s = 0;
    for (int off = 2; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
    if (tid == 0) warp_partial[0] = rsqrtf(s / hd + eps);
    __syncthreads();
    float is = warp_partial[0];
    for (int i = tid; i < hd; i += blockDim.x) d[i] = d[i] * is * weight[i];
}

// RoPE kernel using library's device-side seq_pos (cudaGraph compatible)
// RoPE kernel — reads seq_pos from library's device pointer (set by update_decode_seq_pos)
// Both the kernel AND attention_decode_gqa read from the same device pointer.
// update_decode_seq_pos's captured cudaMemcpyAsync replays each graph launch → correct seq_pos.
__global__ void rope_kernel(float* data, int n_heads, int head_dim, int* d_seq_pos) {
    int h = blockIdx.x; int d = threadIdx.x;
    if (h >= n_heads || d >= head_dim / 2) return;
    // Read pos from device memory (updated by update_decode_seq_pos before kernel launch)
    // Single __threadfence ensures visibility of the async copy result
    int pos = *d_seq_pos;
    __threadfence();
    const float rope_theta = 1000000.0f;
    float theta = (float)pos * powf(rope_theta, -2.0f * (float)d / (float)head_dim);
    float c = cosf(theta), s = sinf(theta);
    int i2 = d * 2;
    float* pair = data + h * head_dim + i2;
    float x = pair[0], y = pair[1];
    pair[0] = x * c - y * s;
    pair[1] = x * s + y * c;
}
struct DevW { int K, N; int8_t* d; float* sc; };
static DevW upload_int8(const char* prefix) {
    char p[256]; snprintf(p, 256, "%s.int8_t", prefix);
    FILE* f = fopen(p, "rb"); if (!f) { fprintf(stderr, "Cannot open %s\n", p); exit(1); }
    int h[5]; (void)fread(h, 4, 5, f);
    std::vector<int8_t> tmp((size_t)h[0] * h[1]); (void)fread(tmp.data(), 1, tmp.size(), f); fclose(f);
    DevW dw{h[0], h[1], nullptr, nullptr};
    cudaMalloc(&dw.d, (size_t)h[0] * h[1]); cudaMemcpy(dw.d, tmp.data(), dw.K * dw.N, cudaMemcpyHostToDevice);
    snprintf(p, 256, "%s.scale_t", prefix); f = fopen(p, "rb"); if (!f) { fprintf(stderr, "Cannot open %s\n", p); exit(1); }
    (void)fread(h, 4, 5, f); size_t ns = (size_t)h[3] * h[4]; std::vector<float> ts(ns); (void)fread(ts.data(), 4, ns, f); fclose(f);
    cudaMalloc(&dw.sc, ns * 4); cudaMemcpy(dw.sc, ts.data(), ns * 4, cudaMemcpyHostToDevice);
    return dw;
}


struct Fp16W { int K, N; __half* d; };
static Fp16W upload_fp16(const char* prefix) {
    char p[256]; snprintf(p, 256, "%s.fp16", prefix);
    FILE* f = fopen(p, "rb"); if (!f) { fprintf(stderr, "Cannot open %s\n", p); exit(1); }
    int h[2]; (void)fread(h, 4, 2, f);
    Fp16W w{h[0], h[1], nullptr};
    size_t sz = (size_t)h[0] * h[1] * 2;
    std::vector<uint8_t> tmp(sz); (void)fread(tmp.data(), 1, sz, f); fclose(f);
    cudaMalloc(&w.d, sz); cudaMemcpy(w.d, tmp.data(), sz, cudaMemcpyHostToDevice);
    return w;
}

struct LW { 
    DevW q, k, v, o, gate, up, down; 
    float* qn; float* kn;
    bool is_fp16;
    __half *fp16_q, *fp16_k, *fp16_v, *fp16_o;
    __half *fp16_gate, *fp16_up, *fp16_down;
};

// Dispatch helper: call gemv_int8_warp or gemv_fp16_warp_launch based on layer type
static void gemv_dispatch(float* y, const LW& layer, const DevW& w, __half* fp16_w,
    const int8_t* x_i8, const float* x_sc, int K, int N, cudaStream_t st) {
    if (layer.is_fp16) {
        blackwell::kernels::gemv_fp16_warp_launch(y, (void*)fp16_w, x_i8, x_sc, K, N, st);
    } else {
        blackwell::kernels::gemv_int8_warp(y, x_i8, x_sc, w.d, w.sc, K, N, st);
    }
}

// Batched gemv dispatch: loops for FP16, uses gemv_int8_batched for INT8.
static void gemv_batched_dispatch(float* y, const LW& layer, const DevW& w, __half* fp16_w,
    const int8_t* x_i8, const float* x_sc, int K, int N, int M, cudaStream_t st) {
    if (layer.is_fp16) {
        for (int m = 0; m < M; m++) {
            blackwell::kernels::gemv_fp16_warp_launch(y + m * N, (void*)fp16_w,
                x_i8 + m * K, x_sc + m * (K / 16), K, N, st);
        }
    } else {
        blackwell::kernels::gemv_int8_batched(y, x_i8, x_sc, w.d, w.sc, K, N, M, st);
    }
}

struct ServerState {
    int NL, H, Q, KV, ID, nqh, nkv, hd, ms, V;
    int fp16_first_n;
    float eps;

    std::vector<LW> layers;
    // Per-layer RMSNorm weights
    std::vector<float*> d_rn_in;    // input layernorm [NL][H]
    std::vector<float*> d_rn_post;  // post-attention layernorm [NL][H]
    float* d_fn;                     // final RMSNorm [H]

    // Embed tokens (host-side for CPU dequant)
    DevW emb;
    DevW lm_head;  // separate lm_head (empty if tied)
    int8_t* h_emb_int8;
    float* h_emb_scale;

    // Per-seq FP32 residual state
    float* d_residual[8];

    // Prefill working buffers
    float* d_x;           // [M][H] combined residual (attn+input+MLP)
    float* d_tmp_save;    // [H] temp save for attn+input before MLP

    int M;
    int8_t* d_xi8; float* d_xi8s;
    float* d_Q, *d_K, *d_V;
    float* d_attn, *d_attn_out;
    int8_t* d_attn_i8; float* d_attn_i8s;
    float* d_gate, *d_up, *d_mlp;
    int8_t* d_mlp_i8; float* d_mlp_i8s;
    float* d_proj;
    float* d_logits;
    int* d_next_id;

    // KV cache
    float* d_kc, *d_vc;
    size_t kv_stride;

    // Device-side seq_pos for graph-compatible RoPE
    int* d_seq_pos;

    // Repetition penalty
    int* d_recent_tokens;        // [M * MAX_RECENT] device buffer
    int max_recent;             // how many recent tokens to track
    float repetition_penalty;   // penalty > 1.0

    cudaStream_t st;
};

enum { MAX_RECENT = 64 };  // tokens to track for repetition penalty

// ── JSON helpers ──────────────────────────────────────────────────────
static std::string read_stdin_line() {
    std::string line; int c;
    while ((c = getchar()) != EOF && c != '\n') line.push_back((char)c);
    return line;
}

static std::vector<std::vector<uint32_t>> parse_prompt_ids(const std::string& json) {
    std::vector<std::vector<uint32_t>> result;
    // Returns empty if prompts are strings (use parse_string_prompts instead)
    const char* p = strstr(json.c_str(), "\"prompts\"");
    if (!p) return result;
    p = strchr(p, '['); if (!p) return result; p++;
    // Check if first non-whitespace is a quote (string prompt)
    const char* ws = p;
    while (*ws == ' ' || *ws == '\t' || *ws == '\n' || *ws == '\r') ws++;
    if (*ws == '"') return result;  // String prompts handled by parse_string_prompts
    // Numeric IDs
    while (*p && *p != ']') {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',') p++;
        if (*p == ']' || !*p) break;
        if (*p == '[') { p++; continue; }
        long val = strtol(p, (char**)&p, 10);
        if (val == 0 && p == json.c_str()) { p++; continue; }
        std::vector<uint32_t> ids;
        ids.push_back((uint32_t)val);
        while (*p == ' ' || *p == ',') p++;
        if (*p == ']') { result.push_back(ids); break; }
        while (*p && *p != ']') {
            while (*p == ' ' || *p == ',' || *p == '\n' || *p == '\r') p++;
            if (*p == ']' || !*p) break;
            val = strtol(p, (char**)&p, 10);
            ids.push_back((uint32_t)val);
        }
        result.push_back(ids);
        if (*p == ']') p++;
    }
    return result;
}

static int find_int(const std::string& json, const char* key, int def) {
    const char* p = strstr(json.c_str(), key); if (!p) return def;
    p = strchr(p, ':'); if (!p) return def; p++;
    while (*p == ' ' || *p == '\t') p++;
    return (int)strtol(p, nullptr, 10);
}

static float find_float(const std::string& json, const char* key, float def) {
    const char* p = strstr(json.c_str(), key); if (!p) return def;
    p = strchr(p, ':'); if (!p) return def; p++;
    while (*p == ' ' || *p == '\t') p++;
    return (float)atof(p);
}

// Encode with special token support: <|im_start|> → 151644, <|im_end|> → 151645
static std::vector<uint32_t> encode_with_special(const std::string& s, blackwell::BpeTokenizer& tokenizer) {
    std::vector<uint32_t> result;
    size_t pos = 0;
    while (pos < s.size()) {
        // Check for <|im_start|>
        if (s.compare(pos, 12, "<|im_start|>") == 0) {
            result.push_back(151644);
            pos += 12;
        } else if (s.compare(pos, 10, "<|im_end|>") == 0) {
            result.push_back(151645);
            pos += 10;
        } else {
            // Find next special token or end
            size_t next_sp = s.find("<|im_", pos);
            if (next_sp == std::string::npos) next_sp = s.size();
            std::string chunk = s.substr(pos, next_sp - pos);
            auto ids = tokenizer.encode(chunk);
            result.insert(result.end(), ids.begin(), ids.end());
            pos = next_sp;
        }
    }
        return result;
}

static std::vector<std::vector<uint32_t>> parse_string_prompts(
    const std::string& json, blackwell::BpeTokenizer& tokenizer) {
    std::vector<std::vector<uint32_t>> result;
    // Try "prompt" key first (OpenAI API style)
    const char* prompt_p = strstr(json.c_str(), "\"prompt\"");
    if (prompt_p && prompt_p[8] == ':') {
        const char* p = strchr(prompt_p, ':'); if (!p) return result; p++;
        while (*p == ' ' || *p == '\t') p++;
        if (*p != '"') return result; p++;
        std::string s;
        while (*p && *p != '"') {
            if (*p == '\\' && *(p+1) == 'n') { s += '\n'; p += 2; }
            else if (*p == '\\' && *(p+1) == 'r') { s += '\r'; p += 2; }
            else if (*p == '\\' && *(p+1) == 't') { s += '\t'; p += 2; }
            else if (*p == '\\' && *(p+1) == '\\') { s += '\\'; p += 2; }
            else { s += *p; p++; }
        }
        result.push_back(encode_with_special(s, tokenizer));
                    return result;
    }
    // Try "prompts" array (batch style)
    const char* p = strstr(json.c_str(), "\"prompts\"");
    if (!p) return result;
    p = strchr(p, '['); if (!p) return result; p++;
    while (*p && *p != ']') {
        while (*p && (*p == ' ' || *p == '\t' || *p == '\n')) p++;
        if (*p == ']' || !*p) break;
        if (*p != '"') { p++; continue; } p++;
        std::string s;
        while (*p && *p != '"') {
            if (*p == '\\' && *(p + 1) == '"') { s += '"'; p += 2; }
            else if (*p == '\\' && *(p + 1) == 'n') { s += '\n'; p += 2; }
            else if (*p == '\\' && *(p + 1) == 'r') { s += '\r'; p += 2; }
            else if (*p == '\\' && *(p + 1) == 't') { s += '\t'; p += 2; }
            else if (*p == '\\' && *(p + 1) == '\\') { s += '\\'; p += 2; }
            else { s += *p; p++; }
        }
        if (*p == '"') p++;
        auto ids = encode_with_special(s, tokenizer);
        result.push_back(ids);
        while (*p && (*p == ',' || *p == ' ' || *p == '\t' || *p == '\n')) p++;
    }
        return result;
}

static void server_write_results(const std::vector<std::vector<uint32_t>>& tokens,
    blackwell::BpeTokenizer& tokenizer) {
    printf("{\"tokens\":[");
    for (size_t s = 0; s < tokens.size(); s++) {
        if (s > 0) printf(",");
        printf("[");
        for (size_t t = 0; t < tokens[s].size(); t++) {
            if (t > 0) printf(",");
            printf("%u", tokens[s][t]);
        }
        printf("]");
    }
    printf("],\"text\":[");
    for (size_t s = 0; s < tokens.size(); s++) {
        if (s > 0) printf(",");
        printf("\"");
        std::string txt;
        for (size_t t = 0; t < tokens[s].size(); t++) txt += tokenizer.decode(tokens[s][t]);
        for (size_t i = 0; i < txt.size(); i++) {
            char c = txt[i];
            if (c == '"') printf("\\\"");
            else if (c == '\\') printf("\\\\");
            else if (c == '\n') printf("\\n");
            else if (c == '\r') printf("\\r");
            else if (c == '\t') printf("\\t");
            else printf("%c", c);
        }
        printf("\"");
    }
    printf("]}\n"); fflush(stdout);
}

// ── Embed: CPU-side dequant, store as FP32 residual ───────────────────
static void embed_batch(ServerState& S, const std::vector<std::vector<uint32_t>>& prompts, int step) {
    int M = (int)prompts.size();
    for (int m = 0; m < M; m++) {
        uint32_t tid = prompts[m][step];
        std::vector<float> h_hidden(S.H);
        for (int d = 0; d < S.H; d++)
            h_hidden[d] = (float)S.h_emb_int8[tid * S.H + d] * S.h_emb_scale[tid * (S.H / 16) + d / 16];
        cudaMemcpy(S.d_residual[m], h_hidden.data(), S.H * 4, cudaMemcpyHostToDevice);
    }
}

// ── Batched decode step (CORRECT model architecture) ──────────────────
static void batched_decode_step(ServerState& S, int seq_pos) {
    int M = S.M;
    size_t kv_seq_stride = (size_t)S.nkv * S.hd * S.ms;

    for (int l = 0; l < S.NL; l++) {
        size_t kv_layer_off = (size_t)l * S.nkv * S.hd * S.ms;

        // ── Input layernorm + quantize (per-sequence) ──
        for (int m = 0; m < M; m++) {
            blackwell::kernels::fused_rmsnorm_quant_int8(
                S.d_xi8 + m * S.H, S.d_xi8s + m * (S.H / 16),
                S.d_residual[m], S.d_rn_in[l], S.H, S.eps, S.st);
        }

        // ── Q/K/V projections + head_norm + RoPE (per-sequence) ──
        for (int m = 0; m < M; m++) {
            gemv_dispatch(S.d_Q + m * S.Q, S.layers[l], S.layers[l].q, S.layers[l].fp16_q,
    S.d_xi8 + m * S.H, S.d_xi8s + m * (S.H / 16),
                S.H, S.Q, S.st);
            gemv_dispatch(S.d_K + m * S.KV, S.layers[l], S.layers[l].k, S.layers[l].fp16_k,
    S.d_xi8 + m * S.H, S.d_xi8s + m * (S.H / 16),
                S.H, S.KV, S.st);
            gemv_dispatch(S.d_V + m * S.KV, S.layers[l], S.layers[l].v, S.layers[l].fp16_v,
    S.d_xi8 + m * S.H, S.d_xi8s + m * (S.H / 16),
                S.H, S.KV, S.st);

            // Q/K head norms
            head_norm_kernel<<<S.nqh, 128, 0, S.st>>>(S.d_Q + m * S.Q, S.layers[l].qn, S.nqh, S.hd, S.eps);
            head_norm_kernel<<<S.nkv, 128, 0, S.st>>>(S.d_K + m * S.KV, S.layers[l].kn, S.nkv, S.hd, S.eps);

            // RoPE: update device seq_pos and sync before kernel reads it
            blackwell::kernels::update_decode_seq_pos(seq_pos, S.st);
            cudaStreamSynchronize(S.st);
            rope_kernel<<<S.nqh, S.hd / 2, 0, S.st>>>(S.d_Q + m * S.Q, S.nqh, S.hd, S.d_seq_pos);
            rope_kernel<<<S.nkv, S.hd / 2, 0, S.st>>>(S.d_K + m * S.KV, S.nkv, S.hd, S.d_seq_pos);

            // KV cache write
            size_t km = (size_t)m * kv_seq_stride + kv_layer_off;
            blackwell::kernels::update_kv_cache(
                S.d_kc + km, S.d_vc + km, S.d_K + m * S.KV, S.d_V + m * S.KV,
                0, seq_pos, S.nkv, S.hd, S.ms, S.st);
        }

        // ── Attention (single-seq for M=1) ──
        if (M == 1) {
            size_t kb = kv_layer_off;
            blackwell::kernels::attention_decode_gqa(
                S.d_attn, S.d_Q, S.d_kc + kb, S.d_vc + kb,
                seq_pos, S.nqh, S.nkv, S.hd, S.ms, S.st);
        } else {
            blackwell::kernels::attention_decode_batched_gqa(
                S.d_attn, S.d_Q, S.d_kc, S.d_vc,
                seq_pos, S.nqh, S.nkv, S.hd, S.ms,
                M, (int)kv_seq_stride, (int)kv_layer_off, S.st);
        }

        // ── Attention output projection + residual 1 (per-sequence) ──
        for (int m = 0; m < M; m++) {
            blackwell::kernels::quantize_int8(S.d_attn_i8 + m * S.Q, S.d_attn_i8s + m * (S.Q / 16), S.d_attn + m * S.Q, S.Q, S.st);
            gemv_dispatch(S.d_proj + m * S.H, S.layers[l], S.layers[l].o, S.layers[l].fp16_o,
    S.d_attn_i8 + m * S.Q, S.d_attn_i8s + m * (S.Q / 16),
                S.Q, S.H, S.st);
            // Residual 1: proj = attn_out + input (before RMSNorm)
            blackwell::kernels::vector_add_fp32(S.d_proj + m * S.H, S.d_proj + m * S.H, S.d_residual[m], S.H, S.st);
            // Save attention output for MLP residual
            cudaMemcpyAsync(S.d_residual[m], S.d_proj + m * S.H, S.H * 4, cudaMemcpyDeviceToDevice, S.st);
        }

        // ── Post-attention layernorm + quantize (per-sequence) ──
        for (int m = 0; m < M; m++) {
            blackwell::kernels::fused_rmsnorm_quant_int8(
                S.d_xi8 + m * S.H, S.d_xi8s + m * (S.H / 16),
                S.d_proj + m * S.H, S.d_rn_post[l], S.H, S.eps, S.st);
        }

        // ── MLP: batched gate/up ──
        gemv_batched_dispatch(S.d_gate, S.layers[l], S.layers[l].gate, S.layers[l].fp16_gate,
    S.d_xi8, S.d_xi8s,
             S.H, S.ID, M, S.st);
        gemv_batched_dispatch(S.d_up, S.layers[l], S.layers[l].up, S.layers[l].fp16_up,
    S.d_xi8, S.d_xi8s,
             S.H, S.ID, M, S.st);

        for (int m = 0; m < M; m++) {
            blackwell::kernels::apply_swiglu(S.d_mlp + m * S.ID, S.d_gate + m * S.ID, S.d_up + m * S.ID, S.ID, S.st);
            blackwell::kernels::quantize_int8(S.d_mlp_i8 + m * S.ID, S.d_mlp_i8s + m * (S.ID / 16), S.d_mlp + m * S.ID, S.ID, S.st);
        }

        // ── MLP down projection (batched) ──
        gemv_batched_dispatch(S.d_proj, S.layers[l], S.layers[l].down, S.layers[l].fp16_down,
    S.d_mlp_i8, S.d_mlp_i8s,
             S.ID, S.H, M, S.st);

        // ── Residual 2: down + attention output (per-sequence) ──
        for (int m = 0; m < M; m++) {
            blackwell::kernels::vector_add_fp32(S.d_proj + m * S.H, S.d_proj + m * S.H, S.d_residual[m], S.H, S.st);
            cudaMemcpyAsync(S.d_residual[m], S.d_proj + m * S.H, S.H * 4, cudaMemcpyDeviceToDevice, S.st);
        }
    }
}

// ── Prefill: process M prompt tokens in parallel through all layers ──
static void batched_prefill(ServerState& S, int M) {
    if (M <= 0) return;
    size_t kv_seq_stride = (size_t)S.nkv * S.hd * S.ms;

    for (int l = 0; l < S.NL; l++) {
        size_t kv_layer_off = (size_t)l * S.nkv * S.hd * S.ms;

        // ── Input layernorm + quantize (per-sequence) ──
        for (int m = 0; m < M; m++) {
            blackwell::kernels::fused_rmsnorm_quant_int8(
                S.d_xi8 + m * S.H, S.d_xi8s + m * (S.H / 16),
                S.d_residual[m], S.d_rn_in[l], S.H, S.eps, S.st);
        }

        // ── Q/K/V projections (per-sequence, batched) ──
        for (int m = 0; m < M; m++) {
            gemv_dispatch(S.d_Q + m * S.Q, S.layers[l], S.layers[l].q, S.layers[l].fp16_q,
    S.d_xi8 + m * S.H, S.d_xi8s + m * (S.H / 16),
                S.H, S.Q, S.st);
            gemv_dispatch(S.d_K + m * S.KV, S.layers[l], S.layers[l].k, S.layers[l].fp16_k,
    S.d_xi8 + m * S.H, S.d_xi8s + m * (S.H / 16),
                S.H, S.KV, S.st);
            gemv_dispatch(S.d_V + m * S.KV, S.layers[l], S.layers[l].v, S.layers[l].fp16_v,
    S.d_xi8 + m * S.H, S.d_xi8s + m * (S.H / 16),
                S.H, S.KV, S.st);

            // Q/K head norms
            head_norm_kernel<<<S.nqh, 128, 0, S.st>>>(S.d_Q + m * S.Q, S.layers[l].qn, S.nqh, S.hd, S.eps);
            head_norm_kernel<<<S.nkv, 128, 0, S.st>>>(S.d_K + m * S.KV, S.layers[l].kn, S.nkv, S.hd, S.eps);

            // RoPE for each position (seq_pos = m)
            blackwell::kernels::update_decode_seq_pos(m, S.st);
            cudaStreamSynchronize(S.st);
            rope_kernel<<<S.nqh, S.hd / 2, 0, S.st>>>(S.d_Q + m * S.Q, S.nqh, S.hd, S.d_seq_pos);
            rope_kernel<<<S.nkv, S.hd / 2, 0, S.st>>>(S.d_K + m * S.KV, S.nkv, S.hd, S.d_seq_pos);
        }

        // ── Write K,V to persistent KV cache (needed by decode step later) ──
        for (int m = 0; m < M; m++) {
            size_t kb = kv_layer_off + (size_t)m * S.nkv * S.hd;
            blackwell::kernels::update_kv_cache(
                S.d_kc + kb, S.d_vc + kb, S.d_K + m * S.KV, S.d_V + m * S.KV,
                0, 0, S.nkv, S.hd, S.ms, S.st);
        }

        // ── Attention: batched M sequences attending to all M ──
        // For prefill, each sequence m attends to all M tokens' K/V (just projected)
        // NOTE: d_K/d_V are per-layer temp buffers, NOT the persistent KV cache.
        // Pass kv_layer_elems=0 because K/V for layer l start at d_K+0 (re-written each layer).
        for (int m = 0; m < M; m++) {
            // Each batch item attends to all M tokens in this layer
            // Note: all kernels write to S.d_attn (shared buffer) — sync before copy
            blackwell::kernels::attention_decode_batched_gqa(
                S.d_attn, S.d_Q, S.d_K, S.d_V,  // Full K,V for all M
                m, S.nqh, S.nkv, S.hd, S.ms,
                M, (int)kv_seq_stride, 0, S.st);
            cudaStreamSynchronize(S.st);  // ensure kernel completes before copy
            cudaMemcpyAsync(S.d_attn_out + m * S.Q, S.d_attn, S.Q * 4, cudaMemcpyDeviceToDevice, S.st);
        }

        // ── Attention output projection + residual 1 (per-sequence) ──
        for (int m = 0; m < M; m++) {
            blackwell::kernels::quantize_int8(S.d_attn_i8 + m * S.Q, S.d_attn_i8s + m * (S.Q / 16), 
                S.d_attn_out + m * S.Q, S.Q, S.st);
            gemv_dispatch(S.d_proj + m * S.H, S.layers[l], S.layers[l].o, S.layers[l].fp16_o,
    S.d_attn_i8 + m * S.Q, S.d_attn_i8s + m * (S.Q / 16),
                S.Q, S.H, S.st);
            // Residual 1: proj = attn_out + input
            blackwell::kernels::vector_add_fp32(S.d_proj + m * S.H, S.d_proj + m * S.H, S.d_residual[m], S.H, S.st);
            cudaMemcpyAsync(S.d_residual[m], S.d_proj + m * S.H, S.H * 4, cudaMemcpyDeviceToDevice, S.st);
        }

        // ── Post-attention layernorm + quantize ──
        for (int m = 0; m < M; m++) {
            blackwell::kernels::fused_rmsnorm_quant_int8(
                S.d_xi8 + m * S.H, S.d_xi8s + m * (S.H / 16),
                S.d_proj + m * S.H, S.d_rn_post[l], S.H, S.eps, S.st);
        }

        // ── MLP: batched gate/up ──
        gemv_batched_dispatch(S.d_gate, S.layers[l], S.layers[l].gate, S.layers[l].fp16_gate,
    S.d_xi8, S.d_xi8s,
             S.H, S.ID, M, S.st);
        gemv_batched_dispatch(S.d_up, S.layers[l], S.layers[l].up, S.layers[l].fp16_up,
    S.d_xi8, S.d_xi8s,
             S.H, S.ID, M, S.st);

        for (int m = 0; m < M; m++) {
            blackwell::kernels::apply_swiglu(S.d_mlp + m * S.ID, S.d_gate + m * S.ID, S.d_up + m * S.ID, S.ID, S.st);
            blackwell::kernels::quantize_int8(S.d_mlp_i8 + m * S.ID, S.d_mlp_i8s + m * (S.ID / 16), S.d_mlp + m * S.ID, S.ID, S.st);
        }

        // ── MLP down projection + residual 2 ──
        gemv_batched_dispatch(S.d_proj, S.layers[l], S.layers[l].down, S.layers[l].fp16_down,
    S.d_mlp_i8, S.d_mlp_i8s,
             S.ID, S.H, M, S.st);

        for (int m = 0; m < M; m++) {
            blackwell::kernels::vector_add_fp32(S.d_proj + m * S.H, S.d_proj + m * S.H, S.d_residual[m], S.H, S.st);
            cudaMemcpyAsync(S.d_residual[m], S.d_proj + m * S.H, S.H * 4, cudaMemcpyDeviceToDevice, S.st);
        }
    }
}

// ── Main ──────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    const char* model = (argc > 1) ? argv[1] : "1.7b";

    ServerState S;
    S.eps = 1e-6f; S.M = 8;

    const char* wdir;  // weights directory
    if (strstr(model, "8b_instruct")) {
        S.NL = 36; S.H = 4096; S.Q = 4096; S.KV = 1024; S.ID = 12288;
        S.nqh = 32; S.nkv = 8; S.hd = 128; S.ms = 2048; S.V = 151936;
        wdir = "weights_int8_qwen3_8b_instruct";
    } else if (strstr(model, "8b")) {
        S.NL = 36; S.H = 4096; S.Q = 4096; S.KV = 1024; S.ID = 12288;
        S.nqh = 32; S.nkv = 8; S.hd = 128; S.ms = 2048; S.V = 151936;
        wdir = "weights_int8_qwen3_8b";
    } else {
        S.NL = 28; S.H = 2048; S.Q = 2048; S.KV = 1024; S.ID = 6144;
        S.nqh = 16; S.nkv = 8; S.hd = 128; S.ms = 2048; S.V = 151936;
        wdir = "weights_int8_bf16";
    }

    fprintf(stderr, "Blackwell INT8 NOFP4 Server v0.6.2 (model=%s, %d layers, H=%d)\n", model, S.NL, S.H);

    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    fprintf(stderr, "  Streaming, batched prefill\n");
    fprintf(stderr, "Device: %s (CC %d.%d)\n", p.name, p.major, p.minor);

    die(cudaStreamCreate(&S.st), "stream");

    blackwell::BpeTokenizer tokenizer;
    if (tokenizer.load("tokenizer_data.bin") != 0) {
        fprintf(stderr, "FAIL: no tokenizer_data.bin\n"); return 1;
    }

    // ── Load INT8 weights ──
    fprintf(stderr, "Loading weights...\n");
    S.layers.resize(S.NL);
    for (int l = 0; l < S.NL; l++) {
        char p[256];
        bool is_fp16 = false;
        // Check if FP16 file exists
        snprintf(p, 256, "%s/%d_self_attn.q_proj.fp16", wdir, l);
        FILE* tst = fopen(p, "rb");
        if (tst) { is_fp16 = true; fclose(tst); }
        S.layers[l].is_fp16 = is_fp16;
        
        if (is_fp16) {
            auto lfp16 = [&](const char* nm, DevW& dw, __half*& dst) {
                snprintf(p, 256, "%s/%d_%s", wdir, l, nm);  // upload_fp16 adds .fp16
                auto w = upload_fp16(p);
                dw.K = w.K; dw.N = w.N; dw.d = nullptr; dw.sc = nullptr;
                dst = w.d;
            };
            lfp16("self_attn.q_proj", S.layers[l].q, S.layers[l].fp16_q);
            lfp16("self_attn.k_proj", S.layers[l].k, S.layers[l].fp16_k);
            lfp16("self_attn.v_proj", S.layers[l].v, S.layers[l].fp16_v);
            lfp16("self_attn.o_proj", S.layers[l].o, S.layers[l].fp16_o);
            lfp16("mlp.gate_proj", S.layers[l].gate, S.layers[l].fp16_gate);
            lfp16("mlp.up_proj", S.layers[l].up, S.layers[l].fp16_up);
            lfp16("mlp.down_proj", S.layers[l].down, S.layers[l].fp16_down);
        } else {
            snprintf(p, 256, "%s/%d_self_attn.q_proj", wdir, l); S.layers[l].q = upload_int8(p);
            snprintf(p, 256, "%s/%d_self_attn.k_proj", wdir, l); S.layers[l].k = upload_int8(p);
            snprintf(p, 256, "%s/%d_self_attn.v_proj", wdir, l); S.layers[l].v = upload_int8(p);
            snprintf(p, 256, "%s/%d_self_attn.o_proj", wdir, l); S.layers[l].o = upload_int8(p);
            snprintf(p, 256, "%s/%d_mlp.gate_proj", wdir, l);  S.layers[l].gate = upload_int8(p);
            snprintf(p, 256, "%s/%d_mlp.up_proj", wdir, l);    S.layers[l].up = upload_int8(p);
            snprintf(p, 256, "%s/%d_mlp.down_proj", wdir, l);  S.layers[l].down = upload_int8(p);
        }
        if (l % 7 == 0) fprintf(stderr, "  layer %d/%d%s\n", l, S.NL, is_fp16 ? " (FP16)" : "");
    }
    S.emb = upload_int8((std::string(wdir) + "/embed_tokens").c_str());

    // Load lm_head if exists (separate from embed_tokens)
    {
        char plm[256]; snprintf(plm, 256, "%s/lm_head.int8_t", wdir);
        FILE* flm = fopen(plm, "rb");
        if (flm) { fclose(flm); S.lm_head = upload_int8((std::string(wdir) + "/lm_head").c_str()); fprintf(stderr, "  lm_head: separate (INT8)\n"); }
        else { S.lm_head.d = nullptr; }
    }

    // Host-side embed copies
    {
        char p[256]; snprintf(p, 256, "%s/embed_tokens.int8_t", wdir);
        FILE* f = fopen(p, "rb"); int h[5]; (void)fread(h, 4, 5, f);
        size_t num = (size_t)h[0] * h[1];
        S.h_emb_int8 = (int8_t*)malloc(num); (void)fread(S.h_emb_int8, 1, num, f); fclose(f);
        snprintf(p, 256, "%s/embed_tokens.scale_t", wdir);
        f = fopen(p, "rb"); (void)fread(h, 4, 5, f);
        size_t ns = (size_t)h[3] * h[4];
        S.h_emb_scale = (float*)malloc(ns * 4); (void)fread(S.h_emb_scale, 4, ns, f); fclose(f);
    }

    // ── Per-layer RMSNorm weights ──
    S.d_rn_in.resize(S.NL); S.d_rn_post.resize(S.NL);
    for (int l = 0; l < S.NL; l++) {
        float* w = (float*)malloc(S.H * 4);
        char p[256];
        snprintf(p, 256, "%s/%d_input_layernorm.f32", wdir, l);
        FILE* f = fopen(p, "rb"); (void)fread(w, 4, S.H, f); fclose(f);
        cudaMalloc(&S.d_rn_in[l], S.H * 4); cudaMemcpy(S.d_rn_in[l], w, S.H * 4, cudaMemcpyHostToDevice);
        snprintf(p, 256, "%s/%d_post_attention_layernorm.f32", wdir, l);
        f = fopen(p, "rb"); (void)fread(w, 4, S.H, f); fclose(f);
        cudaMalloc(&S.d_rn_post[l], S.H * 4); cudaMemcpy(S.d_rn_post[l], w, S.H * 4, cudaMemcpyHostToDevice);
        free(w);
    }

    // ── Per-layer Q/K norms ──
    {
        float* qk_h = (float*)malloc(S.NL * 2 * S.hd * 4);
        FILE* f = fopen((std::string(wdir) + "/qk_norms.f32").c_str(), "rb");
        size_t n_read = fread(qk_h, 4, S.NL * 2 * S.hd, f); fclose(f);
        for (int l = 0; l < S.NL && l < (int)(n_read / (2 * S.hd)); l++) {
            cudaMalloc(&S.layers[l].qn, S.hd * 4);
            cudaMemcpy(S.layers[l].qn, qk_h + l * 2 * S.hd, S.hd * 4, cudaMemcpyHostToDevice);
            cudaMalloc(&S.layers[l].kn, S.hd * 4);
            cudaMemcpy(S.layers[l].kn, qk_h + l * 2 * S.hd + S.hd, S.hd * 4, cudaMemcpyHostToDevice);
        }
        free(qk_h);
    }

    // ── Final norm ──
    {
        float* w = (float*)malloc(S.H * 4);
        FILE* f = fopen((std::string(wdir) + "/final_norm.f32").c_str(), "rb");
        (void)fread(w, 4, S.H, f); fclose(f);
        cudaMalloc(&S.d_fn, S.H * 4); cudaMemcpy(S.d_fn, w, S.H * 4, cudaMemcpyHostToDevice);
        free(w);
    }
    fprintf(stderr, "  done\n");

    // ── Allocate buffers ──
    for (int m = 0; m < S.M; m++) cudaMalloc(&S.d_residual[m], S.H * 4);
    cudaMalloc(&S.d_xi8, S.M * S.H); cudaMalloc(&S.d_xi8s, S.M * (S.H / 16) * 4);
    cudaMalloc(&S.d_Q, S.M * S.Q * 4); cudaMalloc(&S.d_K, S.M * S.KV * 4); cudaMalloc(&S.d_V, S.M * S.KV * 4);
    cudaMalloc(&S.d_attn, S.M * S.Q * 4);
    cudaMalloc(&S.d_attn_out, S.M * S.Q * 4);  // attention output buffer for prefill
    cudaMalloc(&S.d_attn_i8, S.M * S.Q); cudaMalloc(&S.d_attn_i8s, S.M * (S.Q / 16) * 4);
    cudaMalloc(&S.d_gate, S.M * S.ID * 4); cudaMalloc(&S.d_up, S.M * S.ID * 4); cudaMalloc(&S.d_mlp, S.M * S.ID * 4);
    cudaMalloc(&S.d_mlp_i8, S.M * S.ID); cudaMalloc(&S.d_mlp_i8s, S.M * (S.ID / 16) * 4);
    cudaMalloc(&S.d_proj, S.M * S.H * 4);
    cudaMalloc(&S.d_logits, S.V * 4); cudaMalloc(&S.d_next_id, sizeof(int));
    cudaMalloc(&S.d_x, S.M * S.H * 4); cudaMalloc(&S.d_tmp_save, S.H * 4);

    size_t kv_sz = (size_t)S.NL * S.M * S.nkv * S.ms * S.hd * 4;
    cudaMalloc(&S.d_kc, kv_sz); cudaMalloc(&S.d_vc, kv_sz);
    cudaMemset(S.d_kc, 0, kv_sz); cudaMemset(S.d_vc, 0, kv_sz);
    // Get device-side seq_pos from library (for graph-compatible RoPE)
    int* tmp_ptr = nullptr; blackwell::kernels::get_seq_pos_device_ptr(&tmp_ptr); S.d_seq_pos = tmp_ptr;
    // Initialize to 0
    int zero = 0;
    cudaMemcpyAsync(S.d_seq_pos, &zero, sizeof(int), cudaMemcpyHostToDevice, S.st);
    S.kv_stride = (size_t)S.nkv * S.hd * S.ms;

    // Repetition penalty buffers
    cudaMalloc(&S.d_recent_tokens, S.M * MAX_RECENT * sizeof(int));
    cudaMemset(S.d_recent_tokens, 0, S.M * MAX_RECENT * sizeof(int));
    S.max_recent = MAX_RECENT;
    S.repetition_penalty = 1.0f;  // default: no penalty

    fprintf(stderr, "Ready.\n");

    // ── Main request loop ──
    while (true) {
        std::string line = read_stdin_line();
        if (line.empty()) break;

        auto prompts = parse_prompt_ids(line);
        if (prompts.empty()) prompts = parse_string_prompts(line, tokenizer);
        if (prompts.empty()) continue;

        int M = (int)prompts.size();
        if (M > 8) { M = 8; prompts.resize(8); }
        S.M = M;
        int max_tokens = find_int(line, "max_tokens", 30);
        float temperature = find_float(line, "temperature", 0);
        int top_k = find_int(line, "top_k", 0);
        float rep_pen = find_float(line, "repetition_penalty", 1.0f);
        if (rep_pen < 1.0f) rep_pen = 1.0f;
        if (rep_pen > 2.0f) rep_pen = 2.0f;
        S.repetition_penalty = rep_pen;
        bool stream = find_int(line, "stream", 0) == 1;

        int gen_start = (int)prompts[0].size();
        // Prefill: load all M tokens if they fit in batch size, else token-by-token
        if (gen_start > 0 && gen_start <= M) {
            embed_batch(S, prompts, 0);  // load all M tokens' embeddings to d_residual[m]
            batched_prefill(S, gen_start);
            cudaStreamSynchronize(S.st);  // ensure prefill done before gen loop
            // Copy last token's hidden state to d_residual[0] for decode continuation
            if (gen_start > 1) {
                cudaMemcpyAsync(S.d_residual[0], S.d_residual[gen_start - 1], S.H * 4,
                    cudaMemcpyDeviceToDevice, S.st);
            }
        } else if (gen_start > M) {
            // gen_start > M: fall back to token-by-token decode (no batched prefill overflow)
            for (int s = 0; s < gen_start; s++) {
                embed_batch(S, prompts, s);
                batched_decode_step(S, s);
            }
        }

        // Generate (after prefill, start from position gen_start)
        // For M=1: batched path (single batched_decode_step call per token)
        // For M>1: sequential per-item path (avoids KV cache position conflicts)
        std::vector<std::vector<uint32_t>> outputs(M);
        // Repetition penalty: host-side recent tokens buffer
        std::vector<int> h_recent(MAX_RECENT, 0);
        int recent_head = 0;
        int num_recent = 0;

        if (M == 1) {
            // Batched decode: one batched_decode_step call per token
            for (int s = gen_start; s < gen_start + max_tokens; s++) {
                blackwell::kernels::fused_rmsnorm(S.d_residual[0], S.d_residual[0], S.d_fn, S.H, S.eps, S.st);
                blackwell::kernels::quantize_int8(S.d_xi8, S.d_xi8s, S.d_residual[0], S.H, S.st);
                blackwell::kernels::gemv_int8_warp(S.d_logits, S.d_xi8, S.d_xi8s,
                    (S.lm_head.d ? S.lm_head.d : S.emb.d), (S.lm_head.d ? S.lm_head.sc : S.emb.sc), S.H, S.V, S.st);
                // Apply repetition penalty before sampling
                if (S.repetition_penalty > 1.0f && num_recent > 0) {
                    cudaMemcpy(S.d_recent_tokens, h_recent.data(), num_recent * sizeof(int), cudaMemcpyHostToDevice);
                    blackwell::kernels::apply_repetition_penalty(S.d_logits, S.d_recent_tokens, num_recent, S.repetition_penalty, S.V, S.st);
                }
                blackwell::kernels::sample_gpu(S.d_logits, S.V, temperature, top_k,
                    S.d_next_id, 0xdeadbeefLL, s, S.st);
                uint32_t next_id;
                cudaMemcpy(&next_id, S.d_next_id, sizeof(int), cudaMemcpyDeviceToHost);
                // Update recent tokens (circular buffer)
                if (recent_head >= MAX_RECENT) recent_head = 0;
                h_recent[recent_head++] = (int)next_id;
                if (num_recent < MAX_RECENT) num_recent++;
                outputs[0].push_back(next_id);
                if (stream) {
                    std::string tok_txt = tokenizer.decode(next_id);
                    printf("data: {\x22token\x22:%u,\x22text\x22:\x22", next_id); fflush(stdout);
                    for (size_t i = 0; i < tok_txt.size(); i++) {
                        char c = tok_txt[i];
                        if (c == '"') printf("\\\"");
                        else if (c == '\\') printf("\\\\");
                        else if (c == '\n') printf("\\n");
                        else if (c == '\r') printf("\\r");
                        else printf("%c", c);
                    }
                    printf("}\n\n"); fflush(stdout);
                }
                if (next_id == 151643) break;
                std::vector<float> h_hidden(S.H);
                for (int d = 0; d < S.H; d++)
                    h_hidden[d] = (float)S.h_emb_int8[next_id * S.H + d] * S.h_emb_scale[next_id * (S.H / 16) + d / 16];
                cudaMemcpy(S.d_residual[0], h_hidden.data(), S.H * 4, cudaMemcpyHostToDevice);
                batched_decode_step(S, s);
            }
        } else {
            // Sequential per-item generation (M>1): each item's tokens processed
            // sequentially through all layers. Avoids KV cache position conflicts.
            std::vector<int> item_pos(M, gen_start);
            std::vector<bool> item_done(M, false);
            std::vector<std::vector<int>> item_recent(M, std::vector<int>(MAX_RECENT, 0));
            std::vector<int> item_recent_head(M, 0);
            std::vector<int> item_num_recent(M, 0);
            std::vector<int> h_recent(MAX_RECENT);

            for (int t = 0; t < max_tokens; t++) {
                for (int m = 0; m < M; m++) {
                    if (item_done[m]) continue;
                    blackwell::kernels::fused_rmsnorm(S.d_residual[m], S.d_residual[m], S.d_fn, S.H, S.eps, S.st);
                    blackwell::kernels::quantize_int8(S.d_xi8, S.d_xi8s, S.d_residual[m], S.H, S.st);
                    blackwell::kernels::gemv_int8_warp(S.d_logits, S.d_xi8, S.d_xi8s,
                        (S.lm_head.d ? S.lm_head.d : S.emb.d), (S.lm_head.d ? S.lm_head.sc : S.emb.sc), S.H, S.V, S.st);
                    // Apply repetition penalty for this sequence
                    if (S.repetition_penalty > 1.0f && item_num_recent[m] > 0) {
                        for (int i = 0; i < item_num_recent[m]; i++)
                            h_recent[i] = item_recent[m][i];
                        cudaMemcpy(S.d_recent_tokens, h_recent.data(), item_num_recent[m] * sizeof(int), cudaMemcpyHostToDevice);
                        blackwell::kernels::apply_repetition_penalty(S.d_logits, S.d_recent_tokens, item_num_recent[m], S.repetition_penalty, S.V, S.st);
                    }
                    blackwell::kernels::sample_gpu(S.d_logits, S.V, temperature, top_k,
                        S.d_next_id, 0xdeadbeefLL, item_pos[m], S.st);
                    uint32_t next_id;
                    cudaMemcpy(&next_id, S.d_next_id, sizeof(int), cudaMemcpyDeviceToHost);
                    // Update recent tokens for this sequence
                    if (item_recent_head[m] >= MAX_RECENT) item_recent_head[m] = 0;
                    item_recent[m][item_recent_head[m]++] = (int)next_id;
                    if (item_num_recent[m] < MAX_RECENT) item_num_recent[m]++;
                    outputs[m].push_back(next_id);
                    if (stream) {
                        std::string tok_txt = tokenizer.decode(next_id);
                        printf("data: {\x22token\x22:%u,\x22text\x22:\x22", next_id); fflush(stdout);
                        for (size_t i = 0; i < tok_txt.size(); i++) {
                            char c = tok_txt[i];
                            if (c == '"') printf("\\\"");
                            else if (c == '\\') printf("\\\\");
                            else if (c == '\n') printf("\\n");
                            else if (c == '\r') printf("\\r");
                            else printf("%c", c);
                        }
                        printf("}\n\n"); fflush(stdout);
                    }
                    if (next_id == 151643) { item_done[m] = true; continue; }
                    std::vector<float> h_hidden(S.H);
                    for (int d = 0; d < S.H; d++)
                        h_hidden[d] = (float)S.h_emb_int8[next_id * S.H + d] * S.h_emb_scale[next_id * (S.H / 16) + d / 16];
                    cudaMemcpy(S.d_residual[0], h_hidden.data(), S.H * 4, cudaMemcpyHostToDevice);
                    int orig_M = S.M; S.M = 1;
                    batched_decode_step(S, item_pos[m]);
                    S.M = orig_M;
                    cudaMemcpyAsync(S.d_residual[m], S.d_residual[0], S.H * 4, cudaMemcpyDeviceToDevice, S.st);
                    item_pos[m]++;
                }
                bool all_done = true;
                for (int m = 0; m < M; m++) if (!item_done[m]) { all_done = false; break; }
                if (all_done) break;
            }
        }

        if (stream) {
            printf("data: [DONE]\n\n"); fflush(stdout);
        } else {
            server_write_results(outputs, tokenizer);
        }

        cudaMemset(S.d_kc, 0, kv_sz); cudaMemset(S.d_vc, 0, kv_sz);
    }
    return 0;
}
