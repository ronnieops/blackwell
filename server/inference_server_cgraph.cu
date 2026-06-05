// server/inference_server_cgraph.cu — Blackwell INT8 server with CUDA Graph
//
// CUDA Graph captures the full 28-layer decode step. Replayed per token.
// Pre-normalizes: uses update_decode_seq_pos() for device-side seq_pos.
// Uses update_kv_cache_device (no H2D copy in capture).
//
// Build:
//   CUDACXX=/usr/local/cuda-13.3/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include server/inference_server_cgraph.cu build/libblackwell_kernels.a \
//     -o server/inference_server_cgraph

#include <cuda_runtime.h>
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

// ── Inline kernels (must be graph-compatible, no H2D memcpy in capture) ──

__global__ void head_norm_kernel(float* data, const float* weight, int nh, int hd, float eps) {
    int h = blockIdx.x; if (h >= nh) return;
    float* d = data + h * hd;
    __shared__ float wp[4];
    float s = 0; int tid = threadIdx.x;
    for (int i = tid; i < hd; i += blockDim.x) s += d[i] * d[i];
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
    if ((tid & 31) == 0) wp[tid >> 5] = s; __syncthreads();
    if (tid < 4) s = wp[tid]; else s = 0;
    for (int off = 2; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
    if (tid == 0) wp[0] = rsqrtf(s / hd + eps); __syncthreads();
    float is = wp[0]; for (int i = tid; i < hd; i += blockDim.x) d[i] = d[i] * is * weight[i];
}

// RoPE kernel — reads seq_pos from library's device pointer (set by update_decode_seq_pos)
// Both the kernel AND attention_decode_gqa read from the same device pointer.
// update_decode_seq_pos's captured cudaMemcpyAsync replays each graph launch → correct seq_pos.
__global__ void rope_kernel(float* data, int n_heads, int head_dim, int* d_seq_pos) {
    int h = blockIdx.x; int d = threadIdx.x;
    if (h >= n_heads || d >= head_dim / 2) return;
    int pos = (threadIdx.x == 0 && blockIdx.x == 0) ? *d_seq_pos : 0;
    pos = __shfl_sync(0xffffffff, pos, 0);
    const float rope_theta = 1000000.0f;
    float theta = (float)pos * powf(rope_theta, -2.0f * (float)d / (float)head_dim);
    float c = cosf(theta), s = sinf(theta);
    int i2 = d * 2;
    float* pair = data + h * head_dim + i2;
    float x = pair[0], y = pair[1];
    pair[0] = x * c - y * s;
    pair[1] = x * s + y * c;
}

// ── Weight structures ──────────────────────────────────────────────────
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

struct LW { DevW q, k, v, o, gate, up, down; float *qn, *kn; };

struct ServerState {
    int NL, H, Q, KV, ID, nqh, nkv, hd, ms, V;
    float eps;

    std::vector<LW> layers;
    std::vector<float*> d_rn_in, d_rn_post;
    float* d_fn;
    DevW emb;
    int8_t* h_emb_int8; float* h_emb_scale;

    // Working buffers (M=1 target for CUDA Graph)
    float* d_res;
    int8_t* d_xi8; float* d_xi8s;
    float *d_Q, *d_K, *d_V, *d_attn;
    int8_t *d_ai; float *d_as;
    float *d_gate, *d_up, *d_mlp;
    int8_t *d_mi; float *d_ms;
    float *d_proj;
    float *d_logits;
    int* d_next_id;
    int* d_seq_pos; // device-side seq_pos for graph-compatible RoPE

    float* d_kc, *d_vc;
    size_t kv_stride;

    // CUDA Graph
    cudaStream_t graph_stream;
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    bool graph_ready;

    cudaStream_t st;
};

// ── JSON helpers ──────────────────────────────────────────────────────
static std::string read_stdin_line() {
    std::string line; int c;
    while ((c = getchar()) != EOF && c != '\n') line.push_back((char)c);
    return line;
}

static std::vector<std::vector<uint32_t>> parse_prompt_ids(const std::string& json) {
    std::vector<std::vector<uint32_t>> result;
    const char* p = strstr(json.c_str(), "\"prompts\"");
    if (!p) return result;
    p = strchr(p, '['); if (!p) return result; p++;
    while (*p && *p != ']') {
        while (*p && *p != '[') p++;
        if (!*p || *p == ']') break; p++;
        std::vector<uint32_t> ids;
        while (*p && *p != ']') {
            while (*p && (*p == ' ' || *p == ',' || *p == '\n' || *p == '\r')) p++;
            if (*p == ']') break;
            long val = strtol(p, (char**)&p, 10);
            ids.push_back((uint32_t)val);
        }
        if (*p == ']') p++;
        result.push_back(ids);
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

static std::vector<std::vector<uint32_t>> parse_string_prompts(
    const std::string& json, blackwell::BpeTokenizer& tokenizer) {
    std::vector<std::vector<uint32_t>> result;
    const char* p = strstr(json.c_str(), "\"prompts\"");
    if (!p) return result;
    p = strchr(p, '['); if (!p) return result; p++;
    while (*p && *p != ']') {
        while (*p && (*p == ' ' || *p == '\t' || *p == '\n')) p++;
        if (*p == ']' || !*p) break;
        if (*p != '"') { p++; continue; } p++;
        std::string s;
        while (*p && *p != '"') {
            if (*p == '\\' && *(p + 1) == '\"') { s += '"'; p += 2; }
            else if (*p == '\\' && *(p + 1) == 'n') { s += '\n'; p += 2; }
            else if (*p == '\\' && *(p + 1) == 'r') { s += '\r'; p += 2; }
            else if (*p == '\\' && *(p + 1) == 't') { s += '\t'; p += 2; }
            else if (*p == '\\' && *(p + 1) == '\\') { s += '\\'; p += 2; }
            else { s += *p; p++; }
        }
        if (*p == '"') p++;
        auto ids = tokenizer.encode(s);
        result.push_back(ids);
        while (*p && (*p == ',' || *p == ' ' || *p == '\t' || *p == '\n')) p++;
    }
    return result;
}

static void write_results(const std::vector<std::vector<uint32_t>>& tokens,
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

// ── Embed: CPU-side dequant → device (no graph capture) ─────────────────
static void embed_token(ServerState& S, uint32_t token_id) {
    std::vector<float> h_hidden(S.H);
    for (int d = 0; d < S.H; d++)
        h_hidden[d] = (float)S.h_emb_int8[token_id * S.H + d] * S.h_emb_scale[token_id * (S.H / 16) + d / 16];
    cudaMemcpy(S.d_res, h_hidden.data(), S.H * 4, cudaMemcpyHostToDevice);
}

// ── Build CUDA Graph for one decode step (28 layers) ──────────────────────
static void build_decode_graph(ServerState& S) {
    if (S.graph_ready) {
        cudaGraphExecDestroy(S.graph_exec);
        cudaGraphDestroy(S.graph);
    }

    fprintf(stderr, "  Building CUDA Graph (28 layers)...\n");
    fflush(stderr);

    cudaStreamBeginCapture(S.graph_stream, cudaStreamCaptureModeGlobal);

    // 28-layer decode loop — all kernels captured
    for (int l = 0; l < S.NL; l++) {
        size_t kb = (size_t)l * S.nkv * S.hd * S.ms;

        // Input RMSNorm + quantize
        blackwell::kernels::fused_rmsnorm_quant_int8(
            S.d_xi8, S.d_xi8s, S.d_res, S.d_rn_in[l], S.H, S.eps, S.graph_stream);

        // Q/K/V projections
        blackwell::kernels::gemv_int8_warp(S.d_Q, S.d_xi8, S.d_xi8s,
            S.layers[l].q.d, S.layers[l].q.sc, S.H, S.Q, S.graph_stream);
        blackwell::kernels::gemv_int8_warp(S.d_K, S.d_xi8, S.d_xi8s,
            S.layers[l].k.d, S.layers[l].k.sc, S.H, S.KV, S.graph_stream);
        blackwell::kernels::gemv_int8_warp(S.d_V, S.d_xi8, S.d_xi8s,
            S.layers[l].v.d, S.layers[l].v.sc, S.H, S.KV, S.graph_stream);

        // Head norms
        head_norm_kernel<<<S.nqh, 128, 0, S.graph_stream>>>(S.d_Q, S.layers[l].qn, S.nqh, S.hd, S.eps);
        head_norm_kernel<<<S.nkv, 128, 0, S.graph_stream>>>(S.d_K, S.layers[l].kn, S.nkv, S.hd, S.eps);

        // RoPE — reads d_seq_pos (set by update_decode_seq_pos before launch)
        rope_kernel<<<S.nqh, S.hd / 2, 0, S.graph_stream>>>(S.d_Q, S.nqh, S.hd, S.d_seq_pos);
        rope_kernel<<<S.nkv, S.hd / 2, 0, S.graph_stream>>>(S.d_K, S.nkv, S.hd, S.d_seq_pos);

        // KV cache write (device-side seq_pos)
        blackwell::kernels::update_kv_cache_device(
            S.d_kc + kb, S.d_vc + kb, S.d_K, S.d_V,
            0, S.d_seq_pos, S.nkv, S.hd, S.ms, S.graph_stream);

        // Attention
        blackwell::kernels::attention_decode_gqa(
            S.d_attn, S.d_Q, S.d_kc + kb, S.d_vc + kb,
            0 /* placeholder — reads d_seq_pos */, S.nqh, S.nkv, S.hd, S.ms, S.graph_stream);

        // Attention output projection + residual
        blackwell::kernels::quantize_int8(S.d_ai, S.d_as, S.d_attn, S.Q, S.graph_stream);
        blackwell::kernels::gemv_int8_warp(S.d_proj, S.d_ai, S.d_as,
            S.layers[l].o.d, S.layers[l].o.sc, S.Q, S.H, S.graph_stream);
        blackwell::kernels::vector_add_fp32(S.d_proj, S.d_proj, S.d_res, S.H, S.graph_stream);
        // Save for MLP residual
        cudaMemcpyAsync(S.d_res, S.d_proj, S.H * 4, cudaMemcpyDeviceToDevice, S.graph_stream);

        // Post-attention RMSNorm + quantize
        blackwell::kernels::fused_rmsnorm_quant_int8(
            S.d_xi8, S.d_xi8s, S.d_proj, S.d_rn_post[l], S.H, S.eps, S.graph_stream);

        // MLP gate + up
        blackwell::kernels::gemv_int8_warp(S.d_gate, S.d_xi8, S.d_xi8s,
            S.layers[l].gate.d, S.layers[l].gate.sc, S.H, S.ID, S.graph_stream);
        blackwell::kernels::gemv_int8_warp(S.d_up, S.d_xi8, S.d_xi8s,
            S.layers[l].up.d, S.layers[l].up.sc, S.H, S.ID, S.graph_stream);
        blackwell::kernels::apply_swiglu(S.d_mlp, S.d_gate, S.d_up, S.ID, S.graph_stream);

        // MLP down
        blackwell::kernels::quantize_int8(S.d_mi, S.d_ms, S.d_mlp, S.ID, S.graph_stream);
        blackwell::kernels::gemv_int8_warp(S.d_proj, S.d_mi, S.d_ms,
            S.layers[l].down.d, S.layers[l].down.sc, S.ID, S.H, S.graph_stream);

        // Residual 2
        blackwell::kernels::vector_add_fp32(S.d_proj, S.d_proj, S.d_res, S.H, S.graph_stream);
        cudaMemcpyAsync(S.d_res, S.d_proj, S.H * 4, cudaMemcpyDeviceToDevice, S.graph_stream);
    }

    cudaError_t e = cudaStreamEndCapture(S.graph_stream, &S.graph);
    if (e != cudaSuccess) { fprintf(stderr, "FAIL cudaStreamEndCapture: %s\n", cudaGetErrorString(e)); exit(1); }

    e = cudaGraphInstantiate(&S.graph_exec, S.graph, NULL, NULL, 0);
    if (e != cudaSuccess) { fprintf(stderr, "FAIL cudaGraphInstantiate: %s\n", cudaGetErrorString(e)); exit(1); }

    S.graph_ready = true;
    fprintf(stderr, "  Graph built.\n");
}

// ── Prefill: plain per-kernel (no graph capture) ──────────────────────────────
static void prefill(ServerState& S, const std::vector<uint32_t>& tokens) {
    for (size_t s = 0; s < tokens.size(); s++) {
        // Update seq_pos before kernel
        blackwell::kernels::update_decode_seq_pos((int)s, S.st);
        cudaStreamSynchronize(S.st);

        // Embed
        embed_token(S, tokens[s]);

        // Plain per-kernel decode (no capture)
        for (int l = 0; l < S.NL; l++) {
            size_t kb = (size_t)l * S.nkv * S.hd * S.ms;

            blackwell::kernels::fused_rmsnorm_quant_int8(
                S.d_xi8, S.d_xi8s, S.d_res, S.d_rn_in[l], S.H, S.eps, S.st);
            blackwell::kernels::gemv_int8_warp(S.d_Q, S.d_xi8, S.d_xi8s,
                S.layers[l].q.d, S.layers[l].q.sc, S.H, S.Q, S.st);
            blackwell::kernels::gemv_int8_warp(S.d_K, S.d_xi8, S.d_xi8s,
                S.layers[l].k.d, S.layers[l].k.sc, S.H, S.KV, S.st);
            blackwell::kernels::gemv_int8_warp(S.d_V, S.d_xi8, S.d_xi8s,
                S.layers[l].v.d, S.layers[l].v.sc, S.H, S.KV, S.st);
            head_norm_kernel<<<S.nqh, 128, 0, S.st>>>(S.d_Q, S.layers[l].qn, S.nqh, S.hd, S.eps);
            head_norm_kernel<<<S.nkv, 128, 0, S.st>>>(S.d_K, S.layers[l].kn, S.nkv, S.hd, S.eps);
            rope_kernel<<<S.nqh, S.hd / 2, 0, S.st>>>(S.d_Q, S.nqh, S.hd, S.d_seq_pos);
            rope_kernel<<<S.nkv, S.hd / 2, 0, S.st>>>(S.d_K, S.nkv, S.hd, S.d_seq_pos);
            blackwell::kernels::update_kv_cache_device(
                S.d_kc + kb, S.d_vc + kb, S.d_K, S.d_V,
                0, S.d_seq_pos, S.nkv, S.hd, S.ms, S.st);
            blackwell::kernels::attention_decode_gqa(
                S.d_attn, S.d_Q, S.d_kc + kb, S.d_vc + kb, 0, S.nqh, S.nkv, S.hd, S.ms, S.st);
            blackwell::kernels::quantize_int8(S.d_ai, S.d_as, S.d_attn, S.Q, S.st);
            blackwell::kernels::gemv_int8_warp(S.d_proj, S.d_ai, S.d_as,
                S.layers[l].o.d, S.layers[l].o.sc, S.Q, S.H, S.st);
            blackwell::kernels::vector_add_fp32(S.d_proj, S.d_proj, S.d_res, S.H, S.st);
            cudaMemcpyAsync(S.d_res, S.d_proj, S.H * 4, cudaMemcpyDeviceToDevice, S.st);
            blackwell::kernels::fused_rmsnorm_quant_int8(
                S.d_xi8, S.d_xi8s, S.d_proj, S.d_rn_post[l], S.H, S.eps, S.st);
            blackwell::kernels::gemv_int8_warp(S.d_gate, S.d_xi8, S.d_xi8s,
                S.layers[l].gate.d, S.layers[l].gate.sc, S.H, S.ID, S.st);
            blackwell::kernels::gemv_int8_warp(S.d_up, S.d_xi8, S.d_xi8s,
                S.layers[l].up.d, S.layers[l].up.sc, S.H, S.ID, S.st);
            blackwell::kernels::apply_swiglu(S.d_mlp, S.d_gate, S.d_up, S.ID, S.st);
            blackwell::kernels::quantize_int8(S.d_mi, S.d_ms, S.d_mlp, S.ID, S.st);
            blackwell::kernels::gemv_int8_warp(S.d_proj, S.d_mi, S.d_ms,
                S.layers[l].down.d, S.layers[l].down.sc, S.ID, S.H, S.st);
            blackwell::kernels::vector_add_fp32(S.d_proj, S.d_proj, S.d_res, S.H, S.st);
            cudaMemcpyAsync(S.d_res, S.d_proj, S.H * 4, cudaMemcpyDeviceToDevice, S.st);
        }
    }
}

// ── Main ──────────────────────────────────────────────────────────────
int main() {
    ServerState S;
    S.NL = 28; S.H = 2048; S.Q = 2048; S.KV = 1024; S.ID = 6144;
    S.nqh = 16; S.nkv = 8; S.hd = 128; S.ms = 2048; S.V = 151936;
    S.eps = 1e-6f; S.graph_ready = false;

    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    fprintf(stderr, "Blackwell INT8 Server v0.5.0 (CUDA Graph)\n");
    fprintf(stderr, "Device: %s (CC %d.%d)\n", p.name, p.major, p.minor);

    die(cudaStreamCreate(&S.st), "stream st");
    die(cudaStreamCreate(&S.graph_stream), "stream graph");

    // Device-side seq_pos for graph-compatible RoPE
    die(blackwell::kernels::get_seq_pos_device_ptr(&S.d_seq_pos), "get_seq_pos_device_ptr");
    fprintf(stderr, "  d_seq_pos device ptr: %p\n", (void*)S.d_seq_pos);

    blackwell::BpeTokenizer tokenizer;
    if (tokenizer.load("tokenizer_data.bin") != 0) {
        fprintf(stderr, "FAIL: no tokenizer_data.bin\n"); return 1;
    }

    // ── Load weights ──
    fprintf(stderr, "Loading weights...\n");
    S.layers.resize(S.NL);
    for (int l = 0; l < S.NL; l++) {
        char p[256];
        snprintf(p, 256, "weights_int8_bf16/%d_self_attn.q_proj", l); S.layers[l].q = upload_int8(p);
        snprintf(p, 256, "weights_int8_bf16/%d_self_attn.k_proj", l); S.layers[l].k = upload_int8(p);
        snprintf(p, 256, "weights_int8_bf16/%d_self_attn.v_proj", l); S.layers[l].v = upload_int8(p);
        snprintf(p, 256, "weights_int8_bf16/%d_self_attn.o_proj", l); S.layers[l].o = upload_int8(p);
        snprintf(p, 256, "weights_int8_bf16/%d_mlp.gate_proj", l);  S.layers[l].gate = upload_int8(p);
        snprintf(p, 256, "weights_int8_bf16/%d_mlp.up_proj", l);    S.layers[l].up = upload_int8(p);
        snprintf(p, 256, "weights_int8_bf16/%d_mlp.down_proj", l);  S.layers[l].down = upload_int8(p);
        if (l % 7 == 0) fprintf(stderr, "  layer %d/%d\n", l, S.NL);
    }
    S.emb = upload_int8("weights_int8_bf16/embed_tokens");

    // Host-side embed
    {
        char p[256]; snprintf(p, 256, "weights_int8_bf16/embed_tokens.int8_t");
        FILE* f = fopen(p, "rb"); int h[5]; (void)fread(h, 4, 5, f);
        size_t num = (size_t)h[0] * h[1];
        S.h_emb_int8 = (int8_t*)malloc(num); (void)fread(S.h_emb_int8, 1, num, f); fclose(f);
        snprintf(p, 256, "weights_int8_bf16/embed_tokens.scale_t");
        f = fopen(p, "rb"); (void)fread(h, 4, 5, f);
        size_t ns = (size_t)h[3] * h[4];
        S.h_emb_scale = (float*)malloc(ns * 4); (void)fread(S.h_emb_scale, 4, ns, f); fclose(f);
    }

    // Per-layer RMSNorm
    S.d_rn_in.resize(S.NL); S.d_rn_post.resize(S.NL);
    for (int l = 0; l < S.NL; l++) {
        float* w = (float*)malloc(S.H * 4);
        char p[256];
        snprintf(p, 256, "weights_int8_bf16/%d_input_layernorm.f32", l);
        FILE* f = fopen(p, "rb"); (void)fread(w, 4, S.H, f); fclose(f);
        cudaMalloc(&S.d_rn_in[l], S.H * 4); cudaMemcpy(S.d_rn_in[l], w, S.H * 4, cudaMemcpyHostToDevice);
        snprintf(p, 256, "weights_int8_bf16/%d_post_attention_layernorm.f32", l);
        f = fopen(p, "rb"); (void)fread(w, 4, S.H, f); fclose(f);
        cudaMalloc(&S.d_rn_post[l], S.H * 4); cudaMemcpy(S.d_rn_post[l], w, S.H * 4, cudaMemcpyHostToDevice);
        free(w);
    }

    // Per-layer Q/K norms
    {
        float* qk_h = (float*)malloc(28 * 2 * 128 * 4);
        FILE* f = fopen("weights_int8_bf16/qk_norms.f32", "rb");
        (void)fread(qk_h, 4, 28 * 2 * 128, f); fclose(f);
        for (int l = 0; l < S.NL; l++) {
            cudaMalloc(&S.layers[l].qn, 128 * 4);
            cudaMemcpy(S.layers[l].qn, qk_h + l * 2 * 128, 128 * 4, cudaMemcpyHostToDevice);
            cudaMalloc(&S.layers[l].kn, 128 * 4);
            cudaMemcpy(S.layers[l].kn, qk_h + l * 2 * 128 + 128, 128 * 4, cudaMemcpyHostToDevice);
        }
        free(qk_h);
    }

    // Final norm
    {
        float* w = (float*)malloc(S.H * 4);
        FILE* f = fopen("weights_int8_bf16/final_norm.f32", "rb");
        (void)fread(w, 4, S.H, f); fclose(f);
        cudaMalloc(&S.d_fn, S.H * 4); cudaMemcpy(S.d_fn, w, S.H * 4, cudaMemcpyHostToDevice);
        free(w);
    }

    // Allocate buffers
    cudaMalloc(&S.d_res, S.H * 4);
    cudaMalloc(&S.d_xi8, S.H); cudaMalloc(&S.d_xi8s, S.H / 16 * 4);
    cudaMalloc(&S.d_Q, S.Q * 4); cudaMalloc(&S.d_K, S.KV * 4); cudaMalloc(&S.d_V, S.KV * 4);
    cudaMalloc(&S.d_attn, S.Q * 4);
    cudaMalloc(&S.d_ai, S.Q); cudaMalloc(&S.d_as, S.Q / 16 * 4);
    cudaMalloc(&S.d_gate, S.ID * 4); cudaMalloc(&S.d_up, S.ID * 4); cudaMalloc(&S.d_mlp, S.ID * 4);
    cudaMalloc(&S.d_mi, S.ID); cudaMalloc(&S.d_ms, S.ID / 16 * 4);
    cudaMalloc(&S.d_proj, S.H * 4);
    cudaMalloc(&S.d_logits, S.V * 4); cudaMalloc(&S.d_next_id, sizeof(int));

    size_t kv_sz = (size_t)S.NL * S.nkv * S.ms * S.hd * 4;
    cudaMalloc(&S.d_kc, kv_sz); cudaMalloc(&S.d_vc, kv_sz);
    cudaMemset(S.d_kc, 0, kv_sz); cudaMemset(S.d_vc, 0, kv_sz);
    S.kv_stride = (size_t)S.nkv * S.hd * S.ms;

    fprintf(stderr, "  done\n");
    fprintf(stderr, "Building decode graph (28 layers, ~560 kernels)...\n");
    fflush(stderr);

    build_decode_graph(S);

    fprintf(stderr, "Ready.\n");

    // ── Main request loop ──
    while (true) {
        std::string line = read_stdin_line();
        if (line.empty()) break;

        auto prompts = parse_prompt_ids(line);
        if (prompts.empty()) prompts = parse_string_prompts(line, tokenizer);
        if (prompts.empty()) continue;

        int M = (int)prompts.size();
        if (M > 1) { fprintf(stderr, "Warning: batch >1, using M=1\n"); }
        M = 1; // CUDA Graph version supports M=1 only

        int max_tokens = find_int(line, "max_tokens", 30);
        float temperature = find_float(line, "temperature", 0);
        int top_k = find_int(line, "top_k", 0);

        // Prefill
        int gen_start = (int)prompts[0].size();
        prefill(S, std::vector<uint32_t>(prompts[0].begin(), prompts[0].end()));
        cudaStreamSynchronize(S.st);

        // Generate (PER-KERNEL, no graph — for correctness verification)
        std::vector<uint32_t> output;
        for (int step = gen_start; step < gen_start + max_tokens; step++) {
            // Embed next token
            uint32_t next_id = (step == gen_start) ? prompts[0][step] : output.back();
            embed_token(S, next_id);
            cudaStreamSynchronize(S.st);

            // Per-kernel decode (matching nofp4 benchmark exactly)
            for (int l = 0; l < S.NL; l++) {
                size_t kb = (size_t)l * S.nkv * S.hd * S.ms;

                blackwell::kernels::fused_rmsnorm_quant_int8(
                    S.d_xi8, S.d_xi8s, S.d_res, S.d_rn_in[l], S.H, S.eps, S.st);
                blackwell::kernels::gemv_int8_warp(S.d_Q, S.d_xi8, S.d_xi8s,
                    S.layers[l].q.d, S.layers[l].q.sc, S.H, S.Q, S.st);
                blackwell::kernels::gemv_int8_warp(S.d_K, S.d_xi8, S.d_xi8s,
                    S.layers[l].k.d, S.layers[l].k.sc, S.H, S.KV, S.st);
                blackwell::kernels::gemv_int8_warp(S.d_V, S.d_xi8, S.d_xi8s,
                    S.layers[l].v.d, S.layers[l].v.sc, S.H, S.KV, S.st);
                head_norm_kernel<<<S.nqh, 128, 0, S.st>>>(S.d_Q, S.layers[l].qn, S.nqh, S.hd, S.eps);
                head_norm_kernel<<<S.nkv, 128, 0, S.st>>>(S.d_K, S.layers[l].kn, S.nkv, S.hd, S.eps);

                // RoPE: update seq_pos and sync, then apply rotation
                blackwell::kernels::update_decode_seq_pos(step, S.st);
                cudaStreamSynchronize(S.st);
                rope_kernel<<<S.nqh, S.hd / 2, 0, S.st>>>(S.d_Q, S.nqh, S.hd, S.d_seq_pos);
                rope_kernel<<<S.nkv, S.hd / 2, 0, S.st>>>(S.d_K, S.nkv, S.hd, S.d_seq_pos);

                blackwell::kernels::update_kv_cache_device(
                    S.d_kc + kb, S.d_vc + kb, S.d_K, S.d_V,
                    0, S.d_seq_pos, S.nkv, S.hd, S.ms, S.st);
                blackwell::kernels::attention_decode_gqa(
                    S.d_attn, S.d_Q, S.d_kc + kb, S.d_vc + kb,
                    0, S.nqh, S.nkv, S.hd, S.ms, S.st);
                blackwell::kernels::quantize_int8(S.d_ai, S.d_as, S.d_attn, S.Q, S.st);
                blackwell::kernels::gemv_int8_warp(S.d_proj, S.d_ai, S.d_as,
                    S.layers[l].o.d, S.layers[l].o.sc, S.Q, S.H, S.st);
                blackwell::kernels::vector_add_fp32(S.d_proj, S.d_proj, S.d_res, S.H, S.st);
                cudaMemcpyAsync(S.d_res, S.d_proj, S.H * 4, cudaMemcpyDeviceToDevice, S.st);
                blackwell::kernels::fused_rmsnorm_quant_int8(
                    S.d_xi8, S.d_xi8s, S.d_proj, S.d_rn_post[l], S.H, S.eps, S.st);
                blackwell::kernels::gemv_int8_warp(S.d_gate, S.d_xi8, S.d_xi8s,
                    S.layers[l].gate.d, S.layers[l].gate.sc, S.H, S.ID, S.st);
                blackwell::kernels::gemv_int8_warp(S.d_up, S.d_xi8, S.d_xi8s,
                    S.layers[l].up.d, S.layers[l].up.sc, S.H, S.ID, S.st);
                blackwell::kernels::apply_swiglu(S.d_mlp, S.d_gate, S.d_up, S.ID, S.st);
                blackwell::kernels::quantize_int8(S.d_mi, S.d_ms, S.d_mlp, S.ID, S.st);
                blackwell::kernels::gemv_int8_warp(S.d_proj, S.d_mi, S.d_ms,
                    S.layers[l].down.d, S.layers[l].down.sc, S.ID, S.H, S.st);
                blackwell::kernels::vector_add_fp32(S.d_proj, S.d_proj, S.d_res, S.H, S.st);
                cudaMemcpyAsync(S.d_res, S.d_proj, S.H * 4, cudaMemcpyDeviceToDevice, S.st);
            }
            cudaStreamSynchronize(S.st);

            // Final norm + lm_head + sample
            blackwell::kernels::fused_rmsnorm(S.d_res, S.d_res, S.d_fn, S.H, S.eps, S.st);
            blackwell::kernels::quantize_int8(S.d_xi8, S.d_xi8s, S.d_res, S.H, S.st);
            blackwell::kernels::gemv_int8_warp(S.d_logits, S.d_xi8, S.d_xi8s,
                S.emb.d, S.emb.sc, S.H, S.V, S.st);
            blackwell::kernels::sample_gpu(S.d_logits, S.V, temperature, top_k,
                S.d_next_id, 0xdeadbeefLL, step, S.st);
            uint32_t token_id;
            cudaMemcpy(&token_id, S.d_next_id, sizeof(int), cudaMemcpyDeviceToHost);
            output.push_back(token_id);
            if (token_id == 151643) break;
        }

        // Write results
        std::vector<std::vector<uint32_t>> out(1);
        out[0] = output;
        write_results(out, tokenizer);
        cudaMemset(S.d_kc, 0, kv_sz); cudaMemset(S.d_vc, 0, kv_sz);
    }

    if (S.graph_ready) {
        cudaGraphExecDestroy(S.graph_exec);
        cudaGraphDestroy(S.graph);
    }
    return 0;
}