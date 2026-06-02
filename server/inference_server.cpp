// server/inference_server.cpp — Blackwell INT8 continuous batching inference server
//
// Persistent C++ daemon for the M=8 batched CUDA Graph decode path.
// Protocol: JSON lines on stdin/stdout.
//
// Input (stdin, one JSON line per batch):
//   {"prompts":[[id1,id2,...],...], "max_tokens":20, "temperature":0.0, "top_k":0}
//
// Output (stdout):
//   {"tokens":[[tok1,tok2,...],...]}
//
// Build:
//   CUDACXX=/usr/local/cuda-13.3/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include server/inference_server.cpp build/libblackwell_kernels.a \
//     -o server/inference_server
//
// Python server spawns this as subprocess and communicates via stdin/stdout.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <sstream>
#include "blackwell/kernels.h"
#include "blackwell/bpe_tokenizer.h"

static void die(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) { fprintf(stderr, "FAIL %s: %s\n", msg, cudaGetErrorString(e)); exit(1); }
}

static void check_stream(const char* msg) {
    cudaError_t e = cudaPeekAtLastError();
    if (e != cudaSuccess) { fprintf(stderr, "CUDA err after %s: %s\n", msg, cudaGetErrorString(e)); exit(1); }
}

// ── Weight loading (INT8) ────────────────────────────────────────────────
struct DevW { int K, N; int8_t* d; float* sc; };
static DevW upload_int8(const char* prefix) {
    char p[256]; snprintf(p,256,"%s.int8_t",prefix);
    FILE* f = fopen(p,"rb"); if(!f){fprintf(stderr,"Cannot open %s\n",p);exit(1);}
    int h[5]; fread(h,4,5,f);
    std::vector<int8_t> tmp((size_t)h[0]*h[1]); fread(tmp.data(),1,tmp.size(),f); fclose(f);
    DevW dw{h[0],h[1],nullptr,nullptr};
    cudaMalloc(&dw.d,(size_t)h[0]*h[1]); cudaMemcpy(dw.d,tmp.data(),dw.K*dw.N,cudaMemcpyHostToDevice);
    snprintf(p,256,"%s.scale_t",prefix); f=fopen(p,"rb"); if(!f){fprintf(stderr,"Cannot open %s\n",p);exit(1);}
    fread(h,4,5,f); size_t ns=(size_t)h[3]*h[4]; std::vector<float> ts(ns); fread(ts.data(),4,ns,f); fclose(f);
    cudaMalloc(&dw.sc,ns*4); cudaMemcpy(dw.sc,ts.data(),ns*4,cudaMemcpyHostToDevice);
    return dw;
}
static DevW upload_f32(const char* prefix) {
    char p[256]; snprintf(p,256,"%s.f32",prefix);
    FILE* f = fopen(p,"rb"); if(!f){fprintf(stderr,"Cannot open %s\n",p);exit(1);}
    int h[5]; fread(h,4,5,f);
    size_t n = (size_t)h[0]; std::vector<float> tmp(n); fread(tmp.data(),4,n,f); fclose(f);
    DevW dw{(int)n,1,nullptr,nullptr};
    cudaMalloc(&dw.d,n*4); cudaMemcpy(dw.d,tmp.data(),n*4,cudaMemcpyHostToDevice);
    cudaMalloc(&dw.sc,4); float s=1.f; cudaMemcpy(dw.sc,&s,4,cudaMemcpyHostToDevice);
    return dw;
}

struct LW { DevW q,k,v,o,gate,up,down; };
struct ServerState {
    // Model params
    int NL, H, Q, KV, ID, nqh, nkv, hd, ms, V;
    float eps;

    // Weights (GPU)
    std::vector<LW> layers;
    DevW emb;      // INT8 embed_tokens [V × H] (used for embed + lm_head weight tying)
    // Host-side copies for CPU-side dequant of embed rows (GPU ptr can't be dereferenced from host)
    int8_t *h_emb_int8;
    float *h_emb_scale;
    float *d_fn;   // Final RMSNorm weight [H], loaded via upload_f32

    // Batched buffers (M=8 max)
    int M;
    void *d_x_fp4[8]; float *d_xs[8];          // Per-seq FP4 hidden state [H]
    int8_t *d_xi8; float *d_xi8s;              // INT8 input [M*H]
    float *d_Q, *d_K, *d_V;                    // QKV projections [M*Q], [M*KV], [M*KV]
    float *d_attn;                              // Attention output [M*Q]
    int8_t *d_attn_i8; float *d_attn_i8s;      // INT8 attn output [M*Q]
    float *d_gate, *d_up, *d_mlp;              // MLP buffers [M*ID]
    int8_t *d_mlp_i8; float *d_mlp_i8s;        // INT8 MLP output [M*ID]
    float *d_proj;                              // Shared projection buffer [M*H]
    float *d_res;                               // Residual add temp [H]
    float *d_rn;                                // RMSNorm weights temp [H]
    float *d_logits;                            // lm_head output [V]
    float *d_hidden;                            // Decoder hidden state [H]

    // KV cache
    float *d_kc, *d_vc;
    size_t kv_stride;  // bytes per sequence's KV cache (1 layer)

    // GPU sampler
    int *d_next_id;

    // Stream
    cudaStream_t st;
};

// ── Simple JSON reading (no dependencies) ───────────────────────────────
static std::string read_stdin_line() {
    std::string line;
    int c;
    while ((c = getchar()) != EOF && c != '\n') line.push_back((char)c);
    return line;
}

static std::vector<std::vector<uint32_t>> parse_prompt_ids(const std::string& json) {
    std::vector<std::vector<uint32_t>> result;
    // Find "prompts":[ and extract arrays
    const char* p = json.c_str();
    p = strstr(p, "\"prompts\"");
    if (!p) return result;
    p = strchr(p, '[');
    if (!p) return result;
    p++; // skip first [
    while (*p && *p != ']') {
        while (*p && *p != '[') p++;
        if (!*p || *p == ']') break;
        p++; // skip [
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
    const char* p = strstr(json.c_str(), key);
    if (!p) return def;
    p = strchr(p, ':');
    if (!p) return def;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    return (int)strtol(p, nullptr, 10);
}

static float find_float(const std::string& json, const char* key, float def) {
    const char* p = strstr(json.c_str(), key);
    if (!p) return def;
    p = strchr(p, ':');
    if (!p) return def;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    return (float)atof(p);
}

// Parse "prompts":["text1","text2"] → tokenized prompt IDs
static std::vector<std::vector<uint32_t>> parse_string_prompts(
    const std::string& json, blackwell::BpeTokenizer& tokenizer) {
    std::vector<std::vector<uint32_t>> result;
    const char* p = strstr(json.c_str(), "\"prompts\"");
    if (!p) return result;
    p = strchr(p, '[');
    if (!p) return result;
    p++; // skip [
    while (*p && *p != ']') {
        // Skip whitespace
        while (*p && (*p == ' ' || *p == '\t' || *p == '\n')) p++;
        if (*p == ']' || !*p) break;
        // Expect "
        if (*p != '"') { p++; continue; }
        p++; // skip opening "
        // Read string
        std::string s;
        while (*p && *p != '"') {
            if (*p == '\\' && *(p+1) == '"') { s += '"'; p += 2; }
            else { s += *p; p++; }
        }
        if (*p == '"') p++; // skip closing "
        // Tokenize
        auto ids = tokenizer.encode(s);
        result.push_back(ids);
        // Skip comma
        while (*p && (*p == ',' || *p == ' ' || *p == '\t' || *p == '\n')) p++;
    }
    return result;
}

// ── Simple JSON output ─────────────────────────────────────────────────
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
        for (size_t t = 0; t < tokens[s].size(); t++) {
            txt += tokenizer.decode(tokens[s][t]);
        }
        // Escape JSON special characters
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
    printf("]}\n");
    fflush(stdout);
}

// ── Embed batch: lookup tokens from INT8 embed table ────────────────────
static void embed_batch(ServerState& S, const std::vector<std::vector<uint32_t>>& prompts, int step) {
    // For step 0, embed all M prompts' first token
    // For subsequent steps, embed the newly generated tokens
    int M = (int)prompts.size();
    for (int m = 0; m < M; m++) {
        uint32_t tid = prompts[m][step];
        std::vector<float> h_hidden(S.H);
        std::vector<float> h_scales(S.H / 16);
        for (int d = 0; d < S.H; d++) {
            h_hidden[d] = (float)S.h_emb_int8[tid * S.H + d] * S.h_emb_scale[tid * (S.H/16) + d/16];
        }
        // Compute FP4 block scales (absmax/6.0 for FP4 E2M1, max=6)
        for (int b = 0; b < S.H / 16; b++) {
            float amax = 0;
            for (int d = 0; d < 16; d++) amax = fmaxf(amax, fabsf(h_hidden[b*16 + d]));
            h_scales[b] = fmaxf(amax / 6.0f, 1e-9f);
        }
        cudaMemcpy(S.d_hidden, h_hidden.data(), S.H*4, cudaMemcpyHostToDevice);
        cudaMemcpy(S.d_xs[m], h_scales.data(), (S.H/16)*4, cudaMemcpyHostToDevice);
        blackwell::kernels::pack_fp4(S.d_x_fp4[m], S.d_hidden, S.d_xs[m], S.H, S.st);
    }
    check_stream("embed_batch");
}

// ── Batched decode step (all M sequences, 1 token position) ─────────────
static void batched_decode_step(ServerState& S, int seq_pos) {
    int M = S.M;
    size_t kv_seq_stride = (size_t)S.nkv * S.hd * S.ms;

    for (int l = 0; l < S.NL; l++) {
        size_t kv_layer_off = (size_t)l * S.nkv * S.hd * S.ms;

        // Unpack FP4 → INT8 for each sequence
        for (int m = 0; m < M; m++) {
            blackwell::kernels::unpack_fp4_pack_int8(
                S.d_xi8 + m*S.H, S.d_xi8s + m*(S.H/16),
                S.d_x_fp4[m], S.d_xs[m],
                S.d_xi8s + m*(S.H/16), S.H, S.st);
        }

        // Q/K/V projections (serial per-seq, faster than batched)
        for (int m = 0; m < M; m++) {
            blackwell::kernels::gemv_int8_warp(S.d_Q + m*S.Q, S.d_xi8 + m*S.H, S.d_xi8s + m*(S.H/16),
                S.layers[l].q.d, S.layers[l].q.sc, S.H, S.Q, S.st);
            blackwell::kernels::gemv_int8_warp(S.d_K + m*S.KV, S.d_xi8 + m*S.H, S.d_xi8s + m*(S.H/16),
                S.layers[l].k.d, S.layers[l].k.sc, S.H, S.KV, S.st);
            blackwell::kernels::gemv_int8_warp(S.d_V + m*S.KV, S.d_xi8 + m*S.H, S.d_xi8s + m*(S.H/16),
                S.layers[l].v.d, S.layers[l].v.sc, S.H, S.KV, S.st);

            size_t km = (size_t)m * kv_seq_stride + kv_layer_off;
            blackwell::kernels::update_kv_cache(
                S.d_kc + km, S.d_vc + km,
                S.d_K + m*S.KV, S.d_V + m*S.KV,
                0, seq_pos, S.nkv, S.hd, S.ms, S.st);
        }

        // ONE batched attention call
        blackwell::kernels::attention_decode_batched_gqa(
            S.d_attn, S.d_Q, S.d_kc, S.d_vc,
            seq_pos, S.nqh, S.nkv, S.hd, S.ms,
            M, (int)kv_seq_stride, (int)kv_layer_off, S.st);

        // Attn output projection + residual
        for (int m = 0; m < M; m++) {
            blackwell::kernels::pack_int8(S.d_attn_i8 + m*S.Q, S.d_attn + m*S.Q, S.d_attn_i8s + m*(S.Q/16), S.Q, S.st);
            blackwell::kernels::gemv_int8_warp(S.d_proj + m*S.H, S.d_attn_i8 + m*S.Q, S.d_attn_i8s + m*(S.Q/16),
                S.layers[l].o.d, S.layers[l].o.sc, S.Q, S.H, S.st);
            blackwell::kernels::unpack_fp4(S.d_res, S.d_x_fp4[m], S.d_xs[m], S.H, S.st);
            blackwell::kernels::vector_add_fp32(S.d_proj + m*S.H, S.d_proj + m*S.H, S.d_res, S.H, S.st);
            blackwell::kernels::fused_rmsnorm_quant_int8(S.d_xi8 + m*S.H, S.d_xi8s + m*(S.H/16),
                S.d_proj + m*S.H, S.d_rn, S.H, S.eps, S.st);
        }

        // MLP: batched gate/up/down
        blackwell::kernels::gemv_int8_batched(S.d_gate, S.d_xi8, S.d_xi8s,
            S.layers[l].gate.d, S.layers[l].gate.sc, S.H, S.ID, M, S.st);
        blackwell::kernels::gemv_int8_batched(S.d_up, S.d_xi8, S.d_xi8s,
            S.layers[l].up.d, S.layers[l].up.sc, S.H, S.ID, M, S.st);

        for (int m = 0; m < M; m++) {
            blackwell::kernels::apply_swiglu(S.d_mlp + m*S.ID, S.d_gate + m*S.ID, S.d_up + m*S.ID, S.ID, S.st);
            blackwell::kernels::pack_int8(S.d_mlp_i8 + m*S.ID, S.d_mlp + m*S.ID, S.d_mlp_i8s + m*(S.ID/16), S.ID, S.st);
        }

        blackwell::kernels::gemv_int8_batched(S.d_proj, S.d_mlp_i8, S.d_mlp_i8s,
            S.layers[l].down.d, S.layers[l].down.sc, S.ID, S.H, M, S.st);

        // MLP residual + store next hidden state as FP4
        for (int m = 0; m < M; m++) {
            blackwell::kernels::unpack_fp4(S.d_res, S.d_x_fp4[m], S.d_xs[m], S.H, S.st);
            blackwell::kernels::vector_add_fp32(S.d_proj + m*S.H, S.d_proj + m*S.H, S.d_res, S.H, S.st);
            blackwell::kernels::fused_rmsnorm_quant_int8(S.d_xi8 + m*S.H, S.d_xi8s + m*(S.H/16),
                S.d_proj + m*S.H, S.d_rn, S.H, S.eps, S.st);
            blackwell::kernels::pack_fp4(S.d_x_fp4[m], S.d_proj + m*S.H, S.d_xs[m], S.H, S.st);
        }
    }
}

// ── Main ────────────────────────────────────────────────────────────────
int main() {
    ServerState S;
    // Qwen3-1.7B config
    S.NL=28; S.H=2048; S.Q=2048; S.KV=1024; S.ID=6144;
    S.nqh=16; S.nkv=8; S.hd=128; S.ms=2048; S.V=151936;
    S.eps=1e-6f; S.M=8;

    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    fprintf(stderr, "Blackwell INT8 Inference Server v0.2.0\n");
    fprintf(stderr, "Device: %s (CC %d.%d)\n", p.name, p.major, p.minor);
    fprintf(stderr, "Config: layers=%d H=%d Q=%d KV=%d ID=%d nqh=%d nkv=%d\n",
        S.NL, S.H, S.Q, S.KV, S.ID, S.nqh, S.nkv);

    die(cudaStreamCreate(&S.st), "stream");

    // ── Load tokenizer ────────────────────────────────────────────────
    blackwell::BpeTokenizer tokenizer;
    if (tokenizer.load("tokenizer_data.bin") != 0) {
        fprintf(stderr, "FAIL: no tokenizer_data.bin\n");
        return 1;
    }

    // ── Load weights ───────────────────────────────────────────────────
    fprintf(stderr, "Loading weights...\n");
    S.layers.resize(S.NL);
    for (int l = 0; l < S.NL; l++) {
        char p[256];
        snprintf(p,256,"weights_int8_bf16/%d_self_attn.q_proj",l); S.layers[l].q = upload_int8(p);
        snprintf(p,256,"weights_int8_bf16/%d_self_attn.k_proj",l); S.layers[l].k = upload_int8(p);
        snprintf(p,256,"weights_int8_bf16/%d_self_attn.v_proj",l); S.layers[l].v = upload_int8(p);
        snprintf(p,256,"weights_int8_bf16/%d_self_attn.o_proj",l); S.layers[l].o = upload_int8(p);
        snprintf(p,256,"weights_int8_bf16/%d_mlp.gate_proj",l);  S.layers[l].gate = upload_int8(p);
        snprintf(p,256,"weights_int8_bf16/%d_mlp.up_proj",l);    S.layers[l].up = upload_int8(p);
        snprintf(p,256,"weights_int8_bf16/%d_mlp.down_proj",l);  S.layers[l].down = upload_int8(p);
        if (l % 7 == 0) fprintf(stderr, "  layer %d/%d\n", l, S.NL);
    }
    S.emb = upload_int8("weights_int8_bf16/embed_tokens");
    // Qwen ties embed_tokens = lm_head. Use emb for lm_head GEMV.

    // Allocate host-side copies of embed weights for CPU-side dequant
    // (GPU pointers can't be dereferenced from host code)
    {
        char p[256]; snprintf(p,256,"weights_int8_bf16/embed_tokens.int8_t");
        FILE *f = fopen(p,"rb"); if(!f){fprintf(stderr,"Cannot open %s\n",p);exit(1);}
        int h[5]; fread(h,4,5,f);
        size_t num = (size_t)h[0]*h[1];
        S.h_emb_int8 = (int8_t*)malloc(num);
        fread(S.h_emb_int8,1,num,f); fclose(f);

        snprintf(p,256,"weights_int8_bf16/embed_tokens.scale_t");
        f = fopen(p,"rb"); fread(h,4,5,f);
        size_t ns = (size_t)h[3]*h[4];
        S.h_emb_scale = (float*)malloc(ns*4);
        fread(S.h_emb_scale,4,ns,f); fclose(f);
    }
    // Load final norm weights directly into float* buffer (FP32, not quantized)
    {
        float *tmp = (float*)malloc(S.H * 4);
        FILE *f = fopen("weights_int8_bf16/final_norm.f32", "rb");
        if (!f) { fprintf(stderr, "Cannot open final_norm.f32\n"); exit(1); }
        int h[5]; fread(h, 4, 5, f);
        fread(tmp, 4, S.H, f); fclose(f);
        cudaMalloc(&S.d_fn, S.H * 4);
        cudaMemcpy(S.d_fn, tmp, S.H * 4, cudaMemcpyHostToDevice);
        free(tmp);
    }
    fprintf(stderr, "  done\n");

    // ── Allocate buffers (M=8 max) ────────────────────────────────────
    cudaMalloc(&S.d_rn, S.H*4);
    cudaMalloc(&S.d_res, S.H*4);
    cudaMalloc(&S.d_hidden, S.H*4);
    cudaMalloc(&S.d_logits, S.V*4);
    std::vector<float> one_h(S.H,1.f); cudaMemcpy(S.d_rn,one_h.data(),S.H*4,cudaMemcpyHostToDevice);

    for (int m = 0; m < S.M; m++) {
        cudaMalloc(&S.d_x_fp4[m], S.H);
        cudaMalloc(&S.d_xs[m], (S.H/16)*4);
        cudaMemset(S.d_x_fp4[m], 0, S.H);
    }
    cudaMalloc(&S.d_xi8, S.M*S.H);
    cudaMalloc(&S.d_xi8s, S.M*(S.H/16)*4);
    cudaMalloc(&S.d_Q, S.M*S.Q*4); cudaMalloc(&S.d_K, S.M*S.KV*4); cudaMalloc(&S.d_V, S.M*S.KV*4);
    cudaMalloc(&S.d_attn, S.M*S.Q*4);
    cudaMalloc(&S.d_attn_i8, S.M*S.Q); cudaMalloc(&S.d_attn_i8s, S.M*(S.Q/16)*4);
    cudaMalloc(&S.d_gate, S.M*S.ID*4); cudaMalloc(&S.d_up, S.M*S.ID*4); cudaMalloc(&S.d_mlp, S.M*S.ID*4);
    cudaMalloc(&S.d_mlp_i8, S.M*S.ID); cudaMalloc(&S.d_mlp_i8s, S.M*(S.ID/16)*4);
    cudaMalloc(&S.d_proj, S.M*S.H*4);

    // KV cache: NL layers × M seqs × nkv × ms × hd
    size_t kv_sz = (size_t)S.NL * S.M * S.nkv * S.ms * S.hd * 4;
    cudaMalloc(&S.d_kc, kv_sz); cudaMalloc(&S.d_vc, kv_sz);
    cudaMemset(S.d_kc, 0, kv_sz); cudaMemset(S.d_vc, 0, kv_sz);
    S.kv_stride = (size_t)S.nkv * S.hd * S.ms;

    die(cudaMalloc(&S.d_next_id, sizeof(int)), "next_id");
    check_stream("alloc");

    fprintf(stderr, "Ready. Waiting for input on stdin...\n");

    // ── Main request loop ────────────────────────────────────────────
    while (true) {
        std::string line = read_stdin_line();
        if (line.empty()) break;  // EOF

        // ── Parse request ─────────────────────────────────────────────
        auto prompts = parse_prompt_ids(line);
        if (prompts.empty()) {
            // Try parsing as string prompts: "prompts":["text1","text2"]
            prompts = parse_string_prompts(line, tokenizer);
        }
        if (prompts.empty()) {
            fprintf(stderr, "Warning: empty or malformed request\n");
            continue;
        }

        int M = (int)prompts.size();
        if (M > 8) { fprintf(stderr, "Warning: batch >8, truncating\n"); M = 8; prompts.resize(8); }
        S.M = M;

        int max_tokens = find_int(line, "max_tokens", 30);
        float temperature = find_float(line, "temperature", 0);
        int top_k = find_int(line, "top_k", 0);

        fprintf(stderr, "  Batch: %d seqs, max_tokens=%d temp=%.1f top_k=%d\n",
            M, max_tokens, temperature, top_k);
        fflush(stderr);

        // ── Prefill: embed and process all prompt tokens ──────────────
        int gen_start = (int)prompts[0].size();
        for (int s = 0; s < gen_start; s++) {
            embed_batch(S, prompts, s);
            batched_decode_step(S, s);
        }

        // ── Autoregressive generation ─────────────────────────────────
        std::vector<std::vector<uint32_t>> outputs(M);
        for (int s = gen_start; s < gen_start + max_tokens; s++) {
            for (int m = 0; m < M; m++) {
                // Final norm + lm_head + sample in one sequence
                blackwell::kernels::unpack_fp4(S.d_hidden, S.d_x_fp4[m], S.d_xs[m], S.H, S.st);
                blackwell::kernels::fused_rmsnorm(S.d_hidden, S.d_hidden, S.d_fn, S.H, S.eps, S.st);
                blackwell::kernels::quantize_int8(S.d_xi8, S.d_xi8s, S.d_hidden, S.H, S.st);
                blackwell::kernels::gemv_int8_warp(S.d_logits, S.d_xi8, S.d_xi8s,
                    S.emb.d, S.emb.sc, S.H, S.V, S.st);
                blackwell::kernels::sample_gpu(S.d_logits, S.V, temperature, top_k,
                    S.d_next_id, 0xdeadbeefLL, s, S.st);

                uint32_t next_id;
                cudaMemcpy(&next_id, S.d_next_id, sizeof(int), cudaMemcpyDeviceToHost);
                outputs[m].push_back(next_id);

                if (next_id == 151643) break;  // EOS

                // Embed next token for next decode step
                std::vector<float> h_hidden(S.H);
                std::vector<float> h_scales(S.H / 16);
                for (int d = 0; d < S.H; d++) {
                    h_hidden[d] = (float)S.h_emb_int8[next_id * S.H + d] * S.h_emb_scale[next_id * (S.H/16) + d/16];
                }
                for (int b = 0; b < S.H / 16; b++) {
                    float amax = 0;
                    for (int d = 0; d < 16; d++) amax = fmaxf(amax, fabsf(h_hidden[b*16 + d]));
                    h_scales[b] = fmaxf(amax / 6.0f, 1e-9f);
                }
                cudaMemcpy(S.d_hidden, h_hidden.data(), S.H*4, cudaMemcpyHostToDevice);
                cudaMemcpy(S.d_xs[m], h_scales.data(), (S.H/16)*4, cudaMemcpyHostToDevice);
                blackwell::kernels::pack_fp4(S.d_x_fp4[m], S.d_hidden, S.d_xs[m], S.H, S.st);
            }

            batched_decode_step(S, s);
        }

        // Write results
        write_results(outputs, tokenizer);

        // Clear KV cache for next batch
        cudaMemset(S.d_kc, 0, kv_sz); cudaMemset(S.d_vc, 0, kv_sz);
    }

    fprintf(stderr, "Shutting down.\n");
    return 0;
}
