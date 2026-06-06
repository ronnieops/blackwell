// bench/prefill_decode_benchmark.cu — Prefill + decode full pipeline
// Measures: prefill prompt (with attention) → decode tokens
// Compare: prefill+decode vs decode-only
//
// Build: nvcc -O3 -std=c++17 -arch=sm_120a bench/prefill_decode_benchmark.cu
//         build/libblackwell_kernels.a -I include -o bench/prefill_decode_benchmark

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <sys/time.h>
#include "blackwell/kernels.h"

using namespace blackwell::kernels;

static void die(cudaError_t e, const char* m = "") {
    if (e != cudaSuccess) { fprintf(stderr, "FAIL %s: %s\n", m, cudaGetErrorString(e)); exit(1); }
}

// ── Inline kernels (from inference_server) ─────────────────────────────

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

__global__ void rope_kernel(float* data, int nh, int hd, const int* seq_pos) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= nh * (hd / 2)) return;
    int head = h / (hd / 2);
    int dim = h % (hd / 2);
    int pos = *seq_pos;
    float theta = powf(10000.0f, -2.0f * dim / hd);
    float cos_ = cosf(pos * theta);
    float sin_ = sinf(pos * theta);
    float* x = data + head * hd;
    float x0 = x[dim], x1 = x[dim + hd/2];
    x[dim] = x0 * cos_ - x1 * sin_;
    x[dim + hd/2] = x0 * sin_ + x1 * cos_;
}

// ── Weight structures ─────────────────────────────────────────────────

struct W { int K, N; int8_t* d; float* sc; };
struct LW { W q, k, v, o, gate, up, down; float *rn_in, *rn_post, *qn, *kn; };
struct Emb { int8_t* d; float* sc; int H, V; };

static W load_int8(const char* dir, const char* name) {
    char p[256]; snprintf(p, 256, "%s/%s.int8_t", dir, name);
    FILE* f = fopen(p, "rb"); die(f ? cudaSuccess : cudaErrorMemoryAllocation, "open");
    int h[5]; fread(h, 4, 5, f); int K = h[0], N = h[1];
    std::vector<int8_t> tmp((size_t)K * N); fread(tmp.data(), 1, (size_t)K * N, f); fclose(f);
    W w; w.K = K; w.N = N;
    die(cudaMalloc(&w.d, (size_t)K * N), "malloc"); cudaMemcpy(w.d, tmp.data(), (size_t)K * N, cudaMemcpyHostToDevice);
    snprintf(p, 256, "%s/%s.scale_t", dir, name); f = fopen(p, "rb"); die(f ? cudaSuccess : cudaErrorMemoryAllocation, "open");
    fread(h, 4, 5, f); int ns = h[3] * h[4];
    std::vector<float> ts(ns); fread(ts.data(), 4, ns, f); fclose(f);
    die(cudaMalloc(&w.sc, ns * 4), "malloc"); cudaMemcpy(w.sc, ts.data(), ns * 4, cudaMemcpyHostToDevice);
    return w;
}
static W load_int8_transposed(const char* dir, const char* name) {
    char p[256]; snprintf(p, 256, "%s/%s.int8_t", dir, name);
    FILE* f = fopen(p, "rb"); die(f ? cudaSuccess : cudaErrorMemoryAllocation, "open");
    int h[5]; fread(h, 4, 5, f); int K = h[0], N = h[1];
    std::vector<int8_t> tmp((size_t)K * N); fread(tmp.data(), 1, (size_t)K * N, f); fclose(f);
    std::vector<int8_t> dst((size_t)N * K);
    for (int k = 0; k < K; k++) for (int n = 0; n < N; n++) dst[(size_t)n * K + k] = tmp[(size_t)k * N + n];
    W w; w.K = K; w.N = N;
    die(cudaMalloc(&w.d, (size_t)N * K), "malloc"); cudaMemcpy(w.d, dst.data(), (size_t)N * K, cudaMemcpyHostToDevice);
    snprintf(p, 256, "%s/%s.scale_t", dir, name); f = fopen(p, "rb"); die(f ? cudaSuccess : cudaErrorMemoryAllocation, "open");
    fread(h, 4, 5, f); int sk = h[3], sn = h[4];
    std::vector<float> s_src(sk * sn); fread(s_src.data(), 4, sk * sn, f); fclose(f);
    std::vector<float> s_dst(sn * sk);
    for (int r = 0; r < sn; r++) for (int c = 0; c < sk; c++) s_dst[(size_t)r * sk + c] = s_src[(size_t)c * sn + r];
    die(cudaMalloc(&w.sc, sn * sk * 4), "malloc"); cudaMemcpy(w.sc, s_dst.data(), sn * sk * 4, cudaMemcpyHostToDevice);
    return w;
}
static float* load_f32(const char* dir, const char* name, int n) {
    char p[256]; snprintf(p, 256, "%s/%s.f32", dir, name);
    FILE* f = fopen(p, "rb"); die(f ? cudaSuccess : cudaErrorMemoryAllocation, "open");
    float* h = (float*)malloc(n * 4); fread(h, 4, n, f); fclose(f);
    float* d; die(cudaMalloc(&d, n * 4), "malloc"); cudaMemcpy(d, h, n * 4, cudaMemcpyHostToDevice); free(h); return d;
}
static void load_emb(Emb& e, const char* dir) {
    char p[256]; snprintf(p, 256, "%s/embed_tokens.int8_t", dir);
    FILE* f = fopen(p, "rb"); die(f ? cudaSuccess : cudaErrorMemoryAllocation, "open");
    int h[5]; fread(h, 4, 5, f); e.H = h[0]; e.V = h[1];
    std::vector<int8_t> tmp((size_t)e.H * e.V); fread(tmp.data(), 1, (size_t)e.H * e.V, f); fclose(f);
    die(cudaMalloc(&e.d, (size_t)e.H * e.V), "malloc"); cudaMemcpy(e.d, tmp.data(), (size_t)e.H * e.V, cudaMemcpyHostToDevice);
    snprintf(p, 256, "%s/embed_tokens.scale_t", dir); f = fopen(p, "rb"); die(f ? cudaSuccess : cudaErrorMemoryAllocation, "open");
    fread(h, 4, 5, f); int ns = h[3] * h[4];
    std::vector<float> ts(ns); fread(ts.data(), 4, ns, f); fclose(f);
    die(cudaMalloc(&e.sc, ns * 4), "malloc"); cudaMemcpy(e.sc, ts.data(), ns * 4, cudaMemcpyHostToDevice);
}

// ── Prefill step (M tokens, full attention) ───────────────────────────

struct PrefillState {
    float *d_h, *d_h_out;
    int8_t *d_h_i8; float *d_h_sc;
    float *d_Q, *d_K, *d_V;  // Q,K,V for current token
    float *d_K_full, *d_V_full;  // full K,V cache for all tokens
    int8_t *d_attn_i8; float *d_attn_i8s;
    float *d_attn_out;
    float *d_proj, *d_gate, *d_up, *d_mlp;
    int8_t *d_mlp_i8; float *d_mlp_i8s;
    float *d_residual;
    int *d_seq_pos;
};

static void prefill_alloc(PrefillState& P, int M, int H, int Q, int KV, int ID, int nl, int nqh, int nkv, int hd, int ms) {
    size_t kvcache = (size_t)nl * nkv * ms * hd * 4;
    die(cudaMalloc(&P.d_h, M * H * 4), "h");
    die(cudaMalloc(&P.d_h_out, M * H * 4), "h_out");
    die(cudaMalloc(&P.d_h_i8, M * H), "h_i8");
    die(cudaMalloc(&P.d_h_sc, M * (H/16) * 4), "h_sc");
    die(cudaMalloc(&P.d_Q, Q * 4), "Q");
    die(cudaMalloc(&P.d_K, KV * 4), "K");
    die(cudaMalloc(&P.d_V, KV * 4), "V");
    die(cudaMalloc(&P.d_K_full, kvcache), "K_full");
    die(cudaMalloc(&P.d_V_full, kvcache), "V_full");
    die(cudaMalloc(&P.d_attn_i8, Q), "attn_i8");
    die(cudaMalloc(&P.d_attn_i8s, (Q/16) * 4), "attn_i8s");
    die(cudaMalloc(&P.d_attn_out, Q * 4), "attn_out");
    die(cudaMalloc(&P.d_proj, H * 4), "proj");
    die(cudaMalloc(&P.d_gate, ID * 4), "gate");
    die(cudaMalloc(&P.d_up, ID * 4), "up");
    die(cudaMalloc(&P.d_mlp, ID * 4), "mlp");
    die(cudaMalloc(&P.d_mlp_i8, ID), "mlp_i8");
    die(cudaMalloc(&P.d_mlp_i8s, (ID/16) * 4), "mlp_i8s");
    die(cudaMalloc(&P.d_residual, H * 4), "residual");
    die(cudaMalloc(&P.d_seq_pos, sizeof(int)), "seq_pos");
    cudaMemset(P.d_K_full, 0, kvcache);
    cudaMemset(P.d_V_full, 0, kvcache);
}

static void prefill_free(PrefillState& P) {
    cudaFree(P.d_h); cudaFree(P.d_h_out); cudaFree(P.d_h_i8); cudaFree(P.d_h_sc);
    cudaFree(P.d_Q); cudaFree(P.d_K); cudaFree(P.d_V);
    cudaFree(P.d_K_full); cudaFree(P.d_V_full);
    cudaFree(P.d_attn_i8); cudaFree(P.d_attn_i8s); cudaFree(P.d_attn_out);
    cudaFree(P.d_proj); cudaFree(P.d_gate); cudaFree(P.d_up); cudaFree(P.d_mlp); cudaFree(P.d_mlp_i8); cudaFree(P.d_mlp_i8s);
    cudaFree(P.d_residual); cudaFree(P.d_seq_pos);
}

// Run prefill for M tokens through all NL layers
// Returns hidden state of last token in d_last_hidden
static void run_prefill(PrefillState& P, const float* d_input, int M,
                        const LW* layers, int NL, int H, int Q, int KV, int ID,
                        int nqh, int nkv, int hd, int ms, int nl,
                        float* d_last_hidden, cudaStream_t st) {
    // Copy input embeddings to d_h
    cudaMemcpy(P.d_h, d_input, M * H * 4, cudaMemcpyDeviceToDevice);

    size_t kv_layer_stride = (size_t)nkv * hd * ms;
    size_t kv_seq_stride = (size_t)nkv * hd;

    for (int l = 0; l < NL; l++) {
        // Input RMSNorm + quantize per token
        for (int m = 0; m < M; m++) {
            fused_rmsnorm_quant_int8(P.d_h_i8 + m * H, P.d_h_sc + m * (H/16),
                P.d_h + m * H, layers[l].rn_in, H, 1e-6f, st);
        }

        // QKV for each token, write to full K/V cache
        for (int m = 0; m < M; m++) {
            gemv_int8_warp(P.d_Q, P.d_h_i8 + m * H, P.d_h_sc + m * (H/16),
                layers[l].q.d, layers[l].q.sc, H, Q, st);
            gemv_int8_warp(P.d_K, P.d_h_i8 + m * H, P.d_h_sc + m * (H/16),
                layers[l].k.d, layers[l].k.sc, H, KV, st);
            gemv_int8_warp(P.d_V, P.d_h_i8 + m * H, P.d_h_sc + m * (H/16),
                layers[l].v.d, layers[l].v.sc, H, KV, st);

            // Q/K head norms
            head_norm_kernel<<<nqh, 128, 0, st>>>(P.d_Q, layers[l].qn, nqh, hd, 1e-6f);
            head_norm_kernel<<<nkv, 128, 0, st>>>(P.d_K, layers[l].kn, nkv, hd, 1e-6f);

            // RoPE at position m
            int pos = m;
            cudaMemcpyAsync(P.d_seq_pos, &pos, sizeof(int), cudaMemcpyHostToDevice, st);
            rope_kernel<<<nqh, hd/2, 0, st>>>(P.d_Q, nqh, hd, P.d_seq_pos);
            rope_kernel<<<nkv, hd/2, 0, st>>>(P.d_K, nkv, hd, P.d_seq_pos);

            // Write K,V to full cache
            size_t off = (size_t)l * kv_layer_stride + m * kv_seq_stride;
            cudaMemcpyAsync(P.d_K_full + off, P.d_K, KV * 4, cudaMemcpyDeviceToDevice, st);
            cudaMemcpyAsync(P.d_V_full + off, P.d_V, KV * 4, cudaMemcpyDeviceToDevice, st);

            // Attention: Q_m attends to K_0..K_m, V_0..V_m
            attention_decode_batched_gqa(P.d_attn_out, P.d_Q, P.d_K_full, P.d_V_full,
                m, nqh, nkv, hd, ms, M, kv_seq_stride, (int)(l * kv_layer_stride), st);

            // Output projection + residual
            quantize_int8(P.d_attn_i8, P.d_attn_i8s, P.d_attn_out, Q, st);
            gemv_int8_warp(P.d_proj, P.d_attn_i8, P.d_attn_i8s, layers[l].o.d, layers[l].o.sc, Q, H, st);
            // Residual: save for next layer
            cudaMemcpyAsync(P.d_residual, P.d_proj, H * 4, cudaMemcpyDeviceToDevice, st);
            // Add residual
            vector_add_fp32(P.d_h_out + m * H, P.d_proj, P.d_h + m * H, H, st);
        }

        // Post-attention RMSNorm + MLP
        for (int m = 0; m < M; m++) {
            fused_rmsnorm_quant_int8(P.d_h_i8 + m * H, P.d_h_sc + m * (H/16),
                P.d_residual, layers[l].rn_post, H, 1e-6f, st);
        }

        // MLP gate/up
        gemv_int8_batched(P.d_gate, P.d_h_i8, P.d_h_sc,
            layers[l].gate.d, layers[l].gate.sc, H, ID, M, st);
        gemv_int8_batched(P.d_up, P.d_h_i8, P.d_h_sc,
            layers[l].up.d, layers[l].up.sc, H, ID, M, st);

        for (int m = 0; m < M; m++) {
            apply_swiglu(P.d_mlp + m * ID, P.d_gate + m * ID, P.d_up + m * ID, ID, st);
            quantize_int8(P.d_mlp_i8 + m * ID, P.d_mlp_i8s + m * (ID/16), P.d_mlp + m * ID, ID, st);
        }

        gemv_int8_batched(P.d_proj, P.d_mlp_i8, P.d_mlp_i8s,
            layers[l].down.d, layers[l].down.sc, ID, H, M, st);

        for (int m = 0; m < M; m++) {
            // Residual 2: down + attention output
            vector_add_fp32(P.d_h + m * H, P.d_proj + m * H, P.d_residual, H, st);
        }
    }

    // Copy last token's hidden state
    cudaMemcpyAsync(d_last_hidden, P.d_h + (M - 1) * H, H * 4, cudaMemcpyDeviceToDevice, st);
}

// ── Decode step (single token) ───────────────────────────────────────

static void run_decode(float* d_x, float* d_tmp, int8_t* d_xi8, float* d_xi8s,
                       float* d_Q, float* d_K, float* d_V, float* d_proj,
                       float* d_attn, int8_t* d_attn_i8, float* d_attn_i8s,
                       float* d_gate, float* d_up, float* d_mlp, int8_t* d_mlp_i8, float* d_mlp_i8s,
                       float* d_kc, float* d_vc, int* d_seq_pos,
                       const LW* layers, int NL, int H, int Q, int KV, int ID,
                       int nqh, int nkv, int hd, int ms, int nl, int seq_pos,
                       cudaStream_t st) {
    size_t kv_seq_stride = (size_t)nkv * hd;

    for (int l = 0; l < NL; l++) {
        size_t kv_layer_off = (size_t)l * nkv * hd * ms;
        size_t kv_off = kv_layer_off + seq_pos * kv_seq_stride;

        fused_rmsnorm_quant_int8(d_xi8, d_xi8s, d_x, layers[l].rn_in, H, 1e-6f, st);

        gemv_int8_warp(d_Q, d_xi8, d_xi8s, layers[l].q.d, layers[l].q.sc, H, Q, st);
        gemv_int8_warp(d_K, d_xi8, d_xi8s, layers[l].k.d, layers[l].k.sc, H, KV, st);
        gemv_int8_warp(d_V, d_xi8, d_xi8s, layers[l].v.d, layers[l].v.sc, H, KV, st);

        head_norm_kernel<<<nqh, 128, 0, st>>>(d_Q, layers[l].qn, nqh, hd, 1e-6f);
        head_norm_kernel<<<nkv, 128, 0, st>>>(d_K, layers[l].kn, nkv, hd, 1e-6f);

        update_decode_seq_pos(seq_pos, st);
        cudaStreamSynchronize(st);
        rope_kernel<<<nqh, hd/2, 0, st>>>(d_Q, nqh, hd, d_seq_pos);
        rope_kernel<<<nkv, hd/2, 0, st>>>(d_K, nkv, hd, d_seq_pos);

        cudaMemcpyAsync(d_kc + kv_off, d_K, KV * 4, cudaMemcpyDeviceToDevice, st);
        cudaMemcpyAsync(d_vc + kv_off, d_V, KV * 4, cudaMemcpyDeviceToDevice, st);

        attention_decode_gqa(d_attn, d_Q, d_kc + kv_layer_off, d_vc + kv_layer_off,
            seq_pos, nqh, nkv, hd, ms, st);

        quantize_int8(d_attn_i8, d_attn_i8s, d_attn, Q, st);
        gemv_int8_warp(d_proj, d_attn_i8, d_attn_i8s, layers[l].o.d, layers[l].o.sc, Q, H, st);
        vector_add_fp32(d_tmp, d_proj, d_x, H, st);

        fused_rmsnorm_quant_int8(d_xi8, d_xi8s, d_tmp, layers[l].rn_post, H, 1e-6f, st);

        gemv_int8_warp(d_gate, d_xi8, d_xi8s, layers[l].gate.d, layers[l].gate.sc, H, ID, st);
        gemv_int8_warp(d_up, d_xi8, d_xi8s, layers[l].up.d, layers[l].up.sc, H, ID, st);
        apply_swiglu(d_mlp, d_gate, d_up, ID, st);
        quantize_int8(d_mlp_i8, d_mlp_i8s, d_mlp, ID, st);
        gemv_int8_warp(d_proj, d_mlp_i8, d_mlp_i8s, layers[l].down.d, layers[l].down.sc, ID, H, st);

        vector_add_fp32(d_x, d_proj, d_tmp, H, st);
    }
}

// ── Decode-only (no prefill) ─────────────────────────────────────────

static void run_decode_only(float* d_x, float* d_tmp, int8_t* d_xi8, float* d_xi8s,
                            float* d_Q, float* d_K, float* d_V, float* d_proj,
                            float* d_attn, int8_t* d_attn_i8, float* d_attn_i8s,
                            float* d_gate, float* d_up, float* d_mlp, int8_t* d_mlp_i8, float* d_mlp_i8s,
                            float* d_kc, float* d_vc, int* d_seq_pos,
                            const LW* layers, int NL, int H, int Q, int KV, int ID,
                            int nqh, int nkv, int hd, int ms, int nl, int seq_pos,
                            cudaStream_t st) {
    size_t kv_seq_stride = (size_t)nkv * hd;

    for (int l = 0; l < NL; l++) {
        size_t kv_layer_off = (size_t)l * nkv * hd * ms;
        size_t kv_off = kv_layer_off + seq_pos * kv_seq_stride;

        fused_rmsnorm_quant_int8(d_xi8, d_xi8s, d_x, layers[l].rn_in, H, 1e-6f, st);

        gemv_int8_warp(d_Q, d_xi8, d_xi8s, layers[l].q.d, layers[l].q.sc, H, Q, st);
        gemv_int8_warp(d_K, d_xi8, d_xi8s, layers[l].k.d, layers[l].k.sc, H, KV, st);
        gemv_int8_warp(d_V, d_xi8, d_xi8s, layers[l].v.d, layers[l].v.sc, H, KV, st);

        head_norm_kernel<<<nqh, 128, 0, st>>>(d_Q, layers[l].qn, nqh, hd, 1e-6f);
        head_norm_kernel<<<nkv, 128, 0, st>>>(d_K, layers[l].kn, nkv, hd, 1e-6f);

        update_decode_seq_pos(seq_pos, st);
        cudaStreamSynchronize(st);
        rope_kernel<<<nqh, hd/2, 0, st>>>(d_Q, nqh, hd, d_seq_pos);
        rope_kernel<<<nkv, hd/2, 0, st>>>(d_K, nkv, hd, d_seq_pos);

        cudaMemcpyAsync(d_kc + kv_off, d_K, KV * 4, cudaMemcpyDeviceToDevice, st);
        cudaMemcpyAsync(d_vc + kv_off, d_V, KV * 4, cudaMemcpyDeviceToDevice, st);

        attention_decode_gqa(d_attn, d_Q, d_kc + kv_layer_off, d_vc + kv_layer_off,
            seq_pos, nqh, nkv, hd, ms, st);

        quantize_int8(d_attn_i8, d_attn_i8s, d_attn, Q, st);
        gemv_int8_warp(d_proj, d_attn_i8, d_attn_i8s, layers[l].o.d, layers[l].o.sc, Q, H, st);
        vector_add_fp32(d_tmp, d_proj, d_x, H, st);

        fused_rmsnorm_quant_int8(d_xi8, d_xi8s, d_tmp, layers[l].rn_post, H, 1e-6f, st);

        gemv_int8_warp(d_gate, d_xi8, d_xi8s, layers[l].gate.d, layers[l].gate.sc, H, ID, st);
        gemv_int8_warp(d_up, d_xi8, d_xi8s, layers[l].up.d, layers[l].up.sc, H, ID, st);
        apply_swiglu(d_mlp, d_gate, d_up, ID, st);
        quantize_int8(d_mlp_i8, d_mlp_i8s, d_mlp, ID, st);
        gemv_int8_warp(d_proj, d_mlp_i8, d_mlp_i8s, layers[l].down.d, layers[l].down.sc, ID, H, st);

        vector_add_fp32(d_x, d_proj, d_tmp, H, st);
    }
}

// ── Main ─────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    const char* WDIR = "weights_int8_bf16";
    int PRE_SEQ = (argc > 1) ? atoi(argv[1]) : 8;
    int DEC_TOKENS = (argc > 2) ? atoi(argv[2]) : 20;
    if (PRE_SEQ < 1) PRE_SEQ = 1; if (PRE_SEQ > 32) PRE_SEQ = 32;  // limit for now
    if (DEC_TOKENS < 1) DEC_TOKENS = 1; if (DEC_TOKENS > 50) DEC_TOKENS = 50;

    // Qwen3-1.7B config
    const int NL = 28, H = 2048, Q = 2048, KV = 256, ID = 11008;
    const int nqh = 32, nkv = 4, hd = 64, ms = 2048;
    const int V = 151936;

    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("=== Blackwell Prefill + Decode Pipeline ===\n");
    printf("Device: %s\n", p.name);
    printf("Model: Qwen3-1.7B INT8, NL=%d, H=%d\n", NL, H);
    printf("Prefill: %d tokens, Decode: %d tokens\n\n", PRE_SEQ, DEC_TOKENS);

    // Load qk_norms from single file (all layers)
    FILE* f = fopen("weights_int8_bf16/qk_norms.f32", "rb");
    if (!f) { fprintf(stderr, "FAIL: no qk_norms.f32\n"); return 1; }
    std::vector<float> qk_h(NL * 2 * 128); fread(qk_h.data(), 4, NL * 2 * 128, f); fclose(f);
    std::vector<float*> qn(NL), kn(NL);
    for (int l = 0; l < NL; l++) {
        die(cudaMalloc(&qn[l], 128 * 4), "qn");
        die(cudaMalloc(&kn[l], 128 * 4), "kn");
        cudaMemcpy(qn[l], qk_h.data() + l * 2 * 128, 128 * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(kn[l], qk_h.data() + l * 2 * 128 + 128, 128 * 4, cudaMemcpyHostToDevice);
    }

    // Load weights (one layer at a time)
    std::vector<LW> layers(NL);
    printf("Loading weights...\n"); fflush(stdout);
    for (int l = 0; l < NL; l++) {
        layers[l].qn = qn[l];
        layers[l].kn = kn[l];
        char p[256];
        snprintf(p, 256, "%d_self_attn.q_proj", l); layers[l].q = load_int8(WDIR, p);
        snprintf(p, 256, "%d_self_attn.k_proj", l); layers[l].k = load_int8(WDIR, p);
        snprintf(p, 256, "%d_self_attn.v_proj", l); layers[l].v = load_int8(WDIR, p);
        snprintf(p, 256, "%d_self_attn.o_proj", l); layers[l].o = load_int8(WDIR, p);
        snprintf(p, 256, "%d_mlp.gate_proj", l); layers[l].gate = load_int8(WDIR, p);
        snprintf(p, 256, "%d_mlp.up_proj", l); layers[l].up = load_int8(WDIR, p);
        snprintf(p, 256, "%d_mlp.down_proj", l); layers[l].down = load_int8(WDIR, p);
        snprintf(p, 256, "%d_input_layernorm", l); layers[l].rn_in = load_f32(WDIR, p, H);
        snprintf(p, 256, "%d_post_attention_layernorm", l); layers[l].rn_post = load_f32(WDIR, p, H);
    }
    printf("\n"); fflush(stdout);

    // Generate synthetic embeddings (avoid loading 297MB vocab from disk)
    std::vector<float> h_emb(PRE_SEQ * H);
    for (int i = 0; i < (int)h_emb.size(); i++) {
        h_emb[i] = ((rand() % 2000) / 1000.0f - 1.0f) * 0.5f;
    }
    float *d_prefill_input;
    cudaMalloc(&d_prefill_input, PRE_SEQ * H * 4);
    cudaMemcpy(d_prefill_input, h_emb.data(), PRE_SEQ * H * 4, cudaMemcpyHostToDevice);

    // Allocate prefill buffers
    PrefillState P;
    prefill_alloc(P, PRE_SEQ, H, Q, KV, ID, NL, nqh, nkv, hd, ms);

    // Decode buffers (share K/V cache with prefill)
    float *d_x, *d_tmp, *d_last_hidden;
    int8_t *d_xi8;
    float *d_xi8s, *d_proj;
    float *d_attn, *d_gate, *d_up, *d_mlp;
    int8_t *d_attn_i8, *d_mlp_i8;
    float *d_attn_i8s, *d_mlp_i8s;
    int *d_seq_pos;
    cudaMalloc(&d_x, H * 4); cudaMalloc(&d_tmp, H * 4); cudaMalloc(&d_last_hidden, H * 4);
    cudaMalloc(&d_xi8, H); cudaMalloc(&d_xi8s, (H/16) * 4);
    cudaMalloc(&d_proj, H * 4);
    cudaMalloc(&d_attn, Q * 4); cudaMalloc(&d_gate, ID * 4); cudaMalloc(&d_up, ID * 4); cudaMalloc(&d_mlp, ID * 4);
    cudaMalloc(&d_attn_i8, Q); cudaMalloc(&d_mlp_i8, ID);
    cudaMalloc(&d_attn_i8s, (Q/16) * 4); cudaMalloc(&d_mlp_i8s, (ID/16) * 4);
    cudaMalloc(&d_seq_pos, sizeof(int));

    cudaStream_t st;
    cudaStreamCreate(&st);

    // Warmup
    cudaMemset(d_prefill_input, 0, PRE_SEQ * H * 4);
    cudaMemset(d_x, 0, H * 4);
    cudaDeviceSynchronize();

    printf("\n--- Benchmark 1: Decode-only (prompt + generate) ---\n");
    {
        size_t kvcache = (size_t)NL * nkv * ms * hd * 4;
        cudaMemset(P.d_K_full, 0, kvcache);
        cudaMemset(P.d_V_full, 0, kvcache);
        cudaDeviceSynchronize();

        struct timeval t0, t1;
        gettimeofday(&t0, NULL);
        for (int s = 0; s < PRE_SEQ + DEC_TOKENS; s++) {
            // For prompt tokens, use embeddings; for gen tokens, use last hidden
            if (s < PRE_SEQ) {
                cudaMemcpy(d_x, d_prefill_input + s * H, H * 4, cudaMemcpyDeviceToDevice);
            }
            run_decode_only(d_x, d_tmp, d_xi8, d_xi8s, P.d_Q, P.d_K, P.d_V, d_proj,
                           d_attn, d_attn_i8, d_attn_i8s, d_gate, d_up, d_mlp, d_mlp_i8, d_mlp_i8s,
                           P.d_K_full, P.d_V_full, P.d_seq_pos,
                           layers.data(), NL, H, Q, KV, ID, nqh, nkv, hd, ms, NL, s, st);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_usec - t0.tv_usec) / 1000.0;
        printf("  Total: %.1f ms for %d tokens\n", ms, PRE_SEQ + DEC_TOKENS);
        printf("  Throughput: %.1f t/s (%.2f ms/token)\n", 1000.0 * (PRE_SEQ + DEC_TOKENS) / ms, ms / (PRE_SEQ + DEC_TOKENS));
    }

    printf("\n--- Benchmark 2: Prefill + Decode ---\n");
    {
        size_t kvcache = (size_t)NL * nkv * ms * hd * 4;
        cudaMemset(P.d_K_full, 0, kvcache);
        cudaMemset(P.d_V_full, 0, kvcache);
        cudaDeviceSynchronize();

        struct timeval t0, t1;
        gettimeofday(&t0, NULL);

        // Prefill
        run_prefill(P, d_prefill_input, PRE_SEQ, layers.data(), NL, H, Q, KV, ID,
                   nqh, nkv, hd, ms, NL, d_last_hidden, st);
        cudaStreamSynchronize(st);

        // Decode from position PRE_SEQ (reuse K/V cache from prefill)
        for (int s = PRE_SEQ; s < PRE_SEQ + DEC_TOKENS; s++) {
            cudaMemcpy(d_x, d_last_hidden, H * 4, cudaMemcpyDeviceToDevice);
            run_decode(d_x, d_tmp, d_xi8, d_xi8s, P.d_Q, P.d_K, P.d_V, d_proj,
                      d_attn, d_attn_i8, d_attn_i8s, d_gate, d_up, d_mlp, d_mlp_i8, d_mlp_i8s,
                      P.d_K_full, P.d_V_full, P.d_seq_pos,
                      layers.data(), NL, H, Q, KV, ID, nqh, nkv, hd, ms, NL, s, st);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_usec - t0.tv_usec) / 1000.0;
        printf("  Total: %.1f ms (prefill %.1f + decode %.1f)\n",
               ms, 0.0, ms);  // TODO: split times
        printf("  Throughput: %.1f t/s\n", 1000.0 * (PRE_SEQ + DEC_TOKENS) / ms);
    }

    printf("\n--- Benchmark 3: Prefill time breakdown ---\n");
    for (int s = 4; s <= PRE_SEQ && s <= 32; s *= 2) {
        size_t kvcache = (size_t)NL * nkv * ms * hd * 4;
        cudaMemset(P.d_K_full, 0, kvcache);
        cudaMemset(P.d_V_full, 0, kvcache);
        cudaDeviceSynchronize();
        struct timeval t0, t1;
        gettimeofday(&t0, NULL);
        run_prefill(P, d_prefill_input, s, layers.data(), NL, H, Q, KV, ID,
                   nqh, nkv, hd, ms, NL, d_last_hidden, st);
        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_usec - t0.tv_usec) / 1000.0;
        printf("  SEQ=%2d: %.1f ms (%.0f t/s)\n", s, ms, 1000.0 * s / ms);
    }

    cudaStreamDestroy(st);
    prefill_free(P);
    return 0;
}