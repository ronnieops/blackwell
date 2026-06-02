// bench/decode_int4_cgraph.cu — CUDA Graph benchmark for INT4 decode
//
// INT4 decode pipeline: same 14-kernel/layer architecture as INT8,
// but uses gemv_int4_warp instead of gemv_int8_warp.
// Tests if INT4's 2× bandwidth advantage overcomes dequantization overhead.
//
// Build:
//   CUDACXX=/usr/local/cuda-13.3/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/decode_int4_cgraph.cu build/libblackwell_kernels.a \
//     -o bench/decode_int4_cgraph

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include <cstdint>
#include "blackwell/kernels.h"

struct GpuTimer {
    cudaEvent_t s, e;
    GpuTimer() { cudaEventCreate(&s); cudaEventCreate(&e); }
    ~GpuTimer() { cudaEventDestroy(s); cudaEventDestroy(e); }
    void start(cudaStream_t st=0) { cudaEventRecord(s, st); }
    float stop(cudaStream_t st=0) {
        cudaEventRecord(e, st); cudaEventSynchronize(e);
        float ms=0; cudaEventElapsedTime(&ms, s, e); return ms;
    }
};

static void chk(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) { printf("FAIL: %s: %s\n", msg, cudaGetErrorString(e)); exit(1); }
}

// Load INT4 weights: .int4_t + .scale_t files
struct LoadedW4 { int K, N; std::vector<uint8_t> d; std::vector<float> sc; };

static LoadedW4 load_int4_w(const char* prefix) {
    char p[256];
    snprintf(p, 256, "%s.int4_t", prefix);
    FILE* f = fopen(p, "rb");
    if (!f) { printf("Cannot open %s\n", p); exit(1); }
    int h[5]; fread(h, 4, 5, f);
    LoadedW4 w;
    w.K = h[0]; w.N = h[1];
    int packed_size = h[0] * h[1] / 2;  // K × N/2 bytes
    w.d.resize(packed_size);
    fread(w.d.data(), 1, packed_size, f);
    fclose(f);

    snprintf(p, 256, "%s.scale_t", prefix);
    f = fopen(p, "rb");
    fread(h, 4, 5, f);
    int num_kb = h[3];
    w.sc.resize(num_kb * h[4]);  // N * num_K_blks floats
    fread(w.sc.data(), 4, w.sc.size(), f);
    fclose(f);

    printf("  %s: K=%d N=%d packed=%d bytes scales=%d floats\n",
           prefix, w.K, w.N, (int)w.d.size(), (int)w.sc.size());
    return w;
}

// Upload to GPU and transpose to row-major
struct DevW4 { int K, N; uint8_t* d; float* sc; };

static DevW4 upload_transpose(const char* prefix) {
    auto w = load_int4_w(prefix);
    DevW4 dw{w.K, w.N};

    // Upload original
    uint8_t* d_orig;
    cudaMalloc(&d_orig, w.d.size());
    cudaMemcpy(d_orig, w.d.data(), w.d.size(), cudaMemcpyHostToDevice);

    float* sc_orig;
    cudaMalloc(&sc_orig, w.sc.size() * 4);
    cudaMemcpy(sc_orig, w.sc.data(), w.sc.size() * 4, cudaMemcpyHostToDevice);

    // Allocate transposed (row-major): d [N][K/2], sc [N][K/16]
    size_t packed_sz = (size_t)w.N * (w.K / 2);
    size_t scale_sz = (size_t)w.N * (w.K / 16);
    cudaMalloc(&dw.d, packed_sz);
    cudaMalloc(&dw.sc, scale_sz * 4);

    // Transpose
    chk(blackwell::kernels::transpose_int4_weights(
        dw.d, dw.sc, d_orig, sc_orig, w.K, w.N), "transpose_int4");

    cudaFree(d_orig); cudaFree(sc_orig);
    return dw;
}

int main(int argc, char** argv) {
    int num_layers = 4;
    if (argc > 1) num_layers = atoi(argv[1]);
    if (num_layers > 28) num_layers = 28;

    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("# INT4 CUDA Graph Decode Benchmark — Qwen3-1.7B\n");
    printf("Device: %s (%d.%d)\n", p.name, p.major, p.minor);
    printf("Layers: %d\n", num_layers);

    const int H = 2048, Q = 2048, KV = 1024, I = 6144;
    const int nqh = 16, nkv = 8, hd = 128, ms = 2048;
    const float s13 = 1.f / 3.f;

    // Load INT4 weights (transposed to row-major)
    struct LW4 { DevW4 q, k, v, o, g, u, d; };
    printf("Loading + transposing INT4 weights...\n");
    std::vector<LW4> lw(num_layers);
    for (int l = 0; l < num_layers; ++l) {
        char p[256];
        snprintf(p, 256, "weights_int4_qwen3_1.7b/%d_self_attn.q_proj", l); lw[l].q = upload_transpose(p);
        snprintf(p, 256, "weights_int4_qwen3_1.7b/%d_self_attn.k_proj", l); lw[l].k = upload_transpose(p);
        snprintf(p, 256, "weights_int4_qwen3_1.7b/%d_self_attn.v_proj", l); lw[l].v = upload_transpose(p);
        snprintf(p, 256, "weights_int4_qwen3_1.7b/%d_self_attn.o_proj", l); lw[l].o = upload_transpose(p);
        snprintf(p, 256, "weights_int4_qwen3_1.7b/%d_mlp.gate_proj", l);   lw[l].g = upload_transpose(p);
        snprintf(p, 256, "weights_int4_qwen3_1.7b/%d_mlp.up_proj", l);     lw[l].u = upload_transpose(p);
        snprintf(p, 256, "weights_int4_qwen3_1.7b/%d_mlp.down_proj", l);   lw[l].d = upload_transpose(p);
    }

    // ── Buffers ─────────────────────────────────────────────────────────────
    float *d_x32, *d_xs;
    float *d_rn;
    cudaMalloc(&d_x32, H * 4);
    cudaMalloc(&d_xs, (H / 16) * 4);
    cudaMalloc(&d_rn, H * 4);
    std::vector<float> rn_h(H, 1.f);
    cudaMemcpy(d_rn, rn_h.data(), H * 4, cudaMemcpyHostToDevice);

    // Per-layer buffers
    float *d_Q, *d_K, *d_V, *d_attn, *d_proj;
    float *d_gate, *d_up, *d_mlp;
    float *d_res;
    cudaMalloc(&d_Q, Q * 4); cudaMalloc(&d_K, KV * 4); cudaMalloc(&d_V, KV * 4);
    cudaMalloc(&d_attn, Q * 4); cudaMalloc(&d_proj, H * 4);
    cudaMalloc(&d_gate, I * 4); cudaMalloc(&d_up, I * 4); cudaMalloc(&d_mlp, I * 4);
    cudaMalloc(&d_res, I * 4);

    // INT4 activation buffers (packed: K/2 bytes)
    uint8_t *d_x_i4;
    float *d_x_i4_sc;
    uint8_t *d_attn_i4;
    float *d_attn_i4_sc;
    uint8_t *d_mlp_i4;
    float *d_mlp_i4_sc;
    cudaMalloc(&d_x_i4, H / 2);  // packed: K/2 bytes
    cudaMalloc(&d_x_i4_sc, (H / 16) * 4);
    cudaMalloc(&d_attn_i4, Q / 2);
    cudaMalloc(&d_attn_i4_sc, (Q / 16) * 4);
    cudaMalloc(&d_mlp_i4, I / 2);
    cudaMalloc(&d_mlp_i4_sc, (I / 16) * 4);

    // Init scales (uniform)
    std::vector<float> xsh(H / 16, s13);
    cudaMemcpy(d_xs, xsh.data(), (H / 16) * 4, cudaMemcpyHostToDevice);
    float ixv = 1.f / 7.f;
    std::vector<float> ixsh(H / 16, ixv);
    cudaMemcpy(d_x_i4_sc, ixsh.data(), (H / 16) * 4, cudaMemcpyHostToDevice);
    std::vector<float> ai4s(Q / 16, ixv), mi4s(I / 16, ixv);
    cudaMemcpy(d_attn_i4_sc, ai4s.data(), (Q / 16) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mlp_i4_sc, mi4s.data(), (I / 16) * 4, cudaMemcpyHostToDevice);

    // Save initial scales
    float *d_x_i4_sc_init, *d_attn_i4_sc_init, *d_mlp_i4_sc_init;
    cudaMalloc(&d_x_i4_sc_init, (H / 16) * 4); cudaMemcpy(d_x_i4_sc_init, d_x_i4_sc, (H / 16) * 4, cudaMemcpyDeviceToDevice);
    cudaMalloc(&d_attn_i4_sc_init, (Q / 16) * 4); cudaMemcpy(d_attn_i4_sc_init, d_attn_i4_sc, (Q / 16) * 4, cudaMemcpyDeviceToDevice);
    cudaMalloc(&d_mlp_i4_sc_init, (I / 16) * 4); cudaMemcpy(d_mlp_i4_sc_init, d_mlp_i4_sc, (I / 16) * 4, cudaMemcpyDeviceToDevice);

    // KV cache
    float *d_kc, *d_vc;
    size_t kv_sz = (size_t)num_layers * nkv * ms * hd * 4;
    cudaMalloc(&d_kc, kv_sz); cudaMalloc(&d_vc, kv_sz);
    cudaMemset(d_kc, 0, kv_sz); cudaMemset(d_vc, 0, kv_sz);

    // L2 cache hints
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 8 * 1024 * 1024);

    // ── Fill KV cache (seq=0..128) on default stream ──────────────────────
    printf("Filling KV cache (seq=0..128)... ");
    fflush(stdout);

    // Init x = uniform 1.0
    std::vector<float> xh(H, 1.f);
    cudaMemcpy(d_x32, xh.data(), H * 4, cudaMemcpyHostToDevice);

    // For INT4: quantize FP32 → INT4 (pack x to packed format)
    // We'll unpack to FP32 for the GEMV, then quantize for the next layer
    std::vector<float> x_recon(H, 1.f);
    cudaMemcpy(d_x32, x_recon.data(), H * 4, cudaMemcpyHostToDevice);

    int sq = 128;
    for (int s = 0; s <= sq; ++s) {
        for (int l = 0; l < num_layers; ++l) {
            // x is FP32 here (d_x32). For GEMV, unpack FP32 → INT4 for activations,
            // then use gemv_int4_warp. But gemv_int4_warp takes packed INT4 input.
            // Since activations are FP32, we should use FP32 × INT4 kernel.
            // Let's unpack the current x to INT4 format for the next iteration.

            // For this layer: d_x32 is FP32 → quantize to INT4
            // Then use gemv_int4_warp for QKV
            // After proj output, RMSNorm → quantize → next layer
            //
            // Actually: since gemv_int4_warp needs PACKED INT4 input,
            // and our activations start as FP32, we need to quantize first.
            // Let's quantize d_x32 → d_x_i4 (packed), then use gemv_int4_warp.

            // Quantize x to INT4
            chk(blackwell::kernels::quantize_int4(d_x_i4, d_x_i4_sc, d_x32, H), "quantize_x");

            // QKV: each is gemv_int4_warp(x_i4, W_q/k/v)
            // x_packed: [K/2], x_scale: [K/16], W: [N][K/2], W_scale: [N][K/16]
            chk(blackwell::kernels::gemv_int4_warp(d_Q, d_x_i4, d_x_i4_sc,
                lw[l].q.d, lw[l].q.sc, H, Q), "Q");
            chk(blackwell::kernels::gemv_int4_warp(d_K, d_x_i4, d_x_i4_sc,
                lw[l].k.d, lw[l].k.sc, H, KV), "K");
            chk(blackwell::kernels::gemv_int4_warp(d_V, d_x_i4, d_x_i4_sc,
                lw[l].v.d, lw[l].v.sc, H, KV), "V");

            // Update KV cache
            int kb = l * nkv * ms * hd;
            chk(blackwell::kernels::update_kv_cache(
                d_kc + kb, d_vc + kb, d_K, d_V, 0, s, nkv, hd, ms), "kv");

            // Attention
            chk(blackwell::kernels::attention_decode_gqa(
                d_attn, d_Q, d_kc + kb, d_vc + kb, s, nqh, nkv, hd, ms), "attn");

            // Pack attn → INT4
            chk(blackwell::kernels::quantize_int4(d_attn_i4, d_attn_i4_sc, d_attn, Q), "quantize_attn");

            // Wo: gemv_int4_warp(attn_i4, W_o)
            chk(blackwell::kernels::gemv_int4_warp(d_proj, d_attn_i4, d_attn_i4_sc,
                lw[l].o.d, lw[l].o.sc, Q, H), "Wo");

            // Residual: proj += d_x32 (current hidden state), then RMSNorm, then quantize
            chk(blackwell::kernels::fused_residual_norm_int4(d_x_i4, d_x_i4_sc, d_proj, d_x32, d_rn, H, 1e-6f), "fused_rnq_attn");

            // MLP: gate/up → swiglu → quantize → down → residual → RMSNorm
            chk(blackwell::kernels::gemv_int4_warp(d_gate, d_x_i4, d_x_i4_sc,
                lw[l].g.d, lw[l].g.sc, H, I), "gate");
            chk(blackwell::kernels::gemv_int4_warp(d_up, d_x_i4, d_x_i4_sc,
                lw[l].u.d, lw[l].u.sc, H, I), "up");

            chk(blackwell::kernels::fused_swiglu_quant_int4(d_mlp_i4, d_mlp_i4_sc, d_gate, d_up, I), "fused_swiglu_quant");

            chk(blackwell::kernels::gemv_int4_warp(d_proj, d_mlp_i4, d_mlp_i4_sc,
                lw[l].d.d, lw[l].d.sc, I, H), "down");

            // MLP residual: INT4 → d_x_i4 (not d_x32!), FP32 → d_x32
            chk(blackwell::kernels::fused_residual_norm_int4_fp32out(d_x_i4, d_x_i4_sc, d_x32, d_proj, d_x32, d_rn, H, 1e-6f), "fused_rnq_mlp");
        }
    }
    printf("done\n");

    // ── Benchmark decode (single token) ───────────────────────────────────
    printf("\nBenchmarking...\n");

    // Restore scales (overwritten by quantize_int4)
    cudaMemcpy(d_x_i4_sc, d_x_i4_sc_init, (H / 16) * 4, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_attn_i4_sc, d_attn_i4_sc_init, (Q / 16) * 4, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_mlp_i4_sc, d_mlp_i4_sc_init, (I / 16) * 4, cudaMemcpyDeviceToDevice);

    const int warmup = 50, iters = 200;
    GpuTimer t;

    // Per-kernel timing
    cudaDeviceSynchronize();
    t.start();
    for (int i = 0; i < warmup; ++i) {
        for (int l = 0; l < num_layers; ++l) {

        chk(blackwell::kernels::quantize_int4(d_x_i4, d_x_i4_sc, d_x32, H), "quantize_x");
        chk(blackwell::kernels::gemv_int4_warp(d_Q, d_x_i4, d_x_i4_sc, lw[l].q.d, lw[l].q.sc, H, Q), "Q");
        chk(blackwell::kernels::gemv_int4_warp(d_K, d_x_i4, d_x_i4_sc, lw[l].k.d, lw[l].k.sc, H, KV), "K");
        chk(blackwell::kernels::gemv_int4_warp(d_V, d_x_i4, d_x_i4_sc, lw[l].v.d, lw[l].v.sc, H, KV), "V");

        int kb = l * nkv * ms * hd;
        chk(blackwell::kernels::update_kv_cache(d_kc + kb, d_vc + kb, d_K, d_V, 0, sq + 1, nkv, hd, ms), "kv");
        chk(blackwell::kernels::attention_decode_gqa(d_attn, d_Q, d_kc + kb, d_vc + kb, sq + 1, nqh, nkv, hd, ms), "attn");
        chk(blackwell::kernels::quantize_int4(d_attn_i4, d_attn_i4_sc, d_attn, Q), "quantize_attn");
        chk(blackwell::kernels::gemv_int4_warp(d_proj, d_attn_i4, d_attn_i4_sc, lw[l].o.d, lw[l].o.sc, Q, H), "Wo");
        chk(blackwell::kernels::fused_residual_norm_int4(d_x_i4, d_x_i4_sc, d_proj, d_x32, d_rn, H, 1e-6f), "fused_rnq_attn");
        chk(blackwell::kernels::gemv_int4_warp(d_gate, d_x_i4, d_x_i4_sc, lw[l].g.d, lw[l].g.sc, H, I), "gate");
        chk(blackwell::kernels::gemv_int4_warp(d_up, d_x_i4, d_x_i4_sc, lw[l].u.d, lw[l].u.sc, H, I), "up");
        chk(blackwell::kernels::fused_swiglu_quant_int4(d_mlp_i4, d_mlp_i4_sc, d_gate, d_up, I), "fused_swiglu_quant");
        chk(blackwell::kernels::gemv_int4_warp(d_proj, d_mlp_i4, d_mlp_i4_sc, lw[l].d.d, lw[l].d.sc, I, H), "down");
        chk(blackwell::kernels::fused_residual_norm_int4_fp32out(d_x_i4, d_x_i4_sc, d_x32, d_proj, d_x32, d_rn, H, 1e-6f), "fused_rnq_mlp");
        }

        // Restore scales for next iter
        cudaMemcpy(d_x_i4_sc, d_x_i4_sc_init, (H / 16) * 4, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_attn_i4_sc, d_attn_i4_sc_init, (Q / 16) * 4, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_mlp_i4_sc, d_mlp_i4_sc_init, (I / 16) * 4, cudaMemcpyDeviceToDevice);
    }
    t.stop();
    printf("Warmup done\n");

    // Full layer decode timing
    cudaDeviceSynchronize();
    t.start();
    for (int i = 0; i < iters; ++i) {
        for (int l = 0; l < num_layers; ++l) {
            chk(blackwell::kernels::quantize_int4(d_x_i4, d_x_i4_sc, d_x32, H), "quantize_x");
            chk(blackwell::kernels::gemv_int4_warp(d_Q, d_x_i4, d_x_i4_sc, lw[l].q.d, lw[l].q.sc, H, Q), "Q");
            chk(blackwell::kernels::gemv_int4_warp(d_K, d_x_i4, d_x_i4_sc, lw[l].k.d, lw[l].k.sc, H, KV), "K");
            chk(blackwell::kernels::gemv_int4_warp(d_V, d_x_i4, d_x_i4_sc, lw[l].v.d, lw[l].v.sc, H, KV), "V");
            int kb = l * nkv * ms * hd;
            chk(blackwell::kernels::update_kv_cache(d_kc + kb, d_vc + kb, d_K, d_V, 0, sq + 1, nkv, hd, ms), "kv");
            chk(blackwell::kernels::attention_decode_gqa(d_attn, d_Q, d_kc + kb, d_vc + kb, sq + 1, nqh, nkv, hd, ms), "attn");
            chk(blackwell::kernels::quantize_int4(d_attn_i4, d_attn_i4_sc, d_attn, Q), "quantize_attn");
            chk(blackwell::kernels::gemv_int4_warp(d_proj, d_attn_i4, d_attn_i4_sc, lw[l].o.d, lw[l].o.sc, Q, H), "Wo");
            chk(blackwell::kernels::fused_residual_norm_int4(d_x_i4, d_x_i4_sc, d_proj, d_x32, d_rn, H, 1e-6f), "fused_rnq_attn");
            chk(blackwell::kernels::gemv_int4_warp(d_gate, d_x_i4, d_x_i4_sc, lw[l].g.d, lw[l].g.sc, H, I), "gate");
            chk(blackwell::kernels::gemv_int4_warp(d_up, d_x_i4, d_x_i4_sc, lw[l].u.d, lw[l].u.sc, H, I), "up");
            chk(blackwell::kernels::fused_swiglu_quant_int4(d_mlp_i4, d_mlp_i4_sc, d_gate, d_up, I), "fused_swiglu_quant");
            chk(blackwell::kernels::gemv_int4_warp(d_proj, d_mlp_i4, d_mlp_i4_sc, lw[l].d.d, lw[l].d.sc, I, H), "down");
            chk(blackwell::kernels::fused_residual_norm_int4_fp32out(d_x_i4, d_x_i4_sc, d_x32, d_proj, d_x32, d_rn, H, 1e-6f), "fused_rnq_mlp");
        }
    }
    float ms_total = t.stop();

    float ms_per_layer = ms_total / iters;
    float ms_per_seq = ms_per_layer;
    float total_t_s = (num_layers * iters) / (ms_total / 1000.0);
    float per_seq_t_s = 1.0 / (ms_per_seq / 1000.0);

    printf("\n=== Results ===\n");
    printf("Total time: %.3f ms (%d iters)\n", ms_total, iters);
    printf("Per layer:  %.3f ms\n", ms_per_layer);
    printf("Per seq:    %.3f ms\n", ms_per_seq);
    printf("Total t/s:  %.1f\n", total_t_s);
    printf("Per-seq t/s: %.1f\n", per_seq_t_s);
    printf("INT8 ref:    %.1f t/s (181.5)\n", 181.5f);
    printf("Q4_K_M ref: %.1f t/s (293.4)\n", 293.4f);
    printf("vs INT8:     %.0f%%\n", per_seq_t_s / 181.5f * 100.0f);
    printf("vs Q4_K_M:   %.0f%%\n", per_seq_t_s / 293.4f * 100.0f);

    // Cleanup
    for (auto& w : lw) {
        cudaFree(w.q.d); cudaFree(w.q.sc);
        cudaFree(w.k.d); cudaFree(w.k.sc);
        cudaFree(w.v.d); cudaFree(w.v.sc);
        cudaFree(w.o.d); cudaFree(w.o.sc);
        cudaFree(w.g.d); cudaFree(w.g.sc);
        cudaFree(w.u.d); cudaFree(w.u.sc);
        cudaFree(w.d.d); cudaFree(w.d.sc);
    }
    cudaFree(d_x32); cudaFree(d_xs); cudaFree(d_rn);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_attn); cudaFree(d_proj);
    cudaFree(d_gate); cudaFree(d_up); cudaFree(d_mlp);
    cudaFree(d_res);
    cudaFree(d_x_i4); cudaFree(d_x_i4_sc);
    cudaFree(d_attn_i4); cudaFree(d_attn_i4_sc);
    cudaFree(d_mlp_i4); cudaFree(d_mlp_i4_sc);
    cudaFree(d_x_i4_sc_init); cudaFree(d_attn_i4_sc_init); cudaFree(d_mlp_i4_sc_init);
    cudaFree(d_kc); cudaFree(d_vc);

    printf("\nDone.\n");
    return 0;
}