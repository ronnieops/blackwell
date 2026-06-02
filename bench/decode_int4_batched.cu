// bench/decode_int4_batched.cu — INT4 batched decode (M sequences per iteration)
//
// INT4 M-sequence decode with batched GEMVs. Replaces serial per-seq GEMV
// with gemv_int4_batched for Q/K/V, gate, up, down projections.
//
// Build:
//   /usr/local/cuda-13.3/bin/nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/decode_int4_batched.cu build/libblackwell_kernels.a \
//     -o bench/decode_int4_batched

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

// INT4 weight loader (from .int4_t + .scale_t files)
struct LoadedW4 {
    int K, N;
    std::vector<uint8_t> d;   // [K/2] bytes packed INT4
    std::vector<float> sc;    // [N * K/16] FP32 scales
};
static LoadedW4 load_int4_w(const char* prefix) {
    char p[256];
    snprintf(p, 256, "%s.int4_t", prefix);
    FILE* f = fopen(p, "rb"); if (!f) { printf("Cannot open %s\n", p); exit(1); }
    int h[5]; fread(h, 4, 5, f);
    LoadedW4 w; w.K = h[0]; w.N = h[1];
    int packed_sz = h[0] * h[1] / 2;
    w.d.resize(packed_sz); fread(w.d.data(), 1, packed_sz, f); fclose(f);

    snprintf(p, 256, "%s.scale_t", prefix);
    f = fopen(p, "rb"); if (!f) { printf("Cannot open %s\n", p); exit(1); }
    fread(h, 4, 5, f);
    int num_sc = h[3] * h[4];
    w.sc.resize(num_sc); fread(w.sc.data(), 4, num_sc, f); fclose(f);
    return w;
}

struct DevW4 { int K, N; uint8_t* d; float* sc; };
static DevW4 upload_w4(const char* prefix) {
    auto w = load_int4_w(prefix);
    DevW4 dw{w.K, w.N};
    cudaMalloc(&dw.d, w.d.size()); cudaMemcpy(dw.d, w.d.data(), w.d.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&dw.sc, w.sc.size() * 4); cudaMemcpy(dw.sc, w.sc.data(), w.sc.size() * 4, cudaMemcpyHostToDevice);
    return dw;
}

struct LW4 {
    DevW4 q, k, v, o, g, u, d;
};

// Process one sequence through all layers
static void process_seq(
    float* x_out,          // output hidden state [H]
    const float* x_in,     // input [H]
    int l0,                // first layer index
    int num_l,             // number of layers
    const LW4* lw,
    uint8_t* x_i4, float* x_i4_sc,
    float* mlp_sc,         // separate scale buffer for MLP (same H/16 elements)
    float* Q, float* K, float* V,
    float* attn, float* proj,
    uint8_t* attn_i4, float* attn_i4_sc,
    float* gate, float* up,
    uint8_t* mlp_i4, float* mlp_i4_sc,
    float* rn,
    float* kc, float* vc,
    int H, int Qdim, int KVdim, int Idim,
    int nkv, int hd, int ms,
    int kv_seq_stride,     // floats per seq in KV cache
    int pos,               // position for attention
    bool write_kv
) {
    // Quantize input
    blackwell::kernels::quantize_int4(x_i4, x_i4_sc, x_in, H, 0);

    // QKV
    float* Qm = Q, *Km = K, *Vm = V;
    blackwell::kernels::gemv_int4_warp(Qm, x_i4, x_i4_sc, lw->q.d, lw->q.sc, H, Qdim, 0);
    blackwell::kernels::gemv_int4_warp(Km, x_i4, x_i4_sc, lw->k.d, lw->k.sc, H, KVdim, 0);
    blackwell::kernels::gemv_int4_warp(Vm, x_i4, x_i4_sc, lw->v.d, lw->v.sc, H, KVdim, 0);

    // Copy x for residual (we'll need it later)
    // Process each layer
    for (int l = l0; l < l0 + num_l; ++l) {
        size_t kb = (size_t)l * nkv * ms * hd;

        // Write KV cache
        if (write_kv) {
            blackwell::kernels::update_kv_cache(
                kc + kb, vc + kb, Km, Vm, 0, pos, nkv, hd, ms, 0);
        }

        // Attention
        blackwell::kernels::attention_decode_gqa(attn, Qm, kc + kb, vc + kb,
            pos, 16, nkv, hd, ms, 0);

        // O projection
        blackwell::kernels::quantize_int4(attn_i4, attn_i4_sc, attn, Qdim, 0);
        blackwell::kernels::gemv_int4_warp(proj, attn_i4, attn_i4_sc, lw->o.d, lw->o.sc, Qdim, H, 0);

        // Residual: proj += x_in
        blackwell::kernels::vector_add_fp32(proj, proj, x_in, H, 0);

        // RMSNorm and quantize for MLP input
        blackwell::kernels::fused_residual_norm_int4(x_i4, x_i4_sc, proj, x_in, rn, H, 1e-6f, 0);

        // MLP gate
        blackwell::kernels::gemv_int4_warp(gate, x_i4, x_i4_sc, lw->g.d, lw->g.sc, H, Idim, 0);

        // MLP up
        blackwell::kernels::gemv_int4_warp(up, x_i4, x_i4_sc, lw->u.d, lw->u.sc, H, Idim, 0);

        // SwiGLU + quantize
        blackwell::kernels::fused_swiglu_quant_int4(mlp_i4, mlp_i4_sc, gate, up, Idim, 0);

        // MLP down
        blackwell::kernels::gemv_int4_warp(proj, mlp_i4, mlp_i4_sc, lw->d.d, lw->d.sc, Idim, H, 0);

        // Restore x_i4_sc before MLP residual (was overwritten by fused_swiglu_quant_int4)
        blackwell::kernels::quantize_int4(x_i4, x_i4_sc, x_in, H, 0);

        // Residual: proj += x_in (post-attn norm), then RMSNorm -> output
        blackwell::kernels::fused_residual_norm_int4_fp32out(x_out, mlp_sc, proj, x_in, x_in, rn, H, 1e-6f, 0);

        // Next layer: x_in = x_out
        x_in = x_out;
    }
}

int main(int argc, char** argv) {
    int num_layers = 4, M = 4;
    if (argc > 1) num_layers = atoi(argv[1]);
    if (argc > 2) M = atoi(argv[2]);
    if (num_layers > 28) num_layers = 28;
    if (M < 1) M = 1;
    if (M > 8) M = 8;

    const int H = 2048, Q = 2048, KV = 1024, I = 6144;
    const int nkv = 8, hd = 128, ms = 2048;

    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("# INT4 Batched Decode — M=%d sequences\n", M);
    printf("Device: %s\n", p.name);
    printf("Layers: %d\n", num_layers);

    // Load weights
    printf("Loading INT4 weights...\n");
    std::vector<LW4> lw(num_layers);
    for (int l = 0; l < num_layers; ++l) {
        char p[256];
        snprintf(p, 256, "weights_int4_qwen3_1.7b/%d_self_attn.q_proj", l); lw[l].q = upload_w4(p);
        snprintf(p, 256, "weights_int4_qwen3_1.7b/%d_self_attn.k_proj", l); lw[l].k = upload_w4(p);
        snprintf(p, 256, "weights_int4_qwen3_1.7b/%d_self_attn.v_proj", l); lw[l].v = upload_w4(p);
        snprintf(p, 256, "weights_int4_qwen3_1.7b/%d_self_attn.o_proj", l); lw[l].o = upload_w4(p);
        snprintf(p, 256, "weights_int4_qwen3_1.7b/%d_mlp.gate_proj", l);  lw[l].g = upload_w4(p);
        snprintf(p, 256, "weights_int4_qwen3_1.7b/%d_mlp.up_proj", l);    lw[l].u = upload_w4(p);
        snprintf(p, 256, "weights_int4_qwen3_1.7b/%d_mlp.down_proj", l);   lw[l].d = upload_w4(p);
    }

    // ── Per-sequence buffers ────────────────────────────────────────
    float **d_x32_arr = new float*[M];
    uint8_t **d_x_i4_arr = new uint8_t*[M];
    float **d_x_i4_sc_arr = new float*[M];
    for (int m = 0; m < M; ++m) {
        cudaMalloc(&d_x32_arr[m], H * 4);
        cudaMalloc(&d_x_i4_arr[m], (H / 2));
        cudaMalloc(&d_x_i4_sc_arr[m], (H / 16) * 4);
    }

    // Shared intermediates (reused per layer)
    float *d_Q, *d_K, *d_V;
    float *d_attn, *d_proj;
    uint8_t *d_attn_i4;
    float *d_attn_i4_sc;
    float *d_gate, *d_up;
    uint8_t *d_mlp_i4;
    float *d_mlp_i4_sc;
    // Separate scale buffer for MLP residual norm (fused kernels overwrite scale output)
    float *d_mlp_sc_buf;
    cudaMalloc(&d_mlp_sc_buf, (H / 16) * 4);
    float *d_rn;

    cudaMalloc(&d_Q, Q * 4); cudaMalloc(&d_K, KV * 4); cudaMalloc(&d_V, KV * 4);
    cudaMalloc(&d_attn, Q * 4); cudaMalloc(&d_proj, H * 4);
    cudaMalloc(&d_attn_i4, Q / 2); cudaMalloc(&d_attn_i4_sc, (Q / 16) * 4);
    cudaMalloc(&d_gate, I * 4); cudaMalloc(&d_up, I * 4);
    cudaMalloc(&d_mlp_i4, I / 2); cudaMalloc(&d_mlp_i4_sc, (I / 16) * 4);
    cudaMalloc(&d_rn, H * 4);

    std::vector<float> rn_h(H, 1.f), xh(H, 1.0f);
    cudaMemcpy(d_rn, rn_h.data(), H * 4, cudaMemcpyHostToDevice);

    // Init per-seq input to all 1.0
    for (int m = 0; m < M; ++m) {
        cudaMemcpy(d_x32_arr[m], xh.data(), H * 4, cudaMemcpyHostToDevice);
    }

    // Init quantization scales
    std::vector<float> xsh(H / 16, 1.f / 7.f);
    std::vector<float> ash(Q / 16, 1.f / 7.f);
    std::vector<float> msh(I / 16, 1.f / 7.f);
    for (int m = 0; m < M; ++m) {
        cudaMemcpy(d_x_i4_sc_arr[m], xsh.data(), (H / 16) * 4, cudaMemcpyHostToDevice);
    }

    // ── KV cache ──────────────────────────────────────────────────
    // Layout: [num_layers][nkv][ms][hd] floats, per sequence
    size_t kv_seq_stride = (size_t)num_layers * nkv * ms * hd;  // floats
    size_t kv_sz = (size_t)M * kv_seq_stride * 4;               // bytes
    float *d_kc, *d_vc;
    cudaMalloc(&d_kc, kv_sz); cudaMalloc(&d_vc, kv_sz);
    cudaMemset(d_kc, 0, kv_sz); cudaMemset(d_vc, 0, kv_sz);

    // ── Fill KV cache ──────────────────────────────────────────────
    printf("Filling KV cache (M=%d, seq=0..128)... ", M);
    int sq = 128;
    for (int s = 0; s <= sq; ++s) {
        for (int m = 0; m < M; ++m) {
            float* kc_seq = d_kc + (size_t)m * kv_seq_stride;
            float* vc_seq = d_vc + (size_t)m * kv_seq_stride;
            process_seq(
                d_x32_arr[m],
                d_x32_arr[m],
                0, num_layers,
                lw.data(),
                d_x_i4_arr[m], d_x_i4_sc_arr[m],
                d_mlp_sc_buf,
                d_Q, d_K, d_V,
                d_attn, d_proj,
                d_attn_i4, d_attn_i4_sc,
                d_gate, d_up,
                d_mlp_i4, d_mlp_i4_sc,
                d_rn,
                kc_seq, vc_seq,
                H, Q, KV, I,
                nkv, hd, ms,
                kv_seq_stride,
                s, true
            );
        }
    }
    printf("done\n");

    // Re-init inputs for benchmark
    for (int m = 0; m < M; ++m) {
        cudaMemcpy(d_x32_arr[m], xh.data(), H * 4, cudaMemcpyHostToDevice);
    }

    // ── Benchmark ──────────────────────────────────────────────────
    const int warmup = 20, iters = 100;
    printf("Warming up (%d iters)...\n", warmup);

    for (int i = 0; i < warmup; ++i) {
        for (int m = 0; m < M; ++m) {
            float* kc_seq = d_kc + (size_t)m * kv_seq_stride;
            float* vc_seq = d_vc + (size_t)m * kv_seq_stride;
            process_seq(
                d_x32_arr[m], d_x32_arr[m],
                0, num_layers,
                lw.data(),
                d_x_i4_arr[m], d_x_i4_sc_arr[m],
                d_mlp_sc_buf,
                d_Q, d_K, d_V,
                d_attn, d_proj,
                d_attn_i4, d_attn_i4_sc,
                d_gate, d_up,
                d_mlp_i4, d_mlp_i4_sc,
                d_rn,
                kc_seq, vc_seq,
                H, Q, KV, I,
                nkv, hd, ms,
                kv_seq_stride,
                sq + 1, true
            );
        }
    }
    cudaDeviceSynchronize();

    printf("Benchmarking (%d iters)...\n", iters);
    GpuTimer t;
    t.start();

    for (int i = 0; i < iters; ++i) {
        for (int m = 0; m < M; ++m) {
            float* kc_seq = d_kc + (size_t)m * kv_seq_stride;
            float* vc_seq = d_vc + (size_t)m * kv_seq_stride;
            process_seq(
                d_x32_arr[m], d_x32_arr[m],
                0, num_layers,
                lw.data(),
                d_x_i4_arr[m], d_x_i4_sc_arr[m],
                d_mlp_sc_buf,
                d_Q, d_K, d_V,
                d_attn, d_proj,
                d_attn_i4, d_attn_i4_sc,
                d_gate, d_up,
                d_mlp_i4, d_mlp_i4_sc,
                d_rn,
                kc_seq, vc_seq,
                H, Q, KV, I,
                nkv, hd, ms,
                kv_seq_stride,
                sq + 1, true
            );
        }
    }
    float ms_total = t.stop();

    float ms_per_iter = ms_total / iters;
    float ms_per_layer = ms_per_iter / num_layers;
    float ms_per_seq = ms_per_iter / M;
    float t_s = (float)M / (ms_per_seq / 1000.0f);
    float t_s_total = (float)(num_layers * M * iters) / (ms_total / 1000.0f);

    printf("\n=== Results ===\n");
    printf("Total time: %.3f ms (%d iters)\n", ms_total, iters);
    printf("Per iter:   %.3f ms\n", ms_per_iter);
    printf("Per layer:  %.3f ms\n", ms_per_layer);
    printf("Per seq:    %.3f ms\n", ms_per_seq);
    printf("Per-seq t/s: %.1f (M=%d sequences)\n", t_s, M);
    printf("Total t/s:  %.1f\n", t_s_total);
    printf("INT8 ref:   181.5 t/s\n");
    printf("Q4_K_M ref: 293.4 t/s\n");
    printf("vs INT8:    %.0f%%\n", t_s / 181.5f * 100.0f);
    printf("vs Q4_K_M:  %.0f%%\n", t_s / 293.4f * 100.0f);

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
    for (int m = 0; m < M; ++m) {
        cudaFree(d_x32_arr[m]);
        cudaFree(d_x_i4_arr[m]);
        cudaFree(d_x_i4_sc_arr[m]);
    }
    delete[] d_x32_arr; delete[] d_x_i4_arr; delete[] d_x_i4_sc_arr;
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_attn); cudaFree(d_proj);
    cudaFree(d_attn_i4); cudaFree(d_attn_i4_sc);
    cudaFree(d_gate); cudaFree(d_up);
    cudaFree(d_mlp_i4); cudaFree(d_mlp_i4_sc);
    cudaFree(d_mlp_sc_buf);
    cudaFree(d_rn); cudaFree(d_kc); cudaFree(d_vc);

    return 0;
}
