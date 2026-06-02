// bench/decode_int4_batched_attn.cu — INT4 batched decode with batched attention
//
// INT4 M-sequence decode where QKV GEMVs run batched AND attention_decode_batched_gqa
// processes all M sequences in one kernel call instead of M serial calls.
//
// Build:
//   /usr/local/cuda-13.3/bin/nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/decode_int4_batched_attn.cu build/libblackwell_kernels.a \
//     -o bench/decode_int4_batched_attn

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

// INT4 weight loader
struct DevW4 { int K, N; uint8_t* d; float* sc; };
static DevW4 upload_w4(const char* prefix) {
    char p[256]; snprintf(p,256,"%s.int4_t",prefix);
    FILE* f=fopen(p,"rb"); int h[5]; fread(h,4,5,f);
    DevW4 dw; dw.K=h[0]; dw.N=h[1];
    size_t ds=(size_t)h[0]*h[1]/2, ss=(size_t)h[3]*h[4];
    dw.d=(uint8_t*)malloc(ds); // temporary on host
    uint8_t* td=new uint8_t[ds]; fread(td,1,ds,f); fclose(f);
    cudaMalloc(&dw.d,ds); cudaMemcpy(dw.d,td,ds,cudaMemcpyHostToDevice); delete[] td;
    snprintf(p,256,"%s.scale_t",prefix); f=fopen(p,"rb"); fread(h,4,5,f);
    float* ts=new float[ss]; fread(ts,4,ss,f); fclose(f);
    cudaMalloc(&dw.sc,ss*4); cudaMemcpy(dw.sc,ts,ss*4,cudaMemcpyHostToDevice); delete[] ts;
    return dw;
}

struct LW4 { DevW4 q,k,v,o,g,u,d; };

int main(int argc, char** argv) {
    int num_layers = 4, M = 4;
    if (argc > 1) num_layers = atoi(argv[1]);
    if (argc > 2) M = atoi(argv[2]);
    if (num_layers > 28) num_layers = 28;
    if (M < 1) M = 1;
    if (M > 8) M = 8;

    const int H = 2048, Q = 2048, KV = 1024, I = 6144;
    const int nkv = 8, hd = 128, ms = 2048;
    const int nqh = 16;

    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("# INT4 Batched Attn Decode — M=%d\n", M);
    printf("Device: %s  Layers: %d\n", p.name, num_layers);

    // Load weights
    printf("Loading INT4 weights...\n");
    std::vector<LW4> lw(num_layers);
    for (int l = 0; l < num_layers; ++l) {
        char p[256];
        snprintf(p,256,"weights_int4_qwen3_1.7b/%d_self_attn.q_proj",l); lw[l].q=upload_w4(p);
        snprintf(p,256,"weights_int4_qwen3_1.7b/%d_self_attn.k_proj",l); lw[l].k=upload_w4(p);
        snprintf(p,256,"weights_int4_qwen3_1.7b/%d_self_attn.v_proj",l); lw[l].v=upload_w4(p);
        snprintf(p,256,"weights_int4_qwen3_1.7b/%d_self_attn.o_proj",l); lw[l].o=upload_w4(p);
        snprintf(p,256,"weights_int4_qwen3_1.7b/%d_mlp.gate_proj",l);  lw[l].g=upload_w4(p);
        snprintf(p,256,"weights_int4_qwen3_1.7b/%d_mlp.up_proj",l);    lw[l].u=upload_w4(p);
        snprintf(p,256,"weights_int4_qwen3_1.7b/%d_mlp.down_proj",l);   lw[l].d=upload_w4(p);
    }

    // ── Batched buffers [M][d] ─────────────────────────────────────
    float *d_x32, *d_Q_b, *d_K_b, *d_V_b, *d_attn_b, *d_proj_b;
    uint8_t *d_x_i4_b, *d_attn_i4_b;
    float *d_x_i4_sc_b, *d_attn_i4_sc_b;
    float *d_gate_b, *d_up_b, *d_mlp_sc_b;
    uint8_t *d_mlp_i4_b;
    float *d_mlp_i4_sc_b, *d_rn;

    // Batched buffers — contiguous [M] layout for gemv_int4_batched
    cudaMalloc(&d_x32, (size_t)M * H * 4);
    cudaMalloc(&d_x_i4_b, (size_t)M * (H/2));
    cudaMalloc(&d_x_i4_sc_b, (size_t)M * (H/16) * 4);
    cudaMalloc(&d_Q_b, (size_t)M * Q * 4);
    cudaMalloc(&d_K_b, (size_t)M * KV * 4);
    cudaMalloc(&d_V_b, (size_t)M * KV * 4);
    cudaMalloc(&d_attn_b, (size_t)M * Q * 4);
    cudaMalloc(&d_proj_b, (size_t)M * H * 4);
    cudaMalloc(&d_attn_i4_b, (size_t)M * (Q/2));
    cudaMalloc(&d_attn_i4_sc_b, (size_t)M * (Q/16) * 4);
    cudaMalloc(&d_gate_b, (size_t)M * I * 4);
    cudaMalloc(&d_up_b, (size_t)M * I * 4);
    cudaMalloc(&d_mlp_i4_b, (size_t)M * (I/2));
    cudaMalloc(&d_mlp_i4_sc_b, (size_t)M * (I/16) * 4);
    cudaMalloc(&d_mlp_sc_b, (size_t)(H/16) * 4);  // shared — only used per-seq in loop
    cudaMalloc(&d_rn, H * 4);

    std::vector<float> rn_h(H, 1.f), xh(H, 1.f);
    cudaMemcpy(d_rn, rn_h.data(), H*4, cudaMemcpyHostToDevice);
    // Init per-seq hidden states to all 1.0
    for (int m = 0; m < M; ++m)
        cudaMemcpy(d_x32 + (size_t)m * H, xh.data(), H*4, cudaMemcpyHostToDevice);

    std::vector<float> xsh(H/16, 1.f/7.f), ash(Q/16, 1.f/7.f), msh(I/16, 1.f/7.f);
    for (int m = 0; m < M; ++m) {
        cudaMemcpy(d_x_i4_sc_b + (size_t)m*(H/16), xsh.data(), (H/16)*4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_attn_i4_sc_b + (size_t)m*(Q/16), ash.data(), (Q/16)*4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_mlp_i4_sc_b + (size_t)m*(I/16), msh.data(), (I/16)*4, cudaMemcpyHostToDevice);
    }

    // ── KV cache ──────────────────────────────────────────────────
    size_t kv_seq_stride = (size_t)num_layers * nkv * ms * hd;  // floats per seq
    size_t kv_sz = (size_t)M * kv_seq_stride * 4;
    float *d_kc, *d_vc;
    cudaMalloc(&d_kc, kv_sz); cudaMalloc(&d_vc, kv_sz);
    cudaMemset(d_kc, 0, kv_sz); cudaMemset(d_vc, 0, kv_sz);

    // ── Fill KV cache ──────────────────────────────────────────────
    printf("Fill KV cache (M=%d, sq=0..128)... ", M); fflush(stdout);
    int sq = 128;
    for (int s = 0; s <= sq; ++s) {
        // Quantize all M seqs, then QKV batch, then KV update per-seq + attn per-seq
        for (int m = 0; m < M; ++m) {
            float* xs = d_x32 + (size_t)m * H;
            blackwell::kernels::quantize_int4(d_x_i4_b + (size_t)m*(H/2),
                d_x_i4_sc_b + (size_t)m*(H/16), xs, H, 0);
        }
        for (int l = 0; l < num_layers; ++l) {
            // QKV batched
            blackwell::kernels::gemv_int4_batched(d_Q_b + 0, d_x_i4_b, d_x_i4_sc_b,
                lw[l].q.d, lw[l].q.sc, H, Q, M, 0);
            blackwell::kernels::gemv_int4_batched(d_K_b + 0, d_x_i4_b, d_x_i4_sc_b,
                lw[l].k.d, lw[l].k.sc, H, KV, M, 0);
            blackwell::kernels::gemv_int4_batched(d_V_b + 0, d_x_i4_b, d_x_i4_sc_b,
                lw[l].v.d, lw[l].v.sc, H, KV, M, 0);

            size_t kv_layer_off = (size_t)l * nkv * ms * hd;  // floats
            for (int m = 0; m < M; ++m) {
                float* kc_seq = d_kc + (size_t)m * kv_seq_stride;
                float* vc_seq = d_vc + (size_t)m * kv_seq_stride;
                blackwell::kernels::update_kv_cache(
                    kc_seq + kv_layer_off, vc_seq + kv_layer_off,
                    d_K_b + (size_t)m * KV, d_V_b + (size_t)m * KV,
                    0, s, nkv, hd, ms, 0);
            }

            // Batched attention
            blackwell::kernels::attention_decode_batched_gqa(d_attn_b, d_Q_b,
                d_kc, d_vc, s, nqh, nkv, hd, ms, M,
                kv_seq_stride, kv_layer_off, 0);

            // Wo: quantize + batched
            for (int m = 0; m < M; ++m)
                blackwell::kernels::quantize_int4(d_attn_i4_b + (size_t)m*(Q/2),
                    d_attn_i4_sc_b + (size_t)m*(Q/16),
                    d_attn_b + (size_t)m * Q, Q, 0);
            blackwell::kernels::gemv_int4_batched(d_proj_b + 0, d_attn_i4_b, d_attn_i4_sc_b,
                lw[l].o.d, lw[l].o.sc, Q, H, M, 0);

            // Attention residual + RMSNorm + quant for MLP (per-seq)
            for (int m = 0; m < M; ++m) {
                float* xm = d_x32 + (size_t)m * H;
                // fused_residual_norm_int4: proj += residual, then norm + quant
                blackwell::kernels::fused_residual_norm_int4(
                    d_x_i4_b + (size_t)m*(H/2),
                    d_x_i4_sc_b + (size_t)m*(H/16),
                    d_proj_b + (size_t)m * H,  // in/out: Wo output, gets Wo+res
                    xm,                          // residual (x32 before attn)
                    d_rn, H, 1e-6f, 0);
            }

            // MLP gate + up (batched — input is x_i4_b which = quantized post-attn-norm)
            blackwell::kernels::gemv_int4_batched(d_gate_b + 0, d_x_i4_b, d_x_i4_sc_b,
                lw[l].g.d, lw[l].g.sc, H, I, M, 0);
            blackwell::kernels::gemv_int4_batched(d_up_b + 0, d_x_i4_b, d_x_i4_sc_b,
                lw[l].u.d, lw[l].u.sc, H, I, M, 0);

            // SwiGLU + quant (per-seq)
            for (int m = 0; m < M; ++m) {
                float* gm = d_gate_b + (size_t)m * I;
                float* um = d_up_b + (size_t)m * I;
                blackwell::kernels::fused_swiglu_quant_int4(
                    d_mlp_i4_b + (size_t)m*(I/2),
                    d_mlp_i4_sc_b + (size_t)m*(I/16),
                    gm, um, I, 0);
            }

            // MLP down (batched)
            blackwell::kernels::gemv_int4_batched(d_proj_b + 0, d_mlp_i4_b, d_mlp_i4_sc_b,
                lw[l].d.d, lw[l].d.sc, I, H, M, 0);

            // Scale restore + MLP residual + norm (per-seq)
            // fused kernel writes INT4 to d_x_i4_b, FP32 to xm
            for (int m = 0; m < M; ++m) {
                float* xm = d_x32 + (size_t)m * H;
                // fused_residual_norm_int4_fp32out: norm(down + xm) -> INT4(d_x_i4_b) + FP32(xm)
                blackwell::kernels::fused_residual_norm_int4_fp32out(
                    d_x_i4_b + (size_t)m*(H/2),  // INT4 output
                    d_x_i4_sc_b + (size_t)m*(H/16),  // scales (fresh)
                    xm,                              // FP32 normalized output
                    d_proj_b + (size_t)m * H,        // proj_in = MLP down
                    xm,                              // residual = pre-MLP state
                    d_rn, H, 1e-6f, 0);
            }
        }
    }
    printf("done\n");

    // Re-init inputs
    for (int m = 0; m < M; ++m)
        cudaMemcpy(d_x32 + (size_t)m * H, xh.data(), H*4, cudaMemcpyHostToDevice);

    // ── Benchmark ──────────────────────────────────────────────────
    const int warmup = 20, iters = 100;
    printf("Warmup (%d iters)...\n", warmup);

    for (int i = 0; i < warmup; ++i) {
        for (int l = 0; l < num_layers; ++l) {
            // Quantize all M
            for (int m = 0; m < M; ++m) {
                float* xm = d_x32 + (size_t)m * H;
                blackwell::kernels::quantize_int4(d_x_i4_b + (size_t)m*(H/2),
                    d_x_i4_sc_b + (size_t)m*(H/16), xm, H, 0);
            }
            // QKV batched
            blackwell::kernels::gemv_int4_batched(d_Q_b, d_x_i4_b, d_x_i4_sc_b,
                lw[l].q.d, lw[l].q.sc, H, Q, M, 0);
            blackwell::kernels::gemv_int4_batched(d_K_b, d_x_i4_b, d_x_i4_sc_b,
                lw[l].k.d, lw[l].k.sc, H, KV, M, 0);
            blackwell::kernels::gemv_int4_batched(d_V_b, d_x_i4_b, d_x_i4_sc_b,
                lw[l].v.d, lw[l].v.sc, H, KV, M, 0);

            size_t kv_layer_off = (size_t)l * nkv * ms * hd;
            for (int m = 0; m < M; ++m) {
                float* kc_seq = d_kc + (size_t)m * kv_seq_stride;
                float* vc_seq = d_vc + (size_t)m * kv_seq_stride;
                blackwell::kernels::update_kv_cache(
                    kc_seq + kv_layer_off, vc_seq + kv_layer_off,
                    d_K_b + (size_t)m * KV, d_V_b + (size_t)m * KV,
                    0, sq + 1, nkv, hd, ms, 0);
            }

            // Batched attention
            blackwell::kernels::attention_decode_batched_gqa(d_attn_b, d_Q_b,
                d_kc, d_vc, sq + 1, nqh, nkv, hd, ms, M,
                kv_seq_stride, kv_layer_off, 0);

            // Wo: quantize + batched
            for (int m = 0; m < M; ++m)
                blackwell::kernels::quantize_int4(d_attn_i4_b + (size_t)m*(Q/2),
                    d_attn_i4_sc_b + (size_t)m*(Q/16),
                    d_attn_b + (size_t)m * Q, Q, 0);
            blackwell::kernels::gemv_int4_batched(d_proj_b, d_attn_i4_b, d_attn_i4_sc_b,
                lw[l].o.d, lw[l].o.sc, Q, H, M, 0);

            // Attn residual + norm (per-seq)
            for (int m = 0; m < M; ++m) {
                float* xm = d_x32 + (size_t)m * H;
                blackwell::kernels::fused_residual_norm_int4(
                    d_x_i4_b + (size_t)m*(H/2),
                    d_x_i4_sc_b + (size_t)m*(H/16),
                    d_proj_b + (size_t)m * H, xm, d_rn, H, 1e-6f, 0);
            }

            // MLP gate + up (batched)
            blackwell::kernels::gemv_int4_batched(d_gate_b, d_x_i4_b, d_x_i4_sc_b,
                lw[l].g.d, lw[l].g.sc, H, I, M, 0);
            blackwell::kernels::gemv_int4_batched(d_up_b, d_x_i4_b, d_x_i4_sc_b,
                lw[l].u.d, lw[l].u.sc, H, I, M, 0);

            // SwiGLU + quant (per-seq)
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::fused_swiglu_quant_int4(
                    d_mlp_i4_b + (size_t)m*(I/2),
                    d_mlp_i4_sc_b + (size_t)m*(I/16),
                    d_gate_b + (size_t)m * I,
                    d_up_b + (size_t)m * I, I, 0);
            }

            // MLP down (batched)
            blackwell::kernels::gemv_int4_batched(d_proj_b, d_mlp_i4_b, d_mlp_i4_sc_b,
                lw[l].d.d, lw[l].d.sc, I, H, M, 0);

            // Scale restore + MLP residual + norm (per-seq)
            for (int m = 0; m < M; ++m) {
                float* xm = d_x32 + (size_t)m * H;
                blackwell::kernels::fused_residual_norm_int4_fp32out(
                    d_x_i4_b + (size_t)m*(H/2),
                    d_x_i4_sc_b + (size_t)m*(H/16),
                    xm,
                    d_proj_b + (size_t)m * H, xm,
                    d_rn, H, 1e-6f, 0);
            }
        }
    }
    cudaDeviceSynchronize();

    printf("Benchmark (%d iters)...\n", iters);
    GpuTimer t;
    t.start();
    for (int i = 0; i < iters; ++i) {
        for (int l = 0; l < num_layers; ++l) {
            // Quantize all M
            for (int m = 0; m < M; ++m) {
                float* xm = d_x32 + (size_t)m * H;
                blackwell::kernels::quantize_int4(d_x_i4_b + (size_t)m*(H/2),
                    d_x_i4_sc_b + (size_t)m*(H/16), xm, H, 0);
            }
            // QKV batched
            blackwell::kernels::gemv_int4_batched(d_Q_b, d_x_i4_b, d_x_i4_sc_b,
                lw[l].q.d, lw[l].q.sc, H, Q, M, 0);
            blackwell::kernels::gemv_int4_batched(d_K_b, d_x_i4_b, d_x_i4_sc_b,
                lw[l].k.d, lw[l].k.sc, H, KV, M, 0);
            blackwell::kernels::gemv_int4_batched(d_V_b, d_x_i4_b, d_x_i4_sc_b,
                lw[l].v.d, lw[l].v.sc, H, KV, M, 0);

            size_t kv_layer_off = (size_t)l * nkv * ms * hd;
            for (int m = 0; m < M; ++m) {
                float* kc_seq = d_kc + (size_t)m * kv_seq_stride;
                float* vc_seq = d_vc + (size_t)m * kv_seq_stride;
                blackwell::kernels::update_kv_cache(
                    kc_seq + kv_layer_off, vc_seq + kv_layer_off,
                    d_K_b + (size_t)m * KV, d_V_b + (size_t)m * KV,
                    0, sq + 1, nkv, hd, ms, 0);
            }

            // Batched attention
            blackwell::kernels::attention_decode_batched_gqa(d_attn_b, d_Q_b,
                d_kc, d_vc, sq + 1, nqh, nkv, hd, ms, M,
                kv_seq_stride, kv_layer_off, 0);

            // Wo: quantize + batched
            for (int m = 0; m < M; ++m)
                blackwell::kernels::quantize_int4(d_attn_i4_b + (size_t)m*(Q/2),
                    d_attn_i4_sc_b + (size_t)m*(Q/16),
                    d_attn_b + (size_t)m * Q, Q, 0);
            blackwell::kernels::gemv_int4_batched(d_proj_b, d_attn_i4_b, d_attn_i4_sc_b,
                lw[l].o.d, lw[l].o.sc, Q, H, M, 0);

            // Attn residual + norm (per-seq)
            for (int m = 0; m < M; ++m) {
                float* xm = d_x32 + (size_t)m * H;
                blackwell::kernels::fused_residual_norm_int4(
                    d_x_i4_b + (size_t)m*(H/2),
                    d_x_i4_sc_b + (size_t)m*(H/16),
                    d_proj_b + (size_t)m * H, xm, d_rn, H, 1e-6f, 0);
            }

            // MLP gate + up (batched)
            blackwell::kernels::gemv_int4_batched(d_gate_b, d_x_i4_b, d_x_i4_sc_b,
                lw[l].g.d, lw[l].g.sc, H, I, M, 0);
            blackwell::kernels::gemv_int4_batched(d_up_b, d_x_i4_b, d_x_i4_sc_b,
                lw[l].u.d, lw[l].u.sc, H, I, M, 0);

            // SwiGLU + quant (per-seq)
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::fused_swiglu_quant_int4(
                    d_mlp_i4_b + (size_t)m*(I/2),
                    d_mlp_i4_sc_b + (size_t)m*(I/16),
                    d_gate_b + (size_t)m * I,
                    d_up_b + (size_t)m * I, I, 0);
            }

            // MLP down (batched)
            blackwell::kernels::gemv_int4_batched(d_proj_b, d_mlp_i4_b, d_mlp_i4_sc_b,
                lw[l].d.d, lw[l].d.sc, I, H, M, 0);

            // Scale restore + MLP residual + norm (per-seq)
            for (int m = 0; m < M; ++m) {
                float* xm = d_x32 + (size_t)m * H;
                blackwell::kernels::fused_residual_norm_int4_fp32out(
                    d_x_i4_b + (size_t)m*(H/2),
                    d_x_i4_sc_b + (size_t)m*(H/16),
                    xm,
                    d_proj_b + (size_t)m * H, xm,
                    d_rn, H, 1e-6f, 0);
            }
        }
    }
    float ms_total = t.stop();

    float ms_per_iter = ms_total / iters;
    float ms_per_layer = ms_per_iter / num_layers;
    float ms_per_seq = ms_per_iter / M;
    float t_s = (float)M / (ms_per_seq / 1000.0f);
    float t_s_total = (float)(num_layers * M * iters) / (ms_total / 1000.0f);

    printf("\n=== Results ===\n");
    printf("Total:  %.3f ms (%d iters)\n", ms_total, iters);
    printf("Layer:  %.3f ms, Seq: %.3f ms\n", ms_per_layer, ms_per_seq);
    printf("Per-seq t/s: %.1f (M=%d)\n", t_s, M);
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
    cudaFree(d_x32); cudaFree(d_x_i4_b); cudaFree(d_x_i4_sc_b);
    cudaFree(d_Q_b); cudaFree(d_K_b); cudaFree(d_V_b);
    cudaFree(d_attn_b); cudaFree(d_proj_b);
    cudaFree(d_attn_i4_b); cudaFree(d_attn_i4_sc_b);
    cudaFree(d_gate_b); cudaFree(d_up_b);
    cudaFree(d_mlp_i4_b); cudaFree(d_mlp_i4_sc_b);
    cudaFree(d_mlp_sc_b); cudaFree(d_rn);
    cudaFree(d_kc); cudaFree(d_vc);
    return 0;
}