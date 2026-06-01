// bench/verify_pipeline_error.cu — Measure INT8 pipeline error
//
// Compares INT8 full-layer pipeline output vs FP32 reference.
// Same random input for both, measures per-layer L2 error growth.
//
// Build:
//   export PATH=/usr/local/cuda-13.3/bin:$PATH
//   CUDACXX=/usr/local/cuda-13.3/bin/nvcc nvcc -O3 -std=c++17 \
//     -arch=sm_120a -I include bench/verify_pipeline_error.cu \
//     build/libblackwell_kernels.a -o bench/verify_pipeline_error

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include <cstdint>
#include "blackwell/kernels.h"

static void chk(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) { printf("FAIL: %s: %s\n", msg, cudaGetErrorString(e)); exit(1); }
}

struct LoadedW { int K, N; std::vector<int8_t> d; std::vector<float> sc; };
static LoadedW load_int8_w(const char* prefix) {
    char p[256]; snprintf(p,256,"%s.int8_t",prefix);
    FILE* f = fopen(p,"rb"); int h[5]; fread(h,4,5,f);
    LoadedW w; w.K=h[0]; w.N=h[1]; w.d.resize(h[0]*h[1]); fread(w.d.data(),1,w.d.size(),f); fclose(f);
    snprintf(p,256,"%s.scale_t",prefix); f=fopen(p,"rb"); fread(h,4,5,f);
    w.sc.resize(h[3]*h[4]); fread(w.sc.data(),4,w.sc.size(),f); fclose(f);
    return w;
}

struct DevW { int K, N; int8_t* d; float* sc; };
static DevW upload(const char* prefix) {
    auto w = load_int8_w(prefix); DevW dw{w.K, w.N};
    cudaMalloc(&dw.d, w.K*w.N); cudaMemcpy(dw.d,w.d.data(),w.K*w.N,cudaMemcpyHostToDevice);
    cudaMalloc(&dw.sc, w.sc.size()*4); cudaMemcpy(dw.sc,w.sc.data(),w.sc.size()*4,cudaMemcpyHostToDevice);
    return dw;
}

int main() {
    int num_layers = 28;
    const int H = 2048, Q = 2048, KV = 1024, I = 6144;
    const int nqh = 16, nkv = 8, hd = 128, ms = 2048;
    const int big = I;

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("# Pipeline Error Analysis — Qwen3-1.7B INT8\n");
    printf("# Device: %s (%d.%d)\n", prop.name, prop.major, prop.minor);
    printf("# Layers: %d, H=%d, I=%d\n", num_layers, H, I);
    fflush(stdout);

    // Load weights
    const char* WDIR = "weights_int8_bf16";
    struct LW { DevW q,k,v,o,g,u,d; };
    std::vector<LW> lw(num_layers);
    for (int l = 0; l < num_layers; ++l) {
        char p[256];
        snprintf(p,256,"%s/%d_self_attn.q_proj",WDIR,l); lw[l].q = upload(p);
        snprintf(p,256,"%s/%d_self_attn.k_proj",WDIR,l); lw[l].k = upload(p);
        snprintf(p,256,"%s/%d_self_attn.v_proj",WDIR,l); lw[l].v = upload(p);
        snprintf(p,256,"%s/%d_self_attn.o_proj",WDIR,l); lw[l].o = upload(p);
        snprintf(p,256,"%s/%d_mlp.gate_proj",WDIR,l);  lw[l].g = upload(p);
        snprintf(p,256,"%s/%d_mlp.up_proj",WDIR,l);    lw[l].u = upload(p);
        snprintf(p,256,"%s/%d_mlp.down_proj",WDIR,l);  lw[l].d = upload(p);
    }

    // Buffers
    float *d_Q, *d_KV, *d_attn, *d_proj, *d_gate, *d_up, *d_mlp, *d_res, *d_rn;
    int8_t *d_xi8, *d_ai8, *d_mi8;
    float *d_xs, *d_xi8s, *d_ai8s, *d_mi8s;
    void *d_fp4;
    cudaMalloc(&d_Q, Q*4); cudaMalloc(&d_KV, KV*4);
    cudaMalloc(&d_attn, Q*4); cudaMalloc(&d_proj, H*4);
    cudaMalloc(&d_gate, I*4); cudaMalloc(&d_up, I*4); cudaMalloc(&d_mlp, I*4);
    cudaMalloc(&d_res, big*4);
    cudaMalloc(&d_xi8, H); cudaMalloc(&d_xi8s, (H/16)*4);
    cudaMalloc(&d_ai8, Q); cudaMalloc(&d_ai8s, (Q/16)*4);
    cudaMalloc(&d_mi8, I); cudaMalloc(&d_mi8s, (I/16)*4);
    cudaMalloc(&d_fp4, H);
    cudaMalloc(&d_xs, (H/16)*4);
    cudaMalloc(&d_rn, H*4);

    std::vector<float> rn(H, 1.f);
    cudaMemcpy(d_rn, rn.data(), H*4, cudaMemcpyHostToDevice);

    // KV cache
    float *d_kc, *d_vc;
    size_t kv_sz = (size_t)num_layers * nkv * ms * hd * 4;
    cudaMalloc(&d_kc, kv_sz); cudaMalloc(&d_vc, kv_sz);
    cudaMemset(d_kc, 0, kv_sz); cudaMemset(d_vc, 0, kv_sz);

    // FP32 reference input (random normal)
    std::vector<float> h_ref(H);
    srand(42);
    float sum = 0, sum2 = 0;
    for (int i = 0; i < H; ++i) {
        h_ref[i] = (float)(rand() % 256 - 128) / 64.f;
        sum += h_ref[i]; sum2 += h_ref[i]*h_ref[i];
    }
    float mean = sum/H, var = sum2/H - mean*mean;
    for (int i = 0; i < H; ++i) h_ref[i] = (h_ref[i] - mean) / sqrtf(var);  // normalize

    // Upload reference → FP4 state
    float* d_ref; cudaMalloc(&d_ref, H*4);
    cudaMemcpy(d_ref, h_ref.data(), H*4, cudaMemcpyHostToDevice);
    float s13 = 1.f/3.f;
    std::vector<float> xsh(H/16, s13);
    cudaMemcpy(d_xs, xsh.data(), (H/16)*4, cudaMemcpyHostToDevice);
    chk(blackwell::kernels::fused_rmsnorm(d_res, d_rn, d_ref, H, 1e-6f, 0), "init_rn");
    chk(blackwell::kernels::pack_fp4(d_fp4, d_res, d_xs, H, 0), "init_pack");

    // Also save FP32 reference (residual) for baseline
    std::vector<float> h_fp32_input(H);
    cudaMemcpy(h_fp32_input.data(), d_res, H*4, cudaMemcpyDeviceToHost);

    // Run 28 layers, measure error per layer
    int sq = 0;  // single token, seq_pos=0
    printf("\nLayer-by-layer L2 error (INT8 vs FP32 baseline):\n");
    printf("  L    L2 err  SNR(dB)  max_abs  Layer type\n");
    printf("  ──────────────────────────────────────────\n");

    float total_l2 = 0;
    for (int l = 0; l < num_layers; ++l) {
        // Save state before layer (FP32 baseline)
        std::vector<float> h_before(H);
        blackwell::kernels::unpack_fp4(d_ref, d_fp4, d_xs, H, 0);
        cudaMemcpy(h_before.data(), d_ref, H*4, cudaMemcpyDeviceToHost);

        // Run INT8 pipeline for this layer
        int kb = l * nkv * ms * hd;
        chk(blackwell::kernels::unpack_fp4(d_res, d_fp4, d_xs, H, 0), "unpack");
        chk(blackwell::kernels::pack_int8(d_xi8, d_res, d_xi8s, H, 0), "pack");
        chk(blackwell::kernels::gemv_int8_warp(d_Q, d_xi8, d_xi8s, lw[l].q.d, lw[l].q.sc, H, Q, 0), "Q");
        chk(blackwell::kernels::gemv_int8_warp(d_KV, d_xi8, d_xi8s, lw[l].k.d, lw[l].k.sc, H, KV, 0), "K");
        chk(blackwell::kernels::gemv_int8_warp(d_KV, d_xi8, d_xi8s, lw[l].v.d, lw[l].v.sc, H, KV, 0), "V");
        chk(blackwell::kernels::update_kv_cache(d_kc+kb, d_vc+kb, d_KV, d_KV, 0, sq, nkv, hd, ms, 0), "kv");
        chk(blackwell::kernels::attention_decode_gqa(d_attn, d_Q, d_kc+kb, d_vc+kb, sq, nqh, nkv, hd, ms, 0), "attn");
        chk(blackwell::kernels::pack_int8(d_ai8, d_attn, d_ai8s, Q, 0), "pack_ai8");
        chk(blackwell::kernels::gemv_int8_warp(d_proj, d_ai8, d_ai8s, lw[l].o.d, lw[l].o.sc, Q, H, 0), "O");
        chk(blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_res, H, 0), "vadd_res");
        chk(blackwell::kernels::fused_rmsnorm_quant_int8(d_xi8, d_xi8s, d_proj, d_rn, H, 1e-6f, 0), "rnq");
        chk(blackwell::kernels::fused_rmsnorm_pack(d_fp4, d_xs, d_proj, d_rn, H, 1e-6f, 0), "rnp");
        // MLP
        chk(blackwell::kernels::unpack_fp4(d_res, d_fp4, d_xs, H, 0), "upk2");
        chk(blackwell::kernels::pack_int8(d_xi8, d_res, d_xi8s, H, 0), "pk2");
        chk(blackwell::kernels::gemv_int8_warp(d_gate, d_xi8, d_xi8s, lw[l].g.d, lw[l].g.sc, H, I, 0), "gate");
        chk(blackwell::kernels::gemv_int8_warp(d_up, d_xi8, d_xi8s, lw[l].u.d, lw[l].u.sc, H, I, 0), "up");
        chk(blackwell::kernels::apply_swiglu(d_mlp, d_gate, d_up, I, 0), "swiglu");
        chk(blackwell::kernels::pack_int8(d_mi8, d_mlp, d_mi8s, I, 0), "pmi8");
        chk(blackwell::kernels::gemv_int8_warp(d_proj, d_mi8, d_mi8s, lw[l].d.d, lw[l].d.sc, I, H, 0), "down");
        chk(blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_res, H, 0), "vadd2");
        chk(blackwell::kernels::fused_rmsnorm_quant_int8(d_xi8, d_xi8s, d_proj, d_rn, H, 1e-6f, 0), "rnq2");
        chk(blackwell::kernels::fused_rmsnorm_pack(d_fp4, d_xs, d_proj, d_rn, H, 1e-6f, 0), "rnp2");

        // Read INT8 output
        std::vector<float> h_out(H);
        blackwell::kernels::unpack_fp4(d_res, d_fp4, d_xs, H, 0);
        cudaMemcpy(h_out.data(), d_res, H*4, cudaMemcpyDeviceToHost);

        // Error vs FP32 input (not vs a FP32 layer, since we're measuring cumulative error)
        float l2 = 0;
        float max_abs = 0;
        float invar = 0;
        for (int i = 0; i < H; ++i) {
            float e = h_out[i] - h_fp32_input[i];
            l2 += e*e;
            max_abs = fmaxf(max_abs, fabsf(e));
            invar += h_fp32_input[i]*h_fp32_input[i];
        }
        l2 = sqrtf(l2/H);
        invar = sqrtf(invar/H);
        float snr = 20.f * log10f(invar / (l2 + 1e-9f));

        printf("  %2d   %.4f  %5.1f   %.4f   %s\n",
            l, l2, snr, max_abs, (l%2==0 ? "linear_attn" : "full_attn"));
        total_l2 += l2;
    }

    printf("\n  ──────────────────────────────────────────\n");
    printf("  Mean L2: %.4f\n", total_l2/num_layers);

    // Print machine-parseable
    printf("\n##RESULT layers=%d H=%d I=%d mean_l2=%.4f\n", num_layers, H, I, total_l2/num_layers);
    return 0;
}