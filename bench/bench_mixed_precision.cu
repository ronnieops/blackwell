// bench/bench_mixed_precision.cu — Mixed-precision benchmark: INT8 attention + FP4 MLP
//
// Compares 3 decode setups for one full layer:
//   1. All INT8 (gemv_int8_warp) — baseline
//   2. All FP4 packed (gemv_fp4_warp) — pure FP4
//   3. Mixed: INT8 for QKV/O, FP4 for gate/up/down
//
// Build:
//   nvcc -O3 -std=c++17 -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/bench_mixed_precision.cu build/libblackwell_kernels.a \
//     -o bench/bench_mixed_precision

#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>
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

struct LoadedW { int K, N; std::vector<int8_t> d; std::vector<float> sc; };
static LoadedW load_int8_w(const char* prefix) {
    char p[256]; snprintf(p,256,"%s.int8_t",prefix);
    FILE* f = fopen(p,"rb"); if(!f){fprintf(stderr,"FAIL %s\n",p);exit(1);}
    int h[5]; fread(h,4,5,f);
    LoadedW w; w.K=h[0]; w.N=h[1]; w.d.resize(h[0]*h[1]); fread(w.d.data(),1,w.d.size(),f); fclose(f);
    snprintf(p,256,"%s.scale_t",prefix); f=fopen(p,"rb"); fread(h,4,5,f);
    w.sc.resize(h[3]*h[4]); fread(w.sc.data(),4,w.sc.size(),f); fclose(f);
    return w;
}

struct LoadedFP4 { int K, N; std::vector<uint8_t> packed; std::vector<float> sc; };
static LoadedFP4 load_fp4_w(const char* prefix) {
    char p[256]; snprintf(p,256,"%s.packed_fp4",prefix);
    FILE* f = fopen(p,"rb"); if(!f){fprintf(stderr,"FAIL %s\n",p);exit(1);}
    int h[5]; fread(h,4,5,f);
    LoadedFP4 w; w.K=h[0]; w.N=h[1]; w.packed.resize(h[4]); fclose(f);
    int n_sc = h[3] * (h[0]/16);
    f = fopen(p,"rb"); fread(h,4,5,f);
    fseek(f, 20 + h[4], SEEK_SET);
    w.sc.resize(h[3] * (h[0]/16));
    fread(w.sc.data(),4,w.sc.size(),f); fclose(f);
    return w;
}

int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("# Mixed-Precision Decode Benchmark — %s (%d.%d)\n", p.name, p.major, p.minor);
    printf("Layer 0, Qwen3-1.7B (H=2048, I=6144, Qheads=16, KVheads=8, hd=128)\n\n");

    const float s13 = 1.f/3.f;
    int iters = 1000;

    // ── Layer 0 weights ───────────────────────────────────────────────────
    printf("Loading weights...\n");
    auto wq_i8 = load_int8_w("weights_int8_bf16/0_self_attn.q_proj");
    auto wk_i8 = load_int8_w("weights_int8_bf16/0_self_attn.k_proj");
    auto wv_i8 = load_int8_w("weights_int8_bf16/0_self_attn.v_proj");
    auto wo_i8 = load_int8_w("weights_int8_bf16/0_self_attn.o_proj");

    auto wg_fp4 = load_fp4_w("weights_packed_fp4/0_mlp.gate_proj");
    auto wu_fp4 = load_fp4_w("weights_packed_fp4/0_mlp.up_proj");
    auto wd_fp4 = load_fp4_w("weights_packed_fp4/0_mlp.down_proj");
    auto wg_i8 = load_int8_w("weights_int8_bf16/0_mlp.gate_proj");
    auto wu_i8 = load_int8_w("weights_int8_bf16/0_mlp.up_proj");
    auto wd_i8 = load_int8_w("weights_int8_bf16/0_mlp.down_proj");

    const int H = 2048, Q = 2048, KV = 1024, I = 6144;

    // ═══ Device Memory ═══════════════════════════════════════════════════
    // INT8 weights (attention)
    int8_t *d_wq, *d_wk, *d_wv, *d_wo;
    float *d_wq_s, *d_wk_s, *d_wv_s, *d_wo_s;
    auto upload_i8 = [&](auto& w, auto*& d, auto*& ds) {
        cudaMalloc(&d, w.K*w.N); cudaMemcpy(d, w.d.data(), w.K*w.N, cudaMemcpyHostToDevice);
        cudaMalloc(&ds, w.sc.size()*4); cudaMemcpy(ds, w.sc.data(), w.sc.size()*4, cudaMemcpyHostToDevice);
    };
    upload_i8(wq_i8, d_wq, d_wq_s); upload_i8(wk_i8, d_wk, d_wk_s);
    upload_i8(wv_i8, d_wv, d_wv_s); upload_i8(wo_i8, d_wo, d_wo_s);

    // FP4 weights (MLP)
    uint8_t *d_wg4, *d_wu4, *d_wd4;
    float *d_wg4_s, *d_wu4_s, *d_wd4_s;
    auto upload_fp4 = [&](auto& w, auto*& d, auto*& ds) {
        cudaMalloc(&d, w.packed.size()); cudaMemcpy(d, w.packed.data(), w.packed.size(), cudaMemcpyHostToDevice);
        cudaMalloc(&ds, w.sc.size()*4); cudaMemcpy(ds, w.sc.data(), w.sc.size()*4, cudaMemcpyHostToDevice);
    };
    upload_fp4(wg_fp4, d_wg4, d_wg4_s); upload_fp4(wu_fp4, d_wu4, d_wu4_s);
    upload_fp4(wd_fp4, d_wd4, d_wd4_s);

    // INT8 weights (MLP, for all-INT8 baseline)
    int8_t *d_wg_i8, *d_wu_i8, *d_wd_i8;
    float *d_wg_i8s, *d_wu_i8s, *d_wd_i8s;
    upload_i8(wg_i8, d_wg_i8, d_wg_i8s);
    upload_i8(wu_i8, d_wu_i8, d_wu_i8s);
    upload_i8(wd_i8, d_wd_i8, d_wd_i8s);

    // Activations
    int8_t *d_xi8; float *d_xs_i8;
    uint8_t *d_xf4; float *d_xs_f4;
    float *d_x32, *d_y32;
    cudaMalloc(&d_xi8, H); cudaMalloc(&d_xs_i8, (H/16)*4);
    cudaMalloc(&d_xf4, H/2); cudaMalloc(&d_xs_f4, (H/16)*4);
    cudaMalloc(&d_x32, H*4);
    cudaMalloc(&d_y32, max(H,I)*4);

    std::vector<int8_t> xh_i8(H, 42);
    std::vector<float> xsh_i8(H/16, s13);
    cudaMemcpy(d_xi8, xh_i8.data(), H, cudaMemcpyHostToDevice);
    cudaMemcpy(d_xs_i8, xsh_i8.data(), (H/16)*4, cudaMemcpyHostToDevice);

    std::vector<uint8_t> xh_f4(H/2,0x11); // two 0.5 values
    std::vector<float> xsh_f4(H/16, 0.166666f);
    cudaMemcpy(d_xf4, xh_f4.data(), H/2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_xs_f4, xsh_f4.data(), (H/16)*4, cudaMemcpyHostToDevice);

    // Output buffers
    float *d_q, *d_k, *d_v, *d_attn, *d_o, *d_g, *d_u, *d_mlp, *d_tmp;
    cudaMalloc(&d_q, Q*4); cudaMalloc(&d_k, KV*4); cudaMalloc(&d_v, KV*4);
    cudaMalloc(&d_attn, Q*4); cudaMalloc(&d_o, H*4);
    cudaMalloc(&d_g, I*4); cudaMalloc(&d_u, I*4); cudaMalloc(&d_mlp, I*4);
    cudaMalloc(&d_tmp, max(H,I)*4);

    // ═══ Benchmark 1: All INT8 ═════════════════════════════════════════
    printf("\n=== All INT8 (baseline) ===\n");
    // Warmup
    for (int i = 0; i < 10; i++) {
        blackwell::kernels::gemv_int8_warp(d_q, d_xi8, d_xs_i8, d_wq, d_wq_s, H, Q, 0);
        blackwell::kernels::gemv_int8_warp(d_k, d_xi8, d_xs_i8, d_wk, d_wk_s, H, KV, 0);
        blackwell::kernels::gemv_int8_warp(d_v, d_xi8, d_xs_i8, d_wv, d_wv_s, H, KV, 0);
        // Simple attention approximation (just copy Q)
        cudaMemcpy(d_attn, d_q, Q*4, cudaMemcpyDeviceToDevice);
        blackwell::kernels::gemv_int8_warp(d_o, d_attn, d_xs_i8, d_wo, d_wo_s, Q, H, 0);
        blackwell::kernels::gemv_int8_warp(d_g, d_xi8, d_xs_i8, d_wg_i8, d_wg_i8s, H, I, 0);
        blackwell::kernels::gemv_int8_warp(d_u, d_xi8, d_xs_i8, d_wu_i8, d_wu_i8s, H, I, 0);
        blackwell::kernels::apply_swiglu(d_mlp, d_g, d_u, I, 0);
        blackwell::kernels::pack_int8(d_tmp, d_mlp, d_xs_i8, I, 0);
        blackwell::kernels::gemv_int8_warp(d_o, (int8_t*)d_tmp, d_xs_i8, d_wd_i8, d_wd_i8s, I, H, 0);
    }
    cudaDeviceSynchronize();

    // Benchmark
    GpuTimer ti;
    ti.start();
    for (int i = 0; i < iters; i++) {
        blackwell::kernels::gemv_int8_warp(d_q, d_xi8, d_xs_i8, d_wq, d_wq_s, H, Q, 0);
        blackwell::kernels::gemv_int8_warp(d_k, d_xi8, d_xs_i8, d_wk, d_wk_s, H, KV, 0);
        blackwell::kernels::gemv_int8_warp(d_v, d_xi8, d_xs_i8, d_wv, d_wv_s, H, KV, 0);
        cudaMemcpy(d_attn, d_q, Q*4, cudaMemcpyDeviceToDevice);
        blackwell::kernels::gemv_int8_warp(d_o, d_attn, d_xs_i8, d_wo, d_wo_s, Q, H, 0);
        blackwell::kernels::gemv_int8_warp(d_g, d_xi8, d_xs_i8, d_wg_i8, d_wg_i8s, H, I, 0);
        blackwell::kernels::gemv_int8_warp(d_u, d_xi8, d_xs_i8, d_wu_i8, d_wu_i8s, H, I, 0);
        blackwell::kernels::apply_swiglu(d_mlp, d_g, d_u, I, 0);
        blackwell::kernels::pack_int8(d_tmp, d_mlp, d_xs_i8, I, 0);
        blackwell::kernels::gemv_int8_warp(d_o, (int8_t*)d_tmp, d_xs_i8, d_wd_i8, d_wd_i8s, I, H, 0);
    }
    float ms_i8 = ti.stop();
    printf("  Per-layer: %.3f ms\n", ms_i8 / iters);
    printf("  28L est:   %.1f t/s\n", 1000.f / (ms_i8 / iters * 28));

    // ═══ Benchmark 2: All FP4 ═══════════════════════════════════════════
    printf("\n=== All FP4 Packed ===\n");
    for (int i = 0; i < 10; i++) {
        blackwell::kernels::gemv_fp4_warp(d_q, d_xf4, d_xs_f4, (uint8_t*)d_wq, d_wq_s, H, Q, 0);
        blackwell::kernels::gemv_fp4_warp(d_k, d_xf4, d_xs_f4, (uint8_t*)d_wk, d_wk_s, H, KV, 0);
        blackwell::kernels::gemv_fp4_warp(d_v, d_xf4, d_xs_f4, (uint8_t*)d_wv, d_wv_s, H, KV, 0);
        cudaMemcpy(d_attn, d_q, Q*4, cudaMemcpyDeviceToDevice);
        blackwell::kernels::gemv_fp4_warp(d_o, d_attn, d_xs_f4, (uint8_t*)d_wo, d_wo_s, Q, H, 0);
        blackwell::kernels::gemv_fp4_warp(d_g, d_xf4, d_xs_f4, d_wg4, d_wg4_s, H, I, 0);
        blackwell::kernels::gemv_fp4_warp(d_u, d_xf4, d_xs_f4, d_wu4, d_wu4_s, H, I, 0);
        blackwell::kernels::apply_swiglu(d_mlp, d_g, d_u, I, 0);
        cudaMemcpy(d_tmp, d_mlp, I*4, cudaMemcpyDeviceToDevice);
        blackwell::kernels::gemv_fp4_warp(d_o, (uint8_t*)d_tmp, d_xs_f4, d_wd4, d_wd4_s, I, H, 0);
    }
    cudaDeviceSynchronize();

    GpuTimer tf4;
    tf4.start();
    for (int i = 0; i < iters; i++) {
        blackwell::kernels::gemv_fp4_warp(d_q, d_xf4, d_xs_f4, (uint8_t*)d_wq, d_wq_s, H, Q, 0);
        blackwell::kernels::gemv_fp4_warp(d_k, d_xf4, d_xs_f4, (uint8_t*)d_wk, d_wk_s, H, KV, 0);
        blackwell::kernels::gemv_fp4_warp(d_v, d_xf4, d_xs_f4, (uint8_t*)d_wv, d_wv_s, H, KV, 0);
        cudaMemcpy(d_attn, d_q, Q*4, cudaMemcpyDeviceToDevice);
        blackwell::kernels::gemv_fp4_warp(d_o, d_attn, d_xs_f4, (uint8_t*)d_wo, d_wo_s, Q, H, 0);
        blackwell::kernels::gemv_fp4_warp(d_g, d_xf4, d_xs_f4, d_wg4, d_wg4_s, H, I, 0);
        blackwell::kernels::gemv_fp4_warp(d_u, d_xf4, d_xs_f4, d_wu4, d_wu4_s, H, I, 0);
        blackwell::kernels::apply_swiglu(d_mlp, d_g, d_u, I, 0);
        cudaMemcpy(d_tmp, d_mlp, I*4, cudaMemcpyDeviceToDevice);
        blackwell::kernels::gemv_fp4_warp(d_o, (uint8_t*)d_tmp, d_xs_f4, d_wd4, d_wd4_s, I, H, 0);
    }
    float ms_f4 = tf4.stop();
    printf("  Per-layer: %.3f ms\n", ms_f4 / iters);
    printf("  28L est:   %.1f t/s\n", 1000.f / (ms_f4 / iters * 28));

    // ═══ Benchmark 3: Mixed INT8 attention + FP4 MLP ═══════════════════
    printf("\n=== Mixed: INT8 Attn + FP4 MLP ===\n");
    for (int i = 0; i < 10; i++) {
        blackwell::kernels::gemv_int8_warp(d_q, d_xi8, d_xs_i8, d_wq, d_wq_s, H, Q, 0);
        blackwell::kernels::gemv_int8_warp(d_k, d_xi8, d_xs_i8, d_wk, d_wk_s, H, KV, 0);
        blackwell::kernels::gemv_int8_warp(d_v, d_xi8, d_xs_i8, d_wv, d_wv_s, H, KV, 0);
        cudaMemcpy(d_attn, d_q, Q*4, cudaMemcpyDeviceToDevice);
        blackwell::kernels::gemv_int8_warp(d_o, d_attn, d_xs_i8, d_wo, d_wo_s, Q, H, 0);
        blackwell::kernels::gemv_fp4_warp(d_g, d_xf4, d_xs_f4, d_wg4, d_wg4_s, H, I, 0);
        blackwell::kernels::gemv_fp4_warp(d_u, d_xf4, d_xs_f4, d_wu4, d_wu4_s, H, I, 0);
        blackwell::kernels::apply_swiglu(d_mlp, d_g, d_u, I, 0);
        cudaMemcpy(d_tmp, d_mlp, I*4, cudaMemcpyDeviceToDevice);
        blackwell::kernels::gemv_fp4_warp(d_o, (uint8_t*)d_tmp, d_xs_f4, d_wd4, d_wd4_s, I, H, 0);
    }
    cudaDeviceSynchronize();

    GpuTimer tm;
    tm.start();
    for (int i = 0; i < iters; i++) {
        blackwell::kernels::gemv_int8_warp(d_q, d_xi8, d_xs_i8, d_wq, d_wq_s, H, Q, 0);
        blackwell::kernels::gemv_int8_warp(d_k, d_xi8, d_xs_i8, d_wk, d_wk_s, H, KV, 0);
        blackwell::kernels::gemv_int8_warp(d_v, d_xi8, d_xs_i8, d_wv, d_wv_s, H, KV, 0);
        cudaMemcpy(d_attn, d_q, Q*4, cudaMemcpyDeviceToDevice);
        blackwell::kernels::gemv_int8_warp(d_o, d_attn, d_xs_i8, d_wo, d_wo_s, Q, H, 0);
        blackwell::kernels::gemv_fp4_warp(d_g, d_xf4, d_xs_f4, d_wg4, d_wg4_s, H, I, 0);
        blackwell::kernels::gemv_fp4_warp(d_u, d_xf4, d_xs_f4, d_wu4, d_wu4_s, H, I, 0);
        blackwell::kernels::apply_swiglu(d_mlp, d_g, d_u, I, 0);
        cudaMemcpy(d_tmp, d_mlp, I*4, cudaMemcpyDeviceToDevice);
        blackwell::kernels::gemv_fp4_warp(d_o, (uint8_t*)d_tmp, d_xs_f4, d_wd4, d_wd4_s, I, H, 0);
    }
    float ms_mixed = tm.stop();
    printf("  Per-layer: %.3f ms\n", ms_mixed / iters);
    printf("  28L est:   %.1f t/s\n", 1000.f / (ms_mixed / iters * 28));

    // ═══ Results ════════════════════════════════════════════════════════
    printf("\n╔═══════════════════════╦════════════╦═══════════╗\n");
    printf("║ Method               ║ Per-Layer  ║ 28L t/s  ║\n");
    printf("╠═══════════════════════╬════════════╬═══════════╣\n");
    printf("║ All INT8 (warp)      ║ %8.3f ms ║ %9.1f ║\n", ms_i8/iters, 1000.f/(ms_i8/iters*28));
    printf("║ All FP4 packed       ║ %8.3f ms ║ %9.1f ║\n", ms_f4/iters, 1000.f/(ms_f4/iters*28));
    printf("║ Mixed: INT8+FP4      ║ %8.3f ms ║ %9.1f ║\n", ms_mixed/iters, 1000.f/(ms_mixed/iters*28));
    printf("╚═══════════════════════╩════════════╩═══════════╝\n");

    return 0;
}
