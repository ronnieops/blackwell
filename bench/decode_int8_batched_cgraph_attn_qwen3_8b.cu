// bench/decode_int8_batched_cgraph_attn_qwen3_8b.cu
//
// INT8 M-sequence decode with BATCHED attention + CUDA Graph for Qwen3-8B.
// Batched attention fuses M×nqh Q/K/V into single kernel call.
// Batched GEMV fuses M projections (gate, up, down) into single call.
//
// Build:
//   export PATH=/usr/local/cuda-13.3/bin:$PATH
//   CUDACXX=/usr/local/cuda-13.3/bin/nvcc nvcc -O3 -std=c++17 \
//     -arch=sm_120a -I include bench/decode_int8_batched_cgraph_attn_qwen3_8b.cu \
//     build/libblackwell_kernels.a -o bench/decode_int8_batched_cgraph_attn_qwen3_8b
//
// Usage:
//   ./bench/decode_int8_batched_cgraph_attn_qwen3_8b <num_layers> <M>

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

int main(int argc, char** argv) {
    int num_layers = 4;
    int M = 4;
    if (argc > 1) num_layers = atoi(argv[1]);
    if (argc > 2) M = atoi(argv[2]);
    if (num_layers > 36) num_layers = 36;
    if (M < 1) M = 1;
    if (M > 8) M = 8;

    // Qwen3-8B dimensions
    const int H = 4096, Q = 4096, KV = 1024, I = 12288;
    const int nqh = 32, nkv = 8, hd = 128, ms = 2048;
    const float s13 = 1.f/3.f;
    const int big = (H > I) ? H : I;  // max(H,I) for residual buffers
    const char* WDIR = "weights_int8_qwen3_8b";

    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("# INT8 Batched Decode — Qwen3-8B (Batched Attention + CUDA Graph)\n");
    printf("# Device: %s (%d.%d)\n", p.name, p.major, p.minor);
    printf("# Layers: %d, Batch M: %d, H=%d, Q=%d, KV=%d, I=%d\n", num_layers, M, H, Q, KV, I);
    printf("# nqh=%d, nkv=%d, head_dim=%d\n", nqh, nkv, hd);
    fflush(stdout);

    // Load weights
    struct LW { DevW q,k,v,o,g,u,d; };
    printf("Loading INT8 weights from %s...\n", WDIR); fflush(stdout);
    std::vector<LW> lw(num_layers);
    for (int l = 0; l < num_layers; ++l) {
        char pp[256];
        snprintf(pp,256,"%s/%d_self_attn.q_proj",WDIR,l); lw[l].q = upload(pp);
        snprintf(pp,256,"%s/%d_self_attn.k_proj",WDIR,l); lw[l].k = upload(pp);
        snprintf(pp,256,"%s/%d_self_attn.v_proj",WDIR,l); lw[l].v = upload(pp);
        snprintf(pp,256,"%s/%d_self_attn.o_proj",WDIR,l); lw[l].o = upload(pp);
        snprintf(pp,256,"%s/%d_mlp.gate_proj",WDIR,l);  lw[l].g = upload(pp);
        snprintf(pp,256,"%s/%d_mlp.up_proj",WDIR,l);    lw[l].u = upload(pp);
        snprintf(pp,256,"%s/%d_mlp.down_proj",WDIR,l);  lw[l].d = upload(pp);
    }

    // ── Per-seq state (FP4) ──────────────────────────────────────────────────
    void **d_x_fp4_arr = new void*[M];
    float **d_xs_arr = new float*[M];
    for (int m = 0; m < M; ++m) {
        cudaMalloc(&d_x_fp4_arr[m], H);
        cudaMalloc(&d_xs_arr[m], (H/16)*4);
    }

    // ── Batched GEMV buffers (M × dim) ──────────────────────────────────────
    // Q/K/V projections: batched per-sequence
    float *d_Q_b, *d_K_b, *d_V_b;
    // Attention output + O proj
    float *d_attn_b, *d_proj_b;
    // MLP intermediate
    float *d_gate_b, *d_up_b, *d_mlp_b;
    // Residual (max(H,I) for MLP)
    float *d_res_b;
    // INT8 inputs/outputs (packed from residual)
    int8_t *d_xi8_b;
    float *d_xi8s_b;
    int8_t *d_attn_i8_b;
    float *d_attn_i8s_b;
    int8_t *d_mlp_i8_b;
    float *d_mlp_i8s_b;
    // Per-seq unpack intermediate (H elements)
    float *d_res_s;
    int8_t *d_xi8_s;
    float *d_xi8s_s;

    cudaMalloc(&d_Q_b, M * Q * 4);
    cudaMalloc(&d_K_b, M * KV * 4);
    cudaMalloc(&d_V_b, M * KV * 4);
    cudaMalloc(&d_attn_b, M * Q * 4);
    cudaMalloc(&d_proj_b, M * H * 4);
    cudaMalloc(&d_gate_b, M * I * 4);
    cudaMalloc(&d_up_b, M * I * 4);
    cudaMalloc(&d_mlp_b, M * I * 4);
    cudaMalloc(&d_res_b, M * big * 4);
    cudaMalloc(&d_xi8_b, M * H);
    cudaMalloc(&d_xi8s_b, M * (H/16) * 4);
    cudaMalloc(&d_attn_i8_b, M * Q);
    cudaMalloc(&d_attn_i8s_b, M * (Q/16) * 4);
    cudaMalloc(&d_mlp_i8_b, M * I);
    cudaMalloc(&d_mlp_i8s_b, M * (I/16) * 4);
    cudaMalloc(&d_res_s, big * 4);   // shared per-seq unpack buffer
    cudaMalloc(&d_xi8_s, big);       // shared per-seq int8 input
    cudaMalloc(&d_xi8s_s, (big/16) * 4);
    // Per-seq attn output (for O projection)
    float *d_attn_out_s;
    cudaMalloc(&d_attn_out_s, Q * 4);

    // RMSNorm weight
    float* d_rn; cudaMalloc(&d_rn, big * 4);
    std::vector<float> rn(big, 1.f);
    cudaMemcpy(d_rn, rn.data(), big * 4, cudaMemcpyHostToDevice);

    // Init FP4 state for all sequences
    float *d_x32_tmp; cudaMalloc(&d_x32_tmp, H*4);
    std::vector<float> xh(H, 1.f), xsh(H/16, s13);
    cudaMemcpy(d_x32_tmp, xh.data(), H*4, cudaMemcpyHostToDevice);
    for (int m = 0; m < M; ++m) {
        cudaMemcpy(d_xs_arr[m], xsh.data(), (H/16)*4, cudaMemcpyHostToDevice);
        chk(blackwell::kernels::pack_fp4(d_x_fp4_arr[m], d_x32_tmp, d_xs_arr[m], H, 0), "init_pack");
    }
    cudaFree(d_x32_tmp);

    // Init scales
    float ixv = 1.f/127.f;
    std::vector<float> ixsh(H/16, ixv), ai8s(Q/16, ixv), mi8s(I/16, ixv), bixsh(big/16, ixv);
    float *d_bixsh; cudaMalloc(&d_bixsh, (big/16)*4); cudaMemcpy(d_bixsh, bixsh.data(), (big/16)*4, cudaMemcpyHostToDevice);
    for (int m = 0; m < M; ++m) {
        cudaMemcpy(d_xi8s_b + m*(H/16), ixsh.data(), (H/16)*4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_attn_i8s_b + m*(Q/16), ai8s.data(), (Q/16)*4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_mlp_i8s_b + m*(I/16), mi8s.data(), (I/16)*4, cudaMemcpyHostToDevice);
    }
    cudaFree(d_bixsh);

    // ── KV cache: [M][total_layers][nkv][ms][hd] contiguous ─────────────────
    float *d_kc, *d_vc;
    size_t kv_sz = (size_t)M * num_layers * nkv * ms * hd * 4;
    cudaMalloc(&d_kc, kv_sz); cudaMalloc(&d_vc, kv_sz);
    cudaMemset(d_kc, 0, kv_sz); cudaMemset(d_vc, 0, kv_sz);
    size_t kv_seq_stride = (size_t)num_layers * nkv * ms * hd;  // floats per seq

    // L2 persisting cache
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 8*1024*1024);

    // ── Fill KV cache (seq=0..128) ─────────────────────────────────────────
    printf("Filling KV cache (%d sequences, seq=0..128)... ", M); fflush(stdout);
    int sq = 128;
    for (int s = 0; s <= sq; ++s) {
        for (int m = 0; m < M; ++m) {
            for (int l = 0; l < num_layers; ++l) {
                size_t km = m * kv_seq_stride + (size_t)l * nkv * ms * hd;

                // Unpack FP4 → FP32 residual
                blackwell::kernels::unpack_fp4(d_res_s, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
                // Quantize to INT8 for GEMV
                blackwell::kernels::pack_int8(d_xi8_s, d_res_s, d_xi8s_s, H, 0);

                // QKV projections (per-sequence, H=Q=4096 > batched sweet spot)
                blackwell::kernels::gemv_int8_warp(d_Q_b + m*Q, d_xi8_s, d_xi8s_s,
                    lw[l].q.d, lw[l].q.sc, H, Q, 0);
                blackwell::kernels::gemv_int8_warp(d_K_b + m*KV, d_xi8_s, d_xi8s_s,
                    lw[l].k.d, lw[l].k.sc, H, KV, 0);
                blackwell::kernels::gemv_int8_warp(d_V_b + m*KV, d_xi8_s, d_xi8s_s,
                    lw[l].v.d, lw[l].v.sc, H, KV, 0);

                // Update KV cache
                blackwell::kernels::update_kv_cache(
                    d_kc + km, d_vc + km,
                    d_K_b + m*KV, d_V_b + m*KV,
                    0, s, nkv, hd, ms, 0);

                // Attention (serial per-seq — for KV fill correctness)
                blackwell::kernels::attention_decode_gqa(
                    d_attn_out_s, d_Q_b + m*Q,
                    d_kc + km, d_vc + km,
                    s, nqh, nkv, hd, ms, 0);

                // O projection: quantize → INT8 → GEMV → FP32
                blackwell::kernels::pack_int8(d_attn_i8_b + m*Q, d_attn_out_s, d_attn_i8s_b + m*(Q/16), Q, 0);
                blackwell::kernels::gemv_int8_warp(
                    d_proj_b + m*H, d_attn_i8_b + m*Q, d_attn_i8s_b + m*(Q/16),
                    lw[l].o.d, lw[l].o.sc, Q, H, 0);

                // Residual add + RMSNorm → d_res_s
                blackwell::kernels::vector_add_fp32(d_proj_b + m*H, d_proj_b + m*H, d_res_s, H, 0);
                blackwell::kernels::fused_rmsnorm(d_res_s, d_proj_b + m*H, d_rn, H, 1e-6f, 0);

                // Quantize for next GEMV
                blackwell::kernels::pack_int8(d_xi8_s, d_res_s, d_xi8s_s, H, 0);

                // MLP gate + up projections (separate warps, same stream)
                blackwell::kernels::gemv_int8_warp(d_gate_b + m*I, d_xi8_s, d_xi8s_s,
                    lw[l].g.d, lw[l].g.sc, H, I, 0);
                blackwell::kernels::gemv_int8_warp(d_up_b + m*I, d_xi8_s, d_xi8s_s,
                    lw[l].u.d, lw[l].u.sc, H, I, 0);
                blackwell::kernels::apply_swiglu(d_mlp_b + m*I, d_gate_b + m*I, d_up_b + m*I, I, 0);

                // Down projection: quantize → INT8 → GEMV → FP32
                blackwell::kernels::pack_int8(d_mlp_i8_b + m*I, d_mlp_b + m*I, d_mlp_i8s_b + m*(I/16), I, 0);
                blackwell::kernels::gemv_int8_warp(d_proj_b + m*H, d_mlp_i8_b + m*I, d_mlp_i8s_b + m*(I/16),
                    lw[l].d.d, lw[l].d.sc, I, H, 0);

                // Residual add + RMSNorm → d_res_s
                blackwell::kernels::vector_add_fp32(d_proj_b + m*H, d_proj_b + m*H, d_res_s, H, 0);
                blackwell::kernels::fused_rmsnorm(d_res_s, d_proj_b + m*H, d_rn, H, 1e-6f, 0);

                // Repack FP4 state
                blackwell::kernels::fused_rmsnorm(d_proj_b + m*H, d_res_s, d_rn, H, 1e-6f, 0);
                blackwell::kernels::pack_fp4(d_x_fp4_arr[m], d_proj_b + m*H, d_xs_arr[m], H, 0);
            }
        }
    }
    cudaStreamSynchronize(0);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("KV fill: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    printf("done\n"); fflush(stdout);

    printf("Save state...\n"); fflush(stdout);
    void** d_x_fp4_init = new void*[M];
    float** d_xs_init = new float*[M];
    for (int m = 0; m < M; ++m) {
        cudaMalloc(&d_x_fp4_init[m], H);
        cudaMalloc(&d_xs_init[m], (H/16)*4);
        cudaMemcpy(d_x_fp4_init[m], d_x_fp4_arr[m], H, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_xs_init[m], d_xs_arr[m], (H/16)*4, cudaMemcpyDeviceToDevice);
    }
    float *d_kc_save, *d_vc_save;
    cudaMalloc(&d_kc_save, kv_sz); cudaMalloc(&d_vc_save, kv_sz);
    cudaMemcpy(d_kc_save, d_kc, kv_sz, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_vc_save, d_vc, kv_sz, cudaMemcpyDeviceToDevice);

    const int WARMUP = 5, BENCH = 20;

    // ── BASELINE: Serial attention per-seq + batched MLP ─────────────────────
    printf("\nWarmup (serial-attn baseline, M=%d)...\n", M); fflush(stdout);
    for (int w = 0; w < WARMUP; ++w) {
        for (int m = 0; m < M; ++m) {
            for (int l = 0; l < num_layers; ++l) {
                size_t km = m * kv_seq_stride + (size_t)l * nkv * ms * hd;
                blackwell::kernels::unpack_fp4(d_res_s, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
                blackwell::kernels::pack_int8(d_xi8_s, d_res_s, d_xi8s_s, H, 0);
                blackwell::kernels::gemv_int8_warp(d_Q_b + m*Q, d_xi8_s, d_xi8s_s, lw[l].q.d, lw[l].q.sc, H, Q, 0);
                blackwell::kernels::gemv_int8_warp(d_K_b + m*KV, d_xi8_s, d_xi8s_s, lw[l].k.d, lw[l].k.sc, H, KV, 0);
                blackwell::kernels::gemv_int8_warp(d_V_b + m*KV, d_xi8_s, d_xi8s_s, lw[l].v.d, lw[l].v.sc, H, KV, 0);
                blackwell::kernels::update_kv_cache(d_kc + km, d_vc + km, d_K_b + m*KV, d_V_b + m*KV, 0, sq, nkv, hd, ms, 0);
                blackwell::kernels::attention_decode_gqa(d_attn_out_s, d_Q_b + m*Q, d_kc + km, d_vc + km, sq, nqh, nkv, hd, ms, 0);
                blackwell::kernels::pack_int8(d_attn_i8_b + m*Q, d_attn_out_s, d_attn_i8s_b + m*(Q/16), Q, 0);
                blackwell::kernels::gemv_int8_warp(d_proj_b + m*H, d_attn_i8_b + m*Q, d_attn_i8s_b + m*(Q/16), lw[l].o.d, lw[l].o.sc, Q, H, 0);
                blackwell::kernels::vector_add_fp32(d_proj_b + m*H, d_proj_b + m*H, d_res_s, H, 0);
                blackwell::kernels::fused_rmsnorm(d_res_s, d_proj_b + m*H, d_rn, H, 1e-6f, 0);
                blackwell::kernels::pack_int8(d_xi8_s, d_res_s, d_xi8s_s, H, 0);
                blackwell::kernels::gemv_int8_warp(d_gate_b + m*I, d_xi8_s, d_xi8s_s, lw[l].g.d, lw[l].g.sc, H, I, 0);
                blackwell::kernels::gemv_int8_warp(d_up_b + m*I, d_xi8_s, d_xi8s_s, lw[l].u.d, lw[l].u.sc, H, I, 0);
                blackwell::kernels::apply_swiglu(d_mlp_b + m*I, d_gate_b + m*I, d_up_b + m*I, I, 0);
                blackwell::kernels::pack_int8(d_mlp_i8_b + m*I, d_mlp_b + m*I, d_mlp_i8s_b + m*(I/16), I, 0);
                blackwell::kernels::gemv_int8_warp(d_proj_b + m*H, d_mlp_i8_b + m*I, d_mlp_i8s_b + m*(I/16), lw[l].d.d, lw[l].d.sc, I, H, 0);
                blackwell::kernels::vector_add_fp32(d_proj_b + m*H, d_proj_b + m*H, d_res_s, H, 0);
                blackwell::kernels::fused_rmsnorm(d_res_s, d_proj_b + m*H, d_rn, H, 1e-6f, 0);
                blackwell::kernels::fused_rmsnorm(d_proj_b + m*H, d_res_s, d_rn, H, 1e-6f, 0);
                blackwell::kernels::pack_fp4(d_x_fp4_arr[m], d_proj_b + m*H, d_xs_arr[m], H, 0);
            }
        }
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Serial warmup: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    printf("Benchmark serial-attn (%d iters)...\n", BENCH);
    GpuTimer t_serial;
    t_serial.start();
    for (int i = 0; i < BENCH; ++i) {
        for (int m = 0; m < M; ++m) {
            for (int l = 0; l < num_layers; ++l) {
                size_t km = m * kv_seq_stride + (size_t)l * nkv * ms * hd;
                blackwell::kernels::unpack_fp4(d_res_s, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
                blackwell::kernels::pack_int8(d_xi8_s, d_res_s, d_xi8s_s, H, 0);
                blackwell::kernels::gemv_int8_warp(d_Q_b + m*Q, d_xi8_s, d_xi8s_s, lw[l].q.d, lw[l].q.sc, H, Q, 0);
                blackwell::kernels::gemv_int8_warp(d_K_b + m*KV, d_xi8_s, d_xi8s_s, lw[l].k.d, lw[l].k.sc, H, KV, 0);
                blackwell::kernels::gemv_int8_warp(d_V_b + m*KV, d_xi8_s, d_xi8s_s, lw[l].v.d, lw[l].v.sc, H, KV, 0);
                blackwell::kernels::update_kv_cache(d_kc + km, d_vc + km, d_K_b + m*KV, d_V_b + m*KV, 0, sq, nkv, hd, ms, 0);
                blackwell::kernels::attention_decode_gqa(d_attn_out_s, d_Q_b + m*Q, d_kc + km, d_vc + km, sq, nqh, nkv, hd, ms, 0);
                blackwell::kernels::pack_int8(d_attn_i8_b + m*Q, d_attn_out_s, d_attn_i8s_b + m*(Q/16), Q, 0);
                blackwell::kernels::gemv_int8_warp(d_proj_b + m*H, d_attn_i8_b + m*Q, d_attn_i8s_b + m*(Q/16), lw[l].o.d, lw[l].o.sc, Q, H, 0);
                blackwell::kernels::vector_add_fp32(d_proj_b + m*H, d_proj_b + m*H, d_res_s, H, 0);
                blackwell::kernels::fused_rmsnorm(d_res_s, d_proj_b + m*H, d_rn, H, 1e-6f, 0);
                blackwell::kernels::pack_int8(d_xi8_s, d_res_s, d_xi8s_s, H, 0);
                blackwell::kernels::gemv_int8_warp(d_gate_b + m*I, d_xi8_s, d_xi8s_s, lw[l].g.d, lw[l].g.sc, H, I, 0);
                blackwell::kernels::gemv_int8_warp(d_up_b + m*I, d_xi8_s, d_xi8s_s, lw[l].u.d, lw[l].u.sc, H, I, 0);
                blackwell::kernels::apply_swiglu(d_mlp_b + m*I, d_gate_b + m*I, d_up_b + m*I, I, 0);
                blackwell::kernels::pack_int8(d_mlp_i8_b + m*I, d_mlp_b + m*I, d_mlp_i8s_b + m*(I/16), I, 0);
                blackwell::kernels::gemv_int8_warp(d_proj_b + m*H, d_mlp_i8_b + m*I, d_mlp_i8s_b + m*(I/16), lw[l].d.d, lw[l].d.sc, I, H, 0);
                blackwell::kernels::vector_add_fp32(d_proj_b + m*H, d_proj_b + m*H, d_res_s, H, 0);
                blackwell::kernels::fused_rmsnorm(d_res_s, d_proj_b + m*H, d_rn, H, 1e-6f, 0);
                blackwell::kernels::fused_rmsnorm(d_proj_b + m*H, d_res_s, d_rn, H, 1e-6f, 0);
                blackwell::kernels::pack_fp4(d_x_fp4_arr[m], d_proj_b + m*H, d_xs_arr[m], H, 0);
            }
        }
    }
    float serial_ms = t_serial.stop();
    float serial_total = serial_ms / BENCH;
    float serial_tp = 1000.f / serial_total;
    float serial_tp_scaled = serial_tp / M * 8.f;  // normalized to M=8

    // Save serial output
    float* h_serial_out = new float[M * H];
    for (int m = 0; m < M; ++m) {
        float* d_tmp; cudaMalloc(&d_tmp, H*4);
        blackwell::kernels::unpack_fp4(d_tmp, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
        cudaMemcpy(h_serial_out + m*H, d_tmp, H*4, cudaMemcpyDeviceToHost);
        cudaFree(d_tmp);
    }

    // Restore initial state
    for (int m = 0; m < M; ++m) {
        cudaMemcpy(d_x_fp4_arr[m], d_x_fp4_init[m], H, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_xs_arr[m], d_xs_init[m], (H/16)*4, cudaMemcpyDeviceToDevice);
    }
    cudaMemcpy(d_kc, d_kc_save, kv_sz, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_vc, d_vc_save, kv_sz, cudaMemcpyDeviceToDevice);

    // ── BATCHED ATTENTION: benchmark ────────────────────────────────────────
    printf("\nBenchmark batched-attn (per-kernel, %d iters)...\n", BENCH);
    for (int w = 0; w < WARMUP; ++w) {
        for (int l = 0; l < num_layers; ++l) {
            // Unpack all M sequences
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::unpack_fp4(d_res_b + m*big, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
                blackwell::kernels::pack_int8(d_xi8_b + m*H, d_res_b + m*big, d_xi8s_b + m*(H/16), H, 0);
            }
            // QKV projections (serial per-sequence due to H=4096)
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::gemv_int8_warp(d_Q_b + m*Q, d_xi8_b + m*H, d_xi8s_b + m*(H/16), lw[l].q.d, lw[l].q.sc, H, Q, 0);
                blackwell::kernels::gemv_int8_warp(d_K_b + m*KV, d_xi8_b + m*H, d_xi8s_b + m*(H/16), lw[l].k.d, lw[l].k.sc, H, KV, 0);
                blackwell::kernels::gemv_int8_warp(d_V_b + m*KV, d_xi8_b + m*H, d_xi8s_b + m*(H/16), lw[l].v.d, lw[l].v.sc, H, KV, 0);
                size_t km = m * kv_seq_stride + (size_t)l * nkv * ms * hd;
                blackwell::kernels::update_kv_cache(d_kc + km, d_vc + km, d_K_b + m*KV, d_V_b + m*KV, 0, sq, nkv, hd, ms, 0);
            }
            // Batched attention (single kernel, all M sequences)
            size_t kv_base = (size_t)l * nkv * ms * hd;
            chk(blackwell::kernels::attention_decode_batched_gqa(
                d_attn_b, d_Q_b,
                d_kc, d_vc,
                sq, nqh, nkv, hd, ms,
                M,
                kv_seq_stride,   // floats between sequence KV bases
                kv_base          // floats to layer
            ), "batched_attn");
            // O projections (per-sequence, H=Q=4096)
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::pack_int8(d_attn_i8_b + m*Q, d_attn_b + m*Q, d_attn_i8s_b + m*(Q/16), Q, 0);
                blackwell::kernels::gemv_int8_warp(d_proj_b + m*H, d_attn_i8_b + m*Q, d_attn_i8s_b + m*(Q/16), lw[l].o.d, lw[l].o.sc, Q, H, 0);
                blackwell::kernels::vector_add_fp32(d_proj_b + m*H, d_proj_b + m*H, d_res_b + m*big, H, 0);
                blackwell::kernels::fused_rmsnorm(d_res_b + m*big, d_proj_b + m*H, d_rn, H, 1e-6f, 0);
                blackwell::kernels::pack_int8(d_xi8_b + m*H, d_res_b + m*big, d_xi8s_b + m*(H/16), H, 0);
            }
            // MLP gate+up (batched, M×I)
            chk(blackwell::kernels::gemv_int8_batched(d_gate_b, d_xi8_b, d_xi8s_b,
                lw[l].g.d, lw[l].g.sc, H, I, M, 0), "batched_gate");
            chk(blackwell::kernels::gemv_int8_batched(d_up_b, d_xi8_b, d_xi8s_b,
                lw[l].u.d, lw[l].u.sc, H, I, M, 0), "batched_up");
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::apply_swiglu(d_mlp_b + m*I, d_gate_b + m*I, d_up_b + m*I, I, 0);
                blackwell::kernels::pack_int8(d_mlp_i8_b + m*I, d_mlp_b + m*I, d_mlp_i8s_b + m*(I/16), I, 0);
            }
            // MLP down (batched)
            chk(blackwell::kernels::gemv_int8_batched(d_proj_b, d_mlp_i8_b, d_mlp_i8s_b,
                lw[l].d.d, lw[l].d.sc, I, H, M, 0), "batched_down");
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::vector_add_fp32(d_proj_b + m*H, d_proj_b + m*H, d_res_b + m*big, H, 0);
                blackwell::kernels::fused_rmsnorm(d_res_b + m*big, d_proj_b + m*H, d_rn, H, 1e-6f, 0);
                blackwell::kernels::fused_rmsnorm(d_proj_b + m*H, d_res_b + m*big, d_rn, H, 1e-6f, 0);
                blackwell::kernels::pack_fp4(d_x_fp4_arr[m], d_proj_b + m*H, d_xs_arr[m], H, 0);
            }
        }
    }
    cudaDeviceSynchronize();

    GpuTimer t_batch;
    t_batch.start();
    for (int i = 0; i < BENCH; ++i) {
        for (int l = 0; l < num_layers; ++l) {
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::unpack_fp4(d_res_b + m*big, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
                blackwell::kernels::pack_int8(d_xi8_b + m*H, d_res_b + m*big, d_xi8s_b + m*(H/16), H, 0);
            }
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::gemv_int8_warp(d_Q_b + m*Q, d_xi8_b + m*H, d_xi8s_b + m*(H/16), lw[l].q.d, lw[l].q.sc, H, Q, 0);
                blackwell::kernels::gemv_int8_warp(d_K_b + m*KV, d_xi8_b + m*H, d_xi8s_b + m*(H/16), lw[l].k.d, lw[l].k.sc, H, KV, 0);
                blackwell::kernels::gemv_int8_warp(d_V_b + m*KV, d_xi8_b + m*H, d_xi8s_b + m*(H/16), lw[l].v.d, lw[l].v.sc, H, KV, 0);
                size_t km = m * kv_seq_stride + (size_t)l * nkv * ms * hd;
                blackwell::kernels::update_kv_cache(d_kc + km, d_vc + km, d_K_b + m*KV, d_V_b + m*KV, 0, sq, nkv, hd, ms, 0);
            }
            size_t kv_base = (size_t)l * nkv * ms * hd;
            blackwell::kernels::attention_decode_batched_gqa(
                d_attn_b, d_Q_b, d_kc, d_vc, sq, nqh, nkv, hd, ms, M,
                kv_seq_stride, kv_base, 0);
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::pack_int8(d_attn_i8_b + m*Q, d_attn_b + m*Q, d_attn_i8s_b + m*(Q/16), Q, 0);
                blackwell::kernels::gemv_int8_warp(d_proj_b + m*H, d_attn_i8_b + m*Q, d_attn_i8s_b + m*(Q/16), lw[l].o.d, lw[l].o.sc, Q, H, 0);
                blackwell::kernels::vector_add_fp32(d_proj_b + m*H, d_proj_b + m*H, d_res_b + m*big, H, 0);
                blackwell::kernels::fused_rmsnorm(d_res_b + m*big, d_proj_b + m*H, d_rn, H, 1e-6f, 0);
                blackwell::kernels::pack_int8(d_xi8_b + m*H, d_res_b + m*big, d_xi8s_b + m*(H/16), H, 0);
            }
            blackwell::kernels::gemv_int8_batched(d_gate_b, d_xi8_b, d_xi8s_b, lw[l].g.d, lw[l].g.sc, H, I, M, 0);
            blackwell::kernels::gemv_int8_batched(d_up_b, d_xi8_b, d_xi8s_b, lw[l].u.d, lw[l].u.sc, H, I, M, 0);
            for (int m = 0; m < M; ++m)
                blackwell::kernels::apply_swiglu(d_mlp_b + m*I, d_gate_b + m*I, d_up_b + m*I, I, 0);
            for (int m = 0; m < M; ++m)
                blackwell::kernels::pack_int8(d_mlp_i8_b + m*I, d_mlp_b + m*I, d_mlp_i8s_b + m*(I/16), I, 0);
            blackwell::kernels::gemv_int8_batched(d_proj_b, d_mlp_i8_b, d_mlp_i8s_b, lw[l].d.d, lw[l].d.sc, I, H, M, 0);
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::vector_add_fp32(d_proj_b + m*H, d_proj_b + m*H, d_res_b + m*big, H, 0);
                blackwell::kernels::fused_rmsnorm(d_res_b + m*big, d_proj_b + m*H, d_rn, H, 1e-6f, 0);
                blackwell::kernels::fused_rmsnorm(d_proj_b + m*H, d_res_b + m*big, d_rn, H, 1e-6f, 0);
                blackwell::kernels::pack_fp4(d_x_fp4_arr[m], d_proj_b + m*H, d_xs_arr[m], H, 0);
            }
        }
    }
    float batch_ms = t_batch.stop();
    float batch_total = batch_ms / BENCH;
    float batch_tp = M * 1000.f / batch_total;
    float batch_tp_scaled = batch_tp / M * 8.f;

    // Save batch output
    float* h_batch_out = new float[M * H];
    for (int m = 0; m < M; ++m) {
        float* d_tmp; cudaMalloc(&d_tmp, H*4);
        blackwell::kernels::unpack_fp4(d_tmp, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
        cudaMemcpy(h_batch_out + m*H, d_tmp, H*4, cudaMemcpyDeviceToHost);
        cudaFree(d_tmp);
    }

    // Correctness check
    float maxdiff = 0;
    for (int i = 0; i < M*H; ++i)
        maxdiff = fmaxf(maxdiff, fabsf(h_serial_out[i] - h_batch_out[i]));
    delete[] h_serial_out; delete[] h_batch_out;

    // Restore for CUDA Graph benchmark
    for (int m = 0; m < M; ++m) {
        cudaMemcpy(d_x_fp4_arr[m], d_x_fp4_init[m], H, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_xs_arr[m], d_xs_init[m], (H/16)*4, cudaMemcpyDeviceToDevice);
    }
    cudaMemcpy(d_kc, d_kc_save, kv_sz, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_vc, d_vc_save, kv_sz, cudaMemcpyDeviceToDevice);

    // ── CUDA Graph (batched attention) ─────────────────────────────────────
    printf("\nCapturing CUDA Graph (batched attention, %d layers, %d sequences)... ", num_layers, M);
    fflush(stdout);

    cudaStream_t gst; cudaStreamCreate(&gst);

    // L2 persisting hint
    cudaAccessPolicyWindow win;
    win.base_ptr = (void*)d_rn;
    win.num_bytes = big * 4;
    win.hitRatio = 1.0f;
    win.hitProp = cudaAccessPropertyPersisting;
    win.missProp = cudaAccessPropertyStreaming;
    cudaStreamAttrValue av;
    av.accessPolicyWindow = win;
    cudaStreamSetAttribute(gst, cudaStreamAttributeAccessPolicyWindow, &av);

    // Pre-trigger batched attention to set smem config
    chk(blackwell::kernels::attention_decode_batched_gqa(
        d_attn_b, d_Q_b, d_kc, d_vc, sq, nqh, nkv, hd, ms, M,
        kv_seq_stride, 0, gst), "pre_trigger");
    cudaStreamSynchronize(gst);

    cudaGraph_t graph; cudaGraphExec_t exec;
    cudaStreamBeginCapture(gst, cudaStreamCaptureModeGlobal);
    for (int l = 0; l < num_layers; ++l) {
        for (int m = 0; m < M; ++m) {
            blackwell::kernels::unpack_fp4(d_res_b + m*big, d_x_fp4_arr[m], d_xs_arr[m], H, gst);
            blackwell::kernels::pack_int8(d_xi8_b + m*H, d_res_b + m*big, d_xi8s_b + m*(H/16), H, gst);
        }
        for (int m = 0; m < M; ++m) {
            blackwell::kernels::gemv_int8_warp(d_Q_b + m*Q, d_xi8_b + m*H, d_xi8s_b + m*(H/16), lw[l].q.d, lw[l].q.sc, H, Q, gst);
            blackwell::kernels::gemv_int8_warp(d_K_b + m*KV, d_xi8_b + m*H, d_xi8s_b + m*(H/16), lw[l].k.d, lw[l].k.sc, H, KV, gst);
            blackwell::kernels::gemv_int8_warp(d_V_b + m*KV, d_xi8_b + m*H, d_xi8s_b + m*(H/16), lw[l].v.d, lw[l].v.sc, H, KV, gst);
            size_t km = m * kv_seq_stride + (size_t)l * nkv * ms * hd;
            blackwell::kernels::update_kv_cache(d_kc + km, d_vc + km, d_K_b + m*KV, d_V_b + m*KV, 0, sq, nkv, hd, ms, gst);
        }
        size_t kv_base = (size_t)l * nkv * ms * hd;
        blackwell::kernels::attention_decode_batched_gqa(
            d_attn_b, d_Q_b, d_kc, d_vc, sq, nqh, nkv, hd, ms, M,
            kv_seq_stride, kv_base, gst);
        for (int m = 0; m < M; ++m) {
            blackwell::kernels::pack_int8(d_attn_i8_b + m*Q, d_attn_b + m*Q, d_attn_i8s_b + m*(Q/16), Q, gst);
            blackwell::kernels::gemv_int8_warp(d_proj_b + m*H, d_attn_i8_b + m*Q, d_attn_i8s_b + m*(Q/16), lw[l].o.d, lw[l].o.sc, Q, H, gst);
            blackwell::kernels::vector_add_fp32(d_proj_b + m*H, d_proj_b + m*H, d_res_b + m*big, H, gst);
            blackwell::kernels::fused_rmsnorm(d_res_b + m*big, d_proj_b + m*H, d_rn, H, 1e-6f, gst);
            blackwell::kernels::pack_int8(d_xi8_b + m*H, d_res_b + m*big, d_xi8s_b + m*(H/16), H, gst);
        }
        blackwell::kernels::gemv_int8_batched(d_gate_b, d_xi8_b, d_xi8s_b, lw[l].g.d, lw[l].g.sc, H, I, M, gst);
        blackwell::kernels::gemv_int8_batched(d_up_b, d_xi8_b, d_xi8s_b, lw[l].u.d, lw[l].u.sc, H, I, M, gst);
        for (int m = 0; m < M; ++m)
            blackwell::kernels::apply_swiglu(d_mlp_b + m*I, d_gate_b + m*I, d_up_b + m*I, I, gst);
        for (int m = 0; m < M; ++m)
            blackwell::kernels::pack_int8(d_mlp_i8_b + m*I, d_mlp_b + m*I, d_mlp_i8s_b + m*(I/16), I, gst);
        blackwell::kernels::gemv_int8_batched(d_proj_b, d_mlp_i8_b, d_mlp_i8s_b, lw[l].d.d, lw[l].d.sc, I, H, M, gst);
        for (int m = 0; m < M; ++m) {
            blackwell::kernels::vector_add_fp32(d_proj_b + m*H, d_proj_b + m*H, d_res_b + m*big, H, gst);
            blackwell::kernels::fused_rmsnorm(d_res_b + m*big, d_proj_b + m*H, d_rn, H, 1e-6f, gst);
            blackwell::kernels::fused_rmsnorm(d_proj_b + m*H, d_res_b + m*big, d_rn, H, 1e-6f, gst);
            blackwell::kernels::pack_fp4(d_x_fp4_arr[m], d_proj_b + m*H, d_xs_arr[m], H, gst);
        }
    }
    cudaStreamEndCapture(gst, &graph);
    cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
    printf("OK\n"); fflush(stdout);

    for (int i = 0; i < WARMUP; ++i) chk(cudaGraphLaunch(exec, gst), "warmup");
    cudaStreamSynchronize(gst);

    GpuTimer t_graph;
    t_graph.start();
    for (int i = 0; i < BENCH; ++i) chk(cudaGraphLaunch(exec, gst), "launch");
    float graph_ms = t_graph.stop();
    float graph_total = graph_ms / BENCH;
    float graph_tp = M * 1000.f / graph_total;
    float graph_tp_scaled = graph_tp / M * 8.f;

    // ── Print Results ───────────────────────────────────────────────────────
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  Qwen3-8B Batched Decode — %d layers, M=%d                   ║\n", num_layers, M);
    printf("╠════════════════════════╦═══════════════╦═══════════════╦══════════╣\n");
    printf("║ Method                 ║ Per-step     ║ Total t/s    ║ Scaled8 ║\n");
    printf("╠════════════════════════╬═══════════════╬═══════════════╬══════════╣\n");
    printf("║ Serial-attn per-kernel ║ %8.2f ms  ║  %7.1f t/s ║ %6.1f   ║\n", serial_total, serial_tp, serial_tp_scaled);
    printf("║ Batched-attn per-kernel ║ %8.2f ms  ║  %7.1f t/s ║ %6.1f   ║\n", batch_total, batch_tp, batch_tp_scaled);
    printf("║ Batched-attn + CUDA Graph║ %8.2f ms  ║  %7.1f t/s ║ %6.1f   ║\n", graph_total, graph_tp, graph_tp_scaled);
    printf("╠════════════════════════╩═══════════════╩═══════════════╩══════════╣\n");
    printf("║ Batched-attn speedup:  %.2fx vs serial-attn                  ║\n", batch_total / serial_total);
    printf("║ CUDA Graph speedup:    %.2fx vs batched-attn                ║\n", batch_total / graph_total);
    printf("║ Correctness: max_diff=%.6f %s                               ║\n", maxdiff, maxdiff < 0.001f ? "✅" : "❌");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    printf("\n##RESULT model=Qwen3-8B layers=%d M=%d serial=%.2fms=%.1fts batch=%.2fms=%.1fts graph=%.2fms=%.1fts bspeedup=%.2fx gspeedup=%.2fx diff=%.6f\n",
        num_layers, M, serial_total, serial_tp, batch_total, batch_tp, graph_total, graph_tp,
        batch_total/serial_total, batch_total/graph_total, maxdiff);

    cudaStreamDestroy(gst);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    return 0;
}