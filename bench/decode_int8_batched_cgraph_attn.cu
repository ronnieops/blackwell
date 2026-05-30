// bench/decode_int8_batched_cgraph_attn.cu — Batched decode + batched attention + CUDA Graph
//
// INT8 M-sequence decode with batched MLP AND batched attention, captured
// as CUDA Graph. Replaces serial per-seq attention loop with single
// attention_decode_batched_gqa call.
//
// Build:
//   CUDACXX=/usr/local/cuda-13.3/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/decode_int8_batched_cgraph_attn.cu build/libblackwell_kernels.a \
//     -o bench/decode_int8_batched_cgraph_attn

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
    if (num_layers > 28) num_layers = 28;
    if (M < 1) M = 1;
    if (M > 8) M = 8;

    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("# INT8 Batched Decode — Batched Attention + CUDA Graph\n");
    printf("Device: %s (%d.%d)\n", p.name, p.major, p.minor);
    printf("Layers: %d, Batch M: %d\n", num_layers, M);

    const int H = 2048, Q = 2048, KV = 1024, I = 6144;
    const int nqh = 16, nkv = 8, hd = 128, ms = 2048;
    const float s13 = 1.f/3.f;

    // Load weights
    struct LW { DevW q,k,v,o,g,u,d; };
    printf("Loading INT8 weights...\n");
    std::vector<LW> lw(num_layers);
    for (int l = 0; l < num_layers; ++l) {
        char p[256];
        snprintf(p,256,"weights_int8_bf16/%d_self_attn.q_proj",l); lw[l].q = upload(p);
        snprintf(p,256,"weights_int8_bf16/%d_self_attn.k_proj",l); lw[l].k = upload(p);
        snprintf(p,256,"weights_int8_bf16/%d_self_attn.v_proj",l); lw[l].v = upload(p);
        snprintf(p,256,"weights_int8_bf16/%d_self_attn.o_proj",l); lw[l].o = upload(p);
        snprintf(p,256,"weights_int8_bf16/%d_mlp.gate_proj",l);  lw[l].g = upload(p);
        snprintf(p,256,"weights_int8_bf16/%d_mlp.up_proj",l);    lw[l].u = upload(p);
        snprintf(p,256,"weights_int8_bf16/%d_mlp.down_proj",l);  lw[l].d = upload(p);
    }

    // ── Buffers ─────────────────────────────────────────────────────────────
    // Per-seq buffers (for unpack/pack intermediates)
    float *d_res_s, *d_xi8s_s;
    int8_t *d_xi8_s;
    cudaMalloc(&d_res_s, I*4); cudaMalloc(&d_xi8_s, I); cudaMalloc(&d_xi8s_s, (I/16)*4);

    // Per-sequence FP4 state
    void **d_x_fp4_arr = new void*[M];
    float **d_xs_arr = new float*[M];
    for (int m = 0; m < M; ++m) {
        cudaMalloc(&d_x_fp4_arr[m], H);
        cudaMalloc(&d_xs_arr[m], (H/16)*4);
    }

    // Batched buffers
    float *d_Q_b, *d_K_b, *d_V_b, *d_attn_b, *d_proj_b;
    int8_t *d_xi8_b;
    float *d_xi8s_b, *d_attn_i8_b, *d_attn_i8s_b;
    float *d_gate_b, *d_up_b, *d_mlp_b, *d_mlp_i8s_b;
    int8_t *d_mlp_i8_b;
    cudaMalloc(&d_Q_b, M * Q * 4);
    cudaMalloc(&d_K_b, M * KV * 4);
    cudaMalloc(&d_V_b, M * KV * 4);
    cudaMalloc(&d_attn_b, M * Q * 4);
    cudaMalloc(&d_proj_b, M * H * 4);
    cudaMalloc(&d_xi8_b, M * H);
    cudaMalloc(&d_xi8s_b, M * (H/16) * 4);
    cudaMalloc(&d_attn_i8_b, M * Q);
    cudaMalloc(&d_attn_i8s_b, M * (Q/16) * 4);
    cudaMalloc(&d_gate_b, M * I * 4);
    cudaMalloc(&d_up_b, M * I * 4);
    cudaMalloc(&d_mlp_b, M * I * 4);
    cudaMalloc(&d_mlp_i8_b, M * I);
    cudaMalloc(&d_mlp_i8s_b, M * (I/16) * 4);
    // Per-seq output buffer (for post-batched-attention unpack)
    float *d_attn_out_s;
    cudaMalloc(&d_attn_out_s, H * 4);

    float *d_rn; cudaMalloc(&d_rn, H*4);
    std::vector<float> rn_h(H, 1.f);
    cudaMemcpy(d_rn, rn_h.data(), H*4, cudaMemcpyHostToDevice);

    // KV cache: contiguous [M][total_layers][nkv][ms][hd]
    float *d_kc, *d_vc;
    size_t kv_sz = (size_t)M * num_layers * nkv * ms * hd * 4;
    cudaMalloc(&d_kc, kv_sz); cudaMalloc(&d_vc, kv_sz);
    cudaMemset(d_kc, 0, kv_sz); cudaMemset(d_vc, 0, kv_sz);

    // KV cache stride info for batched attention
    size_t kv_seq_stride = (size_t)num_layers * nkv * ms * hd;  // floats

    // Init all sequences
    float *d_x32; cudaMalloc(&d_x32, H*4);
    std::vector<float> xh(H, 1.f), xsh(H/16, s13);
    cudaMemcpy(d_x32, xh.data(), H*4, cudaMemcpyHostToDevice);
    for (int m = 0; m < M; ++m) {
        cudaMemcpy(d_xs_arr[m], xsh.data(), (H/16)*4, cudaMemcpyHostToDevice);
        blackwell::kernels::pack_fp4(d_x_fp4_arr[m], d_x32, d_xs_arr[m], H, 0);
    }

    // ── Fill KV cache ───────────────────────────────────────────────────────
    printf("Filling KV cache (%d sequences, seq=0..128)... ", M);
    fflush(stdout);
    int sq = 128;
    for (int s = 0; s <= sq; ++s) {
        for (int m = 0; m < M; ++m) {
            for (int l = 0; l < num_layers; ++l) {
                size_t kv_layer_off = (size_t)l * nkv * ms * hd;
                size_t km = m * kv_seq_stride + kv_layer_off;
                blackwell::kernels::unpack_fp4(d_res_s, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
                blackwell::kernels::pack_int8(d_xi8_s, d_res_s, d_xi8s_s, H, 0);
                blackwell::kernels::gemv_int8_warp(d_Q_b + m*Q, d_xi8_s, d_xi8s_s,
                    lw[l].q.d, lw[l].q.sc, H, Q, 0);
                blackwell::kernels::gemv_int8_warp(d_K_b + m*KV, d_xi8_s, d_xi8s_s,
                    lw[l].k.d, lw[l].k.sc, H, KV, 0);
                blackwell::kernels::gemv_int8_warp(d_V_b + m*KV, d_xi8_s, d_xi8s_s,
                    lw[l].v.d, lw[l].v.sc, H, KV, 0);
                blackwell::kernels::update_kv_cache(d_kc+km, d_vc+km, d_K_b+m*KV, d_V_b+m*KV, 0, s, nkv, hd, ms, 0);
                blackwell::kernels::attention_decode_gqa(d_attn_out_s, d_Q_b+m*Q, d_kc+km, d_vc+km,
                    s, nqh, nkv, hd, ms, 0);
                blackwell::kernels::pack_int8(d_attn_i8_b + m*Q, d_attn_out_s, d_attn_i8s_b + m*(Q/16), Q, 0);
                blackwell::kernels::gemv_int8_warp(d_proj_b + m*H, d_attn_i8_b + m*Q, d_attn_i8s_b + m*(Q/16),
                    lw[l].o.d, lw[l].o.sc, Q, H, 0);
                blackwell::kernels::unpack_fp4(d_res_s, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
                blackwell::kernels::vector_add_fp32(d_proj_b+m*H, d_proj_b+m*H, d_res_s, H, 0);
                blackwell::kernels::fused_rmsnorm_quant_int8(d_xi8_b+m*H, d_xi8s_b+m*(H/16),
                    d_proj_b+m*H, d_rn, H, 1e-6f, 0);
                blackwell::kernels::fused_rmsnorm_pack(d_x_fp4_arr[m], d_xs_arr[m], d_proj_b+m*H, d_rn, H, 1e-6f, 0);
                // MLP
                blackwell::kernels::unpack_fp4(d_res_s, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
                blackwell::kernels::pack_int8(d_xi8_s, d_res_s, d_xi8s_s, H, 0);
                blackwell::kernels::gemv_int8_warp(d_gate_b+m*I, d_xi8_s, d_xi8s_s,
                    lw[l].g.d, lw[l].g.sc, H, I, 0);
                blackwell::kernels::gemv_int8_warp(d_up_b+m*I, d_xi8_s, d_xi8s_s,
                    lw[l].u.d, lw[l].u.sc, H, I, 0);
                blackwell::kernels::apply_swiglu(d_mlp_b+m*I, d_gate_b+m*I, d_up_b+m*I, I, 0);
                blackwell::kernels::pack_int8(d_mlp_i8_b+m*I, d_mlp_b+m*I, d_mlp_i8s_b+m*(I/16), I, 0);
                blackwell::kernels::gemv_int8_warp(d_proj_b+m*H, d_mlp_i8_b+m*I, d_mlp_i8s_b+m*(I/16),
                    lw[l].d.d, lw[l].d.sc, I, H, 0);
                blackwell::kernels::vector_add_fp32(d_proj_b+m*H, d_proj_b+m*H, d_res_s, H, 0);
                blackwell::kernels::fused_rmsnorm_quant_int8(d_xi8_b+m*H, d_xi8s_b+m*(H/16),
                    d_proj_b+m*H, d_rn, H, 1e-6f, 0);
                blackwell::kernels::fused_rmsnorm_pack(d_x_fp4_arr[m], d_xs_arr[m], d_proj_b+m*H, d_rn, H, 1e-6f, 0);
            }
        }
    }
    printf("done\n");

    // Save initial state
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

    int warm = 5, bench = 20;

    // ── BASELINE: Serial attention per-seq + batched MLP ─────────────────────
    // Uses the original attention_decode_gqa (M×serial)
    printf("Warmup (serial-attn baseline, M=%d)...\n", M);
    for (int w = 0; w < warm; ++w) {
        for (int l = 0; l < num_layers; ++l) {
            size_t kv_layer_off = l * nkv * ms * hd;
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::unpack_fp4_pack_int8(
                    d_xi8_b + m*H, d_xi8s_b + m*(H/16),
                    d_x_fp4_arr[m], d_xs_arr[m],
                    d_xi8s_b + m*(H/16), H, 0);
            }
            for (int m = 0; m < M; ++m) {
                size_t km = m * kv_seq_stride + kv_layer_off;
                blackwell::kernels::gemv_int8_warp(d_Q_b + m*Q, d_xi8_b + m*H, d_xi8s_b + m*(H/16),
                    lw[l].q.d, lw[l].q.sc, H, Q, 0);
                blackwell::kernels::gemv_int8_warp(d_K_b + m*KV, d_xi8_b + m*H, d_xi8s_b + m*(H/16),
                    lw[l].k.d, lw[l].k.sc, H, KV, 0);
                blackwell::kernels::gemv_int8_warp(d_V_b + m*KV, d_xi8_b + m*H, d_xi8s_b + m*(H/16),
                    lw[l].v.d, lw[l].v.sc, H, KV, 0);
                blackwell::kernels::update_kv_cache(d_kc+km, d_vc+km, d_K_b+m*KV, d_V_b+m*KV, 0, sq, nkv, hd, ms, 0);
                blackwell::kernels::attention_decode_gqa(d_attn_b + m*Q, d_Q_b+m*Q, d_kc+km, d_vc+km,
                    sq, nqh, nkv, hd, ms, 0);
                blackwell::kernels::pack_int8(d_attn_i8_b + m*Q, d_attn_b + m*Q, d_attn_i8s_b + m*(Q/16), Q, 0);
                blackwell::kernels::gemv_int8_warp(d_proj_b + m*H, d_attn_i8_b + m*Q, d_attn_i8s_b + m*(Q/16),
                    lw[l].o.d, lw[l].o.sc, Q, H, 0);
                blackwell::kernels::unpack_fp4(d_res_s, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
                blackwell::kernels::vector_add_fp32(d_proj_b+m*H, d_proj_b+m*H, d_res_s, H, 0);
                blackwell::kernels::fused_rmsnorm_quant_int8(d_xi8_b + m*H, d_xi8s_b + m*(H/16),
                    d_proj_b+m*H, d_rn, H, 1e-6f, 0);
                blackwell::kernels::fused_rmsnorm_pack(d_x_fp4_arr[m], d_xs_arr[m], d_proj_b+m*H, d_rn, H, 1e-6f, 0);
            }
            // MLP: batched
            blackwell::kernels::gemv_int8_batched(d_gate_b, d_xi8_b, d_xi8s_b,
                lw[l].g.d, lw[l].g.sc, H, I, M, 0);
            blackwell::kernels::gemv_int8_batched(d_up_b, d_xi8_b, d_xi8s_b,
                lw[l].u.d, lw[l].u.sc, H, I, M, 0);
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::apply_swiglu(d_mlp_b + m*I, d_gate_b + m*I, d_up_b + m*I, I, 0);
                blackwell::kernels::pack_int8(d_mlp_i8_b + m*I, d_mlp_b + m*I, d_mlp_i8s_b + m*(I/16), I, 0);
            }
            blackwell::kernels::gemv_int8_batched(d_proj_b, d_mlp_i8_b, d_mlp_i8s_b,
                lw[l].d.d, lw[l].d.sc, I, H, M, 0);
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::unpack_fp4(d_res_s, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
                blackwell::kernels::vector_add_fp32(d_proj_b+m*H, d_proj_b+m*H, d_res_s, H, 0);
                blackwell::kernels::fused_rmsnorm_quant_int8(d_xi8_b+m*H, d_xi8s_b+m*(H/16), d_proj_b+m*H, d_rn, H, 1e-6f, 0);
                blackwell::kernels::fused_rmsnorm_pack(d_x_fp4_arr[m], d_xs_arr[m], d_proj_b+m*H, d_rn, H, 1e-6f, 0);
            }
        }
    }
    cudaDeviceSynchronize();

    printf("Benchmark serial-attn (%d iters)...\n", bench);
    GpuTimer tpks;
    tpks.start();
    for (int i = 0; i < bench; ++i) {
        for (int l = 0; l < num_layers; ++l) {
            size_t kv_layer_off = l * nkv * ms * hd;
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::unpack_fp4_pack_int8(
                    d_xi8_b + m*H, d_xi8s_b + m*(H/16),
                    d_x_fp4_arr[m], d_xs_arr[m],
                    d_xi8s_b + m*(H/16), H, 0);
            }
            for (int m = 0; m < M; ++m) {
                size_t km = m * kv_seq_stride + kv_layer_off;
                blackwell::kernels::gemv_int8_warp(d_Q_b + m*Q, d_xi8_b + m*H, d_xi8s_b + m*(H/16),
                    lw[l].q.d, lw[l].q.sc, H, Q, 0);
                blackwell::kernels::gemv_int8_warp(d_K_b + m*KV, d_xi8_b + m*H, d_xi8s_b + m*(H/16),
                    lw[l].k.d, lw[l].k.sc, H, KV, 0);
                blackwell::kernels::gemv_int8_warp(d_V_b + m*KV, d_xi8_b + m*H, d_xi8s_b + m*(H/16),
                    lw[l].v.d, lw[l].v.sc, H, KV, 0);
                blackwell::kernels::update_kv_cache(d_kc+km, d_vc+km, d_K_b+m*KV, d_V_b+m*KV, 0, sq, nkv, hd, ms, 0);
                blackwell::kernels::attention_decode_gqa(d_attn_b + m*Q, d_Q_b+m*Q, d_kc+km, d_vc+km,
                    sq, nqh, nkv, hd, ms, 0);
                blackwell::kernels::pack_int8(d_attn_i8_b + m*Q, d_attn_b + m*Q, d_attn_i8s_b + m*(Q/16), Q, 0);
                blackwell::kernels::gemv_int8_warp(d_proj_b + m*H, d_attn_i8_b + m*Q, d_attn_i8s_b + m*(Q/16),
                    lw[l].o.d, lw[l].o.sc, Q, H, 0);
                blackwell::kernels::unpack_fp4(d_res_s, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
                blackwell::kernels::vector_add_fp32(d_proj_b+m*H, d_proj_b+m*H, d_res_s, H, 0);
                blackwell::kernels::fused_rmsnorm_quant_int8(d_xi8_b + m*H, d_xi8s_b + m*(H/16),
                    d_proj_b+m*H, d_rn, H, 1e-6f, 0);
                blackwell::kernels::fused_rmsnorm_pack(d_x_fp4_arr[m], d_xs_arr[m], d_proj_b+m*H, d_rn, H, 1e-6f, 0);
            }
            blackwell::kernels::gemv_int8_batched(d_gate_b, d_xi8_b, d_xi8s_b,
                lw[l].g.d, lw[l].g.sc, H, I, M, 0);
            blackwell::kernels::gemv_int8_batched(d_up_b, d_xi8_b, d_xi8s_b,
                lw[l].u.d, lw[l].u.sc, H, I, M, 0);
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::apply_swiglu(d_mlp_b + m*I, d_gate_b + m*I, d_up_b + m*I, I, 0);
                blackwell::kernels::pack_int8(d_mlp_i8_b + m*I, d_mlp_b + m*I, d_mlp_i8s_b + m*(I/16), I, 0);
            }
            blackwell::kernels::gemv_int8_batched(d_proj_b, d_mlp_i8_b, d_mlp_i8s_b,
                lw[l].d.d, lw[l].d.sc, I, H, M, 0);
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::unpack_fp4(d_res_s, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
                blackwell::kernels::vector_add_fp32(d_proj_b+m*H, d_proj_b+m*H, d_res_s, H, 0);
                blackwell::kernels::fused_rmsnorm_quant_int8(d_xi8_b+m*H, d_xi8s_b+m*(H/16), d_proj_b+m*H, d_rn, H, 1e-6f, 0);
                blackwell::kernels::fused_rmsnorm_pack(d_x_fp4_arr[m], d_xs_arr[m], d_proj_b+m*H, d_rn, H, 1e-6f, 0);
            }
        }
    }
    float pks_ms = tpks.stop();

    // Save serial-attn output
    float *d_pk_out; cudaMalloc(&d_pk_out, M*H*4);
    for (int m = 0; m < M; ++m)
        blackwell::kernels::unpack_fp4(d_pk_out+m*H, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
    std::vector<float> pk_out(M*H);
    cudaMemcpy(pk_out.data(), d_pk_out, M*H*4, cudaMemcpyDeviceToHost);

    // Restore state for graph
    for (int m = 0; m < M; ++m) {
        cudaMemcpy(d_x_fp4_arr[m], d_x_fp4_init[m], H, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_xs_arr[m], d_xs_init[m], (H/16)*4, cudaMemcpyDeviceToDevice);
    }
    for (int m = 0; m < M; ++m) { cudaFree(d_x_fp4_init[m]); cudaFree(d_xs_init[m]); }
    delete[] d_x_fp4_init; delete[] d_xs_init;
    cudaMemcpy(d_kc, d_kc_save, kv_sz, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_vc, d_vc_save, kv_sz, cudaMemcpyDeviceToDevice);
    cudaFree(d_kc_save); cudaFree(d_vc_save);

    // ── BATCHED ATTENTION: single attention_decode_batched_gqa call ───────────
    printf("Benchmark batched-attn (per-kernel, %d iters)...\n", bench);
    for (int w = 0; w < warm; ++w) {
        for (int l = 0; l < num_layers; ++l) {
            size_t kv_layer_off = l * nkv * ms * hd;
            // Attn: Q/K/V GEMV for all M, then ONE batched attention call
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::unpack_fp4_pack_int8(
                    d_xi8_b + m*H, d_xi8s_b + m*(H/16),
                    d_x_fp4_arr[m], d_xs_arr[m],
                    d_xi8s_b + m*(H/16), H, 0);
            }
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::gemv_int8_warp(d_Q_b + m*Q, d_xi8_b + m*H, d_xi8s_b + m*(H/16),
                    lw[l].q.d, lw[l].q.sc, H, Q, 0);
                blackwell::kernels::gemv_int8_warp(d_K_b + m*KV, d_xi8_b + m*H, d_xi8s_b + m*(H/16),
                    lw[l].k.d, lw[l].k.sc, H, KV, 0);
                blackwell::kernels::gemv_int8_warp(d_V_b + m*KV, d_xi8_b + m*H, d_xi8s_b + m*(H/16),
                    lw[l].v.d, lw[l].v.sc, H, KV, 0);
                size_t km = m * kv_seq_stride + kv_layer_off;
                blackwell::kernels::update_kv_cache(d_kc+km, d_vc+km, d_K_b+m*KV, d_V_b+m*KV, 0, sq, nkv, hd, ms, 0);
            }
            // ONE batched attention call instead of M serial calls
            blackwell::kernels::attention_decode_batched_gqa(d_attn_b, d_Q_b, d_kc, d_vc,
                sq, nqh, nkv, hd, ms, M, kv_seq_stride, kv_layer_off, 0);
            // Attn output projection (per-seq Wo after attn)
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::pack_int8(d_attn_i8_b + m*Q, d_attn_b + m*Q, d_attn_i8s_b + m*(Q/16), Q, 0);
                blackwell::kernels::gemv_int8_warp(d_proj_b + m*H, d_attn_i8_b + m*Q, d_attn_i8s_b + m*(Q/16),
                    lw[l].o.d, lw[l].o.sc, Q, H, 0);
                blackwell::kernels::unpack_fp4(d_res_s, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
                blackwell::kernels::vector_add_fp32(d_proj_b+m*H, d_proj_b+m*H, d_res_s, H, 0);
                blackwell::kernels::fused_rmsnorm_quant_int8(d_xi8_b+m*H, d_xi8s_b+m*(H/16),
                    d_proj_b+m*H, d_rn, H, 1e-6f, 0);
                blackwell::kernels::fused_rmsnorm_pack(d_x_fp4_arr[m], d_xs_arr[m], d_proj_b+m*H, d_rn, H, 1e-6f, 0);
            }
            // MLP: batched (same as before)
            blackwell::kernels::gemv_int8_batched(d_gate_b, d_xi8_b, d_xi8s_b,
                lw[l].g.d, lw[l].g.sc, H, I, M, 0);
            blackwell::kernels::gemv_int8_batched(d_up_b, d_xi8_b, d_xi8s_b,
                lw[l].u.d, lw[l].u.sc, H, I, M, 0);
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::apply_swiglu(d_mlp_b+m*I, d_gate_b+m*I, d_up_b+m*I, I, 0);
                blackwell::kernels::pack_int8(d_mlp_i8_b+m*I, d_mlp_b+m*I, d_mlp_i8s_b+m*(I/16), I, 0);
            }
            blackwell::kernels::gemv_int8_batched(d_proj_b, d_mlp_i8_b, d_mlp_i8s_b,
                lw[l].d.d, lw[l].d.sc, I, H, M, 0);
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::unpack_fp4(d_res_s, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
                blackwell::kernels::vector_add_fp32(d_proj_b+m*H, d_proj_b+m*H, d_res_s, H, 0);
                blackwell::kernels::fused_rmsnorm_quant_int8(d_xi8_b+m*H, d_xi8s_b+m*(H/16), d_proj_b+m*H, d_rn, H, 1e-6f, 0);
                blackwell::kernels::fused_rmsnorm_pack(d_x_fp4_arr[m], d_xs_arr[m], d_proj_b+m*H, d_rn, H, 1e-6f, 0);
            }
        }
    }
    cudaDeviceSynchronize();

    GpuTimer tpkb;
    tpkb.start();
    for (int i = 0; i < bench; ++i) {
        for (int l = 0; l < num_layers; ++l) {
            size_t kv_layer_off = l * nkv * ms * hd;
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::unpack_fp4_pack_int8(
                    d_xi8_b + m*H, d_xi8s_b + m*(H/16),
                    d_x_fp4_arr[m], d_xs_arr[m],
                    d_xi8s_b + m*(H/16), H, 0);
            }
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::gemv_int8_warp(d_Q_b + m*Q, d_xi8_b + m*H, d_xi8s_b + m*(H/16),
                    lw[l].q.d, lw[l].q.sc, H, Q, 0);
                blackwell::kernels::gemv_int8_warp(d_K_b + m*KV, d_xi8_b + m*H, d_xi8s_b + m*(H/16),
                    lw[l].k.d, lw[l].k.sc, H, KV, 0);
                blackwell::kernels::gemv_int8_warp(d_V_b + m*KV, d_xi8_b + m*H, d_xi8s_b + m*(H/16),
                    lw[l].v.d, lw[l].v.sc, H, KV, 0);
                size_t km = m * kv_seq_stride + kv_layer_off;
                blackwell::kernels::update_kv_cache(d_kc+km, d_vc+km, d_K_b+m*KV, d_V_b+m*KV, 0, sq, nkv, hd, ms, 0);
            }
            blackwell::kernels::attention_decode_batched_gqa(d_attn_b, d_Q_b, d_kc, d_vc,
                sq, nqh, nkv, hd, ms, M, kv_seq_stride, kv_layer_off, 0);
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::pack_int8(d_attn_i8_b + m*Q, d_attn_b + m*Q, d_attn_i8s_b + m*(Q/16), Q, 0);
                blackwell::kernels::gemv_int8_warp(d_proj_b + m*H, d_attn_i8_b + m*Q, d_attn_i8s_b + m*(Q/16),
                    lw[l].o.d, lw[l].o.sc, Q, H, 0);
                blackwell::kernels::unpack_fp4(d_res_s, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
                blackwell::kernels::vector_add_fp32(d_proj_b+m*H, d_proj_b+m*H, d_res_s, H, 0);
                blackwell::kernels::fused_rmsnorm_quant_int8(d_xi8_b+m*H, d_xi8s_b+m*(H/16),
                    d_proj_b+m*H, d_rn, H, 1e-6f, 0);
                blackwell::kernels::fused_rmsnorm_pack(d_x_fp4_arr[m], d_xs_arr[m], d_proj_b+m*H, d_rn, H, 1e-6f, 0);
            }
            blackwell::kernels::gemv_int8_batched(d_gate_b, d_xi8_b, d_xi8s_b,
                lw[l].g.d, lw[l].g.sc, H, I, M, 0);
            blackwell::kernels::gemv_int8_batched(d_up_b, d_xi8_b, d_xi8s_b,
                lw[l].u.d, lw[l].u.sc, H, I, M, 0);
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::apply_swiglu(d_mlp_b+m*I, d_gate_b+m*I, d_up_b+m*I, I, 0);
                blackwell::kernels::pack_int8(d_mlp_i8_b+m*I, d_mlp_b+m*I, d_mlp_i8s_b+m*(I/16), I, 0);
            }
            blackwell::kernels::gemv_int8_batched(d_proj_b, d_mlp_i8_b, d_mlp_i8s_b,
                lw[l].d.d, lw[l].d.sc, I, H, M, 0);
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::unpack_fp4(d_res_s, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
                blackwell::kernels::vector_add_fp32(d_proj_b+m*H, d_proj_b+m*H, d_res_s, H, 0);
                blackwell::kernels::fused_rmsnorm_quant_int8(d_xi8_b+m*H, d_xi8s_b+m*(H/16), d_proj_b+m*H, d_rn, H, 1e-6f, 0);
                blackwell::kernels::fused_rmsnorm_pack(d_x_fp4_arr[m], d_xs_arr[m], d_proj_b+m*H, d_rn, H, 1e-6f, 0);
            }
        }
    }
    float pkb_ms = tpkb.stop();

    // ── CUDA Graph capture (batched attention path) ──────────────────────────
    printf("\n=== CUDA Graph (batched attention) ===\n");
    cudaDeviceSynchronize();
    cudaError_t cerr = cudaPeekAtLastError();
    if (cerr != cudaSuccess) { printf("  Pre-capture: %s\n", cudaGetErrorString(cerr)); cudaGetLastError(); }

    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 8 * 1024 * 1024);
    cudaStream_t graph_stream;
    cudaStreamCreate(&graph_stream);

    cudaAccessPolicyWindow norm_policy;
    norm_policy.base_ptr = (void*)d_rn;
    norm_policy.num_bytes = H * 4;
    norm_policy.hitRatio = 1.0f;
    norm_policy.hitProp = cudaAccessPropertyPersisting;
    norm_policy.missProp = cudaAccessPropertyStreaming;
    cudaStreamAttrValue norm_attr;
    norm_attr.accessPolicyWindow = norm_policy;
    cudaStreamSetAttribute(graph_stream, cudaStreamAttributeAccessPolicyWindow, &norm_attr);

    // Pre-trigger batched attention
    blackwell::kernels::attention_decode_batched_gqa(d_attn_b, d_Q_b, d_kc, d_vc,
        sq, nqh, nkv, hd, ms, M, kv_seq_stride, 0, graph_stream);
    cudaStreamSynchronize(graph_stream);

    printf("  Capturing %d layers × %d seqs... ", num_layers, M);
    fflush(stdout);

    cudaStreamBeginCapture(graph_stream, cudaStreamCaptureModeGlobal);
    for (int l = 0; l < num_layers; ++l) {
        size_t kv_layer_off = l * nkv * ms * hd;

        // 1. Fused unpack+pack all M (1 kernel instead of 2)
        for (int m = 0; m < M; ++m) {
            blackwell::kernels::unpack_fp4_pack_int8(
                d_xi8_b + m*H, d_xi8s_b + m*(H/16),
                d_x_fp4_arr[m], d_xs_arr[m],
                d_xi8s_b + m*(H/16), H, graph_stream);
        }

        // 2. Q/K/V GEMV + KV cache (per-seq)
        for (int m = 0; m < M; ++m) {
            size_t km = m * kv_seq_stride + kv_layer_off;
            blackwell::kernels::gemv_int8_warp(d_Q_b + m*Q, d_xi8_b + m*H, d_xi8s_b + m*(H/16),
                lw[l].q.d, lw[l].q.sc, H, Q, graph_stream);
            blackwell::kernels::gemv_int8_warp(d_K_b + m*KV, d_xi8_b + m*H, d_xi8s_b + m*(H/16),
                lw[l].k.d, lw[l].k.sc, H, KV, graph_stream);
            blackwell::kernels::gemv_int8_warp(d_V_b + m*KV, d_xi8_b + m*H, d_xi8s_b + m*(H/16),
                lw[l].v.d, lw[l].v.sc, H, KV, graph_stream);
            blackwell::kernels::update_kv_cache(d_kc+km, d_vc+km, d_K_b+m*KV, d_V_b+m*KV, 0, sq, nkv, hd, ms, graph_stream);
        }

        // 3. ONE batched attention
        blackwell::kernels::attention_decode_batched_gqa(d_attn_b, d_Q_b, d_kc, d_vc,
            sq, nqh, nkv, hd, ms, M, kv_seq_stride, kv_layer_off, graph_stream);

        // 4. Wo + residual + rmsnorm (per-seq)
        for (int m = 0; m < M; ++m) {
            blackwell::kernels::pack_int8(d_attn_i8_b + m*Q, d_attn_b + m*Q, d_attn_i8s_b + m*(Q/16), Q, graph_stream);
            blackwell::kernels::gemv_int8_warp(d_proj_b + m*H, d_attn_i8_b + m*Q, d_attn_i8s_b + m*(Q/16),
                lw[l].o.d, lw[l].o.sc, Q, H, graph_stream);
            blackwell::kernels::unpack_fp4(d_res_s, d_x_fp4_arr[m], d_xs_arr[m], H, graph_stream);
            blackwell::kernels::vector_add_fp32(d_proj_b+m*H, d_proj_b+m*H, d_res_s, H, graph_stream);
            blackwell::kernels::fused_rmsnorm_quant_int8(d_xi8_b+m*H, d_xi8s_b+m*(H/16),
                d_proj_b+m*H, d_rn, H, 1e-6f, graph_stream);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4_arr[m], d_xs_arr[m], d_proj_b+m*H, d_rn, H, 1e-6f, graph_stream);
        }

        // 5. Batched MLP
        blackwell::kernels::gemv_int8_batched(d_gate_b, d_xi8_b, d_xi8s_b,
            lw[l].g.d, lw[l].g.sc, H, I, M, graph_stream);
        blackwell::kernels::gemv_int8_batched(d_up_b, d_xi8_b, d_xi8s_b,
            lw[l].u.d, lw[l].u.sc, H, I, M, graph_stream);
        for (int m = 0; m < M; ++m) {
            blackwell::kernels::apply_swiglu(d_mlp_b+m*I, d_gate_b+m*I, d_up_b+m*I, I, graph_stream);
            blackwell::kernels::pack_int8(d_mlp_i8_b+m*I, d_mlp_b+m*I, d_mlp_i8s_b+m*(I/16), I, graph_stream);
        }
        blackwell::kernels::gemv_int8_batched(d_proj_b, d_mlp_i8_b, d_mlp_i8s_b,
            lw[l].d.d, lw[l].d.sc, I, H, M, graph_stream);
        for (int m = 0; m < M; ++m) {
            blackwell::kernels::unpack_fp4(d_res_s, d_x_fp4_arr[m], d_xs_arr[m], H, graph_stream);
            blackwell::kernels::vector_add_fp32(d_proj_b+m*H, d_proj_b+m*H, d_res_s, H, graph_stream);
            blackwell::kernels::fused_rmsnorm_quant_int8(d_xi8_b+m*H, d_xi8s_b+m*(H/16), d_proj_b+m*H, d_rn, H, 1e-6f, graph_stream);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4_arr[m], d_xs_arr[m], d_proj_b+m*H, d_rn, H, 1e-6f, graph_stream);
        }
    }

    cudaGraph_t graph;
    cerr = cudaStreamEndCapture(graph_stream, &graph);
    if (cerr != cudaSuccess) { printf("FAIL capture: %s\n", cudaGetErrorString(cerr)); return 1; }

    cudaGraphExec_t graph_exec;
    cerr = cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
    if (cerr != cudaSuccess) { printf("FAIL instantiate: %s\n", cudaGetErrorString(cerr)); return 1; }
    printf("OK\n");

    // Graph warmup + bench
    printf("  Graph warmup...\n");
    for (int i = 0; i < warm; ++i) cudaGraphLaunch(graph_exec, graph_stream);
    cudaStreamSynchronize(graph_stream);

    printf("  Graph benchmark (%d iters)...\n", bench);
    GpuTimer tg;
    tg.start(graph_stream);
    for (int i = 0; i < bench; ++i) cudaGraphLaunch(graph_exec, graph_stream);
    cudaStreamSynchronize(graph_stream);
    float graph_ms = tg.stop();

    // Correctness check
    float *d_gr_out; cudaMalloc(&d_gr_out, M*H*4);
    for (int m = 0; m < M; ++m)
        blackwell::kernels::unpack_fp4(d_gr_out+m*H, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
    std::vector<float> gr_out(M*H);
    cudaMemcpy(gr_out.data(), d_gr_out, M*H*4, cudaMemcpyDeviceToHost);
    cudaFree(d_gr_out);

    float max_diff = 0, pk_l1 = 0, gr_l1 = 0;
    // Compare against serial-attn per-kernel baseline
    for (int m = 0; m < M; ++m) {
        for (int i = 0; i < H; ++i) {
            int idx = m * H + i;
            pk_l1 += fabsf(pk_out[idx]);
            gr_l1 += fabsf(gr_out[idx]);
            max_diff = fmaxf(max_diff, fabsf(pk_out[idx] - gr_out[idx]));
        }
    }
    printf("\n=== Correctness (graph vs serial-attn baseline) ===\n");
    printf("  Max diff: %.6f %s\n", max_diff,
        max_diff < 1e-3 ? "✅ MATCH" : max_diff < 0.1 ? "⚠️ CLOSE" : "❌ MISMATCH");
    cudaFree(d_pk_out);

    // ── Results ──────────────────────────────────────────────────────────────
    float pks_pt = pks_ms / bench;
    float pkb_pt = pkb_ms / bench;
    float gr_pt = graph_ms / bench;

    float tps_s = M * 1000.f / pks_pt;
    float s28_s = 1000.f / (pks_pt * 28.f / num_layers);
    float tps_b = M * 1000.f / pkb_pt;
    float s28_b = 1000.f / (pkb_pt * 28.f / num_layers);
    float tps_g = M * 1000.f / gr_pt;
    float s28_g = 1000.f / (gr_pt * 28.f / num_layers);

    printf("\n=== Results (M=%d, %d layers) ===\n", M, num_layers);
    printf("  %-30s  %8s  %10s  %8s\n", "Method", "Per-step", "Total t/s", "Scaled28");
    printf("  %-30s  %7.3fms  %8.1f    %7.1f\n", "Serial-attn per-kernel (old)",
        pks_pt, tps_s, s28_s);
    printf("  %-30s  %7.3fms  %8.1f    %7.1f\n", "Batched-attn per-kernel",
        pkb_pt, tps_b, s28_b);
    printf("  %-30s  %7.3fms  %8.1f    %7.1f\n", "Batched-attn + CUDA Graph",
        gr_pt, tps_g, s28_g);
    printf("  Batched-attn speedup: %.2fx (%.1f%%) over serial\n",
        pks_pt / pkb_pt, (1.f - pkb_pt/pks_pt)*100.f);
    printf("  CUDA Graph speedup:  %.2fx (%.1f%%) over batched-attn\n",
        pkb_pt / gr_pt, (1.f - gr_pt/pkb_pt)*100.f);
    printf("  Target: llama.cpp 276.0 t/s\n");

    // Cleanup
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(graph_stream);

    for (int m = 0; m < M; ++m) { cudaFree(d_x_fp4_arr[m]); cudaFree(d_xs_arr[m]); }
    delete[] d_x_fp4_arr; delete[] d_xs_arr;
    for (auto& l : lw) {
        cudaFree(l.q.d); cudaFree(l.q.sc); cudaFree(l.k.d); cudaFree(l.k.sc);
        cudaFree(l.v.d); cudaFree(l.v.sc); cudaFree(l.o.d); cudaFree(l.o.sc);
        cudaFree(l.g.d); cudaFree(l.g.sc); cudaFree(l.u.d); cudaFree(l.u.sc);
        cudaFree(l.d.d); cudaFree(l.d.sc);
    }
    cudaFree(d_Q_b); cudaFree(d_K_b); cudaFree(d_V_b);
    cudaFree(d_attn_b); cudaFree(d_proj_b);
    cudaFree(d_xi8_b); cudaFree(d_xi8s_b);
    cudaFree(d_attn_i8_b); cudaFree(d_attn_i8s_b);
    cudaFree(d_gate_b); cudaFree(d_up_b); cudaFree(d_mlp_b);
    cudaFree(d_mlp_i8_b); cudaFree(d_mlp_i8s_b);
    cudaFree(d_res_s); cudaFree(d_xi8_s); cudaFree(d_xi8s_s);
    cudaFree(d_attn_out_s);
    cudaFree(d_rn); cudaFree(d_x32);
    cudaFree(d_kc); cudaFree(d_vc);
    return 0;
}
