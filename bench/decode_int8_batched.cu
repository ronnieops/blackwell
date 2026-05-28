// bench/decode_int8_batched.cu — Multi-sequence INT8 decode benchmark
//
// Simulates M concurrent decode sequences sharing the same model weights.
// Uses gemv_int8_batched for GEMV, serial per-token kernels for attention.
// Demonstrates weight-load amortization across sequences.
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/decode_int8_batched.cu build/libblackwell_kernels.a \
//     -o bench/decode_int8_batched

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
    int M = 4;  // batch size (sequences)
    if (argc > 1) num_layers = atoi(argv[1]);
    if (argc > 2) M = atoi(argv[2]);
    if (num_layers > 28) num_layers = 28;
    if (M < 1) M = 1;
    if (M > 8) M = 8;

    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("# INT8 Batched Decode Benchmark — Qwen3-1.7B\n");
    printf("Device: %s (%d.%d)\n", p.name, p.major, p.minor);
    printf("Layers: %d, Batch M: %d\n", num_layers, M);

    const int H = 2048, Q = 2048, KV = 1024, I = 6144;
    const int nqh = 16, nkv = 8, hd = 128, ms = 2048;
    const float s13 = 1.f/3.f, ixv = 1.f/127.f;

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

    // ── Buffers (M copies of per-token state) ────────────────────────────────
    // Batched GEMV I/O
    float *d_y;          // [M * max(Q,KV,H,I)]
    int8_t *d_x_batch;   // [M * max(H,I)]
    float *d_xs_batch;   // [M * max(H,I)/16]
    cudaMalloc(&d_y, M * I * 4);    // worst case: M * I
    cudaMalloc(&d_x_batch, M * I);
    cudaMalloc(&d_xs_batch, M * (I/16) * 4);

    // Per-sequence state
    float *d_Q_s, *d_K_s, *d_V_s, *d_attn_s, *d_proj_s;
    float *d_gate_s, *d_up_s, *d_mlp_s, *d_res_s;
    int8_t *d_xi8_s;
    float *d_xi8s_s, *d_attn_i8_s, *d_attn_i8s_s, *d_mlp_i8_s, *d_mlp_i8s_s;
    // Allocate M copies for attention (per-sequence KV, Q, K, V)
    // For simplicity: process attention sequentially per sequence, MLP batched
    
    // Per-seq buffers (reused across sequences in serial attention)
    cudaMalloc(&d_Q_s, Q*4); cudaMalloc(&d_K_s, KV*4); cudaMalloc(&d_V_s, KV*4);
    cudaMalloc(&d_attn_s, Q*4); cudaMalloc(&d_proj_s, H*4);
    cudaMalloc(&d_gate_s, I*4); cudaMalloc(&d_up_s, I*4); cudaMalloc(&d_mlp_s, I*4);
    cudaMalloc(&d_res_s, I*4);
    cudaMalloc(&d_xi8_s, I); cudaMalloc(&d_xi8s_s, (I/16)*4);
    cudaMalloc(&d_attn_i8_s, Q); cudaMalloc(&d_attn_i8s_s, (Q/16)*4);
    cudaMalloc(&d_mlp_i8_s, I); cudaMalloc(&d_mlp_i8s_s, (I/16)*4);

    // Per-sequence FP4 state (x_fp4 + xs + residual)
    void **d_x_fp4_arr = new void*[M];
    float **d_xs_arr = new float*[M];
    for (int m = 0; m < M; ++m) {
        cudaMalloc(&d_x_fp4_arr[m], H);
        cudaMalloc(&d_xs_arr[m], (H/16)*4);
    }
    float *d_rn; cudaMalloc(&d_rn, H*4);
    std::vector<float> rn_h(H, 1.f);
    cudaMemcpy(d_rn, rn_h.data(), H*4, cudaMemcpyHostToDevice);

    // KV cache per sequence per layer
    float *d_kc, *d_vc;
    size_t kv_sz = (size_t)M * num_layers * nkv * ms * hd * 4;
    cudaMalloc(&d_kc, kv_sz); cudaMalloc(&d_vc, kv_sz);
    cudaMemset(d_kc, 0, kv_sz); cudaMemset(d_vc, 0, kv_sz);

    // Init all sequences with same input (uniform 1.0)
    float *d_x32; cudaMalloc(&d_x32, H*4);
    std::vector<float> xh(H, 1.f), xsh(H/16, s13);
    cudaMemcpy(d_x32, xh.data(), H*4, cudaMemcpyHostToDevice);
    for (int m = 0; m < M; ++m) {
        cudaMemcpy(d_xs_arr[m], xsh.data(), (H/16)*4, cudaMemcpyHostToDevice);
        blackwell::kernels::pack_fp4(d_x_fp4_arr[m], d_x32, d_xs_arr[m], H, 0);
    }

    // ── Fill KV cache for all sequences ──────────────────────────────────────
    printf("Filling KV cache (%d sequences, seq=0..128)... ", M);
    fflush(stdout);
    int sq = 128;
    for (int s = 0; s <= sq; ++s) {
        for (int m = 0; m < M; ++m) {
            for (int l = 0; l < num_layers; ++l) {
                int kb = (m * num_layers + l) * nkv * ms * hd;
                blackwell::kernels::unpack_fp4(d_res_s, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
                blackwell::kernels::pack_int8(d_xi8_s, d_res_s, d_xi8s_s, H, 0);
                blackwell::kernels::gemv_int8(d_Q_s, d_xi8_s, d_xi8s_s, lw[l].q.d, lw[l].q.sc, H, Q, 0);
                blackwell::kernels::gemv_int8(d_K_s, d_xi8_s, d_xi8s_s, lw[l].k.d, lw[l].k.sc, H, KV, 0);
                blackwell::kernels::gemv_int8(d_V_s, d_xi8_s, d_xi8s_s, lw[l].v.d, lw[l].v.sc, H, KV, 0);
                blackwell::kernels::update_kv_cache(d_kc+kb, d_vc+kb, d_K_s, d_V_s, 0, s, nkv, hd, ms, 0);
                blackwell::kernels::attention_decode_gqa(d_attn_s, d_Q_s, d_kc+kb, d_vc+kb, s, nqh, nkv, hd, ms, 0);
                blackwell::kernels::pack_int8(d_attn_i8_s, d_attn_s, d_attn_i8s_s, Q, 0);
                blackwell::kernels::gemv_int8(d_proj_s, d_attn_i8_s, d_attn_i8s_s, lw[l].o.d, lw[l].o.sc, Q, H, 0);
                blackwell::kernels::vector_add_fp32(d_proj_s, d_proj_s, d_res_s, H, 0);
                blackwell::kernels::fused_rmsnorm_quant_int8(d_xi8_s, d_xi8s_s, d_proj_s, d_rn, H, 1e-6f, 0);
                blackwell::kernels::fused_rmsnorm_pack(d_x_fp4_arr[m], d_xs_arr[m], d_proj_s, d_rn, H, 1e-6f, 0);
                // MLP
                blackwell::kernels::unpack_fp4(d_res_s, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
                blackwell::kernels::pack_int8(d_xi8_s, d_res_s, d_xi8s_s, H, 0);
                blackwell::kernels::gemv_int8(d_gate_s, d_xi8_s, d_xi8s_s, lw[l].g.d, lw[l].g.sc, H, I, 0);
                blackwell::kernels::gemv_int8(d_up_s, d_xi8_s, d_xi8s_s, lw[l].u.d, lw[l].u.sc, H, I, 0);
                blackwell::kernels::apply_swiglu(d_mlp_s, d_gate_s, d_up_s, I, 0);
                blackwell::kernels::pack_int8(d_mlp_i8_s, d_mlp_s, d_mlp_i8s_s, I, 0);
                blackwell::kernels::gemv_int8(d_proj_s, d_mlp_i8_s, d_mlp_i8s_s, lw[l].d.d, lw[l].d.sc, I, H, 0);
                blackwell::kernels::vector_add_fp32(d_proj_s, d_proj_s, d_res_s, H, 0);
                blackwell::kernels::fused_rmsnorm_quant_int8(d_xi8_s, d_xi8s_s, d_proj_s, d_rn, H, 1e-6f, 0);
                blackwell::kernels::fused_rmsnorm_pack(d_x_fp4_arr[m], d_xs_arr[m], d_proj_s, d_rn, H, 1e-6f, 0);
            }
        }
    }
    printf("done\n");

    // ── Benchmark modes ─────────────────────────────────────────────────────
    int warm = 5, bench = 20;

    // Mode A: M independent sequences, each using single-token gemv_int8 (baseline)
    printf("Warmup (M=%d single-token)... \n", M);
    for (int w = 0; w < warm; ++w) {
        for (int l = 0; l < num_layers; ++l) {
            for (int m = 0; m < M; ++m) {
                int kb = (m * num_layers + l) * nkv * ms * hd;
                blackwell::kernels::unpack_fp4(d_res_s, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
                blackwell::kernels::pack_int8(d_xi8_s, d_res_s, d_xi8s_s, H, 0);
                blackwell::kernels::gemv_int8(d_Q_s, d_xi8_s, d_xi8s_s, lw[l].q.d, lw[l].q.sc, H, Q, 0);
                blackwell::kernels::gemv_int8(d_K_s, d_xi8_s, d_xi8s_s, lw[l].k.d, lw[l].k.sc, H, KV, 0);
                blackwell::kernels::gemv_int8(d_V_s, d_xi8_s, d_xi8s_s, lw[l].v.d, lw[l].v.sc, H, KV, 0);
                blackwell::kernels::update_kv_cache(d_kc+kb, d_vc+kb, d_K_s, d_V_s, 0, sq, nkv, hd, ms, 0);
                blackwell::kernels::attention_decode_gqa(d_attn_s, d_Q_s, d_kc+kb, d_vc+kb, sq, nqh, nkv, hd, ms, 0);
                blackwell::kernels::pack_int8(d_attn_i8_s, d_attn_s, d_attn_i8s_s, Q, 0);
                blackwell::kernels::gemv_int8(d_proj_s, d_attn_i8_s, d_attn_i8s_s, lw[l].o.d, lw[l].o.sc, Q, H, 0);
                blackwell::kernels::vector_add_fp32(d_proj_s, d_proj_s, d_res_s, H, 0);
                blackwell::kernels::fused_rmsnorm_quant_int8(d_xi8_s, d_xi8s_s, d_proj_s, d_rn, H, 1e-6f, 0);
                blackwell::kernels::fused_rmsnorm_pack(d_x_fp4_arr[m], d_xs_arr[m], d_proj_s, d_rn, H, 1e-6f, 0);
                blackwell::kernels::unpack_fp4(d_res_s, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
                blackwell::kernels::pack_int8(d_xi8_s, d_res_s, d_xi8s_s, H, 0);
                blackwell::kernels::gemv_int8(d_gate_s, d_xi8_s, d_xi8s_s, lw[l].g.d, lw[l].g.sc, H, I, 0);
                blackwell::kernels::gemv_int8(d_up_s, d_xi8_s, d_xi8s_s, lw[l].u.d, lw[l].u.sc, H, I, 0);
                blackwell::kernels::apply_swiglu(d_mlp_s, d_gate_s, d_up_s, I, 0);
                blackwell::kernels::pack_int8(d_mlp_i8_s, d_mlp_s, d_mlp_i8s_s, I, 0);
                blackwell::kernels::gemv_int8(d_proj_s, d_mlp_i8_s, d_mlp_i8s_s, lw[l].d.d, lw[l].d.sc, I, H, 0);
                blackwell::kernels::vector_add_fp32(d_proj_s, d_proj_s, d_res_s, H, 0);
                blackwell::kernels::fused_rmsnorm_quant_int8(d_xi8_s, d_xi8s_s, d_proj_s, d_rn, H, 1e-6f, 0);
                blackwell::kernels::fused_rmsnorm_pack(d_x_fp4_arr[m], d_xs_arr[m], d_proj_s, d_rn, H, 1e-6f, 0);
            }
        }
    }
    cudaDeviceSynchronize();

    // Benchmark Mode A: M×single-token
    printf("Benchmark M=%d single-token (%d iters)... \n", M, bench);
    GpuTimer ta;
    ta.start();
    for (int i = 0; i < bench; ++i) {
        for (int l = 0; l < num_layers; ++l) {
            for (int m = 0; m < M; ++m) {
                int kb = (m * num_layers + l) * nkv * ms * hd;
                blackwell::kernels::unpack_fp4(d_res_s, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
                blackwell::kernels::pack_int8(d_xi8_s, d_res_s, d_xi8s_s, H, 0);
                blackwell::kernels::gemv_int8(d_Q_s, d_xi8_s, d_xi8s_s, lw[l].q.d, lw[l].q.sc, H, Q, 0);
                blackwell::kernels::gemv_int8(d_K_s, d_xi8_s, d_xi8s_s, lw[l].k.d, lw[l].k.sc, H, KV, 0);
                blackwell::kernels::gemv_int8(d_V_s, d_xi8_s, d_xi8s_s, lw[l].v.d, lw[l].v.sc, H, KV, 0);
                blackwell::kernels::update_kv_cache(d_kc+kb, d_vc+kb, d_K_s, d_V_s, 0, sq, nkv, hd, ms, 0);
                blackwell::kernels::attention_decode_gqa(d_attn_s, d_Q_s, d_kc+kb, d_vc+kb, sq, nqh, nkv, hd, ms, 0);
                blackwell::kernels::pack_int8(d_attn_i8_s, d_attn_s, d_attn_i8s_s, Q, 0);
                blackwell::kernels::gemv_int8(d_proj_s, d_attn_i8_s, d_attn_i8s_s, lw[l].o.d, lw[l].o.sc, Q, H, 0);
                blackwell::kernels::vector_add_fp32(d_proj_s, d_proj_s, d_res_s, H, 0);
                blackwell::kernels::fused_rmsnorm_quant_int8(d_xi8_s, d_xi8s_s, d_proj_s, d_rn, H, 1e-6f, 0);
                blackwell::kernels::fused_rmsnorm_pack(d_x_fp4_arr[m], d_xs_arr[m], d_proj_s, d_rn, H, 1e-6f, 0);
                blackwell::kernels::unpack_fp4(d_res_s, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
                blackwell::kernels::pack_int8(d_xi8_s, d_res_s, d_xi8s_s, H, 0);
                // Use batched GEMV for gate+up (2 biggest weights)
                // For single-token baseline, just call gemv_int8 twice
                blackwell::kernels::gemv_int8(d_gate_s, d_xi8_s, d_xi8s_s, lw[l].g.d, lw[l].g.sc, H, I, 0);
                blackwell::kernels::gemv_int8(d_up_s, d_xi8_s, d_xi8s_s, lw[l].u.d, lw[l].u.sc, H, I, 0);
                blackwell::kernels::apply_swiglu(d_mlp_s, d_gate_s, d_up_s, I, 0);
                blackwell::kernels::pack_int8(d_mlp_i8_s, d_mlp_s, d_mlp_i8s_s, I, 0);
                blackwell::kernels::gemv_int8(d_proj_s, d_mlp_i8_s, d_mlp_i8s_s, lw[l].d.d, lw[l].d.sc, I, H, 0);
                blackwell::kernels::vector_add_fp32(d_proj_s, d_proj_s, d_res_s, H, 0);
                blackwell::kernels::fused_rmsnorm_quant_int8(d_xi8_s, d_xi8s_s, d_proj_s, d_rn, H, 1e-6f, 0);
                blackwell::kernels::fused_rmsnorm_pack(d_x_fp4_arr[m], d_xs_arr[m], d_proj_s, d_rn, H, 1e-6f, 0);
            }
        }
    }
    float ms_a = ta.stop();

    // ── Mode B: Batched GEMV for gate+up+down ───────────────────────────────
    // Pack M sequences' INT8 inputs, call gemv_int8_batched with M
    // Attention still serial per-sequence (KV cache is per-seq)
    // MLP: batched gate+up GEMV, serial swiglu per-seq, batched down GEMV

    printf("Benchmark M=%d batched (%d iters)... \n", M, bench);
    // Need per-sequence gate/up/down output buffers packed for batched
    float *d_gate_b, *d_up_b, *d_mlp_b, *d_proj_b;
    int8_t *d_mlp_i8_b;
    float *d_mlp_i8s_b;
    cudaMalloc(&d_gate_b, M * I * 4);
    cudaMalloc(&d_up_b, M * I * 4);
    cudaMalloc(&d_mlp_b, M * I * 4);
    cudaMalloc(&d_proj_b, M * H * 4);
    cudaMalloc(&d_mlp_i8_b, M * I);
    cudaMalloc(&d_mlp_i8s_b, M * (I/16) * 4);

    // Per-seq xi8 packed for batched
    int8_t *d_xi8_b;
    float *d_xi8s_b;
    cudaMalloc(&d_xi8_b, M * H);
    cudaMalloc(&d_xi8s_b, M * (H/16) * 4);

    GpuTimer tb;
    tb.start();
    for (int i = 0; i < bench; ++i) {
        for (int l = 0; l < num_layers; ++l) {
            // Pack all M sequences' INT8 inputs
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::unpack_fp4(d_res_s, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
                blackwell::kernels::pack_int8(d_xi8_b + m*H, d_res_s, d_xi8s_b + m*(H/16), H, 0);
            }

            // Attention: serial per-sequence (KV cache unique)
            for (int m = 0; m < M; ++m) {
                int kb = (m * num_layers + l) * nkv * ms * hd;
                blackwell::kernels::gemv_int8(d_Q_s, d_xi8_b + m*H, d_xi8s_b + m*(H/16), lw[l].q.d, lw[l].q.sc, H, Q, 0);
                blackwell::kernels::gemv_int8(d_K_s, d_xi8_b + m*H, d_xi8s_b + m*(H/16), lw[l].k.d, lw[l].k.sc, H, KV, 0);
                blackwell::kernels::gemv_int8(d_V_s, d_xi8_b + m*H, d_xi8s_b + m*(H/16), lw[l].v.d, lw[l].v.sc, H, KV, 0);
                blackwell::kernels::update_kv_cache(d_kc+kb, d_vc+kb, d_K_s, d_V_s, 0, sq, nkv, hd, ms, 0);
                blackwell::kernels::attention_decode_gqa(d_attn_s, d_Q_s, d_kc+kb, d_vc+kb, sq, nqh, nkv, hd, ms, 0);
                blackwell::kernels::pack_int8(d_attn_i8_s, d_attn_s, d_attn_i8s_s, Q, 0);
                blackwell::kernels::gemv_int8(d_proj_s, d_attn_i8_s, d_attn_i8s_s, lw[l].o.d, lw[l].o.sc, Q, H, 0);
                blackwell::kernels::unpack_fp4(d_res_s, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
                blackwell::kernels::vector_add_fp32(d_proj_s, d_proj_s, d_res_s, H, 0);
                blackwell::kernels::fused_rmsnorm_quant_int8(d_xi8_s, d_xi8s_s, d_proj_s, d_rn, H, 1e-6f, 0);
                // Copy INT8 x back to batched buffer
                cudaMemcpy(d_xi8_b + m*H, d_xi8_s, H, cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_xi8s_b + m*(H/16), d_xi8s_s, (H/16)*4, cudaMemcpyDeviceToDevice);
                blackwell::kernels::fused_rmsnorm_pack(d_x_fp4_arr[m], d_xs_arr[m], d_proj_s, d_rn, H, 1e-6f, 0);
            }

            // MLP: BATCHED gate+up GEMV (weight reuse!)
            blackwell::kernels::gemv_int8_batched(d_gate_b, d_xi8_b, d_xi8s_b, lw[l].g.d, lw[l].g.sc, H, I, M, 0);
            blackwell::kernels::gemv_int8_batched(d_up_b, d_xi8_b, d_xi8s_b, lw[l].u.d, lw[l].u.sc, H, I, M, 0);

            // Swiglu: per-sequence (elementwise, cheap)
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::apply_swiglu(d_mlp_b + m*I, d_gate_b + m*I, d_up_b + m*I, I, 0);
                blackwell::kernels::pack_int8(d_mlp_i8_b + m*I, d_mlp_b + m*I, d_mlp_i8s_b + m*(I/16), I, 0);
            }

            // BATCHED down_proj GEMV (weight reuse!)
            blackwell::kernels::gemv_int8_batched(d_proj_b, d_mlp_i8_b, d_mlp_i8s_b, lw[l].d.d, lw[l].d.sc, I, H, M, 0);

            // Per-sequence: residual + rmsnorm
            for (int m = 0; m < M; ++m) {
                blackwell::kernels::unpack_fp4(d_res_s, d_x_fp4_arr[m], d_xs_arr[m], H, 0);
                blackwell::kernels::vector_add_fp32(d_proj_b + m*H, d_proj_b + m*H, d_res_s, H, 0);
                blackwell::kernels::fused_rmsnorm_quant_int8(d_xi8_s, d_xi8s_s, d_proj_b + m*H, d_rn, H, 1e-6f, 0);
                blackwell::kernels::fused_rmsnorm_pack(d_x_fp4_arr[m], d_xs_arr[m], d_proj_b + m*H, d_rn, H, 1e-6f, 0);
            }
        }
    }
    float ms_b = tb.stop();

    // ── Results ──────────────────────────────────────────────────────────────
    float pt_a = ms_a / bench;
    float pt_b = ms_b / bench;
    float tps_a = M * 1000.f / pt_a;   // total tokens/sec across all sequences
    float tps_b = M * 1000.f / pt_b;
    float s28_a = 1000.f / (pt_a * 28.f / num_layers);
    float s28_b = 1000.f / (pt_b * 28.f / num_layers);

    printf("\n=== Results (M=%d sequences, %d layers) ===\n", M, num_layers);
    printf("  %-25s  %8s  %10s  %8s\n", "Method", "Per-step", "Total t/s", "Scaled28");
    printf("  %-25s  %7.3fms  %8.1f    %7.1f\n", "M×single-token",
        pt_a, tps_a, s28_a);
    printf("  %-25s  %7.3fms  %8.1f    %7.1f\n", "Batched MLP",
        pt_b, tps_b, s28_b);
    printf("  Speedup: %.2fx\n", pt_a / pt_b);
    printf("  Per-sequence throughput: %.1f t/s (batched)\n", 1000.f / pt_b);
    printf("  Effective throughput: %.1f t/s (batched, %d seq)\n", tps_b, M);

    // Cleanup
    for (int m = 0; m < M; ++m) { cudaFree(d_x_fp4_arr[m]); cudaFree(d_xs_arr[m]); }
    delete[] d_x_fp4_arr; delete[] d_xs_arr;
    for (auto& l : lw) {
        cudaFree(l.q.d); cudaFree(l.q.sc); cudaFree(l.k.d); cudaFree(l.k.sc);
        cudaFree(l.v.d); cudaFree(l.v.sc); cudaFree(l.o.d); cudaFree(l.o.sc);
        cudaFree(l.g.d); cudaFree(l.g.sc); cudaFree(l.u.d); cudaFree(l.u.sc);
        cudaFree(l.d.d); cudaFree(l.d.sc);
    }
    cudaFree(d_x32); cudaFree(d_rn); cudaFree(d_y);
    cudaFree(d_x_batch); cudaFree(d_xs_batch);
    cudaFree(d_Q_s); cudaFree(d_K_s); cudaFree(d_V_s);
    cudaFree(d_attn_s); cudaFree(d_proj_s);
    cudaFree(d_gate_s); cudaFree(d_up_s); cudaFree(d_mlp_s); cudaFree(d_res_s);
    cudaFree(d_xi8_s); cudaFree(d_xi8s_s);
    cudaFree(d_attn_i8_s); cudaFree(d_attn_i8s_s);
    cudaFree(d_mlp_i8_s); cudaFree(d_mlp_i8s_s);
    cudaFree(d_gate_b); cudaFree(d_up_b); cudaFree(d_mlp_b); cudaFree(d_proj_b);
    cudaFree(d_mlp_i8_b); cudaFree(d_mlp_i8s_b);
    cudaFree(d_xi8_b); cudaFree(d_xi8s_b);
    cudaFree(d_kc); cudaFree(d_vc);
    return 0;
}
