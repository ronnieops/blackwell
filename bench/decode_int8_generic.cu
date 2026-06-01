// bench/decode_int8_generic.cu — Generic INT8 CUDA Graph decode benchmark
//
// Supports Qwen3-0.6B, 1.7B, 8B and variants. Accepts model dims as args.
//
// Usage:
//   ./bench/decode_int8_generic <num_layers> <weight_dir> <H> <Q> <KV> <I> <nqh> <nkv> <model_name>
//
// Examples:
//   ./bench/decode_int8_generic 28 weights_int8_qwen3_06b 1024 1024 512 3072 8 4 "Qwen3-0.6B"
//   ./bench/decode_int8_generic 28 weights_int8_bf16 2048 2048 1024 6144 16 8 "Qwen3-1.7B"
//   ./bench/decode_int8_generic 36 weights_int8_qwen3_8b 4096 4096 1024 12288 32 8 "Qwen3-8B"
//
// Build:
//   export PATH=/usr/local/cuda-13.3/bin:$PATH
//   CUDACXX=/usr/local/cuda-13.3/bin/nvcc nvcc -O3 -std=c++17 \
//     -arch=sm_120a -I include bench/decode_int8_generic.cu \
//     build/libblackwell_kernels.a -o bench/decode_int8_generic

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

// Each layer uses these buffers. All reused across layers.
struct LayerBufs {
    float *d_Q, *d_K, *d_V;
    float *d_attn;         // attention output (Q-sized)
    float *d_proj;         // temporary (H-sized)
    float *d_gate, *d_up, *d_mlp;  // MLP (I-sized)
    float *d_res;          // residual + unpack output (max(H,I)-sized)
    void *d_x_fp4;         // FP4 packed state (H bytes)
    float *d_xs;          // FP4 scales ((H/16)*4 bytes)
    int8_t *d_x_int8;      // INT8 input (H bytes)
    float *d_x_int8_s;    // INT8 scales ((H/16)*4 bytes)
    int8_t *d_attn_i8;    // INT8 attn output (Q bytes)
    float *d_attn_i8s;    // attn scales ((Q/16)*4 bytes)
    int8_t *d_mlp_i8;      // INT8 MLP output (I bytes)
    float *d_mlp_i8s;     // MLP scales ((I/16)*4 bytes)
};

int main(int argc, char** argv) {
    if (argc < 10) {
        printf("Usage: %s <num_layers> <weight_dir> <H> <Q> <KV> <I> <nqh> <nkv> <model_name>\n", argv[0]);
        printf("Examples:\n");
        printf("  %s 28 weights_int8_qwen3_06b 1024 1024 512 3072 8 4 \"Qwen3-0.6B\"\n", argv[0]);
        printf("  %s 28 weights_int8_bf16 2048 2048 1024 6144 16 8 \"Qwen3-1.7B\"\n", argv[0]);
        printf("  %s 36 weights_int8_qwen3_8b 4096 4096 1024 12288 32 8 \"Qwen3-8B\"\n", argv[0]);
        exit(1);
    }

    int num_layers = atoi(argv[1]);
    const char* WDIR = argv[2];
    int H  = atoi(argv[3]);
    int Q  = atoi(argv[4]);      // Q = H for self-attention
    int KV = atoi(argv[5]);      // KV = num_kv_heads * head_dim (typically 512 or 1024)
    int I  = atoi(argv[6]);      // intermediate size
    int nqh = atoi(argv[7]);     // num_q_heads
    int nkv = atoi(argv[8]);     // num_kv_heads
    const char* MODEL_NAME = argv[9];
    const int head_dim = 128;
    const int max_seq = 2048;
    const int big = (H > I) ? H : I;  // max(H,I) for d_res

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop,0);
    printf("# INT8 CUDA Graph Decode — %s\n", MODEL_NAME);
    printf("# Device: %s (%d.%d), Layers: %d\n", prop.name, prop.major, prop.minor, num_layers);
    printf("# H=%d Q=%d KV=%d I=%d nqh=%d nkv=%d head_dim=%d big=%d\n",
        H, Q, KV, I, nqh, nkv, head_dim, big);
    fflush(stdout);

    // Load weights
    struct LW { DevW q,k,v,o,g,u,d; };
    printf("Loading INT8 weights from %s...\n", WDIR); fflush(stdout);
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

    // Allocate buffers (d_res = big = max(H,I) to handle MLP residual and fused_rmsnorm output)
    LayerBufs b;
    cudaMalloc(&b.d_Q, Q*4); cudaMalloc(&b.d_K, KV*4); cudaMalloc(&b.d_V, KV*4);
    cudaMalloc(&b.d_attn, Q*4);
    cudaMalloc(&b.d_proj, H*4);    // attention o_proj: Q→H
    cudaMalloc(&b.d_gate, I*4); cudaMalloc(&b.d_up, I*4); cudaMalloc(&b.d_mlp, I*4);
    cudaMalloc(&b.d_res, big*4);    // unpack output, MLP residual, RMSNorm output
    cudaMalloc(&b.d_x_fp4, H); cudaMalloc(&b.d_xs, (H/16)*4);
    cudaMalloc(&b.d_x_int8, H); cudaMalloc(&b.d_x_int8_s, (H/16)*4);
    cudaMalloc(&b.d_attn_i8, Q); cudaMalloc(&b.d_attn_i8s, (Q/16)*4);
    cudaMalloc(&b.d_mlp_i8, I); cudaMalloc(&b.d_mlp_i8s, (I/16)*4);

    // RMSNorm weight (all ones, tiny)
    float* d_rn; cudaMalloc(&d_rn, big*4);
    std::vector<float> rn(big, 1.f);
    cudaMemcpy(d_rn, rn.data(), big*4, cudaMemcpyHostToDevice);

    // Init x = uniform 1.0 → RMSNorm (separate kernels, no 2048 limit)
    float s13 = 1.f/3.f, ixv = 1.f/127.f;
    std::vector<float> x_init(H, 1.f), xsh(H/16, s13);
    float* d_x_tmp; cudaMalloc(&d_x_tmp, H*4);
    cudaMemcpy(d_x_tmp, x_init.data(), H*4, cudaMemcpyHostToDevice);
    cudaMemcpy(b.d_xs, xsh.data(), (H/16)*4, cudaMemcpyHostToDevice);
    chk(blackwell::kernels::fused_rmsnorm(d_x_tmp, d_rn, d_x_tmp, H, 1e-6f, 0), "init_rn");
    chk(blackwell::kernels::pack_fp4(b.d_x_fp4, d_x_tmp, b.d_xs, H, 0), "init_pack");
    cudaFree(d_x_tmp);
    // Init scales
    std::vector<float> ixsh(H/16, ixv), ai8s(Q/16, ixv), mi8s(I/16, ixv);
    cudaMemcpy(b.d_x_int8_s, ixsh.data(), (H/16)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(b.d_attn_i8s, ai8s.data(), (Q/16)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(b.d_mlp_i8s, mi8s.data(), (I/16)*4, cudaMemcpyHostToDevice);

    // KV cache
    float *d_kc, *d_vc;
    size_t kv_sz = (size_t)num_layers * nkv * max_seq * head_dim * 4;
    cudaMalloc(&d_kc, kv_sz); cudaMalloc(&d_vc, kv_sz);
    cudaMemset(d_kc, 0, kv_sz); cudaMemset(d_vc, 0, kv_sz);

    // L2 persisting cache for norm weights
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 8*1024*1024);

    // ── Fill KV cache (seq=0..128) ──────────────────────────────────────────
    printf("Filling KV cache (seq=0..128)... "); fflush(stdout);
    int sq = 128;
    for (int s = 0; s <= sq; ++s) {
        for (int l = 0; l < num_layers; ++l) {
            // Unpack FP4 state → FP32 residual
            blackwell::kernels::unpack_fp4(b.d_res, b.d_x_fp4, b.d_xs, H, 0);
            // Quantize residual to INT8 for GEMV
            blackwell::kernels::pack_int8(b.d_x_int8, b.d_res, b.d_x_int8_s, H, 0);

            int kb = l * nkv * max_seq * head_dim;

            // QKV projections
            chk(blackwell::kernels::gemv_int8_warp(b.d_Q, b.d_x_int8, b.d_x_int8_s,
                lw[l].q.d, lw[l].q.sc, H, Q, 0), "Q");
            chk(blackwell::kernels::gemv_int8_warp(b.d_K, b.d_x_int8, b.d_x_int8_s,
                lw[l].k.d, lw[l].k.sc, H, KV, 0), "K");
            chk(blackwell::kernels::gemv_int8_warp(b.d_V, b.d_x_int8, b.d_x_int8_s,
                lw[l].v.d, lw[l].v.sc, H, KV, 0), "V");

            // Update KV cache
            chk(blackwell::kernels::update_kv_cache(
                d_kc+kb, d_vc+kb, b.d_K, b.d_V, 0, s, nkv, head_dim, max_seq, 0), "kv");

            // Attention
            chk(blackwell::kernels::attention_decode_gqa(
                b.d_attn, b.d_Q, d_kc+kb, d_vc+kb,
                s, nqh, nkv, head_dim, max_seq, 0), "attn");

            // O projection: quantize attn output → INT8 → GEMV → FP32
            chk(blackwell::kernels::pack_int8(b.d_attn_i8, b.d_attn, b.d_attn_i8s, Q, 0), "pack_attn");
            chk(blackwell::kernels::gemv_int8_warp(b.d_proj, b.d_attn_i8, b.d_attn_i8s,
                lw[l].o.d, lw[l].o.sc, Q, H, 0), "O");

            // Residual add + fused RMSNorm+quant → d_x_int8 (saves d_res for MLP)
            blackwell::kernels::vector_add_fp32(b.d_proj, b.d_proj, b.d_res, H, 0);
            chk(blackwell::kernels::fused_rmsnorm_quant_int8(
                b.d_x_int8, b.d_x_int8_s, b.d_proj, d_rn, H, 1e-6f, 0), "rnq_attn");
            chk(blackwell::kernels::fused_rmsnorm_pack(
                b.d_x_fp4, b.d_xs, b.d_proj, d_rn, H, 1e-6f, 0), "rnp_attn");

            // MLP: unpack new FP4 state, quantize to INT8
            blackwell::kernels::unpack_fp4(b.d_res, b.d_x_fp4, b.d_xs, H, 0);
            blackwell::kernels::pack_int8(b.d_x_int8, b.d_res, b.d_x_int8_s, H, 0);
                lw[l].g.d, lw[l].g.sc, H, I, 0), "gate");
            chk(blackwell::kernels::gemv_int8_warp(b.d_up, b.d_x_int8, b.d_x_int8_s,
                lw[l].u.d, lw[l].u.sc, H, I, 0), "up");
            chk(blackwell::kernels::apply_swiglu(b.d_mlp, b.d_gate, b.d_up, I, 0), "swiglu");

            // Down projection: quantize MLP output → INT8 → GEMV → FP32
            chk(blackwell::kernels::pack_int8(b.d_mlp_i8, b.d_mlp, b.d_mlp_i8s, I, 0), "pack_mlp");
            chk(blackwell::kernels::gemv_int8_warp(b.d_proj, b.d_mlp_i8, b.d_mlp_i8s,
                lw[l].d.d, lw[l].d.sc, I, H, 0), "down");

            // MLP residual add + fused RMSNorm+quant + repack
            blackwell::kernels::vector_add_fp32(b.d_proj, b.d_proj, b.d_res, H, 0);
            chk(blackwell::kernels::fused_rmsnorm_quant_int8(
                b.d_x_int8, b.d_x_int8_s, b.d_proj, d_rn, H, 1e-6f, 0), "rnq_mlp");
            chk(blackwell::kernels::fused_rmsnorm_pack(
                b.d_x_fp4, b.d_xs, b.d_proj, d_rn, H, 1e-6f, 0), "rnp_mlp");
        }
    }
    cudaStreamSynchronize(0);
    printf("done\n"); fflush(stdout);

    // Save initial state for correctness comparison
    void* d_x_fp4_saved; float* d_xs_saved;
    cudaMalloc(&d_x_fp4_saved, H); cudaMalloc(&d_xs_saved, (H/16)*4);
    cudaMemcpy(d_x_fp4_saved, b.d_x_fp4, H, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_xs_saved, b.d_xs, (H/16)*4, cudaMemcpyDeviceToDevice);
    float* d_kc_saved; float* d_vc_saved;
    cudaMalloc(&d_kc_saved, kv_sz); cudaMalloc(&d_vc_saved, kv_sz);
    cudaMemcpy(d_kc_saved, d_kc, kv_sz, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_vc_saved, d_vc, kv_sz, cudaMemcpyDeviceToDevice);

    const int WARMUP = 5, ITERS = 20;

    // ── Per-kernel benchmark ────────────────────────────────────────────────
    printf("Per-kernel benchmark (%d warmup + %d iters)...\n", WARMUP, ITERS); fflush(stdout);

    for (int w = 0; w < WARMUP; ++w) {
        for (int l = 0; l < num_layers; ++l) {
            blackwell::kernels::unpack_fp4(b.d_res, b.d_x_fp4, b.d_xs, H, 0);
            blackwell::kernels::pack_int8(b.d_x_int8, b.d_res, b.d_x_int8_s, H, 0);
            int kb = l * nkv * max_seq * head_dim;
            blackwell::kernels::gemv_int8_warp(b.d_Q, b.d_x_int8, b.d_x_int8_s, lw[l].q.d, lw[l].q.sc, H, Q, 0);
            blackwell::kernels::gemv_int8_warp(b.d_K, b.d_x_int8, b.d_x_int8_s, lw[l].k.d, lw[l].k.sc, H, KV, 0);
            blackwell::kernels::gemv_int8_warp(b.d_V, b.d_x_int8, b.d_x_int8_s, lw[l].v.d, lw[l].v.sc, H, KV, 0);
            blackwell::kernels::update_kv_cache(d_kc+kb, d_vc+kb, b.d_K, b.d_V, 0, sq, nkv, head_dim, max_seq, 0);
            blackwell::kernels::attention_decode_gqa(b.d_attn, b.d_Q, d_kc+kb, d_vc+kb, sq, nqh, nkv, head_dim, max_seq, 0);
            blackwell::kernels::pack_int8(b.d_attn_i8, b.d_attn, b.d_attn_i8s, Q, 0);
            blackwell::kernels::gemv_int8_warp(b.d_proj, b.d_attn_i8, b.d_attn_i8s, lw[l].o.d, lw[l].o.sc, Q, H, 0);
            blackwell::kernels::vector_add_fp32(b.d_proj, b.d_proj, b.d_res, H, 0);
            blackwell::kernels::fused_rmsnorm_quant_int8(
                b.d_x_int8, b.d_x_int8_s, b.d_proj, d_rn, H, 1e-6f, 0);
            blackwell::kernels::fused_rmsnorm_pack(
                b.d_x_fp4, b.d_xs, b.d_proj, d_rn, H, 1e-6f, 0);
            // MLP: unpack new FP4 state
            blackwell::kernels::unpack_fp4(b.d_res, b.d_x_fp4, b.d_xs, H, 0);
            blackwell::kernels::pack_int8(b.d_x_int8, b.d_res, b.d_x_int8_s, H, 0);
            blackwell::kernels::gemv_int8_warp(b.d_gate, b.d_x_int8, b.d_x_int8_s, lw[l].g.d, lw[l].g.sc, H, I, 0);
            blackwell::kernels::gemv_int8_warp(b.d_up, b.d_x_int8, b.d_x_int8_s, lw[l].u.d, lw[l].u.sc, H, I, 0);
            blackwell::kernels::apply_swiglu(b.d_mlp, b.d_gate, b.d_up, I, 0);
            blackwell::kernels::pack_int8(b.d_mlp_i8, b.d_mlp, b.d_mlp_i8s, I, 0);
            blackwell::kernels::gemv_int8_warp(b.d_proj, b.d_mlp_i8, b.d_mlp_i8s, lw[l].d.d, lw[l].d.sc, I, H, 0);
            blackwell::kernels::vector_add_fp32(b.d_proj, b.d_proj, b.d_res, H, 0);
            blackwell::kernels::fused_rmsnorm_quant_int8(
                b.d_x_int8, b.d_x_int8_s, b.d_proj, d_rn, H, 1e-6f, 0);
            blackwell::kernels::fused_rmsnorm_pack(
                b.d_x_fp4, b.d_xs, b.d_proj, d_rn, H, 1e-6f, 0);
        }
    }
    cudaDeviceSynchronize();

    GpuTimer t0;
    t0.start();
    for (int i = 0; i < ITERS; ++i) {
        for (int l = 0; l < num_layers; ++l) {
            blackwell::kernels::unpack_fp4(b.d_res, b.d_x_fp4, b.d_xs, H, 0);
            blackwell::kernels::pack_int8(b.d_x_int8, b.d_res, b.d_x_int8_s, H, 0);
            int kb = l * nkv * max_seq * head_dim;
            blackwell::kernels::gemv_int8_warp(b.d_Q, b.d_x_int8, b.d_x_int8_s, lw[l].q.d, lw[l].q.sc, H, Q, 0);
            blackwell::kernels::gemv_int8_warp(b.d_K, b.d_x_int8, b.d_x_int8_s, lw[l].k.d, lw[l].k.sc, H, KV, 0);
            blackwell::kernels::gemv_int8_warp(b.d_V, b.d_x_int8, b.d_x_int8_s, lw[l].v.d, lw[l].v.sc, H, KV, 0);
            blackwell::kernels::update_kv_cache(d_kc+kb, d_vc+kb, b.d_K, b.d_V, 0, sq, nkv, head_dim, max_seq, 0);
            blackwell::kernels::attention_decode_gqa(b.d_attn, b.d_Q, d_kc+kb, d_vc+kb, sq, nqh, nkv, head_dim, max_seq, 0);
            blackwell::kernels::pack_int8(b.d_attn_i8, b.d_attn, b.d_attn_i8s, Q, 0);
            blackwell::kernels::gemv_int8_warp(b.d_proj, b.d_attn_i8, b.d_attn_i8s, lw[l].o.d, lw[l].o.sc, Q, H, 0);
            blackwell::kernels::vector_add_fp32(b.d_proj, b.d_proj, b.d_res, H, 0);
            blackwell::kernels::fused_rmsnorm_quant_int8(
                b.d_x_int8, b.d_x_int8_s, b.d_proj, d_rn, H, 1e-6f, 0);
            blackwell::kernels::fused_rmsnorm_pack(
                b.d_x_fp4, b.d_xs, b.d_proj, d_rn, H, 1e-6f, 0);
            // MLP: unpack new FP4 state
            blackwell::kernels::unpack_fp4(b.d_res, b.d_x_fp4, b.d_xs, H, 0);
            blackwell::kernels::pack_int8(b.d_x_int8, b.d_res, b.d_x_int8_s, H, 0);
            blackwell::kernels::gemv_int8_warp(b.d_gate, b.d_x_int8, b.d_x_int8_s, lw[l].g.d, lw[l].g.sc, H, I, 0);
            blackwell::kernels::gemv_int8_warp(b.d_up, b.d_x_int8, b.d_x_int8_s, lw[l].u.d, lw[l].u.sc, H, I, 0);
            blackwell::kernels::apply_swiglu(b.d_mlp, b.d_gate, b.d_up, I, 0);
            blackwell::kernels::pack_int8(b.d_mlp_i8, b.d_mlp, b.d_mlp_i8s, I, 0);
            blackwell::kernels::gemv_int8_warp(b.d_proj, b.d_mlp_i8, b.d_mlp_i8s, lw[l].d.d, lw[l].d.sc, I, H, 0);
            blackwell::kernels::vector_add_fp32(b.d_proj, b.d_proj, b.d_res, H, 0);
            blackwell::kernels::fused_rmsnorm_quant_int8(
                b.d_x_int8, b.d_x_int8_s, b.d_proj, d_rn, H, 1e-6f, 0);
            blackwell::kernels::fused_rmsnorm_pack(
                b.d_x_fp4, b.d_xs, b.d_proj, d_rn, H, 1e-6f, 0);
        }
    }
    float pk_ms = t0.stop() / ITERS;
    float pk_ts = 1000.f / pk_ms;
    float pk_s28 = pk_ts * 28.f / num_layers;

    // Save per-kernel final state
    std::vector<float> pk_out(H);
    float* d_tmp; cudaMalloc(&d_tmp, H*4);
    blackwell::kernels::unpack_fp4(d_tmp, b.d_x_fp4, b.d_xs, H, 0);
    cudaMemcpy(pk_out.data(), d_tmp, H*4, cudaMemcpyDeviceToHost);

    // Restore initial state
    cudaMemcpy(b.d_x_fp4, d_x_fp4_saved, H, cudaMemcpyDeviceToDevice);
    cudaMemcpy(b.d_xs, d_xs_saved, (H/16)*4, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_kc, d_kc_saved, kv_sz, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_vc, d_vc_saved, kv_sz, cudaMemcpyDeviceToDevice);
    cudaFree(d_x_fp4_saved); cudaFree(d_xs_saved); cudaFree(d_kc_saved); cudaFree(d_vc_saved);

    // ── CUDA Graph benchmark ─────────────────────────────────────────────────
    printf("\nCapturing CUDA Graph (%d layers, %d kernels)... ", num_layers, num_layers*20);
    fflush(stdout);

    cudaStream_t gst; cudaStreamCreate(&gst);

    // L2 persisting hint on graph stream
    cudaAccessPolicyWindow win;
    win.base_ptr = (void*)d_rn;
    win.num_bytes = big * 4;
    win.hitRatio = 1.0f;
    win.hitProp = cudaAccessPropertyPersisting;
    win.missProp = cudaAccessPropertyStreaming;
    cudaStreamAttrValue av;
    av.accessPolicyWindow = win;
    cudaStreamSetAttribute(gst, cudaStreamAttributeAccessPolicyWindow, &av);

    // Pre-trigger attention kernel to set smem config
    int kb0 = 0 * nkv * max_seq * head_dim;
    blackwell::kernels::attention_decode_gqa(b.d_attn, b.d_Q, d_kc+kb0, d_vc+kb0, sq, nqh, nkv, head_dim, max_seq, gst);
    cudaStreamSynchronize(gst);

    cudaGraph_t graph; cudaGraphExec_t exec;
    cudaStreamBeginCapture(gst, cudaStreamCaptureModeGlobal);
    for (int l = 0; l < num_layers; ++l) {
        int kb = l * nkv * max_seq * head_dim;
        blackwell::kernels::unpack_fp4(b.d_res, b.d_x_fp4, b.d_xs, H, gst);
        blackwell::kernels::pack_int8(b.d_x_int8, b.d_res, b.d_x_int8_s, H, gst);
        blackwell::kernels::gemv_int8_warp(b.d_Q, b.d_x_int8, b.d_x_int8_s, lw[l].q.d, lw[l].q.sc, H, Q, gst);
        blackwell::kernels::gemv_int8_warp(b.d_K, b.d_x_int8, b.d_x_int8_s, lw[l].k.d, lw[l].k.sc, H, KV, gst);
        blackwell::kernels::gemv_int8_warp(b.d_V, b.d_x_int8, b.d_x_int8_s, lw[l].v.d, lw[l].v.sc, H, KV, gst);
        blackwell::kernels::update_kv_cache(d_kc+kb, d_vc+kb, b.d_K, b.d_V, 0, sq, nkv, head_dim, max_seq, gst);
        blackwell::kernels::attention_decode_gqa(b.d_attn, b.d_Q, d_kc+kb, d_vc+kb, sq, nqh, nkv, head_dim, max_seq, gst);
        blackwell::kernels::pack_int8(b.d_attn_i8, b.d_attn, b.d_attn_i8s, Q, gst);
        blackwell::kernels::gemv_int8_warp(b.d_proj, b.d_attn_i8, b.d_attn_i8s, lw[l].o.d, lw[l].o.sc, Q, H, gst);
        blackwell::kernels::vector_add_fp32(b.d_proj, b.d_proj, b.d_res, H, gst);
        blackwell::kernels::fused_rmsnorm_quant_int8(
            b.d_x_int8, b.d_x_int8_s, b.d_proj, d_rn, H, 1e-6f, gst);
        blackwell::kernels::fused_rmsnorm_pack(
            b.d_x_fp4, b.d_xs, b.d_proj, d_rn, H, 1e-6f, gst);
        // MLP: unpack new FP4 state
        blackwell::kernels::unpack_fp4(b.d_res, b.d_x_fp4, b.d_xs, H, gst);
        blackwell::kernels::pack_int8(b.d_x_int8, b.d_res, b.d_x_int8_s, H, gst);
        blackwell::kernels::gemv_int8_warp(b.d_gate, b.d_x_int8, b.d_x_int8_s, lw[l].g.d, lw[l].g.sc, H, I, gst);
        blackwell::kernels::gemv_int8_warp(b.d_up, b.d_x_int8, b.d_x_int8_s, lw[l].u.d, lw[l].u.sc, H, I, gst);
        blackwell::kernels::apply_swiglu(b.d_mlp, b.d_gate, b.d_up, I, gst);
        blackwell::kernels::pack_int8(b.d_mlp_i8, b.d_mlp, b.d_mlp_i8s, I, gst);
        blackwell::kernels::gemv_int8_warp(b.d_proj, b.d_mlp_i8, b.d_mlp_i8s, lw[l].d.d, lw[l].d.sc, I, H, gst);
        blackwell::kernels::vector_add_fp32(b.d_proj, b.d_proj, b.d_res, H, gst);
        blackwell::kernels::fused_rmsnorm_quant_int8(
            b.d_x_int8, b.d_x_int8_s, b.d_proj, d_rn, H, 1e-6f, gst);
        blackwell::kernels::fused_rmsnorm_pack(
            b.d_x_fp4, b.d_xs, b.d_proj, d_rn, H, 1e-6f, gst);
    }
    cudaStreamEndCapture(gst, &graph);
    cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);
    printf("OK\n"); fflush(stdout);

    for (int i = 0; i < WARMUP; ++i) chk(cudaGraphLaunch(exec, gst), "warmup");
    cudaStreamSynchronize(gst);

    GpuTimer t1;
    t1.start();
    for (int i = 0; i < ITERS; ++i) chk(cudaGraphLaunch(exec, gst), "launch");
    float cg_ms = t1.stop() / ITERS;
    float cg_ts = 1000.f / cg_ms;
    float cg_s28 = cg_ts * 28.f / num_layers;

    // Save graph output
    std::vector<float> cg_out(H);
    blackwell::kernels::unpack_fp4(d_tmp, b.d_x_fp4, b.d_xs, H, gst);
    cudaMemcpy(cg_out.data(), d_tmp, H*4, cudaMemcpyDeviceToHost);

    // Correctness
    float maxdiff = 0;
    for (int i = 0; i < H; ++i) maxdiff = fmaxf(maxdiff, fabsf(pk_out[i]-cg_out[i]));

    // ── Print Results ───────────────────────────────────────────────────────
    float speedup = pk_ms / cg_ms;
    float graph_benefit = (pk_ms - cg_ms) / pk_ms * 100.f;

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  %-14s  %3d layers                           ║\n", MODEL_NAME, num_layers);
    printf("╠════════════╦═══════════════╦═══════════════╦════════════════╣\n");
    printf("║ Method     ║   ms/token    ║    t/s       ║   Scaled-28   ║\n");
    printf("╠════════════╬═══════════════╬═══════════════╬════════════════╣\n");
    printf("║ Per-kernel ║   %7.3f ms  ║  %7.1f t/s  ║  %7.1f t/s  ║\n", pk_ms, pk_ts, pk_s28);
    printf("║ CUDA Graph ║   %7.3f ms  ║  %7.1f t/s  ║  %7.1f t/s  ║\n", cg_ms, cg_ts, cg_s28);
    printf("╠════════════╩═══════════════╩═══════════════╩════════════════╣\n");
    printf("║ Speedup: %.2fx  |  Graph benefit: +%.1f%%                      ║\n", speedup, graph_benefit);
    printf("║ Correctness: max_diff=%.6f %s                            ║\n", maxdiff, maxdiff < 0.001f ? "✅" : "❌");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    // Machine-parseable summary
    printf("\n##RESULT model=%s layers=%d H=%d Q=%d I=%d nqh=%d nkv=%d pk=%.3fms=%.1fts cg=%.3fms=%.1fts speedup=%.2fx diff=%.6f\n",
        MODEL_NAME, num_layers, H, Q, I, nqh, nkv, pk_ms, pk_ts, cg_ms, cg_ts, speedup, maxdiff);

    cudaStreamDestroy(gst);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaFree(d_tmp);
    return 0;
}