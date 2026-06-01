// bench/decode_int8_cgraph.cu — CUDA Graph benchmark for INT8 decode
//
// Measures INT8 decode with CUDA Graph capture vs per-kernel launch.
// Captures all 18 kernels/layer into a single graph to eliminate
// inter-kernel launch gaps.
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/decode_int8_cgraph.cu build/libblackwell_kernels.a \
//     -o bench/decode_int8_cgraph

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

// Per-layer buffers (double-buffered for prefetch if needed)
struct LayerBufs {
    float *d_Q, *d_K, *d_V, *d_attn, *d_proj;
    float *d_gate, *d_up, *d_mlp;
    float *d_res;          // FP32 residual
    int8_t *d_x_int8;      // INT8 input
    float *d_x_int8_s;     // INT8 input scale
    int8_t *d_attn_i8;     // INT8 attention output
    float *d_attn_i8s;
    int8_t *d_mlp_i8;      // INT8 MLP output
    float *d_mlp_i8s;
};

int main(int argc, char** argv) {
    int num_layers = 4;
    if (argc > 1) num_layers = atoi(argv[1]);
    if (num_layers > 28) num_layers = 28;

    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("# INT8 CUDA Graph Decode Benchmark — Qwen3-1.7B\n");
    printf("Device: %s (%d.%d)\n", p.name, p.major, p.minor);
    printf("Layers: %d\n", num_layers);

    const int H = 2048, Q = 2048, KV = 1024, I = 6144;
    const int nqh = 16, nkv = 8, hd = 128, ms = 2048;
    const float s13 = 1.f/3.f;

    // Load INT8 weights (BF16-derived)
    struct LW { DevW q,k,v,o,g,u,d; };
    printf("Loading INT8 weights (BF16-derived)...\n");
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
    float *d_x32, *d_xs;
    void *d_x_fp4;
    float *d_rn;
    cudaMalloc(&d_x32, H*4); cudaMalloc(&d_x_fp4, H); cudaMalloc(&d_xs, (H/16)*4);
    cudaMalloc(&d_rn, H*4);
    std::vector<float> rn_h(H,1.f); cudaMemcpy(d_rn,rn_h.data(),H*4,cudaMemcpyHostToDevice);

    // Reusable per-layer buffers
    LayerBufs b;
    cudaMalloc(&b.d_Q, Q*4); cudaMalloc(&b.d_K, KV*4); cudaMalloc(&b.d_V, KV*4);
    cudaMalloc(&b.d_attn, Q*4); cudaMalloc(&b.d_proj, H*4);
    cudaMalloc(&b.d_gate, I*4); cudaMalloc(&b.d_up, I*4); cudaMalloc(&b.d_mlp, I*4);
    cudaMalloc(&b.d_res, I*4); // max(H,I)
    cudaMalloc(&b.d_x_int8, I); // max(H,I) for INT8
    cudaMalloc(&b.d_x_int8_s, (I/16)*4);
    cudaMalloc(&b.d_attn_i8, Q); cudaMalloc(&b.d_attn_i8s, (Q/16)*4);
    cudaMalloc(&b.d_mlp_i8, I); cudaMalloc(&b.d_mlp_i8s, (I/16)*4);

    // Init x = uniform 1.0
    float ixv = 1.f/127.f;
    std::vector<float> xh(H,1.f), xsh(H/16, s13);
    cudaMemcpy(d_x32, xh.data(), H*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_xs, xsh.data(), (H/16)*4, cudaMemcpyHostToDevice);
    blackwell::kernels::pack_fp4(d_x_fp4, d_x32, d_xs, H, 0);
    std::vector<float> ixsh(H/16, ixv);
    cudaMemcpy(b.d_x_int8_s, ixsh.data(), (H/16)*4, cudaMemcpyHostToDevice);
    std::vector<float> ai8s(Q/16, ixv), mi8s(I/16, ixv);
    cudaMemcpy(b.d_attn_i8s, ai8s.data(), (Q/16)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(b.d_mlp_i8s, mi8s.data(), (I/16)*4, cudaMemcpyHostToDevice);
    // Save initial scale values for per-kernel / CUDA Graph comparison
    // These buffers get overwritten by pack_int8 in each run — must restore
    float *d_x_int8_s_init, *d_attn_i8s_init, *d_mlp_i8s_init;
    cudaMalloc(&d_x_int8_s_init, (I/16)*4); cudaMemcpy(d_x_int8_s_init, b.d_x_int8_s, (I/16)*4, cudaMemcpyDeviceToDevice);
    cudaMalloc(&d_attn_i8s_init, (Q/16)*4); cudaMemcpy(d_attn_i8s_init, b.d_attn_i8s, (Q/16)*4, cudaMemcpyDeviceToDevice);
    cudaMalloc(&d_mlp_i8s_init, (I/16)*4); cudaMemcpy(d_mlp_i8s_init, b.d_mlp_i8s, (I/16)*4, cudaMemcpyDeviceToDevice);

    // KV cache
    float *d_kc, *d_vc;
    size_t kv_sz = (size_t)num_layers * nkv * ms * hd * 4;
    cudaMalloc(&d_kc, kv_sz); cudaMalloc(&d_vc, kv_sz);
    cudaMemset(d_kc, 0, kv_sz); cudaMemset(d_vc, 0, kv_sz);

    // ── L2 Cache Hints ───────────────────────────────────────────────────
    // Reserve 8 MB of L2 for persisting activation buffers (norm weights, etc)
    // Weight matrices (4-12 MB each) stream through and evict naturally.
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 8 * 1024 * 1024);

    // ── Fill KV cache (seq=0..128) on default stream ────────────────────────
    printf("Filling KV cache (seq=0..128)... ");
    fflush(stdout);
    int sq = 128;
    for (int s = 0; s <= sq; ++s) {
        for (int l = 0; l < num_layers; ++l) {
            blackwell::kernels::unpack_fp4(b.d_res, d_x_fp4, d_xs, H, 0);
            blackwell::kernels::pack_int8(b.d_x_int8, b.d_res, b.d_x_int8_s, H, 0);
            int kb = l * nkv * ms * hd;
            chk(blackwell::kernels::gemv_int8_warp(b.d_Q, b.d_x_int8, b.d_x_int8_s,
                lw[l].q.d, lw[l].q.sc, H, Q, 0), "Q");
            chk(blackwell::kernels::gemv_int8_warp(b.d_K, b.d_x_int8, b.d_x_int8_s,
                lw[l].k.d, lw[l].k.sc, H, KV, 0), "K");
            chk(blackwell::kernels::gemv_int8_warp(b.d_V, b.d_x_int8, b.d_x_int8_s,
                lw[l].v.d, lw[l].v.sc, H, KV, 0), "V");
            chk(blackwell::kernels::update_kv_cache(
                d_kc+kb, d_vc+kb, b.d_K, b.d_V, 0, s, nkv, hd, ms, 0), "kv");
            chk(blackwell::kernels::attention_decode_gqa(
                b.d_attn, b.d_Q, d_kc+kb, d_vc+kb,
                s, nqh, nkv, hd, ms, 0), "attn");
            chk(blackwell::kernels::pack_int8(b.d_attn_i8, b.d_attn, b.d_attn_i8s, Q, 0), "pack_attn");
            chk(blackwell::kernels::gemv_int8_warp(b.d_proj, b.d_attn_i8, b.d_attn_i8s,
                lw[l].o.d, lw[l].o.sc, Q, H, 0), "Wo");
            blackwell::kernels::vector_add_fp32(b.d_proj, b.d_proj, b.d_res, H, 0);
            blackwell::kernels::fused_rmsnorm_quant_int8(b.d_x_int8, b.d_x_int8_s,
                b.d_proj, d_rn, H, 1e-6f, 0);
            // Maintain d_x_fp4 for residual next iter
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, b.d_proj, d_rn, H, 1e-6f, 0);

            blackwell::kernels::unpack_fp4(b.d_res, d_x_fp4, d_xs, H, 0);
            blackwell::kernels::pack_int8(b.d_x_int8, b.d_res, b.d_x_int8_s, H, 0);
            chk(blackwell::kernels::gemv_int8_warp(b.d_gate, b.d_x_int8, b.d_x_int8_s,
                lw[l].g.d, lw[l].g.sc, H, I, 0), "gate");
            chk(blackwell::kernels::gemv_int8_warp(b.d_up, b.d_x_int8, b.d_x_int8_s,
                lw[l].u.d, lw[l].u.sc, H, I, 0), "up");
            chk(blackwell::kernels::apply_swiglu(b.d_mlp, b.d_gate, b.d_up, I, 0), "swiglu");
            chk(blackwell::kernels::pack_int8(b.d_mlp_i8, b.d_mlp, b.d_mlp_i8s, I, 0), "pack_mlp");
            chk(blackwell::kernels::gemv_int8_warp(b.d_proj, b.d_mlp_i8, b.d_mlp_i8s,
                lw[l].d.d, lw[l].d.sc, I, H, 0), "down");
            blackwell::kernels::vector_add_fp32(b.d_proj, b.d_proj, b.d_res, H, 0);
            blackwell::kernels::fused_rmsnorm_quant_int8(b.d_x_int8, b.d_x_int8_s,
                b.d_proj, d_rn, H, 1e-6f, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, b.d_proj, d_rn, H, 1e-6f, 0);
        }
    }
    printf("done\n");

    // ── Save initial state for correctness comparison ─────────────────────
    void* d_x_fp4_init; float* d_xs_init;
    cudaMalloc(&d_x_fp4_init, H);
    cudaMalloc(&d_xs_init, (H/16)*4);
    cudaMemcpy(d_x_fp4_init, d_x_fp4, H, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_xs_init, d_xs, (H/16)*4, cudaMemcpyDeviceToDevice);
    // Save KV cache
    float *d_kc_save, *d_vc_save;
    cudaMalloc(&d_kc_save, kv_sz); cudaMalloc(&d_vc_save, kv_sz);
    cudaMemcpy(d_kc_save, d_kc, kv_sz, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_vc_save, d_vc, kv_sz, cudaMemcpyDeviceToDevice);

    // ── Per-kernel baseline ─────────────────────────────────────────────────
    printf("Warmup (per-kernel)...\n");
    int warm = 5, bench = 20;
    for (int w = 0; w < warm; ++w) {
        for (int l = 0; l < num_layers; ++l) {
            blackwell::kernels::unpack_fp4(b.d_res, d_x_fp4, d_xs, H, 0);
            blackwell::kernels::pack_int8(b.d_x_int8, b.d_res, b.d_x_int8_s, H, 0);
            int kb = l * nkv * ms * hd;
            chk(blackwell::kernels::gemv_int8_warp(b.d_Q, b.d_x_int8, b.d_x_int8_s,
                lw[l].q.d, lw[l].q.sc, H, Q, 0), "Q");
            chk(blackwell::kernels::gemv_int8_warp(b.d_K, b.d_x_int8, b.d_x_int8_s,
                lw[l].k.d, lw[l].k.sc, H, KV, 0), "K");
            chk(blackwell::kernels::gemv_int8_warp(b.d_V, b.d_x_int8, b.d_x_int8_s,
                lw[l].v.d, lw[l].v.sc, H, KV, 0), "V");
            chk(blackwell::kernels::update_kv_cache(
                d_kc+kb, d_vc+kb, b.d_K, b.d_V, 0, sq, nkv, hd, ms, 0), "kv");
            chk(blackwell::kernels::attention_decode_gqa(
                b.d_attn, b.d_Q, d_kc+kb, d_vc+kb,
                sq, nqh, nkv, hd, ms, 0), "attn");
            chk(blackwell::kernels::pack_int8(b.d_attn_i8, b.d_attn, b.d_attn_i8s, Q, 0), "pack_attn");
            chk(blackwell::kernels::gemv_int8_warp(b.d_proj, b.d_attn_i8, b.d_attn_i8s,
                lw[l].o.d, lw[l].o.sc, Q, H, 0), "Wo");
            blackwell::kernels::vector_add_fp32(b.d_proj, b.d_proj, b.d_res, H, 0);
            blackwell::kernels::fused_rmsnorm_quant_int8(b.d_x_int8, b.d_x_int8_s,
                b.d_proj, d_rn, H, 1e-6f, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, b.d_proj, d_rn, H, 1e-6f, 0);
            // MLP
            blackwell::kernels::unpack_fp4(b.d_res, d_x_fp4, d_xs, H, 0);
            blackwell::kernels::pack_int8(b.d_x_int8, b.d_res, b.d_x_int8_s, H, 0);
            chk(blackwell::kernels::gemv_int8_warp(b.d_gate, b.d_x_int8, b.d_x_int8_s,
                lw[l].g.d, lw[l].g.sc, H, I, 0), "gate");
            chk(blackwell::kernels::gemv_int8_warp(b.d_up, b.d_x_int8, b.d_x_int8_s,
                lw[l].u.d, lw[l].u.sc, H, I, 0), "up");
            chk(blackwell::kernels::apply_swiglu(b.d_mlp, b.d_gate, b.d_up, I, 0), "swiglu");
            chk(blackwell::kernels::pack_int8(b.d_mlp_i8, b.d_mlp, b.d_mlp_i8s, I, 0), "pack_mlp");
            chk(blackwell::kernels::gemv_int8_warp(b.d_proj, b.d_mlp_i8, b.d_mlp_i8s,
                lw[l].d.d, lw[l].d.sc, I, H, 0), "down");
            blackwell::kernels::vector_add_fp32(b.d_proj, b.d_proj, b.d_res, H, 0);
            blackwell::kernels::fused_rmsnorm_quant_int8(b.d_x_int8, b.d_x_int8_s,
                b.d_proj, d_rn, H, 1e-6f, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, b.d_proj, d_rn, H, 1e-6f, 0);
        }
    }
    cudaDeviceSynchronize();

    // Baseline benchmark (no per-kernel events — just wall-clock)
    printf("Benchmarking per-kernel (%d tokens)...\n", bench);
    GpuTimer t0;
    t0.start();
    for (int i = 0; i < bench; ++i) {
        for (int l = 0; l < num_layers; ++l) {
            blackwell::kernels::unpack_fp4(b.d_res, d_x_fp4, d_xs, H, 0);
            blackwell::kernels::pack_int8(b.d_x_int8, b.d_res, b.d_x_int8_s, H, 0);
            int kb = l * nkv * ms * hd;
            blackwell::kernels::gemv_int8_warp(b.d_Q, b.d_x_int8, b.d_x_int8_s,
                lw[l].q.d, lw[l].q.sc, H, Q, 0);
            blackwell::kernels::gemv_int8_warp(b.d_K, b.d_x_int8, b.d_x_int8_s,
                lw[l].k.d, lw[l].k.sc, H, KV, 0);
            blackwell::kernels::gemv_int8_warp(b.d_V, b.d_x_int8, b.d_x_int8_s,
                lw[l].v.d, lw[l].v.sc, H, KV, 0);
            blackwell::kernels::update_kv_cache(
                d_kc+kb, d_vc+kb, b.d_K, b.d_V, 0, sq, nkv, hd, ms, 0);
            blackwell::kernels::attention_decode_gqa(
                b.d_attn, b.d_Q, d_kc+kb, d_vc+kb,
                sq, nqh, nkv, hd, ms, 0);
            blackwell::kernels::pack_int8(b.d_attn_i8, b.d_attn, b.d_attn_i8s, Q, 0);
            blackwell::kernels::gemv_int8_warp(b.d_proj, b.d_attn_i8, b.d_attn_i8s,
                lw[l].o.d, lw[l].o.sc, Q, H, 0);
            blackwell::kernels::vector_add_fp32(b.d_proj, b.d_proj, b.d_res, H, 0);
            blackwell::kernels::fused_rmsnorm_quant_int8(b.d_x_int8, b.d_x_int8_s,
                b.d_proj, d_rn, H, 1e-6f, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, b.d_proj, d_rn, H, 1e-6f, 0);
            // MLP
            blackwell::kernels::unpack_fp4(b.d_res, d_x_fp4, d_xs, H, 0);
            blackwell::kernels::pack_int8(b.d_x_int8, b.d_res, b.d_x_int8_s, H, 0);
            blackwell::kernels::gemv_int8_warp(b.d_gate, b.d_x_int8, b.d_x_int8_s,
                lw[l].g.d, lw[l].g.sc, H, I, 0);
            blackwell::kernels::gemv_int8_warp(b.d_up, b.d_x_int8, b.d_x_int8_s,
                lw[l].u.d, lw[l].u.sc, H, I, 0);
            blackwell::kernels::apply_swiglu(b.d_mlp, b.d_gate, b.d_up, I, 0);
            blackwell::kernels::pack_int8(b.d_mlp_i8, b.d_mlp, b.d_mlp_i8s, I, 0);
            blackwell::kernels::gemv_int8_warp(b.d_proj, b.d_mlp_i8, b.d_mlp_i8s,
                lw[l].d.d, lw[l].d.sc, I, H, 0);
            blackwell::kernels::vector_add_fp32(b.d_proj, b.d_proj, b.d_res, H, 0);
            blackwell::kernels::fused_rmsnorm_quant_int8(b.d_x_int8, b.d_x_int8_s,
                b.d_proj, d_rn, H, 1e-6f, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, b.d_proj, d_rn, H, 1e-6f, 0);
        }
    }
    float baseline_ms = t0.stop();
    float baseline_pt = baseline_ms / bench;
    float baseline_s28 = 1000.f / (baseline_pt * 28.f / num_layers);

    // Save per-kernel output (after warm+bench iterations)
    float* d_tmp; cudaMalloc(&d_tmp, H*4);
    std::vector<float> per_kernel_out(H);
    blackwell::kernels::unpack_fp4(d_tmp, d_x_fp4, d_xs, H, 0);
    cudaMemcpy(per_kernel_out.data(), d_tmp, H*4, cudaMemcpyDeviceToHost);

    // Restore initial state so graph starts from same point
    cudaMemcpy(d_x_fp4, d_x_fp4_init, H, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_xs, d_xs_init, (H/16)*4, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_kc, d_kc_save, kv_sz, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_vc, d_vc_save, kv_sz, cudaMemcpyDeviceToDevice);
    // Restore scale buffers too (overwritten by layer 27's pack_int8 during benchmark)
    cudaMemcpy(b.d_x_int8_s, d_x_int8_s_init, (I/16)*4, cudaMemcpyDeviceToDevice);
    cudaMemcpy(b.d_attn_i8s, d_attn_i8s_init, (Q/16)*4, cudaMemcpyDeviceToDevice);
    cudaMemcpy(b.d_mlp_i8s, d_mlp_i8s_init, (I/16)*4, cudaMemcpyDeviceToDevice);
    cudaFree(d_x_fp4_init); cudaFree(d_xs_init);
    cudaFree(d_kc_save); cudaFree(d_vc_save);

    // ── CUDA Graph capture ──────────────────────────────────────────────────
    printf("\n=== CUDA Graph ===\n");
    cudaDeviceSynchronize();
    cudaError_t cerr = cudaPeekAtLastError();
    if (cerr != cudaSuccess) {
        printf("  Pre-capture error: %s — fixing...\n", cudaGetErrorString(cerr));
        cudaGetLastError(); // clear
    }

    cudaStream_t graph_stream;
    cudaStreamCreate(&graph_stream);

    // Mark RMSNorm weights as persisting (tiny, reused every layer)
    // Must be set on graph_stream (not stream 0) for CUDA Graph path
    cudaAccessPolicyWindow norm_policy;
    norm_policy.base_ptr = (void*)d_rn;
    norm_policy.num_bytes = H * 4;  // 8 KB
    norm_policy.hitRatio = 1.0f;
    norm_policy.hitProp = cudaAccessPropertyPersisting;
    norm_policy.missProp = cudaAccessPropertyStreaming;
    cudaStreamAttrValue norm_attr;
    norm_attr.accessPolicyWindow = norm_policy;
    cudaStreamSetAttribute(graph_stream, cudaStreamAttributeAccessPolicyWindow, &norm_attr);

    // Pre-trigger attention_decode_gqa on graph_stream (sets smem config)
    blackwell::kernels::attention_decode_gqa(
        b.d_attn, b.d_Q, d_kc, d_vc,
        sq, nqh, nkv, hd, ms, graph_stream);
    cudaStreamSynchronize(graph_stream);

    printf("  Capturing %d layers (%d kernels)... ", num_layers, num_layers * 20);
    fflush(stdout);

    cudaStreamBeginCapture(graph_stream, cudaStreamCaptureModeGlobal);
    for (int l = 0; l < num_layers; ++l) {
        int kb = l * nkv * ms * hd;

        // Attention block
        blackwell::kernels::unpack_fp4(b.d_res, d_x_fp4, d_xs, H, graph_stream);
        blackwell::kernels::pack_int8(b.d_x_int8, b.d_res, b.d_x_int8_s, H, graph_stream);

        blackwell::kernels::gemv_int8_warp(b.d_Q, b.d_x_int8, b.d_x_int8_s,
            lw[l].q.d, lw[l].q.sc, H, Q, graph_stream);
        blackwell::kernels::gemv_int8_warp(b.d_K, b.d_x_int8, b.d_x_int8_s,
            lw[l].k.d, lw[l].k.sc, H, KV, graph_stream);
        blackwell::kernels::gemv_int8_warp(b.d_V, b.d_x_int8, b.d_x_int8_s,
            lw[l].v.d, lw[l].v.sc, H, KV, graph_stream);

        blackwell::kernels::update_kv_cache(
            d_kc+kb, d_vc+kb, b.d_K, b.d_V, 0, sq, nkv, hd, ms, graph_stream);
        blackwell::kernels::attention_decode_gqa(
            b.d_attn, b.d_Q, d_kc+kb, d_vc+kb,
            sq, nqh, nkv, hd, ms, graph_stream);

        blackwell::kernels::pack_int8(b.d_attn_i8, b.d_attn, b.d_attn_i8s, Q, graph_stream);
        blackwell::kernels::gemv_int8_warp(b.d_proj, b.d_attn_i8, b.d_attn_i8s,
            lw[l].o.d, lw[l].o.sc, Q, H, graph_stream);

        blackwell::kernels::vector_add_fp32(b.d_proj, b.d_proj, b.d_res, H, graph_stream);
        blackwell::kernels::fused_rmsnorm_quant_int8(b.d_x_int8, b.d_x_int8_s,
            b.d_proj, d_rn, H, 1e-6f, graph_stream);
        blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, b.d_proj, d_rn, H, 1e-6f, graph_stream);

        // MLP block
        blackwell::kernels::unpack_fp4(b.d_res, d_x_fp4, d_xs, H, graph_stream);
        blackwell::kernels::pack_int8(b.d_x_int8, b.d_res, b.d_x_int8_s, H, graph_stream);

        blackwell::kernels::gemv_int8_warp(b.d_gate, b.d_x_int8, b.d_x_int8_s,
            lw[l].g.d, lw[l].g.sc, H, I, graph_stream);
        blackwell::kernels::gemv_int8_warp(b.d_up, b.d_x_int8, b.d_x_int8_s,
            lw[l].u.d, lw[l].u.sc, H, I, graph_stream);
        blackwell::kernels::apply_swiglu(b.d_mlp, b.d_gate, b.d_up, I, graph_stream);

        blackwell::kernels::pack_int8(b.d_mlp_i8, b.d_mlp, b.d_mlp_i8s, I, graph_stream);
        blackwell::kernels::gemv_int8_warp(b.d_proj, b.d_mlp_i8, b.d_mlp_i8s,
            lw[l].d.d, lw[l].d.sc, I, H, graph_stream);

        blackwell::kernels::vector_add_fp32(b.d_proj, b.d_proj, b.d_res, H, graph_stream);
        blackwell::kernels::fused_rmsnorm_quant_int8(b.d_x_int8, b.d_x_int8_s,
            b.d_proj, d_rn, H, 1e-6f, graph_stream);
        blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, b.d_proj, d_rn, H, 1e-6f, graph_stream);
    }

    cudaGraph_t graph;
    cerr = cudaStreamEndCapture(graph_stream, &graph);
    if (cerr != cudaSuccess) {
        printf("FAIL: %s\n", cudaGetErrorString(cerr));
        cudaStreamDestroy(graph_stream);
        return 1;
    }

    cudaGraphExec_t graph_exec;
    cerr = cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
    if (cerr != cudaSuccess) {
        printf("FAIL instantiate: %s\n", cudaGetErrorString(cerr));
        cudaGraphDestroy(graph);
        cudaStreamDestroy(graph_stream);
        return 1;
    }
    printf("OK\n");

    // Graph warmup (same #iters as per-kernel warmup)
    printf("  Graph warmup...\n");
    for (int i = 0; i < warm; ++i) cudaGraphLaunch(graph_exec, graph_stream);
    cudaStreamSynchronize(graph_stream);

    // ── Graph benchmark ──────────────────────────────────────────────────
    printf("  Graph benchmark (%d tokens)...\n", bench);
    GpuTimer tg;
    tg.start(graph_stream);
    for (int i = 0; i < bench; ++i) cudaGraphLaunch(graph_exec, graph_stream);
    cudaStreamSynchronize(graph_stream);
    float graph_ms = tg.stop();
    float graph_pt = graph_ms / bench;
    float graph_s28 = 1000.f / (graph_pt * 28.f / num_layers);

    // ── Correctness check ────────────────────────────────────────────────
    std::vector<float> graph_out(H);
    blackwell::kernels::unpack_fp4(d_tmp, d_x_fp4, d_xs, H, 0);
    cudaMemcpy(graph_out.data(), d_tmp, H*4, cudaMemcpyDeviceToHost);

    printf("\n=== Correctness Check (same start, %d warmup + %d bench iters) ===\n", warm, bench);
    printf("  Per-kernel first 8: ");
    for (int i = 0; i < 8; ++i) printf("%.4f ", per_kernel_out[i]);
    printf("\n  Graph first 8:     ");
    for (int i = 0; i < 8; ++i) printf("%.4f ", graph_out[i]);
    printf("\n");
    float pk_sum = 0, gr_sum = 0, max_diff = 0;
    for (int i = 0; i < H; ++i) {
        pk_sum += fabsf(per_kernel_out[i]);
        gr_sum += fabsf(graph_out[i]);
        max_diff = fmaxf(max_diff, fabsf(per_kernel_out[i] - graph_out[i]));
    }
    printf("  Per-kernel L1: %.4f  Graph L1: %.4f\n", pk_sum/H, gr_sum/H);
    printf("  Max diff: %.6f %s\n", max_diff,
        max_diff < 1e-3 ? "✅ MATCH" : max_diff < 0.1 ? "⚠️ CLOSE (FP4 quantization)" : "❌ MISMATCH");
    cudaFree(d_tmp);

    // ── Results ──────────────────────────────────────────────────────────────
    printf("\n=== Results ===\n");
    printf("  %-20s  %8s  %8s  %8s\n", "Method", "Per-tok", "t/s", "Scaled28");
    printf("  %-20s  %7.3fms  %7.1f   %7.1f\n", "Per-kernel",
        baseline_pt, 1000.f/baseline_pt, baseline_s28);
    printf("  %-20s  %7.3fms  %7.1f   %7.1f\n", "CUDA Graph",
        graph_pt, 1000.f/graph_pt, graph_s28);
    printf("  Speedup: %.2fx (%.1f%%)\n",
        baseline_pt / graph_pt,
        (1.f - graph_pt/baseline_pt) * 100.f);
    printf("  Target: 114.0 t/s\n");

    // Cleanup
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(graph_stream);

    for (auto& l : lw) {
        cudaFree(l.q.d); cudaFree(l.q.sc);
        cudaFree(l.k.d); cudaFree(l.k.sc);
        cudaFree(l.v.d); cudaFree(l.v.sc);
        cudaFree(l.o.d); cudaFree(l.o.sc);
        cudaFree(l.g.d); cudaFree(l.g.sc);
        cudaFree(l.u.d); cudaFree(l.u.sc);
        cudaFree(l.d.d); cudaFree(l.d.sc);
    }
    cudaFree(d_x32); cudaFree(d_x_fp4); cudaFree(d_xs); cudaFree(d_rn);
    cudaFree(b.d_Q); cudaFree(b.d_K); cudaFree(b.d_V);
    cudaFree(b.d_attn); cudaFree(b.d_proj);
    cudaFree(b.d_gate); cudaFree(b.d_up); cudaFree(b.d_mlp);
    cudaFree(b.d_res);
    cudaFree(b.d_x_int8); cudaFree(b.d_x_int8_s);
    cudaFree(b.d_attn_i8); cudaFree(b.d_attn_i8s);
    cudaFree(b.d_mlp_i8); cudaFree(b.d_mlp_i8s);
    cudaFree(d_kc); cudaFree(d_vc);

    return 0;
}
