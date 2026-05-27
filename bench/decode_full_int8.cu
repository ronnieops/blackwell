// bench/decode_full_int8.cu — Full decode with INT8 GEMV (BF16-derived weights)
//
// Same pipeline as decode_full.cu but replaces gemv_fp4_v2 calls with gemv_int8.
// Also replaces gemv_fp4_splitk (down_proj) with gemv_int8.
//
// INT8 input conversion: x_fp4 → unpack_fp4 → pack_int8 (once per layer)
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120,code=sm_120 \
//     -I include bench/decode_full_int8.cu build/libblackwell_kernels.a \
//     -o bench/decode_full_int8

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include <cstdint>
#include "blackwell/kernels.h"

// ── Timer ──────────────────────────────────────────────────────────────────
struct GpuTimer {
    cudaEvent_t s, e;
    GpuTimer() { cudaEventCreate(&s); cudaEventCreate(&e); }
    ~GpuTimer() { cudaEventDestroy(s); cudaEventDestroy(e); }
    void start() { cudaEventRecord(s); }
    float stop() { cudaEventRecord(e); cudaEventSynchronize(e);
                   float ms=0; cudaEventElapsedTime(&ms, s, e); return ms; }
};
struct KernelTimer {
    const char* name; double tot = 0; int n = 0;
    KernelTimer(const char* n) : name(n) {}
    void add(float ms) { tot += ms; ++n; }
    void print() const { printf("  %-30s  %7.3f ms (%d)\n", name, tot, n); }
};

// ── File I/O ───────────────────────────────────────────────────────────────
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

// ── Main ───────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    int num_layers = 4;
    if (argc > 1) num_layers = atoi(argv[1]);
    if (num_layers > 28) num_layers = 28;

    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("# INT8 Full Decode Benchmark — Qwen3-1.7B\n");
    printf("Device: %s (%d.%d)\n", p.name, p.major, p.minor);
    printf("Layers: %d\n", num_layers);

    const int H = 2048, Q = 2048, KV = 1024, I = 6144;
    const int nqh = 16, nkv = 8, hd = 128, ms = 2048;
    const float s13 = 1.f/3.f;

    // Load INT8 weights (BF16-derived)
    struct LW { DevW q,k,v,o,g,u,d; };
    auto lfmt = [](char* buf, const char* fmt, int l, const char* name) {
        snprintf(buf,256,"weights_int8_bf16/%s_int8.%s", fmt, name);
        // Actually the format is weights_int8_bf16/0_self_attn.q_proj
        snprintf(buf,256,"weights_int8_bf16/%d_%s", l, name);
    };

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
    float *d_x32, *d_xs, *d_Q, *d_K, *d_V, *d_attn, *d_proj;
    float *d_gate, *d_up, *d_mlp;
    float *d_res;          // FP32 residual (reused for attn and MLP)
    void *d_x_fp4;         // FP4 input to kernels
    float *d_attn_s, *d_mlp_s;
    void *d_attn_fp4, *d_mlp_fp4;
    int8_t *d_x_int8;      // INT8 input to gemv_int8
    float *d_x_int8_s;

    cudaMalloc(&d_x32, H*4); cudaMalloc(&d_x_fp4, H);  cudaMalloc(&d_xs, (H/16)*4);
    cudaMalloc(&d_Q, Q*4); cudaMalloc(&d_K, KV*4); cudaMalloc(&d_V, KV*4);
    cudaMalloc(&d_attn, Q*4); cudaMalloc(&d_proj, H*4);
    cudaMalloc(&d_gate, I*4); cudaMalloc(&d_up, I*4); cudaMalloc(&d_mlp, I*4);
    cudaMalloc(&d_res, I*4); // sized for max(H,I) = max(2048,6144)
    cudaMalloc(&d_attn_fp4, Q); cudaMalloc(&d_attn_s, (Q/16)*4);
    cudaMalloc(&d_mlp_fp4, I); cudaMalloc(&d_mlp_s, (I/16)*4);
    cudaMalloc(&d_x_int8, H); cudaMalloc(&d_x_int8_s, (H/16)*4);

    float *d_rn; cudaMalloc(&d_rn, H*4);
    std::vector<float> rn_h(H,1.f); cudaMemcpy(d_rn,rn_h.data(),H*4,cudaMemcpyHostToDevice);

    // KV cache
    float *d_kc, *d_vc;
    size_t kv_sz = (size_t)num_layers * nkv * ms * hd * 4;
    cudaMalloc(&d_kc, kv_sz); cudaMalloc(&d_vc, kv_sz);
    cudaMemset(d_kc, 0, kv_sz); cudaMemset(d_vc, 0, kv_sz);

    // Init x = uniform 1.0
    std::vector<float> xh(H, 1.f), xsh(H/16, s13);
    cudaMemcpy(d_x32, xh.data(), H*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_xs, xsh.data(), (H/16)*4, cudaMemcpyHostToDevice);
    blackwell::kernels::pack_fp4(d_x_fp4, d_x32, d_xs, H, 0);
    std::vector<float> as(Q/16,s13), ms2(I/16,s13);
    cudaMemcpy(d_attn_s, as.data(), (Q/16)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mlp_s, ms2.data(), (I/16)*4, cudaMemcpyHostToDevice);

    // Pre-quantize x to INT8 for GEMV input
    float ixv = 1.f/127.f;
    std::vector<float> ixsh(H/16, ixv);
    cudaMemcpy(d_x_int8_s, ixsh.data(), (H/16)*4, cudaMemcpyHostToDevice);
    blackwell::kernels::pack_int8(d_x_int8, d_x32, d_x_int8_s, H, 0);

    // ── Fill KV cache (seq=0..128) ──────────────────────────────────────────
    printf("Filling KV cache (seq=0..128)... ");
    fflush(stdout);
    for (int sq = 0; sq <= 128; ++sq) {
        for (int l = 0; l < num_layers; ++l) {
            // Convert FP4 x → INT8 x for GEMV
            blackwell::kernels::unpack_fp4(d_res, d_x_fp4, d_xs, H, 0);
            blackwell::kernels::pack_int8(d_x_int8, d_res, d_x_int8_s, H, 0);

            int kb = l * nkv * ms * hd;
            chk(blackwell::kernels::gemv_int8(d_Q, d_x_int8, d_x_int8_s,
                lw[l].q.d, lw[l].q.sc, H, Q, 0), "Q");
            chk(blackwell::kernels::gemv_int8(d_K, d_x_int8, d_x_int8_s,
                lw[l].k.d, lw[l].k.sc, H, KV, 0), "K");
            chk(blackwell::kernels::gemv_int8(d_V, d_x_int8, d_x_int8_s,
                lw[l].v.d, lw[l].v.sc, H, KV, 0), "V");
            chk(blackwell::kernels::update_kv_cache(
                d_kc+kb, d_vc+kb, d_K, d_V, 0, sq, nkv, hd, ms, 0), "kv_cache");
            chk(blackwell::kernels::attention_decode_gqa(
                d_attn, d_Q, d_kc+kb, d_vc+kb,
                sq, nqh, nkv, hd, ms, 0), "attn");
            chk(blackwell::kernels::pack_fp4(d_attn_fp4, d_attn, d_attn_s, Q, 0), "pack_attn");

            // Wo: attn_fp4 → FP32 → INT8 for gemv_int8
            chk(blackwell::kernels::unpack_fp4(d_proj, d_attn_fp4, d_attn_s, Q, 0), "unpack_attn");
            // Use d_mlp as temp INT8 buffer (reuse — not used yet)
            // Actually need separate INT8 buffer for attn → Wo
            // Reuse d_mlp_fp4 as INT8 buffer (different size but Wo output is H=2048)
            // Better: create a small INT8 buffer
            // For now: quantize attn output to INT8 using temp buffer
            // d_res is not needed yet — reuse as FP32 attn output
            // d_res already has attn FP32 from unpack. Pack to INT8 in-place:
            // Use d_gate as temp INT8 (H=2048)
            int8_t* d_attn_int8 = reinterpret_cast<int8_t*>(d_mlp_fp4); // Q=2048 bytes
            float* d_attn_int8_s = d_mlp_s; // Q/16 floats
            std::vector<float> ai8s(Q/16, ixv);
            cudaMemcpy(d_attn_int8_s, ai8s.data(), (Q/16)*4, cudaMemcpyHostToDevice);
            chk(blackwell::kernels::pack_int8(d_attn_int8, d_proj, d_attn_int8_s, Q, 0), "pack_attn_i8");

            chk(blackwell::kernels::gemv_int8(d_proj, d_attn_int8, d_attn_int8_s,
                lw[l].o.d, lw[l].o.sc, Q, H, 0), "Wo");
            // Resid: proj += x_residual (saved in d_res — keep original x_residual)
            // d_res holds x FP32 from earlier unpack. We need to keep it.
            // Actually we overwrote d_res with unpack_fp4... but that's x.
            // The residual is the unpacked x. Use it.
            // After Wo: d_proj += d_res (x residual from before attention)
            blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_res, H, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, d_proj, d_rn, H, 1e-5f, 0);

            // MLP: x_fp4 → INT8 x
            blackwell::kernels::unpack_fp4(d_res, d_x_fp4, d_xs, H, 0);
            blackwell::kernels::pack_int8(d_x_int8, d_res, d_x_int8_s, H, 0);

            chk(blackwell::kernels::gemv_int8(d_gate, d_x_int8, d_x_int8_s,
                lw[l].g.d, lw[l].g.sc, H, I, 0), "gate");
            chk(blackwell::kernels::gemv_int8(d_up, d_x_int8, d_x_int8_s,
                lw[l].u.d, lw[l].u.sc, H, I, 0), "up");
            chk(blackwell::kernels::apply_swiglu(d_mlp, d_gate, d_up, I, 0), "swiglu");
            chk(blackwell::kernels::pack_fp4(d_mlp_fp4, d_mlp, d_mlp_s, I, 0), "pack_mlp");

            // Down: mlp_fp4 → FP32 → INT8
            chk(blackwell::kernels::unpack_fp4(d_proj, d_mlp_fp4, d_mlp_s, I, 0), "unpack_mlp");
            // INT8 buffer for MLP: reuse d_gate (I=6144 bytes, enough)
            int8_t* d_mlp_int8 = reinterpret_cast<int8_t*>(d_gate);
            float* d_mlp_int8_s = d_gate + I; // wrong — d_gate is float*
            // Actually just use separate allocation
            int8_t *d_mlp_i8_tmp;
            float *d_mlp_i8s_tmp;
            if (sq == 0 && l == 0) {
                // Lazy allocate once
                cudaMalloc(&d_mlp_i8_tmp, I);
                cudaMalloc(&d_mlp_i8s_tmp, (I/16)*4);
                std::vector<float> mi8s(I/16, ixv);
                cudaMemcpy(d_mlp_i8s_tmp, mi8s.data(), (I/16)*4, cudaMemcpyHostToDevice);
            }
            // Actually for quick fill, just use separate buffers
            // Already allocated at top
        }
    }

    // ── Proper allocation ───────────────────────────────────────────────────
    int8_t *d_attn_i8, *d_mlp_i8;
    float *d_attn_i8s, *d_mlp_i8s;
    cudaMalloc(&d_attn_i8, Q); cudaMalloc(&d_attn_i8s, (Q/16)*4);
    cudaMalloc(&d_mlp_i8, I); cudaMalloc(&d_mlp_i8s, (I/16)*4);
    std::vector<float> ai8s(Q/16, ixv), mi8s(I/16, ixv);
    cudaMemcpy(d_attn_i8s, ai8s.data(), (Q/16)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mlp_i8s, mi8s.data(), (I/16)*4, cudaMemcpyHostToDevice);

    // Redo cache fill properly
    cudaMemset(d_kc, 0, kv_sz); cudaMemset(d_vc, 0, kv_sz);
    printf("done\nFilling KV cache (retry)... ");
    fflush(stdout);
    for (int sq = 0; sq <= 128; ++sq) {
        for (int l = 0; l < num_layers; ++l) {
            blackwell::kernels::unpack_fp4(d_res, d_x_fp4, d_xs, H, 0);
            blackwell::kernels::pack_int8(d_x_int8, d_res, d_x_int8_s, H, 0);
            int kb = l * nkv * ms * hd;
            chk(blackwell::kernels::gemv_int8(d_Q, d_x_int8, d_x_int8_s,
                lw[l].q.d, lw[l].q.sc, H, Q, 0), "Q");
            chk(blackwell::kernels::gemv_int8(d_K, d_x_int8, d_x_int8_s,
                lw[l].k.d, lw[l].k.sc, H, KV, 0), "K");
            chk(blackwell::kernels::gemv_int8(d_V, d_x_int8, d_x_int8_s,
                lw[l].v.d, lw[l].v.sc, H, KV, 0), "V");
            chk(blackwell::kernels::update_kv_cache(
                d_kc+kb, d_vc+kb, d_K, d_V, 0, sq, nkv, hd, ms, 0), "kv");
            chk(blackwell::kernels::attention_decode_gqa(
                d_attn, d_Q, d_kc+kb, d_vc+kb,
                sq, nqh, nkv, hd, ms, 0), "attn");
            chk(blackwell::kernels::pack_fp4(d_attn_fp4, d_attn, d_attn_s, Q, 0), "pack_a");
            chk(blackwell::kernels::unpack_fp4(d_proj, d_attn_fp4, d_attn_s, Q, 0), "unpack_a");
            chk(blackwell::kernels::pack_int8(d_attn_i8, d_proj, d_attn_i8s, Q, 0), "pack_ai8");
            chk(blackwell::kernels::gemv_int8(d_proj, d_attn_i8, d_attn_i8s,
                lw[l].o.d, lw[l].o.sc, Q, H, 0), "Wo");
            blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_res, H, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, d_proj, d_rn, H, 1e-5f, 0);
            // MLP
            blackwell::kernels::unpack_fp4(d_res, d_x_fp4, d_xs, H, 0);
            blackwell::kernels::pack_int8(d_x_int8, d_res, d_x_int8_s, H, 0);
            chk(blackwell::kernels::gemv_int8(d_gate, d_x_int8, d_x_int8_s,
                lw[l].g.d, lw[l].g.sc, H, I, 0), "gate");
            chk(blackwell::kernels::gemv_int8(d_up, d_x_int8, d_x_int8_s,
                lw[l].u.d, lw[l].u.sc, H, I, 0), "up");
            chk(blackwell::kernels::apply_swiglu(d_mlp, d_gate, d_up, I, 0), "swiglu");
            chk(blackwell::kernels::pack_fp4(d_mlp_fp4, d_mlp, d_mlp_s, I, 0), "pack_m");
            chk(blackwell::kernels::unpack_fp4(d_proj, d_mlp_fp4, d_mlp_s, I, 0), "unpack_m");
            chk(blackwell::kernels::pack_int8(d_mlp_i8, d_proj, d_mlp_i8s, I, 0), "pack_mi8");
            chk(blackwell::kernels::gemv_int8(d_proj, d_mlp_i8, d_mlp_i8s,
                lw[l].d.d, lw[l].d.sc, I, H, 0), "down");
            blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_res, H, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, d_proj, d_rn, H, 1e-5f, 0);
        }
    }
    printf("done\n");

    // ── Warmup ──────────────────────────────────────────────────────────────
    printf("Warmup...\n");
    int sq = 128, warm = 5, bench = 20;
    for (int w = 0; w < warm; ++w) {
        for (int l = 0; l < num_layers; ++l) {
            blackwell::kernels::unpack_fp4(d_res, d_x_fp4, d_xs, H, 0);
            blackwell::kernels::pack_int8(d_x_int8, d_res, d_x_int8_s, H, 0);
            int kb = l * nkv * ms * hd;
            chk(blackwell::kernels::gemv_int8(d_Q, d_x_int8, d_x_int8_s,
                lw[l].q.d, lw[l].q.sc, H, Q, 0), "Q");
            chk(blackwell::kernels::gemv_int8(d_K, d_x_int8, d_x_int8_s,
                lw[l].k.d, lw[l].k.sc, H, KV, 0), "K");
            chk(blackwell::kernels::gemv_int8(d_V, d_x_int8, d_x_int8_s,
                lw[l].v.d, lw[l].v.sc, H, KV, 0), "V");
            chk(blackwell::kernels::update_kv_cache(
                d_kc+kb, d_vc+kb, d_K, d_V, 0, sq, nkv, hd, ms, 0), "kv");
            chk(blackwell::kernels::attention_decode_gqa(
                d_attn, d_Q, d_kc+kb, d_vc+kb,
                sq, nqh, nkv, hd, ms, 0), "attn");
            chk(blackwell::kernels::pack_fp4(d_attn_fp4, d_attn, d_attn_s, Q, 0), "pack_a");
            chk(blackwell::kernels::unpack_fp4(d_proj, d_attn_fp4, d_attn_s, Q, 0), "unpack_a");
            chk(blackwell::kernels::pack_int8(d_attn_i8, d_proj, d_attn_i8s, Q, 0), "pack_ai8");
            chk(blackwell::kernels::gemv_int8(d_proj, d_attn_i8, d_attn_i8s,
                lw[l].o.d, lw[l].o.sc, Q, H, 0), "Wo");
            blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_res, H, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, d_proj, d_rn, H, 1e-5f, 0);
            // MLP
            blackwell::kernels::unpack_fp4(d_res, d_x_fp4, d_xs, H, 0);
            blackwell::kernels::pack_int8(d_x_int8, d_res, d_x_int8_s, H, 0);
            chk(blackwell::kernels::gemv_int8(d_gate, d_x_int8, d_x_int8_s,
                lw[l].g.d, lw[l].g.sc, H, I, 0), "gate");
            chk(blackwell::kernels::gemv_int8(d_up, d_x_int8, d_x_int8_s,
                lw[l].u.d, lw[l].u.sc, H, I, 0), "up");
            chk(blackwell::kernels::apply_swiglu(d_mlp, d_gate, d_up, I, 0), "swiglu");
            chk(blackwell::kernels::pack_fp4(d_mlp_fp4, d_mlp, d_mlp_s, I, 0), "pack_m");
            chk(blackwell::kernels::unpack_fp4(d_proj, d_mlp_fp4, d_mlp_s, I, 0), "unpack_m");
            chk(blackwell::kernels::pack_int8(d_mlp_i8, d_proj, d_mlp_i8s, I, 0), "pack_mi8");
            chk(blackwell::kernels::gemv_int8(d_proj, d_mlp_i8, d_mlp_i8s,
                lw[l].d.d, lw[l].d.sc, I, H, 0), "down");
            blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_res, H, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, d_proj, d_rn, H, 1e-5f, 0);
        }
    }
    cudaDeviceSynchronize();

    // ── Benchmark ───────────────────────────────────────────────────────────
    KernelTimer timers[] = {
        {"pack_int8(x) + gemv Q"}, {"gemv K"}, {"gemv V"},
        {"update_kv_cache"}, {"attention_decode_gqa"},
        {"pack_fp4(attn) + i8_convert + gemv Wo"}, {"fused_rmsnorm_pack(attn)"},
        {"gemv gate"}, {"gemv up"}, {"apply_swiglu"},
        {"pack_fp4(mlp) + i8_convert + gemv down"}, {"fused_rmsnorm_pack(mlp)"}
    };
    GpuTimer kt[12];

    printf("Benchmarking %d tokens...\n", bench);
    GpuTimer t; t.start();
    for (int i = 0; i < bench; ++i) {
        for (int l = 0; l < num_layers; ++l) {
            GpuTimer loop_t;
            // Save x residual
            blackwell::kernels::unpack_fp4(d_res, d_x_fp4, d_xs, H, 0);
            blackwell::kernels::pack_int8(d_x_int8, d_res, d_x_int8_s, H, 0);

            int kb = l * nkv * ms * hd;
            loop_t.start();
            blackwell::kernels::gemv_int8(d_Q, d_x_int8, d_x_int8_s,
                lw[l].q.d, lw[l].q.sc, H, Q, 0);
            timers[0].add(loop_t.stop());
            loop_t.start();
            blackwell::kernels::gemv_int8(d_K, d_x_int8, d_x_int8_s,
                lw[l].k.d, lw[l].k.sc, H, KV, 0);
            timers[1].add(loop_t.stop());
            loop_t.start();
            blackwell::kernels::gemv_int8(d_V, d_x_int8, d_x_int8_s,
                lw[l].v.d, lw[l].v.sc, H, KV, 0);
            timers[2].add(loop_t.stop());

            loop_t.start();
            blackwell::kernels::update_kv_cache(
                d_kc+kb, d_vc+kb, d_K, d_V, 0, sq, nkv, hd, ms, 0);
            timers[3].add(loop_t.stop());
            loop_t.start();
            blackwell::kernels::attention_decode_gqa(
                d_attn, d_Q, d_kc+kb, d_vc+kb,
                sq, nqh, nkv, hd, ms, 0);
            timers[4].add(loop_t.stop());

            // Attn_out → Wo
            blackwell::kernels::pack_fp4(d_attn_fp4, d_attn, d_attn_s, Q, 0);
            blackwell::kernels::unpack_fp4(d_proj, d_attn_fp4, d_attn_s, Q, 0);
            blackwell::kernels::pack_int8(d_attn_i8, d_proj, d_attn_i8s, Q, 0);
            loop_t.start();
            blackwell::kernels::gemv_int8(d_proj, d_attn_i8, d_attn_i8s,
                lw[l].o.d, lw[l].o.sc, Q, H, 0);
            timers[5].add(loop_t.stop());

            blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_res, H, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, d_proj, d_rn, H, 1e-5f, 0);

            // MLP
            blackwell::kernels::unpack_fp4(d_res, d_x_fp4, d_xs, H, 0);
            blackwell::kernels::pack_int8(d_x_int8, d_res, d_x_int8_s, H, 0);

            loop_t.start();
            blackwell::kernels::gemv_int8(d_gate, d_x_int8, d_x_int8_s,
                lw[l].g.d, lw[l].g.sc, H, I, 0);
            timers[7].add(loop_t.stop());
            loop_t.start();
            blackwell::kernels::gemv_int8(d_up, d_x_int8, d_x_int8_s,
                lw[l].u.d, lw[l].u.sc, H, I, 0);
            timers[8].add(loop_t.stop());

            loop_t.start();
            blackwell::kernels::apply_swiglu(d_mlp, d_gate, d_up, I, 0);
            timers[9].add(loop_t.stop());

            blackwell::kernels::pack_fp4(d_mlp_fp4, d_mlp, d_mlp_s, I, 0);
            blackwell::kernels::unpack_fp4(d_proj, d_mlp_fp4, d_mlp_s, I, 0);
            blackwell::kernels::pack_int8(d_mlp_i8, d_proj, d_mlp_i8s, I, 0);
            loop_t.start();
            blackwell::kernels::gemv_int8(d_proj, d_mlp_i8, d_mlp_i8s,
                lw[l].d.d, lw[l].d.sc, I, H, 0);
            timers[10].add(loop_t.stop());

            blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_res, H, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, d_proj, d_rn, H, 1e-5f, 0);
        }
    }
    float total_ms = t.stop();
    float ptm = total_ms / bench;
    float tps = 1000.f / ptm;
    float s28 = 1000.f / (ptm * 28.f / num_layers);

    printf("\n=== INT8 Decode Results ===\n");
    printf("  Layers:         %d\n", num_layers);
    printf("  Tokens:         %d\n", bench);
    printf("  Total:          %.2f ms\n", total_ms);
    printf("  Per-token:      %.3f ms\n", ptm);
    printf("  Throughput:     %.1f t/s\n", tps);
    printf("  Scaled 28:      %.1f t/s\n", s28);

    printf("\n=== Per-Kernel Timing ===\n");
    double kt_total = 0;
    for (int i = 0; i < 12; ++i) { timers[i].print(); kt_total += timers[i].tot; }
    printf("  %-30s  %7.3f ms\n", "TOTAL KERNEL", kt_total);

    // ── Cleanup ─────────────────────────────────────────────────────────────
    for (auto& l : lw) {
        cudaFree(l.q.d); cudaFree(l.q.sc);
        cudaFree(l.k.d); cudaFree(l.k.sc);
        cudaFree(l.v.d); cudaFree(l.v.sc);
        cudaFree(l.o.d); cudaFree(l.o.sc);
        cudaFree(l.g.d); cudaFree(l.g.sc);
        cudaFree(l.u.d); cudaFree(l.u.sc);
        cudaFree(l.d.d); cudaFree(l.d.sc);
    }
    cudaFree(d_x32); cudaFree(d_x_fp4); cudaFree(d_xs);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_attn); cudaFree(d_proj);
    cudaFree(d_gate); cudaFree(d_up); cudaFree(d_mlp);
    cudaFree(d_res); cudaFree(d_rn);
    cudaFree(d_attn_fp4); cudaFree(d_attn_s);
    cudaFree(d_mlp_fp4); cudaFree(d_mlp_s);
    cudaFree(d_x_int8); cudaFree(d_x_int8_s);
    cudaFree(d_attn_i8); cudaFree(d_attn_i8s);
    cudaFree(d_mlp_i8); cudaFree(d_mlp_i8s);
    cudaFree(d_kc); cudaFree(d_vc);
    return 0;
}