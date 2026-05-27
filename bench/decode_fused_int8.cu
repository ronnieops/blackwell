// bench/decode_fused_int8.cu — Full decode with gemv_int8_from_fp4 (fused)
//
// Replaces decode_full_int8.cu: instead of unpack_fp4 + pack_int8 + gemv_int8,
// uses gemv_int8_from_fp4 which converts FP4 input to INT8 inline in one kernel.
// Saves 2 kernel launches per GEMV call.
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120,code=sm_120 \
//     -I include bench/decode_fused_int8.cu build/libblackwell_kernels.a \
//     -o bench/decode_fused_int8

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
    if (argc > 1) num_layers = atoi(argv[1]);
    if (num_layers > 28) num_layers = 28;

    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("# Fused INT8 Decode Benchmark — gemv_int8_from_fp4\n");
    printf("Device: %s (%d.%d)\n", p.name, p.major, p.minor);
    printf("Layers: %d\n", num_layers);

    const int H = 2048, Q = 2048, KV = 1024, I = 6144;
    const int nqh = 16, nkv = 8, hd = 128, ms = 2048;
    const float s13 = 1.f/3.f;

    // Load INT8 weights (BF16-derived, transposed)
    printf("Loading INT8 weights (BF16-derived)...\n");
    std::vector<DevW> qw(num_layers), kw(num_layers), vw(num_layers);
    std::vector<DevW> ow(num_layers), gw(num_layers), uw(num_layers), dw(num_layers);

    for (int l = 0; l < num_layers; ++l) {
        char p[256];
        snprintf(p,256,"weights_int8_bf16/%d_self_attn.q_proj",l); qw[l] = upload(p);
        snprintf(p,256,"weights_int8_bf16/%d_self_attn.k_proj",l); kw[l] = upload(p);
        snprintf(p,256,"weights_int8_bf16/%d_self_attn.v_proj",l); vw[l] = upload(p);
        snprintf(p,256,"weights_int8_bf16/%d_self_attn.o_proj",l); ow[l] = upload(p);
        snprintf(p,256,"weights_int8_bf16/%d_mlp.gate_proj",l);  gw[l] = upload(p);
        snprintf(p,256,"weights_int8_bf16/%d_mlp.up_proj",l);    uw[l] = upload(p);
        snprintf(p,256,"weights_int8_bf16/%d_mlp.down_proj",l);  dw[l] = upload(p);
    }

    // ── Buffers ─────────────────────────────────────────────────────────────
    float *d_x32, *d_xs;
    float *d_Q, *d_K, *d_V, *d_attn, *d_proj;
    float *d_gate, *d_up;
    float *d_res;          // FP32 residual / temp (size max(H,I))
    void *d_x_fp4;         // FP4 input to kernels
    float *d_attn_s, *d_mlp_s;
    void *d_attn_fp4, *d_mlp_fp4;

    cudaMalloc(&d_x32, H*4);
    cudaMalloc(&d_x_fp4, H);  cudaMalloc(&d_xs, (H/16)*4);
    cudaMalloc(&d_Q, Q*4); cudaMalloc(&d_K, KV*4); cudaMalloc(&d_V, KV*4);
    cudaMalloc(&d_attn, Q*4); cudaMalloc(&d_proj, H*4);
    cudaMalloc(&d_gate, I*4); cudaMalloc(&d_up, I*4);
    cudaMalloc(&d_res, I*4);
    cudaMalloc(&d_attn_fp4, Q); cudaMalloc(&d_attn_s, (Q/16)*4);
    cudaMalloc(&d_mlp_fp4, I); cudaMalloc(&d_mlp_s, (I/16)*4);

    float *d_rn; cudaMalloc(&d_rn, H*4);
    std::vector<float> rn_h(H,1.f); cudaMemcpy(d_rn,rn_h.data(),H*4,cudaMemcpyHostToDevice);

    // KV cache
    float *d_kc, *d_vc;
    size_t kv_sz = (size_t)num_layers * nkv * ms * hd * 4;
    cudaMalloc(&d_kc, kv_sz); cudaMalloc(&d_vc, kv_sz);
    cudaMemset(d_kc, 0, kv_sz); cudaMemset(d_vc, 0, kv_sz);

    // Init x
    std::vector<float> xh(H, 1.f), xsh(H/16, s13);
    cudaMemcpy(d_x32, xh.data(), H*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_xs, xsh.data(), (H/16)*4, cudaMemcpyHostToDevice);
    blackwell::kernels::pack_fp4(d_x_fp4, d_x32, d_xs, H, 0);
    std::vector<float> as(Q/16,s13), ms2(I/16,s13);
    cudaMemcpy(d_attn_s, as.data(), (Q/16)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mlp_s, ms2.data(), (I/16)*4, cudaMemcpyHostToDevice);

    // ── Fill KV cache ───────────────────────────────────────────────────────
    printf("Filling KV cache (seq=0..128)... ");
    fflush(stdout);
    for (int sq = 0; sq <= 128; ++sq) {
        for (int l = 0; l < num_layers; ++l) {
            int kb = l * nkv * ms * hd;

            // Attention: Q,K,V from x_fp4 using fused kernel
            chk(blackwell::kernels::gemv_int8_from_fp4(d_Q, d_x_fp4, d_xs,
                qw[l].d, qw[l].sc, H, Q, 0), "Q");
            chk(blackwell::kernels::gemv_int8_from_fp4(d_K, d_x_fp4, d_xs,
                kw[l].d, kw[l].sc, H, KV, 0), "K");
            chk(blackwell::kernels::gemv_int8_from_fp4(d_V, d_x_fp4, d_xs,
                vw[l].d, vw[l].sc, H, KV, 0), "V");
            chk(blackwell::kernels::update_kv_cache(
                d_kc+kb, d_vc+kb, d_K, d_V, 0, sq, nkv, hd, ms, 0), "kv");
            chk(blackwell::kernels::attention_decode_gqa(
                d_attn, d_Q, d_kc+kb, d_vc+kb,
                sq, nqh, nkv, hd, ms, 0), "attn");
            chk(blackwell::kernels::pack_fp4(d_attn_fp4, d_attn, d_attn_s, Q, 0), "pack_a");

            // Wo: attn_fp4 → gemv_int8_from_fp4 (fused!)
            chk(blackwell::kernels::gemv_int8_from_fp4(d_proj, d_attn_fp4, d_attn_s,
                ow[l].d, ow[l].sc, Q, H, 0), "Wo");

            // Residual + norm
            blackwell::kernels::unpack_fp4(d_res, d_x_fp4, d_xs, H, 0);
            blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_res, H, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, d_proj, d_rn, H, 1e-5f, 0);

            // MLP: gate + up
            chk(blackwell::kernels::gemv_int8_from_fp4(d_gate, d_x_fp4, d_xs,
                gw[l].d, gw[l].sc, H, I, 0), "gate");
            chk(blackwell::kernels::gemv_int8_from_fp4(d_up, d_x_fp4, d_xs,
                uw[l].d, uw[l].sc, H, I, 0), "up");
            chk(blackwell::kernels::apply_swiglu(d_res, d_gate, d_up, I, 0), "swiglu");
            // Use d_gate as mlp output buffer (I floats, reuse)
            float* d_mlp = d_gate;
            cudaMemcpy(d_mlp, d_res, I*4, cudaMemcpyDeviceToDevice);
            chk(blackwell::kernels::pack_fp4(d_mlp_fp4, d_mlp, d_mlp_s, I, 0), "pack_m");

            // Down: mlp_fp4 → gemv_int8_from_fp4 (fused!)
            chk(blackwell::kernels::gemv_int8_from_fp4(d_proj, d_mlp_fp4, d_mlp_s,
                dw[l].d, dw[l].sc, I, H, 0), "down");

            // Residual + norm
            blackwell::kernels::unpack_fp4(d_res, d_x_fp4, d_xs, H, 0);
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
            int kb = l * nkv * ms * hd;
            chk(blackwell::kernels::gemv_int8_from_fp4(d_Q, d_x_fp4, d_xs,
                qw[l].d, qw[l].sc, H, Q, 0), "Q");
            chk(blackwell::kernels::gemv_int8_from_fp4(d_K, d_x_fp4, d_xs,
                kw[l].d, kw[l].sc, H, KV, 0), "K");
            chk(blackwell::kernels::gemv_int8_from_fp4(d_V, d_x_fp4, d_xs,
                vw[l].d, vw[l].sc, H, KV, 0), "V");
            chk(blackwell::kernels::update_kv_cache(
                d_kc+kb, d_vc+kb, d_K, d_V, 0, sq, nkv, hd, ms, 0), "kv");
            chk(blackwell::kernels::attention_decode_gqa(
                d_attn, d_Q, d_kc+kb, d_vc+kb,
                sq, nqh, nkv, hd, ms, 0), "attn");
            chk(blackwell::kernels::pack_fp4(d_attn_fp4, d_attn, d_attn_s, Q, 0), "pack_a");
            chk(blackwell::kernels::gemv_int8_from_fp4(d_proj, d_attn_fp4, d_attn_s,
                ow[l].d, ow[l].sc, Q, H, 0), "Wo");
            blackwell::kernels::unpack_fp4(d_res, d_x_fp4, d_xs, H, 0);
            blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_res, H, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, d_proj, d_rn, H, 1e-5f, 0);
            // MLP
            chk(blackwell::kernels::gemv_int8_from_fp4(d_gate, d_x_fp4, d_xs,
                gw[l].d, gw[l].sc, H, I, 0), "gate");
            chk(blackwell::kernels::gemv_int8_from_fp4(d_up, d_x_fp4, d_xs,
                uw[l].d, uw[l].sc, H, I, 0), "up");
            chk(blackwell::kernels::apply_swiglu(d_res, d_gate, d_up, I, 0), "swiglu");
            chk(blackwell::kernels::pack_fp4(d_mlp_fp4, d_res, d_mlp_s, I, 0), "pack_m");
            chk(blackwell::kernels::gemv_int8_from_fp4(d_proj, d_mlp_fp4, d_mlp_s,
                dw[l].d, dw[l].sc, I, H, 0), "down");
            blackwell::kernels::unpack_fp4(d_res, d_x_fp4, d_xs, H, 0);
            blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_res, H, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, d_proj, d_rn, H, 1e-5f, 0);
        }
    }
    cudaDeviceSynchronize();

    // ── Benchmark ───────────────────────────────────────────────────────────
    KernelTimer timers[] = {
        {"gemv_int8_from_fp4 (Q)"}, {"gemv_int8_from_fp4 (K)"},
        {"gemv_int8_from_fp4 (V)"},
        {"update_kv_cache"}, {"attention_decode_gqa"},
        {"pack_fp4(attn)"}, {"gemv_int8_from_fp4 (Wo)"},
        {"fused_rmsnorm_pack(attn)"},
        {"gemv_int8_from_fp4 (gate)"}, {"gemv_int8_from_fp4 (up)"},
        {"apply_swiglu"}, {"pack_fp4(mlp)"},
        {"gemv_int8_from_fp4 (down)"}, {"fused_rmsnorm_pack(mlp)"}
    };

    printf("Benchmarking %d tokens...\n", bench);
    GpuTimer t; t.start();
    for (int i = 0; i < bench; ++i) {
        for (int l = 0; l < num_layers; ++l) {
            GpuTimer lt;
            int kb = l * nkv * ms * hd;

            lt.start();
            blackwell::kernels::gemv_int8_from_fp4(d_Q, d_x_fp4, d_xs,
                qw[l].d, qw[l].sc, H, Q, 0);
            timers[0].add(lt.stop());
            lt.start();
            blackwell::kernels::gemv_int8_from_fp4(d_K, d_x_fp4, d_xs,
                kw[l].d, kw[l].sc, H, KV, 0);
            timers[1].add(lt.stop());
            lt.start();
            blackwell::kernels::gemv_int8_from_fp4(d_V, d_x_fp4, d_xs,
                vw[l].d, vw[l].sc, H, KV, 0);
            timers[2].add(lt.stop());

            lt.start();
            blackwell::kernels::update_kv_cache(
                d_kc+kb, d_vc+kb, d_K, d_V, 0, sq, nkv, hd, ms, 0);
            timers[3].add(lt.stop());
            lt.start();
            blackwell::kernels::attention_decode_gqa(
                d_attn, d_Q, d_kc+kb, d_vc+kb,
                sq, nqh, nkv, hd, ms, 0);
            timers[4].add(lt.stop());
            lt.start();
            blackwell::kernels::pack_fp4(d_attn_fp4, d_attn, d_attn_s, Q, 0);
            timers[5].add(lt.stop());

            lt.start();
            blackwell::kernels::gemv_int8_from_fp4(d_proj, d_attn_fp4, d_attn_s,
                ow[l].d, ow[l].sc, Q, H, 0);
            timers[6].add(lt.stop());

            // Residual: save x_fp32 before overwrite
            blackwell::kernels::unpack_fp4(d_res, d_x_fp4, d_xs, H, 0);
            blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_res, H, 0);
            lt.start();
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, d_proj, d_rn, H, 1e-5f, 0);
            timers[7].add(lt.stop());

            // MLP: gate + up (fused)
            lt.start();
            blackwell::kernels::gemv_int8_from_fp4(d_gate, d_x_fp4, d_xs,
                gw[l].d, gw[l].sc, H, I, 0);
            timers[8].add(lt.stop());
            lt.start();
            blackwell::kernels::gemv_int8_from_fp4(d_up, d_x_fp4, d_xs,
                uw[l].d, uw[l].sc, H, I, 0);
            timers[9].add(lt.stop());

            lt.start();
            blackwell::kernels::apply_swiglu(d_res, d_gate, d_up, I, 0);
            timers[10].add(lt.stop());
            lt.start();
            blackwell::kernels::pack_fp4(d_mlp_fp4, d_res, d_mlp_s, I, 0);
            timers[11].add(lt.stop());

            lt.start();
            blackwell::kernels::gemv_int8_from_fp4(d_proj, d_mlp_fp4, d_mlp_s,
                dw[l].d, dw[l].sc, I, H, 0);
            timers[12].add(lt.stop());

            // Residual
            blackwell::kernels::unpack_fp4(d_res, d_x_fp4, d_xs, H, 0);
            blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_res, H, 0);
            lt.start();
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, d_proj, d_rn, H, 1e-5f, 0);
            timers[13].add(lt.stop());
        }
    }
    float total_ms = t.stop();
    float ptm = total_ms / bench;
    float tps = 1000.f / ptm;
    float s28 = 1000.f / (ptm * 28.f / num_layers);

    printf("\n=== Fused INT8 Decode Results ===\n");
    printf("  Layers:         %d\n", num_layers);
    printf("  Tokens:         %d\n", bench);
    printf("  Total:          %.2f ms\n", total_ms);
    printf("  Per-token:      %.3f ms\n", ptm);
    printf("  Throughput:     %.1f t/s\n", tps);
    printf("  Scaled 28:      %.1f t/s\n", s28);

    printf("\n=== Per-Kernel Timing ===\n");
    double kt_total = 0;
    for (int i = 0; i < 14; ++i) { timers[i].print(); kt_total += timers[i].tot; }
    printf("  %-30s  %7.3f ms\n", "TOTAL KERNEL", kt_total);

    // ── Cleanup ─────────────────────────────────────────────────────────────
    for (int l = 0; l < num_layers; ++l) {
        cudaFree(qw[l].d); cudaFree(qw[l].sc);
        cudaFree(kw[l].d); cudaFree(kw[l].sc);
        cudaFree(vw[l].d); cudaFree(vw[l].sc);
        cudaFree(ow[l].d); cudaFree(ow[l].sc);
        cudaFree(gw[l].d); cudaFree(gw[l].sc);
        cudaFree(uw[l].d); cudaFree(uw[l].sc);
        cudaFree(dw[l].d); cudaFree(dw[l].sc);
    }
    cudaFree(d_x32); cudaFree(d_x_fp4); cudaFree(d_xs);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_attn); cudaFree(d_proj);
    cudaFree(d_gate); cudaFree(d_up);
    cudaFree(d_res); cudaFree(d_rn);
    cudaFree(d_attn_fp4); cudaFree(d_attn_s);
    cudaFree(d_mlp_fp4); cudaFree(d_mlp_s);
    cudaFree(d_kc); cudaFree(d_vc);
    return 0;
}