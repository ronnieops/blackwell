// bench/decode_int8_direct.cu — INT8-only decode pipeline (no FP4)
//
// Architecture:
//   Layer n: d_x_fp32 (FP32 residual) → input to next layer
//   After each layer: fused_rmsnorm_quant_int8 → d_x_int8
//   Next layer: GEMVs consume d_x_int8 directly (never touch FP4)
//   FP4 only used for output storage/sampling
//
// Benchmark: INT8 direct path vs FP4 round-trip path
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/decode_int8_direct.cu build/libblackwell_kernels.a \
//     -o bench/decode_int8_direct

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
    void start(cudaStream_t st = 0) { cudaEventRecord(s, st); }
    float stop(cudaStream_t st = 0) {
        cudaEventRecord(e, st); cudaEventSynchronize(e);
        float ms = 0; cudaEventElapsedTime(&ms, s, e); return ms;
    }
};

static void chk(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) { printf("FAIL: %s: %s\n", msg, cudaGetErrorString(e)); exit(1); }
}

struct LoadedW { int K, N; std::vector<int8_t> d; std::vector<float> sc; };
static LoadedW load_int8_w(const char* prefix) {
    char p[256]; snprintf(p,256,"%s.int8_t",prefix);
    FILE* f = fopen(p,"rb"); int h[5]; fread(h,4,5,f);
    LoadedW w; w.K=h[0]; w.N=h[1];
    w.d.resize(h[0]*h[1]); fread(w.d.data(),1,w.d.size(),f); fclose(f);
    snprintf(p,256,"%s.scale_t",prefix); f=fopen(p,"rb"); fread(h,4,5,f);
    w.sc.resize(h[3]*h[4]); fread(w.sc.data(),4,w.sc.size(),f); fclose(f);
    return w;
}

struct DevW { int K, N; int8_t* d; float* sc; };
static DevW upload(const LoadedW& w) {
    DevW dw; dw.K=w.K; dw.N=w.N;
    cudaMalloc(&dw.d, w.d.size()); cudaMemcpy(dw.d, w.d.data(), w.d.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&dw.sc, w.sc.size()*4); cudaMemcpy(dw.sc, w.sc.data(), w.sc.size()*4, cudaMemcpyHostToDevice);
    return dw;
}

struct LayerW { DevW q,k,v,o,g,u,d; };  // qkv+mlp per layer

int main(int argc, char** argv) {
    int num_layers = 2, iters = 100;
    if (argc > 1) num_layers = atoi(argv[1]);
    if (argc > 2) iters = atoi(argv[2]);

    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("# INT8 Direct Decode (No FP4) — Qwen3-1.7B\n");
    printf("Device: %s (CC %d.%d)\n", p.name, p.major, p.minor);
    printf("Layers: %d, Iters: %d\n\n", num_layers, iters);

    // ── Config ──────────────────────────────────────────────────────────────
    const int H = 2048, Q = 2048, KV = 1024, I = 6144;
    const int nqh = 12, nkv = 12, hd = 64, ms = 128;  // GQA
    const float ixv = 1.f / 127.f;

    // ── Allocate persistent working buffers (NO per-layer alloc) ────────────
    // Direct INT8 path: d_x_fp32 is the residual in FP32.
    // Each layer: rmsnorm(d_x_fp32) → pack to d_x_int8 → GEMVs
    float *d_x_fp32, *d_res;
    int8_t *d_x_int8;
    float *d_xs_int8;  // scales for d_x_int8
    cudaMalloc(&d_x_fp32, H*4);
    cudaMalloc(&d_res, H*4);
    cudaMalloc(&d_x_int8, H);
    cudaMalloc(&d_xs_int8, (H/16)*4);

    // Attention inputs/outputs
    float *d_Q, *d_K, *d_V, *d_attn;
    int8_t *d_attn_i8;
    float *d_attn_i8s;
    cudaMalloc(&d_Q, Q*4); cudaMalloc(&d_K, KV*4); cudaMalloc(&d_V, KV*4);
    cudaMalloc(&d_attn, Q*4);
    cudaMalloc(&d_attn_i8, Q);
    cudaMalloc(&d_attn_i8s, (Q/16)*4);

    // MLP intermediates
    float *d_gate, *d_up, *d_mlp;
    int8_t *d_mlp_i8;
    float *d_mlp_i8s;
    cudaMalloc(&d_gate, I*4); cudaMalloc(&d_up, I*4); cudaMalloc(&d_mlp, I*4);
    cudaMalloc(&d_mlp_i8, I);
    cudaMalloc(&d_mlp_i8s, (I/16)*4);

    // Output
    float *d_proj;

    // KV cache
    size_t kv_sz = (size_t)num_layers * nkv * ms * hd * 4;
    float *d_kc, *d_vc;
    cudaMalloc(&d_kc, kv_sz); cudaMalloc(&d_vc, kv_sz);
    cudaMemset(d_kc, 0, kv_sz); cudaMemset(d_vc, 0, kv_sz);

    // RMSNorm weight
    float *d_rn;
    cudaMalloc(&d_rn, H*4);

    // Initialize: seed d_x_fp32 with random-like input
    std::vector<float> init_x(H);
    for (int i = 0; i < H; ++i) init_x[i] = (i % 17 - 8) * 0.01f;
    cudaMemcpy(d_x_fp32, init_x.data(), H*4, cudaMemcpyHostToDevice);

    std::vector<float> ix8s(H/16, ixv), attn_i8s(Q/16, ixv), mlp_i8s(I/16, ixv);
    cudaMemcpy(d_xs_int8, ix8s.data(), (H/16)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_attn_i8s, attn_i8s.data(), (Q/16)*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mlp_i8s, mlp_i8s.data(), (I/16)*4, cudaMemcpyHostToDevice);

    // ── Load all layer weights ──────────────────────────────────────────────
    printf("Loading %d layers...\n", num_layers);
    std::vector<LayerW> lw(num_layers);
    for (int l = 0; l < num_layers; ++l) {
        char prefix[256];
        snprintf(prefix,256,"weights_int8_bf16/%d_self_attn.q_proj",l);
        lw[l].q = upload(load_int8_w(prefix));
        snprintf(prefix,256,"weights_int8_bf16/%d_self_attn.k_proj",l);
        lw[l].k = upload(load_int8_w(prefix));
        snprintf(prefix,256,"weights_int8_bf16/%d_self_attn.v_proj",l);
        lw[l].v = upload(load_int8_w(prefix));
        snprintf(prefix,256,"weights_int8_bf16/%d_self_attn.o_proj",l);
        lw[l].o = upload(load_int8_w(prefix));
        snprintf(prefix,256,"weights_int8_bf16/%d_mlp.gate_proj",l);
        lw[l].g = upload(load_int8_w(prefix));
        snprintf(prefix,256,"weights_int8_bf16/%d_mlp.up_proj",l);
        lw[l].u = upload(load_int8_w(prefix));
        snprintf(prefix,256,"weights_int8_bf16/%d_mlp.down_proj",l);
        lw[l].d = upload(load_int8_w(prefix));
    }
    // RMSNorm weight (use layer 0's)
    {
        FILE* f = fopen("weights_int8_bf16/0_self_attn.q_proj.scale_t","rb"); int h[5]; fread(h,4,5,f); fclose(f);
        std::vector<float> rn(H, 1.f);
        cudaMemcpy(d_rn, rn.data(), H*4, cudaMemcpyHostToDevice);
    }
    printf("Weights loaded. Starting benchmark.\n");

    // ── Per-layer decode (direct INT8, no FP4) ───────────────────────────────
    // Pipeline:
    //   d_x_fp32 (residual) → RMSNorm → quant → d_x_int8
    //   d_x_int8 → GEMVs → attention → Wo → residual+out
    //   residual → RMSNorm → quant → d_x_int8
    //   d_x_int8 → gate/up → SwiGLU → down → residual+out
    //   NO FP4 anywhere

    GpuTimer t;
    t.start();
    for (int i = 0; i < iters; ++i) {
        for (int l = 0; l < num_layers; ++l) {
            int kb = l * nkv * ms * hd;

            // Phase 1: RMSNorm + quantize → INT8 for QKV
            blackwell::kernels::fused_rmsnorm_quant_int8(d_x_int8, d_xs_int8,
                d_x_fp32, d_rn, H, 1e-6f, 0);

            // QKV GEMVs (consume d_x_int8 directly)
            blackwell::kernels::gemv_int8(d_Q, d_x_int8, d_xs_int8, lw[l].q.d, lw[l].q.sc, H, Q, 0);
            blackwell::kernels::gemv_int8(d_K, d_x_int8, d_xs_int8, lw[l].k.d, lw[l].k.sc, H, KV, 0);
            blackwell::kernels::gemv_int8(d_V, d_x_int8, d_xs_int8, lw[l].v.d, lw[l].v.sc, H, KV, 0);

            // KV cache + attention
            blackwell::kernels::update_kv_cache(d_kc+kb, d_vc+kb, d_K, d_V, 0, ms-1, nkv, hd, ms, 0);
            blackwell::kernels::attention_decode_gqa(d_attn, d_Q, d_kc+kb, d_vc+kb, ms-1, nqh, nkv, hd, ms, 0);

            // Wo + residual add
            blackwell::kernels::pack_int8(d_attn_i8, d_attn, d_attn_i8s, Q, 0);
            d_proj = d_gate;  // reuse buffer
            blackwell::kernels::gemv_int8(d_proj, d_attn_i8, d_attn_i8s, lw[l].o.d, lw[l].o.sc, Q, H, 0);
            blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_x_fp32, H, 0);

            // Phase 1 end: d_proj is residual for next RMSNorm
            // Reuse d_gate as output buffer (we'll compute gate GEMV in place)
            // Save d_proj to d_res for phase 2
            cudaMemcpy(d_res, d_proj, H*4, cudaMemcpyDeviceToDevice);

            // Phase 2: RMSNorm on attn result → quant → INT8 for MLP
            // d_proj currently holds attn result. RMSNorm it.
            blackwell::kernels::fused_rmsnorm_quant_int8(d_x_int8, d_xs_int8,
                d_proj, d_rn, H, 1e-6f, 0);
            // Store d_proj (attn-out after residual) in d_res for next layer's residual add
            // d_res holds the attn output (pre-RMSNorm residual add val)
            // Actually: d_x_fp32 is no longer needed after we've RMSNorm'd it.
            // Let's use d_x_fp32 as the input buffer for the RMSNorm above.
            // After RMSNorm+quant, d_proj is free. Compute gate GEMV into it.

            // MLP GEMVs
            blackwell::kernels::gemv_int8(d_gate, d_x_int8, d_xs_int8, lw[l].g.d, lw[l].g.sc, H, I, 0);
            blackwell::kernels::gemv_int8(d_up, d_x_int8, d_xs_int8, lw[l].u.d, lw[l].u.sc, H, I, 0);
            blackwell::kernels::apply_swiglu(d_mlp, d_gate, d_up, I, 0);
            blackwell::kernels::pack_int8(d_mlp_i8, d_mlp, d_mlp_i8s, I, 0);
            blackwell::kernels::gemv_int8(d_proj, d_mlp_i8, d_mlp_i8s, lw[l].d.d, lw[l].d.sc, I, H, 0);

            // Residual add: d_proj(out) + d_res(original residual from attn)
            blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_res, H, 0);

            // Copy back to d_x_fp32 for next layer
            // At this point d_proj = final FP32 (after down_proj + residual)
            // This feeds into next layer's RMSNorm as input.
            // No final RMSNorm needed at last layer? Let's add it for correctness.
            blackwell::kernels::fused_rmsnorm_quant_int8(d_x_int8, d_xs_int8,
                d_proj, d_rn, H, 1e-6f, 0);
            // Re-convert to FP32 for next layer's input
            // Actually we need FP32 for the next layer's input.
            // Skip: quantize only happens *before* GEMVs.
            // Let's do: d_proj already FP32, copy to d_x_fp32 for next layer.
            // Let d_x_fp32 be the FP32 residual buffer.
            // At layer end, store output in d_x_fp32.
            // The quant happens at layer start.
            cudaMemcpy(d_x_fp32, d_proj, H*4, cudaMemcpyDeviceToDevice);
        }
    }
    cudaDeviceSynchronize();
    float ms_direct = t.stop();

    float pt_direct = ms_direct / iters;
    float tps_direct = 1000.f / (pt_direct * 28.f / num_layers);
    printf("=== Direct INT8 (No FP4) ===\n");
    printf("  Per-token: %.2f ms (%.1f GB/s)\n", pt_direct, 4.f*H*2*3*num_layers*iters/(pt_direct*1e3));
    printf("  t/s:       %.1f  (28L scaled)\n", tps_direct);
    printf("  Total:     %.2f ms (%.2f us/layer)\n", ms_direct, pt_direct/num_layers*1000);

    // For comparison: same code with FP4 round-trip
    float *d_x_fp4, *d_xs_fp4;
    cudaMalloc(&d_x_fp4, H);
    cudaMalloc(&d_xs_fp4, (H/16)*4);
    cudaMemcpy(d_x_fp32, init_x.data(), H*4, cudaMemcpyHostToDevice);
    cudaMemset(d_kc, 0, kv_sz); cudaMemset(d_vc, 0, kv_sz);

    GpuTimer t2;
    t2.start();
    for (int i = 0; i < iters; ++i) {
        for (int l = 0; l < num_layers; ++l) {
            int kb = l * nkv * ms * hd;

            // FP4 path: unpack → pack_int8
            blackwell::kernels::unpack_fp4(d_res, d_x_fp4, d_xs_fp4, H, 0);
            blackwell::kernels::pack_int8(d_x_int8, d_res, d_xs_int8, H, 0);

            blackwell::kernels::gemv_int8(d_Q, d_x_int8, d_xs_int8, lw[l].q.d, lw[l].q.sc, H, Q, 0);
            blackwell::kernels::gemv_int8(d_K, d_x_int8, d_xs_int8, lw[l].k.d, lw[l].k.sc, H, KV, 0);
            blackwell::kernels::gemv_int8(d_V, d_x_int8, d_xs_int8, lw[l].v.d, lw[l].v.sc, H, KV, 0);

            blackwell::kernels::update_kv_cache(d_kc+kb, d_vc+kb, d_K, d_V, 0, ms-1, nkv, hd, ms, 0);
            blackwell::kernels::attention_decode_gqa(d_attn, d_Q, d_kc+kb, d_vc+kb, ms-1, nqh, nkv, hd, ms, 0);

            blackwell::kernels::pack_int8(d_attn_i8, d_attn, d_attn_i8s, Q, 0);
            d_proj = d_gate;
            blackwell::kernels::gemv_int8(d_proj, d_attn_i8, d_attn_i8s, lw[l].o.d, lw[l].o.sc, Q, H, 0);
            blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_res, H, 0);

            blackwell::kernels::fused_rmsnorm_quant_int8(d_x_int8, d_xs_int8, d_proj, d_rn, H, 1e-6f, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs_fp4, d_proj, d_rn, H, 1e-6f, 0);

            // MLP
            blackwell::kernels::unpack_fp4(d_res, d_x_fp4, d_xs_fp4, H, 0);
            blackwell::kernels::pack_int8(d_x_int8, d_res, d_xs_int8, H, 0);
            blackwell::kernels::gemv_int8(d_gate, d_x_int8, d_xs_int8, lw[l].g.d, lw[l].g.sc, H, I, 0);
            blackwell::kernels::gemv_int8(d_up, d_x_int8, d_xs_int8, lw[l].u.d, lw[l].u.sc, H, I, 0);
            blackwell::kernels::apply_swiglu(d_mlp, d_gate, d_up, I, 0);
            blackwell::kernels::pack_int8(d_mlp_i8, d_mlp, d_mlp_i8s, I, 0);
            blackwell::kernels::gemv_int8(d_proj, d_mlp_i8, d_mlp_i8s, lw[l].d.d, lw[l].d.sc, I, H, 0);
            blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_res, H, 0);

            blackwell::kernels::fused_rmsnorm_quant_int8(d_x_int8, d_xs_int8, d_proj, d_rn, H, 1e-6f, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs_fp4, d_proj, d_rn, H, 1e-6f, 0);
        }
    }
    cudaDeviceSynchronize();
    float ms_fp4 = t2.stop();

    float pt_fp4 = ms_fp4 / iters;
    float tps_fp4 = 1000.f / (pt_fp4 * 28.f / num_layers);
    printf("\n=== FP4 Round-trip (baseline) ===\n");
    printf("  Per-token: %.2f ms\n", pt_fp4);
    printf("  t/s:       %.1f  (28L scaled)\n", tps_fp4);
    printf("  Total:     %.2f ms (%.2f us/layer)\n", ms_fp4, pt_fp4/num_layers*1000);

    printf("\n=== Speedup ===\n");
    printf("  Direct vs FP4: %.2fx (%.1f%% faster)\n", ms_fp4/ms_direct, 100*(ms_fp4-ms_direct)/ms_fp4);
    printf("  Layer saving:  %.2f us/layer\n", (ms_fp4-ms_direct)/iters/num_layers*1000);

    // Real decode comparison: per-kernel wall-clock
    // Reset buffers
    cudaMemcpy(d_x_fp32, init_x.data(), H*4, cudaMemcpyHostToDevice);
    cudaMemset(d_kc, 0, kv_sz); cudaMemset(d_vc, 0, kv_sz);

    printf("\n=== Real Decode Timeline (per-kernel events) ===\n");
    GpuTimer tk;
    struct { const char* n; float m; } pt[20];
    int npt = 0;
    auto tick=[&](const char* n){
        tk.start();
        pt[npt].n=n; pt[npt].m=0; npt++;
    };
    auto tack=[&](){
        pt[npt-1].m = tk.stop();
    };
    auto phase=[&](int l){
        int kb=l*nkv*ms*hd;
        tick("rmsnorm_q"); chk(blackwell::kernels::fused_rmsnorm_quant_int8(d_x_int8,d_xs_int8,d_x_fp32,d_rn,H,1e-6f,0),"rn"); tack();

        tick("q_gemv"); blackwell::kernels::gemv_int8(d_Q,d_x_int8,d_xs_int8,lw[l].q.d,lw[l].q.sc,H,Q,0); tack();
        tick("k_gemv"); blackwell::kernels::gemv_int8(d_K,d_x_int8,d_xs_int8,lw[l].k.d,lw[l].k.sc,H,KV,0); tack();
        tick("v_gemv"); blackwell::kernels::gemv_int8(d_V,d_x_int8,d_xs_int8,lw[l].v.d,lw[l].v.sc,H,KV,0); tack();

        tick("kv_upd"); blackwell::kernels::update_kv_cache(d_kc+kb,d_vc+kb,d_K,d_V,0,ms-1,nkv,hd,ms,0); tack();
        tick("attn"); blackwell::kernels::attention_decode_gqa(d_attn,d_Q,d_kc+kb,d_vc+kb,ms-1,nqh,nkv,hd,ms,0); tack();

        tick("pack_attn"); blackwell::kernels::pack_int8(d_attn_i8,d_attn,d_attn_i8s,Q,0); tack();
        d_proj = d_gate;
        tick("o_gemv"); blackwell::kernels::gemv_int8(d_proj,d_attn_i8,d_attn_i8s,lw[l].o.d,lw[l].o.sc,Q,H,0); tack();
        tick("add_res1"); blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_x_fp32,H,0); tack();

        cudaMemcpy(d_res, d_proj, H*4, cudaMemcpyDeviceToDevice);
        tick("rmsnorm_m"); chk(blackwell::kernels::fused_rmsnorm_quant_int8(d_x_int8,d_xs_int8,d_proj,d_rn,H,1e-6f,0),"rn2"); tack();

        tick("gate"); blackwell::kernels::gemv_int8(d_gate,d_x_int8,d_xs_int8,lw[l].g.d,lw[l].g.sc,H,I,0); tack();
        tick("up"); blackwell::kernels::gemv_int8(d_up,d_x_int8,d_xs_int8,lw[l].u.d,lw[l].u.sc,H,I,0); tack();
        tick("swiglu"); blackwell::kernels::apply_swiglu(d_mlp,d_gate,d_up,I,0); tack();
        tick("pack_mlp"); blackwell::kernels::pack_int8(d_mlp_i8,d_mlp,d_mlp_i8s,I,0); tack();
        tick("down"); blackwell::kernels::gemv_int8(d_proj,d_mlp_i8,d_mlp_i8s,lw[l].d.d,lw[l].d.sc,I,H,0); tack();
        tick("add_res2"); blackwell::kernels::vector_add_fp32(d_proj,d_proj,d_res,H,0); tack();

        cudaMemcpy(d_x_fp32, d_proj, H*4, cudaMemcpyDeviceToDevice);
    };

    // Warmup
    for(int w=0;w<5;++w) phase(0);
    cudaDeviceSynchronize();

    // Time each step
    GpuTimer tper;
    for(int i=0;i<2;++i){
        npt=0; tper.start();
        phase(0);
        pt[npt-1].m = tper.stop();
    }
    float layer_ms = pt[npt-1].m;
    printf("  1 layer = %.1f us\n", layer_ms*1000);
    for(int i=0;i<npt;++i){
        printf("    %-12s %7.2f us (%5.1f%%)\n", pt[i].n, pt[i].m*1000, 100*pt[i].m/layer_ms);
    }

    float tps = 1000.f / (layer_ms * 28.f / num_layers);
    printf("\n  Direct INT8 throughput: %.1f t/s (28L scaled)\n", tps);
    printf("  vs FP4 throughput:       %.1f t/s\n", tps_fp4);
    printf("  Speedup: %.2fx\n", tps/tps_fp4);

    cudaFree(d_x_fp32); cudaFree(d_res); cudaFree(d_x_int8); cudaFree(d_xs_int8);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_attn);
    cudaFree(d_attn_i8); cudaFree(d_attn_i8s);
    cudaFree(d_gate); cudaFree(d_up); cudaFree(d_mlp);
    cudaFree(d_mlp_i8); cudaFree(d_mlp_i8s);
    cudaFree(d_proj); cudaFree(d_x_fp4); cudaFree(d_xs_fp4);
    cudaFree(d_kc); cudaFree(d_vc); cudaFree(d_rn);
    return 0;
}