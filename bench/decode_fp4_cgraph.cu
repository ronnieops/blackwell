// bench/decode_fp4_cgraph.cu — FP4 packed pipeline with CUDA Graph
//
// FP32 activations × packed FP4 weights throughout. No activation quantization.
// All 7 projections per layer use gemv_fp32_fp4_warp.
// CUDA Graph capture for minimum launch overhead.
//
// Build:
//   nvcc -O3 -std=c++17 -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/decode_fp4_cgraph.cu build/libblackwell_kernels.a \
//     -o bench/decode_fp4_cgraph

#include <cuda_runtime.h>
#include <cuda_fp4.h>
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

struct LoadedFP4 { int K, N; std::vector<uint8_t> packed; std::vector<float> sc; };
static LoadedFP4 load_fp4_w(const char* prefix) {
    char p[256]; snprintf(p,256,"%s.packed_fp4",prefix);
    FILE* f = fopen(p,"rb"); int h[5]; fread(h,4,5,f);
    LoadedFP4 w; w.K=h[0]; w.N=h[1]; w.packed.resize(h[4]);
    fread(w.packed.data(),1,w.packed.size(),f); fclose(f);
    f = fopen(p,"rb"); fread(h,4,5,f);
    fseek(f, 20 + h[4], SEEK_SET);
    w.sc.resize(h[3] * (h[0]/16));
    fread(w.sc.data(),4,w.sc.size(),f); fclose(f);
    return w;
}

struct DevFP4 { int K, N; uint8_t* d; float* sc; };
static DevFP4 upload_fp4(const char* prefix) {
    auto w = load_fp4_w(prefix); DevFP4 dw{w.K, w.N};
    size_t psz = (size_t)w.K * w.N / 2;
    cudaMalloc(&dw.d, psz); cudaMemcpy(dw.d, w.packed.data(), psz, cudaMemcpyHostToDevice);
    cudaMalloc(&dw.sc, w.sc.size()*4);
    cudaMemcpy(dw.sc, w.sc.data(), w.sc.size()*4, cudaMemcpyHostToDevice);
    return dw;
}

struct LayerBufs {
    float *d_Q, *d_K, *d_V, *d_attn, *d_proj;
    float *d_gate, *d_up, *d_mlp;
    float *d_res;           // FP32 residual (full precision)
};

int main(int argc, char** argv) {
    int num_layers = 4;
    if (argc > 1) num_layers = atoi(argv[1]);
    if (num_layers > 28) num_layers = 28;

    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("# FP4 Packed CUDA Graph Decode Benchmark — Qwen3-1.7B\n");
    printf("Device: %s (%d.%d)\n", p.name, p.major, p.minor);
    printf("Layers: %d\n", num_layers);

    const int H = 2048, Q = 2048, KV = 1024, I = 6144;
    const int nqh = 16, nkv = 8, hd = 128, ms = 2048;

    // Load packed FP4 weights
    struct LW { DevFP4 q,k,v,o,g,u,d; };
    printf("Loading packed FP4 weights...\n");
    std::vector<LW> lw(num_layers);
    for (int l = 0; l < num_layers; ++l) {
        char p[256];
        snprintf(p,256,"weights_packed_fp4/%d_self_attn.q_proj",l); lw[l].q = upload_fp4(p);
        snprintf(p,256,"weights_packed_fp4/%d_self_attn.k_proj",l); lw[l].k = upload_fp4(p);
        snprintf(p,256,"weights_packed_fp4/%d_self_attn.v_proj",l); lw[l].v = upload_fp4(p);
        snprintf(p,256,"weights_packed_fp4/%d_self_attn.o_proj",l); lw[l].o = upload_fp4(p);
        snprintf(p,256,"weights_packed_fp4/%d_mlp.gate_proj",l);  lw[l].g = upload_fp4(p);
        snprintf(p,256,"weights_packed_fp4/%d_mlp.up_proj",l);    lw[l].u = upload_fp4(p);
        snprintf(p,256,"weights_packed_fp4/%d_mlp.down_proj",l);  lw[l].d = upload_fp4(p);
    }

    // ── Buffers ─────────────────────────────────────────────────────────────
    float *d_rn;  // RMSNorm weight (all-ones for synthetic benchmark)
    cudaMalloc(&d_rn, H*4);
    std::vector<float> rn_h(H,1.f); cudaMemcpy(d_rn, rn_h.data(), H*4, cudaMemcpyHostToDevice);

    // Reusable per-layer buffers
    LayerBufs b;
    cudaMalloc(&b.d_Q, Q*4); cudaMalloc(&b.d_K, KV*4); cudaMalloc(&b.d_V, KV*4);
    cudaMalloc(&b.d_attn, Q*4); cudaMalloc(&b.d_proj, H*4);
    cudaMalloc(&b.d_gate, I*4); cudaMalloc(&b.d_up, I*4); cudaMalloc(&b.d_mlp, I*4);
    cudaMalloc(&b.d_res, H*4);

    // Init residual = uniform 1.0 (FP32, full precision)
    std::vector<float> res_h(H, 0.5f);
    cudaMemcpy(b.d_res, res_h.data(), H*4, cudaMemcpyHostToDevice);

    // KV cache
    float *d_kc, *d_vc;
    size_t kv_sz = (size_t)num_layers * nkv * ms * hd * 4;
    cudaMalloc(&d_kc, kv_sz); cudaMalloc(&d_vc, kv_sz);
    cudaMemset(d_kc, 0, kv_sz); cudaMemset(d_vc, 0, kv_sz);

    // ── Fill KV cache (seq=0..128) ──────────────────────────────────────────
    printf("Filling KV cache (seq=0..128)... ");
    fflush(stdout);
    int sq = 128;
    for (int s = 0; s <= sq; ++s) {
        for (int l = 0; l < num_layers; ++l) {
            // Same per-layer logic as benchmark
            chk(blackwell::kernels::fused_rmsnorm(b.d_proj, b.d_res, d_rn, H, 1e-6f, 0), "rms_pre");
            chk(blackwell::kernels::gemv_fp32_fp4_warp(b.d_Q, b.d_proj, lw[l].q.d, lw[l].q.sc, H, Q, 0), "Q");
            chk(blackwell::kernels::gemv_fp32_fp4_warp(b.d_K, b.d_proj, lw[l].k.d, lw[l].k.sc, H, KV, 0), "K");
            chk(blackwell::kernels::gemv_fp32_fp4_warp(b.d_V, b.d_proj, lw[l].v.d, lw[l].v.sc, H, KV, 0), "V");
            int kb = l * nkv * ms * hd;
            chk(blackwell::kernels::update_kv_cache(d_kc+kb, d_vc+kb, b.d_K, b.d_V, 0, s, nkv, hd, ms, 0), "kv");
            chk(blackwell::kernels::attention_decode_gqa(b.d_attn, b.d_Q, d_kc+kb, d_vc+kb,
                s, nqh, nkv, hd, ms, 0), "attn");
            chk(blackwell::kernels::gemv_fp32_fp4_warp(b.d_proj, b.d_attn, lw[l].o.d, lw[l].o.sc, Q, H, 0), "O");
            chk(blackwell::kernels::vector_add_fp32(b.d_mlp, b.d_proj, b.d_res, H, 0), "res_attn");
            cudaMemcpy(b.d_res, b.d_mlp, H*4, cudaMemcpyDeviceToDevice);

            chk(blackwell::kernels::fused_rmsnorm(b.d_proj, b.d_res, d_rn, H, 1e-6f, 0), "rms_mlp");
            chk(blackwell::kernels::gemv_fp32_fp4_warp(b.d_gate, b.d_proj, lw[l].g.d, lw[l].g.sc, H, I, 0), "gate");
            chk(blackwell::kernels::gemv_fp32_fp4_warp(b.d_up, b.d_proj, lw[l].u.d, lw[l].u.sc, H, I, 0), "up");
            chk(blackwell::kernels::apply_swiglu(b.d_mlp, b.d_gate, b.d_up, I, 0), "swiglu");
            chk(blackwell::kernels::gemv_fp32_fp4_warp(b.d_proj, b.d_mlp, lw[l].d.d, lw[l].d.sc, I, H, 0), "down");
            chk(blackwell::kernels::vector_add_fp32(b.d_mlp, b.d_proj, b.d_res, H, 0), "res_mlp");
            cudaMemcpy(b.d_res, b.d_mlp, H*4, cudaMemcpyDeviceToDevice);
        }
    }
    printf("done\n");

    // ── Save initial state for correctness ──────────────────────────────────
    // Save KV cache
    float *d_kc_save, *d_vc_save;
    cudaMalloc(&d_kc_save, kv_sz); cudaMalloc(&d_vc_save, kv_sz);
    cudaMemcpy(d_kc_save, d_kc, kv_sz, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_vc_save, d_vc, kv_sz, cudaMemcpyDeviceToDevice);

    // Save residual
    float *d_res_save;
    cudaMalloc(&d_res_save, H*4);
    cudaMemcpy(d_res_save, b.d_res, H*4, cudaMemcpyDeviceToDevice);

    // ── Per-kernel baseline ─────────────────────────────────────────────────
    printf("Warmup (per-kernel)...\n");
    int warm = 5, bench = 20;
    for (int w = 0; w < warm; ++w) {
        for (int l = 0; l < num_layers; ++l) {
            chk(blackwell::kernels::fused_rmsnorm(b.d_proj, b.d_res, d_rn, H, 1e-6f, 0), "rms_pre");
            chk(blackwell::kernels::gemv_fp32_fp4_warp(b.d_Q, b.d_proj, lw[l].q.d, lw[l].q.sc, H, Q, 0), "Q");
            chk(blackwell::kernels::gemv_fp32_fp4_warp(b.d_K, b.d_proj, lw[l].k.d, lw[l].k.sc, H, KV, 0), "K");
            chk(blackwell::kernels::gemv_fp32_fp4_warp(b.d_V, b.d_proj, lw[l].v.d, lw[l].v.sc, H, KV, 0), "V");
            int kb = l * nkv * ms * hd;
            chk(blackwell::kernels::update_kv_cache(d_kc+kb, d_vc+kb, b.d_K, b.d_V, 0, sq, nkv, hd, ms, 0), "kv");
            chk(blackwell::kernels::attention_decode_gqa(b.d_attn, b.d_Q, d_kc+kb, d_vc+kb,
                sq, nqh, nkv, hd, ms, 0), "attn");
            chk(blackwell::kernels::gemv_fp32_fp4_warp(b.d_proj, b.d_attn, lw[l].o.d, lw[l].o.sc, Q, H, 0), "O");
            // Attention residual: proj += residual (but we need temp since vector_add overwrites)
            // vector_add(out, a, b) → out[i]=a[i]+b[i]
            // d_res = d_proj + d_res → but d_proj and d_res share? No, different buffers.
            // Actually: d_res (residual) + d_proj (output) → d_proj
            // But then d_res is unchanged, which is wrong for next iter.
            // Fix: use d_proj as first arg (overwrites with sum), then copy to d_res
            // ... or just use d_mlp as temp since I > H
            // Let me just fix it: vector_add(d_mlp, d_proj, d_res) → d_mlp = d_proj + d_res
            // Then copy d_mlp to d_res
            chk(blackwell::kernels::vector_add_fp32(b.d_mlp, b.d_proj, b.d_res, H, 0), "res_attn");
            cudaMemcpy(b.d_res, b.d_mlp, H*4, cudaMemcpyDeviceToDevice);

            // MLP
            chk(blackwell::kernels::fused_rmsnorm(b.d_proj, b.d_res, d_rn, H, 1e-6f, 0), "rms_mlp");
            chk(blackwell::kernels::gemv_fp32_fp4_warp(b.d_gate, b.d_proj, lw[l].g.d, lw[l].g.sc, H, I, 0), "gate");
            chk(blackwell::kernels::gemv_fp32_fp4_warp(b.d_up, b.d_proj, lw[l].u.d, lw[l].u.sc, H, I, 0), "up");
            chk(blackwell::kernels::apply_swiglu(b.d_mlp, b.d_gate, b.d_up, I, 0), "swiglu");
            chk(blackwell::kernels::gemv_fp32_fp4_warp(b.d_proj, b.d_mlp, lw[l].d.d, lw[l].d.sc, I, H, 0), "down");
            chk(blackwell::kernels::vector_add_fp32(b.d_mlp, b.d_proj, b.d_res, H, 0), "res_mlp");
            cudaMemcpy(b.d_res, b.d_mlp, H*4, cudaMemcpyDeviceToDevice);
        }
    }
    cudaDeviceSynchronize();

    // Baseline benchmark
    printf("Benchmarking per-kernel (%d tokens)...\n", bench);
    GpuTimer t0;
    t0.start();
    for (int i = 0; i < bench; ++i) {
        for (int l = 0; l < num_layers; ++l) {
            chk(blackwell::kernels::fused_rmsnorm(b.d_proj, b.d_res, d_rn, H, 1e-6f, 0), "rms_pre");
            chk(blackwell::kernels::gemv_fp32_fp4_warp(b.d_Q, b.d_proj, lw[l].q.d, lw[l].q.sc, H, Q, 0), "Q");
            chk(blackwell::kernels::gemv_fp32_fp4_warp(b.d_K, b.d_proj, lw[l].k.d, lw[l].k.sc, H, KV, 0), "K");
            chk(blackwell::kernels::gemv_fp32_fp4_warp(b.d_V, b.d_proj, lw[l].v.d, lw[l].v.sc, H, KV, 0), "V");
            int kb = l * nkv * ms * hd;
            chk(blackwell::kernels::update_kv_cache(d_kc+kb, d_vc+kb, b.d_K, b.d_V, 0, sq, nkv, hd, ms, 0), "kv");
            chk(blackwell::kernels::attention_decode_gqa(b.d_attn, b.d_Q, d_kc+kb, d_vc+kb,
                sq, nqh, nkv, hd, ms, 0), "attn");
            chk(blackwell::kernels::gemv_fp32_fp4_warp(b.d_proj, b.d_attn, lw[l].o.d, lw[l].o.sc, Q, H, 0), "O");
            chk(blackwell::kernels::vector_add_fp32(b.d_mlp, b.d_proj, b.d_res, H, 0), "res_attn");
            cudaMemcpy(b.d_res, b.d_mlp, H*4, cudaMemcpyDeviceToDevice);

            chk(blackwell::kernels::fused_rmsnorm(b.d_proj, b.d_res, d_rn, H, 1e-6f, 0), "rms_mlp");
            chk(blackwell::kernels::gemv_fp32_fp4_warp(b.d_gate, b.d_proj, lw[l].g.d, lw[l].g.sc, H, I, 0), "gate");
            chk(blackwell::kernels::gemv_fp32_fp4_warp(b.d_up, b.d_proj, lw[l].u.d, lw[l].u.sc, H, I, 0), "up");
            chk(blackwell::kernels::apply_swiglu(b.d_mlp, b.d_gate, b.d_up, I, 0), "swiglu");
            chk(blackwell::kernels::gemv_fp32_fp4_warp(b.d_proj, b.d_mlp, lw[l].d.d, lw[l].d.sc, I, H, 0), "down");
            chk(blackwell::kernels::vector_add_fp32(b.d_mlp, b.d_proj, b.d_res, H, 0), "res_mlp");
            cudaMemcpy(b.d_res, b.d_mlp, H*4, cudaMemcpyDeviceToDevice);
        }
    }
    float baseline_ms = t0.stop();
    float baseline_pt = baseline_ms / bench;
    float baseline_s28 = 1000.f / (baseline_pt * 28.f / num_layers);

    // Save per-kernel output
    float* d_tmp; cudaMalloc(&d_tmp, H*4);
    std::vector<float> per_kernel_out(H);
    cudaMemcpy(d_tmp, b.d_res, H*4, cudaMemcpyDeviceToDevice);
    cudaMemcpy(per_kernel_out.data(), d_tmp, H*4, cudaMemcpyDeviceToHost);

    // Restore initial state
    cudaMemcpy(b.d_res, d_res_save, H*4, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_kc, d_kc_save, kv_sz, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_vc, d_vc_save, kv_sz, cudaMemcpyDeviceToDevice);
    cudaFree(d_res_save); cudaFree(d_kc_save); cudaFree(d_vc_save);

    // ── CUDA Graph capture ──────────────────────────────────────────────────
    printf("\n=== CUDA Graph ===\n");
    cudaDeviceSynchronize();
    cudaError_t cerr = cudaPeekAtLastError();
    if (cerr != cudaSuccess) {
        printf("  Pre-capture error: %s — fixing...\n", cudaGetErrorString(cerr));
        cudaGetLastError();
    }

    cudaStream_t graph_stream;
    cudaStreamCreate(&graph_stream);

    // Pre-trigger attention_decode_gqa
    blackwell::kernels::attention_decode_gqa(
        b.d_attn, b.d_Q, d_kc, d_vc, sq, nqh, nkv, hd, ms, graph_stream);
    cudaStreamSynchronize(graph_stream);

    printf("  Capturing %d layers... ", num_layers);
    fflush(stdout);

    cudaStreamBeginCapture(graph_stream, cudaStreamCaptureModeGlobal);
    for (int l = 0; l < num_layers; ++l) {
        int kb = l * nkv * ms * hd;

        // Attention block (no chk during capture — cudaPeekAtLastError breaks graph capture)
        blackwell::kernels::fused_rmsnorm(b.d_proj, b.d_res, d_rn, H, 1e-6f, graph_stream);
        blackwell::kernels::gemv_fp32_fp4_warp(b.d_Q, b.d_proj, lw[l].q.d, lw[l].q.sc, H, Q, graph_stream);
        blackwell::kernels::gemv_fp32_fp4_warp(b.d_K, b.d_proj, lw[l].k.d, lw[l].k.sc, H, KV, graph_stream);
        blackwell::kernels::gemv_fp32_fp4_warp(b.d_V, b.d_proj, lw[l].v.d, lw[l].v.sc, H, KV, graph_stream);
        blackwell::kernels::update_kv_cache(d_kc+kb, d_vc+kb, b.d_K, b.d_V, 0, sq, nkv, hd, ms, graph_stream);
        blackwell::kernels::attention_decode_gqa(b.d_attn, b.d_Q, d_kc+kb, d_vc+kb,
            sq, nqh, nkv, hd, ms, graph_stream);
        blackwell::kernels::gemv_fp32_fp4_warp(b.d_proj, b.d_attn, lw[l].o.d, lw[l].o.sc, Q, H, graph_stream);
        blackwell::kernels::vector_add_fp32(b.d_mlp, b.d_proj, b.d_res, H, graph_stream);
        cudaMemcpyAsync(b.d_res, b.d_mlp, H*4, cudaMemcpyDeviceToDevice, graph_stream);

        // MLP block
        blackwell::kernels::fused_rmsnorm(b.d_proj, b.d_res, d_rn, H, 1e-6f, graph_stream);
        blackwell::kernels::gemv_fp32_fp4_warp(b.d_gate, b.d_proj, lw[l].g.d, lw[l].g.sc, H, I, graph_stream);
        blackwell::kernels::gemv_fp32_fp4_warp(b.d_up, b.d_proj, lw[l].u.d, lw[l].u.sc, H, I, graph_stream);
        blackwell::kernels::apply_swiglu(b.d_mlp, b.d_gate, b.d_up, I, graph_stream);
        blackwell::kernels::gemv_fp32_fp4_warp(b.d_proj, b.d_mlp, lw[l].d.d, lw[l].d.sc, I, H, graph_stream);
        blackwell::kernels::vector_add_fp32(b.d_mlp, b.d_proj, b.d_res, H, graph_stream);
        cudaMemcpyAsync(b.d_res, b.d_mlp, H*4, cudaMemcpyDeviceToDevice, graph_stream);
    }

    cudaGraph_t graph;
    cerr = cudaStreamEndCapture(graph_stream, &graph);
    if (cerr != cudaSuccess) {
        printf("FAIL: %s\n", cudaGetErrorString(cerr));
        return 1;
    }

    cudaGraphExec_t graph_exec;
    cerr = cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
    if (cerr != cudaSuccess) {
        printf("FAIL instantiate: %s\n", cudaGetErrorString(cerr));
        return 1;
    }
    printf("OK\n");

    // Graph warmup
    printf("  Graph warmup...\n");
    for (int i = 0; i < warm; ++i) cudaGraphLaunch(graph_exec, graph_stream);
    cudaStreamSynchronize(graph_stream);

    // Graph benchmark
    printf("  Graph benchmark (%d tokens)...\n", bench);
    GpuTimer tg;
    tg.start(graph_stream);
    for (int i = 0; i < bench; ++i) cudaGraphLaunch(graph_exec, graph_stream);
    cudaStreamSynchronize(graph_stream);
    float graph_ms = tg.stop();
    float graph_pt = graph_ms / bench;
    float graph_s28 = 1000.f / (graph_pt * 28.f / num_layers);

    // ── Correctness ───────────────────────────────────────────────────────
    std::vector<float> graph_out(H);
    cudaMemcpy(d_tmp, b.d_res, H*4, cudaMemcpyDeviceToDevice);
    cudaMemcpy(graph_out.data(), d_tmp, H*4, cudaMemcpyDeviceToHost);

    printf("\n=== Correctness Check ===\n");
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
        max_diff < 1e-3 ? "✅ MATCH" : max_diff < 0.1 ? "⚠️ CLOSE (FP4)" : "❌ MISMATCH");
    cudaFree(d_tmp);

    // ── Results ──────────────────────────────────────────────────────────────
    printf("\n=== FP4 Pipeline Results ===\n");
    printf("  %-20s  %8s  %8s  %8s\n", "Method", "Per-tok", "t/s", "Scaled28");
    printf("  %-20s  %7.3fms  %7.1f   %7.1f\n", "Per-kernel (FP4)",
        baseline_pt, 1000.f/baseline_pt, baseline_s28);
    printf("  %-20s  %7.3fms  %7.1f   %7.1f\n", "CUDA Graph (FP4)",
        graph_pt, 1000.f/graph_pt, graph_s28);
    printf("  Speedup: %.2fx (%.1f%%)\n",
        baseline_pt / graph_pt,
        (1.f - graph_pt/baseline_pt) * 100.f);
    printf("  vs llama.cpp Q4_K_M (253 t/s): %.1f%%\n", graph_s28 / 253.0f * 100);
    printf("  vs our INT8 CUDA Graph (174 t/s): %.1f%%\n", graph_s28 / 174.0f * 100);

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
    cudaFree(d_rn); cudaFree(d_kc); cudaFree(d_vc);
    cudaFree(b.d_Q); cudaFree(b.d_K); cudaFree(b.d_V);
    cudaFree(b.d_attn); cudaFree(b.d_proj);
    cudaFree(b.d_gate); cudaFree(b.d_up); cudaFree(b.d_mlp);
    cudaFree(b.d_res);

    return 0;
}
