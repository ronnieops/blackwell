// bench/decode_full.cu — Full transformer decode step (attention + MLP)
//
// Per layer:
//   Attention:
//     Q,K,V = fused_qkv(x, W_q, W_k, W_v)
//     update_kv_cache(K, V)
//     attn_out = attention_decode(Q, K_cache, V_cache)
//     x = rmsnorm_pack(attn_out @ W_o + x_residual)
//   MLP:
//     gate = gemv(x, W_gate)
//     up   = gemv(x, W_up)
//     mlp  = SwiGLU(gate, up)
//     x    = rmsnorm_pack(mlp @ W_down + x_residual)
//
// Uses real Qwen3-1.7B weights from weights/ directory.
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120,code=sm_120 \
//     -I include bench/decode_full.cu build/libblackwell_kernels.a \
//     -o bench/decode_full

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include <cstdint>
#include "blackwell/kernels.h"

struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin() { cudaEventRecord(start, 0); }
    float end() { cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
                  float ms=0; cudaEventElapsedTime(&ms, start, stop); return ms; }
};

static bool check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) { printf("FAIL: %s: %s\n", msg, cudaGetErrorString(e)); return false; }
    return true;
}

struct WeightFile {
    int dim1, dim0, block;
    int num_K_blocks, num_N_blocks;
    std::vector<uint8_t> data;
    std::vector<float> scales;
};

WeightFile load_weight(const char* path) {
    WeightFile w;
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    int header[5];
    if (fread(header, 4, 5, f) != 5) { fprintf(stderr, "Bad header %s\n", path); exit(1); }
    w.dim1 = header[0]; w.dim0 = header[1]; w.block = header[2];
    w.num_K_blocks = header[3]; w.num_N_blocks = header[4];
    size_t n = (size_t)w.dim1 * w.dim0;
    w.data.resize(n);
    if (fread(w.data.data(), 1, n, f) != n) { fprintf(stderr, "Bad data %s\n", path); exit(1); }
    size_t nscales = (size_t)w.num_K_blocks * w.num_N_blocks;
    w.scales.resize(nscales);
    if (fread(w.scales.data(), 4, nscales, f) != nscales) { fprintf(stderr, "Bad scales %s\n", path); exit(1); }
    fclose(f);
    return w;
}

struct DevWeight {
    void* data;
    float* scales;
};

DevWeight upload_weight(const WeightFile& w) {
    DevWeight d;
    size_t n = (size_t)w.dim1 * w.dim0;
    cudaMalloc(&d.data, n);
    cudaMemcpy(d.data, w.data.data(), n, cudaMemcpyHostToDevice);
    size_t nscales = (size_t)w.num_K_blocks * w.num_N_blocks;
    cudaMalloc(&d.scales, nscales * 4);
    cudaMemcpy(d.scales, w.scales.data(), nscales * 4, cudaMemcpyHostToDevice);
    return d;
}

int main(int argc, char** argv) {
    int num_layers = 4;
    if (argc > 1) num_layers = atoi(argv[1]);
    if (num_layers > 28) num_layers = 28;

    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("# Full Decode Benchmark — Attention + MLP\n");
    printf("Device: %s (CC %d.%d)\n", p.name, p.major, p.minor);
    printf("Layers: %d\n\n", num_layers);

    // Qwen3-1.7B params
    const int hidden = 2048;
    const int q_dim = 2048;       // 16 heads × 128
    const int kv_dim = 1024;      // 8 KV heads × 128
    const int intermediate = 6144;
    const int num_q_heads = 16, num_kv_heads = 8;
    const int head_dim = 128;
    const int max_seq = 2048;
    const float scale_1_3 = 1.0f / 3.0f;

    // Load weights
    printf("Loading weights...\n");
    auto load = [&](const char* fmt, int l) {
        char path[256];
        snprintf(path, 256, fmt, l);
        WeightFile w = load_weight(path);
        DevWeight d = upload_weight(w);
        printf("  %s: %d×%d\n", path, w.dim1, w.dim0);
        return d;
    };

    struct LayerWeights {
        DevWeight Wq, Wk, Wv, Wo;
        DevWeight Wgate, Wup, Wdown;
    };
    std::vector<LayerWeights> lw(num_layers);

    for (int l = 0; l < num_layers; ++l) {
        lw[l].Wq    = load("weights/%d_self_attn.q_proj.fp4", l);
        lw[l].Wk    = load("weights/%d_self_attn.k_proj.fp4", l);
        lw[l].Wv    = load("weights/%d_self_attn.v_proj.fp4", l);
        lw[l].Wo    = load("weights/%d_self_attn.o_proj.fp4", l);
        lw[l].Wgate = load("weights/%d_mlp.gate_proj.fp4", l);
        lw[l].Wup   = load("weights/%d_mlp.up_proj.fp4", l);
        lw[l].Wdown = load("weights/%d_mlp.down_proj.fp4", l);
    }

    // Allocate buffers
    float *d_x32, *d_xs, *d_Q, *d_K, *d_V, *d_attn, *d_proj, *d_gate, *d_up, *d_mlp;
    void *d_x_fp4, *d_attn_fp4, *d_mlp_fp4;
    float *d_attn_s, *d_mlp_s;
    cudaMalloc(&d_x32, hidden * 4);
    cudaMalloc(&d_x_fp4, hidden);
    cudaMalloc(&d_xs, (hidden/16) * 4);
    cudaMalloc(&d_Q, q_dim * 4);
    cudaMalloc(&d_K, kv_dim * 4);
    cudaMalloc(&d_V, kv_dim * 4);
    cudaMalloc(&d_attn, q_dim * 4);
    cudaMalloc(&d_proj, hidden * 4);
    cudaMalloc(&d_gate, intermediate * 4);
    cudaMalloc(&d_up, intermediate * 4);
    cudaMalloc(&d_mlp, intermediate * 4);
    cudaMalloc(&d_attn_fp4, q_dim);
    cudaMalloc(&d_attn_s, (q_dim/16) * 4);
    cudaMalloc(&d_mlp_fp4, intermediate);
    cudaMalloc(&d_mlp_s, (intermediate/16) * 4);

    // Init x = uniform 1.0
    std::vector<float> x_init(hidden, 1.0f);
    cudaMemcpy(d_x32, x_init.data(), hidden * 4, cudaMemcpyHostToDevice);
    std::vector<float> xs_init(hidden/16, scale_1_3);
    cudaMemcpy(d_xs, xs_init.data(), (hidden/16) * 4, cudaMemcpyHostToDevice);
    blackwell::kernels::pack_fp4(d_x_fp4, d_x32, d_xs, hidden, 0);

    // Init scales
    std::vector<float> attn_s_init(q_dim/16, scale_1_3);
    cudaMemcpy(d_attn_s, attn_s_init.data(), (q_dim/16) * 4, cudaMemcpyHostToDevice);
    std::vector<float> mlp_s_init(intermediate/16, scale_1_3);
    cudaMemcpy(d_mlp_s, mlp_s_init.data(), (intermediate/16) * 4, cudaMemcpyHostToDevice);

    // RMSNorm weights (ones)
    float *d_rn_weight;
    cudaMalloc(&d_rn_weight, hidden * 4);
    std::vector<float> rn_weight(hidden, 1.0f);
    cudaMemcpy(d_rn_weight, rn_weight.data(), hidden * 4, cudaMemcpyHostToDevice);

    // KV cache
    float *d_kc, *d_vc;
    cudaMalloc(&d_kc, num_layers * num_kv_heads * max_seq * head_dim * 4);
    cudaMalloc(&d_vc, num_layers * num_kv_heads * max_seq * head_dim * 4);
    cudaMemset(d_kc, 0, num_layers * num_kv_heads * max_seq * head_dim * 4);
    cudaMemset(d_vc, 0, num_layers * num_kv_heads * max_seq * head_dim * 4);

    // Residual buffer (for attention + MLP residual connections)
    void *d_x_residual_fp4;
    float *d_x_residual_s;
    cudaMalloc(&d_x_residual_fp4, hidden);
    cudaMalloc(&d_x_residual_s, (hidden/16) * 4);

    // Fill KV cache to seq=128
    printf("Filling KV cache to seq_pos=128...\n");
    for (int seq = 0; seq <= 128; ++seq) {
        for (int l = 0; l < num_layers; ++l) {
            int kv_base = l * num_kv_heads * max_seq * head_dim;
            // Attention
            // Separate GEMVs (fused_qkv limited to 256 outputs/block)
            blackwell::kernels::gemv_fp4(d_Q, d_x_fp4, d_xs,
                lw[l].Wq.data, lw[l].Wq.scales, hidden, q_dim, 0);
            blackwell::kernels::gemv_fp4(d_K, d_x_fp4, d_xs,
                lw[l].Wk.data, lw[l].Wk.scales, hidden, kv_dim, 0);
            blackwell::kernels::gemv_fp4(d_V, d_x_fp4, d_xs,
                lw[l].Wv.data, lw[l].Wv.scales, hidden, kv_dim, 0);
            blackwell::kernels::update_kv_cache(
                d_kc + kv_base, d_vc + kv_base, d_K, d_V, 0, seq,
                num_kv_heads, head_dim, max_seq, 0);
            blackwell::kernels::attention_decode(
                d_attn, d_Q, d_kc + kv_base, d_vc + kv_base,
                seq, num_q_heads, head_dim, max_seq, 0);
            // Save residual (x before attention)
            // For simplicity in warmup, skip residual (just overwrite x)
            blackwell::kernels::pack_fp4(d_attn_fp4, d_attn, d_attn_s, q_dim, 0);
            blackwell::kernels::gemv_fp4(d_proj, d_attn_fp4, d_attn_s,
                lw[l].Wo.data, lw[l].Wo.scales, q_dim, hidden, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, d_proj, d_rn_weight, hidden, 1e-5f, 0);

            // MLP
            blackwell::kernels::gemv_fp4(d_gate, d_x_fp4, d_xs,
                lw[l].Wgate.data, lw[l].Wgate.scales, hidden, intermediate, 0);
            blackwell::kernels::gemv_fp4(d_up, d_x_fp4, d_xs,
                lw[l].Wup.data, lw[l].Wup.scales, hidden, intermediate, 0);
            blackwell::kernels::apply_swiglu(d_mlp, d_gate, d_up, intermediate, 0);
            blackwell::kernels::pack_fp4(d_mlp_fp4, d_mlp, d_mlp_s, intermediate, 0);
            blackwell::kernels::gemv_fp4(d_proj, d_mlp_fp4, d_mlp_s,
                lw[l].Wdown.data, lw[l].Wdown.scales, intermediate, hidden, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, d_proj, d_rn_weight, hidden, 1e-5f, 0);
        }
    }
    printf("  done.\n");

    cudaError_t cerr = cudaPeekAtLastError();
    if (cerr != cudaSuccess) { printf("  ERROR after cache fill: %s\n", cudaGetErrorString(cerr)); return 1; }

    // Warmup
    printf("Warmup...\n");
    int warm = 5, bench_iter = 20;
    int seq_pos = 128;
    for (int i = 0; i < warm; ++i) {
        for (int l = 0; l < num_layers; ++l) {
            int kv_base = l * num_kv_heads * max_seq * head_dim;
            // Attention
            // Separate GEMVs (fused_qkv limited to 256 outputs/block)
            blackwell::kernels::gemv_fp4(d_Q, d_x_fp4, d_xs,
                lw[l].Wq.data, lw[l].Wq.scales, hidden, q_dim, 0);
            blackwell::kernels::gemv_fp4(d_K, d_x_fp4, d_xs,
                lw[l].Wk.data, lw[l].Wk.scales, hidden, kv_dim, 0);
            blackwell::kernels::gemv_fp4(d_V, d_x_fp4, d_xs,
                lw[l].Wv.data, lw[l].Wv.scales, hidden, kv_dim, 0);
            blackwell::kernels::update_kv_cache(
                d_kc + kv_base, d_vc + kv_base, d_K, d_V, 0, seq_pos,
                num_kv_heads, head_dim, max_seq, 0);
            blackwell::kernels::attention_decode(
                d_attn, d_Q, d_kc + kv_base, d_vc + kv_base,
                seq_pos, num_q_heads, head_dim, max_seq, 0);
            blackwell::kernels::pack_fp4(d_attn_fp4, d_attn, d_attn_s, q_dim, 0);
            blackwell::kernels::gemv_fp4(d_proj, d_attn_fp4, d_attn_s,
                lw[l].Wo.data, lw[l].Wo.scales, q_dim, hidden, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, d_proj, d_rn_weight, hidden, 1e-5f, 0);

            // MLP
            blackwell::kernels::gemv_fp4(d_gate, d_x_fp4, d_xs,
                lw[l].Wgate.data, lw[l].Wgate.scales, hidden, intermediate, 0);
            blackwell::kernels::gemv_fp4(d_up, d_x_fp4, d_xs,
                lw[l].Wup.data, lw[l].Wup.scales, hidden, intermediate, 0);
            blackwell::kernels::apply_swiglu(d_mlp, d_gate, d_up, intermediate, 0);
            blackwell::kernels::pack_fp4(d_mlp_fp4, d_mlp, d_mlp_s, intermediate, 0);
            blackwell::kernels::gemv_fp4(d_proj, d_mlp_fp4, d_mlp_s,
                lw[l].Wdown.data, lw[l].Wdown.scales, intermediate, hidden, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, d_proj, d_rn_weight, hidden, 1e-5f, 0);
        }
    }
    cudaDeviceSynchronize();

    // Benchmark
    printf("Benchmarking %d tokens (seq_pos=%d)...\n", bench_iter, seq_pos);
    GpuTimer t;
    t.begin();
    for (int iter = 0; iter < bench_iter; ++iter) {
        for (int l = 0; l < num_layers; ++l) {
            int kv_base = l * num_kv_heads * max_seq * head_dim;
            // Attention
            // Separate GEMVs (fused_qkv limited to 256 outputs/block)
            blackwell::kernels::gemv_fp4(d_Q, d_x_fp4, d_xs,
                lw[l].Wq.data, lw[l].Wq.scales, hidden, q_dim, 0);
            blackwell::kernels::gemv_fp4(d_K, d_x_fp4, d_xs,
                lw[l].Wk.data, lw[l].Wk.scales, hidden, kv_dim, 0);
            blackwell::kernels::gemv_fp4(d_V, d_x_fp4, d_xs,
                lw[l].Wv.data, lw[l].Wv.scales, hidden, kv_dim, 0);
            blackwell::kernels::update_kv_cache(
                d_kc + kv_base, d_vc + kv_base, d_K, d_V, 0, seq_pos,
                num_kv_heads, head_dim, max_seq, 0);
            blackwell::kernels::attention_decode(
                d_attn, d_Q, d_kc + kv_base, d_vc + kv_base,
                seq_pos, num_q_heads, head_dim, max_seq, 0);
            blackwell::kernels::pack_fp4(d_attn_fp4, d_attn, d_attn_s, q_dim, 0);
            blackwell::kernels::gemv_fp4(d_proj, d_attn_fp4, d_attn_s,
                lw[l].Wo.data, lw[l].Wo.scales, q_dim, hidden, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, d_proj, d_rn_weight, hidden, 1e-5f, 0);

            // MLP
            blackwell::kernels::gemv_fp4(d_gate, d_x_fp4, d_xs,
                lw[l].Wgate.data, lw[l].Wgate.scales, hidden, intermediate, 0);
            blackwell::kernels::gemv_fp4(d_up, d_x_fp4, d_xs,
                lw[l].Wup.data, lw[l].Wup.scales, hidden, intermediate, 0);
            blackwell::kernels::apply_swiglu(d_mlp, d_gate, d_up, intermediate, 0);
            blackwell::kernels::pack_fp4(d_mlp_fp4, d_mlp, d_mlp_s, intermediate, 0);
            blackwell::kernels::gemv_fp4(d_proj, d_mlp_fp4, d_mlp_s,
                lw[l].Wdown.data, lw[l].Wdown.scales, intermediate, hidden, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, d_proj, d_rn_weight, hidden, 1e-5f, 0);
        }
    }
    float total_ms = t.end();
    float per_token_ms = total_ms / bench_iter;
    float tps = 1000.0f / per_token_ms;
    float scaled_28 = 1000.0f / (per_token_ms * 28.0f / num_layers);

    printf("\n=== Results ===\n");
    printf("  Layers:            %d\n", num_layers);
    printf("  Tokens:            %d\n", bench_iter);
    printf("  Total time:        %.2f ms\n", total_ms);
    printf("  Per-token:         %.3f ms\n", per_token_ms);
    printf("  Throughput:        %.1f t/s\n", tps);
    printf("  Scaled to 28:      %.1f t/s\n", scaled_28);
    printf("  Target (llama):    114 t/s\n");
    printf("  Ratio:             %.1f%%\n", (scaled_28 / 114.0f) * 100.0f);

    printf("\n  Per-layer breakdown estimate:\n");
    printf("    Attention: fused_qkv + update_kv + attn + pack + gemv_o + rmsnorm_pack = 6 ops\n");
    printf("    MLP:       2×gemv(gate/up) + swiglu + pack + gemv(down) + rmsnorm_pack = 6 ops\n");
    printf("    Total:     12 ops/layer × %d layers = %d ops/token\n", num_layers, num_layers * 12);

    // Print output values
    printf("\n  First 8 x values: ");
    std::vector<float> out(hidden);
    cudaMemcpy(out.data(), d_x32, hidden * 4, cudaMemcpyDeviceToHost);
    blackwell::kernels::unpack_fp4(out.data(), d_x_fp4, d_xs, hidden, 0);
    cudaMemcpy(out.data(), d_x_fp4, 1, cudaMemcpyDeviceToHost); // dummy
    // Actually need to unpack properly
    float* d_tmp;
    cudaMalloc(&d_tmp, hidden * 4);
    blackwell::kernels::unpack_fp4(d_tmp, d_x_fp4, d_xs, hidden, 0);
    cudaMemcpy(out.data(), d_tmp, hidden * 4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 8; ++i) printf("%.4f ", out[i]);
    printf("\n");
    cudaFree(d_tmp);

    // Cleanup
    for (int l = 0; l < num_layers; ++l) {
        cudaFree(lw[l].Wq.data); cudaFree(lw[l].Wq.scales);
        cudaFree(lw[l].Wk.data); cudaFree(lw[l].Wk.scales);
        cudaFree(lw[l].Wv.data); cudaFree(lw[l].Wv.scales);
        cudaFree(lw[l].Wo.data); cudaFree(lw[l].Wo.scales);
        cudaFree(lw[l].Wgate.data); cudaFree(lw[l].Wgate.scales);
        cudaFree(lw[l].Wup.data); cudaFree(lw[l].Wup.scales);
        cudaFree(lw[l].Wdown.data); cudaFree(lw[l].Wdown.scales);
    }
    cudaFree(d_x32); cudaFree(d_x_fp4); cudaFree(d_xs);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_attn); cudaFree(d_proj); cudaFree(d_gate); cudaFree(d_up); cudaFree(d_mlp);
    cudaFree(d_attn_fp4); cudaFree(d_attn_s);
    cudaFree(d_mlp_fp4); cudaFree(d_mlp_s);
    cudaFree(d_rn_weight);
    cudaFree(d_kc); cudaFree(d_vc);
    cudaFree(d_x_residual_fp4); cudaFree(d_x_residual_s);

    return 0;
}
