// bench/decode_real.cu - Decode benchmark with real Qwen3-1.7B weights
// Loads FP4 weights extracted by tools/extract_weights.cpp
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

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

int main(int argc, char** argv) {
    int num_layers = 4;
    if (argc > 1) num_layers = atoi(argv[1]);
    if (num_layers > 28) num_layers = 28;

    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("# Decode Benchmark - Real Qwen3-1.7B Weights\n");
    printf("Device: %s (CC %d.%d)\n", p.name, p.major, p.minor);
    printf("Layers: %d\n\n", num_layers);

    const int hidden = 2048;
    const int q_dim = 2048;
    const int kv_dim = 1024;
    const int num_q_heads = 16, num_kv_heads = 8;
    const int head_dim = 128;
    const int max_seq = 2048;
    const float scale_one_third = 1.0f / 3.0f;

    // Load weights
    printf("Loading weights...\n");
    std::vector<WeightFile> W_q(num_layers), W_k(num_layers), W_v(num_layers), W_o(num_layers);

    for (int l = 0; l < num_layers; ++l) {
        char path[256];
        snprintf(path, 256, "/mnt/data/dev/projects/blackwell/weights/%d_self_attn.q_proj.fp4", l);
        W_q[l] = load_weight(path);
        snprintf(path, 256, "/mnt/data/dev/projects/blackwell/weights/%d_self_attn.k_proj.fp4", l);
        W_k[l] = load_weight(path);
        snprintf(path, 256, "/mnt/data/dev/projects/blackwell/weights/%d_self_attn.v_proj.fp4", l);
        W_v[l] = load_weight(path);
        snprintf(path, 256, "/mnt/data/dev/projects/blackwell/weights/%d_self_attn.o_proj.fp4", l);
        W_o[l] = load_weight(path);
        printf("  Layer %2d: q=%dx%d k=%dx%d v=%dx%d o=%dx%d\n", l,
               W_q[l].dim1, W_q[l].dim0,
               W_k[l].dim1, W_k[l].dim0,
               W_v[l].dim1, W_v[l].dim0,
               W_o[l].dim1, W_o[l].dim0);
    }

    // Upload weights to GPU
    printf("\nUploading to GPU...\n");
    std::vector<void*> d_Wq(num_layers), d_Wk(num_layers), d_Wv(num_layers), d_Wo(num_layers);
    std::vector<float*> d_Wqs(num_layers), d_Wks(num_layers), d_Wvs(num_layers), d_Wos(num_layers);

    for (int l = 0; l < num_layers; ++l) {
        auto upload = [](const WeightFile& w, void*& d_data, float*& d_scales) {
            size_t n = (size_t)w.dim1 * w.dim0;
            cudaMalloc(&d_data, n);
            cudaMemcpy(d_data, w.data.data(), n, cudaMemcpyHostToDevice);

            size_t nscales = (size_t)w.num_K_blocks * w.num_N_blocks;
            cudaMalloc(&d_scales, nscales * 4);
            cudaMemcpy(d_scales, w.scales.data(), nscales * 4, cudaMemcpyHostToDevice);
        };
        upload(W_q[l], d_Wq[l], d_Wqs[l]);
        upload(W_k[l], d_Wk[l], d_Wks[l]);
        upload(W_v[l], d_Wv[l], d_Wvs[l]);
        upload(W_o[l], d_Wo[l], d_Wos[l]);
    }

    // Allocate buffers
    float *d_x32, *d_x_fp4, *d_xs, *d_Q, *d_K, *d_V, *d_attn, *d_proj, *d_attn_fp4, *d_attn_s;
    void *d_attn_fp4_v, *d_proj_fp4_v;
    cudaMalloc(&d_x32, hidden * 4);
    cudaMalloc(&d_x_fp4, hidden);
    cudaMalloc(&d_xs, (hidden/16) * 4);
    cudaMalloc(&d_Q, q_dim * 4);
    cudaMalloc(&d_K, kv_dim * 4);
    cudaMalloc(&d_V, kv_dim * 4);
    cudaMalloc(&d_attn, q_dim * 4);
    cudaMalloc(&d_proj, hidden * 4);
    cudaMalloc(&d_attn_fp4_v, q_dim);
    cudaMalloc(&d_attn_s, (q_dim/16) * 4);
    cudaMalloc(&d_proj_fp4_v, hidden);

    // Init x = uniform 1.0
    std::vector<float> x_init(hidden, 1.0f);
    cudaMemcpy(d_x32, x_init.data(), hidden * 4, cudaMemcpyHostToDevice);
    std::vector<float> xs_init(hidden/16, scale_one_third);
    cudaMemcpy(d_xs, xs_init.data(), (hidden/16) * 4, cudaMemcpyHostToDevice);
    blackwell::kernels::pack_fp4(d_x_fp4, d_x32, d_xs, hidden, 0);

    // Init attn scales
    std::vector<float> attn_scale_init(q_dim/16, scale_one_third);
    cudaMemcpy(d_attn_s, attn_scale_init.data(), (q_dim/16) * 4, cudaMemcpyHostToDevice);
    
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

    // Populate KV cache with real projected K/V values
    printf("\nFilling KV cache to seq_pos=128...\n");
    for (int seq = 0; seq <= 128; ++seq) {
        for (int l = 0; l < num_layers; ++l) {
            int kv_base = l * num_kv_heads * max_seq * head_dim;
            blackwell::kernels::fused_qkv_gemv(d_Q, d_K, d_V,
                d_x_fp4, d_xs, d_Wq[l], d_Wqs[l], d_Wk[l], d_Wks[l], d_Wv[l], d_Wvs[l],
                hidden, q_dim, kv_dim, 0);
            blackwell::kernels::update_kv_cache(
                d_kc + kv_base, d_vc + kv_base, d_K, d_V, 0, seq,
                num_kv_heads, head_dim, max_seq, 0);
            blackwell::kernels::attention_decode(
                d_attn, d_Q, d_kc + kv_base, d_vc + kv_base,
                seq, num_q_heads, head_dim, max_seq, 0);
            blackwell::kernels::pack_fp4(d_attn_fp4_v, d_attn, d_attn_s, q_dim, 0);
            blackwell::kernels::gemv_fp4(d_proj, d_attn_fp4_v, d_attn_s,
                d_Wo[l], d_Wos[l], q_dim, hidden, 0);
            // RMSNorm
            blackwell::kernels::fused_rmsnorm(d_x32, d_proj, d_rn_weight, hidden, 1e-5f, 0);
            blackwell::kernels::pack_fp4(d_x_fp4, d_x32, d_xs, hidden, 0);
        }
        if (seq % 32 == 0) { printf("  seq=%d\r", seq); fflush(stdout); }
    }
    printf("  done.\n");
    
    // Check for errors
    cudaError_t cerr = cudaPeekAtLastError();
    if (cerr != cudaSuccess) { printf("  ERROR: %s\n", cudaGetErrorString(cerr)); return 1; }

    // Benchmark
    int warm = 5, bench = 20;
    int seq_pos = 128;
    
    for (int i = 0; i < warm; ++i) {
        for (int l = 0; l < num_layers; ++l) {
            int kv_base = l * num_kv_heads * max_seq * head_dim;
            blackwell::kernels::fused_qkv_gemv(d_Q, d_K, d_V,
                d_x_fp4, d_xs, d_Wq[l], d_Wqs[l], d_Wk[l], d_Wks[l], d_Wv[l], d_Wvs[l],
                hidden, q_dim, kv_dim, 0);
            blackwell::kernels::update_kv_cache(
                d_kc + kv_base, d_vc + kv_base, d_K, d_V, 0, seq_pos,
                num_kv_heads, head_dim, max_seq, 0);
            blackwell::kernels::attention_decode(
                d_attn, d_Q, d_kc + kv_base, d_vc + kv_base,
                seq_pos, num_q_heads, head_dim, max_seq, 0);
            blackwell::kernels::pack_fp4(d_attn_fp4_v, d_attn, d_attn_s, q_dim, 0);
            blackwell::kernels::gemv_fp4(d_proj, d_attn_fp4_v, d_attn_s,
                d_Wo[l], d_Wos[l], q_dim, hidden, 0);
            blackwell::kernels::fused_rmsnorm(d_x32, d_proj, d_rn_weight, hidden, 1e-5f, 0);
            blackwell::kernels::pack_fp4(d_x_fp4, d_x32, d_xs, hidden, 0);
        }
    }
    cudaDeviceSynchronize();
    cerr = cudaPeekAtLastError();
    if (cerr != cudaSuccess) { printf("  WARM ERROR: %s\n", cudaGetErrorString(cerr)); return 1; }

    printf("\nBenchmarking %d tokens (seq_pos=%d)...\n", bench, seq_pos);
    fflush(stdout);

    GpuTimer t;
    t.begin();
    for (int iter = 0; iter < bench; ++iter) {
        seq_pos = 128 + iter;
        for (int l = 0; l < num_layers; ++l) {
            int kv_base = l * num_kv_heads * max_seq * head_dim;
            blackwell::kernels::fused_qkv_gemv(d_Q, d_K, d_V,
                d_x_fp4, d_xs, d_Wq[l], d_Wqs[l], d_Wk[l], d_Wks[l], d_Wv[l], d_Wvs[l],
                hidden, q_dim, kv_dim, 0);
            blackwell::kernels::update_kv_cache(
                d_kc + kv_base, d_vc + kv_base, d_K, d_V, 0, seq_pos,
                num_kv_heads, head_dim, max_seq, 0);
            blackwell::kernels::attention_decode(
                d_attn, d_Q, d_kc + kv_base, d_vc + kv_base,
                seq_pos, num_q_heads, head_dim, max_seq, 0);
            blackwell::kernels::pack_fp4(d_attn_fp4_v, d_attn, d_attn_s, q_dim, 0);
            blackwell::kernels::gemv_fp4(d_proj, d_attn_fp4_v, d_attn_s,
                d_Wo[l], d_Wos[l], q_dim, hidden, 0);
            blackwell::kernels::fused_rmsnorm(d_x32, d_proj, d_rn_weight, hidden, 1e-5f, 0);
            blackwell::kernels::pack_fp4(d_x_fp4, d_x32, d_xs, hidden, 0);
        }
    }
    float total_ms = t.end();
    float per_token_ms = total_ms / bench;
    float tps = 1000.0f / per_token_ms;
    float scaled = tps * (28.0f / num_layers);

    printf("\n=== Results ===\n");
    printf("  Layers:            %d\n", num_layers);
    printf("  Seq pos:           ~128-148\n");
    printf("  Tokens:            %d\n", bench);
    printf("  Total time:        %.2f ms\n", total_ms);
    printf("  Per-token:         %.3f ms\n", per_token_ms);
    printf("  Throughput:        %.1f t/s\n", tps);
    printf("  Scaled to 28:      %.1f t/s\n", scaled);
    printf("  Target (llama):    114 t/s\n");
    printf("  Ratio to target:   %.1f%%\n", (scaled / 114.0f) * 100.0f);

    // Compare output to synthetic baseline
    printf("\n  First 8 x values: ");
    std::vector<float> out(hidden);
    cudaMemcpy(out.data(), d_x32, hidden * 4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 8; ++i) printf("%.4f ", out[i]);
    printf("\n");

    // Cleanup
    for (int l = 0; l < num_layers; ++l) {
        cudaFree(d_Wq[l]); cudaFree(d_Wqs[l]);
        cudaFree(d_Wk[l]); cudaFree(d_Wks[l]);
        cudaFree(d_Wv[l]); cudaFree(d_Wvs[l]);
        cudaFree(d_Wo[l]); cudaFree(d_Wos[l]);
    }
    cudaFree(d_x32); cudaFree(d_x_fp4); cudaFree(d_xs);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_attn); cudaFree(d_proj);
    cudaFree(d_attn_fp4_v); cudaFree(d_attn_s); cudaFree(d_proj_fp4_v);
    cudaFree(d_kc); cudaFree(d_vc);

    return 0;
}
