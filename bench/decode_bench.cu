// bench/decode_bench.cu — End-to-end single-token decode benchmark
//
// Models a simplified transformer decode step:
//   For each layer:
//     Q = x @ W_q  (GEMV, FP4 weights, FP4 input)
//     K = x @ W_k
//     V = x @ W_v
//     update_kv_cache(K, V, seq_pos)   (K/V dequantized → FP32 cache)
//     attn_out = attention_decode(Q, K_cache, V_cache, seq_pos)
//     out = attn_out @ W_o
//     out = rmsnorm(out) (simplified: no residual for bench)
//     x = out
//
// Hidden dim = 2048, 16 Q heads, 4 KV heads, head_dim = 128
// FP4 block-scaled weights and input, FP32 KV cache

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
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
    if (e != cudaSuccess) {
        printf("FAIL: %s: %s\n", msg, cudaGetErrorString(e));
        return false;
    }
    return true;
}

// Qwen3-1.7B parameters
constexpr int kHiddenDim    = 2048;
constexpr int kNumQHeads    = 16;
constexpr int kNumKVHeads   = 4;
constexpr int kHeadDim      = 128;
constexpr int kMaxSeqLen    = 2048;
constexpr int kNumLayers    = 4;
constexpr int kFP4BlockSize = 16;
constexpr float kScaleOneThird = 1.0f / 3.0f;  // uniform 1.0 input → scale

struct DeviceBuffers {
    float *x_fp32;       // hidden state (hidden_dim) FP32
    void  *x_fp4;        // hidden state packed to FP4
    float *x_scale;      // per-block scales for x_fp4

    float *Q, *K, *V;    // projections (FP32 for attention: Q: 2048, K/V: 512)
    float *attn_out;     // attention output (2048)
    float *proj_out;     // after output projection (2048)

    // Per-layer KV cache (all layers combined for simplicity)
    float *k_cache;      // [layers * kv_heads * max_seq * head_dim]
    float *v_cache;      // same
};

int main(int argc, char** argv) {
    int kNumLayers = 4;
    if (argc > 1) {
        kNumLayers = atoi(argv[1]);
        if (kNumLayers < 1) kNumLayers = 1;
        if (kNumLayers > 28) kNumLayers = 28;
    }
    constexpr int kAllocLayers = 4;  // Always allocate for 4 layers

    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("# Decode Benchmark (End-to-End)\n");
    printf("Device: %s (CC %d.%d)\n", p.name, p.major, p.minor);
    printf("Config: hidden=%d, Qheads=%d, KVheads=%d, head_dim=%d, layers=%d (alloc=%d), max_seq=%d\n\n",
           kHiddenDim, kNumQHeads, kNumKVHeads, kHeadDim, kNumLayers, kAllocLayers, kMaxSeqLen);

    int warmup = 3, bench = 20;

    // All weights: W_q (2048×2048), W_k (2048×512), W_v (2048×512), W_o (2048×2048)
    int q_dim = kNumQHeads * kHeadDim;
    int kv_dim = kNumKVHeads * kHeadDim;
    int num_w_blks_q = (kHiddenDim / kFP4BlockSize) * (q_dim / kFP4BlockSize);
    int num_w_blks_kv = (kHiddenDim / kFP4BlockSize) * (kv_dim / kFP4BlockSize);
    int num_w_blks_o = (q_dim / kFP4BlockSize) * (kHiddenDim / kFP4BlockSize);

    // Init buffers
    DeviceBuffers b;
    cudaMalloc(&b.x_fp32, kHiddenDim * 4);
    cudaMalloc(&b.x_fp4, kHiddenDim);  // sizeof(__nv_fp4_e2m1)=1
    cudaMalloc(&b.x_scale, (kHiddenDim / kFP4BlockSize) * 4);
    cudaMalloc(&b.Q, q_dim * 4);
    cudaMalloc(&b.K, kv_dim * 4);
    cudaMalloc(&b.V, kv_dim * 4);
    cudaMalloc(&b.attn_out, q_dim * 4);
    cudaMalloc(&b.proj_out, kHiddenDim * 4);

    cudaMalloc(&b.k_cache, kAllocLayers * kNumKVHeads * kMaxSeqLen * kHeadDim * 4);
    cudaMalloc(&b.v_cache, kAllocLayers * kNumKVHeads * kMaxSeqLen * kHeadDim * 4);
    cudaMemset(b.k_cache, 0, kAllocLayers * kNumKVHeads * kMaxSeqLen * kHeadDim * 4);
    cudaMemset(b.v_cache, 0, kAllocLayers * kNumKVHeads * kMaxSeqLen * kHeadDim * 4);

    // Initialize x to uniform 1.0
    std::vector<float> x_init(kHiddenDim, 1.0f);
    cudaMemcpy(b.x_fp32, x_init.data(), kHiddenDim * 4, cudaMemcpyHostToDevice);
    std::vector<float> x_scales(kHiddenDim / kFP4BlockSize, kScaleOneThird);
    cudaMemcpy(b.x_scale, x_scales.data(), (kHiddenDim / kFP4BlockSize) * 4, cudaMemcpyHostToDevice);

    // Allocate per-layer weight arrays (always alloc 4 layers)
    std::vector<void*> W_q_fp4(kAllocLayers), W_k_fp4(kAllocLayers), W_v_fp4(kAllocLayers), W_o_fp4(kAllocLayers);
    std::vector<float*> W_q_scale(kAllocLayers), W_k_scale(kAllocLayers), W_v_scale(kAllocLayers), W_o_scale(kAllocLayers);
    std::vector<float*> rmsnorm_w(kAllocLayers);

    std::vector<float> all_scales(num_w_blks_q, kScaleOneThird);
    std::vector<float> all_one(kHiddenDim, 1.0f);

    for (int l = 0; l < kNumLayers; ++l) {
        // W_q
        cudaMalloc(&W_q_fp4[l], kHiddenDim * q_dim);
        cudaMalloc(&W_q_scale[l], num_w_blks_q * 4);
        cudaMemcpy(W_q_scale[l], all_scales.data(), num_w_blks_q * 4, cudaMemcpyHostToDevice);
        float *tmp;
        cudaMalloc(&tmp, kHiddenDim * q_dim * 4);
        cudaMemcpy(tmp, all_one.data(), kHiddenDim * 4, cudaMemcpyHostToDevice);
        // broadcast: fill with 1.0
        for (int r = 1; r < q_dim; ++r)
            cudaMemcpy(tmp + r * kHiddenDim, tmp, kHiddenDim * 4, cudaMemcpyHostToDevice);
        check(blackwell::kernels::pack_fp4(W_q_fp4[l], tmp, W_q_scale[l], kHiddenDim * q_dim, 0), "pack W_q");
        cudaFree(tmp);

        // W_k (simpler: use init_fp4_weights inline)
        cudaMalloc(&W_k_fp4[l], kHiddenDim * kv_dim);
        cudaMalloc(&W_k_scale[l], num_w_blks_kv * 4);
        cudaMemcpy(W_k_scale[l], all_scales.data(), num_w_blks_kv * 4, cudaMemcpyHostToDevice);
        cudaMalloc(&tmp, kHiddenDim * kv_dim * 4);
        std::vector<float> ones_h(kHiddenDim * kv_dim, 1.0f);
        cudaMemcpy(tmp, ones_h.data(), kHiddenDim * kv_dim * 4, cudaMemcpyHostToDevice);
        check(blackwell::kernels::pack_fp4(W_k_fp4[l], tmp, W_k_scale[l], kHiddenDim * kv_dim, 0), "pack W_k");
        cudaFree(tmp);

        // W_v
        cudaMalloc(&W_v_fp4[l], kHiddenDim * kv_dim);
        cudaMalloc(&W_v_scale[l], num_w_blks_kv * 4);
        cudaMemcpy(W_v_scale[l], all_scales.data(), num_w_blks_kv * 4, cudaMemcpyHostToDevice);
        cudaMalloc(&tmp, kHiddenDim * kv_dim * 4);
        cudaMemcpy(tmp, ones_h.data(), kHiddenDim * kv_dim * 4, cudaMemcpyHostToDevice);
        check(blackwell::kernels::pack_fp4(W_v_fp4[l], tmp, W_v_scale[l], kHiddenDim * kv_dim, 0), "pack W_v");
        cudaFree(tmp);

        // W_o: q_dim × hidden
        cudaMalloc(&W_o_fp4[l], q_dim * kHiddenDim);
        cudaMalloc(&W_o_scale[l], num_w_blks_o * 4);
        cudaMemcpy(W_o_scale[l], all_scales.data(), num_w_blks_o * 4, cudaMemcpyHostToDevice);
        cudaMalloc(&tmp, q_dim * kHiddenDim * 4);
        std::vector<float> ones_oh(q_dim * kHiddenDim, 1.0f);
        cudaMemcpy(tmp, ones_oh.data(), q_dim * kHiddenDim * 4, cudaMemcpyHostToDevice);
        check(blackwell::kernels::pack_fp4(W_o_fp4[l], tmp, W_o_scale[l], q_dim * kHiddenDim, 0), "pack W_o");
        cudaFree(tmp);

        // RMSNorm weights
        cudaMalloc(&rmsnorm_w[l], kHiddenDim * 4);
        std::vector<float> rn(kHiddenDim, 1.0f);
        cudaMemcpy(rmsnorm_w[l], rn.data(), kHiddenDim * 4, cudaMemcpyHostToDevice);
    }

    // Pack initial x to FP4
    check(blackwell::kernels::pack_fp4(b.x_fp4, b.x_fp32, b.x_scale, kHiddenDim, 0), "pack x init");

    // Scratch FP4 buffer for attn_out
    void *attn_fp4;
    float *attn_scale;
    cudaMalloc(&attn_fp4, q_dim);
    cudaMalloc(&attn_scale, (q_dim / kFP4BlockSize) * 4);
    std::vector<float> attn_scales_h(q_dim / kFP4BlockSize, kScaleOneThird);
    cudaMemcpy(attn_scale, attn_scales_h.data(), (q_dim / kFP4BlockSize) * 4, cudaMemcpyHostToDevice);

    // Warmup
    printf("Warmup %d tokens... ", warmup);
    fflush(stdout);
    for (int iter = 0; iter < warmup; ++iter) {
        int seq_pos = iter % kMaxSeqLen;

        for (int l = 0; l < kNumLayers; ++l) {
            int kvcache_base = l * kNumKVHeads * kMaxSeqLen * kHeadDim;

            check(blackwell::kernels::fused_qkv_gemv(b.Q, b.K, b.V,
                b.x_fp4, b.x_scale,
                W_q_fp4[l], W_q_scale[l],
                W_k_fp4[l], W_k_scale[l],
                W_v_fp4[l], W_v_scale[l],
                kHiddenDim, q_dim, kv_dim, 0), "fused_qkv");
            check(blackwell::kernels::update_kv_cache(
                b.k_cache + kvcache_base, b.v_cache + kvcache_base,
                b.K, b.V, 0, seq_pos,
                kNumKVHeads, kHeadDim, kMaxSeqLen, 0), "update_kv");
            check(blackwell::kernels::attention_decode(
                b.attn_out, b.Q,
                b.k_cache + kvcache_base, b.v_cache + kvcache_base,
                seq_pos, kNumQHeads, kHeadDim, kMaxSeqLen, 0), "attn");
            // Pack attn_out FP32 → FP4 for GEMV
            check(blackwell::kernels::pack_fp4(attn_fp4, b.attn_out, attn_scale, q_dim, 0), "pack attn");
            check(blackwell::kernels::gemv_fp4(b.proj_out, attn_fp4, attn_scale,
                W_o_fp4[l], W_o_scale[l], q_dim, kHiddenDim, 0), "gemv_o");
            // Fused RMSNorm + FP4 pack (was 2 kernels: rmsnorm + pack)
            check(blackwell::kernels::fused_rmsnorm_pack(
                b.x_fp4, b.x_scale, b.proj_out, rmsnorm_w[l],
                kHiddenDim, 1e-5f, 0), "rmsnorm_pack");
        }
    }
    cudaDeviceSynchronize();
    printf("done\n");

    // First, populate cache with seq=128 tokens so attention has realistic workload
    printf("Filling KV cache to seq_pos=128... ");
    fflush(stdout);
    // Reset x
    cudaMemcpy(b.x_fp32, x_init.data(), kHiddenDim * 4, cudaMemcpyHostToDevice);
    check(blackwell::kernels::pack_fp4(b.x_fp4, b.x_fp32, b.x_scale, kHiddenDim, 0), "pack x reset");
    for (int seq = 0; seq <= 128; ++seq) {
        for (int l = 0; l < kNumLayers; ++l) {
            int kvcache_base = l * kNumKVHeads * kMaxSeqLen * kHeadDim;
            blackwell::kernels::gemv_fp4(b.Q, b.x_fp4, b.x_scale,
                W_q_fp4[l], W_q_scale[l], kHiddenDim, q_dim, 0);
            blackwell::kernels::gemv_fp4(b.K, b.x_fp4, b.x_scale,
                W_k_fp4[l], W_k_scale[l], kHiddenDim, kv_dim, 0);
            blackwell::kernels::gemv_fp4(b.V, b.x_fp4, b.x_scale,
                W_v_fp4[l], W_v_scale[l], kHiddenDim, kv_dim, 0);
            blackwell::kernels::update_kv_cache(
                b.k_cache + kvcache_base, b.v_cache + kvcache_base,
                b.K, b.V, 0, seq,
                kNumKVHeads, kHeadDim, kMaxSeqLen, 0);
            blackwell::kernels::attention_decode(
                b.attn_out, b.Q,
                b.k_cache + kvcache_base, b.v_cache + kvcache_base,
                seq, kNumQHeads, kHeadDim, kMaxSeqLen, 0);
            blackwell::kernels::pack_fp4(attn_fp4, b.attn_out, attn_scale, q_dim, 0);
            blackwell::kernels::gemv_fp4(b.proj_out, attn_fp4, attn_scale,
                W_o_fp4[l], W_o_scale[l], q_dim, kHiddenDim, 0);
            blackwell::kernels::fused_rmsnorm_pack(
                b.x_fp4, b.x_scale, b.proj_out, rmsnorm_w[l],
                kHiddenDim, 1e-5f, 0);
        }
    }
    printf("done\n");

    const int graph_seq_pos = 128;
    printf("Capturing decode graph at seq_pos=%d... ", graph_seq_pos);
    fflush(stdout);

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    cudaStream_t graph_stream;
    cudaStreamCreate(&graph_stream);

    cudaStreamBeginCapture(graph_stream, cudaStreamCaptureModeGlobal);
    for (int l = 0; l < kNumLayers; ++l) {
        int kvcache_base = l * kNumKVHeads * kMaxSeqLen * kHeadDim;
        blackwell::kernels::fused_qkv_gemv(b.Q, b.K, b.V,
            b.x_fp4, b.x_scale,
            W_q_fp4[l], W_q_scale[l],
            W_k_fp4[l], W_k_scale[l],
            W_v_fp4[l], W_v_scale[l],
            kHiddenDim, q_dim, kv_dim, graph_stream);
        blackwell::kernels::update_kv_cache(
            b.k_cache + kvcache_base, b.v_cache + kvcache_base,
            b.K, b.V, 0, graph_seq_pos,
            kNumKVHeads, kHeadDim, kMaxSeqLen, graph_stream);
        blackwell::kernels::attention_decode(
            b.attn_out, b.Q,
            b.k_cache + kvcache_base, b.v_cache + kvcache_base,
            graph_seq_pos, kNumQHeads, kHeadDim, kMaxSeqLen, graph_stream);
        blackwell::kernels::pack_fp4(attn_fp4, b.attn_out, attn_scale, q_dim, graph_stream);
        blackwell::kernels::gemv_fp4(b.proj_out, attn_fp4, attn_scale,
            W_o_fp4[l], W_o_scale[l], q_dim, kHiddenDim, graph_stream);
        blackwell::kernels::fused_rmsnorm_pack(
            b.x_fp4, b.x_scale, b.proj_out, rmsnorm_w[l],
            kHiddenDim, 1e-5f, graph_stream);
    }

    cudaStreamEndCapture(graph_stream, &graph);
    cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
    printf("done\n");

    // Benchmark with CUDA Graph
    printf("Benchmarking %d tokens via CUDA Graph (seq_pos=%d)...\n", bench, graph_seq_pos);
    fflush(stdout);

    GpuTimer t;
    t.begin();
    for (int iter = 0; iter < bench; ++iter) {
        cudaGraphLaunch(graph_exec, graph_stream);
    }
    cudaStreamSynchronize(graph_stream);
    float total_ms = t.end();

    float per_token_ms = total_ms / bench;
    float tps = 1000.0f / per_token_ms;
    float scaled = tps * (28.0f / kNumLayers);

    printf("\n=== Graph Results (seq_pos=%d) ===\n", graph_seq_pos);
    printf("  Tokens:            %d\n", bench);
    printf("  Total time:        %.2f ms\n", total_ms);
    printf("  Per-token:         %.3f ms\n", per_token_ms);
    printf("  Throughput:        %.1f t/s (%d layers)\n", tps, kNumLayers);
    printf("  Scaled to 28:      %.1f t/s\n", scaled);
    printf("  Target (llama):    114 t/s (28 layers)\n\n");

    // Compare with non-graph launch
    printf("Benchmarking %d tokens via direct launch (no graph)...\n", bench);
    fflush(stdout);
    GpuTimer t2;
    t2.begin();
    for (int iter = 0; iter < bench; ++iter) {
        for (int l = 0; l < kNumLayers; ++l) {
            int kvcache_base = l * kNumKVHeads * kMaxSeqLen * kHeadDim;
            blackwell::kernels::fused_qkv_gemv(b.Q, b.K, b.V,
                b.x_fp4, b.x_scale,
                W_q_fp4[l], W_q_scale[l],
                W_k_fp4[l], W_k_scale[l],
                W_v_fp4[l], W_v_scale[l],
                kHiddenDim, q_dim, kv_dim, 0);
            blackwell::kernels::update_kv_cache(
                b.k_cache + kvcache_base, b.v_cache + kvcache_base,
                b.K, b.V, 0, graph_seq_pos,
                kNumKVHeads, kHeadDim, kMaxSeqLen, 0);
            blackwell::kernels::attention_decode(
                b.attn_out, b.Q,
                b.k_cache + kvcache_base, b.v_cache + kvcache_base,
                graph_seq_pos, kNumQHeads, kHeadDim, kMaxSeqLen, 0);
            blackwell::kernels::pack_fp4(attn_fp4, b.attn_out, attn_scale, q_dim, 0);
            blackwell::kernels::gemv_fp4(b.proj_out, attn_fp4, attn_scale,
                W_o_fp4[l], W_o_scale[l], q_dim, kHiddenDim, 0);
            blackwell::kernels::fused_rmsnorm_pack(
                b.x_fp4, b.x_scale, b.proj_out, rmsnorm_w[l],
                kHiddenDim, 1e-5f, 0);
        }
    }
    float total_direct = t2.end();
    printf("  Direct per-token:  %.3f ms\n", total_direct / bench);
    printf("  Graph perf vs direct: %.1f%%\n\n",
           (total_direct / total_ms) * 100.0f);

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(graph_stream);

    cudaFree(attn_fp4); cudaFree(attn_scale);

    // Cleanup
    for (int l = 0; l < kNumLayers; ++l) {
        cudaFree(W_q_fp4[l]); cudaFree(W_q_scale[l]);
        cudaFree(W_k_fp4[l]); cudaFree(W_k_scale[l]);
        cudaFree(W_v_fp4[l]); cudaFree(W_v_scale[l]);
        cudaFree(W_o_fp4[l]); cudaFree(W_o_scale[l]);
        cudaFree(rmsnorm_w[l]);
    }
    cudaFree(b.x_fp32); cudaFree(b.x_fp4); cudaFree(b.x_scale);
    cudaFree(b.Q); cudaFree(b.K); cudaFree(b.V);
    cudaFree(b.attn_out); cudaFree(b.proj_out);
    cudaFree(b.k_cache); cudaFree(b.v_cache);

    return 0;
}
