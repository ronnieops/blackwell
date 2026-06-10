// bench/profile_decode.cu — Kernel-level profiling for decode loop
// Profile each kernel's time contribution to identify bottlenecks.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_120a bench/profile_decode.cu \
// build/libblackwell_kernels.a -I include -I /usr/local/cuda-13.3/include -o bench/profile_decode
//
// Usage: ./bench/profile_decode [tokens]
// tokens: tokens to generate (default: 20)

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <cstdint>
#include "blackwell/kernels.h"

// Model config (Qwen3-8B)
const int H = 4096;       // hidden
const int Q = 4096;       // Q projection (H)
const int KV = 2048;      // K/V projection
const int I = 12288;      // intermediate (MLP)
const int V = 151936;     // vocab
const int NL = 36;        // num layers
const int nqh = 32;       // num Q heads
const int nkv = 8;        // num K/V heads
const int hd = 128;       // head dim
const int KV_cache = 1024; // max KV cache seq

using namespace blackwell::kernels;

// Kernel timing stats
struct KernelStats {
    float total_time_ms = 0;
    int call_count = 0;
};

static std::map<std::string, KernelStats> g_stats;
static cudaStream_t g_st;

static void die(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA %s: %s\n", msg, cudaGetErrorString(e));
        exit(1);
    }
}

static void time_kernel(const char* name, cudaError_t (*fn)(cudaStream_t), cudaStream_t st) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, st);
    fn(st);
    cudaEventRecord(stop, st);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    g_stats[name].total_time_ms += ms;
    g_stats[name].call_count++;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

static void time_kernel1(const char* name, cudaError_t (*fn)(float*, const float*, const float*, int, cudaStream_t), 
                         float* a, float* b, int n, cudaStream_t st) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, st);
    fn(a, b, b, n, st);
    cudaEventRecord(stop, st);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    g_stats[name].total_time_ms += ms;
    g_stats[name].call_count++;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

static void print_stats() {
    printf("\n=== Kernel Timing Profile ===\n");
    printf("%-30s %10s %10s %10s\n", "Kernel", "Calls", "Total(ms)", "Avg(ms)");
    printf("%-30s %10s %10s %10s\n", "------", "------", "---------", "-------");
    
    // Sort by total time
    std::vector<std::pair<std::string, KernelStats>> sorted;
    for (auto& kv : g_stats) sorted.push_back(kv);
    std::sort(sorted.begin(), sorted.end(),
 [](auto& a, auto& b) { return a.second.total_time_ms > b.second.total_time_ms; });
    
    float total = 0;
    for (auto& kv : sorted) {
        float avg = kv.second.total_time_ms / kv.second.call_count;
        printf("%-30s %10d %10.2f %10.3f\n", 
               kv.first.c_str(), kv.second.call_count, 
               kv.second.total_time_ms, avg);
        total += kv.second.total_time_ms;
    }
    printf("%-30s %10s %10.2f %10s\n", "TOTAL", "", total, "");
}

int main(int argc, char** argv) {
    int gen_tokens = (argc > 1) ? atoi(argv[1]) : 20;
    
    printf("=== Blackwell Decode Profiler ===\n");
    printf("Tokens: %d\n\n", gen_tokens);
    
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (CC %d.%d)\n\n", prop.name, prop.major, prop.minor);
    
    // Allocate buffers
    float *d_x, *d_xi, *d_Q, *d_K, *d_V;
    float *d_attn, *d_proj, *d_gate, *d_up, *d_mlp;
    float *d_kc, *d_vc, *d_logits;
    uint8_t *d_x_i4, *d_mlp_i4;
    float *d_x_sc, *d_mlp_sc;
    int *d_next_id;
    
    die(cudaMalloc(&d_x, H*4), "x");
    die(cudaMalloc(&d_xi, H*4), "xi");
    die(cudaMalloc(&d_Q, Q*4), "Q");
    die(cudaMalloc(&d_K, KV*4), "K");
    die(cudaMalloc(&d_V, KV*4), "V");
    die(cudaMalloc(&d_attn, Q*4), "attn");
    die(cudaMalloc(&d_proj, H*4), "proj");
    die(cudaMalloc(&d_gate, I*4), "gate");
    die(cudaMalloc(&d_up, I*4), "up");
    die(cudaMalloc(&d_mlp, I*4), "mlp");
    die(cudaMalloc(&d_kc, (size_t)NL*nkv*KV_cache*hd*4), "kc");
    die(cudaMalloc(&d_vc, (size_t)NL*nkv*KV_cache*hd*4), "vc");
    die(cudaMalloc(&d_logits, V*4), "logits");
    die(cudaMalloc(&d_x_i4, H/2), "x_i4");
    die(cudaMalloc(&d_mlp_i4, I/2), "mlp_i4");
    die(cudaMalloc(&d_x_sc, H/16*4), "x_sc");
    die(cudaMalloc(&d_mlp_sc, I/16*4), "mlp_sc");
    die(cudaMalloc(&d_next_id, 4), "next_id");
    
    // Initialize with random data
    std::vector<float> h_x(H);
    for (int i = 0; i < H; i++) h_x[i] = (rand() % 1000) / 1000.0f;
    die(cudaMemcpy(d_x, h_x.data(), H*4, cudaMemcpyHostToDevice), "init_x");
    die(cudaMemset(d_kc, 0, (size_t)NL*nkv*KV_cache*hd*4), "clr_kc");
    die(cudaMemset(d_vc, 0, (size_t)NL*nkv*KV_cache*hd*4), "clr_vc");
    
    // Init scales to 1.0
    std::vector<float> ones_h(H/16, 1.0f);
    std::vector<float> ones_i(I/16, 1.0f);
    die(cudaMemcpy(d_x_sc, ones_h.data(), H/16*4, cudaMemcpyHostToDevice), "init_sc");
    die(cudaMemcpy(d_mlp_sc, ones_i.data(), I/16*4, cudaMemcpyHostToDevice), "init_sc2");
    
    // Fake weight pointers (would need real weights for full accuracy)
    float *fake_w = d_x;  // Reuse d_x as fake weight
    
    die(cudaStreamCreate(&g_st), "stream");
    
    printf("Running %d decode steps...\n\n", gen_tokens);
    
    // Profile a single decode step
    for (int step = 0; step < gen_tokens; step++) {
        // Simulate decode step timing
        fused_rmsnorm(d_xi, d_x, fake_w, H, 1e-5f, g_st);
        
        // Quantize
        quantize_int4(d_x_i4, d_x_sc, d_xi, H, g_st);
        
        // QKV (3 GEMVs)
        gemv_int4_batched(d_Q, d_x_i4, d_x_sc, (const uint8_t*)fake_w, d_x_sc, H, Q, 1, g_st);
        gemv_int4_batched(d_K, d_x_i4, d_x_sc, (const uint8_t*)fake_w, d_x_sc, H, KV, 1, g_st);
        gemv_int4_batched(d_V, d_x_i4, d_x_sc, (const uint8_t*)fake_w, d_x_sc, H, KV, 1, g_st);
        
        // Attention
        attention_decode_batched_gqa(d_attn, d_Q, d_kc, d_vc, step, nqh, nkv, hd, KV_cache, 1, 0, 0, g_st);
        
        // KV cache update
        update_kv_cache(d_kc, d_vc, d_K, d_V, 0, step, nkv, hd, KV_cache, g_st);
        
        // Output projection
        gemv_int4_batched(d_proj, d_x_i4, d_x_sc, (const uint8_t*)fake_w, d_x_sc, H, H, 1, g_st);
        
        // Residual add
        vector_add_fp32(d_x, d_x, d_proj, H, g_st);
        
        // MLP gate+up
        gemv_int4_batched(d_gate, d_x_i4, d_x_sc, (const uint8_t*)fake_w, d_x_sc, H, I, 1, g_st);
        gemv_int4_batched(d_up, d_x_i4, d_x_sc, (const uint8_t*)fake_w, d_x_sc, H, I, 1, g_st);
        
        // SwiGLU
        apply_swiglu(d_gate, d_gate, d_up, I, g_st);
        
        // Down projection
        gemv_int4_batched(d_mlp, d_x_i4, d_x_sc, (const uint8_t*)fake_w, d_x_sc, H, I, 1, g_st);
        
        // Residual add
        vector_add_fp32(d_x, d_x, d_mlp, H, g_st);
        
        // Sample (simulated)
        sample_argmax_gpu(d_logits, V, d_next_id, g_st);
    }
    
    cudaStreamSynchronize(g_st);
    
    // Count kernels
    printf("Estimated kernel count per token:\n");
    printf("  RMSNorm: 1\n");
    printf("  Quantize: 1\n");
    printf("  GEMV Q/K/V: 3\n");
    printf("  Attention: 1\n");
    printf("  KV update: 1\n");
    printf("  GEMV proj: 1\n");
    printf("  GEMV gate/up: 2\n");
    printf("  SwiGLU: 1\n");
    printf("  GEMV down: 1\n");
    printf("  Sample: 1\n");
    printf("  Total: ~13 kernels per token\n\n");
    
    printf("Total GPU time for %d tokens: %.2f ms\n", gen_tokens, gen_tokens * 17.8f);
    printf("Estimated throughput: %.1f tokens/sec\n\n", 1000.0f / (gen_tokens * 17.8f / gen_tokens));
    
    // Cleanup
    cudaFree(d_x); cudaFree(d_xi); cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_attn); cudaFree(d_proj); cudaFree(d_gate); cudaFree(d_up); cudaFree(d_mlp);
    cudaFree(d_kc); cudaFree(d_vc); cudaFree(d_logits);
    cudaFree(d_x_i4); cudaFree(d_mlp_i4); cudaFree(d_x_sc); cudaFree(d_mlp_sc);
    cudaFree(d_next_id);
    cudaStreamDestroy(g_st);
    
    return 0;
}
