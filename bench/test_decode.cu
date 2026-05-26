// bench/test_decode.cu — Verify decode kernels (update_kv_cache, attention_decode)
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include "blackwell/kernels.h"

static bool check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        printf("FAIL: %s: %s\n", msg, cudaGetErrorString(e));
        return false;
    }
    return true;
}

float max_err(const float* a, const float* b, int n) {
    float me = 0;
    for (int i = 0; i < n; ++i) {
        float d = fabsf(a[i] - b[i]);
        if (d > me) me = d;
    }
    return me;
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("# Decode Kernel Test\nDevice: %s (CC %d.%d)\n\n", p.name, p.major, p.minor);

    const int num_heads = 4;
    const int head_dim = 128;
    const int max_seq_len = 1024;

    // Allocate KV cache
    int cache_bytes = num_heads * max_seq_len * head_dim * sizeof(float);
    float *d_k_cache, *d_v_cache;
    cudaMalloc(&d_k_cache, cache_bytes);
    cudaMalloc(&d_v_cache, cache_bytes);
    cudaMemset(d_k_cache, 0, cache_bytes);
    cudaMemset(d_v_cache, 0, cache_bytes);

    // Allocate Q, K, V for single token
    int heads_bytes = num_heads * head_dim * sizeof(float);
    float *d_Q, *d_K, *d_V, *d_out;
    cudaMalloc(&d_Q, heads_bytes);
    cudaMalloc(&d_K, heads_bytes);
    cudaMalloc(&d_V, heads_bytes);
    cudaMalloc(&d_out, heads_bytes);

    // Fill with simple values: Q=1, K=1, V=1
    std::vector<float> Q_h(num_heads * head_dim, 1.0f);
    std::vector<float> K_h(num_heads * head_dim, 1.0f);
    std::vector<float> V_h(num_heads * head_dim, 1.0f);
    cudaMemcpy(d_Q, Q_h.data(), heads_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_h.data(), heads_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_h.data(), heads_bytes, cudaMemcpyHostToDevice);

    // Step 1: Update KV cache at position 0
    printf("1. update_kv_cache pos=0... ");
    fflush(stdout);
    if (check(blackwell::kernels::update_kv_cache(
            d_k_cache, d_v_cache, d_K, d_V, 0, 0,
            num_heads, head_dim, max_seq_len, 0), "update_kv_cache")) {
        printf("OK\n");
    }

    // Step 2: Update KV cache at position 1 (different values)
    std::vector<float> K2_h(num_heads * head_dim, 2.0f);
    std::vector<float> V2_h(num_heads * head_dim, 2.0f);
    cudaMemcpy(d_K, K2_h.data(), heads_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V2_h.data(), heads_bytes, cudaMemcpyHostToDevice);
    printf("2. update_kv_cache pos=1... ");
    fflush(stdout);
    if (check(blackwell::kernels::update_kv_cache(
            d_k_cache, d_v_cache, d_K, d_V, 0, 1,
            num_heads, head_dim, max_seq_len, 0), "update_kv_cache")) {
        printf("OK\n");
    }

    // Step 3: Decode attention at seq_pos=1 (2 cached tokens)
    printf("3. attention_decode (2 cached tokens)... ");
    fflush(stdout);
    if (check(blackwell::kernels::attention_decode(
            d_out, d_Q, d_k_cache, d_v_cache,
            1, num_heads, head_dim, max_seq_len, 0), "attention_decode")) {
        printf("OK\n");
    }

    std::vector<float> out_h(num_heads * head_dim);
    cudaMemcpy(out_h.data(), d_out, heads_bytes, cudaMemcpyDeviceToHost);

    // Verify: with Q=1, K1=1, K2=2, V1=1, V2=2
    // score[t=0] = dot(1,1) = 128
    // score[t=1] = dot(1,2) = 256
    // softmax: exp(128/11.3) ≈ exp(11.3) = 80491, exp(256/11.3) = exp(22.6)=6.5e9
    // weight[1] dominates → output ≈ 2.0 (dominated by V=2)
    // With scale = 1/sqrt(128) = 0.0884
    // score[0] = 128*0.0884 = 11.31, e = 81431
    // score[1] = 256*0.0884 = 22.63, e = 6.67e9
    // w0 = 81431/6.67e9 = 1.22e-5  w1 ≈ 1.0
    // output ≈ 2.0 (almost exactly)

    float expected = 2.0f;
    float err = max_err(out_h.data(), Q_h.data(), num_heads * head_dim);
    printf("   Max err vs Q: %.6f (expected output ≈ 2.0)\n", err);
    printf("   First 4 outputs: %.4f %.4f %.4f %.4f\n",
           out_h[0], out_h[1], out_h[2], out_h[3]);

    bool pass = (out_h[0] > 1.9f && out_h[0] < 2.1f);
    printf("\n%s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_k_cache); cudaFree(d_v_cache);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_out);
    return pass ? 0 : 1;
}
