// Measure attention_decode kernel alone
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include "blackwell/kernels.h"

int main() {
    const int num_heads = 16, head_dim = 128, max_seq = 2048;
    
    float *d_K, *d_V, *d_Q, *d_out;
    cudaMalloc(&d_K, num_heads * max_seq * head_dim * 4);
    cudaMalloc(&d_V, num_heads * max_seq * head_dim * 4);
    cudaMalloc(&d_Q, num_heads * head_dim * 4);
    cudaMalloc(&d_out, num_heads * head_dim * 4);
    cudaMemset(d_K, 0, num_heads * max_seq * head_dim * 4);
    cudaMemset(d_V, 0, num_heads * max_seq * head_dim * 4);
    cudaMemset(d_Q, 0, num_heads * head_dim * 4);

    for (int seq_pos : {8, 32, 64, 128, 256, 512}) {
        cudaEvent_t s, e;
        cudaEventCreate(&s); cudaEventCreate(&e);
        
        int warm = 10, bench = 100;
        for (int i = 0; i < warm; ++i) {
            blackwell::kernels::attention_decode(d_out, d_Q, d_K, d_V, seq_pos, num_heads, head_dim, max_seq, 0);
        }
        cudaDeviceSynchronize();
        
        cudaEventRecord(s, 0);
        for (int i = 0; i < bench; ++i) {
            blackwell::kernels::attention_decode(d_out, d_Q, d_K, d_V, seq_pos, num_heads, head_dim, max_seq, 0);
        }
        cudaEventRecord(e, 0);
        cudaEventSynchronize(e);
        float ms;
        cudaEventElapsedTime(&ms, s, e);
        ms /= bench;
        
        double flops = (double)(seq_pos + 1) * head_dim * 2 * num_heads;  // QK + WV
        double tflops = flops / ms / 1e6;
        printf("seq_pos=%4d  ms=%.4f  t/s=%8.0f  GFLOPS=%.1f\n", 
               seq_pos, ms, 1000.0f / ms, tflops);
        
        cudaEventDestroy(s); cudaEventDestroy(e);
    }
    
    cudaFree(d_K); cudaFree(d_V); cudaFree(d_Q); cudaFree(d_out);
    return 0;
}
