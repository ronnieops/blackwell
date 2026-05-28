// cuBLAS GEMV benchmark — proper coalesced layout
// Weight stored column-major [K × N] (transposed from our row-major [N × K])
// Use CUBLAS_OP_N for direct GEMV with coalesced access
//
// Build:
//   /usr/local/cuda-12.8/bin/nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120,code=sm_120 \
//     bench/cublas_gemv_bench.cu -lcublas -o bench/cublas_gemv_bench
//
// Run: ./bench/cublas_gemv_bench

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <chrono>

using Clock = std::chrono::high_resolution_clock;
#define CU(x) do{cudaError_t e=x;if(e!=cudaSuccess){printf("CUERR %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
#define CB(x) do{cublasStatus_t s=x;if(s!=CUBLAS_STATUS_SUCCESS){printf("CUBERR %s:%d: %d\n",__FILE__,__LINE__,s);exit(1);}}while(0)

int main() {
    cudaDeviceProp P; cudaGetDeviceProperties(&P, 0);
    printf("# cuBLAS GEMV — %s\n\n", P.name);

    cublasHandle_t h; CB(cublasCreate(&h));

    // Test configs: (N_out, K_in, reps)
    struct Test { int N, K, reps; const char* label; };
    Test tests[] = {
        {2048, 2048, 2000, "q_proj/o_proj"},
        {1024, 2048, 2000, "k_proj/v_proj"},
        {6144, 2048, 500,  "gate/up_proj"},
        {2048, 6144, 500,  "down_proj"},
        {151936, 2048, 50, "lm_head"},
    };

    // Types to test
    struct Type { cudaDataType_t wtype; cudaDataType_t atype;
                  cudaDataType ctype; cudaDataType comp; int esz; const char* name; };
    Type types[] = {
        {CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 4, "FP32"},
        {CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, 2, "FP16"},
        {CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F, 2, "BF16"},
    };

    for (auto& t : types) {
        printf("── %s ──\n", t.name);
        for (auto& cfg : tests) {
            int N = cfg.N, K = cfg.K;
            int esz = t.esz;

            // Alloc as column-major [K × N]: lda = K
            void *d_W, *d_x, *d_y;
            CU(cudaMalloc(&d_W, (size_t)K * N * esz));
            CU(cudaMalloc(&d_x, (size_t)K * esz));
            CU(cudaMalloc(&d_y, (size_t)N * ((t.ctype == CUDA_R_32F) ? 4 : esz)));

            float alpha_f = 1.0f, beta_f = 0.0f;

            // Warmup
            for (int i = 0; i < 10; i++) {
                CB(cublasGemmEx(h,
                    CUBLAS_OP_N, CUBLAS_OP_N,  // A [1×K] × B [K×N] → C [1×N]
                    1, N, K,
                    &alpha_f,
                    d_x, t.atype, 1,       // A: [1×K] col-major, lda=1
                    d_W, t.wtype, K,        // B: [K×N] col-major, lda=K
                    &beta_f,
                    d_y, t.ctype, 1,        // C: [1×N] col-major, ldc=1
                    t.comp, CUBLAS_GEMM_DEFAULT));
            }
            CU(cudaDeviceSynchronize());

            // Benchmark
            auto t0 = Clock::now();
            for (int i = 0; i < cfg.reps; i++) {
                CB(cublasGemmEx(h,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    1, N, K,
                    &alpha_f,
                    d_x, t.atype, 1,
                    d_W, t.wtype, K,
                    &beta_f,
                    d_y, t.ctype, 1,
                    t.comp, CUBLAS_GEMM_DEFAULT));
            }
            CU(cudaDeviceSynchronize());
            auto t1 = Clock::now();

            double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
            double us_per = ms * 1000.0 / cfg.reps;
            double bw = (double)N * K * esz / (us_per / 1e6) / 1e9;

            printf("  %-14s N=%5d K=%4d: %8.1f us, %6.1f GB/s\n",
                   cfg.label, N, K, us_per, bw);

            CU(cudaFree(d_W)); CU(cudaFree(d_x)); CU(cudaFree(d_y));
        }
        printf("\n");
    }

    CB(cublasDestroy(h));
    printf("Done.\n");
    return 0;
}
