// bench/verify_gemm_dp4a.cu — Verify INT8×INT8 __dp4a GEMM correctness
//
// For each weight matrix (Q/K/V/O/gate/up/down):
//   1. pack_int8: FP32 input → INT8 activations + scales
//   2. gemm_int8_dp4a: INT8×INT8 GEMM via __dp4a
//   3. FP32 reference: dequant weights + FP32 GEMM
//   4. Compare max error and cosine similarity

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include "blackwell/kernels.h"

static void die(cudaError_t e, const char* m) {
    if(e!=cudaSuccess){printf("FAIL %s: %s\n",m,cudaGetErrorString(e));exit(1);}
}

static void die_cond(bool ok, const char* m) {
    if(!ok){printf("FAIL %s\n",m);exit(1);}
}

// Dequantize INT8 weights to FP32: W_fp32[n][k] = W_i8[n][k] * W_sc[n][k/16]
void dequant_weights(float* out, const int8_t* w_i8, const float* w_sc,
                     int N, int K) {
    int num_K_blks = K / 16;
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) {
            int kb = k / 16;
            out[n * K + k] = (float)w_i8[n * K + k] * w_sc[n * num_K_blks + kb];
        }
}

// Dequantize INT8 activations to FP32
void dequant_acts(float* out, const int8_t* a_i8, const float* a_sc,
                  int M, int K) {
    int num_K_blks = K / 16;
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) {
            int kb = k / 16;
            out[m * K + k] = (float)a_i8[m * K + k] * a_sc[m * num_K_blks + kb];
        }
}

// CPU GEMM reference: C[M×N] = A[M×K] × B^T[N×K]
void cpu_gemm_fp32(float* C, const float* A, const float* Wt,
                   int M, int N, int K) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[m * K + k] * Wt[n * K + k];
            C[m * N + n] = sum;
        }
}

struct LW { std::vector<int8_t> d; std::vector<float> sc; };
static LW lw(const char* p) {
    char x[256]; snprintf(x,256,"%s.int8_t",p);
    FILE* f=fopen(x,"rb"); (void)f; die_cond(f!=0,"open int8_t");
    int h[5]; (void)fread(h,4,5,f); LW w;
    w.d.resize(h[0]*h[1]); (void)fread(w.d.data(),1,w.d.size(),f); fclose(f);
    snprintf(x,256,"%s.scale_t",p); f=fopen(x,"rb"); die_cond(f!=0,"open scale_t");
    (void)fread(h,4,5,f);
    w.sc.resize(h[3]*h[4]); (void)fread(w.sc.data(),4,w.sc.size(),f); fclose(f);
    return w;
}

const int H=2048, QD=2048, KV=1024, ID=6144;

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 128;
    printf("=== INT8×INT8 __dp4a GEMM Correctness Verification ===\n");
    printf("  M=%d (prefill tokens)\n\n", M);

    struct TestCase { const char* name; int N; int K; const char* path; };
    TestCase tests[] = {
        {"Q proj", QD, H, "weights_int8_bf16/0_self_attn.q_proj"},
        {"K proj", KV, H, "weights_int8_bf16/0_self_attn.k_proj"},
        {"V proj", KV, H, "weights_int8_bf16/0_self_attn.v_proj"},
        {"O proj", H, QD, "weights_int8_bf16/0_self_attn.o_proj"},
        {"gate",   ID, H, "weights_int8_bf16/0_mlp.gate_proj"},
        {"up",     ID, H, "weights_int8_bf16/0_mlp.up_proj"},
        {"down",   H, ID, "weights_int8_bf16/0_mlp.down_proj"},
    };
    int n_tests = sizeof(tests) / sizeof(tests[0]);

    // Synthetic FP32 input (max K = ID = 6144)
    int maxK = ID;
    std::vector<float> h_A(M * maxK);
    for (int i = 0; i < M * maxK; ++i)
        h_A[i] = ((i * 31 + 7) % 127 - 63) * 0.01f;

    // GPU buffers
    float *d_A, *d_C;
    die(cudaMalloc(&d_A, M * maxK * sizeof(float)), "d_A");
    die(cudaMalloc(&d_C, M * ID * sizeof(float)), "d_C");
    die(cudaMemcpy(d_A, h_A.data(), M * maxK * sizeof(float), cudaMemcpyHostToDevice), "cpy_A");

    // Pre-quantized activation buffers (max K = ID = 6144 → K/16 = 384 scales per row)
    int8_t *d_A_i8;
    float *d_A_sc;
    int a_elems = M * maxK;
    int a_nblks = a_elems / 16;
    die(cudaMalloc(&d_A_i8, a_elems), "d_A_i8");
    die(cudaMalloc(&d_A_sc, a_nblks * sizeof(float)), "d_A_sc");

    int pass = 0, fail = 0;

    for (int t = 0; t < n_tests; ++t) {
        auto& tc = tests[t];
        int N = tc.N, K = tc.K;

        printf("--- %s (M=%d, N=%d, K=%d) ---\n", tc.name, M, N, K);

        // Load INT8 weights
        LW w = lw(tc.path);

        // Pre-quantize FP32 input to INT8 (compute absmax scales on host)
        {
            int num_K_blks = K / 16;
            std::vector<float> h_A_quant(M * K);
            std::vector<int8_t> h_A_i8_host(M * K);
            std::vector<float> h_A_sc_host(M * num_K_blks);
            for (int m = 0; m < M; ++m) {
                for (int kb = 0; kb < num_K_blks; ++kb) {
                    float block_max = 0.0f;
                    for (int j = 0; j < 16; ++j) {
                        float v = h_A[m * K + kb * 16 + j];
                        float av = fabsf(v);
                        if (av > block_max) block_max = av;
                    }
                    float sc = block_max / 127.0f;
                    if (sc < 1e-10f) sc = 1.0f;
                    h_A_sc_host[m * num_K_blks + kb] = sc;
                    for (int j = 0; j < 16; ++j) {
                        float v = h_A[m * K + kb * 16 + j] / sc;
                        v = fminf(127.0f, fmaxf(-127.0f, roundf(v)));
                        h_A_i8_host[m * K + kb * 16 + j] = (int8_t)(int)v;
                    }
                }
            }
            die(cudaMemcpy(d_A_i8, h_A_i8_host.data(), M * K, cudaMemcpyHostToDevice), "cpy_Ai8");
            die(cudaMemcpy(d_A_sc, h_A_sc_host.data(), M * num_K_blks * sizeof(float), cudaMemcpyHostToDevice), "cpy_Asc");
        }

        // GPU: INT8×INT8 __dp4a GEMM
        {
            int8_t* d_wi8; float* d_wsc;
            die(cudaMalloc((void**)&d_wi8, w.d.size()), "d_wi8");
            die(cudaMalloc((void**)&d_wsc, w.sc.size() * sizeof(float)), "d_wsc");
            die(cudaMemcpy(d_wi8, w.d.data(), w.d.size(), cudaMemcpyHostToDevice), "cpy_wi8");
            die(cudaMemcpy(d_wsc, w.sc.data(), w.sc.size() * sizeof(float), cudaMemcpyHostToDevice), "cpy_wsc");

            die(cudaMemset(d_C, 0, M * N * sizeof(float)), "memset_C");
            die(blackwell::kernels::gemm_int8_dp4a(d_C, d_A_i8, d_A_sc, d_wi8, d_wsc, M, N, K, 0), "gemm_int8_dp4a");

            std::vector<float> h_C_gpu(M * N);
            die(cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost), "cpy_C_gpu");

            // CPU: FP32 reference using dequantized activations
            std::vector<float> W_fp32(N * K);
            dequant_weights(W_fp32.data(), w.d.data(), w.sc.data(), N, K);

            // Dequant INT8 activations back to FP32
            std::vector<float> A_fp32(M * K);
            {
                std::vector<int8_t> h_A_i8(M * K);
                std::vector<float> h_A_sc(M * (K / 16));
                die(cudaMemcpy(h_A_i8.data(), d_A_i8, M * K, cudaMemcpyDeviceToHost), "cpy_Ai8");
                die(cudaMemcpy(h_A_sc.data(), d_A_sc, M * (K / 16) * sizeof(float), cudaMemcpyDeviceToHost), "cpy_Asc");
                dequant_acts(A_fp32.data(), h_A_i8.data(), h_A_sc.data(), M, K);
            }

            std::vector<float> h_C_ref(M * N);
            cpu_gemm_fp32(h_C_ref.data(), A_fp32.data(), W_fp32.data(), M, N, K);

            // Compare
            float max_err = 0.0f;
            double sum_sq_err = 0.0;
            double sum_sq_ref = 0.0;
            for (int i = 0; i < M * N; ++i) {
                float err = fabsf(h_C_gpu[i] - h_C_ref[i]);
                if (err > max_err) max_err = err;
                sum_sq_err += (double)err * err;
                sum_sq_ref += (double)h_C_ref[i] * h_C_ref[i];
            }
            float rmse = sqrtf(sum_sq_err / (M * N));
            double denom = sqrt(sum_sq_ref) * sqrt(sum_sq_err + sum_sq_ref) + 1e-12;
            float cosine = (float)(1.0 - sum_sq_err / denom);
            if (cosine < -1.0f) cosine = -1.0f;
            if (cosine > 1.0f) cosine = 1.0f;

            printf("  GPU[0..3]: %.4f %.4f %.4f %.4f\n",
                   h_C_gpu[0], h_C_gpu[1], h_C_gpu[2], h_C_gpu[3]);
            printf("  REF[0..3]: %.4f %.4f %.4f %.4f\n",
                   h_C_ref[0], h_C_ref[1], h_C_ref[2], h_C_ref[3]);
            printf("  Max err: %.6f, RMSE: %.6f, Cosine: %.8f\n",
                   max_err, rmse, cosine);

            bool ok = (cosine > 0.999f) && (rmse < 0.1f);
            printf("  Result: %s\n\n", ok ? "PASS ✅" : "FAIL ❌");
            if (ok) pass++; else fail++;

            cudaFree(d_wi8);
            cudaFree(d_wsc);
        }
    }

    printf("=== Summary: %d/%d passed ===\n", pass, n_tests);

    cudaFree(d_A_i8);
    cudaFree(d_A_sc);
    cudaFree(d_A);
    cudaFree(d_C);
    return fail > 0 ? 1 : 0;
}
