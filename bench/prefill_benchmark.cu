// bench/prefill_benchmark.cu — Prefill vs decode GEMM throughput
// Measures: how fast can we process N tokens in parallel (prefill)?
// vs: decode processes 1 token per step
// Run: ./bench/prefill_benchmark [seq_len]
// Default: 128 tokens

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sys/time.h>
#include "blackwell/kernels.h"

using blackwell::kernels::gemv_int8_warp;
using blackwell::kernels::gemv_int8_batched;
using blackwell::kernels::quantize_int8;
using blackwell::kernels::fused_rmsnorm_batched;
using blackwell::kernels::apply_swiglu;
using blackwell::kernels::vector_add_fp32;

static void chk(cudaError_t e, const char* m = "") {
    if (e != cudaSuccess) { printf("FAIL %s: %s\n", m, cudaGetErrorString(e)); exit(1); }
}
struct GpuTimer {
    cudaEvent_t s, e;
    GpuTimer() { cudaEventCreate(&s); cudaEventCreate(&e); }
    ~GpuTimer() { cudaEventDestroy(s); cudaEventDestroy(e); }
    void start(cudaStream_t st=0) { cudaEventRecord(s, st); }
    float stop(cudaStream_t st=0) { cudaEventRecord(e, st); cudaEventSynchronize(e); float ms=0; cudaEventElapsedTime(&ms, s, e); return ms; }
};

struct W { int K, N; int8_t* d; float* sc; };
struct LW { W q, k, v, o, gate, up, down; float *rn_in, *rn_post; };

static W load_int8(const char* dir, const char* name) {
    char p[1024]; snprintf(p, 1024, "%s/%s.int8_t", dir, name);
    FILE* f = fopen(p, "rb"); if (!f) { printf("FAIL open %s\n", p); exit(1); }
    int h[5]; (void)fread(h, 4, 5, f); W w{h[0], h[1], nullptr, nullptr};
    std::vector<int8_t> tmp((size_t)w.K * w.N); (void)fread(tmp.data(), 1, tmp.size(), f); fclose(f);
    cudaMalloc(&w.d, (size_t)w.K * w.N); cudaMemcpy(w.d, tmp.data(), (size_t)w.K * w.N, cudaMemcpyHostToDevice);
    snprintf(p, 1024, "%s/%s.scale_t", dir, name); f = fopen(p, "rb"); if (!f) { printf("FAIL open %s\n", p); exit(1); }
    (void)fread(h, 4, 5, f); size_t ns = (size_t)h[3] * h[4];
    std::vector<float> ts(ns); (void)fread(ts.data(), 4, ns, f); fclose(f);
    cudaMalloc(&w.sc, ns * 4); cudaMemcpy(w.sc, ts.data(), ns * 4, cudaMemcpyHostToDevice);
    return w;
}
static float* load_f32(const char* dir, const char* name, int n) {
    char p[1024]; snprintf(p, 1024, "%s/%s.f32", dir, name);
    FILE* f = fopen(p, "rb"); if (!f) { printf("FAIL open %s\n", p); exit(1); }
    float* h = (float*)malloc(n*4); (void)fread(h, 4, n, f); fclose(f);
    float* d; cudaMalloc(&d, n*4); cudaMemcpy(d, h, n*4, cudaMemcpyHostToDevice); free(h); return d;
}

// Batched GEMV: runs gemv_int8_warp with M rows in parallel
// M = seq_len for prefill, M = 1 for decode
// Each thread block processes 1 row, warp processes 32 rows cooperatively
// d_A_i8: [M × K] INT8, d_A_sc: [M × K/16] scales
// d_B: [N × K] INT8 transposed, d_B_sc: [N × K/16] scales
// d_C: [M × N] FP32 output
static void batched_gemv(float* d_C, int8_t* d_A_i8, float* d_A_sc,
                          const W& B, int M, int N, cudaStream_t st) {
    // Process M rows in chunks of 32 (warp-cooperative)
    // For large M, this saturates the GPU better than serial per-row
    int nblocks = (M + 31) / 32;
    // Call gemv_int8_warp in a loop for each row — same as decode
    // gemv_int8_warp processes 1 row at a time (warp-cooperative within row)
    // For M=128, this is 128 calls — same time as 128 decode steps for GEMV
    // But done in parallel across the batch: GPU runs all 128 simultaneously
    // Key: gemv_int8_warp uses warp-cooperative reduction — fast for large K
    for (int m = 0; m < M; m++) {
        gemv_int8_warp(d_C + m * N,
                       d_A_i8 + m * B.K, d_A_sc + m * (B.K / 16),
                       B.d, B.sc, B.K, N, st);
    }
}

// True batched GEMV: process all M rows in one kernel launch
// Uses the existing gemv_int8_batched if available, else fallback
static __global__ void batch_gemv_kernel(float* C, const int8_t* A_i8, const float* A_sc,
                                           const int8_t* B, const float* B_sc,
                                           int M, int K, int N) {
    int row = blockIdx.x; if (row >= M) return;
    int tid = threadIdx.x;
    // dp4a: 8 elements per thread (32-bit × 2 int8)
    int nwarps = blockDim.x / 32;
    int lane = tid & 31;
    int warp_id = tid / 32;
    int nloops = N / 8;
    float acc = 0.f;
    for (int l = warp_id; l < nloops; l += nwarps) {
        int noff = l * 8;
        // Load 8 B elements (INT8) and their scales
        int bblk = noff / 16;
        float bsc = B_sc[bblk];
        int a_blk = (row * K + noff) / 16;
        float asc = A_sc[a_blk];
        int4 b8 = *(int4*)(B + noff);
        char4 a_lo = *(char4*)(A_i8 + row * K + noff);
        char4 a_hi = *(char4*)(A_i8 + row * K + noff + 4);
        acc += ((float)a_lo.x - 128.f) * ((float)(b8.x & 0xFF) - 128.f) * asc * bsc;
        acc += ((float)a_lo.y - 128.f) * ((float)(b8.x >> 8 & 0xFF) - 128.f) * asc * bsc;
        acc += ((float)a_lo.z - 128.f) * ((float)(b8.x >> 16 & 0xFF) - 128.f) * asc * bsc;
        acc += ((float)a_lo.w - 128.f) * ((float)(b8.x >> 24 & 0xFF) - 128.f) * asc * bsc;
        acc += ((float)a_hi.x - 128.f) * ((float)(b8.y & 0xFF) - 128.f) * asc * bsc;
        acc += ((float)a_hi.y - 128.f) * ((float)(b8.y >> 8 & 0xFF) - 128.f) * asc * bsc;
        acc += ((float)a_hi.z - 128.f) * ((float)(b8.y >> 16 & 0xFF) - 128.f) * asc * bsc;
        acc += ((float)a_hi.w - 128.f) * ((float)(b8.y >> 24 & 0xFF) - 128.f) * asc * bsc;
    }
    // Warp reduce
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, off);
    if (lane == 0) C[row * N + warp_id] = acc;
}

int main(int argc, char** argv) {
    const char* WDIR = "weights_int8_bf16";
    int SEQ = (argc > 1) ? atoi(argv[1]) : 128;
    if (SEQ < 1) SEQ = 1; if (SEQ > 512) SEQ = 512;
    const int NL = 28, H = 2048, Q = 2048, KV = 256, ID = 11008;

    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("=== Blackwell Prefill Benchmark ===\n");
    printf("Device: %s\n", p.name);
    printf("Model: Qwen3-1.7B INT8, NL=%d, H=%d\n", NL, H);
    printf("Sequence length: %d tokens\n\n", SEQ);

    // Load weights
    std::vector<LW> layers(NL);
    printf("Loading weights...\n"); fflush(stdout);
    for (int l = 0; l < NL; l++) {
        char p[256];
        snprintf(p, 256, "%d_self_attn.q_proj", l); layers[l].q = load_int8(WDIR, p);
        snprintf(p, 256, "%d_self_attn.k_proj", l); layers[l].k = load_int8(WDIR, p);
        snprintf(p, 256, "%d_self_attn.v_proj", l); layers[l].v = load_int8(WDIR, p);
        snprintf(p, 256, "%d_mlp.gate_proj", l);   layers[l].gate = load_int8(WDIR, p);
        snprintf(p, 256, "%d_mlp.up_proj", l);     layers[l].up = load_int8(WDIR, p);
        snprintf(p, 256, "%d_mlp.down_proj", l);   layers[l].down = load_int8(WDIR, p);
        snprintf(p, 256, "%d_input_layernorm", l); layers[l].rn_in = load_f32(WDIR, p, H);
        if ((l+1) % 7 == 0) printf("  %d/%d\n", l+1, NL);
    }
    printf("Done.\n\n"); fflush(stdout);
    printf("Allocating buffers...\n"); fflush(stdout);
    float *d_h, *d_h_i8, *d_h_sc;
    float *d_q, *d_k, *d_v;
    float *d_proj, *d_gate, *d_up;
    int8_t *d_mlp_i8; float *d_mlp_sc;
    chk(cudaMalloc(&d_h, SEQ * H * 4));
    chk(cudaMalloc(&d_h_i8, SEQ * H));
    chk(cudaMalloc(&d_h_sc, SEQ * (H / 16) * 4));
    chk(cudaMalloc(&d_q, SEQ * Q * 4));
    chk(cudaMalloc(&d_k, SEQ * KV * 4));
    chk(cudaMalloc(&d_v, SEQ * KV * 4));
    chk(cudaMalloc(&d_proj, SEQ * H * 4));
    chk(cudaMalloc(&d_gate, SEQ * ID * 4));
    chk(cudaMalloc(&d_up, SEQ * ID * 4));
    chk(cudaMalloc(&d_mlp_i8, SEQ * ID));
    chk(cudaMalloc(&d_mlp_sc, SEQ * (ID / 16) * 4));
    float* d_h_out; chk(cudaMalloc(&d_h_out, SEQ * H * 4));
    printf("All buffers allocated\n"); fflush(stdout);
    cudaStream_t st;
    cudaError_t e = cudaStreamCreate(&st);
    printf("Stream created: %s\n", cudaGetErrorString(e)); fflush(stdout);
    if (e != cudaSuccess) { printf("Stream create FAIL: %s\n", cudaGetErrorString(e)); return 1; }

    // Zero init
    cudaMemset(d_h, 0, SEQ * H * 4);

    // Warmup
    for (int w = 0; w < 3; w++) {
        cudaMemset(d_h_i8, 0, SEQ * H);
        cudaMemset(d_q, 0, SEQ * Q * 4);
    }
    cudaDeviceSynchronize();

    // === PREFILL: batched GEMM ===
    const int M_BATCH = 8;  // gemv_int8_batched max M
    printf("--- Prefill (seq_len=%d, batch=%d) ---\n", SEQ, M_BATCH); fflush(stdout);
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    for (int l = 0; l < NL; l++) {
        fused_rmsnorm_batched(d_h_out, d_h, layers[l].rn_in, H, 1e-5f, SEQ, st);
        cudaError_t e = cudaPeekAtLastError();
        if (e != cudaSuccess) { printf("FAIL l=%d rmsnorm: %s\n", l, cudaGetErrorString(e)); break; }
        cudaMemcpy(d_h, d_h_out, SEQ * H * 4, cudaMemcpyDeviceToDevice);
        quantize_int8(d_h_i8, d_h_sc, d_h, H * SEQ, st);
        e = cudaPeekAtLastError();
        if (e != cudaSuccess) { printf("FAIL l=%d quantize: %s\n", l, cudaGetErrorString(e)); break; }
        // QKV GEMVs (batched in groups of M_BATCH)
        for (int mb = 0; mb < SEQ; mb += M_BATCH) {
            int M = (SEQ - mb < M_BATCH) ? (SEQ - mb) : M_BATCH;
            gemv_int8_batched(d_q + mb * Q, d_h_i8 + mb * H, d_h_sc + mb * (H/16),
                              layers[l].q.d, layers[l].q.sc, H, Q, M, st);
            gemv_int8_batched(d_k + mb * KV, d_h_i8 + mb * H, d_h_sc + mb * (H/16),
                              layers[l].k.d, layers[l].k.sc, H, KV, M, st);
            gemv_int8_batched(d_v + mb * KV, d_h_i8 + mb * H, d_h_sc + mb * (H/16),
                              layers[l].v.d, layers[l].v.sc, H, KV, M, st);
        }
        // MLP
        quantize_int8(d_mlp_i8, d_mlp_sc, d_h, H * SEQ, st);
        for (int mb = 0; mb < SEQ; mb += M_BATCH) {
            int M = (SEQ - mb < M_BATCH) ? (SEQ - mb) : M_BATCH;
            gemv_int8_batched(d_gate + mb * ID, d_mlp_i8 + mb * H, d_mlp_sc + mb * (H/16),
                              layers[l].gate.d, layers[l].gate.sc, H, ID, M, st);
            gemv_int8_batched(d_up + mb * ID, d_mlp_i8 + mb * H, d_mlp_sc + mb * (H/16),
                              layers[l].up.d, layers[l].up.sc, H, ID, M, st);
        }
        apply_swiglu(d_gate, d_gate, d_up, SEQ * ID, st);
        quantize_int8(d_mlp_i8, d_mlp_sc, d_gate, SEQ * ID, st);
        for (int mb = 0; mb < SEQ; mb += M_BATCH) {
            int M = (SEQ - mb < M_BATCH) ? (SEQ - mb) : M_BATCH;
            gemv_int8_batched(d_proj + mb * H, d_mlp_i8 + mb * ID, d_mlp_sc + mb * (ID/16),
                              layers[l].down.d, layers[l].down.sc, ID, H, M, st);
        }
        vector_add_fp32(d_h, d_proj, d_h, H * SEQ, st);
    }
    cudaStreamSynchronize(st);
    gettimeofday(&t1, NULL);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_usec - t0.tv_usec) / 1000.0;
    printf("\n  Total: %.1f ms for %d tokens\n", ms, SEQ);
    printf("  Prefill throughput: %.0f tokens/s (%.1f ms/token)\n\n",
           1000.0 * SEQ / ms, ms / SEQ);
    printf("Note: GEMM-only prefill (attention not included).\n");
    printf("      Attention adds O(n^2) cost per layer.\n");
    printf("      Real prefill = GEMM time + attention.\n\n");

    cudaStreamDestroy(st);
    return 0;
}