// bench/decode_qwen35_9b_batched.cu — M=8 batched decode for Qwen3.5-9B GatedDeltaNet
// 32 layers: 24 linear_attention (GatedDeltaNet) + 8 full_attention (GQA)
// Build: nvcc -O3 -std=c++17 -gencode=arch=compute_120a,code=sm_120a -I include bench/decode_qwen35_9b_batched.cu build/libblackwell_kernels.a -o bench/decode_qwen35_9b_batched
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include <cstdint>
#include "blackwell/kernels.h"

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
struct Int8W { int K, N; int8_t* d; float* sc; };
static Int8W load_int8(const char* dir, const char* name) {
    char path[512]; snprintf(path, 512, "%s/%s.int8_t", dir, name);
    FILE* f = fopen(path, "rb"); if (!f) { printf("FAIL open %s\n", path); exit(1); }
    int h[5]; fread(h, 4, 5, f); Int8W w; w.K = h[0]; w.N = h[1];
    size_t db = (size_t)w.K * w.N; int8_t* hd = (int8_t*)malloc(db); fread(hd, 1, db, f); fclose(f);
    snprintf(path, 512, "%s/%s.scale_t", dir, name); f = fopen(path, "rb"); fread(h, 4, 5, f);
    size_t sb = (size_t)h[3] * h[4] * 4; float* hs = (float*)malloc(sb); fread(hs, 4, h[3]*h[4], f); fclose(f);
    cudaMalloc(&w.d, db); cudaMemcpy(w.d, hd, db, cudaMemcpyHostToDevice);
    cudaMalloc(&w.sc, sb); cudaMemcpy(w.sc, hs, sb, cudaMemcpyHostToDevice);
    free(hd); free(hs); return w;
}
static float* load_f32(const char* dir, const char* name, int n) {
    char path[512]; snprintf(path, 512, "%s/%s.f32", dir, name);
    FILE* f = fopen(path, "rb"); if (!f) { printf("FAIL open %s\n", path); exit(1); }
    float* h = (float*)malloc(n*4); fread(h, 4, n, f); fclose(f);
    float* d; cudaMalloc(&d, n*4); cudaMemcpy(d, h, n*4, cudaMemcpyHostToDevice); free(h); return d;
}
static float* load_bf16(const char* dir, const char* name, int n) {
    char path[512]; snprintf(path, 512, "%s/%s.f16", dir, name);
    FILE* f = fopen(path, "rb"); if (!f) { printf("FAIL open %s\n", path); exit(1); }
    uint16_t* h16 = (uint16_t*)malloc(n*2); fread(h16, 2, n, f); fclose(f);
    float* h32 = (float*)malloc(n*4);
    for (int i = 0; i < n; i++) { uint32_t u = (uint32_t)h16[i] << 16; memcpy(&h32[i], &u, 4); }
    float* d; cudaMalloc(&d, n*4); cudaMemcpy(d, h32, n*4, cudaMemcpyHostToDevice); free(h16); free(h32); return d;
}
static const char* LTYPE[32] = {"lin","lin","lin","full","lin","lin","lin","full","lin","lin","lin","full","lin","lin","lin","full","lin","lin","lin","full","lin","lin","lin","full","lin","lin","lin","full","lin","lin","lin","full"};
struct LinW { Int8W qkv, a, b, z, out; float* conv_w; float* A_log; float* dt_bias; float* norm_w; };
struct FullW { Int8W q, k, v, o; float* qn; float* kn; };
struct MlpW { Int8W gate, up, down; };

__global__ void head_norm_k(float* d, const float* w, int nh, int hd, float eps) {
    int h = blockIdx.x; if (h >= nh) return;
    float* p = d + h * hd; __shared__ float wp[4];
    float s = 0; int tid = threadIdx.x;
    for (int i = tid; i < hd; i += blockDim.x) s += p[i]*p[i];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if ((tid & 31) == 0) wp[tid >> 5] = s; __syncthreads();
    if (tid < 32) { float v = (tid < 4) ? wp[tid] : 0.f;
        for (int o = 4; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (tid == 0) wp[0] = rsqrtf(v/hd+eps); }
    __syncthreads(); float is = wp[0];
    for (int i = tid; i < hd; i += blockDim.x) p[i] = p[i] * is * w[i];
}
__global__ void rope_k(float* d, int nh, int hd, const int* sp, int pd) {
    int h = blockIdx.x, t = threadIdx.x; if (h >= nh || t >= pd/2) return;
    int pos = *sp, i2 = t*2; float* pair = d + h*hd + i2;
    float th = (float)pos * powf(10000000.f, -2.f*(float)t/(float)pd);
    float c = cosf(th), s = sinf(th), x = pair[0], y = pair[1];
    pair[0] = x*c - y*s; pair[1] = x*s + y*c;
}
__global__ void compute_g_k(float* g, const float* a, const float* al, const float* dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float sp = logf(1.f + expf(a[i] + dt[i]));
    g[i] = -expf(al[i]) * sp;
}

int main(int argc, char** argv) {
    const char* WDIR = "weights_int8_qwen35_9b";
    int M = (argc > 1) ? atoi(argv[1]) : 1;
    int NT = (argc > 2) ? atoi(argv[2]) : 20;
    if (M < 1) M = 1; if (M > 8) M = 8;
    const int H=4096, I=12288, V=248320, NK=16, NV=32, HD=128, NQ=16, NKV=4, HDA=256;
    const int CD=NK*HD*2+NV*HD, CK=4, MS=2048, PD=HDA/4;
    cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf("# Qwen3.5-9B INT8 Batched Decode — %s\n", p.name);
    printf("# 32 layers, M=%d, H=%d I=%d\n\n", M, H, I);

    LinW lw[32]; FullW fw[32]; MlpW mw[32]; float *lni[32], *lnp[32];
    printf("Loading weights...\n"); fflush(stdout);
    for (int l = 0; l < 32; l++) {
        char p[256]; bool il = (LTYPE[l][0]=='l');
        if (il) {
            snprintf(p,256,"%d_linear_attn.in_proj_qkv",l); lw[l].qkv = load_int8(WDIR,p);
            snprintf(p,256,"%d_linear_attn.in_proj_a",l); lw[l].a = load_int8(WDIR,p);
            snprintf(p,256,"%d_linear_attn.in_proj_b",l); lw[l].b = load_int8(WDIR,p);
            snprintf(p,256,"%d_linear_attn.in_proj_z",l); lw[l].z = load_int8(WDIR,p);
            snprintf(p,256,"%d_linear_attn.out_proj",l); lw[l].out = load_int8(WDIR,p);
            snprintf(p,256,"%d_linear_attn.A_log",l); lw[l].A_log = load_f32(WDIR,p,NV);
            snprintf(p,256,"%d_linear_attn.dt_bias",l); lw[l].dt_bias = load_f32(WDIR,p,NV);
            snprintf(p,256,"%d_linear_attn.norm",l); lw[l].norm_w = load_f32(WDIR,p,HD);
            snprintf(p,256,"%d_linear_attn.conv1d.weight",l); lw[l].conv_w = load_bf16(WDIR,p,CD*CK);
        } else {
            snprintf(p,256,"%d_self_attn.q_proj",l); fw[l].q = load_int8(WDIR,p);
            snprintf(p,256,"%d_self_attn.k_proj",l); fw[l].k = load_int8(WDIR,p);
            snprintf(p,256,"%d_self_attn.v_proj",l); fw[l].v = load_int8(WDIR,p);
            snprintf(p,256,"%d_self_attn.o_proj",l); fw[l].o = load_int8(WDIR,p);
            snprintf(p,256,"%d_self_attn.q_norm",l); fw[l].qn = load_f32(WDIR,p,HDA);
            snprintf(p,256,"%d_self_attn.k_norm",l); fw[l].kn = load_f32(WDIR,p,HDA);
        }
        snprintf(p,256,"%d_mlp.gate_proj",l); mw[l].gate = load_int8(WDIR,p);
        snprintf(p,256,"%d_mlp.up_proj",l); mw[l].up = load_int8(WDIR,p);
        snprintf(p,256,"%d_mlp.down_proj",l); mw[l].down = load_int8(WDIR,p);
        snprintf(p,256,"%d_input_layernorm",l); lni[l] = load_f32(WDIR,p,H);
        snprintf(p,256,"%d_post_attention_layernorm",l); lnp[l] = load_f32(WDIR,p,H);
        if ((l+1)%8==0) printf("  %d/32\n", l+1);
    }
    float* fn = load_f32(WDIR,"final_norm",H); printf("Done.\n\n");

    // ── Buffers (M=8 batched) ──
    float *d_x, *d_xn, *d_proj;
    int8_t *d_ai; float *d_as;
    float *d_qkv, *d_qkvc, *d_g, *d_beta, *d_ao, *d_z;
    float *d_cs, *d_rs;
    float *dQ, *dK, *dV, *d_kc, *d_vc;
    int *d_step; int h_step = 0;

    chk(cudaMalloc(&d_x, M * H * 4));
    chk(cudaMalloc(&d_xn, M * H * 4));
    chk(cudaMalloc(&d_proj, M * H * 4));
    chk(cudaMalloc(&d_ai, M * H));
    chk(cudaMalloc(&d_as, M * (H/16) * 4));
    chk(cudaMalloc(&d_qkv, M * CD * 4));  // per-seq temp (conv input)
    chk(cudaMalloc(&d_qkvc, M * CD * 4));  // per-seq temp (conv output)
    // Contiguous arrays for batched recurrent_step
    float *d_q_cnt, *d_k_cnt, *d_v_cnt;
    chk(cudaMalloc(&d_q_cnt, M * NK * HD * 4));
    chk(cudaMalloc(&d_k_cnt, M * NK * HD * 4));
    chk(cudaMalloc(&d_v_cnt, M * NV * HD * 4));
    chk(cudaMalloc(&d_g, M * NV * 4));
    chk(cudaMalloc(&d_beta, M * NV * 4));
    chk(cudaMalloc(&d_ao, M * NV * HD * 4));
    chk(cudaMalloc(&d_z, M * H * 4));
    // SSM state: [M, 32 layers, CD, CK-1]
    // Conv state: [32, M, CD, CK-1] — per-layer contiguous
    chk(cudaMalloc(&d_cs, 32 * M * CD * (CK-1) * 4));
    cudaMemset(d_cs, 0, 32 * M * CD * (CK-1) * 4);
    // Recurrent state: [32, M, NV, HD, HD] — per-layer contiguous for batched call
    chk(cudaMalloc(&d_rs, 32 * M * NV * HD * HD * 4));
    cudaMemset(d_rs, 0, 32 * M * NV * HD * HD * 4);
    // Full attention buffers
    chk(cudaMalloc(&dQ, M * NQ * HDA * 4));
    chk(cudaMalloc(&dK, M * NKV * HDA * 4));
    chk(cudaMalloc(&dV, M * NKV * HDA * 4));
    size_t kvs = (size_t)8 * NKV * MS * HDA * 4;  // 8 full attn layers
    chk(cudaMalloc(&d_kc, kvs)); chk(cudaMalloc(&d_vc, kvs));
    cudaMemset(d_kc, 0, kvs); cudaMemset(d_vc, 0, kvs);

    // Per-sequence MLP temp buffers (reused in per-seq loop)
    // Broadcast temp buffers (recurrent_step: NK→NV)
    float *d_q_bc, *d_k_bc;
    chk(cudaMalloc(&d_q_bc, M * NV * HD * 4));
    chk(cudaMalloc(&d_k_bc, M * NV * HD * 4));
    // Per-seq MLP temp buffers
    float *d_mlp_res; int8_t *d_mlp_ai; float *d_mlp_as;
    chk(cudaMalloc(&d_mlp_res, I * 2 * 4));
    chk(cudaMalloc(&d_mlp_ai, I));
    chk(cudaMalloc(&d_mlp_as, (I/16) * 4));

    // Init state
    float *d_init; chk(cudaMalloc(&d_init, H*4));
    std::vector<float> init(H, 0.1f);
    cudaMemcpy(d_init, init.data(), H*4, cudaMemcpyHostToDevice);
    for (int m = 0; m < M; m++) cudaMemcpy(d_x + m*H, d_init, H*4, cudaMemcpyDeviceToDevice);
    chk(cudaMalloc(&d_step, 4));

    cudaStream_t st; chk(cudaStreamCreate(&st));
    // VRAM check
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("  VRAM: free=%.1fGB total=%.1fGB\n", free_mem/1e9, total_mem/1e9);
    printf("Decoding %d tokens (M=%d)...\n", NT, M); fflush(stdout);
    GpuTimer timer; timer.start(st);
    int fai = 0;

    for (int step = 0; step < NT; step++) {
        h_step = step; cudaMemcpy(d_step, &h_step, 4, cudaMemcpyHostToDevice);
        fai = 0;
        for (int l = 0; l < 32; l++) {
            bool il = (LTYPE[l][0] == 'l');
            // RMSNorm from FP32 state for all M
            for (int m = 0; m < M; m++) {
                chk(blackwell::kernels::fused_rmsnorm(d_xn + m*H, d_x + m*H, lni[l], H, 1e-6f, st), "rn");
            }

            if (il) {
                // ── Linear attention layer (GatedDeltaNet) ──
                // Quantize + project per-seq
                for (int m = 0; m < M; m++) {
                    chk(blackwell::kernels::quantize_int8(d_ai + m*H, d_as + m*(H/16), d_xn + m*H, H, st), "lq");
                    chk(blackwell::kernels::gemv_int8_warp(d_qkv + m*CD, d_ai + m*H, d_as + m*(H/16),
                        lw[l].qkv.d, lw[l].qkv.sc, H, CD, st), "lqkv");
                    chk(blackwell::kernels::gemv_int8_warp(d_g + m*NV, d_ai + m*H, d_as + m*(H/16),
                        lw[l].a.d, lw[l].a.sc, H, NV, st), "la");
                    chk(blackwell::kernels::gemv_int8_warp(d_beta + m*NV, d_ai + m*H, d_as + m*(H/16),
                        lw[l].b.d, lw[l].b.sc, H, NV, st), "lb");
                    chk(blackwell::kernels::gemv_int8_warp(d_z + m*H, d_ai + m*H, d_as + m*(H/16),
                        lw[l].z.d, lw[l].z.sc, H, H, st), "lz");
                }
                for (int m = 0; m < M; m++) {
                    chk(blackwell::kernels::gated_delta_conv1d_update(
                        d_cs + l * M * CD * (CK-1) + m * CD * (CK-1),
                        d_qkv + m*CD, lw[l].conv_w, d_qkvc + m*CD, st), "lcv");
                    cudaMemcpyAsync(d_q_cnt + m*NK*HD, d_qkvc + m*CD, NK*HD*4, cudaMemcpyDeviceToDevice, st);
                    cudaMemcpyAsync(d_k_cnt + m*NK*HD, d_qkvc + m*CD + NK*HD, NK*HD*4, cudaMemcpyDeviceToDevice, st);
                    cudaMemcpyAsync(d_v_cnt + m*NV*HD, d_qkvc + m*CD + NK*HD*2, NV*HD*4, cudaMemcpyDeviceToDevice, st);
                }
                // Compute g = -A_log.exp() * softplus(a + dt_bias) for all M
                for (int b = 0; b < M; b++) {
                    compute_g_k<<<(NV+255)/256, 256, 0, st>>>(
                        d_g + b*NV, d_g + b*NV, lw[l].A_log, lw[l].dt_bias, NV);
                }
                // Batched recurrent step
                chk(blackwell::kernels::gated_delta_recurrent_step(
                    d_q_cnt, d_k_cnt, d_v_cnt,
                    d_g, d_beta,
                    d_q_bc, d_k_bc, d_rs + l * M * NV * HD * HD, d_ao,
                    M, st), "lrs");
                // Batched RMSNormGated
                chk(blackwell::kernels::gated_delta_rmsnorm_gated(
                    d_proj, d_ao, d_z, lw[l].norm_w, M, 1e-6f, st), "lng");
                // Quantize + out_proj per-seq
                for (int m = 0; m < M; m++) {
                    chk(blackwell::kernels::quantize_int8(d_ai + m*H, d_as + m*(H/16), d_proj + m*H, H, st), "lqo");
                    chk(blackwell::kernels::gemv_int8_warp(d_proj + m*H, d_ai + m*H, d_as + m*(H/16),
                        lw[l].out.d, lw[l].out.sc, H, H, st), "lo");
                }
            } else {
                // ── Full attention layer ──
                for (int m = 0; m < M; m++) {
                    chk(blackwell::kernels::quantize_int8(d_ai + m*H, d_as + m*(H/16), d_xn + m*H, H, st), "fq");
                    chk(blackwell::kernels::gemv_int8_warp(dQ + m*NQ*HDA, d_ai + m*H, d_as + m*(H/16),
                        fw[l].q.d, fw[l].q.sc, H, NQ*HDA, st), "fqv");
                    chk(blackwell::kernels::gemv_int8_warp(dK + m*NKV*HDA, d_ai + m*H, d_as + m*(H/16),
                        fw[l].k.d, fw[l].k.sc, H, NKV*HDA, st), "fkv");
                    chk(blackwell::kernels::gemv_int8_warp(dV + m*NKV*HDA, d_ai + m*H, d_as + m*(H/16),
                        fw[l].v.d, fw[l].v.sc, H, NKV*HDA, st), "fvv");
                    head_norm_k<<<M*NQ, 128, 0, st>>>(dQ, fw[l].qn, M*NQ, HDA, 1e-6f);
                    head_norm_k<<<M*NKV, 128, 0, st>>>(dK, fw[l].kn, M*NKV, HDA, 1e-6f);
                }
                // RoPE per-seq (has device seq_pos read)
                for (int m = 0; m < M; m++) {
                    int* sp = d_step;  // same for all
                    rope_k<<<NQ, PD/2, 0, st>>>(dQ + m*NQ*HDA, NQ, HDA, sp, PD);
                    rope_k<<<NKV, PD/2, 0, st>>>(dK + m*NKV*HDA, NKV, HDA, sp, PD);
                }
                // Update KV cache per-seq
                int ko = fai * NKV * MS * HDA;
                for (int m = 0; m < M; m++) {
                    chk(blackwell::kernels::update_kv_cache(d_kc+ko, d_vc+ko,
                        dK + m*NKV*HDA, dV + m*NKV*HDA, 0, step, NKV, HDA, MS, st), "fkvc");
                }
                // Batched attention (single kernel, all M)
                chk(blackwell::kernels::attention_decode_batched_gqa(d_proj, dQ, d_kc, d_vc,
                    step, NQ, NKV, HDA, MS, M, 0, ko, st), "fatt");
                fai++;
                // Quantize + O projection per-seq
                for (int m = 0; m < M; m++) {
                    chk(blackwell::kernels::quantize_int8(d_ai + m*H, d_as + m*(H/16), d_proj + m*H, H, st), "fqo");
                    chk(blackwell::kernels::gemv_int8_warp(d_proj + m*H, d_ai + m*H, d_as + m*(H/16),
                        fw[l].o.d, fw[l].o.sc, H, H, st), "fo");
                }
            }

            // ── MLP (per-seq, three GEMVs) ──
            for (int m = 0; m < M; m++) {
                // Residual add
                chk(blackwell::kernels::vector_add_fp32(d_x + m*H, d_x + m*H, d_proj + m*H, H, st), "r1");
                // Post-attention norm
                chk(blackwell::kernels::fused_rmsnorm(d_xn + m*H, d_x + m*H, lnp[l], H, 1e-6f, st), "rnp");
                // Gate + Up GEMVs
                chk(blackwell::kernels::quantize_int8(d_ai + m*H, d_as + m*(H/16), d_xn + m*H, H, st), "mq");
                chk(blackwell::kernels::gemv_int8_warp(d_mlp_res, d_ai + m*H, d_as + m*(H/16),
                    mw[l].gate.d, mw[l].gate.sc, H, I, st), "mg");
                chk(blackwell::kernels::gemv_int8_warp(d_mlp_res + I, d_ai + m*H, d_as + m*(H/16),
                    mw[l].up.d, mw[l].up.sc, H, I, st), "mu");
                chk(blackwell::kernels::apply_swiglu(d_mlp_res, d_mlp_res, d_mlp_res + I, I, st), "ms");
                chk(blackwell::kernels::quantize_int8(d_mlp_ai, d_mlp_as, d_mlp_res, I, st), "mdq");
                chk(blackwell::kernels::gemv_int8_warp(d_proj + m*H, d_mlp_ai, d_mlp_as,
                    mw[l].down.d, mw[l].down.sc, I, H, st), "md");
                chk(blackwell::kernels::vector_add_fp32(d_x + m*H, d_x + m*H, d_proj + m*H, H, st), "r2");
            }
        }
        if (step % 5 == 0) { cudaStreamSynchronize(st); printf("  %d/%d\n", step+1, NT); fflush(stdout); }
    }
    float ms = timer.stop(), pt = ms / NT, tps = M * 1000.f / pt;
    printf("\nResults: %.1f ms total, %.2f ms/step, %.1f total t/s (%.0f%% of 71.4)\n", ms, pt, tps, tps/71.4f*100.f);
    printf("  Per-seq: %.2f ms/seq = %.1f seqs/s\n", pt/M, 1000.f*M/pt);
    return 0;
}