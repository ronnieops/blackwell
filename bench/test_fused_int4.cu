// bench/test_fused_int4.cu — Correctness test for fused_rmsnorm_quant_int4
// and fused_swiglu_quant_int4 at N=4096 and N=12288 (the previously broken size).
//
// Compares fused kernel output vs sequential (fused_rmsnorm + quantize_int4,
// apply_swiglu + quantize_int4). Verifies bit-identical (or near-identical)
// packed weights and scales.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_120a bench/test_fused_int4.cu \
//          build/libblackwell_kernels.a -I include -o bench/test_fused_int4 \
//          -lcudart -lpthread
// Run:   ./bench/test_fused_int4

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include "blackwell/kernels.h"

using blackwell::kernels::fused_rmsnorm_quant_int4;
using blackwell::kernels::fused_swiglu_quant_int4;
using blackwell::kernels::fused_rmsnorm;
using blackwell::kernels::apply_swiglu;
using blackwell::kernels::quantize_int4;

// CPU reference: RMSNorm + INT4 quant (block-16, absmax/7, offset-binary)
static void cpu_rmsnorm_quant_int4(const float* proj, const float* weight,
                                    uint8_t* packed, float* scales,
                                    int N, float eps) {
    std::vector<float> normed(N);
    float sumsq = 0.0f;
    for (int i = 0; i < N; ++i) sumsq += proj[i]*proj[i];
    float rstd = 1.0f / std::sqrt(sumsq / N + eps);
    for (int i = 0; i < N; ++i) normed[i] = proj[i] * weight[i] * rstd;

    int nblocks = N / 16;
    for (int b = 0; b < nblocks; ++b) {
        float am = 0.0f;
        for (int i = 0; i < 16; ++i) am = std::fmax(am, std::fabs(normed[b*16+i]));
        scales[b] = (am > 1e-10f) ? (am / 7.0f) : (1.0f / 7.0f);
    }
    for (int b = 0; b < N/2; ++b) {
        float lo = normed[b*2]   / scales[(b*2)/16];
        float hi = normed[b*2+1] / scales[(b*2+1)/16];
        int loq = (int)std::round(std::fmin(7,std::fmax(-8,lo)));
        int hiq = (int)std::round(std::fmin(7,std::fmax(-8,hi)));
        packed[b] = (uint8_t)(((loq+8)&0x0F) | (((hiq+8)&0x0F) << 4));
    }
}

// CPU reference: SwiGLU + INT4 quant
static void cpu_swiglu_quant_int4(const float* gate, const float* up,
                                   uint8_t* packed, float* scales, int N) {
    std::vector<float> act(N);
    for (int i = 0; i < N; ++i) {
        float s = 1.0f / (1.0f + std::exp(-gate[i]));
        act[i] = gate[i] * s * up[i];
    }
    int nblocks = N / 16;
    for (int b = 0; b < nblocks; ++b) {
        float am = 0.0f;
        for (int i = 0; i < 16; ++i) am = std::fmax(am, std::fabs(act[b*16+i]));
        scales[b] = (am > 1e-10f) ? (am / 7.0f) : (1.0f / 7.0f);
    }
    for (int b = 0; b < N/2; ++b) {
        float lo = act[b*2]   / scales[(b*2)/16];
        float hi = act[b*2+1] / scales[(b*2+1)/16];
        int loq = (int)std::round(std::fmin(7,std::fmax(-8,lo)));
        int hiq = (int)std::round(std::fmin(7,std::fmax(-8,hi)));
        packed[b] = (uint8_t)(((loq+8)&0x0F) | (((hiq+8)&0x0F) << 4));
    }
}

int main() {
    int sizes[] = {4096, 12288};
    int nfail = 0;
    float eps = 1e-6f;

    for (int si = 0; si < 2; ++si) {
        int N = sizes[si];
        printf("══ N=%d ══\n", N);

        // Random inputs
        std::vector<float> proj(N), weight(N), gate(N), up(N);
        unsigned int seed = 42 + si;
        for (int i = 0; i < N; ++i) {
            proj[i]   = ((float)rand_r(&seed)/RAND_MAX - 0.5f) * 4.0f;
            weight[i] = ((float)rand_r(&seed)/RAND_MAX) * 0.5f + 0.5f;
            gate[i]   = ((float)rand_r(&seed)/RAND_MAX - 0.5f) * 6.0f;
            up[i]     = ((float)rand_r(&seed)/RAND_MAX - 0.5f) * 6.0f;
        }

        // CPU reference
        std::vector<uint8_t> ref_packed(N/2);
        std::vector<float> ref_scales(N/16);
        cpu_rmsnorm_quant_int4(proj.data(), weight.data(), ref_packed.data(),
                               ref_scales.data(), N, eps);

        // GPU fused
        float *d_proj, *d_weight, *d_gate, *d_up;
        uint8_t *d_out; float *d_scales;
        cudaMalloc(&d_proj, N*4);    cudaMemcpy(d_proj, proj.data(), N*4, cudaMemcpyHostToDevice);
        cudaMalloc(&d_weight, N*4);  cudaMemcpy(d_weight, weight.data(), N*4, cudaMemcpyHostToDevice);
        cudaMalloc(&d_gate, N*4);    cudaMemcpy(d_gate, gate.data(), N*4, cudaMemcpyHostToDevice);
        cudaMalloc(&d_up, N*4);      cudaMemcpy(d_up, up.data(), N*4, cudaMemcpyHostToDevice);
        cudaMalloc(&d_out, N/2);
        cudaMalloc(&d_scales, (N/16)*4);

        // --- RMSNorm fused ---
        cudaError_t e = fused_rmsnorm_quant_int4(d_out, d_scales, d_proj, d_weight, N, eps, 0);
        if (e != cudaSuccess) { printf("  RMSNorm launch FAIL: %d\n", e); ++nfail; continue; }
        cudaDeviceSynchronize();

        std::vector<uint8_t> got_packed(N/2);
        std::vector<float> got_scales(N/16);
        cudaMemcpy(got_packed.data(), d_out, N/2, cudaMemcpyDeviceToHost);
        cudaMemcpy(got_scales.data(), d_scales, (N/16)*4, cudaMemcpyDeviceToHost);

        // Compare scales
        float max_sc_diff = 0;
        for (int i = 0; i < N/16; ++i)
            max_sc_diff = std::fmax(max_sc_diff, std::fabs(got_scales[i]-ref_scales[i]));
        // Compare packed (count mismatches)
        int nmismatch = 0;
        for (int i = 0; i < N/2; ++i)
            if (got_packed[i] != ref_packed[i]) ++nmismatch;

        printf("  RMSNorm fused: scale max diff=%.6f, packed mismatches=%d/%d (%.1f%%)\n",
               max_sc_diff, nmismatch, N/2, 100.0f*nmismatch/(N/2));
        if (nmismatch == 0) printf("  ✅ RMSNorm fused: BIT-IDENTICAL to CPU ref\n");
        else { printf("  ⚠️  RMSNorm: %d nibble mismatches (FP rounding OK if <1%%)\n", nmismatch); ++nfail; }

        // --- SwiGLU fused ---
        cpu_swiglu_quant_int4(gate.data(), up.data(), ref_packed.data(), ref_scales.data(), N);
        e = fused_swiglu_quant_int4(d_out, d_scales, d_gate, d_up, N, 0);
        if (e != cudaSuccess) { printf("  SwiGLU launch FAIL: %d\n", e); ++nfail; continue; }
        cudaDeviceSynchronize();
        cudaMemcpy(got_packed.data(), d_out, N/2, cudaMemcpyDeviceToHost);
        cudaMemcpy(got_scales.data(), d_scales, (N/16)*4, cudaMemcpyDeviceToHost);

        max_sc_diff = 0;
        for (int i = 0; i < N/16; ++i)
            max_sc_diff = std::fmax(max_sc_diff, std::fabs(got_scales[i]-ref_scales[i]));
        nmismatch = 0;
        for (int i = 0; i < N/2; ++i)
            if (got_packed[i] != ref_packed[i]) ++nmismatch;

        printf("  SwiGLU fused: scale max diff=%.6f, packed mismatches=%d/%d (%.1f%%)\n",
               max_sc_diff, nmismatch, N/2, 100.0f*nmismatch/(N/2));
        if (nmismatch == 0) printf("  ✅ SwiGLU fused: BIT-IDENTICAL to CPU ref\n");
        else { printf("  ⚠️  SwiGLU: %d nibble mismatches\n", nmismatch); ++nfail; }

        cudaFree(d_proj); cudaFree(d_weight); cudaFree(d_gate); cudaFree(d_up);
        cudaFree(d_out); cudaFree(d_scales);
    }

    printf("\n══ Summary ══\n");
    printf("%s\n", nfail == 0 ? "✅ ALL TESTS PASSED" : "❌ TESTS FAILED");
    return nfail ? 1 : 0;
}
