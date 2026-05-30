// bench/convert_weights_packed_fp4.cu — Convert INT8 weights to packed FP4 (2 vals/byte)
//
// Reads INT8 weights (per-row scales), quantizes to FP4 E2M1,
// packs 2 values per byte, writes packed FP4 weights with per-row scales.
//
// Build:
//   nvcc -O3 -std=c++17 -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/convert_weights_packed_fp4.cu \
//     build/libblackwell_kernels.a -o bench/convert_weights_packed_fp4

#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include "blackwell/kernels.h"

// Kernel: INT8 → FP4 packed quantization on GPU
// Each block of 16 INT8 values → 8 packed bytes + 1 FP32 scale
__global__ void int8_to_packed_fp4_kernel(
    uint8_t* __restrict__ out_packed,   // [N][K/2] packed FP4
    float*   __restrict__ out_scale,    // [N][K/16] per-row scales
    const int8_t* __restrict__ in_i8,   // [N][K] INT8
    const float*  __restrict__ in_sc,   // [N][K/16] INT8 scales (per-row)
    int K, int N)
{
    constexpr int B = 16;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_K_blks = K / B;
    int total = N * num_K_blks;

    if (idx >= total) return;

    int n_out = idx / num_K_blks;
    int kb = idx % num_K_blks;

    // Load INT8 scale for this block
    float i8_sc = in_sc[n_out * num_K_blks + kb];

    // Load 16 INT8 values and dequantize to FP32
    const int8_t* ptr = &in_i8[n_out * K + kb * B];
    float vals[B];
    float block_max = 0.0f;
    for (int j = 0; j < B; j++) {
        vals[j] = static_cast<float>(ptr[j]) * i8_sc;
        block_max = fmaxf(block_max, fabsf(vals[j]));
    }

    // FP4 E2M1 scale: range is ±3, so scale = block_max / 3.0
    float fp4_sc = block_max / 3.0f;
    if (fp4_sc < 1e-10f) fp4_sc = 1e-10f;
    out_scale[n_out * num_K_blks + kb] = fp4_sc;

    // Quantize to FP4 E2M1 and pack 2 per byte
    uint8_t* out_ptr = &out_packed[(size_t)n_out * (K / 2) + kb * (B / 2)];
    for (int j = 0; j < B / 2; j++) {
        float v0 = vals[j * 2] / fp4_sc;
        float v1 = vals[j * 2 + 1] / fp4_sc;
        v0 = fmaxf(-3.0f, fminf(3.0f, v0));
        v1 = fmaxf(-3.0f, fminf(3.0f, v1));

        __nv_fp4_e2m1 f0(v0), f1(v1);
        uint8_t b0, b1;
        memcpy(&b0, &f0, 1);
        memcpy(&b1, &f1, 1);

        // Pack: low nibble = first value, high nibble = second value
        out_ptr[j] = (b0 & 0x0F) | ((b1 & 0x0F) << 4);
    }
}

struct LoadedW { int K, N; std::vector<int8_t> d; std::vector<float> sc; };
static LoadedW load_int8_w(const char* prefix) {
    char p[256]; snprintf(p,256,"%s.int8_t",prefix);
    FILE* f = fopen(p,"rb"); if(!f){fprintf(stderr,"FAIL open %s\n",p);exit(1);}
    int h[5]; fread(h,4,5,f);
    LoadedW w; w.K=h[0]; w.N=h[1]; w.d.resize(h[0]*h[1]); fread(w.d.data(),1,w.d.size(),f); fclose(f);
    snprintf(p,256,"%s.scale_t",prefix); f=fopen(p,"rb"); fread(h,4,5,f);
    w.sc.resize(h[3]*h[4]); fread(w.sc.data(),4,w.sc.size(),f); fclose(f);
    return w;
}

static void save_packed_fp4(const char* prefix, int K, int N,
    const std::vector<uint8_t>& packed, const std::vector<float>& scales) {
    char p[256];
    snprintf(p,256,"%s.packed_fp4",prefix);
    FILE* f = fopen(p,"wb"); if(!f){fprintf(stderr,"FAIL create %s\n",p);exit(1);}
    int h[5] = {K, N, K/16, N, (int)packed.size()};
    fwrite(h,4,5,f);
    fwrite(packed.data(),1,packed.size(),f);
    fwrite(scales.data(),4,scales.size(),f);
    fclose(f);
    printf("  Saved %s: %d bytes packed, %d scales\n", p, (int)packed.size(), (int)scales.size());
}

int main() {
    printf("# Converting INT8 weights to packed FP4 (2 vals/byte)\n\n");

    const char* projections[] = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
    };

    for (int l = 0; l < 28; l++) {
        printf("Layer %d:\n", l);
        for (int p = 0; p < 7; p++) {
            char prefix[256];
            snprintf(prefix, 256, "weights_int8_bf16/%d_%s", l, projections[p]);

            auto w = load_int8_w(prefix);
            int K = w.K, N = w.N;

            // Device buffers
            int8_t* d_in; float* d_sc;
            uint8_t* d_packed; float* d_fp4sc;
            cudaMalloc(&d_in, K*N);
            cudaMalloc(&d_sc, N*(K/16)*4);
            cudaMemcpy(d_in, w.d.data(), K*N, cudaMemcpyHostToDevice);
            cudaMemcpy(d_sc, w.sc.data(), N*(K/16)*4, cudaMemcpyHostToDevice);

            size_t packed_size = (size_t)N * (K/2);
            size_t scale_size = (size_t)N * (K/16);
            cudaMalloc(&d_packed, packed_size);
            cudaMalloc(&d_fp4sc, scale_size * 4);

            int total = N * (K/16);
            int threads = 256;
            int blocks = (total + threads - 1) / threads;

            int8_to_packed_fp4_kernel<<<blocks, threads, 0, 0>>>(
                d_packed, d_fp4sc, d_in, d_sc, K, N);

            std::vector<uint8_t> packed_h(packed_size);
            std::vector<float> scales_h(scale_size);
            cudaMemcpy(packed_h.data(), d_packed, packed_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(scales_h.data(), d_fp4sc, scale_size*4, cudaMemcpyDeviceToHost);

            // Save
            char out_prefix[256];
            snprintf(out_prefix, 256, "weights_packed_fp4/%d_%s", l, projections[p]);
            save_packed_fp4(out_prefix, K, N, packed_h, scales_h);

            cudaFree(d_in); cudaFree(d_sc);
            cudaFree(d_packed); cudaFree(d_fp4sc);
        }
    }

    printf("\nDone. Files in weights_packed_fp4/\n");
    return 0;
}
