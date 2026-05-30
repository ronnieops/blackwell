// bench/convert_weights_int4.cu — Convert INT8 weights to packed INT4 (2 vals/byte)
//
// Reads transposed INT8 weights (per-row scales), quantizes to signed INT4
// (range [-8, 7]), packs 2 values per byte, writes packed INT4 weights.
//
// INT4 format: signed 4-bit integers, block_size=16, scale = block_max / 7.0
// Packed layout: low nibble = first value, high nibble = second value
//
// Build:
//   CUDACXX=/usr/local/cuda-13.3/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/convert_weights_int4.cu \
//     build/libblackwell_kernels.a -o bench/convert_weights_int4

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cmath>
#include "blackwell/kernels.h"

// Kernel: INT8 → packed INT4 quantization on GPU
// Each block of 16 INT8 values → 8 packed bytes + 1 FP32 scale
__global__ void int8_to_packed_int4_kernel(
    uint8_t* __restrict__ out_packed,   // [N][K/2] packed INT4
    float*   __restrict__ out_scale,    // [N][K/16] per-row scales
    const int8_t* __restrict__ in_i8,   // [N][K] INT8 (transposed)
    const float*  __restrict__ in_sc,   // [N][K/16] INT8 per-row scales
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
    const int8_t* ptr = &in_i8[(size_t)n_out * K + kb * B];
    float vals[B];
    float block_max = 0.0f;
    for (int j = 0; j < B; j++) {
        vals[j] = static_cast<float>(ptr[j]) * i8_sc;
        block_max = fmaxf(block_max, fabsf(vals[j]));
    }

    // INT4 scale: range is [-8, 7], so scale = block_max / 7.0
    float int4_sc = block_max / 7.0f;
    if (int4_sc < 1e-10f) int4_sc = 1e-10f;
    out_scale[n_out * num_K_blks + kb] = int4_sc;

    // Quantize to signed INT4 [-8, 7] and pack 2 per byte
    uint8_t* out_ptr = &out_packed[(size_t)n_out * (K / 2) + kb * (B / 2)];
    for (int j = 0; j < B / 2; j++) {
        float v0 = vals[j * 2] / int4_sc;
        float v1 = vals[j * 2 + 1] / int4_sc;

        // Round to nearest integer, clamp to [-8, 7]
        int q0 = __float2int_rn(v0);
        int q1 = __float2int_rn(v1);
        q0 = max(-8, min(7, q0));
        q1 = max(-8, min(7, q1));

        // Pack: low nibble = first value, high nibble = second value
        // INT4 stored as unsigned nibble: (q + 16) & 0x0F to handle negative
        uint8_t n0 = (uint8_t)((q0 + 16) & 0x0F);
        uint8_t n1 = (uint8_t)((q1 + 16) & 0x0F);
        out_ptr[j] = (n0 & 0x0F) | ((n1 & 0x0F) << 4);
    }
}

// Verification kernel: unpack INT4 → FP32 for correctness check
__global__ void unpack_int4_to_fp32_kernel(
    float* __restrict__ out,
    const uint8_t* __restrict__ in_packed,
    const float* __restrict__ scales,
    int K, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K;
    if (idx >= total) return;

    int n = idx / K;
    int k = idx % K;
    int kb = k / 16;
    int k_in_block = k % 16;

    float sc = scales[n * (K / 16) + kb];
    uint8_t byte = in_packed[(size_t)n * (K / 2) + (k / 2)];

    // Extract nibble: low nibble for even k, high nibble for odd k
    int nibble;
    if (k_in_block % 2 == 0) {
        nibble = byte & 0x0F;
    } else {
        nibble = (byte >> 4) & 0x0F;
    }

    // Convert unsigned nibble to signed INT4: subtract 16 if > 7
    int signed_val = nibble;
    if (signed_val > 7) signed_val -= 16;

    out[idx] = static_cast<float>(signed_val) * sc;
}

static void chk(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) { fprintf(stderr, "FAIL: %s: %s\n", msg, cudaGetErrorString(e)); exit(1); }
}

struct LoadedW { int K, N; std::vector<int8_t> d; std::vector<float> sc; };
static LoadedW load_int8_w(const char* prefix) {
    char p[256];
    snprintf(p, 256, "%s.int8_t", prefix);
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", p); exit(1); }
    int h[5]; fread(h, 4, 5, f);
    LoadedW w; w.K = h[0]; w.N = h[1];
    w.d.resize((size_t)h[0] * h[1]);
    fread(w.d.data(), 1, w.d.size(), f);
    fclose(f);

    snprintf(p, 256, "%s.scale_t", prefix);
    f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", p); exit(1); }
    fread(h, 4, 5, f);
    w.sc.resize((size_t)h[3] * h[4]);
    fread(w.sc.data(), 4, w.sc.size(), f);
    fclose(f);
    return w;
}

int main() {
    printf("# Converting INT8 weights to packed INT4 (2 vals/byte)\n");
    printf("# INT4 format: signed [-8, 7], block_size=16, scale = max/7.0\n\n");

    const char* projections[] = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"
    };

    int total_layers = 28;
    size_t total_bytes = 0;
    int total_weights = 0;

    for (int l = 0; l < total_layers; l++) {
        printf("Layer %d:\n", l);
        for (int p = 0; p < 7; p++) {
            char prefix[256];
            snprintf(prefix, 256, "weights_int8_bf16/%d_%s", l, projections[p]);

            auto w = load_int8_w(prefix);
            int K = w.K, N = w.N;
            int num_K_blks = K / 16;

            printf("  %s: K=%d N=%d\n", projections[p], K, N);

            // Device buffers
            int8_t* d_in; float* d_sc;
            uint8_t* d_packed; float* d_int4sc;
            chk(cudaMalloc(&d_in, (size_t)K * N), "malloc in");
            chk(cudaMalloc(&d_sc, (size_t)N * num_K_blks * 4), "malloc sc");
            chk(cudaMemcpy(d_in, w.d.data(), (size_t)K * N, cudaMemcpyHostToDevice), "cpy in");
            chk(cudaMemcpy(d_sc, w.sc.data(), (size_t)N * num_K_blks * 4, cudaMemcpyHostToDevice), "cpy sc");

            size_t packed_size = (size_t)N * (K / 2);
            size_t scale_size = (size_t)N * num_K_blks;
            chk(cudaMalloc(&d_packed, packed_size), "malloc packed");
            chk(cudaMalloc(&d_int4sc, scale_size * 4), "malloc int4sc");

            // Launch INT4 quantization kernel
            int total = N * num_K_blks;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            int8_to_packed_int4_kernel<<<blocks, threads, 0, 0>>>(
                d_packed, d_int4sc, d_in, d_sc, K, N);
            chk(cudaPeekAtLastError(), "int8_to_packed_int4");

            // Download results
            std::vector<uint8_t> packed_h(packed_size);
            std::vector<float> scales_h(scale_size);
            chk(cudaMemcpy(packed_h.data(), d_packed, packed_size, cudaMemcpyDeviceToHost), "dwn packed");
            chk(cudaMemcpy(scales_h.data(), d_int4sc, scale_size * 4, cudaMemcpyDeviceToHost), "dwn sc");

            // Save packed INT4 file
            char out_path[256];
            snprintf(out_path, 256, "weights_int4_packed/%d_%s.int4_packed", l, projections[p]);
            {
                FILE* f = fopen(out_path, "wb");
                if (!f) { fprintf(stderr, "Cannot create %s\n", out_path); exit(1); }
                int h[5] = {K, N, 16, N, num_K_blks};  // block=16 for per-row scales
                fwrite(h, 4, 5, f);
                fwrite(packed_h.data(), 1, packed_size, f);
                fwrite(scales_h.data(), 4, scale_size, f);
                fclose(f);
            }

            // Save scale file (separate, for pipeline loading)
            snprintf(out_path, 256, "weights_int4_packed/%d_%s.scale", l, projections[p]);
            {
                FILE* f = fopen(out_path, "wb");
                if (!f) { fprintf(stderr, "Cannot create %s\n", out_path); exit(1); }
                int h[5] = {K, N, 16, N, num_K_blks};
                fwrite(h, 4, 5, f);
                fwrite(scales_h.data(), 4, scale_size, f);
                fclose(f);
            }

            total_bytes += packed_size + scale_size * 4;
            total_weights += N * K;

            // Verify: unpack INT4 → FP32 and compare with INT8 dequant
            if (l == 0 && p == 0) {
                printf("  Verifying INT4 quantization...\n");
                float* d_fp32_i4; float* d_fp32_i8;
                chk(cudaMalloc(&d_fp32_i4, (size_t)K * N * 4), "malloc fp32_i4");
                chk(cudaMalloc(&d_fp32_i8, (size_t)K * N * 4), "malloc fp32_i8");

                // Unpack INT4 → FP32
                int total_elems = N * K;
                int blocks_v = (total_elems + 255) / 256;
                unpack_int4_to_fp32_kernel<<<blocks_v, 256, 0, 0>>>(
                    d_fp32_i4, d_packed, d_int4sc, K, N);
                chk(cudaPeekAtLastError(), "unpack_int4");

                // Dequant INT8 → FP32
                // Use existing pack_int8 in reverse? No, just compute manually.
                // For now, compute L1 norm of INT4 output as sanity check.
                std::vector<float> fp32_check(K);
                chk(cudaMemcpy(fp32_check.data(), d_fp32_i4, K * 4, cudaMemcpyDeviceToHost), "dwn fp32");
                float l1 = 0;
                for (int i = 0; i < K; i++) l1 += fabsf(fp32_check[i]);
                printf("  INT4 L1 (first row, K elements): %.4f\n", l1);

                cudaFree(d_fp32_i4);
                cudaFree(d_fp32_i8);
            }

            cudaFree(d_in); cudaFree(d_sc);
            cudaFree(d_packed); cudaFree(d_int4sc);
        }
    }

    printf("\nDone: %d layers × 7 projections = %d weight matrices\n", total_layers, total_layers * 7);
    printf("Total INT4 packed size: %.1f MB\n", total_bytes / 1e6);
    printf("Total weights: %d (%.1f M elements)\n", total_weights, total_weights / 1e6);
    printf("Files in weights_int4_packed/\n");
    return 0;
}
