// src/kernels/gemv_fp8.cu — FP8 E4M3 per-row-scaled GEMV
//
// Weight format: W_fp8 [N×K] as uint8_t (FP8 E4M3 values).
// Scale format:  W_scale [N] FP32 (one scale per output row).
// Activation:    x_fp32 [K] FP32.
//
// 1 warp per output row, 32 threads cooperatively iterate K/16 blocks.
// FP8 → FP32 dequant on the fly, FP32×FP32 multiply-accumulate.
//
// Quality: FP8 per-row PPL=41.75 on 1.7B (vs BF16 12.4, INT8 block-16 18.65).

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

// FP8 E4M3 → FP32 device conversion (inline)
__device__ __forceinline__ float fp8_e4m3_to_f32(uint8_t b) {
    int sign = (b >> 7) & 1;
    int exp  = (b >> 3) & 0xF;
    int mant = b & 0x7;
    float value;
    if (exp > 0) {
        value = (1.0f + (float)mant / 8.0f) * powf(2.0f, (float)(exp - 7));
    } else {
        // Subnormal
        value = ((float)mant / 8.0f) * powf(2.0f, -6.0f);
    }
    return sign ? -value : value;
}

// ---------------------------------------------------------------------------
// FP8 GEMV: FP32 activations × FP8 per-row-scaled weights → FP32 output
// 1 warp (32 threads) per output row
// ---------------------------------------------------------------------------
__launch_bounds__(32, 4)
__global__ void gemv_fp8_row_kernel(
    float* __restrict__       y_out,
    const float* __restrict__ x_fp32,
    const uint8_t* __restrict__ W_fp8,
    const float* __restrict__ W_scale,
    int K, int N)
{
    constexpr int B = 16;  // process 16 K-elements per iteration
    int n_out = blockIdx.x;
    int tid   = threadIdx.x;
    if (n_out >= N) return;

    int num_K_blks = K / B;
    float row_scale = W_scale[n_out];
    float acc = 0.0f;

    // Each thread processes every 32nd K-block (warp-stride loop)
    for (int kb = tid; kb < num_K_blks; kb += 32) {
        int x_off = kb * B;

        // Load 16 FP32 activation values (4 × float4 = 16 floats)
        float4 v0 = reinterpret_cast<const float4*>(&x_fp32[x_off])[0];
        float4 v1 = reinterpret_cast<const float4*>(&x_fp32[x_off])[1];
        float4 v2 = reinterpret_cast<const float4*>(&x_fp32[x_off])[2];
        float4 v3 = reinterpret_cast<const float4*>(&x_fp32[x_off])[3];

        // Load 16 FP8 weight values as 4 × uint4 (16 bytes)
        const uint4* w_ptr = reinterpret_cast<const uint4*>(&W_fp8[n_out * K + x_off]);
        uint4 w_pack = *w_ptr;
        const uint8_t* w_bytes = reinterpret_cast<const uint8_t*>(&w_pack);

        // Dequant FP8 → FP32 and dot product
        float sum_b = 0.0f;
        sum_b += fp8_e4m3_to_f32(w_bytes[0])  * v0.x;
        sum_b += fp8_e4m3_to_f32(w_bytes[1])  * v0.y;
        sum_b += fp8_e4m3_to_f32(w_bytes[2])  * v0.z;
        sum_b += fp8_e4m3_to_f32(w_bytes[3])  * v0.w;
        sum_b += fp8_e4m3_to_f32(w_bytes[4])  * v1.x;
        sum_b += fp8_e4m3_to_f32(w_bytes[5])  * v1.y;
        sum_b += fp8_e4m3_to_f32(w_bytes[6])  * v1.z;
        sum_b += fp8_e4m3_to_f32(w_bytes[7])  * v1.w;
        sum_b += fp8_e4m3_to_f32(w_bytes[8])  * v2.x;
        sum_b += fp8_e4m3_to_f32(w_bytes[9])  * v2.y;
        sum_b += fp8_e4m3_to_f32(w_bytes[10]) * v2.z;
        sum_b += fp8_e4m3_to_f32(w_bytes[11]) * v2.w;
        sum_b += fp8_e4m3_to_f32(w_bytes[12]) * v3.x;
        sum_b += fp8_e4m3_to_f32(w_bytes[13]) * v3.y;
        sum_b += fp8_e4m3_to_f32(w_bytes[14]) * v3.z;
        sum_b += fp8_e4m3_to_f32(w_bytes[15]) * v3.w;

        acc += sum_b;
    }

    // Warp shuffle reduction
    acc += __shfl_xor_sync(0xffffffff, acc, 16);
    acc += __shfl_xor_sync(0xffffffff, acc, 8);
    acc += __shfl_xor_sync(0xffffffff, acc, 4);
    acc += __shfl_xor_sync(0xffffffff, acc, 2);
    acc += __shfl_xor_sync(0xffffffff, acc, 1);

    if (tid == 0) y_out[n_out] = acc * row_scale;
}

// ---------------------------------------------------------------------------
// FP8 GEMV with INT8 activation input: INT8 act × FP8 per-row weights → FP32
// For drop-in replacement of gemv_int8_warp (same activation format)
// ---------------------------------------------------------------------------
__launch_bounds__(32, 4)
__global__ void gemv_fp8_int8act_kernel(
    float* __restrict__       y_out,
    const int8_t* __restrict__ x_int8,
    const float* __restrict__ x_scale,
    const uint8_t* __restrict__ W_fp8,
    const float* __restrict__ W_scale,
    int K, int N)
{
    constexpr int B = 16;
    int n_out = blockIdx.x;
    int tid   = threadIdx.x;
    if (n_out >= N) return;

    int num_K_blks = K / B;
    float row_scale = W_scale[n_out];
    float acc = 0.0f;

    for (int kb = tid; kb < num_K_blks; kb += 32) {
        int x_off = kb * B;

        // Load 16 INT8 activation values
        const int8_t* x_ptr = &x_int8[x_off];
        int x0 = reinterpret_cast<const int*>(x_ptr)[0];
        int x1 = reinterpret_cast<const int*>(x_ptr)[1];
        int x2 = reinterpret_cast<const int*>(x_ptr)[2];
        int x3 = reinterpret_cast<const int*>(x_ptr)[3];
        float x_sc = x_scale[kb];

        // Load 16 FP8 weight values
        const uint4* w_ptr = reinterpret_cast<const uint4*>(&W_fp8[n_out * K + x_off]);
        uint4 w_pack = *w_ptr;
        const uint8_t* w_bytes = reinterpret_cast<const uint8_t*>(&w_pack);

        // INT8 sign-extend to float, multiply by FP8 dequant weight
        #define SE(i) (float)(int8_t)((i) & 0xFF)
        float sum_b = 0.0f;
        sum_b += fp8_e4m3_to_f32(w_bytes[0])  * SE(x0 >>  0);
        sum_b += fp8_e4m3_to_f32(w_bytes[1])  * SE(x0 >>  8);
        sum_b += fp8_e4m3_to_f32(w_bytes[2])  * SE(x0 >> 16);
        sum_b += fp8_e4m3_to_f32(w_bytes[3])  * SE(x0 >> 24);
        sum_b += fp8_e4m3_to_f32(w_bytes[4])  * SE(x1 >>  0);
        sum_b += fp8_e4m3_to_f32(w_bytes[5])  * SE(x1 >>  8);
        sum_b += fp8_e4m3_to_f32(w_bytes[6])  * SE(x1 >> 16);
        sum_b += fp8_e4m3_to_f32(w_bytes[7])  * SE(x1 >> 24);
        sum_b += fp8_e4m3_to_f32(w_bytes[8])  * SE(x2 >>  0);
        sum_b += fp8_e4m3_to_f32(w_bytes[9])  * SE(x2 >>  8);
        sum_b += fp8_e4m3_to_f32(w_bytes[10]) * SE(x2 >> 16);
        sum_b += fp8_e4m3_to_f32(w_bytes[11]) * SE(x2 >> 24);
        sum_b += fp8_e4m3_to_f32(w_bytes[12]) * SE(x3 >>  0);
        sum_b += fp8_e4m3_to_f32(w_bytes[13]) * SE(x3 >>  8);
        sum_b += fp8_e4m3_to_f32(w_bytes[14]) * SE(x3 >> 16);
        sum_b += fp8_e4m3_to_f32(w_bytes[15]) * SE(x3 >> 24);
        #undef SE

        acc += sum_b * x_sc;
    }

    // Warp shuffle reduction
    acc += __shfl_xor_sync(0xffffffff, acc, 16);
    acc += __shfl_xor_sync(0xffffffff, acc, 8);
    acc += __shfl_xor_sync(0xffffffff, acc, 4);
    acc += __shfl_xor_sync(0xffffffff, acc, 2);
    acc += __shfl_xor_sync(0xffffffff, acc, 1);

    if (tid == 0) y_out[n_out] = acc * row_scale;
}

// ---------------------------------------------------------------------------
// FP8 quantize: FP32 → FP8 E4M3 per-row scaling
// ---------------------------------------------------------------------------
__launch_bounds__(256, 1)
__global__ void quantize_fp8_row_kernel(
    uint8_t* __restrict__ out_fp8,
    float* __restrict__ out_scale,
    const float* __restrict__ x_fp32,
    int K, int N)
{
    // One block per row, 256 threads
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;

    // Find row max abs
    extern __shared__ float smem[];
    float my_max = 0.0f;
    for (int i = tid; i < K; i += blockDim.x) {
        float v = fabsf(x_fp32[row * K + i]);
        if (v > my_max) my_max = v;
    }
    smem[tid] = my_max;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if (tid < o && smem[tid + o] > smem[tid]) smem[tid] = smem[tid + o];
        __syncthreads();
    }
    float row_max = smem[0];
    __syncthreads();

    // Scale = max / 448 (FP8 E4M3 max)
    float scale = fmaxf(row_max / 448.0f, 1e-30f);
    if (tid == 0) out_scale[row] = scale;

    // Quantize: x_scaled = x / scale, then FP8 encode
    for (int i = tid; i < K; i += blockDim.x) {
        float val = x_fp32[row * K + i] / scale;
        // FP8 E4M3 encoding
        int sign = (val < 0) ? 1 : 0;
        float av = fabsf(val);
        av = fminf(av, 448.0f);

        int exp_b, mant;
        if (av == 0.0f) {
            exp_b = 0; mant = 0;
        } else {
            int exp_u = (int)floorf(log2f(av));
            exp_b = exp_u + 7;
            if (exp_b <= 0) {
                // Subnormal
                mant = (int)fminf(roundf(av * 512.0f), 7.0f);
                exp_b = 0;
            } else {
                float m_f = av * powf(2.0f, (float)(7 - exp_b)) * 8.0f - 8.0f;
                mant = (int)roundf(m_f);
                if (mant >= 8) { exp_b++; mant = 0; }
                if (mant < 0) mant = 0;
            }
        }
        exp_b = (exp_b > 15) ? 15 : exp_b;
        out_fp8[row * K + i] = (uint8_t)((sign << 7) | (exp_b << 3) | (mant & 0x7));
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

cudaError_t gemv_fp8_fp32act(
    float*        y_out,
    const float*  x_fp32,
    const void*   W_fp8,
    const float*  W_scale,
    int           K,
    int           N,
    cudaStream_t  stream)
{
    if (N <= 0 || K <= 0) return cudaSuccess;
    gemv_fp8_row_kernel<<<dim3(N), dim3(32), 0, stream>>>(
        y_out, x_fp32,
        reinterpret_cast<const uint8_t*>(W_fp8),
        W_scale, K, N);
    return cudaGetLastError();
}

cudaError_t gemv_fp8_int8act(
    float*        y_out,
    const void*   x_int8,
    const float*  x_scale,
    const void*   W_fp8,
    const float*  W_scale,
    int           K,
    int           N,
    cudaStream_t  stream)
{
    if (N <= 0 || K <= 0) return cudaSuccess;
    gemv_fp8_int8act_kernel<<<dim3(N), dim3(32), 0, stream>>>(
        y_out,
        reinterpret_cast<const int8_t*>(x_int8),
        x_scale,
        reinterpret_cast<const uint8_t*>(W_fp8),
        W_scale, K, N);
    return cudaGetLastError();
}

cudaError_t quantize_fp8_row(
    void*         out_fp8,
    float*        out_scale,
    const float*  x_fp32,
    int           K,
    int           N,
    cudaStream_t  stream)
{
    if (N <= 0 || K <= 0) return cudaSuccess;
    int shmem = sizeof(float) * 256;
    quantize_fp8_row_kernel<<<dim3(N), dim3(256), shmem, stream>>>(
        reinterpret_cast<uint8_t*>(out_fp8),
        out_scale,
        x_fp32, K, N);
    return cudaGetLastError();
}

} // namespace kernels
} // namespace blackwell
