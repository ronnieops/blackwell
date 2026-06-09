// src/kernels/gemv_int8.cu — INT8 block-scaled GEMV + pack + transpose
//
// INT8 GEMV path: eliminates FP4 cast overhead (static_cast<float> per element).
// Block-scaled quantization: weights in INT8 [-128..127], scales in FP32 per 16×16 block.
// Activations in INT8, scales per 16-element K-block.
//
// Weight format: W_t [N×K] INT8 (transposed row-major).
// Scale format:  W_scale [N/16 × K/16] FP32 per block (one scale per 16×16 block).
// Activation:    x_int8 [K], x_scale [K/16] FP32 per block.

#include <cuda_runtime.h>
#include <cuda/std/cmath>
#include <cuda_fp4.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace blackwell {
namespace kernels {
namespace {

constexpr int kINT8Block = 64;

// ---------------------------------------------------------------------------
// INT8 GEMV kernel — per-thread dot products, transposed weights
// ---------------------------------------------------------------------------
__launch_bounds__(kINT8Block, 1)
__global__ void gemv_int8_kernel(
    float* __restrict__ y_out,
    const int8_t* __restrict__ x_int8,
    const float* __restrict__ x_scale,
    const int8_t* __restrict__ W_t_int8,
    const float* __restrict__ W_t_scale,
    int K, int N)
{
    constexpr int B = 16;
    int tid = threadIdx.x;
    int n_out = blockIdx.x * kINT8Block + tid;
    if (n_out >= N) return;

    int num_K_blks = K / B;

    float acc = 0.0f;

    for (int kb = 0; kb < num_K_blks; ++kb) {
        // Load 16 INT8 weight + activation values via vectorized uint4 loads
        const int8_t* w_ptr = &W_t_int8[n_out * K + kb * B];
        alignas(16) int8_t w_buf[B];
        alignas(16) int8_t x_buf[B];
        *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);
        *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_int8 + kb * B);

        float w_sc = W_t_scale[n_out * num_K_blks + kb];
        float x_sc = x_scale[kb];
        float prod_scale = w_sc * x_sc;

        // __dp4a: 4-way int8 SIMD dot product per iteration (4 × 4 = 16 total)
        const int* w32 = reinterpret_cast<const int*>(w_buf);
        const int* x32 = reinterpret_cast<const int*>(x_buf);
        int sumi = 0;
        sumi = __dp4a(w32[0], x32[0], sumi);
        sumi = __dp4a(w32[1], x32[1], sumi);
        sumi = __dp4a(w32[2], x32[2], sumi);
        sumi = __dp4a(w32[3], x32[3], sumi);
        acc += static_cast<float>(sumi) * prod_scale;
    }

    y_out[n_out] = acc;
}

// ---------------------------------------------------------------------------
// Fused INT8 GEMV kernel — reads FP4 input, converts to INT8 inline
// Eliminates: unpack_fp4 + pack_int8 + gemv_int8 (3 launches → 1 launch)
// ---------------------------------------------------------------------------
__launch_bounds__(kINT8Block, 1)
__global__ void gemv_int8_from_fp4_kernel(
    float* __restrict__ y_out,
    const __nv_fp4_e2m1* __restrict__ x_fp4,
    const float* __restrict__ x_fp4_scale,      // [K/16]
    const int8_t* __restrict__ W_t_int8,
    const float* __restrict__ W_t_scale,        // [N/16 × K/16]
    int K, int N)
{
    constexpr int B = 16;
    int tid = threadIdx.x;
    int n_out = blockIdx.x * kINT8Block + tid;
    if (n_out >= N) return;

    int num_K_blks = K / B;

    float acc = 0.0f;

    for (int kb = 0; kb < num_K_blks; ++kb) {
        // Load 16 FP4 input values (16 bytes = uint4)
        alignas(16) __nv_fp4_e2m1 x_buf[B];
        *reinterpret_cast<uint4*>(x_buf) = 
            *reinterpret_cast<const uint4*>(&x_fp4[kb * B]);

        // Apply FP4 scales to get FP32 values and compute INT8 block scale
        float fp4_sc = x_fp4_scale[kb];
        float vals[B];
        float block_max = 0.0f;
        #pragma unroll
        for (int j = 0; j < B; ++j) {
            float v = static_cast<float>(x_buf[j]) * fp4_sc;
            vals[j] = v;
            float av = fabsf(v);
            if (av > block_max) block_max = av;
        }

        // Compute INT8 scale for this 16-element input block
        float i8_sc = block_max / 127.0f;
        if (i8_sc < 1e-10f) i8_sc = 1e-10f;

        // Load 16 INT8 weight values
        const int8_t* w_ptr = &W_t_int8[n_out * K + kb * B];
        alignas(16) int8_t w_buf[B];
        *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);

        float w_sc = W_t_scale[n_out * num_K_blks + kb];

        // Quantize x values to INT8 and accumulate
        #pragma unroll
        for (int j = 0; j < B; ++j) {
            float x_qf = roundf(vals[j] / i8_sc);
            x_qf = fminf(127.0f, fmaxf(-127.0f, x_qf));
            int8_t x_q = static_cast<int8_t>(static_cast<int>(x_qf));
            acc += static_cast<float>(x_q) * i8_sc *
                   static_cast<float>(w_buf[j]) * w_sc;
        }
    }

    y_out[n_out] = acc;
}

// ---------------------------------------------------------------------------
// INT8 pack kernel: FP32 → INT8 with per-block scales
// Block size = 16. Scale = absmax(block) / 127.0
// ---------------------------------------------------------------------------
__global__ void pack_int8_kernel(
    int8_t* __restrict__ out,
    const float* __restrict__ in,
    const float* __restrict__ scales,   // [num_block] pre-computed scales
    int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    int blk = idx / 16;
    float sc = scales[blk];
    float v = in[idx] / sc;
    v = fminf(127.0f, fmaxf(-127.0f, roundf(v)));
    out[idx] = static_cast<int8_t>(static_cast<int>(v));
}

// ---------------------------------------------------------------------------
// INT8 transpose: W (K×N) → W_t (N×K)
// ---------------------------------------------------------------------------
__global__ void transpose_int8_kernel(
    int8_t* __restrict__ dst,      // [N × K]
    const int8_t* __restrict__ src, // [K × N]
    int K, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * N;
    if (idx >= total) return;
    int k = idx / N;
    int n = idx % N;
    dst[n * K + k] = src[k * N + n];
}

// ---------------------------------------------------------------------------
// INT8 scale transpose: W_scale (K/16 × N/16) → W_t_scale (N/16 × K/16)
// ---------------------------------------------------------------------------
__global__ void transpose_scales_int8_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int num_K_blks, int num_N_blks)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_K_blks * num_N_blks;
    if (idx >= total) return;
    int kb = idx / num_N_blks;
    int nb = idx % num_N_blks;
    dst[nb * num_K_blks + kb] = src[kb * num_N_blks + nb];
}

// ===========================================================================
// Warp-cooperative INT8 GEMV — 1 warp per output row
// 32 threads cooperatively compute one dot product via shuffle reduction.
// Key benefit: all threads read same weight row → perfectly coalesced loads.
// Each thread handles scattered K-blocks (stride-32), reducing register pressure
// from 57 (full-row) to ~25 (K/32 elements per thread).
// ===========================================================================
__launch_bounds__(32, 8)
__global__ void gemv_int8_warp_kernel(
    float* __restrict__ y_out,
    const int8_t* __restrict__ x_int8,
    const float* __restrict__ x_scale,
    const int8_t* __restrict__ W_t_int8,
    const float* __restrict__ W_t_scale,
    int K, int N)
{
    constexpr int B = 16;
    int n_out = blockIdx.x;
    int tid = threadIdx.x;  // 0..31

    int num_K_blks = K / B;

    float acc = 0.0f;

    // Each thread processes K-blocks at stride-32: tid, tid+32, tid+64, ...
    // Warp-wide: 32 threads cover 32 consecutive K-blocks per iteration.
    // Memory access: all 32 threads read from same row n_out * K → coalesced.
    for (int kb = tid; kb < num_K_blks; kb += 32) {
        // Load 16 INT8 weight values (coalesced — same row, sequential offsets)
        const int8_t* w_ptr = &W_t_int8[n_out * K + kb * B];
        alignas(16) int8_t w_buf[B];
        *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);

        // Load 16 INT8 activation values (also coalesced across warp)
        alignas(16) int8_t x_buf[B];
        *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_int8 + kb * B);

        float w_sc = W_t_scale[n_out * num_K_blks + kb];
        float x_sc = x_scale[kb];
        float prod_scale = w_sc * x_sc;

        // dp4a: 4 × 4-way SIMD dot product
        const int* w32 = reinterpret_cast<const int*>(w_buf);
        const int* x32 = reinterpret_cast<const int*>(x_buf);
        int sumi = 0;
        sumi = __dp4a(w32[0], x32[0], sumi);
        sumi = __dp4a(w32[1], x32[1], sumi);
        sumi = __dp4a(w32[2], x32[2], sumi);
        sumi = __dp4a(w32[3], x32[3], sumi);
        acc += static_cast<float>(sumi) * prod_scale;
    }

    // Warp shuffle reduction: sum partial products across 32 lanes
    acc += __shfl_xor_sync(0xffffffff, acc, 16);
    acc += __shfl_xor_sync(0xffffffff, acc, 8);
    acc += __shfl_xor_sync(0xffffffff, acc, 4);
    acc += __shfl_xor_sync(0xffffffff, acc, 2);
    acc += __shfl_xor_sync(0xffffffff, acc, 1);

    // Thread 0 writes the final result
    if (tid == 0) y_out[n_out] = acc;
}

// ===========================================================================
// Warp-cooperative FP32×INT8 per-row GEMV
// FP32 activations × INT8 weights with per-row scales.
// Same warp-per-row strategy for coalesced weight access.
// ===========================================================================
__launch_bounds__(32, 8)
__global__ void gemv_fp32_int8_per_row_warp_kernel(
    float* __restrict__ y_out,
    const float* __restrict__ x_fp32,
    const int8_t* __restrict__ W_t_int8,
    const float* __restrict__ W_t_scale,
    int K, int N)
{
    constexpr int B = 16;
    int n_out = blockIdx.x;
    int tid = threadIdx.x;

    int num_K_blks = K / B;

    float acc = 0.0f;

    for (int kb = tid; kb < num_K_blks; kb += 32) {
        // Load 16 FP32 activation values
        int x_off = kb * B;
        float4 v0 = reinterpret_cast<const float4*>(&x_fp32[x_off])[0];
        float4 v1 = reinterpret_cast<const float4*>(&x_fp32[x_off])[1];
        float4 v2 = reinterpret_cast<const float4*>(&x_fp32[x_off])[2];
        float4 v3 = reinterpret_cast<const float4*>(&x_fp32[x_off])[3];

        // Load 16 INT8 weight values (coalesced)
        const int8_t* w_ptr = &W_t_int8[n_out * K + kb * B];
        int w0 = reinterpret_cast<const int*>(w_ptr)[0];
        int w1 = reinterpret_cast<const int*>(w_ptr)[1];
        int w2 = reinterpret_cast<const int*>(w_ptr)[2];
        int w3 = reinterpret_cast<const int*>(w_ptr)[3];

        float w_sc = W_t_scale[n_out * num_K_blks + kb];

        #define SE_W(i) (float)(int8_t)((i) & 0xFF)
        float sum_b = 0.0f;
        sum_b += SE_W(w0 >>  0) * v0.x;  sum_b += SE_W(w0 >>  8) * v0.y;
        sum_b += SE_W(w0 >> 16) * v0.z;  sum_b += SE_W(w0 >> 24) * v0.w;
        sum_b += SE_W(w1 >>  0) * v1.x;  sum_b += SE_W(w1 >>  8) * v1.y;
        sum_b += SE_W(w1 >> 16) * v1.z;  sum_b += SE_W(w1 >> 24) * v1.w;
        sum_b += SE_W(w2 >>  0) * v2.x;  sum_b += SE_W(w2 >>  8) * v2.y;
        sum_b += SE_W(w2 >> 16) * v2.z;  sum_b += SE_W(w2 >> 24) * v2.w;
        sum_b += SE_W(w3 >>  0) * v3.x;  sum_b += SE_W(w3 >>  8) * v3.y;
        sum_b += SE_W(w3 >> 16) * v3.z;  sum_b += SE_W(w3 >> 24) * v3.w;
        #undef SE_W
        acc += sum_b * w_sc;
    }

    // Warp shuffle reduction
    acc += __shfl_xor_sync(0xffffffff, acc, 16);
    acc += __shfl_xor_sync(0xffffffff, acc, 8);
    acc += __shfl_xor_sync(0xffffffff, acc, 4);
    acc += __shfl_xor_sync(0xffffffff, acc, 2);
    acc += __shfl_xor_sync(0xffffffff, acc, 1);

    if (tid == 0) y_out[n_out] = acc;
}

// ===========================================================================
// Warp-cooperative FP4 GEMV — 1 warp per output row, packed 2 vals/byte
// FP4 E2M1 weights packed as 2 nibbles per byte. Each byte holds 2 values.
// Weight layout: W_packed [N][K/2], scale layout: W_scale [N][K/16] FP32.
// 2× less bandwidth than INT8 for same K×N matrix.
// ===========================================================================

// Device-side: extract FP4 E2M1 nibble and convert to float
__device__ __forceinline__ float fp4_nibble_to_float(uint8_t packed, int nibble_idx) {
    // nibble_idx: 0 = low nibble, 1 = high nibble
    uint8_t raw = (nibble_idx == 0) ? (packed & 0x0F) : (packed >> 4);
    // E2M1: 1 sign + 2 exp + 1 mantissa
    // Values: 0=+0, 1=+0.5, 2=+1, 3=+2, 4=+3, 5=+inf, 6=NaN, 7=NaN
    //         8=-0, 9=-0.5, 10=-1, 11=-2, 12=-3, 13=-inf, 14=NaN, 15=NaN
    // Use __nv_fp4_e2m1 conversion via union
    __nv_fp4_e2m1 v;
    memcpy(&v, &raw, 1);  // only low byte matters
    return static_cast<float>(v);
}

__launch_bounds__(32, 8)
__global__ void gemv_fp4_warp_kernel(
    float* __restrict__ y_out,
    const uint8_t* __restrict__ x_packed,    // [K/2] packed FP4 activations
    const float* __restrict__ x_scale,       // [K/16] FP32 activation scales
    const uint8_t* __restrict__ W_packed,    // [N][K/2] packed FP4 weights
    const float* __restrict__ W_scale,       // [N][K/16] FP32 weight scales
    int K, int N)
{
    constexpr int B = 16;    // quantization block size
    constexpr int PB = 8;    // packed bytes per block (B/2)
    int n_out = blockIdx.x;
    int tid = threadIdx.x;

    int num_K_blks = K / B;
    float acc = 0.0f;

    // Stride-32 loop: each thread handles scattered K-blocks
    for (int kb = tid; kb < num_K_blks; kb += 32) {
        // Load 8 packed bytes = 16 FP4 values from weight row
        const uint8_t* w_ptr = &W_packed[(size_t)n_out * (K / 2) + kb * PB];
        alignas(16) uint8_t w_packed[PB];
        *reinterpret_cast<uint2*>(w_packed) = *reinterpret_cast<const uint2*>(w_ptr);

        // Load 8 packed bytes = 16 FP4 values from activation
        const uint8_t* x_ptr = &x_packed[kb * PB];
        alignas(16) uint8_t x_packed_buf[PB];
        *reinterpret_cast<uint2*>(x_packed_buf) = *reinterpret_cast<const uint2*>(x_ptr);

        float w_sc = W_scale[(size_t)n_out * num_K_blks + kb];
        float x_sc = x_scale[kb];
        float prod_scale = w_sc * x_sc;

        // Proper E2M1 conversion: nibble → __nv_fp4_e2m1 → float → accumulate
        float sum_f = 0.0f;
        #pragma unroll
        for (int j = 0; j < PB; ++j) {
            uint8_t lo_w = w_packed[j] & 0x0F;
            uint8_t hi_w = (w_packed[j] >> 4) & 0x0F;
            uint8_t lo_x = x_packed_buf[j] & 0x0F;
            uint8_t hi_x = (x_packed_buf[j] >> 4) & 0x0F;

            // Convert nibbles to FP4 via __nv_fp4_e2m1 cast
            __nv_fp4_e2m1 fw_lo, fw_hi, fx_lo, fx_hi;
            memcpy(&fw_lo, &lo_w, 1);
            memcpy(&fw_hi, &hi_w, 1);
            memcpy(&fx_lo, &lo_x, 1);
            memcpy(&fx_hi, &hi_x, 1);

            sum_f += static_cast<float>(fw_lo) * static_cast<float>(fx_lo);
            sum_f += static_cast<float>(fw_hi) * static_cast<float>(fx_hi);
        }
        acc += sum_f * prod_scale;
    }

    // Warp shuffle reduction
    acc += __shfl_xor_sync(0xffffffff, acc, 16);
    acc += __shfl_xor_sync(0xffffffff, acc, 8);
    acc += __shfl_xor_sync(0xffffffff, acc, 4);
    acc += __shfl_xor_sync(0xffffffff, acc, 2);
    acc += __shfl_xor_sync(0xffffffff, acc, 1);

    if (tid == 0) y_out[n_out] = acc;
}

// ===========================================================================
// Warp-cooperative FP4 GEMV — FP32 activations × packed FP4 weights
// Same as above but activations are FP32 (no quantization needed)
// ===========================================================================
__launch_bounds__(32, 8)
__global__ void gemv_fp32_fp4_warp_kernel(
    float* __restrict__ y_out,
    const float* __restrict__ x_fp32,        // [K] FP32 activations
    const uint8_t* __restrict__ W_packed,    // [N][K/2] packed FP4 weights
    const float* __restrict__ W_scale,       // [N][K/16] FP32 weight scales
    int K, int N)
{
    constexpr int B = 16;
    constexpr int PB = 8;
    int n_out = blockIdx.x;
    int tid = threadIdx.x;

    int num_K_blks = K / B;
    float acc = 0.0f;

    for (int kb = tid; kb < num_K_blks; kb += 32) {
        // Load 8 packed bytes = 16 FP4 values from weight row
        const uint8_t* w_ptr = &W_packed[(size_t)n_out * (K / 2) + kb * PB];
        alignas(16) uint8_t w_packed[PB];
        *reinterpret_cast<uint2*>(w_packed) = *reinterpret_cast<const uint2*>(w_ptr);

        float w_sc = W_scale[(size_t)n_out * num_K_blks + kb];

        // Load 16 FP32 activation values
        int x_off = kb * B;
        float4 v0 = reinterpret_cast<const float4*>(&x_fp32[x_off])[0];
        float4 v1 = reinterpret_cast<const float4*>(&x_fp32[x_off])[1];
        float4 v2 = reinterpret_cast<const float4*>(&x_fp32[x_off])[2];
        float4 v3 = reinterpret_cast<const float4*>(&x_fp32[x_off])[3];

        // Unpack FP4 nibbles to float via E2M1, then dot with FP32 activations
        float a16[16];
        a16[0]=v0.x; a16[1]=v0.y; a16[2]=v0.z; a16[3]=v0.w;
        a16[4]=v1.x; a16[5]=v1.y; a16[6]=v1.z; a16[7]=v1.w;
        a16[8]=v2.x; a16[9]=v2.y; a16[10]=v2.z; a16[11]=v2.w;
        a16[12]=v3.x; a16[13]=v3.y; a16[14]=v3.z; a16[15]=v3.w;

        float sum_b = 0.0f;
        #pragma unroll
        for (int j = 0; j < PB; ++j) {
            __nv_fp4_e2m1 fp4_lo, fp4_hi;
            uint8_t lo = w_packed[j] & 0x0F;
            uint8_t hi = (w_packed[j] >> 4) & 0x0F;
            memcpy(&fp4_lo, &lo, 1);
            memcpy(&fp4_hi, &hi, 1);
            sum_b += static_cast<float>(fp4_lo) * a16[j * 2];
            sum_b += static_cast<float>(fp4_hi) * a16[j * 2 + 1];
        }
        acc += sum_b * w_sc;
    }

    // Warp shuffle reduction
    acc += __shfl_xor_sync(0xffffffff, acc, 16);
    acc += __shfl_xor_sync(0xffffffff, acc, 8);
    acc += __shfl_xor_sync(0xffffffff, acc, 4);
    acc += __shfl_xor_sync(0xffffffff, acc, 2);
    acc += __shfl_xor_sync(0xffffffff, acc, 1);

    if (tid == 0) y_out[n_out] = acc;
}

// ===========================================================================
// Warp-cooperative INT4 GEMV — 1 warp per output row, packed 2 vals/byte
// Signed INT4 weights [-8,7] packed as 2 nibbles per byte.
// Strategy: scalar FP32 multiply-accumulate (no dp4a).
// Unpack nibble → float, multiply, accumulate. Simpler than dp4a path.
// Weight layout: W_packed [N][K/2], scale layout: W_scale [N][K/16] FP32.
// ===========================================================================

// Device-side: unpack 1 byte (2 INT4 nibbles) to 2 floats
__device__ __forceinline__ void int4_byte_to_floats(uint8_t b, float &f0, float &f1) {
    // Nibble stores q+8 (offset-binary, [0..15] for [-8..7]).
    // Convert back: val = nib - 8.
    int lo = (b & 0x0F) - 8;
    int hi = ((b >> 4) & 0x0F) - 8;
    f0 = static_cast<float>(lo);
    f1 = static_cast<float>(hi);
}

__launch_bounds__(32, 8)
__global__ void gemv_int4_warp_kernel(
    float* __restrict__ y_out,
    const uint8_t* __restrict__ x_packed,    // [K/2] packed INT4 activations
    const float* __restrict__ x_scale,       // [K/16] FP32 activation scales
    const uint8_t* __restrict__ W_packed,    // [N][K/2] packed INT4 weights
    const float* __restrict__ W_scale,       // [N][K/16] FP32 weight scales
    int K, int N)
{
    constexpr int B = 16;    // quantization block size
    constexpr int PB = 8;    // packed bytes per block (B/2)
    int n_out = blockIdx.x;
    int tid = threadIdx.x;

    int num_K_blks = K / B;
    float acc = 0.0f;

    // Stride-32 loop: each thread handles scattered K-blocks
    for (int kb = tid; kb < num_K_blks; kb += 32) {
        // Load 8 packed bytes = 16 INT4 values from weight row
        const uint8_t* w_ptr = &W_packed[(size_t)n_out * (K / 2) + kb * PB];
        uint2 w_packed = *reinterpret_cast<const uint2*>(w_ptr);

        // Load 8 packed bytes = 16 INT4 values from activation
        const uint8_t* x_ptr = &x_packed[kb * PB];
        uint2 x_packed_val = *reinterpret_cast<const uint2*>(x_ptr);

        float w_sc = W_scale[(size_t)n_out * num_K_blks + kb];
        float x_sc = x_scale[kb];
        float prod_scale = w_sc * x_sc;

        // Scalar unpack + dot product: 8 bytes → 16 floats → 16 multiplies
        const uint8_t* wb = reinterpret_cast<const uint8_t*>(&w_packed);
        const uint8_t* xb = reinterpret_cast<const uint8_t*>(&x_packed_val);

        float sum_f = 0.0f;
        #pragma unroll
        for (int j = 0; j < PB; ++j) {
            float w0, w1, x0, x1;
            int4_byte_to_floats(wb[j], w0, w1);
            int4_byte_to_floats(xb[j], x0, x1);
            sum_f += w0 * x0 + w1 * x1;
        }
        acc += sum_f * prod_scale;
    }

    // Warp shuffle reduction
    acc += __shfl_xor_sync(0xffffffff, acc, 16);
    acc += __shfl_xor_sync(0xffffffff, acc, 8);
    acc += __shfl_xor_sync(0xffffffff, acc, 4);
    acc += __shfl_xor_sync(0xffffffff, acc, 2);
    acc += __shfl_xor_sync(0xffffffff, acc, 1);

    if (tid == 0) y_out[n_out] = acc;
}

} // anonymous namespace

// ===========================================================================
// FP32×INT8 GEMV — FP32 activations × INT8 weights
// ===========================================================================
__launch_bounds__(kINT8Block, 1)
__global__ void gemv_fp32_int8_kernel(
    float* __restrict__ y_out,
    const float* __restrict__ x_fp32,
    const int8_t* __restrict__ W_t_int8,
    const float* __restrict__ W_t_scale,
    int K, int N)
{
    constexpr int B = 16;
    int tid = threadIdx.x;
    int n_out = blockIdx.x * kINT8Block + tid;
    if (n_out >= N) return;

    int num_K_blks = K / B;

    float acc = 0.0f;

    for (int kb = 0; kb < num_K_blks; ++kb) {
        // Load 16 FP32 activation values (4× float4 = 16 floats)
        int x_off = kb * B;
        float4 v0 = reinterpret_cast<const float4*>(&x_fp32[x_off])[0];
        float4 v1 = reinterpret_cast<const float4*>(&x_fp32[x_off])[1];
        float4 v2 = reinterpret_cast<const float4*>(&x_fp32[x_off])[2];
        float4 v3 = reinterpret_cast<const float4*>(&x_fp32[x_off])[3];

        // Load 16 INT8 weight values (4× int vectors)
        const int8_t* w_ptr = &W_t_int8[n_out * K + kb * B];
        int w0 = reinterpret_cast<const int*>(w_ptr)[0];
        int w1 = reinterpret_cast<const int*>(w_ptr)[1];
        int w2 = reinterpret_cast<const int*>(w_ptr)[2];
        int w3 = reinterpret_cast<const int*>(w_ptr)[3];

        float w_sc = W_t_scale[n_out * num_K_blks + kb];

        // Unpack int8 (sign-extend) → float and multiply-add (16-wide)
        #define SE(i) (float)(int8_t)((i) & 0xFF)
        float sum_b = 0.0f;
        sum_b += SE(w0 >>  0) * v0.x;  sum_b += SE(w0 >>  8) * v0.y;
        sum_b += SE(w0 >> 16) * v0.z;  sum_b += SE(w0 >> 24) * v0.w;
        sum_b += SE(w1 >>  0) * v1.x;  sum_b += SE(w1 >>  8) * v1.y;
        sum_b += SE(w1 >> 16) * v1.z;  sum_b += SE(w1 >> 24) * v1.w;
        sum_b += SE(w2 >>  0) * v2.x;  sum_b += SE(w2 >>  8) * v2.y;
        sum_b += SE(w2 >> 16) * v2.z;  sum_b += SE(w2 >> 24) * v2.w;
        sum_b += SE(w3 >>  0) * v3.x;  sum_b += SE(w3 >>  8) * v3.y;
        sum_b += SE(w3 >> 16) * v3.z;  sum_b += SE(w3 >> 24) * v3.w;
        #undef SE
        acc += sum_b * w_sc;
    }

    y_out[n_out] = acc;
}

// ---------------------------------------------------------------------------
// Per-row INT8 GEMV kernel — each output row has independent scales.
// Scale layout: W_t_scale [N × K/16] (vs old 2D [N/16 × K/16]).
// ---------------------------------------------------------------------------
__launch_bounds__(kINT8Block, 1)
__global__ void gemv_int8_per_row_kernel(
    float* __restrict__ y_out,
    const int8_t* __restrict__ x_int8,
    const float* __restrict__ x_scale,
    const int8_t* __restrict__ W_t_int8,
    const float* __restrict__ W_t_scale,
    int K, int N)
{
    constexpr int B = 16;
    int tid = threadIdx.x;
    int n_out = blockIdx.x * kINT8Block + tid;
    if (n_out >= N) return;

    int num_K_blks = K / B;

    float acc = 0.0f;

    for (int kb = 0; kb < num_K_blks; ++kb) {
        const int8_t* w_ptr = &W_t_int8[n_out * K + kb * B];
        alignas(16) int8_t w_buf[B];
        alignas(16) int8_t x_buf[B];
        *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);
        *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_int8 + kb * B);

        float w_sc = W_t_scale[n_out * num_K_blks + kb];  // per-row scale
        float x_sc = x_scale[kb];
        float prod_scale = w_sc * x_sc;

        const int* w32 = reinterpret_cast<const int*>(w_buf);
        const int* x32 = reinterpret_cast<const int*>(x_buf);
        int sumi = 0;
        sumi = __dp4a(w32[0], x32[0], sumi);
        sumi = __dp4a(w32[1], x32[1], sumi);
        sumi = __dp4a(w32[2], x32[2], sumi);
        sumi = __dp4a(w32[3], x32[3], sumi);
        acc += static_cast<float>(sumi) * prod_scale;
    }

    y_out[n_out] = acc;
}

// ===========================================================================
// FP32×INT8 per-row GEMV — FP32 activations × INT8 weights with per-row scales
// ===========================================================================
__launch_bounds__(kINT8Block, 1)
__global__ void gemv_fp32_int8_per_row_kernel(
    float* __restrict__ y_out,
    const float* __restrict__ x_fp32,
    const int8_t* __restrict__ W_t_int8,
    const float* __restrict__ W_t_scale,
    int K, int N)
{
    constexpr int B = 16;
    int tid = threadIdx.x;
    int n_out = blockIdx.x * kINT8Block + tid;
    if (n_out >= N) return;

    int num_K_blks = K / B;

    float acc = 0.0f;

    for (int kb = 0; kb < num_K_blks; ++kb) {
        int x_off = kb * B;
        float4 v0 = reinterpret_cast<const float4*>(&x_fp32[x_off])[0];
        float4 v1 = reinterpret_cast<const float4*>(&x_fp32[x_off])[1];
        float4 v2 = reinterpret_cast<const float4*>(&x_fp32[x_off])[2];
        float4 v3 = reinterpret_cast<const float4*>(&x_fp32[x_off])[3];

        const int8_t* w_ptr = &W_t_int8[n_out * K + kb * B];
        int w0 = reinterpret_cast<const int*>(w_ptr)[0];
        int w1 = reinterpret_cast<const int*>(w_ptr)[1];
        int w2 = reinterpret_cast<const int*>(w_ptr)[2];
        int w3 = reinterpret_cast<const int*>(w_ptr)[3];

        float w_sc = W_t_scale[n_out * num_K_blks + kb];  // per-row scale

        #define SE2(i) (float)(int8_t)((i) & 0xFF)
        float sum_b = 0.0f;
        sum_b += SE2(w0 >>  0) * v0.x;  sum_b += SE2(w0 >>  8) * v0.y;
        sum_b += SE2(w0 >> 16) * v0.z;  sum_b += SE2(w0 >> 24) * v0.w;
        sum_b += SE2(w1 >>  0) * v1.x;  sum_b += SE2(w1 >>  8) * v1.y;
        sum_b += SE2(w1 >> 16) * v1.z;  sum_b += SE2(w1 >> 24) * v1.w;
        sum_b += SE2(w2 >>  0) * v2.x;  sum_b += SE2(w2 >>  8) * v2.y;
        sum_b += SE2(w2 >> 16) * v2.z;  sum_b += SE2(w2 >> 24) * v2.w;
        sum_b += SE2(w3 >>  0) * v3.x;  sum_b += SE2(w3 >>  8) * v3.y;
        sum_b += SE2(w3 >> 16) * v3.z;  sum_b += SE2(w3 >> 24) * v3.w;
        #undef SE2
        acc += sum_b * w_sc;
    }

    y_out[n_out] = acc;
}

// ===========================================================================
// Public API
// ===========================================================================

cudaError_t pack_int8(
    void*           out_int8,
    const float*    in_fp32,
    const float*    scale_out,
    int             num_elements,
    cudaStream_t    stream)
{
    if (num_elements <= 0 || num_elements % 16 != 0)
        return cudaErrorInvalidValue;

    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    pack_int8_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<int8_t*>(out_int8), in_fp32, scale_out, num_elements);
    return cudaPeekAtLastError();
}

// ===========================================================================
// Fused INT8 quantize: compute absmax scales + pack in one kernel
// Grid = num_elements / 16 (one block per 16-element group)
// Block = 32 threads (warp). Threads 0-15 load, all warp-reduce absmax.
// ===========================================================================
__launch_bounds__(32, 4)
__global__ void quantize_int8_kernel(
    int8_t* __restrict__ out,
    float*  __restrict__ scales,
    const float* __restrict__ in,
    int n)
{
    int blk = blockIdx.x;
    int tid = threadIdx.x;
    int idx = blk * 16 + tid;

    // Load (16 values across 32 threads; lanes 0-15 get data, 16-31 load 0)
    float v = (tid < 16 && idx < n) ? in[idx] : 0.0f;

    // Full-warp absmax reduce (32 lanes, but only 0-15 have meaningful data)
    float av = fabsf(v);
    av = fmaxf(av, __shfl_xor_sync(0xffffffff, av, 8));
    av = fmaxf(av, __shfl_xor_sync(0xffffffff, av, 4));
    av = fmaxf(av, __shfl_xor_sync(0xffffffff, av, 2));
    av = fmaxf(av, __shfl_xor_sync(0xffffffff, av, 1));

    float sc = av / 127.0f;
    if (sc < 1e-10f) sc = 1.0f;
    if (tid == 0) scales[blk] = sc;

    // Quantize and store (only lanes 0-15)
    float qf = roundf(v / sc);
    qf = fminf(127.0f, fmaxf(-127.0f, qf));
    if (tid < 16 && idx < n)
        out[idx] = static_cast<int8_t>(static_cast<int>(qf));
}

cudaError_t quantize_int8(
    void*           out_int8,
    float*          out_scale,
    const float*    in_fp32,
    int             num_elements,
    cudaStream_t    stream)
{
    if (num_elements <= 0 || num_elements % 16 != 0)
        return cudaErrorInvalidValue;

    int blocks = num_elements / 16;
    quantize_int8_kernel<<<blocks, 32, 0, stream>>>(
        static_cast<int8_t*>(out_int8), out_scale, in_fp32, num_elements);
    return cudaPeekAtLastError();
}

cudaError_t gemv_int8(
    float*          y_out,
    const void*     x_int8,
    const float*    x_scale,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || N % 16 != 0)
        return cudaErrorInvalidValue;

    int nb = (N + kINT8Block - 1) / kINT8Block;
    gemv_int8_kernel<<<dim3(nb), dim3(kINT8Block), 0, stream>>>(
        y_out,
        static_cast<const int8_t*>(x_int8), x_scale,
        static_cast<const int8_t*>(W_t_int8), W_t_scale,
        K, N);
    return cudaPeekAtLastError();
}

// ===========================================================================
// Warp-cooperative INT8 GEMV — 1 warp/row, shuffle reduce
// ===========================================================================
cudaError_t gemv_int8_warp(
    float*          y_out,
    const void*     x_int8,
    const float*    x_scale,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || N % 16 != 0)
        return cudaErrorInvalidValue;

    // 1 block per output row, 32 threads per block (1 warp)
    gemv_int8_warp_kernel<<<dim3(N), dim3(32), 0, stream>>>(
        y_out,
        static_cast<const int8_t*>(x_int8), x_scale,
        static_cast<const int8_t*>(W_t_int8), W_t_scale,
        K, N);
    return cudaPeekAtLastError();
}

// Warp-cooperative FP32×INT8 per-row GEMV
cudaError_t gemv_fp32_int8_per_row_warp(
    float*          y_out,
    const float*    x_fp32,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || N % 16 != 0)
        return cudaErrorInvalidValue;

    gemv_fp32_int8_per_row_warp_kernel<<<dim3(N), dim3(32), 0, stream>>>(
        y_out, x_fp32,
        static_cast<const int8_t*>(W_t_int8), W_t_scale,
        K, N);
    return cudaPeekAtLastError();
}

// ===========================================================================
// Packed FP4 warp GEMV — packed FP4 activations × packed FP4 weights
// ===========================================================================
cudaError_t gemv_fp4_warp(
    float*          y_out,
    const void*     x_packed,      // [K/2] packed FP4
    const float*    x_scale,       // [K/16] FP32
    const void*     W_packed,      // [N][K/2] packed FP4
    const float*    W_scale,       // [N][K/16] FP32 per-row
    int             K,
    int             N,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || N % 16 != 0)
        return cudaErrorInvalidValue;

    gemv_fp4_warp_kernel<<<dim3(N), dim3(32), 0, stream>>>(
        y_out,
        static_cast<const uint8_t*>(x_packed), x_scale,
        static_cast<const uint8_t*>(W_packed), W_scale,
        K, N);
    return cudaPeekAtLastError();
}

// FP32 activations × packed FP4 weights
cudaError_t gemv_fp32_fp4_warp(
    float*          y_out,
    const float*    x_fp32,
    const void*     W_packed,      // [N][K/2] packed FP4
    const float*    W_scale,       // [N][K/16] FP32 per-row
    int             K,
    int             N,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || N % 16 != 0)
        return cudaErrorInvalidValue;

    gemv_fp32_fp4_warp_kernel<<<dim3(N), dim3(32), 0, stream>>>(
        y_out, x_fp32,
        static_cast<const uint8_t*>(W_packed), W_scale,
        K, N);
    return cudaPeekAtLastError();
}

// ===========================================================================
// Packed INT4 warp GEMV — signed INT4 activations × signed INT4 weights
// 2× less bandwidth than INT8. Uses __dp4a after nibble→int8 unpack.
// ===========================================================================
cudaError_t gemv_int4_warp(
    float*          y_out,
    const void*     x_packed,      // [K/2] packed INT4
    const float*    x_scale,       // [K/16] FP32
    const void*     W_packed,      // [N][K/2] packed INT4
    const float*    W_scale,       // [N][K/16] FP32 per-row
    int             K,
    int             N,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || N % 16 != 0)
        return cudaErrorInvalidValue;

    gemv_int4_warp_kernel<<<dim3(N), dim3(32), 0, stream>>>(
        y_out,
        static_cast<const uint8_t*>(x_packed), x_scale,
        static_cast<const uint8_t*>(W_packed), W_scale,
        K, N);
    return cudaPeekAtLastError();
}

// ===========================================================================
// Batched INT4 GEMV — processes M sequences in parallel
// Weight loaded once per K-block, reused across M tokens via template.
// ===========================================================================
namespace {

constexpr int kINT4Block = 32;  // threads per block (1 warp)

// Batched INT4 GEMV: M sequences × same weight matrix in one kernel launch.
// Grid: N blocks, 32 threads/block (1 warp). Each warp computes 1 output row
// for all M sequences. Weight loaded once, reused across M activations.
template <int M>
__launch_bounds__(32, 8)
__global__ void gemv_int4_batched_kernel(
    float* __restrict__ y_out,            // [M][N] output
    const uint8_t* __restrict__ x_packed,  // [M][K/2] packed INT4 activations
    const float* __restrict__ x_scale,    // [M][K/16] activation scales
    const uint8_t* __restrict__ W_packed, // [N][K/2] packed INT4 weights
    const float* __restrict__ W_scale,    // [N][K/16] weight scales
    int K, int N)
{
    constexpr int B = 16;    // quantization block size
    constexpr int PB = 8;    // packed bytes per block (B/2)
    int n_out = blockIdx.x;
    if (n_out >= N) return;
    int tid = threadIdx.x;

    int num_K_blks = K / B;

    // Accumulators for M tokens (template unroll for M=1..8)
    float acc[M];
    #pragma unroll
    for (int mi = 0; mi < M; ++mi) acc[mi] = 0.0f;

    // Stride-32 loop: each thread handles scattered K-blocks
    for (int kb = tid; kb < num_K_blks; kb += 32) {
        // Load 8 packed bytes = 16 INT4 values from weight row (shared across M)
        const uint8_t* w_ptr = &W_packed[(size_t)n_out * (K / 2) + kb * PB];
        uint2 w_packed = *reinterpret_cast<const uint2*>(w_ptr);
        float w_sc = W_scale[(size_t)n_out * num_K_blks + kb];

        // Load activation for each of M tokens (strided access)
        #pragma unroll
        for (int mi = 0; mi < M; ++mi) {
            const uint8_t* x_ptr = &x_packed[(size_t)mi * (K / 2) + kb * PB];
            uint2 x_packed_val = *reinterpret_cast<const uint2*>(x_ptr);
            float x_sc = x_scale[(size_t)mi * num_K_blks + kb];
            float prod_scale = w_sc * x_sc;

            // Scalar unpack + dot product
            const uint8_t* wb = reinterpret_cast<const uint8_t*>(&w_packed);
            const uint8_t* xb = reinterpret_cast<const uint8_t*>(&x_packed_val);

            float sum_f = 0.0f;
            #pragma unroll
            for (int j = 0; j < PB; ++j) {
                float w0, w1, x0, x1;
                int4_byte_to_floats(wb[j], w0, w1);
                int4_byte_to_floats(xb[j], x0, x1);
                sum_f += w0 * x0 + w1 * x1;
            }
            acc[mi] += sum_f * prod_scale;
        }
    }

    // Warp shuffle reduction for each of M accumulators
    #pragma unroll
    for (int mi = 0; mi < M; ++mi) {
        acc[mi] += __shfl_xor_sync(0xffffffff, acc[mi], 16);
        acc[mi] += __shfl_xor_sync(0xffffffff, acc[mi], 8);
        acc[mi] += __shfl_xor_sync(0xffffffff, acc[mi], 4);
        acc[mi] += __shfl_xor_sync(0xffffffff, acc[mi], 2);
        acc[mi] += __shfl_xor_sync(0xffffffff, acc[mi], 1);
    }

    // Thread 0 writes all M outputs
    if (tid == 0) {
        #pragma unroll
        for (int mi = 0; mi < M; ++mi) {
            y_out[(size_t)mi * N + n_out] = acc[mi];
        }
    }
}

}  // anonymous namespace

cudaError_t gemv_int4_batched(
    float*          y_out,
    const uint8_t*  x_packed,
    const float*    x_scale,
    const uint8_t*  W_packed,
    const float*    W_scale,
    int             K,
    int             N,
    int             M,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || N % 16 != 0 || M < 1 || M > 8)
        return cudaErrorInvalidValue;

    dim3 grid(N, 1);

    switch (M) {
        case 1: gemv_int4_batched_kernel<1><<<grid, kINT4Block, 0, stream>>>(y_out, (const uint8_t*)x_packed, x_scale, (const uint8_t*)W_packed, W_scale, K, N); break;
        case 2: gemv_int4_batched_kernel<2><<<grid, kINT4Block, 0, stream>>>(y_out, (const uint8_t*)x_packed, x_scale, (const uint8_t*)W_packed, W_scale, K, N); break;
        case 3: gemv_int4_batched_kernel<3><<<grid, kINT4Block, 0, stream>>>(y_out, (const uint8_t*)x_packed, x_scale, (const uint8_t*)W_packed, W_scale, K, N); break;
        case 4: gemv_int4_batched_kernel<4><<<grid, kINT4Block, 0, stream>>>(y_out, (const uint8_t*)x_packed, x_scale, (const uint8_t*)W_packed, W_scale, K, N); break;
        case 5: gemv_int4_batched_kernel<5><<<grid, kINT4Block, 0, stream>>>(y_out, (const uint8_t*)x_packed, x_scale, (const uint8_t*)W_packed, W_scale, K, N); break;
        case 6: gemv_int4_batched_kernel<6><<<grid, kINT4Block, 0, stream>>>(y_out, (const uint8_t*)x_packed, x_scale, (const uint8_t*)W_packed, W_scale, K, N); break;
        case 7: gemv_int4_batched_kernel<7><<<grid, kINT4Block, 0, stream>>>(y_out, (const uint8_t*)x_packed, x_scale, (const uint8_t*)W_packed, W_scale, K, N); break;
        case 8: gemv_int4_batched_kernel<8><<<grid, kINT4Block, 0, stream>>>(y_out, (const uint8_t*)x_packed, x_scale, (const uint8_t*)W_packed, W_scale, K, N); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaPeekAtLastError();
}

cudaError_t gemv_int8_from_fp4(
    float*          y_out,
    const void*     x_fp4,
    const float*    x_fp4_scale,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || N % 16 != 0)
        return cudaErrorInvalidValue;

    int nb = (N + kINT8Block - 1) / kINT8Block;
    gemv_int8_from_fp4_kernel<<<dim3(nb), dim3(kINT8Block), 0, stream>>>(
        y_out,
        static_cast<const __nv_fp4_e2m1*>(x_fp4), x_fp4_scale,
        static_cast<const int8_t*>(W_t_int8), W_t_scale,
        K, N);
    return cudaPeekAtLastError();
}

cudaError_t gemv_fp32_int8(
    float*          y_out,
    const float*    x_fp32,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || N % 16 != 0)
        return cudaErrorInvalidValue;

    int nb = (N + kINT8Block - 1) / kINT8Block;
    gemv_fp32_int8_kernel<<<dim3(nb), dim3(kINT8Block), 0, stream>>>(
        y_out, x_fp32,
        static_cast<const int8_t*>(W_t_int8), W_t_scale,
        K, N);
    return cudaPeekAtLastError();
}

cudaError_t gemv_fp32_int8_per_row(
    float*          y_out,
    const float*    x_fp32,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || N % 16 != 0)
        return cudaErrorInvalidValue;

    int nb = (N + kINT8Block - 1) / kINT8Block;
    gemv_fp32_int8_per_row_kernel<<<dim3(nb), dim3(kINT8Block), 0, stream>>>(
        y_out, x_fp32,
        static_cast<const int8_t*>(W_t_int8), W_t_scale,
        K, N);
    return cudaPeekAtLastError();
}

cudaError_t gemv_int8_per_row(
    float*          y_out,
    const void*     x_int8,
    const float*    x_scale,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || N % 16 != 0)
        return cudaErrorInvalidValue;

    int nb = (N + kINT8Block - 1) / kINT8Block;
    gemv_int8_per_row_kernel<<<dim3(nb), dim3(kINT8Block), 0, stream>>>(
        y_out,
        static_cast<const int8_t*>(x_int8), x_scale,
        static_cast<const int8_t*>(W_t_int8), W_t_scale,
        K, N);
    return cudaPeekAtLastError();
}

cudaError_t transpose_int8_weights(
    void*           dst,
    float*          dst_scale,
    const void*     src,
    const float*    src_scale,
    int             K,
    int             N,
    cudaStream_t    stream)
{
    int total = K * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    transpose_int8_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<int8_t*>(dst), static_cast<const int8_t*>(src), K, N);

    int num_K_blks = K / 16;
    int num_N_blks = N / 16;
    int total_scales = num_K_blks * num_N_blks;
    blocks = (total_scales + threads - 1) / threads;

    transpose_scales_int8_kernel<<<blocks, threads, 0, stream>>>(
        dst_scale, src_scale, num_K_blks, num_N_blks);

    return cudaPeekAtLastError();
}


// ---------------------------------------------------------------------------
// INT8 GEMV Split-K kernel: K split into K_splits, AtomicAdd reduction.
// Grid: (N/256, K_splits). Each block computes partial dot product over
// K/K_splits columns. AtomicAdd to reduce to same output row.
// Caller MUST zero y_out before launch (cudaMemset).
//
// Targets N=6144 down_proj where N/256 = 24 blocks < 36 SMs.
// K_splits=2 → 48 blocks, K_splits=3 → 72 blocks > 36 SMs.
// ---------------------------------------------------------------------------
__launch_bounds__(kINT8Block, 1)
__global__ void gemv_int8_splitk_kernel(
    float* __restrict__ y_out,
    const int8_t* __restrict__ x_int8,
    const float* __restrict__ x_scale,
    const int8_t* __restrict__ W_t_int8,
    const float* __restrict__ W_t_scale,
    int K, int N, int K_splits)
{
    constexpr int B = 16;
    int tid = threadIdx.x;
    int n_out = blockIdx.x * kINT8Block + tid;
    if (n_out >= N) return;

    int split_id = blockIdx.y;
    int num_K_blks = K / B;

    // Each split handles K/K_splits columns
    int split_blks = num_K_blks / K_splits;
    int kb_start = split_id * split_blks;
    int kb_end = (split_id == K_splits - 1) ? num_K_blks : (kb_start + split_blks);

    float acc = 0.0f;

    for (int kb = kb_start; kb < kb_end; ++kb) {
        // Load 16 INT8 weight + activation values via vectorized uint4 loads
        const int8_t* w_ptr = &W_t_int8[n_out * K + kb * B];
        alignas(16) int8_t w_buf[B];
        alignas(16) int8_t x_buf[B];
        *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);
        *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_int8 + kb * B);

        float w_sc = W_t_scale[n_out * num_K_blks + kb];
        float x_sc = x_scale[kb];
        float prod_scale = w_sc * x_sc;

        // __dp4a: 4-way int8 SIMD dot product per iteration (4 × 4 = 16 total)
        const int* w32 = reinterpret_cast<const int*>(w_buf);
        const int* x32 = reinterpret_cast<const int*>(x_buf);
        int sumi = 0;
        sumi = __dp4a(w32[0], x32[0], sumi);
        sumi = __dp4a(w32[1], x32[1], sumi);
        sumi = __dp4a(w32[2], x32[2], sumi);
        sumi = __dp4a(w32[3], x32[3], sumi);
        acc += static_cast<float>(sumi) * prod_scale;
    }

    // AtomicAdd reduction on shared output row
    atomicAdd(&y_out[n_out], acc);
}

cudaError_t gemv_int8_splitk(
    float*          y_out,
    const void*     x_int8,
    const float*    x_scale,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    int             K_splits,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || N % 16 != 0 || K % K_splits != 0)
        return cudaErrorInvalidValue;

    dim3 grid((N + kINT8Block - 1) / kINT8Block, K_splits);
    gemv_int8_splitk_kernel<<<grid, kINT8Block, 0, stream>>>(
        y_out,
        static_cast<const int8_t*>(x_int8), x_scale,
        static_cast<const int8_t*>(W_t_int8), W_t_scale,
        K, N, K_splits);
    return cudaPeekAtLastError();
}


// ---------------------------------------------------------------------------
// INT8 Batched GEMV: process M tokens simultaneously, reuse weights across them.
// Grid: (ceil(N/256), M). Block: 256 threads.
// Weights loaded once per K-block, activations loaded per-token.
// Eliminates M-1 weight loads vs launching M separate gemv_int8 kernels.
// Best batch sizes: 2-8 tokens (matching llama.cpp MMVQ_MAX_BATCH_SIZE).
// ---------------------------------------------------------------------------
template<int M>
__global__ void gemv_int8_batched_kernel(
    float* __restrict__ y_out,
    const int8_t* __restrict__ x_int8,
    const float* __restrict__ x_scale,
    const int8_t* __restrict__ W_t_int8,
    const float* __restrict__ W_t_scale,
    int K, int N)
{
    constexpr int B = 16;
    int tid = threadIdx.x;
    int n_out = blockIdx.x * kINT8Block + tid;
    int m = blockIdx.y;
    if (n_out >= N) return;

    int num_K_blks = K / B;

    float acc = 0.0f;

    for (int kb = 0; kb < num_K_blks; ++kb) {
        // Load 16 weight values once (shared across M tokens via template unrolling)
        const int8_t* w_ptr = &W_t_int8[n_out * K + kb * B];
        alignas(16) int8_t w_buf[B];
        *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);

        float w_sc = W_t_scale[n_out * num_K_blks + kb];

        // Load this token's activation
        const int8_t* x_ptr = &x_int8[m * K + kb * B];
        alignas(16) int8_t x_buf[B];
        *reinterpret_cast<uint4*>(x_buf) = *reinterpret_cast<const uint4*>(x_ptr);

        float x_sc = x_scale[m * num_K_blks + kb];
        float prod_scale = w_sc * x_sc;

        const int* w32 = reinterpret_cast<const int*>(w_buf);
        const int* x32 = reinterpret_cast<const int*>(x_buf);
        int sumi = 0;
        sumi = __dp4a(w32[0], x32[0], sumi);
        sumi = __dp4a(w32[1], x32[1], sumi);
        sumi = __dp4a(w32[2], x32[2], sumi);
        sumi = __dp4a(w32[3], x32[3], sumi);
        acc += static_cast<float>(sumi) * prod_scale;
    }

    y_out[m * N + n_out] = acc;
}

// ---------------------------------------------------------------------------
// Batched GEMV dispatch: routes to templated kernel based on M.
// ---------------------------------------------------------------------------
cudaError_t gemv_int8_batched(
    float*          y_out,
    const void*     x_int8,
    const float*    x_scale,
    const void*     W_t_int8,
    const float*    W_t_scale,
    int             K,
    int             N,
    int             M,
    cudaStream_t    stream)
{
    if (K % 16 != 0 || N % 16 != 0 || M < 1)
        return cudaErrorInvalidValue;

    int nb = (N + kINT8Block - 1) / kINT8Block;

    // For M>8, launch multiple kernels (each handles up to 8 sequences)
    for (int base = 0; base < M; base += 8) {
        int m = min(M - base, 8);
        dim3 grid(nb, m);
        size_t x_off = (size_t)base * K;
        size_t y_off = (size_t)base * N;
        const float* xs = x_scale + (size_t)base * (K / 16);
        switch (m) {
            case 1: gemv_int8_batched_kernel<1><<<grid, kINT8Block, 0, stream>>>(y_out + y_off, (const int8_t*)(x_int8) + x_off, xs, (const int8_t*)(W_t_int8), W_t_scale, K, N); break;
            case 2: gemv_int8_batched_kernel<2><<<grid, kINT8Block, 0, stream>>>(y_out + y_off, (const int8_t*)(x_int8) + x_off, xs, (const int8_t*)(W_t_int8), W_t_scale, K, N); break;
            case 3: gemv_int8_batched_kernel<3><<<grid, kINT8Block, 0, stream>>>(y_out + y_off, (const int8_t*)(x_int8) + x_off, xs, (const int8_t*)(W_t_int8), W_t_scale, K, N); break;
            case 4: gemv_int8_batched_kernel<4><<<grid, kINT8Block, 0, stream>>>(y_out + y_off, (const int8_t*)(x_int8) + x_off, xs, (const int8_t*)(W_t_int8), W_t_scale, K, N); break;
            case 5: gemv_int8_batched_kernel<5><<<grid, kINT8Block, 0, stream>>>(y_out + y_off, (const int8_t*)(x_int8) + x_off, xs, (const int8_t*)(W_t_int8), W_t_scale, K, N); break;
            case 6: gemv_int8_batched_kernel<6><<<grid, kINT8Block, 0, stream>>>(y_out + y_off, (const int8_t*)(x_int8) + x_off, xs, (const int8_t*)(W_t_int8), W_t_scale, K, N); break;
            case 7: gemv_int8_batched_kernel<7><<<grid, kINT8Block, 0, stream>>>(y_out + y_off, (const int8_t*)(x_int8) + x_off, xs, (const int8_t*)(W_t_int8), W_t_scale, K, N); break;
            case 8: gemv_int8_batched_kernel<8><<<grid, kINT8Block, 0, stream>>>(y_out + y_off, (const int8_t*)(x_int8) + x_off, xs, (const int8_t*)(W_t_int8), W_t_scale, K, N); break;
        }
    }
    return cudaPeekAtLastError();
}

// ===========================================================================
// INT8 GEMM — C[M×N] = A[M×K] × B^T[N×K]
// A is FP32 activations, B is INT8 weights [N×K] with scales [N × K/16]
// Uses 4×4 register tiling with vectorized loads.
// ===========================================================================
__launch_bounds__(256, 1)
__global__ void gemm_int8_kernel(
    float* __restrict__ C,          // [M×N]
    const float* __restrict__ A,    // [M×K] FP32
    const int8_t* __restrict__ B_i8, // [N×K] INT8 transposed
    const float* __restrict__ B_sc,  // [N × K/16] scales
    int M, int N, int K)
{
    constexpr int TILE_M = 4;
    constexpr int TILE_N = 4;
    constexpr int THREADS_M = 16;
    constexpr int THREADS_N = 16;
    constexpr int BSIZE = 16;

    int bm = blockIdx.y * THREADS_M * TILE_M;
    int bn = blockIdx.x * THREADS_N * TILE_N;
    int tm = threadIdx.y;
    int tn = threadIdx.x;
    int m = bm + tm * TILE_M;
    int n = bn + tn * TILE_N;
    int num_K_blks = K / BSIZE;

    float acc[TILE_M][TILE_N] = {};

    for (int kb = 0; kb < num_K_blks; ++kb) {
        float w_sc[TILE_N];
        #pragma unroll
        for (int j = 0; j < TILE_N; ++j) {
            int nj = n + j;
            w_sc[j] = (nj < N) ? B_sc[nj * num_K_blks + kb] : 0.0f;
        }

        float a_vals[TILE_M][BSIZE];
        #pragma unroll
        for (int i = 0; i < TILE_M; ++i) {
            int mi = m + i;
            if (mi < M) {
                const float* a_ptr = &A[mi * K + kb * BSIZE];
                *reinterpret_cast<float4*>(&a_vals[i][0]) = *reinterpret_cast<const float4*>(a_ptr);
                *reinterpret_cast<float4*>(&a_vals[i][4]) = *reinterpret_cast<const float4*>(a_ptr + 4);
                *reinterpret_cast<float4*>(&a_vals[i][8]) = *reinterpret_cast<const float4*>(a_ptr + 8);
                *reinterpret_cast<float4*>(&a_vals[i][12]) = *reinterpret_cast<const float4*>(a_ptr + 12);
            }
        }

        #pragma unroll
        for (int j = 0; j < TILE_N; ++j) {
            int nj = n + j;
            if (nj < N) {
                const int8_t* w_ptr = &B_i8[nj * K + kb * BSIZE];
                alignas(16) int8_t w_buf[BSIZE];
                *reinterpret_cast<uint4*>(w_buf) = *reinterpret_cast<const uint4*>(w_ptr);

                #pragma unroll
                for (int i = 0; i < TILE_M; ++i) {
                    float block_sum = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < BSIZE; ++k) {
                        block_sum += (float)w_buf[k] * a_vals[i][k];
                    }
                    acc[i][j] += block_sum * w_sc[j];
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < TILE_M; ++i) {
        int mi = m + i;
        if (mi < M) {
            #pragma unroll
            for (int j = 0; j < TILE_N; ++j) {
                int nj = n + j;
                if (nj < N) C[mi * N + nj] = acc[i][j];
            }
        }
    }
}

cudaError_t gemm_int8(
    float*          C,              // [M×N] output
    const float*    A,              // [M×K] FP32 activations
    const void*     B_int8,         // [N×K] INT8 transposed weights
    const float*    B_scale,        // [N × K/16] weight scales
    int             M, int N, int K,
    cudaStream_t    stream)
{
    if (K % 16 != 0) return cudaErrorInvalidValue;
    constexpr int TILE = 4;
    constexpr int THREADS = 16;
    dim3 block(THREADS, THREADS);
    dim3 grid((N + THREADS * TILE - 1) / (THREADS * TILE),
              (M + THREADS * TILE - 1) / (THREADS * TILE));
    gemm_int8_kernel<<<grid, block, 0, stream>>>(
        C, A, static_cast<const int8_t*>(B_int8), B_scale, M, N, K);
    return cudaPeekAtLastError();
}

// ===========================================================================
// INT8×INT8 GEMM with __dp4a — pre-quantized activations
// C[M×N] = A_i8[M×K] × B_i8[N×K]^T
// Uses 4×4 register tiling (matching gemm_int8_kernel tile layout).
// Each 16×16 thread block covers 64×64 output region.
// Each thread computes acc[4][4] = 16 outputs.
// Inner K-loop uses __dp4a SIMD dot product.
// ===========================================================================
__launch_bounds__(256, 1)
__global__ void gemm_int8_dp4a_kernel(
    float* __restrict__ C,
    const int8_t* __restrict__ A_i8,
    const float* __restrict__ A_sc,
    const int8_t* __restrict__ B_i8,
    const float* __restrict__ B_sc,
    int M, int N, int K)
{
    constexpr int TILE_M = 4;
    constexpr int TILE_N = 4;
    constexpr int T_M = 16;  // threads along M
    constexpr int T_N = 16;  // threads along N
    constexpr int B = 16;

    int bm = blockIdx.y * TILE_M * T_M;
    int bn = blockIdx.x * TILE_N * T_N;
    int tm = threadIdx.y;
    int tn = threadIdx.x;
    int m_base = bm + tm * TILE_M;
    int n_base = bn + tn * TILE_N;

    int num_K_blks = K / B;
    float acc[TILE_M][TILE_N] = {};

    for (int kb = 0; kb < num_K_blks; ++kb) {
        // Load activation blocks for TILE_M rows
        int8_t a_tile[TILE_M][B];
        #pragma unroll
        for (int i = 0; i < TILE_M; ++i) {
            int mi = m_base + i;
            if (mi < M) {
                *reinterpret_cast<uint4*>(&a_tile[i][0]) =
                    *reinterpret_cast<const uint4*>(&A_i8[mi * K + kb * B]);
            }
        }

        // Load weight blocks for TILE_N columns
        int8_t w_tile[TILE_N][B];
        #pragma unroll
        for (int j = 0; j < TILE_N; ++j) {
            int nj = n_base + j;
            if (nj < N) {
                *reinterpret_cast<uint4*>(&w_tile[j][0]) =
                    *reinterpret_cast<const uint4*>(&B_i8[nj * K + kb * B]);
            }
        }

        // Compute dot products via __dp4a
        #pragma unroll
        for (int i = 0; i < TILE_M; ++i) {
            int mi = m_base + i;
            if (mi >= M) continue;
            float a_sc = A_sc[mi * num_K_blks + kb];

            #pragma unroll
            for (int j = 0; j < TILE_N; ++j) {
                int nj = n_base + j;
                if (nj >= N) continue;
                float w_sc = B_sc[nj * num_K_blks + kb];

                // __dp4a: 4 × 4-way SIMD
                const int* a32 = reinterpret_cast<const int*>(&a_tile[i][0]);
                const int* w32 = reinterpret_cast<const int*>(&w_tile[j][0]);
                int sumi = 0;
                sumi = __dp4a(a32[0], w32[0], sumi);
                sumi = __dp4a(a32[1], w32[1], sumi);
                sumi = __dp4a(a32[2], w32[2], sumi);
                sumi = __dp4a(a32[3], w32[3], sumi);

                acc[i][j] += static_cast<float>(sumi) * a_sc * w_sc;
            }
        }
    }

    // Store results
    #pragma unroll
    for (int i = 0; i < TILE_M; ++i) {
        int mi = m_base + i;
        if (mi >= M) continue;
        #pragma unroll
        for (int j = 0; j < TILE_N; ++j) {
            int nj = n_base + j;
            if (nj >= N) continue;
            C[mi * N + nj] = acc[i][j];
        }
    }
}

cudaError_t gemm_int8_dp4a(
    float*          C,
    const int8_t*   A_int8,
    const float*    A_scale,
    const int8_t*   B_int8,
    const float*    B_scale,
    int             M, int N, int K,
    cudaStream_t    stream)
{
    if (K % 16 != 0) return cudaErrorInvalidValue;
    constexpr int TILE = 4;
    constexpr int THREADS = 16;
    dim3 block(THREADS, THREADS);
    dim3 grid((N + THREADS * TILE - 1) / (THREADS * TILE),
              (M + THREADS * TILE - 1) / (THREADS * TILE));
    gemm_int8_dp4a_kernel<<<grid, block, 0, stream>>>(
        C, A_int8, A_scale, B_int8, B_scale, M, N, K);
    return cudaPeekAtLastError();
}

// ===========================================================================
// INT4 support functions (defined alongside INT4 warp GEMV)
// ===========================================================================

// Transpose INT4 weights: W (K×N/2) → W_t (N×K/2), scales transposed
__global__ void transpose_int4_kernel(
    uint8_t* __restrict__ dst,
    const uint8_t* __restrict__ src,
    int K, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int packed_elements = K * (N / 2);  // K × N/2 bytes
    if (idx >= packed_elements) return;
    int k = idx / (N / 2);
    int n_byte = idx % (N / 2);
    dst[n_byte * K + k] = src[k * (N / 2) + n_byte];
}

__global__ void transpose_scales_int4_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int num_K_blks, int num_N_blks)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_K_blks * num_N_blks;
    if (idx >= total) return;
    int kb = idx / num_N_blks;
    int nb = idx % num_N_blks;
    dst[nb * num_K_blks + kb] = src[kb * num_N_blks + nb];
}

cudaError_t transpose_int4_weights(
    void*           dst,
    float*          dst_scale,
    const void*     src,
    const float*    src_scale,
    int             K,
    int             N,
    cudaStream_t    stream)
{
    int packed_elements = K * (N / 2);
    int threads = 256;
    int blocks = (packed_elements + threads - 1) / threads;
    transpose_int4_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<uint8_t*>(dst), static_cast<const uint8_t*>(src), K, N);

    int num_K_blks = K / 16;
    int num_N_blks = N / 16;
    int total_scales = num_K_blks * num_N_blks;
    blocks = (total_scales + threads - 1) / threads;
    transpose_scales_int4_kernel<<<blocks, threads, 0, stream>>>(
        dst_scale, src_scale, num_K_blks, num_N_blks);

    return cudaPeekAtLastError();
}

// Unpack packed INT4 → FP32
__global__ void unpack_int4_fp32_kernel(
    float* __restrict__ x_out,
    const uint8_t* __restrict__ x_packed,
    const float* __restrict__ x_scale,
    int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K) return;
    int kb = idx / 16;
    int off = idx % 16;
    int byte_idx = idx / 2;
    uint8_t byte = x_packed[byte_idx];
    uint8_t nibble = (off % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
    int8_t val = static_cast<int8_t>(nibble - 8);  // [0..15] → [-8..7]
    x_out[idx] = static_cast<float>(val) * x_scale[kb];
}

cudaError_t unpack_int4_fp32(
    float*          x_out,
    const void*     x_packed,
    const float*    x_scale,
    int             K,
    cudaStream_t    stream)
{
    int threads = 256;
    int blocks = (K + threads - 1) / threads;
    unpack_int4_fp32_kernel<<<blocks, threads, 0, stream>>>(
        x_out, static_cast<const uint8_t*>(x_packed), x_scale, K);
    return cudaPeekAtLastError();
}

// Quantize FP32 → packed INT4 (with per-block scales)
// x_out_packed: [K/2] bytes, x_out_sc: [K/16] FP32
// Block size = 16. Per-block: absmax / 7 → quantize [-7..7] → nibble-pack.
__global__ void quantize_int4_kernel(
    uint8_t* __restrict__ x_out,
    float* __restrict__ x_sc,
    const float* __restrict__ in_fp32,
    int K,
    int M)  // batch size (scales laid out as [M][K/16])
{
    constexpr int B = 16;
    int kb = blockIdx.x;
    int num_kb = K / B;
    if (kb >= num_kb) return;
    
    // Sequence index from grid y dimension (for M>1 batches)
    int m = (gridDim.y > 1) ? blockIdx.y : 0;
    int off = kb * B;
    
    // Compute scales from input
    float absmax = 0.f;
    for (int i = 0; i < B; ++i) {
        float v = fabsf(in_fp32[m * K + off + i]);
        if (v > absmax) absmax = v;
    }
    float sc = (absmax > 1e-10f) ? (absmax / 7.f) : (1.f / 7.f);
    
    // Write scale at correct position for batched layout
    x_sc[m * num_kb + kb] = sc;

    for (int i = 0; i < B / 2; ++i) {
        float v0 = in_fp32[m * K + off + i * 2];
        float v1 = in_fp32[m * K + off + i * 2 + 1];
        int q0 = (int)roundf(v0 / sc);
        int q1 = (int)roundf(v1 / sc);
        q0 = max(-8, min(7, q0));
        q1 = max(-8, min(7, q1));
        uint8_t nib0 = (uint8_t)((q0 + 8) & 0x0F);
        uint8_t nib1 = (uint8_t)((q1 + 8) & 0x0F);
        x_out[m * (K / 2) + kb * (B / 2) + i] = nib0 | (nib1 << 4);
    }
}

cudaError_t quantize_int4(
    void*           x_out_packed,
    float*          x_out_sc,
    const float*    in_fp32,
    int             K,
    cudaStream_t    stream)
{
    // For backward compatibility with M=1 (no batch dimension)
    int num_kb = K / 16;
    quantize_int4_kernel<<<dim3(num_kb, 1), 1, 0, stream>>>(
        static_cast<uint8_t*>(x_out_packed),
        x_out_sc,
        in_fp32,
        K,
        1);  // M=1
    return cudaPeekAtLastError();
}

cudaError_t quantize_int4_batched(
    void*           x_out_packed,
    float*          x_out_sc,
    const float*    in_fp32,
    int             K,
    int             M,
    cudaStream_t    stream)
{
    // Batched version: scales are [M][K/16], activations are [M][K]
    int num_kb = K / 16;
    dim3 grid(num_kb, M);  // [num_kb, M] blocks
    quantize_int4_kernel<<<grid, 1, 0, stream>>>(
        static_cast<uint8_t*>(x_out_packed),
        x_out_sc,
        in_fp32,
        K,
        M);
    return cudaPeekAtLastError();
}

} // namespace kernels
} // namespace blackwell
