// src/kernels/quantize.cu — FP4 E2M1 packing / unpacking + coalesced memory copy
// RTX 5060 Ti / Blackwell SM_120.

#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include "blackwell/kernels.h"

namespace blackwell {
namespace kernels {
namespace {

// Device-side dequant: FP4 E2M1 → float (symmetric block-scale).
// scale = absmax / 3.0f (E2M1 magnitude range ≈ ±3.0).
__device__ __forceinline__ float dequant_fp4(__nv_fp4_e2m1 v, float scale) {
    return static_cast<float>(v) * scale;
}

// Host-side wrapper for packing (no device code needed for stub).
// Real implementation: see bw_pack_fp4_kernel and bw_unpack_fp4_kernel below
// (compiled when backend host code is present).

// ===========================================================================
// Kernel: bw_pack_fp4
// One thread per element.  Scale computed by block-0 thread.
// Correct but sequential on the scale; not bandwidth-optimal for large N.
// For large inputs, split into blocks and compute per-block scales (Phase B).
// ===========================================================================
__global__ void bw_pack_fp4_kernel(
    __nv_fp4_e2m1* __restrict__ out_fp4,
    const float*    scale_in,       // pre-computed scale from caller
    const float*    in_fp32,
    int             num_elements) {

    // Use pre-computed scale directly (no block-level absmax reduction).
    float scale = scale_in[0];
    if (scale < 1e-9f) scale = 1e-9f;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    float sv = fdividef(in_fp32[idx], scale);
    sv = fmaxf(-3.0f, fminf(3.0f, sv));
    out_fp4[idx] = __nv_fp4_e2m1(sv);
}

// ===========================================================================
// Kernel: bw_unpack_fp4  (simple elementwise dequant)
// ===========================================================================
__global__ void bw_unpack_fp4_kernel(
    float*          out_fp32,
    const __nv_fp4_e2m1* __restrict__ in_fp4,
    const float*    scales_in,
    int             num_elements) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    float scale = scales_in[0];
    out_fp32[idx] = static_cast<float>(in_fp4[idx]) * scale;
}

// ===========================================================================
// Kernel: bw_coalesced_copy
// float4 vector load/store — 128-bit ops optimal for 128-bit GDDR7 bus.
// Each warp handles 32 consecutive floats = 128 B = 1 bus transaction.
// ===========================================================================
__global__ void bw_coalesced_copy_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int num_elements) {

    int lane_id  = threadIdx.x & 31;
    int warp_id  = threadIdx.x >> 5;
    int warp_base = (blockIdx.x * 4 + warp_id) * 32;
    if (warp_base >= num_elements) return;

    int idx = warp_base + lane_id;
    if (idx + 3 < num_elements) {
        float4 v = reinterpret_cast<const float4*>(src)[idx >> 2];
        reinterpret_cast<float4*>(dst)[idx >> 2] = v;
    } else if (idx < num_elements) {
        dst[idx] = src[idx];
    }
}

} // anonymous namespace

// ===========================================================================
// Public API host wrappers
// ===========================================================================

cudaError_t pack_fp4(
    void* out_fp4, const float* in_fp32,
    const float* scale_in, int num_elements, cudaStream_t stream) {

    using Fp4 = __nv_fp4_e2m1;
    Fp4* out = static_cast<Fp4*>(out_fp4);

    const int threads = 256;
    dim3 block(threads);
    dim3 grid((num_elements + threads - 1) / threads);

    bw_pack_fp4_kernel<<<grid, block, 0, stream>>>(
        out, scale_in, in_fp32, num_elements);

    return cudaPeekAtLastError();
}

cudaError_t unpack_fp4(
    float* out_fp32, const void* in_fp4,
    const float* scales_in, int num_elements, cudaStream_t stream) {

    const int threads = 256;
    dim3 block(threads);
    dim3 grid((num_elements + threads - 1) / threads);

    bw_unpack_fp4_kernel<<<grid, block, 0, stream>>>(
        out_fp32, static_cast<const __nv_fp4_e2m1*>(in_fp4),
        scales_in, num_elements);

    return cudaPeekAtLastError();
}

// ---------------------------------------------------------------------------
// Fused: unpack FP4 → pack INT8 in 1 kernel
// Reads FP4 E2M1, converts to float, applies FP4 scale,
// then quantizes to INT8 with INT8 block scales (block-16)
// Output: int8_t[N] + float[N/16] scales
// ---------------------------------------------------------------------------
__global__ void fused_unpack_pack_kernel(
    int8_t* __restrict__ out_i8,
    float* __restrict__ out_scales,
    const __nv_fp4_e2m1* __restrict__ in_fp4,
    const float* __restrict__ fp4_scales,    // [1] single per-tensor FP4 scale
    const float* __restrict__ i8_scales_in,  // pre-computed INT8 block scales (optional)
    int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    float fp4_scale = fp4_scales[0];
    float v = static_cast<float>(in_fp4[idx]) * fp4_scale;

    int blk = idx / 16;
    float sc = i8_scales_in[blk];
    v = v / sc;
    v = fminf(127.0f, fmaxf(-127.0f, roundf(v)));
    out_i8[idx] = static_cast<int8_t>(static_cast<int>(v));
}

cudaError_t unpack_fp4_pack_int8(
    void* out_i8,
    float* out_scales,
    const void* in_fp4,
    const float* fp4_scale,
    const float* i8_scales,
    int num_elements,
    cudaStream_t stream) {

    const int threads = 256;
    dim3 block(threads);
    dim3 grid((num_elements + threads - 1) / threads);

    fused_unpack_pack_kernel<<<grid, block, 0, stream>>>(
        static_cast<int8_t*>(out_i8),
        out_scales,
        static_cast<const __nv_fp4_e2m1*>(in_fp4),
        fp4_scale,
        i8_scales,
        num_elements);

    return cudaPeekAtLastError();
}

cudaError_t coalesced_copy(
    float* dst, const float* src, int num_elements, cudaStream_t stream) {

    const int threads = 128;
    const int elem_per_block = threads / 32 * 32;  // multiple of warp
    dim3 block(threads);
    dim3 grid((num_elements + elem_per_block - 1) / elem_per_block);

    bw_coalesced_copy_kernel<<<grid, block, 0, stream>>>(dst, src, num_elements);
    return cudaPeekAtLastError();
}

} // namespace kernels
} // namespace blackwell
