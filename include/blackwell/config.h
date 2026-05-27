// Public configuration header for RTX 5060 Ti / Blackwell (SM_120)
// ---------------------------------------------------------------------------
// Blackwell compute capability 12.0 constants and device-side helpers.
// This header is intended to be lightweight and usable from both host and
// device code (where noted).

#pragma once
#ifndef BLACKWELL_CONFIG_H
#define BLACKWELL_CONFIG_H

// Device C++ required for constexpr CUDA math
#if defined(__CUDACC__)
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#endif

namespace blackwell {

// ---------------------------------------------------------------------------
// Hardware limits — CC 12.0 RTX 5060 Ti
// ---------------------------------------------------------------------------
constexpr int kSMArchitecture      = 120;   // sm_120 / compute_120
// NOTE: Blackwell FP4 block-scaled MMA requires '120a' arch suffix:
//   -gencode=arch=compute_120a,code=sm_120a
// Plain sm_120 rejects mma.sync.aligned.kind::mxf4.block_scale PTX.
constexpr int kMaxWarpsPerSM        = 48;
constexpr int kMaxRegistersPerSM   = 65536;
constexpr int kMaxRegistersPerThread = 255;
constexpr int kMaxSharedMemBytesPerSM  = 131072;  // 128 KB
constexpr int kMaxSharedMemBytesPerBlock = 101376; // 99 KB
constexpr int kMaxThreadsPerBlock  = 1024;
constexpr int kWarpSize            = 32;

// ---------------------------------------------------------------------------
// FP4 / NVFP4 constants
// ---------------------------------------------------------------------------
// FP4 E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit
// Range: encoded values 0–7 map to {-2,-1,-0.5,-0.25,0.25,0.5,1,2}
constexpr int  kFP4BlockSize    = 16;   // shared with block-scaled MMA
constexpr int  kFP4ScaleBits   = 8;    // scale factor stored as FP8 or FP16

// ---------------------------------------------------------------------------
// GEMM tile sizes (must fit within 99 KB shared mem per block)
// ---------------------------------------------------------------------------
// 99 KB = 101376 bytes
// M-tile × K-tile × (A_elem_bytes + scale_bytes)
// 64×64×2 = 8192 bytes  →  fine
// GEMM (prefill): one 16×16 output per WMMA mma_sync, K=64 tiles.
// Grid dimensions cover (M/16, N/16) in the public API.
constexpr int kGEMMTileM = 128; // CTA tile size (8 × 16 WMMA fragments)
constexpr int kGEMMTileN = 128;
constexpr int kGEMMTileK = 64;

// GEMM warp configuration
constexpr int kGEMMWarps     = 8;   // 8 warps per CTA
constexpr int kGEMMThreads   = kGEMMWarps * kWarpSize; // 256
constexpr int kGEMMWarpsM    = 4;   // warps along M dimension
constexpr int kGEMMWarpsN    = 2;   // warps along N dimension

// WMMA fragment constants
constexpr int kWMMAFragM = 16;
constexpr int kWMMAFragN = 16;
constexpr int kWMMAFragK = 16;

// Fragments per warp along each dimension
constexpr int kFragsPerWarpM = kGEMMTileM / kWMMAFragM / kGEMMWarpsM; // 2
constexpr int kFragsPerWarpN = kGEMMTileN / kWMMAFragN / kGEMMWarpsN; // 4

// For decode (GEMV-style, use smaller tiles):
constexpr int kGEMVTileM = 8;
constexpr int kGEMVTileN = 64;
constexpr int kGEMVTileK = 64;

// ---------------------------------------------------------------------------
// Decode batched GEMV (multiple tokens, weight reuse)
// ---------------------------------------------------------------------------
// llama.cpp MMVQ processes up to 8 tokens simultaneously.
// ncols_dst = number of tokens to batch (2-8 for sweet spot)
constexpr int kMaxBatchSize  = 8;  // MMVQ_MAX_BATCH_SIZE from llama.cpp
constexpr int kINT8BlockSize = 16; // INT8 block-scaled block (matching B=16)
constexpr int kWarpCount     = 4;  // warps per block for ncols_dst=1-4
constexpr int kRowsPerBlock  = 1;  // output rows per CTA (1 for single-token)

// ---------------------------------------------------------------------------
// CUDA kernel launch grid helpers
// ---------------------------------------------------------------------------
inline constexpr int num_blocks(int totalThreads, int blockSize) {
    return (totalThreads + blockSize - 1) / blockSize;
}

} // namespace blackwell
#endif // BLACKWELL_CONFIG_H
