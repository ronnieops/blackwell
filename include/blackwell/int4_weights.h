// include/blackwell/int4_weights.h — INT4 weight loader + embed dequantizer
//
// Shared implementations for bench and server files:
//   DevW4f16       — GPU INT4 weight with FP16 per-block scales
//   upload_w4_f16sc — load from .int4_t + .scale_t files to GPU
//   dequant_embed_row — CPU-side INT4 embed table dequant (host row → FP32)
//
// Avoids 20+ copies of identical code across bench/ and server/ files.

#pragma once
#ifndef BLACKWELL_INT4_WEIGHTS_H
#define BLACKWELL_INT4_WEIGHTS_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

namespace blackwell {
namespace weights {

// ── INT4 weight struct (FP16 per-block scales) ───────────────────────────
struct DevW4f16 {
    int K, N;
    uint8_t* d;     // packed INT4 data on GPU [K/2 * N]
    __half* sc16;   // per-block FP16 scales on GPU [N * (K/16)]
};

// ── Upload INT4 weight from .int4_t + .scale_t files to GPU ─────────────
// Reads header, allocates GPU memory, copies data + scales.
inline DevW4f16 upload_w4_f16sc(const char* prefix) {
    char p[264]; snprintf(p,264,"%s.int4_t",prefix);
    FILE* f=fopen(p,"rb"); if(!f){fprintf(stderr,"FAIL open %s\n",p);exit(1);}
    int h[5]; size_t nr=fread(h,4,5,f); if(nr!=5){fprintf(stderr,"FAIL read header %s\n",p);exit(1);}
    DevW4f16 dw; dw.K=h[0]; dw.N=h[1];
    size_t ds=(size_t)h[0]*h[1]/2;
    uint8_t* td=new uint8_t[ds]; nr=fread(td,1,ds,f); if(nr!=ds){fprintf(stderr,"FAIL read data %s\n",p);exit(1);} fclose(f);
    cudaMalloc(&dw.d,ds); cudaMemcpy(dw.d,td,ds,cudaMemcpyHostToDevice); delete[] td;
    snprintf(p,264,"%s.scale_t",prefix); f=fopen(p,"rb"); if(!f){fprintf(stderr,"FAIL open %s\n",p);exit(1);} nr=fread(h,4,5,f); if(nr!=5){fprintf(stderr,"FAIL read scale header %s\n",p);exit(1);}
    size_t ss=(size_t)h[3]*h[4];
    __half* ts=new __half[ss]; nr=fread(ts,2,ss,f); if(nr!=ss){fprintf(stderr,"FAIL read scales %s\n",p);exit(1);} fclose(f);
    cudaMalloc(&dw.sc16,ss*2); cudaMemcpy(dw.sc16,ts,ss*2,cudaMemcpyHostToDevice); delete[] ts;
    return dw;
}

// ── CPU-side INT4 embed dequantizer ─────────────────────────────────────
// Dequantizes one row from packed INT4 embed table to FP32 output vector.
// out: [K] FP32 (host buffer, pre-allocated)
// token: row index into embed table
// host_w: [V * K/2] packed INT4 data (host)
// host_sc: [V * (K/16)] per-block FP32 scales (host, pre-converted from FP16)
// K: hidden dimension
inline void dequant_embed_row(float* out, int token, const uint8_t* host_w,
    const float* host_sc, int K)
{
    int kblocks = K / 16;
    for (int b = 0; b < kblocks; ++b) {
        float sc = host_sc[token * kblocks + b];
        for (int i = 0; i < 16; ++i) {
            size_t byte_idx = (size_t)token * K / 2 + (size_t)b * 8 + i / 2;
            uint8_t byte = host_w[byte_idx];
            int nib = (i & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
            out[b * 16 + i] = (float)(nib - 8) * sc;
        }
    }
}

} // namespace weights
} // namespace blackwell

#endif // BLACKWELL_INT4_WEIGHTS_H