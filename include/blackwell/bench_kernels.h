// include/blackwell/bench_kernels.h — Shared CUDA kernels for bench files
//
// head_norm_kernel: per-head RMSNorm for Q/K after GEMV
// apply_rope_kernel: in-place RoPE for a single position
//
// Avoids 14+ copies of identical code across bench/ and server/ files.

#pragma once
#ifndef BLACKWELL_BENCH_KERNELS_H
#define BLACKWELL_BENCH_KERNELS_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ── Per-head RMSNorm ─────────────────────────────────────────────────────
// Launched as <<<nh, 128>>>. Applies weight[0..hd-1] to data[h*hd .. h*hd+hd-1].
__global__ void head_norm_kernel(float* data, const float* weight, int nh, int hd, float eps) {
    int h=blockIdx.x; if(h>=nh) return;
    float* d=data+h*hd;
    __shared__ float wp[4]; float s=0;
    for(int i=threadIdx.x;i<hd;i+=blockDim.x) s+=d[i]*d[i];
    for(int off=16;off>0;off>>=1) s+=__shfl_xor_sync(0xffffffff,s,off);
    if((threadIdx.x&31)==0) wp[threadIdx.x>>5]=s; __syncthreads();
    if(threadIdx.x<4) s=wp[threadIdx.x]; else s=0;
    for(int off=2;off>0;off>>=1) s+=__shfl_xor_sync(0xffffffff,s,off);
    if(threadIdx.x==0) wp[0]=rsqrtf(s/hd+eps); __syncthreads();
    float is=wp[0];
    for(int i=threadIdx.x;i<hd;i+=blockDim.x) d[i]=d[i]*is*weight[i];
}

// ── In-place RoPE (single position) ──────────────────────────────────────
// Launched as <<<n_heads, head_dim/2>>>. Applies cos/sin rotation to each pair.
__global__ void apply_rope_kernel(float* data, int n_heads, int head_dim, int pos, float rope_theta) {
    int h=blockIdx.x; int d=threadIdx.x;
    if(h>=n_heads||d>=head_dim/2) return;
    float* pair=data+h*head_dim+d*2;
    float theta=(float)pos*powf(rope_theta,-2.0f*(float)d/(float)head_dim);
    float c=cosf(theta),s=sinf(theta),x=pair[0],y=pair[1];
    pair[0]=x*c-y*s; pair[1]=x*s+y*c;
}

#endif // BLACKWELL_BENCH_KERNELS_H
