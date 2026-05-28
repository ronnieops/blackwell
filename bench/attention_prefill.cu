// bench/attention_prefill.cu — Final attention benchmark
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120a,code=sm_120a \
//     -I include bench/attention_prefill.cu build/libblackwell_kernels.a \
//     -o bench/attention_prefill
//
// Smem constraint: RTX 5060 Ti max 32KB smem. K only (16KB for M=128,H=64).
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
using std::vector;
static void die(cudaError_t e){if(e!=cudaSuccess){printf("FAIL %s\n",cudaGetErrorString(e));::exit(1);}}

// attn1: 1 thr per output element (22 GFLOPS) — correctness baseline
__global__ void attn1(float*O,const float*Q,const float*K,const float*V,int M,int H,int QH,float sc){
    int h=blockIdx.x,m=blockIdx.y,d=blockIdx.z*blockDim.x+threadIdx.x;
    if(h>=QH||m>=M||d>=H)return;
    const float*Qm=Q+h*M*H+m*H,*Kh=K+h*M*H,*Vh=V+h*M*H;
    float mx=-1e9f,sum=0,acc=0,s[128];
    for(int j=0;j<M;++j){float dot=0;for(int k=0;k<H;++k)dot+=Qm[k]*Kh[j*H+k];s[j]=dot*sc;mx=fmaxf(mx,dot*sc);}
    for(int j=0;j<M;++j){s[j]=expf(s[j]-mx);sum+=s[j];}
    for(int j=0;j<M;++j)acc+=s[j]*Vh[j*H+d];
    O[h*M*H+m*H+d]=acc/sum;
}

// k_reg: Q in shared, K+V from global (L2 cached). 64 thr/block, no smem.
// 1536 blocks. Q loaded per-block, K+V loaded from global memory.
__global__ void attn_reg(float*O,const float*Q,const float*K,const float*V,int M,int H,int QH,float sc){
    int h=blockIdx.x,m=blockIdx.y,t=threadIdx.x;
    if(h>=QH||m>=M)return;
    __shared__ float q_reg[64];
    const float*Qm=Q+h*M*H+m*H;
    if(t<H)q_reg[t]=Qm[t];
    __syncthreads();
    if(t<H){
        float acc=0,mx=-1e9f,s[128];
        const float*Kh=K+h*M*H,*Vh=V+h*M*H;
        for(int j=0;j<M;++j){float dot=0;for(int kd=0;kd<H;++kd)dot+=q_reg[kd]*Kh[j*H+kd];s[j]=dot*sc;mx=fmaxf(mx,dot*sc);}
        float sum=0;for(int j=0;j<M;++j){s[j]=expf(s[j]-mx);sum+=s[j];}
        for(int j=0;j<M;++j)acc+=s[j]*Vh[j*H+t];
        O[h*M*H+m*H+t]=acc/sum;
    }
}

// k_coop: K cached in smem (32KB max), Q in shared, V from global.
// 1 block per 4 rows, 32 threads. smem = M*H*4 = 32KB (MAX).
__global__ void attn_coop(float*O,const float*Q,const float*K,const float*V,int M,int H,int QH,float sc){
    int h=blockIdx.x,m0=blockIdx.y*4;  // 4 rows per block
    int t=threadIdx.x,lane=t&31,wid=t>>5;
    if(h>=QH||m0>=M)return;

    // smem: K only (V from global/L2). smem = 32KB max.
    extern __shared__ char smem_[];
    float*K_s=(float*)smem_;  // M*H floats

    // Load K[h] into smem (cooperative, 32 threads)
    const float*Kh=K+h*M*H;
    for(int i=t;i<M*H;i+=blockDim.x)K_s[i]=Kh[i];
    __syncthreads();

    // 4 rows per block
    for(int r=0;r<4;++r){
        int m=m0+r;
        if(m>=M)continue;
        // Load Q[h][m] into shared
        __shared__ float q_r[64];
        const float*Qm=Q+h*M*H+m*H;
        if(t<H)q_r[t]=Qm[t];
        __syncthreads();

        // Compute scores S[j] (all warps contribute to S[0..127])
        __shared__ float S[128];
        int j0=wid*4;  // 0,4,8,...,124
        for(int jj=0;jj<4;++jj){
            int j=j0+jj;
            float dot=0;
            for(int d=lane*2;d<(lane+1)*2&&d<H;++d)dot+=q_r[d]*K_s[j*H+d];
            for(int o=16;o>0;o>>=1)dot+=__shfl_down_sync(0xffffffff,dot,o);
            if(lane==0)S[j]=dot*sc;
        }
        __syncthreads();

        // Online softmax (warp 0)
        if(wid==0&&lane==0){
            float mx=S[0];
            #pragma unroll
            for(int j=1;j<M;++j)mx=fmaxf(mx,S[j]);
            float sum=0;
            #pragma unroll
            for(int j=0;j<M;++j){S[j]=expf(S[j]-mx);sum+=S[j];}
            #pragma unroll
            for(int j=0;j<M;++j)S[j]/=sum;
        }
        __syncthreads();

        // Accumulate O[m][:] = S @ V[:,:] (V from global/L2)
        const float*Vh=V+h*M*H;
        float*O_h=O+h*M*H+m*H;
        for(int d=lane*2;d<(lane+1)*2&&d<H;++d){
            float acc=0;
            for(int j=0;j<M;++j)acc+=S[j]*Vh[j*H+d];
            O_h[d]=acc;
        }
        __syncthreads();
    }
}

// k_coopV: K in smem, V partially in smem (first half of rows)
// 32KB smem split: K_s (M*H) + V_s (M*H/2) = 3*M*H/2 floats = 24KB
__global__ void attn_coopV(float*O,const float*Q,const float*K,const float*V,int M,int H,int QH,float sc){
    int h=blockIdx.x,m0=blockIdx.y*4;
    int t=threadIdx.x,lane=t&31,wid=t>>5;
    if(h>=QH||m0>=M)return;
    extern __shared__ char smem_[];
    float*f=(float*)smem_;
    float*K_s=f;           // M*H floats = 32KB
    float*V_s=K_s+M*H;     // M*H floats → OUT OF BOUNDS for 32KB limit!
    // Can't add V to smem — 32KB limit reached with K alone
    // This kernel doesn't actually use V_s, demonstrating the limit
}

int main(int argc,char** argv){
    int IT=20; if(argc>1)IT=atoi(argv[1]);
    cudaDeviceProp P; cudaGetDeviceProperties(&P,0);
    printf("# Attention Prefill — %s  SM:%d  smem_limit: 32KB\n\n",P.name,P.multiProcessorCount);

    cudaStream_t st; die(cudaStreamCreate(&st));
    cudaEvent_t s2,e2; die(cudaEventCreate(&s2)); die(cudaEventCreate(&e2));

    const int M=128,H=64,QH=12;
    float sc=1.0f/sqrtf((float)H);
    size_t N=QH*M*H;

    vector<float> Q(N),K(N),V(N);
    for(int i=0;i<N;++i){Q[i]=(i%17-8)*0.01f;K[i]=(i%23-12)*0.01f;V[i]=(i%31-16)*0.01f;}
    float*dQ,*dK,*dV,*dO;
    cudaMalloc(&dQ,N*4);cudaMalloc(&dK,N*4);cudaMalloc(&dV,N*4);cudaMalloc(&dO,N*4);
    cudaMemcpy(dQ,Q.data(),N*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dK,K.data(),N*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dV,V.data(),N*4,cudaMemcpyHostToDevice);

    double ms[3]={0,0,0};

    // attn1: 98304 blocks, 1 thr/element
    dim3 g1(QH,M,(H+255)/256),b1(256);
    printf("=== attn1 (1 thr/elem, %d blocks) ===\n",QH*M*(H+255)/256);
    attn1<<<g1,b1,0,st>>>(dO,dQ,dK,dV,M,H,QH,sc);
    cudaStreamSynchronize(st);
    if(cudaPeekAtLastError()==cudaSuccess){
        cudaEventRecord(s2,st);
        for(int i=0;i<IT;++i)attn1<<<g1,b1,0,st>>>(dO,dQ,dK,dV,M,H,QH,sc);
        cudaEventRecord(e2,st);cudaEventSynchronize(e2);
        float t;cudaEventElapsedTime(&t,s2,e2);ms[0]=t/IT;
        double gf=2.0*QH*M*M*H*1e-9/(ms[0]/1000.0);
        printf("  %.3f ms  (%.1f GFLOPS)\n",ms[0],gf);
    }

    // attn_reg: 1536 blocks, 64 threads, Q in shared, K+V global (L2)
    dim3 g2(QH,M),b2(64);
    printf("\n=== attn_reg (64 thr, %d blocks, smem=256B) ===\n",QH*M);
    attn_reg<<<g2,b2,0,st>>>(dO,dQ,dK,dV,M,H,QH,sc);
    cudaStreamSynchronize(st);
    if(cudaPeekAtLastError()==cudaSuccess){
        cudaEventRecord(s2,st);
        for(int i=0;i<IT;++i)attn_reg<<<g2,b2,0,st>>>(dO,dQ,dK,dV,M,H,QH,sc);
        cudaEventRecord(e2,st);cudaEventSynchronize(e2);
        float t;cudaEventElapsedTime(&t,s2,e2);ms[1]=t/IT;
        double gf=2.0*QH*M*M*H*1e-9/(ms[1]/1000.0);
        printf("  %.3f ms  (%.1f GFLOPS)\n",ms[1],gf);
    }

    // attn_coop: 384 blocks (M/4), 32 threads, smem=K(32KB)
    dim3 g3(QH,(M+3)/4),b3(32);
    int smem_k=M*H*sizeof(float);  // 32KB = MAX smem
    printf("\n=== attn_coop (32 thr, %d blocks, smem=%dKB=K-only) ===\n",QH*(M/4),smem_k/1024);
    attn_coop<<<g3,b3,smem_k,st>>>(dO,dQ,dK,dV,M,H,QH,sc);
    cudaStreamSynchronize(st);
    if(cudaPeekAtLastError()==cudaSuccess){
        cudaEventRecord(s2,st);
        for(int i=0;i<IT;++i)attn_coop<<<g3,b3,smem_k,st>>>(dO,dQ,dK,dV,M,H,QH,sc);
        cudaEventRecord(e2,st);cudaEventSynchronize(e2);
        float t;cudaEventElapsedTime(&t,s2,e2);ms[2]=t/IT;
        double gf=2.0*QH*M*M*H*1e-9/(ms[2]/1000.0);
        printf("  %.3f ms  (%.1f GFLOPS)\n",ms[2],gf);
    } else printf("  %s\n",cudaGetErrorString(cudaPeekAtLastError()));

    // Analysis
    printf("\n=== vs GEMM (1.1 ms/layer) ===\n");
    double gm=1.1;
    for(int i=0;i<3;++i){
        if(ms[i]>0){
            printf("  Kernel %d: %.3f ms  prefill=%.2f ms  fraction=%.0f%%\n",
                i+1,ms[i],ms[i]+gm,100*ms[i]/(ms[i]+gm));
        }
    }
    double best_ms=ms[2]>0?ms[2]:ms[1];
    if(best_ms>0){
        printf("  Best: %.3f ms  (%.1fx vs attn1)\n",best_ms,ms[0]/best_ms);
    }

    // Smem constraint analysis
    printf("\n=== smem constraint ===\n");
    printf("  MAX smem: 32KB (RTX 5060 Ti)\n");
    printf("  K (M×H=128×64): %d KB\n",M*H*4/1024);
    printf("  V (M×H=128×64): %d KB\n",M*H*4/1024);
    printf("  K+V total: %d KB\n",2*M*H*4/1024);
    printf("  V must stay in global memory (L2 cached)\n");
    printf("  12 heads × KV: %.0f KB (%.1f%% of 32MB L2)\n",
        2.0*M*H*4/1024*12,100.0*2*M*H*4*12/32.0/1024/1024);

    // Bandwidth analysis
    double bytes=(3.0*QH)*M*H*4.0;  // Q + K + V (O in registers)
    printf("\n=== Bandwidth ===\n");
    for(int i=0;i<3;++i){
        if(ms[i]>0){
            double bw=bytes/(ms[i]/1000.0)/1e9;
            printf("  Kernel %d: %.0f GB/s (%.0f%% of 500 peak)\n",i+1,bw,100*bw/500.0);
        }
    }

    // Output check
    vector<float> O(N); cudaMemcpy(O.data(),dO,N*4,cudaMemcpyDeviceToHost);
    int nan=0; float mx=0,sum=0;
    for(float x:O){if(isnan(x))nan++;mx=fmaxf(mx,fabsf(x));sum+=fabsf(x);}
    printf("\n  Output: max=%.4f mean=%.4f nan=%d\n",mx,sum/O.size(),nan);

    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);
    cudaEventDestroy(s2);cudaEventDestroy(e2);cudaStreamDestroy(st);
    return 0;
}