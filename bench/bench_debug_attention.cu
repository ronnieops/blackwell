// bench_debug_attention.cu — Debug flash attention with H=64, simple values
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "blackwell/kernels.h"

static void die(cudaError_t e) {
    if (e != cudaSuccess) { printf("FAIL: %s\n", cudaGetErrorString(e)); exit(1); }
}

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

int main() {
    const int M=128, H=64, QH=12, kvH=1;
    float sc=1.0f/sqrtf(H);
    size_t N=QH*M*H;
    size_t Nk=kvH*M*H;

    // Deterministic data (same pattern as main benchmark)
    std::vector<float> Q(N), K(Nk), V(Nk);
    for(int i=0;i<(int)N;++i) Q[i]=((i*17+13)%127-63)*0.01f;
    for(int i=0;i<(int)Nk;++i){K[i]=((i*23+7)%127-63)*0.01f;V[i]=((i*31+11)%127-63)*0.01f;}

    float *dQ,*dK,*dV,*dO1,*dO2;
    die(cudaMalloc(&dQ,N*4));die(cudaMalloc(&dK,N*4));die(cudaMalloc(&dV,N*4)); // replicated for attn1
    die(cudaMalloc(&dO1,N*4));die(cudaMalloc(&dO2,N*4));
    die(cudaMemcpy(dQ,Q.data(),N*4,cudaMemcpyHostToDevice));
    // Replicate K/V across QH heads for attn1 (doesn't support GQA)
    for(int h=0;h<QH;++h){
        die(cudaMemcpy(dK+h*M*H,K.data(),M*H*4,cudaMemcpyHostToDevice));
        die(cudaMemcpy(dV+h*M*H,V.data(),M*H*4,cudaMemcpyHostToDevice));
    }

    // attn1
    dim3 g1(QH,M,(H+255)/256);dim3 b1(256);
    attn1<<<g1,b1,0>>>(dO1,dQ,dK,dV,M,H,QH,sc);
    die(cudaDeviceSynchronize());die(cudaPeekAtLastError());

    die(blackwell::kernels::attention_prefill(dO2,dQ,dK,dV,M,H,QH,kvH,QH/kvH,sc,0));
    die(cudaStreamSynchronize(0));die(cudaPeekAtLastError());

    std::vector<float> O1(N),O2(N);
    die(cudaMemcpy(O1.data(),dO1,N*4,cudaMemcpyDeviceToHost));
    die(cudaMemcpy(O2.data(),dO2,N*4,cudaMemcpyDeviceToHost));

    // Q = [1,0,0,...], K[j] = [0..j, j+1, 0...] (j at diagonal)
    // Q·K[0] = 1*1 + 0*... = 1
    // Q·K[j>0] = 0 (Q is all zeros except Q[0], K[j][0]=0 for j>0)
    // score[0] = 1 * sc = 1/sqrt(64) = 0.125
    // score[1..7] = 0
    // softmax: exp(0.125) / (exp(0.125) + 7*exp(0)) = 1.133 / (1.133+7) = 0.139
    //           each wrong pos = 1 / (1.133+7) = 0.123
    // O[0] = 0.139*V[0] + 0.123*(V[1]+...+V[7])
    printf("M=%d, H=%d, QH=%d, kvH=%d\n",M,H,QH,kvH);
    printf("Head 0 row 0 d=0..7:\n  attn1: "); for(int d=0;d<8;++d) printf("%+.4e ",O1[0*H+d]); printf("\n");
    printf("  flash: "); for(int d=0;d<8;++d) printf("%+.4e ",O2[0*H+d]); printf("\n");
    printf("  diff:  "); for(int d=0;d<8;++d) printf("%+.2e ",O1[0*H+d]-O2[0*H+d]); printf("\n");

    int bad=0; float md=0;
    for(int i=0;i<(int)N;++i){
        float d=fabsf(O1[i]-O2[i]);
        if(d>0.001f)bad++;
        if(d>md)md=d;
    }
    printf("\nBad: %d/%zu, max_diff=%.2e\n",bad,N,md);

    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO1);cudaFree(dO2);
    return bad>0?1:0;
}