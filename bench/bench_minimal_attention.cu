// bench_minimal_attention.cu — Minimal correctness test for flash attention
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "blackwell/kernels.h"

static void die(cudaError_t e) {
    if (e != cudaSuccess) { printf("FAIL: %s\n", cudaGetErrorString(e)); exit(1); }
}

// attn1: single-element baseline
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
    // Minimal test: QH=1, kvH=1, M=1, H=4 (tiny)
    const int M=1, H=4, QH=1, kvH=1;
    float sc=1.0f/sqrtf(H);
    size_t N=QH*M*H; // 4

    std::vector<float> Q(N), K(N), V(N);
    // Simple values that let us hand-verify
    Q = {1.0f, 0.0f, 0.0f, 0.0f};  // Q[0] = (1,0,0,0)
    K = {2.0f, 0.0f, 0.0f, 0.0f};  // K[0] = (2,0,0,0)
    V = {5.0f, 6.0f, 7.0f, 8.0f};  // V[0] = (5,6,7,8)

    float *dQ, *dK, *dV, *dO1, *dO2;
    die(cudaMalloc(&dQ,N*4)); die(cudaMalloc(&dK,N*4));
    die(cudaMalloc(&dV,N*4)); die(cudaMalloc(&dO1,N*4)); die(cudaMalloc(&dO2,N*4));
    die(cudaMemcpy(dQ,Q.data(),N*4,cudaMemcpyHostToDevice));
    die(cudaMemcpy(dK,K.data(),N*4,cudaMemcpyHostToDevice));
    die(cudaMemcpy(dV,V.data(),N*4,cudaMemcpyHostToDevice));

    // attn1
    dim3 g1(QH,M,(H+255)/256); dim3 b1(256);
    attn1<<<g1,b1,0>>>(dO1,dQ,dK,dV,M,H,QH,sc);
    die(cudaDeviceSynchronize()); die(cudaPeekAtLastError());

    // Expected: Q·K = 1*2 + 0*0 + 0*0 + 0*0 = 2.0
    // scale = 1/sqrt(4) = 0.5
    // score = 2.0 * 0.5 = 1.0
    // softmax over 1 element: exp(1.0)/exp(1.0) = 1.0
    // O = 1.0 * (5,6,7,8) = (5,6,7,8)

    // our kernel
    die(blackwell::kernels::attention_prefill(dO2,dQ,dK,dV,M,H,QH,kvH,QH/kvH,sc,0));
    die(cudaStreamSynchronize(0)); die(cudaPeekAtLastError());

    std::vector<float> O1(N),O2(N);
    die(cudaMemcpy(O1.data(),dO1,N*4,cudaMemcpyDeviceToHost));
    die(cudaMemcpy(O2.data(),dO2,N*4,cudaMemcpyDeviceToHost));

    printf("M=1, H=4, QH=1, kvH=1\n");
    printf("Expected O = (5, 6, 7, 8)\n");
    printf("attn1:   ");
    for(auto x:O1) printf("%+.4f ",x);
    printf("\nflash:   ");
    for(auto x:O2) printf("%+.4f ",x);
    printf("\ndiff:    ");
    for(int i=0;i<(int)N;++i) printf("%+.2e ",O2[i]-O1[i]);
    printf("\n");

    // Test with M=4, simpler values
    printf("\n--- Test M=4 ---\n");
    const int M2=4, H2=4, QH2=1, kvH2=1;
    size_t N2=QH2*M2*H2;
    std::vector<float> Q2 = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}; // identity matrix
    std::vector<float> K2 = {2,0,0,0, 0,3,0,0, 0,0,4,0, 0,0,0,5}; // diagonal
    std::vector<float> V2 = {10,11,12,13, 20,21,22,23, 30,31,32,33, 40,41,42,43};
    
    float *dQ2,*dK2,*dV2,*dOa,*dOb;
    die(cudaMalloc(&dQ2,N2*4)); die(cudaMalloc(&dK2,N2*4));
    die(cudaMalloc(&dV2,N2*4)); die(cudaMalloc(&dOa,N2*4)); die(cudaMalloc(&dOb,N2*4));
    die(cudaMemcpy(dQ2,Q2.data(),N2*4,cudaMemcpyHostToDevice));
    die(cudaMemcpy(dK2,K2.data(),N2*4,cudaMemcpyHostToDevice));
    die(cudaMemcpy(dV2,V2.data(),N2*4,cudaMemcpyHostToDevice));

    dim3 g2a(QH2,M2,(H2+255)/256); dim3 b2a(256);
    attn1<<<g2a,b2a,0>>>(dOa,dQ2,dK2,dV2,M2,H2,QH2,sc);
    die(cudaDeviceSynchronize()); die(cudaPeekAtLastError());

    die(blackwell::kernels::attention_prefill(dOb,dQ2,dK2,dV2,M2,H2,QH2,kvH2,QH2/kvH2,sc,0));
    die(cudaStreamSynchronize(0)); die(cudaPeekAtLastError());

    std::vector<float> Oa(N2),Ob(N2);
    die(cudaMemcpy(Oa.data(),dOa,N2*4,cudaMemcpyDeviceToHost));
    die(cudaMemcpy(Ob.data(),dOb,N2*4,cudaMemcpyDeviceToHost));

    printf("  attn1: "); for(auto x:Oa) printf("%+.4f ",x); printf("\n");
    printf("  flash: "); for(auto x:Ob) printf("%+.4f ",x); printf("\n");
    printf("  diff:  "); for(int i=0;i<(int)N2;++i) printf("%+.2e ",Ob[i]-Oa[i]); printf("\n");

    int bad=0; float md=0;
    for(int i=0;i<(int)N2;++i){
        float d=fabsf(Ob[i]-Oa[i]);
        if(d>0.001f)bad++;
        if(d>md)md=d;
    }
    printf("  mismatches: %d/%zu, max_diff=%.2e\n",bad,N2,md);

    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO1);cudaFree(dO2);
    cudaFree(dQ2);cudaFree(dK2);cudaFree(dV2);cudaFree(dOa);cudaFree(dOb);
    return bad>0?1:0;
}