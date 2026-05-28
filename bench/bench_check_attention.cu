// bench_check_attention.cu — Debug flash attention correctness
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "blackwell/kernels.h"

static void die(cudaError_t e) {
    if (e != cudaSuccess) { printf("FAIL: %s\n", cudaGetErrorString(e)); exit(1); }
}

// attn1: 1 thr per element
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
    const int M=128,H=64,QH=12,kvH=1;
    float sc=1.0f/sqrtf(H);
    size_t N=QH*M*H, Nk=kvH*M*H;
    std::vector<float> Q(N),K(Nk),V(Nk);
    for(int i=0;i<(int)N;++i)Q[i]=((i*17+13)%127-63)*0.01f;
    for(int i=0;i<(int)Nk;++i){K[i]=((i*23+7)%127-63)*0.01f;V[i]=((i*31+11)%127-63)*0.01f;}

    float *dQ,*dK,*dV,*dO1,*dO2;
    die(cudaMalloc(&dQ,N*4));
    die(cudaMalloc(&dK,N*4));  // replicated for attn1
    die(cudaMalloc(&dV,N*4));
    die(cudaMalloc(&dO1,N*4));
    die(cudaMalloc(&dO2,N*4));
    die(cudaMemcpy(dQ,Q.data(),N*4,cudaMemcpyHostToDevice));
    for(int h=0;h<QH;++h){
        die(cudaMemcpy(dK+h*M*H,K.data(),Nk*4,cudaMemcpyHostToDevice));
        die(cudaMemcpy(dV+h*M*H,V.data(),Nk*4,cudaMemcpyHostToDevice));
    }

    // attn1
    dim3 g1(QH,M,(H+255)/256); dim3 b1(256);
    attn1<<<g1,b1,0>>>(dO1,dQ,dK,dV,M,H,QH,sc);
    die(cudaDeviceSynchronize());
    die(cudaPeekAtLastError());

    // our kernel
    die(blackwell::kernels::attention_prefill(dO2,dQ,dK,dV,M,H,QH,kvH,QH/kvH,sc,0));
    die(cudaStreamSynchronize(0));
    die(cudaPeekAtLastError());

    std::vector<float> O1(N),O2(N);
    die(cudaMemcpy(O1.data(),dO1,N*4,cudaMemcpyDeviceToHost));
    die(cudaMemcpy(O2.data(),dO2,N*4,cudaMemcpyDeviceToHost));

    // Compare first 20 elements of head 0, row 0
    printf("Head 0, row 0 (d=0..19):\n  baseline: ");
    for(int d=0;d<20;++d) printf("%+.6f ",O1[0*M*H+0*H+d]);
    printf("\n  flash:    ");
    for(int d=0;d<20;++d) printf("%+.6f ",O2[0*M*H+0*H+d]);
    printf("\n  diff:     ");
    for(int d=0;d<20;++d) printf("%+.2e ",O2[0*M*H+0*H+d]-O1[0*M*H+0*H+d]);
    printf("\n");

    // Check S_s values for head 0, row 0
    // We can't access S_s from host, but compare scores
    printf("\nHead 0, row 1 (d=0..19):\n  baseline: ");
    for(int d=0;d<20;++d) printf("%+.6f ",O1[0*M*H+1*H+d]);
    printf("\n  flash:    ");
    for(int d=0;d<20;++d) printf("%+.6f ",O2[0*M*H+1*H+d]);
    printf("\n  diff:     ");
    for(int d=0;d<20;++d) printf("%+.2e ",O2[0*M*H+1*H+d]-O1[0*M*H+1*H+d]);
    printf("\n");

    // Count mismatches
    int bad=0; float max_diff=0;
    for(int i=0;i<(int)N;++i){
        float d=fabsf(O2[i]-O1[i]);
        float r=d/(fabsf(O1[i])+1e-9f);
        if(r>0.001f)bad++;
        if(d>max_diff)max_diff=d;
    }
    printf("\nMismatches (>0.1%%): %d/%zu\n",bad,N);
    printf("Max diff: %.2e\n",max_diff);

    cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO1);cudaFree(dO2);
    return 0;
}