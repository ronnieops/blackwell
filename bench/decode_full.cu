// bench/decode_full.cu — Full transformer decode step (attention + MLP)
//
// Per layer:
//   Attention:
//     Q,K,V = fused_qkv(x, W_q, W_k, W_v)
//     update_kv_cache(K, V)
//     attn_out = attention_decode(Q, K_cache, V_cache)
//     x = rmsnorm_pack(attn_out @ W_o + x_residual)
//   MLP:
//     gate = gemv(x, W_gate)
//     up   = gemv(x, W_up)
//     mlp  = SwiGLU(gate, up)
//     x    = rmsnorm_pack(mlp @ W_down + x_residual)
//
// Uses real Qwen3-1.7B weights from weights/ directory.
//
// Build:
//   CUDACXX=/usr/local/cuda-12.8/bin/nvcc nvcc -O3 -std=c++17 \
//     -gencode=arch=compute_120,code=sm_120 \
//     -I include bench/decode_full.cu build/libblackwell_kernels.a \
//     -o bench/decode_full

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include <cstdint>
#include "blackwell/kernels.h"

struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin() { cudaEventRecord(start, 0); }
    float end() { cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
                  float ms=0; cudaEventElapsedTime(&ms, start, stop); return ms; }
};

struct KernelTimer {
    const char* name;
    double total_ms = 0;
    int count = 0;
    KernelTimer(const char* n) : name(n) {}
    void add(float ms) { total_ms += ms; ++count; }
    double avg() const { return count > 0 ? total_ms / count : 0; }
    void print() const { printf("  %-30s  %7.3f ms (%d calls)\n", name, total_ms, count); }
};

static bool check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) { printf("FAIL: %s: %s\n", msg, cudaGetErrorString(e)); return false; }
    return true;
}

struct WeightFile {
    int dim1, dim0, block;
    int num_K_blocks, num_N_blocks;
    std::vector<uint8_t> data;
    std::vector<float> scales;
};

WeightFile load_weight(const char* path) {
    WeightFile w;
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    int header[5];
    if (fread(header, 4, 5, f) != 5) { fprintf(stderr, "Bad header %s\n", path); exit(1); }
    w.dim1 = header[0]; w.dim0 = header[1]; w.block = header[2];
    w.num_K_blocks = header[3]; w.num_N_blocks = header[4];
    size_t n = (size_t)w.dim1 * w.dim0;
    w.data.resize(n);
    if (fread(w.data.data(), 1, n, f) != n) { fprintf(stderr, "Bad data %s\n", path); exit(1); }
    size_t nscales = (size_t)w.num_K_blocks * w.num_N_blocks;
    w.scales.resize(nscales);
    if (fread(w.scales.data(), 4, nscales, f) != nscales) { fprintf(stderr, "Bad scales %s\n", path); exit(1); }
    fclose(f);
    return w;
}

struct DevWeight {
    void* data;
    float* scales;
};

DevWeight upload_weight(const WeightFile& w) {
    DevWeight d;
    size_t n = (size_t)w.dim1 * w.dim0;
    cudaMalloc(&d.data, n);
    cudaMemcpy(d.data, w.data.data(), n, cudaMemcpyHostToDevice);
    // L2 persistence hint: keep weight data resident in L2 cache
    cudaMemAdvise(d.data, n, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(d.data, n, cudaMemAdviseSetAccessedBy, 0);
    size_t nscales = (size_t)w.num_K_blocks * w.num_N_blocks;
    cudaMalloc(&d.scales, nscales * 4);
    cudaMemcpy(d.scales, w.scales.data(), nscales * 4, cudaMemcpyHostToDevice);
    return d;
}

// Transposed weight for v2 GEMV
struct DevWeightT {
    void* data;      // [dim0 × dim1] transposed
    float* scales;   // [dim0/16 × dim1/16] transposed
};

DevWeightT upload_transposed_weight(const WeightFile& w) {
    DevWeightT d;
    size_t n = (size_t)w.dim1 * w.dim0;  // same total bytes
    cudaMalloc(&d.data, n);
    size_t nscales = (size_t)w.num_K_blocks * w.num_N_blocks;
    cudaMalloc(&d.scales, nscales * 4);

    // Upload original, then transpose on GPU
    DevWeight orig = upload_weight(w);
    blackwell::kernels::transpose_fp4_weights(
        d.data, d.scales, orig.data, orig.scales,
        w.dim1, w.dim0, 0);
    cudaFree(orig.data);
    cudaFree(orig.scales);

    // L2 persistence hint: keep transposed weight data resident
    cudaMemAdvise(d.data, n, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(d.data, n, cudaMemAdviseSetAccessedBy, 0);
    cudaMemAdvise(d.scales, nscales * 4, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    cudaMemAdvise(d.scales, nscales * 4, cudaMemAdviseSetAccessedBy, 0);

    return d;
}

int main(int argc, char** argv) {
    int num_layers = 4;
    if (argc > 1) num_layers = atoi(argv[1]);
    if (num_layers > 28) num_layers = 28;

    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("# Full Decode Benchmark — Attention + MLP\n");
    printf("Device: %s (CC %d.%d)\n", p.name, p.major, p.minor);
    printf("Layers: %d\n\n", num_layers);

    // L2 persistence: reserve 256KB for persisting weight matrix access.
    // Weight matrices (1.5-4MB each) reused across split-K blocks stay cached.
    cudaError_t l2e = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 256 * 1024);
    if (l2e != cudaSuccess) {
        printf("  Note: L2 persisting = %s\n", cudaGetErrorString(l2e));
    } else {
        printf("  L2 persisting: 256KB reserved\n");
    }

    // Qwen3-1.7B params
    const int hidden = 2048;
    const int q_dim = 2048;       // 16 heads × 128
    const int kv_dim = 1024;      // 8 KV heads × 128
    const int intermediate = 6144;
    const int num_q_heads = 16, num_kv_heads = 8;
    const int head_dim = 128;
    const int max_seq = 2048;
    const float scale_1_3 = 1.0f / 3.0f;

    // Load weights
    printf("Loading weights...\n");
    auto load = [&](const char* fmt, int l) {
        char path[256];
        snprintf(path, 256, fmt, l);
        WeightFile w = load_weight(path);
        DevWeightT d = upload_transposed_weight(w);
        printf("  %s: %d×%d (transposed)\n", path, w.dim1, w.dim0);
        return d;
    };

    struct LayerWeights {
        DevWeightT Wq, Wk, Wv, Wo;
        DevWeightT Wgate, Wup, Wdown;
    };
    std::vector<LayerWeights> lw(num_layers);

    for (int l = 0; l < num_layers; ++l) {
        lw[l].Wq    = load("weights/%d_self_attn.q_proj.fp4", l);
        lw[l].Wk    = load("weights/%d_self_attn.k_proj.fp4", l);
        lw[l].Wv    = load("weights/%d_self_attn.v_proj.fp4", l);
        lw[l].Wo    = load("weights/%d_self_attn.o_proj.fp4", l);
        lw[l].Wgate = load("weights/%d_mlp.gate_proj.fp4", l);
        lw[l].Wup   = load("weights/%d_mlp.up_proj.fp4", l);
        lw[l].Wdown = load("weights/%d_mlp.down_proj.fp4", l);
    }

    // Allocate buffers
    float *d_x32, *d_xs, *d_Q, *d_K, *d_V, *d_attn, *d_proj;
    float *d_gate, *d_up, *d_mlp;          // MLP intermediates
    float *d_mlp_out;                      // MLP FP32 output (+residual)
    void *d_x_fp4, *d_attn_fp4, *d_mlp_fp4;
    float *d_attn_s, *d_mlp_s;
    cudaMalloc(&d_x32, hidden * 4);
    cudaMalloc(&d_x_fp4, hidden);
    cudaMalloc(&d_xs, (hidden/16) * 4);
    cudaMalloc(&d_Q, q_dim * 4);
    cudaMalloc(&d_K, kv_dim * 4);
    cudaMalloc(&d_V, kv_dim * 4);
    cudaMalloc(&d_attn, q_dim * 4);
    cudaMalloc(&d_proj, hidden * 4);
    cudaMalloc(&d_gate, intermediate * 4);
    cudaMalloc(&d_up, intermediate * 4);
    cudaMalloc(&d_mlp, intermediate * 4);
    cudaMalloc(&d_mlp_out, intermediate * 4);  // MLP FP32 + residual
    cudaMalloc(&d_attn_fp4, q_dim);
    cudaMalloc(&d_attn_s, (q_dim/16) * 4);
    cudaMalloc(&d_mlp_fp4, intermediate);
    cudaMalloc(&d_mlp_s, (intermediate/16) * 4);

    // Init x = uniform 1.0
    std::vector<float> x_init(hidden, 1.0f);
    cudaMemcpy(d_x32, x_init.data(), hidden * 4, cudaMemcpyHostToDevice);
    std::vector<float> xs_init(hidden/16, scale_1_3);
    cudaMemcpy(d_xs, xs_init.data(), (hidden/16) * 4, cudaMemcpyHostToDevice);
    blackwell::kernels::pack_fp4(d_x_fp4, d_x32, d_xs, hidden, 0);

    // Init scales
    std::vector<float> attn_s_init(q_dim/16, scale_1_3);
    cudaMemcpy(d_attn_s, attn_s_init.data(), (q_dim/16) * 4, cudaMemcpyHostToDevice);
    std::vector<float> mlp_s_init(intermediate/16, scale_1_3);
    cudaMemcpy(d_mlp_s, mlp_s_init.data(), (intermediate/16) * 4, cudaMemcpyHostToDevice);

    // RMSNorm weights (ones)
    float *d_rn_weight;
    cudaMalloc(&d_rn_weight, hidden * 4);
    std::vector<float> rn_weight(hidden, 1.0f);
    cudaMemcpy(d_rn_weight, rn_weight.data(), hidden * 4, cudaMemcpyHostToDevice);

    // KV cache
    float *d_kc, *d_vc;
    cudaMalloc(&d_kc, num_layers * num_kv_heads * max_seq * head_dim * 4);
    cudaMalloc(&d_vc, num_layers * num_kv_heads * max_seq * head_dim * 4);
    cudaMemset(d_kc, 0, num_layers * num_kv_heads * max_seq * head_dim * 4);
    cudaMemset(d_vc, 0, num_layers * num_kv_heads * max_seq * head_dim * 4);

    // Residual buffer: reuse d_mlp_out for MLP +resid
    // (attention residual uses d_x32 saved before layer)
    // Attention residual buffer: d_mlp_out also serves this
    // d_mlp_out reused across layers


    // Fill KV cache to seq=128
    printf("Filling KV cache to seq_pos=128...\n");
    for (int seq = 0; seq <= 128; ++seq) {
        for (int l = 0; l < num_layers; ++l) {
            int kv_base = l * num_kv_heads * max_seq * head_dim;
            // Attention
            // Separate GEMVs (fused_qkv limited to 256 outputs/block)
            blackwell::kernels::gemv_fp4_v2(d_Q, d_x_fp4, d_xs,
                lw[l].Wq.data, lw[l].Wq.scales, hidden, q_dim, 0);
            blackwell::kernels::gemv_fp4_v2(d_K, d_x_fp4, d_xs,
                lw[l].Wk.data, lw[l].Wk.scales, hidden, kv_dim, 0);
            blackwell::kernels::gemv_fp4_v2(d_V, d_x_fp4, d_xs,
                lw[l].Wv.data, lw[l].Wv.scales, hidden, kv_dim, 0);
            blackwell::kernels::update_kv_cache(
                d_kc + kv_base, d_vc + kv_base, d_K, d_V, 0, seq,
                num_kv_heads, head_dim, max_seq, 0);
            blackwell::kernels::attention_decode_gqa(
                d_attn, d_Q, d_kc + kv_base, d_vc + kv_base,
                seq, num_q_heads, num_kv_heads, head_dim, max_seq, 0);
            // Save x_fp32 residual before attention modifies it
            blackwell::kernels::unpack_fp4(d_mlp_out, d_x_fp4, d_xs, hidden, 0);
            blackwell::kernels::pack_fp4(d_attn_fp4, d_attn, d_attn_s, q_dim, 0);
            blackwell::kernels::gemv_fp4_v2(d_proj, d_attn_fp4, d_attn_s,
                lw[l].Wo.data, lw[l].Wo.scales, q_dim, hidden, 0);
            // Resid: proj += x_residual via FP32 intermediates
            // Resid: proj += x_residual (add FP32 values, then rmsnorm+pack)
            blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_mlp_out, hidden, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, d_proj, d_rn_weight, hidden, 1e-6f, 0);
            // Save attention output as residual for MLP
            blackwell::kernels::unpack_fp4(d_mlp_out, d_x_fp4, d_xs, hidden, 0);

            // MLP
            blackwell::kernels::fused_gate_up_gemv(
                d_gate, d_up,
                d_x_fp4, d_xs,
                lw[l].Wgate.data, lw[l].Wgate.scales,
                lw[l].Wup.data, lw[l].Wup.scales,
                hidden, intermediate, 0);
            blackwell::kernels::apply_swiglu(d_mlp, d_gate, d_up, intermediate, 0);
            blackwell::kernels::pack_fp4(d_mlp_fp4, d_mlp, d_mlp_s, intermediate, 0);
            // Zero output, then split-K for down_proj
            cudaMemsetAsync(d_proj, 0, hidden * 4, 0);
            blackwell::kernels::gemv_fp4_splitk(d_proj, d_mlp_fp4, d_mlp_s,
                lw[l].Wdown.data, lw[l].Wdown.scales, intermediate, hidden, 2, 0);
            // Resid: down_proj_out += attn_residual via FP32 intermediates
            // Resid: down_proj_out += attn_residual (both [hidden=2048])
            blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_mlp_out, hidden, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, d_proj, d_rn_weight, hidden, 1e-6f, 0);
        }
    }
    printf("  done.\n");

    cudaError_t cerr = cudaPeekAtLastError();
    if (cerr != cudaSuccess) { printf("  ERROR after cache fill: %s\n", cudaGetErrorString(cerr)); return 1; }

    // Warmup
    printf("Warmup...\n");
    int warm = 5, bench_iter = 20;
    int seq_pos = 128;
    for (int i = 0; i < warm; ++i) {
        for (int l = 0; l < num_layers; ++l) {
            int kv_base = l * num_kv_heads * max_seq * head_dim;
            // Attention
            // Save x_fp32 residual before attention modifies it (inter-layer)
            blackwell::kernels::unpack_fp4(d_mlp_out, d_x_fp4, d_xs, hidden, 0);
            // Separate GEMVs (fused_qkv limited to 256 outputs/block)
            blackwell::kernels::gemv_fp4_v2(d_Q, d_x_fp4, d_xs,
                lw[l].Wq.data, lw[l].Wq.scales, hidden, q_dim, 0);
            blackwell::kernels::gemv_fp4_v2(d_K, d_x_fp4, d_xs,
                lw[l].Wk.data, lw[l].Wk.scales, hidden, kv_dim, 0);
            blackwell::kernels::gemv_fp4_v2(d_V, d_x_fp4, d_xs,
                lw[l].Wv.data, lw[l].Wv.scales, hidden, kv_dim, 0);
            blackwell::kernels::update_kv_cache(
                d_kc + kv_base, d_vc + kv_base, d_K, d_V, 0, seq_pos,
                num_kv_heads, head_dim, max_seq, 0);
            blackwell::kernels::attention_decode_gqa(
                d_attn, d_Q, d_kc + kv_base, d_vc + kv_base,
                seq_pos, num_q_heads, num_kv_heads, head_dim, max_seq, 0);
            blackwell::kernels::pack_fp4(d_attn_fp4, d_attn, d_attn_s, q_dim, 0);
            blackwell::kernels::gemv_fp4_v2(d_proj, d_attn_fp4, d_attn_s,
                lw[l].Wo.data, lw[l].Wo.scales, q_dim, hidden, 0);
            // Resid: proj += x_residual via FP32 intermediates
            // Resid: proj += x_residual (add FP32 values, then rmsnorm+pack)
            blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_mlp_out, hidden, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, d_proj, d_rn_weight, hidden, 1e-6f, 0);
            // Save attention output as residual for MLP phase
            blackwell::kernels::unpack_fp4(d_mlp_out, d_x_fp4, d_xs, hidden, 0);
            // MLP
            blackwell::kernels::fused_gate_up_gemv(
                d_gate, d_up,
                d_x_fp4, d_xs,
                lw[l].Wgate.data, lw[l].Wgate.scales,
                lw[l].Wup.data, lw[l].Wup.scales,
                hidden, intermediate, 0);
            blackwell::kernels::apply_swiglu(d_mlp, d_gate, d_up, intermediate, 0);
            blackwell::kernels::pack_fp4(d_mlp_fp4, d_mlp, d_mlp_s, intermediate, 0);
            cudaMemsetAsync(d_proj, 0, hidden * 4, 0);
            blackwell::kernels::gemv_fp4_splitk(d_proj, d_mlp_fp4, d_mlp_s,
                lw[l].Wdown.data, lw[l].Wdown.scales, intermediate, hidden, 2, 0);
            blackwell::kernels::vector_add_fp32(d_proj, d_proj, d_mlp_out, hidden, 0);
            blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, d_proj, d_rn_weight, hidden, 1e-6f, 0);
        }
    }
    cudaDeviceSynchronize();

    KernelTimer timers[] = {
        {"gemv_fp4_v2 (Q)"}, {"gemv_fp4_v2 (K)"}, {"gemv_fp4_v2 (V)"},
        {"update_kv_cache"}, {"attention_decode_gqa"},
        {"pack_fp4 (attn)"}, {"gemv_fp4_v2 (Wo)"}, {"fused_rmsnorm_pack (attn)"},
        {"fused_gate_up_gemv"}, {"apply_swiglu"},
        {"pack_fp4 (mlp)"}, {"gemv_fp4_splitk (down_proj)"}, {"fused_rmsnorm_pack (mlp)"}
    };
    GpuTimer kt[14];
#define TIME_KERNEL(idx, call) do { kt[idx].begin(); call; timers[idx].add(kt[idx].end()); } while(0)

    // Benchmark
    printf("Benchmarking %d tokens (seq_pos=%d)...\n", bench_iter, seq_pos);
    GpuTimer t;
    t.begin();
    for (int iter = 0; iter < bench_iter; ++iter) {
        for (int l = 0; l < num_layers; ++l) {
            int kv_base = l * num_kv_heads * max_seq * head_dim;
            // Save x_fp32 residual before attention modifies it
            blackwell::kernels::unpack_fp4(d_mlp_out, d_x_fp4, d_xs, hidden, 0);
            // Attention
            TIME_KERNEL(0, blackwell::kernels::gemv_fp4_v2(d_Q, d_x_fp4, d_xs, lw[l].Wq.data, lw[l].Wq.scales, hidden, q_dim, 0));
            TIME_KERNEL(1, blackwell::kernels::gemv_fp4_v2(d_K, d_x_fp4, d_xs, lw[l].Wk.data, lw[l].Wk.scales, hidden, kv_dim, 0));
            TIME_KERNEL(2, blackwell::kernels::gemv_fp4_v2(d_V, d_x_fp4, d_xs, lw[l].Wv.data, lw[l].Wv.scales, hidden, kv_dim, 0));
            TIME_KERNEL(3, blackwell::kernels::update_kv_cache(d_kc + kv_base, d_vc + kv_base, d_K, d_V, 0, seq_pos, num_kv_heads, head_dim, max_seq, 0));
            TIME_KERNEL(4, blackwell::kernels::attention_decode_gqa(d_attn, d_Q, d_kc + kv_base, d_vc + kv_base, seq_pos, num_q_heads, num_kv_heads, head_dim, max_seq, 0));
            TIME_KERNEL(5, blackwell::kernels::pack_fp4(d_attn_fp4, d_attn, d_attn_s, q_dim, 0));
            TIME_KERNEL(6, blackwell::kernels::gemv_fp4_v2(d_proj, d_attn_fp4, d_attn_s, lw[l].Wo.data, lw[l].Wo.scales, q_dim, hidden, 0));
            // Resid: proj += x_residual
            blackwell::kernels::fused_rmsnorm(d_proj, d_proj, d_rn_weight, hidden, 1e-6f, 0);
            blackwell::kernels::vector_add_fp32(d_x32, d_proj, d_mlp_out, hidden, 0);
            TIME_KERNEL(7, blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, d_x32, d_rn_weight, hidden, 1e-6f, 0));
            // Save attn output as MLP residual
            blackwell::kernels::unpack_fp4(d_mlp_out, d_x_fp4, d_xs, hidden, 0);
            // MLP
#ifdef USE_UNFUSED_GATE_UP
            TIME_KERNEL(8, blackwell::kernels::gemv_fp4_v2(d_gate, d_x_fp4, d_xs, lw[l].Wgate.data, lw[l].Wgate.scales, hidden, intermediate, 0));
            TIME_KERNEL(9, blackwell::kernels::gemv_fp4_v2(d_up, d_x_fp4, d_xs, lw[l].Wup.data, lw[l].Wup.scales, hidden, intermediate, 0));
            blackwell::kernels::apply_swiglu(d_mlp, d_gate, d_up, intermediate, 0);
#else
            TIME_KERNEL(8, blackwell::kernels::fused_gate_up_gemv(d_gate, d_up, d_x_fp4, d_xs, lw[l].Wgate.data, lw[l].Wgate.scales, lw[l].Wup.data, lw[l].Wup.scales, hidden, intermediate, 0));
            TIME_KERNEL(9, blackwell::kernels::apply_swiglu(d_mlp, d_gate, d_up, intermediate, 0));
#endif
            TIME_KERNEL(9, blackwell::kernels::apply_swiglu(d_mlp, d_gate, d_up, intermediate, 0));
            TIME_KERNEL(10, blackwell::kernels::pack_fp4(d_mlp_fp4, d_mlp, d_mlp_s, intermediate, 0));
            // Zero output for atomic split-K partial sums
            cudaMemsetAsync(d_proj, 0, hidden * 4, 0);
            TIME_KERNEL(11, blackwell::kernels::gemv_fp4_splitk(d_proj, d_mlp_fp4, d_mlp_s, lw[l].Wdown.data, lw[l].Wdown.scales, intermediate, hidden, 4, 0));
            // Resid: down_proj_out += attn_residual
            blackwell::kernels::fused_rmsnorm(d_proj, d_proj, d_rn_weight, hidden, 1e-6f, 0);
            blackwell::kernels::vector_add_fp32(d_mlp_out, d_proj, d_mlp_out, intermediate, 0);
            TIME_KERNEL(12, blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, d_mlp_out, d_rn_weight, hidden, 1e-6f, 0));
        }
    }
    float total_ms = t.end();
    float per_token_ms = total_ms / bench_iter;
    float tps = 1000.0f / per_token_ms;
    float scaled_28 = 1000.0f / (per_token_ms * 28.0f / num_layers);

    printf("\n=== Results ===\n");
    printf("  Layers:            %d\n", num_layers);
    printf("  Tokens:            %d\n", bench_iter);
    printf("  Total time:        %.2f ms\n", total_ms);
    printf("  Per-token:         %.3f ms\n", per_token_ms);
    printf("  Throughput:        %.1f t/s\n", tps);
    printf("  Scaled to 28:      %.1f t/s\n", scaled_28);
    printf("  Target (llama):    114 t/s\n");
    printf("  Ratio:             %.1f%%\n", (scaled_28 / 114.0f) * 100.0f);

    printf("\n=== Per-Kernel Timing (%d layers × %d tokens) ===\n", num_layers, bench_iter);
    double kernel_total = 0;
    for (int i = 0; i < 13; ++i) { timers[i].print(); kernel_total += timers[i].total_ms; }
    printf("  %-30s  %7.3f ms (total)\n", "TOTAL KERNEL", kernel_total);
    printf("\n  Per-layer breakdown estimate:\n");
    printf("    Attention: gemv(Q,K,V) + update_kv + attn + pack + gemv(Wo) + rmsnorm_pack = 8 ops\n");
    printf("    MLP:       gate_up + swiglu + pack + gemv(down) + rmsnorm_pack = 5 ops\n");
    printf("    Total:     13 ops/layer × %d layers = %d ops/token\n", num_layers, num_layers * 13);

    // =========================================================================
    // CUDA Graph benchmark — capture full decode loop, benchmark via graphLaunch
    // =========================================================================
    cudaDeviceSynchronize();  // CRITICAL: clear stream=0 error state
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    cudaStream_t graph_stream;
    cudaStreamCreate(&graph_stream);

    // Pre-trigger attention_decode_gqa on graph_stream (sets smem attr)
    blackwell::kernels::attention_decode_gqa(
        d_attn, d_Q, d_kc + 0, d_vc + 0,
        seq_pos, num_q_heads, num_kv_heads, head_dim, max_seq, graph_stream);
    cudaStreamSynchronize(graph_stream);

    printf("\n=== CUDA Graph Benchmark ===\n");
    printf("  Capturing full decode (%d layers)... ", num_layers);
    fflush(stdout);
    cudaStreamBeginCapture(graph_stream, cudaStreamCaptureModeGlobal);
    for (int l = 0; l < num_layers; ++l) {
        int kv_base = l * num_kv_heads * max_seq * head_dim;
        blackwell::kernels::unpack_fp4(d_mlp_out, d_x_fp4, d_xs, hidden, graph_stream);
        blackwell::kernels::gemv_fp4_v2(d_Q, d_x_fp4, d_xs, lw[l].Wq.data, lw[l].Wq.scales, hidden, q_dim, graph_stream);
        blackwell::kernels::gemv_fp4_v2(d_K, d_x_fp4, d_xs, lw[l].Wk.data, lw[l].Wk.scales, hidden, kv_dim, graph_stream);
        blackwell::kernels::gemv_fp4_v2(d_V, d_x_fp4, d_xs, lw[l].Wv.data, lw[l].Wv.scales, hidden, kv_dim, graph_stream);
        blackwell::kernels::update_kv_cache(d_kc + kv_base, d_vc + kv_base, d_K, d_V, 0, seq_pos, num_kv_heads, head_dim, max_seq, graph_stream);
        blackwell::kernels::attention_decode_gqa(d_attn, d_Q, d_kc + kv_base, d_vc + kv_base, seq_pos, num_q_heads, num_kv_heads, head_dim, max_seq, graph_stream);
        blackwell::kernels::pack_fp4(d_attn_fp4, d_attn, d_attn_s, q_dim, graph_stream);
        blackwell::kernels::gemv_fp4_v2(d_proj, d_attn_fp4, d_attn_s, lw[l].Wo.data, lw[l].Wo.scales, q_dim, hidden, graph_stream);
        blackwell::kernels::fused_rmsnorm(d_proj, d_proj, d_rn_weight, hidden, 1e-6f, graph_stream);
        blackwell::kernels::vector_add_fp32(d_x32, d_proj, d_mlp_out, hidden, graph_stream);
        blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, d_x32, d_rn_weight, hidden, 1e-6f, graph_stream);
        blackwell::kernels::unpack_fp4(d_mlp_out, d_x_fp4, d_xs, hidden, graph_stream);
        blackwell::kernels::fused_gate_up_gemv(d_gate, d_up, d_x_fp4, d_xs, lw[l].Wgate.data, lw[l].Wgate.scales, lw[l].Wup.data, lw[l].Wup.scales, hidden, intermediate, graph_stream);
        blackwell::kernels::apply_swiglu(d_mlp, d_gate, d_up, intermediate, graph_stream);
        blackwell::kernels::pack_fp4(d_mlp_fp4, d_mlp, d_mlp_s, intermediate, graph_stream);
        cudaMemsetAsync(d_proj, 0, hidden * 4, graph_stream);
        blackwell::kernels::gemv_fp4_splitk(d_proj, d_mlp_fp4, d_mlp_s, lw[l].Wdown.data, lw[l].Wdown.scales, intermediate, hidden, 4, graph_stream);
        blackwell::kernels::fused_rmsnorm(d_proj, d_proj, d_rn_weight, hidden, 1e-6f, graph_stream);
        blackwell::kernels::vector_add_fp32(d_mlp_out, d_proj, d_mlp_out, intermediate, graph_stream);
        blackwell::kernels::fused_rmsnorm_pack(d_x_fp4, d_xs, d_mlp_out, d_rn_weight, hidden, 1e-6f, graph_stream);
    }
    cudaError_t cap_err = cudaStreamEndCapture(graph_stream, &graph);
    if (cap_err != cudaSuccess) {
        printf("FAIL: %s\n", cudaGetErrorString(cap_err));
        cudaStreamDestroy(graph_stream);
    } else {
        cap_err = cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
        if (cap_err != cudaSuccess) {
            printf("FAIL: cudaGraphInstantiate: %s\n", cudaGetErrorString(cap_err));
            cudaStreamDestroy(graph_stream);
        } else {
            printf("done\n");
            GpuTimer tg;
            tg.begin();
            for (int i = 0; i < bench_iter; ++i) cudaGraphLaunch(graph_exec, graph_stream);
            cudaStreamSynchronize(graph_stream);
            float graph_ms = tg.end();
            float graph_per_token = graph_ms / bench_iter;
            float graph_tps = 1000.0f / graph_per_token;
            printf("  Graph per-token:      %.3f ms\n", graph_per_token);
            printf("  Graph throughput:     %.1f t/s\n", graph_tps);
            printf("  Graph scaled 28:     %.1f t/s\n", 1000.0f / (graph_per_token * 28.0f / num_layers));
            printf("  vs per-kernel:       %.1f%%\n", (graph_tps / tps) * 100.0f);
            cudaGraphExecDestroy(graph_exec);
            cudaGraphDestroy(graph);
            cudaStreamDestroy(graph_stream);
        }
    }

    // Print output values
    float* d_tmp;
    cudaMalloc(&d_tmp, hidden * 4);
    blackwell::kernels::unpack_fp4(d_tmp, d_x_fp4, d_xs, hidden, 0);
    std::vector<float> out(hidden);
    cudaMemcpy(out.data(), d_tmp, hidden * 4, cudaMemcpyDeviceToHost);
    printf("\n  First 8 x values: ");
    for (int i = 0; i < 8; ++i) printf("%.4f ", out[i]);
    printf("\n");
    // Verify outputs show residual connection effect (not all same)
    float sum_abs = 0;
    for (int i = 0; i < hidden; ++i) sum_abs += fabsf(out[i]);
    printf("  L1 norm: %.2f  (non-zero = residual working)\n", sum_abs / hidden);
    cudaFree(d_tmp);

    // Cleanup
    for (int l = 0; l < num_layers; ++l) {
        cudaFree(lw[l].Wq.data); cudaFree(lw[l].Wq.scales);
        cudaFree(lw[l].Wk.data); cudaFree(lw[l].Wk.scales);
        cudaFree(lw[l].Wv.data); cudaFree(lw[l].Wv.scales);
        cudaFree(lw[l].Wo.data); cudaFree(lw[l].Wo.scales);
        cudaFree(lw[l].Wgate.data); cudaFree(lw[l].Wgate.scales);
        cudaFree(lw[l].Wup.data); cudaFree(lw[l].Wup.scales);
        cudaFree(lw[l].Wdown.data); cudaFree(lw[l].Wdown.scales);
    }
    cudaFree(d_x32); cudaFree(d_x_fp4); cudaFree(d_xs);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_attn); cudaFree(d_proj); cudaFree(d_gate); cudaFree(d_up);
    cudaFree(d_mlp); cudaFree(d_mlp_out);
    cudaFree(d_attn_fp4); cudaFree(d_attn_s);
    cudaFree(d_mlp_fp4); cudaFree(d_mlp_s);
    cudaFree(d_rn_weight);
    cudaFree(d_kc); cudaFree(d_vc);

    return 0;
}
