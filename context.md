# Code Context — blackwell Repo Inspection

## Files Retrieved

1. `build/libblackwell_kernels.a` (754,912 bytes, May 26 20:15) — static lib
2. `HANDOFF.md` — throughput 40.6 t/s, 26 symbols, all optimization flags
3. `src/kernels/gemv_v2.cu` (6,110 bytes, May 26 19:47) — GEMV v2 kernel
4. `src/kernels/gemm.cu` (16,715 bytes, May 26 16:23) — GEMV v1 kernel
5. `src/kernels/fused_decode.cu` (3,163 bytes, May 26 19:50) — fused_qkv
6. `bench/decode_full` (1,637,984 bytes, May 26 20:15) — benchmark binary

## Verified Facts

| Check | Result | Detail |
|-------|--------|--------|
| Build exists | ✅ | 754,912 bytes, May 26 20:15 |
| Symbol count | ✅ 26 public, 43 total | 26 outside anonymous NS (matches HANDOFF.md exactly) |
| GEMV v2 dynamic K | ✅ **NOT hardcoded** | `gemv_fp4_v2` takes `int K` param, loops `num_K_blks = K/16` |
| GEMV v1 dynamic K | ✅ **NOT hardcoded** | `gemv_fp4_kernel` takes `int K, int N`, loops `for (int k = 0; k < K; ++k)` |
| fused_qkv multi-block | ✅ | Grid(3, tiles), tiles = ceil(max(q_dim, kv_dim)/256), comment confirms multi-block |
| Throughput (HANDOFF.md) | 40.6 t/s | 28-layer estimate, 35.6% of llama.cpp 114 t/s target |
| Modified .cu since build | ✅ None | git status: only HANDOFF.md modified, no .cu files |
| bench/decode_full binary | ✅ Exists | 1.6 MB, May 26 20:15 |

## Symbol Table (26 Public API)

All `blackwell::kernels::` top-level, no anonymous namespace:

```
dispatch_matmul, gemm_fp4_block_scaled, gemv_fp4, gemv_fp4_v2,
transpose_fp4_weights, fused_gate_up_gemv, fused_rmsnorm_pack,
fused_o_norm_pack, fused_rmsnorm, fused_rope, apply_swiglu,
pack_fp4, unpack_fp4, coalesced_copy, shared_copy_async,
async_pipeline_stage, attention_fp4, run_prefill_layer,
attention_decode, attention_decode_gqa, load_kv_cache_qkgv,
update_kv_cache, fused_qkv_gemv, launch_decode_graph,
capture_decode_graph, destroy_decode_graph
```

17 internal symbols in anonymous namespace (not counted in HANDOFF.md 26).

## Key Code

**GEMV v2 dynamic K** — `src/kernels/gemv_v2.cu:37-84`:
```cpp
__global__ void gemv_fp4_v2_kernel(..., int K, int N) {
    int num_K_blks = K / B;  // B=16
    for (int kb = 0; kb < num_K_blks; ++kb) {
        // uint4 load from W_t[n_out*K + kb*B]
        // Sequential K access — full cache line utilization
    }
}
```

**fused_qkv multi-block** — `src/kernels/fused_decode.cu:84-88`:
```cpp
int max_dim = (q_dim > kv_dim) ? q_dim : kv_dim;
int tiles = (max_dim + kFuseBlockThreads - 1) / kFuseBlockThreads;
dim3 grid(3, tiles);  // blockIdx.x: 0=Q,1=K,2=V; blockIdx.y: output tile
```

## Discrepancies

| Item | AGENTS.md says | Actual | Impact |
|------|---------------|--------|--------|
| GEMV "K=64 hardcoded" | TODO #2: dynamic K needed | K is fully dynamic in both v1 and v2 | **Stale TODO** — already done |
| HANDOFF.md modified | not checked | `git diff HEAD -- HANDO` only | Unrelated to code state |

AGENTS.md `TODO #2: dynamic K` flagged as pending, but `gemv_fp4_v2` accepts `int K` at runtime. Real-world call in `decode_full.cu` uses `hidden=2048`, `intermediate=6144` — no K=64 restriction.

## Start Here

Open `src/kernels/gemv_v2.cu` for the active decode GEMV path. All decode_full calls route through `gemv_fp4_v2`. For fused QKV path, open `src/kernels/fused_decode.cu` — note it's not used by decode_full (uses separate gemv_fp4_v2 calls instead, comment says "fused_qkv limited to 256 outputs/block").

## Architecture

```
decode_full.cu
  ├── gemv_fp4_v2 (Q,K,V,W_proj,Wo) — separate calls per weight matrix
  ├── fused_gate_up_gemv (W_gate, W_up) — one kernel for both
  ├── fused_rmsnorm_pack — rmsnorm + pack in one kernel
  ├── attention_decode_gqa — GQA attention
  └── update_kv_cache — KV cache update

gemv_fp4_v2:
  - Input: transposed weights W_t [N×K]
  - uint4 vector load across K (16 FP4/txn)
  - smem broadcast of x values
  - 256 threads/block, N/256 blocks
```
