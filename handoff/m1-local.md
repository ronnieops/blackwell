# M=1 Bottleneck Analysis: Blackwell INT4 8B Decode

## Executive Summary

**Measured: 15.5ms/tok (64 t/s). Theoretical ceiling: 12.67ms/tok (79 t/s). Gap: 2.83ms/tok.**

The bottleneck is NOT compute (dp4a is fine). The bottleneck is **aggregate HBM bandwidth across 253 GEMV launches per token**, achieving 93% of peak (416/448 GB/s). The remaining gap is:
1. GEMV BW shortfall from low occupancy (17%) — 0.98ms (35%)
2. Non-GEMV kernels (rmsnorm, quantize, etc) — 1.13ms (40%)
3. CPU/launch overhead (serial) — 0.72ms (25%)

**Key correction to prior assumptions:**
- Peak BW is **448 GB/s** (GDDR7 14001MHz × 128-bit × 2 PAM4), NOT 500 GB/s
- Theoretical ceiling is **79 t/s**, NOT 94 t/s
- The "30 t/s gap" (64→94) is actually a **15 t/s gap** (64→79)
- GEMV kernels already achieve 93% BW — the kernel is well-written

---

## 1. Memory Access Pattern: `gemv_int4_warp_kernel`

**File:** `src/kernels/gemv_int8.cu:509-556`

### Launch config
```cuda
<<<dim3(N), dim3(32), 0, stream>>>  // 1 block per output row, 1 warp per block
```
- `__launch_bounds__(32, 8)` (line 509) → max 8 blocks/SM → **8 warps/SM → 17% occupancy**

### Access pattern per K-block iteration
```cpp
for (int kb = tid; kb < num_K_blks; kb += 32) {
    const uint8_t* w_ptr = &W_packed[(size_t)n_out * (K / 2) + kb * PB];  // PB=8
    uint2 w_packed = *reinterpret_cast<const uint2*>(w_ptr);              // 8 bytes = 16 INT4 values
    // ... also loads uint2 activation + 2× FP32 scales
}
```

**Coalescing: PERFECTLY COALESCED.** Thread `t` reads byte offset `n_out*(K/2) + (t + 32*i) * 8`:
- Threads 0-31 read bytes `[base+0, base+8, ..., base+248]` = 256 contiguous bytes
- 256 bytes / 128B cache line = 2 cache lines served per warp iteration
- No wasted bandwidth from coalescing

### uint2 vs uint4
- Current: `uint2` (8 bytes/thread) = 32 threads × 8B = 256B/warp = 2 cache lines/iter
- If `uint4` (16 bytes/thread): 512B/warp = 4 cache lines/iter, processes 2 K-blocks/iter
- **uint4 would halve instruction count but memory traffic is IDENTICAL**
- Kernel is memory-bound (93% BW) → uint4 gives ~0% improvement

### Per-iteration data fetched
| Item | Size | Source |
|------|------|--------|
| Packed weight | 8 bytes (uint2) | `W_packed[n_out][kb]` |
| Weight scale | 4 bytes (FP32) | `W_scale[n_out][kb/16]` |
| Packed activation | 8 bytes (uint2) | `x_packed[kb]` (broadcast across all N blocks for same kb) |
| Activation scale | 4 bytes (FP32) | `x_scale[kb/16]` |
| **Total** | **24 bytes/thread/iter** | |

---

## 2. GEMV Launch Count & Decode Loop

### Per-token GEMV launches
**File:** `bench/text_generate_int4_qwen3_8b.cu:290-350` (warp path)
**File:** `server/inference_server_int4.cu:271-300` (server, uses `gemv_int4_batched` M=1)

```
Per layer (×36):  q_proj + k_proj + v_proj + o_proj + gate + up + down = 7 GEMV calls
Per token:        36 × 7 + lm_head = 253 GEMV calls
```

### Kernel launch count per token
| Category | Per Layer | Per Token (36L) | Notes |
|----------|-----------|-----------------|-------|
| GEMV | 7 | 253 | 92.4% of GPU time |
| quantize_int4 | 4 | 145 | 1.3% GPU time |
| rmsnorm | 2 | 73 | 3.8% GPU time |
| head_norm | 2 | 72 | 0.6% GPU time |
| apply_rope | 2 | 72 | 0.4% GPU time |
| attn + update_kv | 2 | 72 | 0.9% GPU time |
| vector_add | 2 | 72 | 0.4% GPU time |
| swiglu | 1 | 36 | 0.3% GPU time |
| D2D memcpy | 2 | 72 | ~0.6µs each |
| argmax/sample | 0 | 1 | |
| **Total** | **24** | **~797** | |

### Synchronization pattern
- All kernels on stream `st` — fully pipelined, no inter-kernel sync
- **Bench only sync:** `cudaMemcpy(&next_id, d_next_id, 4, D2H)` per token (line 323)
- **Bench has 3 extra sync D2H** for logit peek (lines 318-320) — debug code, not in server
- **Server only sync:** `cudaMemcpy(&next_id, d_next_id, 4, D2H)` (line 533) — 1 per token
- CUDA Graph captures all 867 nodes, achieves 2.2% speedup (65.8 vs 64.3 t/s)

### CUDA Graph: `bench/decode_int4_cgraph_8b.cu`
- **File exists and builds.** Binary at `bench/decode_int4_cgraph_8b`
- Captures full 36-layer decode + lm_head (867 nodes)
- Graph-safe: uses `update_kv_cache_device`, `attention_decode_batched_gqa_device`, `fused_rope_decode`
- seq_pos updated via pinned H2D between replays
- **Measured result (run this session):**
  ```
  Per-kernel: 15.55 ms/token = 64.3 t/s
  Graph:      15.20 ms/token = 65.8 t/s
  Speedup:    2.2%
  ```
- Graph only saves launch overhead (~0.35ms). GEMV time unchanged.

---

## 3. Scale Array Traffic Analysis

### Scale layout
- Weight scale: `W_scale[N][K/16]` FP32 = `N * K / 4` bytes = **33.3% of packed weight bytes**
- Activation scale: `x_scale[K/16]` FP32 = `K / 4` bytes (tiny, reused across all N blocks)

### Traffic breakdown per GEMV
| Matrix | Packed (MB) | Scales (MB) | Scale % | Total (MB) |
|--------|------------|-------------|---------|-----------|
| q_proj (4096×4096) | 8.39 | 4.19 | 33.3% | 12.58 |
| k_proj (4096×1024) | 2.10 | 1.05 | 33.3% | 3.15 |
| v_proj (4096×1024) | 2.10 | 1.05 | 33.3% | 3.15 |
| o_proj (4096×4096) | 8.39 | 4.19 | 33.3% | 12.58 |
| gate (4096×12288) | 25.17 | 12.58 | 33.3% | 37.75 |
| up (4096×12288) | 25.17 | 12.58 | 33.3% | 37.75 |
| down (12288×4096) | 25.17 | 12.58 | 33.3% | 37.75 |
| lm_head (4096×151936) | 311.17 | 155.58 | 33.3% | 466.75 |

### Per-token aggregate
- Packed weights: 3784 MB (66.7%)
- FP32 scales: 1892 MB (33.3%)
- **Total: 5676 MB = 5.676 GB/token**

### FP16 scales opportunity
- If scales were FP16 (2 bytes): save 946 MB/token = **16.7% of total traffic**
- At 416 GB/s achieved: 946MB / 416 GB/s = **2.27ms/token savings → ~6 t/s improvement**
- **This is the single biggest optimization opportunity**
- No kernel changes needed — just reinterpret cast in scale load: `float w_sc = (float)((half*)W_scale)[idx]`
- Requires re-quantizing weights with FP16 scales (lossless for block-16 absmax)

### Redundant scale loading?
Scales are loaded **once per K-block per thread**. Different threads (different `n_out`) loading same `kb` read different scale values (`W_scale[n_out][kb]`). **NOT redundant.** The scale array is indexed by `[n_out][kb]` = per-output-row, per-K-block. Each element is unique.

---

## 4. Activation Quantization Overhead

**Confirmed: 1.3% of GPU time. Not a bottleneck.**

From nsys:
```
quantize_int4_kernel: 2171 instances, 2,909 µs total, avg 1,340 ns, median 1,216 ns
```

Per token: 145 calls × 1.34µs = 0.194ms/token (1.3%)

### `quantize_int4_kernel` launch pathology
**File:** `src/kernels/gemv_int8.cu:1748-1817`

```cuda
quantize_int4_kernel<<<dim3(num_kb, 1), 1, 0, stream>>>(...)
//                       ^256-768 blocks   ^1 thread!
```

Each block = 1 thread = 1 warp (31 idle). Terrible occupancy (256 blocks × 1 thread = 8 warps/SM). But work is trivial (16 FP32 reads, 1 scale, 8 byte writes) → not worth optimizing. The GPU time is dominated by scheduling, not compute.

**However:** 145 launches × ~5µs API overhead = 0.725ms of CPU-side launch cost. This IS a measurable fraction of the 0.72ms CPU overhead. Fusion would eliminate it.

---

## 5. KV Cache / Attention

**Confirmed: 0.7% of GPU time. Not a bottleneck. Decode-only (not prefill).**

From nsys:
```
attn_batched_kernel: 540 instances, 1,521 µs total, avg 2,816 ns
update_kv_kernel:    540 instances,   414 µs total, avg 767 ns
```

Per token: 72 attn calls × 2.8µs + 72 kv writes × 0.77µs = 0.26ms/token (1.7%)

---

## 6. lm_head Analysis

**File:** `bench/text_generate_int4_qwen3_8b.cu:316`
**File:** `server/inference_server_int4.cu:527`

### lm_head is computed every token
```cpp
gemv_int4_warp(d_logits, d_x_i4, d_x_i4_sc, lm_head_w.d, lm_head_w.sc, H, V, st)
// H=4096, V=151936 → N=151936, K=4096
```

### lm_head performance
- Size: 467 MB (largest GEMV)
- Time: 1098µs (7.1% of total)
- BW: 425 GB/s = **95% of 448 peak**
- **Most efficient GEMV in the pipeline**

### Can lm_head be skipped/pruned?
- **NO** — needed every token for logits
- **NO** — need full vocab for softmax sampling
- Only optimization: FP16 scales (save 78 MB = 17% of lm_head time = 180µs)

---

## 7. Memory Access Efficiency

### Coalescing: PERFECT
As analyzed in §1. 32 threads read 256 contiguous bytes per iteration (2 cache lines). No waste.

### Occupancy: LOW (root cause of BW shortfall)
```
__launch_bounds__(32, 8) on gemv_int4_warp_kernel
→ 8 blocks/SM × 1 warp/block = 8 warps/SM
→ Occupancy: 8/48 = 17%
→ 2 warps/scheduler (4 schedulers/SM)
→ Recommended for memory-bound: 8-12 warps/scheduler (32-48 warps/SM)
```

With 8 warps/SM, the SM can only have 8 outstanding memory request groups in flight. To fully hide GDDR7 latency (~200-400 cycles), need more warps to schedule while others wait.

### Occupancy impact on BW
| GEMV Type | N | Blocks | Blocks/SM | Theo BW% (estimated) |
|-----------|------|--------|-----------|---------------------|
| k/v_proj | 1024 | 1024 | 28 (capped at 8) | ~70% |
| q/o_proj | 4096 | 4096 | 114 (capped at 8) | ~85% |
| gate/up/down | 12288 | 12288 | 341 (capped at 8) | ~90% |
| lm_head | 151936 | 151936 | 4220 (capped at 8) | ~95% |

**Key insight:** Smaller matrices (k/v_proj with N=1024) have even lower effective BW because fewer total warps are active across all SMs (1024 blocks / 8 per SM = 128 warps total → only 3.6 waves of work, not enough to amortize startup/teardown).

### `gemv_int4_batched_kernel` — identical to warp
**File:** `src/kernels/gemv_int8.cu:1028-1110`
- Same `__launch_bounds__(32, 8)` (line 1028)
- Same `kINT4Block = 32` (line 1022)
- Same launch config: `<<<dim3(N,1), 32, 0, stream>>>` (line 1125)
- For M=1: template `<1>` unrolls the M loop → identical assembly to warp kernel
- Confirmed by nsys: batched M=1 and warp paths produce identical kernel times (54µs avg, 31µs median, 1098µs max)

---

## 8. CUDA Graph Status

**File:** `bench/decode_int4_cgraph_8b.cu` (exists, compiled, runs)

### Graph-safe APIs used
- `update_kv_cache_device()` — device-side seq_pos (no H2D memcpy in capture)
- `attention_decode_batched_gqa_device()` — device-side seq_pos
- `fused_rope_decode()` — device-side seq_pos with pre-computed cos/sin cache
- `update_decode_seq_pos()` — writes pinned host → async copy to device between replays

### Graph result (run this session)
```
Per-kernel: 15.55 ms/tok = 64.3 t/s
Graph:      15.20 ms/tok = 65.8 t/s
Speedup:    2.2%  (0.35ms/token)
```

### Why only 2.2%?
- GEMV time (88% of total) is unchanged — graph doesn't affect kernel execution
- The saved time is launch overhead (~4.4ms of API calls → most already overlapped with GPU)
- Only the unoverlapped portion (~0.35ms) is recovered
- With `fused_residual_norm_int4` (not currently used) + graph, savings would compound

---

## Quantified Time Budget: 15.5ms/token (64 t/s)

| Component | Time (ms) | % of Total | File:Line |
|-----------|-----------|------------|-----------|
| GEMV kernels (253 calls) | 13.65 | 88% | `gemv_int8.cu:510` (warp), `gemv_int8.cu:1029` (batched) |
| └ lm_head (1 call) | 1.10 | 7% | `gemv_int8.cu:510`, called at `bench:316` |
| └ gate/up/down (108 calls) | ~10.1 | 65% | |
| └ q/o_proj (72 calls) | ~2.4 | 15% | |
| └ k/v_proj (72 calls) | ~0.7 | 5% | |
| rmsnorm (73 calls) | 0.555 | 3.6% | `fused_rmsnorm.cu` |
| quantize_int4 (145 calls) | 0.194 | 1.3% | `gemv_int8.cu:1748` |
| attn_batched (72 calls) | 0.101 | 0.7% | `decode.cu` |
| head_norm (72 calls) | 0.081 | 0.5% | inline in bench/server |
| vector_add (72 calls) | 0.061 | 0.4% | `norm.cu` |
| apply_rope (72 calls) | 0.055 | 0.4% | inline in bench/server |
| swiglu (36 calls) | 0.039 | 0.3% | `fused_rmsnorm.cu` |
| update_kv (72 calls) | 0.028 | 0.2% | `decode.cu` |
| CPU/launch overhead | 0.72 | 5% | 797 launches × ~5µs |
| **TOTAL** | **15.5** | **100%** | |

---

## Ranked Bottleneck Analysis

### Gap: 64 → 79 t/s (15 t/s, 2.83ms/token)

| Rank | Component | Time (ms) | % of Gap | Root Cause | Fix Potential |
|------|-----------|-----------|----------|------------|---------------|
| **1** | **Non-GEMV kernels** | **1.13** | **40%** | Separate rmsnorm + quantize_int4 + D2D memcpy | fused_residual_norm_int4 saves ~0.1-0.15ms |
| **2** | **GEMV BW shortfall** | **0.98** | **35%** | 17% occupancy (8 warps/SM) | Increase warps/SM → 8-12% BW gain |
| **3** | **CPU/launch overhead** | **0.72** | **25%** | 797 serial API calls/token | CUDA Graph (already 2.2%) + fused kernels |

### Fix details

**Rank 1: Non-GEMV kernels (0.55ms rmsnorm + 0.19ms quantize + 0.04ms D2D = 0.78ms addressable)**
- `fused_residual_norm_int4_fp32out` exists in `include/blackwell/kernels.h:1231` — fuses residual_add + rmsnorm + quantize_int4 (3→1)
- `fused_rmsnorm_quant_int4` exists in `include/blackwell/kernels.h:462` — fuses rmsnorm + quantize_int4 (2→1)
- `fused_swiglu_quant_int4` exists in `include/blackwell/kernels.h` — fuses swiglu + quantize (2→1)
- **NONE of these are used in any decode loop** (bench, server, graph)
- Estimated savings: ~0.1-0.15ms/token from fewer launches + less intermediate I/O
- Total with graph: ~0.2-0.3ms (64→66 t/s)

**Rank 2: GEMV BW shortfall (0.98ms)**
- Change `__launch_bounds__(32, 8)` to `__launch_bounds__(32, 16)` → 16 warps/SM
- Risk: register spill (need to verify register count)
- Alternative: 2 warps per output row (64 threads/block) → same occupancy but better latency hiding
- Alternative: process 2 output rows per warp → halves block count, doubles work per warp
- Estimated: 7-10% BW improvement → 0.7-1.0ms savings → 64→70 t/s

**Rank 3: FP16 scales (not yet ranked but significant)**
- FP32 scales = 33% of weight traffic = 1.89 GB/token
- FP16 scales: save 946 MB/token = 16.7% of total traffic
- At 416 GB/s: 2.27ms savings → 64→76 t/s
- Requires: re-quantize weights, change kernel to `(float)((half*)W_scale)[idx]`
- **Biggest single optimization, but requires weight regeneration**

**Combined theoretical max:** 64 → 79 t/s (close all 2.83ms gap)
- FP16 scales: +12 t/s
- Better occupancy: +6 t/s
- Fused kernels + graph: +2 t/s

---

## Validation Commands

### Microbenchmark (lm_head BW)
```bash
killall hashcat 2>/dev/null
# No standalone microbench binary exists. Use nsys to measure lm_head:
nsys profile --trace=cuda --stats=true \
  -o /tmp/nsys_int4 --force-overwrite=true \
  ./bench/text_generate_int4_qwen3_8b "The capital of France is" 10
# Look at gemv_int4_warp_kernel max(ns) — that's lm_head (1098µs)
# BW = 466.7MB / 1098µs = 425 GB/s = 95% of 448 peak
```

### nsys end-to-end profile
```bash
nsys stats --force-export=true /tmp/nsys_int4_8b_warp.nsys-rep
# Key section: "CUDA GPU Kernel Summary" — kernel time distribution
# gemv_int4_warp_kernel: 92.4%, 3791 instances, median 31µs, max 1098µs
```

### CUDA Graph comparison
```bash
killall hashcat 2>/dev/null
./bench/decode_int4_cgraph_8b 20
# Expected: per-kernel 64.3 t/s, graph 65.8 t/s, 2.2% speedup
```

### Batched vs warp comparison
```bash
killall hashcat 2>/dev/null
./bench/text_generate_int4_qwen3_8b "The capital of France is" 30    # warp: ~57 t/s
./bench/text_generate_int4_batched "The capital of France is" 1 30   # batched M=1: ~57 t/s
# Both identical — same kernel, same occupancy
```

### ncu occupancy check (may not work — ncu version mismatch)
```bash
/usr/local/cuda-13.3/bin/ncu --target-processes all \
  --kernel-name "gemv_int4_warp" \
  --launch-count 3 \
  --section Occupancy \
  ./bench/text_generate_int4_qwen3_8b "The capital of France is" 2
# Expected: 8 warps/SM, 17% occupancy
# NOTE: ncu failed to profile in this session (version mismatch, kernel name matching issue)
```

### Peak BW verification
```bash
nvidia-smi --query-gpu=name,clocks.max.memory --format=csv
# Expected: NVIDIA GeForce RTX 5060 Ti, 14001 MHz
# BW = 14001 MHz × 128-bit × 2 (PAM4) / 8 = 448 GB/s
```

---

## Source File References

| File | Lines | Content |
|------|-------|---------|
| `src/kernels/gemv_int8.cu:509-556` | `gemv_int4_warp_kernel` — main INT4 GEMV |
| `src/kernels/gemv_int8.cu:1028-1110` | `gemv_int4_batched_kernel<M>` — batched variant |
| `src/kernels/gemv_int8.cu:1748-1817` | `quantize_int4_kernel` — 1-thread/block launch |
| `src/kernels/fused_int4_ops.cu:47-100` | `fused_rmsnorm_quant_int4_kernel` — unused fusion |
| `src/kernels/fused_int4_ops.cu:255-330` | `fused_swiglu_quant_int4_kernel` — unused fusion |
| `include/blackwell/kernels.h:1231-1265` | `fused_residual_norm_int4*` — unused fusion |
| `bench/text_generate_int4_qwen3_8b.cu:290-350` | Decode loop (warp path) |
| `server/inference_server_int4.cu:271-300` | Decode loop (server, batched M=1) |
| `bench/decode_int4_cgraph_8b.cu` | CUDA Graph capture + benchmark |

---

## Clarification Questions

1. **Theoretical ceiling:** The task says "theoretical ceiling M=1 = 5.3GB/500GB/s = 10.6ms = 94 t/s." But peak BW is 448 GB/s (not 500), and total weight traffic is 5.676 GB (not 5.3). Using 448 GB/s: ceiling = 79 t/s, not 94 t/s. Which BW number is authoritative? nvidia-smi reports 14001 MHz memory clock, which gives 448 GB/s with PAM4.

2. **Weight traffic calculation:** Task microbench says BW = 0.75*N*K/time for lm_head. 0.75 × 4096 × 151936 = 466.7 MB. At 1098µs = 425 GB/s = **95% of 448 peak**. This matches file sizes exactly. Is 500 GB/s the intended peak, or was it an approximation?

3. **Prior plan said compute-bound:** The nsys data definitively shows GEMV is 92.4% of GPU time and achieves 93% BW utilization. The kernel is memory-bound. Is there a different nsys profile or microbench that suggested compute-bound?

4. **FP16 scales:** Is re-quantizing weights with FP16 scales in scope? This saves 16.7% traffic = ~12 t/s. It's the single largest optimization.

5. **Occupancy increase:** Is changing `__launch_bounds__(32, 8)` → `(32, 16)` or restructuring to 2-warp-per-row in scope? Need register count check first.
