# Research Handoff: 2-warp GEMV & AWQ+fusion PPL

## Task 6: 2-warp GEMV Experiment

### Hypothesis
llama.cpp uses 2 warps (64 threads) per output row for Q4_K GEMV. Blackwell uses 1 warp (32 threads). If the memory controller can service more concurrent requests with 2 warps, throughput may improve.

### Current evidence (from AGENTS.md)
- Microbench shows 95% of 448 GB/s peak with 1 warp
- Occupancy experiment (8→16 warps/SM): 0% gain
- GEMV is memory-saturated at 1 warp

### Experiment design

1. **Create benchmark** `bench/bench_2warp_gemv.cu`
   - Microbench: measure BW efficiency at 1 vs 2 warps
   - Use `gemv_int4_warp_kernel` as base
   - 2-warp variant: 64 threads/block, stride-64 over K-blocks
   - Block size: `(nwarps * 32)` threads, where nwarps=1 or 2
   - Measure: BW (GB/s), efficiency (% of 448 GB/s peak)

2. **Kernel template parameter**
   ```cuda
   template<int NWARPS>  // 1 or 2
   __global__ void gemv_int4_warp_kernel_2w(
       float* y_out,
       const uint8_t* x_packed,
       const float* x_scale,
       const uint8_t* W_packed,
       const WScaleT* W_scale,
       int K, int N)
   ```
   - NWARPS=1: 32 threads, stride-32 (current)
   - NWARPS=2: 64 threads, stride-64, 2× warp shuffle reduce + smem

3. **Test matrices** (Qwen3-8B sizes)
   - Q: 4096×4096
   - K: 1024×4096
   - gate: 12288×4096
   - down: 4096×12288
   - lm_head: 151936×4096

4. **Success criteria**
   - ≥5% BW improvement over 1-warp
   - No PPL regression (verify with bench_ppl_int4_8b)

5. **If it works**: Modify `gemv_int4_batched_f16wsc` to use 2 warps for M=1. Update server and bench.

### Resources
- `src/kernels/gemv_int8.cu` lines 529-588 (current warp kernel)
- `src/kernels/gemv_int8.cu` lines 1074-1170 (current batched kernel)
- llama.cpp reference: `vecdotq.cuh` lines 505-528 (VMMQ variant, 2-warps)

---

## Task 7: AWQ + Fusion PPL

### Observation
- AWQ per-layer α + FP16 scales: PPL 21.98
- Fusion kernels (no AWQ) + FP16 scales: PPL 21.98
- These are identical. Either AWQ doesn't help on top of fusion, or both produce the same PPL for different reasons.

### Experiment design

1. **Test configurations**

   | Config | Weights | Kernels | Expected PPL |
   |--------|---------|---------|-------------|
   | A | non-AWQ FP16sc | non-fused | 24.39 |
   | B | non-AWQ FP16sc | fused | 21.98 |
   | C | AWQ FP16sc | non-fused | 21.98 |
   | D | AWQ FP16sc | fused | ? |

   Config D is unknown. If AWQ + fusion gives lower PPL than either alone, they're complementary.

2. **How to test**
   - Config A, B: already measured (24.39, 21.98)
   - Config C: already measured (21.98)
   - Config D: need to run `bench_ppl_int4_8b` with AWQ weights but fused kernels disabled

3. **The fusion effect** (from AGENTS.md)
   - `fused_rmsnorm_quant_int4` replaced `fused_rmsnorm` + `quantize_int4` (2→1 kernel)
   - PPL changed from 24.39 to 21.98
   - The PPL improvement came from different FP32 reduction order (128→256 threads)
   - FP32 reduction is NOT associative — different summation order → different rounding

4. **Hypothesis**
   - AWQ improves weight quantization (better INT4 nibbles)
   - Fusion improves activation quantization (different rounding in RMSNorm+quant)
   - These are orthogonal improvements. Config D should give PPL < 21.98.
   - Expected: ~21.5-21.8

5. **Procedure**
   ```bash
   # Config D: AWQ weights with fused kernels
   # The bench_ppl_int4_8b uses fused kernels by default
   ./bench/bench_ppl_int4_8b weights_int4_qwen3_8b_awq_perlayer_fp16sc
   
   # Config C: AWQ weights with NON-fused kernels (for comparison)
   # Need to modify bench or use a version without fusion
   ```

6. **Success criteria**
   - PPL < 21.98 (improvement over AWQ alone or fusion alone)
   - If PPL = 21.98, then AWQ and fusion affect different parts of the pipeline but produce same final error

### Key files
- `bench/bench_ppl_int4_8b.cu` — PPL benchmark (uses fused kernels)
- `src/kernels/fused_int4_ops.cu` — fusion kernels (fused_rmsnorm_quant_int4, fused_swiglu_quant_int4)

### Risk
The PPL benchmark uses the same token-by-token decode path as generation. To test Config D (AWQ + fused), just run the existing benchmark with AWQ weight dir. It already uses fused kernels.

```bash
# Quick test
./bench/bench_ppl_int4_8b weights_int4_qwen3_8b_awq_perlayer_fp16sc
```

If this gives PPL 21.98 (same as AWQ alone), run a version without fusion to compare:
- Modify bench to call `fused_rmsnorm` + `quantize_int4` separately instead of `fused_rmsnorm_quant_int4`
- Or comment out the fusion kernel calls in the decode loop