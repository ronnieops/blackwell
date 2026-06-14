# INT4 M=1 GEMV Bottleneck Analysis — Context for Closing the Throughput Gap

## 1. Current Performance Snapshot

| Config | t/s | ms/tok | vs llama.cpp Q4_K_M (84 t/s) |
|--------|-----|--------|------------------------------|
| Blackwell INT4 M=1 (8B) | **56** | 17.9 | 67% |
| Blackwell INT4 M=8 | 119 | 8.4 | 142% |
| Blackwell INT4 M=16 | 138 | 7.2 | 164% |

**nsys profile** (per AGENTS.md):
- `gemv_int4_batched`: 92.2% of decode time, 54.8 µs avg, 8851 instances
- `rmsnorm_batched`: 3.7%, 7.6 µs
- `quantize_int4`: 1.3%, 1.3 µs
- `attn_batched`: 0.9%, 3.9 µs
- `head_norm`: 0.5%, 1.1 µs

**Key insight from AGENTS.md**: "Our GEMV is compute-bound at M=1 (unpack overhead), not memory-bound."

---

## 2. Production M=1 Kernel — Exact Location & Code

### 2a. Production kernel: `gemv_int4_batched_kernel<1>`

**File**: `src/kernels/gemv_int8.cu`
**Kernel template**: lines 998–1060
**Launch wrapper**: `gemv_int4_batched()` lines 1071–1104

```
Grid:   dim3(N, 1)          — 1 block per output row
Block:  dim3(kINT4Block=32)  — 1 warp per block
Launch bounds: __launch_bounds__(32, 8)  — up to 8 warps/SM
```

For M=1, this is selected via the switch at line 1088:
```cpp
case 1: gemv_int4_batched_kernel<1><<<grid, kINT4Block, 0, stream>>>(...);
```

### 2b. Alternative M=1 kernel: `gemv_int4_warp_kernel`

**File**: `src/kernels/gemv_int8.cu`
**Lines**: 482–537
**Wrapper**: `gemv_int4_warp()` lines 964–985

Same grid/block as batched M=1 (1 warp/row), same scalar unpack path. Used by `decode_int4_cgraph_8b.cu` and `text_generate_int4_qwen3_8b.cu` (bench). Production server (`inference_server_int4.cu`) migrated to `gemv_int4_batched` per commit 35337ef.

**Both kernels are identical in algorithm**: 1 warp per output row, stride-32 K-loop, scalar nibble→float unpack + FP32 FMA, warp shuffle reduce.

### 2c. Nibble unpack helper (THE compute bottleneck)

**File**: `src/kernels/gemv_int8.cu`, lines 472–478

```cpp
__device__ __forceinline__ void int4_byte_to_floats(uint8_t b, float &f0, float &f1) {
    // Nibble stores q+8 (offset-binary, [0..15] for [-8..7]).
    // Convert back: val = nib - 8.
    int lo = (b & 0x0F) - 8;
    int hi = ((b >> 4) & 0x0F) - 8;
    f0 = static_cast<float>(lo);
    f1 = static_cast<float>(hi);
}
```

Comment at line 466: **"Strategy: scalar FP32 multiply-accumulate (no dp4a)."**

### 2d. The unpack+dot loop (THE compute-bound code)

**File**: `src/kernels/gemv_int8.cu`, lines 1019–1034 (batched kernel)

```cpp
for (int kb = tid; kb < num_K_blks; kb += 32) {
    const uint8_t* w_ptr = &W_packed[(size_t)n_out * (K / 2) + kb * PB];
    uint2 w_packed = *reinterpret_cast<const uint2*>(w_ptr);  // 8 bytes = 16 nibbles
    float w_sc = W_scale[(size_t)n_out * num_K_blks + kb];

    // ... for each M:
    const uint8_t* x_ptr = &x_packed[(size_t)mi * (K / 2) + kb * PB];
    uint2 x_packed_val = *reinterpret_cast<const uint2*>(x_ptr);
    float x_sc = x_scale[(size_t)mi * num_K_blks + kb];
    float prod_scale = w_sc * x_sc;

    const uint8_t* wb = reinterpret_cast<const uint8_t*>(&w_packed);
    const uint8_t* xb = reinterpret_cast<const uint8_t*>(&x_packed_val);

    float sum_f = 0.0f;
    #pragma unroll
    for (int j = 0; j < PB; ++j) {       // PB = 8
        float w0, w1, x0, x1;
        int4_byte_to_floats(wb[j], w0, w1);
        int4_byte_to_floats(xb[j], x0, x1);
        sum_f += w0 * x0 + w1 * x1;      // 2 FMA per iteration
    }
    acc[mi] += sum_f * prod_scale;
}
```

**Per K-block (16 elements):** 4× `int4_byte_to_floats` calls × 4 instructions each = ~16 int ops + 16 `int→float` conversions + 8 FMA + 1 scale multiply. **~41 scalar FP32/int instructions per 16 elements.**

**Per output element (K=4096):** 256 K-blocks / 32 threads = 8 K-blocks/thread × ~41 instr = ~328 instructions for unpack+dot. Plus 2 memory loads (weight + activation) of 8 bytes each.

### 2e. Reduction

**File**: `src/kernels/gemv_int8.cu`, lines 1037–1043

```cpp
// Warp shuffle reduction for each of M accumulators
#pragma unroll
for (int mi = 0; mi < M; ++mi) {
    acc[mi] += __shfl_xor_sync(0xffffffff, acc[mi], 16);
    acc[mi] += __shfl_xor_sync(0xffffffff, acc[mi], 8);
    acc[mi] += __shfl_xor_sync(0xffffffff, acc[mi], 4);
    acc[mi] += __shfl_xor_sync(0xffffffff, acc[mi], 2);
    acc[mi] += __shfl_xor_sync(0xffffffff, acc[mi], 1);
}
```

Thread 0 writes output (line 1048).

---

## 3. Memory Access Pattern

**Weight layout**: `W_packed[N][K/2]` — row-major, 1 output row = `K/2` contiguous bytes.
**Scale layout**: `W_scale[N][K/16]` — row-major, 1 scale per 16 elements.

**Access within kernel** (per thread, per K-block iteration):
- Weight: `W_packed[n_out * K/2 + kb * 8]` — 8-byte (`uint2`) load.
- Activation: `x_packed[kb * 8]` — same 8-byte load.

**Coalescing analysis**:
- 32 threads in the warp each load from different `kb` positions: `kb = tid, tid+32, tid+64, ...`
- Weight addresses: `n_out * K/2 + tid * 8`, stride 8 between consecutive threads.
- For `uint2` (8-byte) loads with stride 8: threads 0-31 read bytes at offsets `0, 8, 16, ..., 248` → 256 bytes per warp = **8 cache lines (32-byte each)**. This is a **strided pattern**: not fully coalesced for 4-byte transactions but coalesced for 8-byte transactions since consecutive threads read consecutive 8-byte blocks.

**Actually coalesced** for the 8-byte `uint2` loads — 32 threads × 8 bytes = 256 bytes = 8 sectors of 32B. L2 cache handles this well.

**Scale loads**: `W_scale[n_out * K/16 + kb]` and `x_scale[kb]` — 4-byte float, stride 4 between threads. Fully coalesced.

**Memory traffic per output element (M=1, K=4096)**:
- Weight: 4096/2 = 2048 bytes
- Weight scales: 4096/16 × 4 = 1024 bytes
- Activation: 2048 bytes (shared across all N threads)
- Activation scales: 1024 bytes (shared)
- Total per row: ~6144 bytes (weight side; activation amortized across N)

For q_proj (N=4096, K=4096): total data = 4096 × (2048 + 1024) = ~12.6 MB weight. At 500 GB/s peak: theoretical minimum = 12.6 MB / 500 GB/s = ~25 µs. Current kernel takes ~54.8 µs (nsys) → ~46% of peak BW. **This confirms compute-bound, not memory-bound.**

---

## 4. Why It's Compute-Bound — Detailed Analysis

### Root cause: scalar FP32 unpack instead of dp4a

The INT8 production kernel (`gemv_int8_warp_kernel`, lines 40–70) uses `__dp4a`:
```cpp
sumi = __dp4a(w32[0], x32[0], sumi);  // 4 INT8 multiply-adds in 1 instruction
```
This processes **16 INT8 values in 4 instructions** (4 `__dp4a` calls).

The INT4 kernel unpacks to **float** and does scalar FMA:
```cpp
int4_byte_to_floats(wb[j], w0, w1);  // 2 int ops + 2 int→float conversions
sum_f += w0 * x0 + w1 * x1;          // 2 FMA
```
This processes **16 INT4 values in ~41 instructions** (8 iterations × ~5 instructions).

**Instruction count ratio: INT4 scalar vs INT8 dp4a ≈ 41/4 ≈ 10×.**

### Arithmetic intensity comparison

| Kernel | Bytes/elem | Instr/elem | Arithmetic:Memory ratio |
|--------|-----------|-----------|------------------------|
| INT8 dp4a | 1.0 | ~0.25 (dp4a) | Low (memory-bound) |
| INT4 scalar float | 0.5 | ~2.5 (unpack+FMA) | High (compute-bound) |
| INT4 dp4a (potential) | 0.5 | ~0.75 (unpack + dp4a) | Balanced |

INT4 halves memory traffic vs INT8 but doesn't halve compute — it actually *increases* compute (unpack overhead) while reducing memory. This flips the bottleneck from memory to compute.

### Compute capacity check

RTX 5060 Ti (GB206, SM_120):
- 36 SMs, ~2048 FP32 ALUs/SM = 73728 FP32 ALUs
- At ~2.4 GHz: ~177 TOPS FP32
- INT4 kernel per-token: 36 layers × 8 GEMV calls (q,k,v,o,gate,up,down,lm_head) × ~1.4B FP32 ops = ~400B FP32 ops/token
- At 177 TOPS: theoretical = 400B/177T = 2.3 ms → **432 t/s compute limit**
- Actual: 56 t/s (17.9 ms) → **utilizing only 13% of FP32 compute**

The kernel is wasting compute on int→float conversions and scalar FMA that dp4a or tensor cores could do far more efficiently. The issue is instruction overhead (unpack, conversion, address calc), not raw FLOPs.

### Multi-thread per row (M>1) confirmation

At M=8 (119 t/s), weight traffic is amortized across 8 activations. Weight loaded once, 8 activations dot-producted. The compute-per-byte ratio improves (weight unpack done once per 8 uses). At M=8 the kernel transitions back toward memory-bound, hence the superlinear scaling.

---

## 5. Block-16 Scale Layout vs llama.cpp Q4_K Super-Blocks

### Our format (block-16 symmetric INT4)

- **Block size**: 16 elements
- **Scale**: 1× FP32 per block (4 bytes) — `absmax/7.0f`
- **Values**: 8 bytes (16 × 4-bit packed nibbles, offset-binary: `nib-8` maps to `[-8,7]`)
- **Bytes per block**: 8 (data) + 4 (scale) = 12 bytes for 16 values = **0.75 bytes/value**
- **Effective bits**: 4 bits value + 2 bits scale overhead = 6 bits/effective value
- **Weight file**: `.int4_t` (header 5 ints: K, N, ?, ?, ?) + `.scale_t` (header 5 ints, scale count = h[3]*h[4])
- **Storage**: `W_packed[N][K/2]` + `W_scale[N][K/16]` — separate arrays

**File headers** (from `upload_w4()` in bench files, line 39):
```cpp
int h[5]; fread(h, 4, 5, f);  // Read 5 ints
// h[0] = K, h[1] = N
// int4_t data: K*N/2 bytes follow
// scale_t: h[3]*h[4] = N*(K/16) floats follow
```

### llama.cpp Q4_K format

- **Super-block size**: 256 elements
- **Scale**: 1× FP16 (d) + 1× FP16 (d_min) per super-block + 8× 6-bit sub-block scales/mins
- **Sub-blocks**: 8 sub-blocks of 32 elements each within super-block
- **Values**: 4-bit quantized per sub-block (with sub-block scale delta)
- **Bytes per super-block**: 128 (data) + 2 (d) + 2 (d_min) + 6 (6-bit scales) + 6 (6-bit mins) = 144 bytes for 256 values = **0.5625 bytes/value** (25% less overhead than our 0.75)
- **Key difference**: Nested 2-level quantization (coarse super-block scale + fine sub-block delta) → better accuracy per bit

### What would change to adopt Q4_K-style super-blocks

1. **Re-quantize all weights** (5.3 GB, 7 weight matrices × 36 layers + embed + lm_head)
2. **New kernel**: unpack 6-bit sub-block scales, apply 2-level scale
3. **No quality improvement needed** — our PPL (23.52) is already acceptable; Q4_K would *improve* quality to ~12-14 range with same 4-bit weights due to better scale granularity
4. **More complex unpack** — would *increase* compute per element, not decrease it
5. **Net effect on M=1 speed**: NEGATIVE — more complex unpack = more compute. Q4_K doesn't help with the compute-bound problem.

**Conclusion**: Switching to Q4_K format does NOT help close the M=1 gap. The problem is unpack instruction count, not quantization granularity. However, Q4_K is how llama.cpp achieves 84 t/s — because llama.cpp uses **tensor cores** for GEMV, not scalar unpack.

---

## 6. Tensor-Core Feasibility Analysis

### 6a. Existing WMMA INT8 GEMM kernels

**`gemm_int8_wmma.cu`** (lines 1–80+):
- `wmma::mma_sync` with `m16n16k16` INT8 fragments
- Designed for prefill (M≥16), NOT decode (M=1)
- Block config: `__launch_bounds__(32, 2)`, 1 warp per 16×16 tile
- Per-block scales applied via FP32 after INT32 accumulation

**`gemm_int8_wmma_fast.cu`** (lines 1–100+):
- 32×32 tiles (4 WMMA blocks per CTA)
- 4 warps per CTA, `__launch_bounds__(128, 1)`
- Direct FP32 accumulation from `c_frag.x[i]`
- 1.5-2× faster than `gemm_int8_wmma` for M≥32

**`gemm_int8_mma.cu`**: STUB — returns `cudaErrorNotSupported`

**`gemm_int8_dp4a`** (`gemv_int8.cu`, line 1600): DP4A-based GEMM, works but not tensor-core.

### 6b. INT4 → INT8 upcast (the key enabler)

**Existing upcast kernel**: `unpack_fp4_pack_int8()` in `src/kernels/quantize.cu` (lines 124–167)
- But this converts FP4 E2M1 → INT8, not INT4 → INT8
- Operates element-wise, not fused with GEMV

**No existing INT4→INT8 upcast kernel exists.** Would need to write one.

**INT4→INT8 upcast is trivial**: `int8_val = nib - 8` (already offset-binary). Each nibble maps directly to `int8_t`. 2 nibbles → 2 int8_t, expand 1 byte to 2 bytes.

### 6c. Tensor-core GEMV for M=1: feasibility

**WMMA m16n16k16 INT8 fragments**:
- Fragment A: 16×16 tile (M dimension) — for M=1, only 1 row used (15 wasted)
- Fragment B: 16×16 tile (N/K dimensions) — 16 output columns per fragment
- For GEMV (M=1), this is 16× overprovisioned on the M dimension

**Problem**: WMMA is designed for GEMM (M≥16). For GEMV (M=1), using WMMA wastes 15/16 = 94% of the compute. However, since we're currently at 13% compute utilization, even 6% utilization of tensor cores (1/16 of m16n16k16) might be faster than scalar FP32.

**Better approach for M=1**: `dp4a` with INT4→INT8 upcast. Load 16 INT4 nibbles (8 bytes), upcast to 16 INT8 (16 bytes), use 4× `__dp4a` (same as INT8 kernel). This eliminates the scalar float conversion entirely.

**Estimated speedup from dp4a INT4 path**:
- Current: ~41 instructions per 16 elements (scalar float)
- dp4a: ~12 instructions per 16 elements (8 nibble→int8 unpacks + 4 dp4a)
- Speedup: ~3.4× on the compute path
- If compute-bound: 56 t/s × 3.4 = ~190 t/s (memory-bound ceiling would limit this)
- Memory-bound ceiling at INT4: K=4096, N=4096 → 12.6 MB / 500 GB/s = 25 µs → ~400 t/s (for q_proj alone)

**Realistic estimate**: dp4a INT4 would push M=1 from compute-bound toward memory-bound. Expected: **100-140 t/s** (memory-bound limited by total weight traffic across all 8 GEMV layers).

### 6d. WMMA for M=1: speculative

A WMMA m16n16k16 tile with M padded to 16 (replicate activation 16×) could use tensor cores. The overhead:
- 16× more activation data (negligible — activation is tiny)
- Each tile computes 16 output rows simultaneously (need N/16 tiles)
- Weight fragment B is 16×16 from column-major B — need weight in transposed layout or shuffles

This would require weight layout change (col-major INT8 with upcast) and is more complex than dp4a. Lower priority.

---

## 7. Decode Loop — Full Per-Token Kernel Sequence

### Production server path (`inference_server_int4.cu`)

**`decode_one_token()`** (lines 265–300): called once per generated token.

Per layer (×36 layers):
1. `quantize_int4_batched` (M=1) — quantize activation to INT4
2. `gemv_int4_batched` (M=1, H×Q) — q_proj
3. `gemv_int4_batched` (M=1, H×KV) — k_proj
4. `gemv_int4_batched` (M=1, H×KV) — v_proj
5. `head_norm_kernel` × 2 — Q and K head norms
6. `apply_rope_kernel` × 2 — Q and K RoPE
7. `update_kv_cache` — write K,V to cache
8. `attention_decode_batched_gqa` — attention
9. `quantize_int4_batched` (M=1) — quantize attn output
10. `gemv_int4_batched` (M=1, Q×H) — o_proj
11. `vector_add_fp32` — residual
12. `quantize_int4_batched` (M=1) — quantize MLP input
13. `gemv_int4_batched` (M=1, H×I) — gate_proj
14. `gemv_int4_batched` (M=1, H×I) — up_proj
15. `apply_swiglu` — activation
16. `quantize_int4` — quantize MLP intermediate
17. `gemv_int4_batched` (M=1, I×H) — down_proj
18. `vector_add_fp32` — residual

Per token (after 36 layers):
19. `fused_rmsnorm` — final norm
20. `quantize_int4_batched` — quantize for lm_head
21. `gemv_int4_batched` (M=1, H×V) — lm_head (largest: N=151936)
22. `apply_repetition_penalty` (optional)
23. `sample_gpu` — sampling

**Total GEMV calls per token**: 7 × 36 + 1 = **253 GEMV launches**

### GEMV sizes (8B Qwen3)

| Projection | K | N | Weight bytes | Notes |
|-----------|---|---|--------------|-------|
| q_proj | 4096 | 4096 | 8.4 MB | |
| k_proj | 4096 | 1024 | 2.1 MB | |
| v_proj | 4096 | 1024 | 2.1 MB | |
| o_proj | 4096 | 4096 | 8.4 MB | |
| gate_proj | 4096 | 12288 | 25.2 MB | Largest attention-side |
| up_proj | 4096 | 12288 | 25.2 MB | |
| down_proj | 12288 | 4096 | 25.2 MB | K=12288 (MLP dim) |
| lm_head | 4096 | 151936 | 311 MB | Once per token |

**Total weight traffic per token**: (8.4+2.1+2.1+8.4+25.2+25.2+25.2) MB × 36 layers + 311 MB = **3485 MB + 311 MB ≈ 3.8 GB**

At 500 GB/s: theoretical minimum = 3.8 GB / 500 GB/s = **7.6 ms/token = 131 t/s** (memory-bound ceiling for M=1)

Current: 56 t/s = 17.9 ms → we're at **42% of memory-bound ceiling**. The compute overhead (scalar unpack) is eating the other 58%.

---

## 8. Quantization Format Details

### Block-16 symmetric INT4 (our format)

**Quantization kernel**: `quantize_int4_kernel()` in `src/kernels/gemv_int8.cu` (lines 1711–1757)

```cpp
// Block size = 16. Per-block: absmax / 7 → quantize [-7..7] → nibble-pack.
float sc = (absmax > 1e-10f) ? (absmax / 7.f) : (1.f / 7.f);
// ...
int q0 = (int)roundf(v0 / sc);
q0 = max(-8, min(7, q0));
uint8_t nib0 = (uint8_t)((q0 + 8) & 0x0F);  // offset-binary: 0=-8, 15=7
```

- Scale = `absmax(block) / 7.0` (FP32)
- Quantized value range: `[-8, 7]` (4 bits, offset-binary)
- Dequantized: `(nib - 8) * scale`
- Storage: nibbles packed 2/byte, scales stored separately as FP32 array

### AWQ calibration (applied)

**Script**: `scripts/quantize_awq_int4_8b.py`
- Alpha = 0.5 (script default, AGENTS.md says 0.6 best)
- Per-channel activation-aware scaling folded into block scales
- `w_sc_new[n] = w_sc[n] / s[n]` — no kernel changes needed
- Current best PPL: 21.82 (with AWQ) vs 23.52 (without)

### Weight file layout

```
weights_int4_qwen3_8b/
  embed_tokens.int4_t       — header[5 ints] + K*N/2 bytes packed nibbles
  embed_tokens.scale_t      — header[5 ints] + N*(K/16) FP32 scales
  0_self_attn.q_proj.int4_t
  0_self_attn.q_proj.scale_t
  ... (7 weight matrices × 36 layers = 252 pairs)
  0_self_attn.k_proj.*
  0_self_attn.v_proj.*
  0_self_attn.o_proj.*
  0_mlp.gate_proj.*
  0_mlp.up_proj.*
  0_mlp.down_proj.*
  qk_norms.f32               — Q/K head norm weights (FP32, NL*2*hd floats)
  {L}_input_layernorm.f32    — per-layer RMSNorm weights (FP32)
  {L}_post_attention_layernorm.f32
  final_norm.f32
  lm_head.int4_t
  lm_head.scale_t
```

---

## 9. Existing INT8 dp4a Kernel (for adaptation reference)

**File**: `src/kernels/gemv_int8.cu`, lines 40–70

```cpp
// gemv_int8_warp_kernel (production INT8)
for (int kb = tid; kb < num_K_blks; kb += 32) {
    // Load 16 INT8 weight values (4× uint32 = 16 bytes)
    const uint8_t* w_ptr = &W_int8[(size_t)n_out * K + kb * B];
    uint4 w16 = *reinterpret_cast<const uint4*>(w_ptr);
    // ... same for activation
    uint4 x16 = *reinterpret_cast<const uint4*>(x_ptr);

    int sumi = 0;
    int* w32 = reinterpret_cast<int*>(&w16);
    int* x32 = reinterpret_cast<int*>(&x16);
    // __dp4a: 4-way int8 SIMD dot product per iteration (4 × 4 = 16 total)
    sumi = __dp4a(w32[0], x32[0], sumi);
    sumi = __dp4a(w32[1], x32[1], sumi);
    sumi = __dp4a(w32[2], x32[2], sumi);
    sumi = __dp4a(w32[3], x32[3], sumi);
    acc += (float)sumi * w_sc * x_sc;
}
```

**Key difference**: INT8 loads 16 bytes (16 values), INT4 loads 8 bytes (16 values). INT8 uses `uint4` (16 bytes), INT4 uses `uint2` (8 bytes). The dp4a path processes the same 16 elements in 4 instructions vs 41 for scalar float.

---

## 10. Constraints (from AGENTS.md + codebase)

### Build
- **CUDA 13.3**: `/usr/local/cuda-13.3/bin/nvcc`
- **SM_120a** (not `compute_120`): `-gencode=arch=compute_120a,code=sm_120a`
- **CMake**: `CUDACXX` env var must be set before `project()` in CMakeLists.txt
- **Build**: `CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build && cmake --build build --parallel`
- **Kernel count**: 189 exported symbols in `build/libblackwell_kernels.a`

### Runtime
- **killall hashcat** before any measurement (60s respawn window, steals GPU)
- RTX 5060 Ti: 36 SMs, ~500 GB/s GDDR7, 16 GB VRAM
- INT4 8B weights: 5.3 GB → fits with room for KV cache

### Model dimensions (8B Qwen3)
- `H=4096, Q=4096, KV=1024, I=12288`
- `nqh=32, nkv=8, hd=128` (GQA: 4 KV groups)
- `NL=36` layers
- `V=151936` vocab size
- `rope_theta=1000000.0f`
- `MAXSEQ=4096` (server), `MAXSEQ=512` (batched bench), `MAXSEQ=2048` (inference_server_int4)

### Kernel API constraints
- `gemv_int4_batched`: K%16==0, N%16==0, M=1..16
- `quantize_int4`: block-16 symmetric, absmax/7, offset-binary nibbles
- `_pos` variants for prefill loops (direct seq_pos arg, no pinned buffer race)
- `update_decode_seq_pos` writes pinned host → cudaMemcpyAsync to device (graph-safe)
- All weight matrices exceed L2 cache (32 MB) — no caching benefit

### Server constraints
- Repetition penalty (default 1.5 in `inference_server_int4.cu` line 610, default 1.0 in batched)
- Rate limiting: 5 req/s, burst 10, all endpoints
- Payload limit: 1MB, max_tokens clamp [1,2048]
- Multi-chunk prefill working (v0.11.0)

---

## 11. Benchmarks & Validation Commands

### End-to-end text generation (correctness + throughput)
```bash
killall hashcat 2>/dev/null
./bench/text_generate_int4_qwen3_8b "The capital of France is" 30  # INT4 M=1: 59 t/s
./bench/text_generate_int4_batched "prompt" M gen_tokens           # INT4 batched
```

### CUDA Graph INT4 8B
```bash
./bench/decode_int4_cgraph_8b [tokens]  # CUDA Graph decode
```

### Microbenchmark (per-GEMV timing)
```bash
./bench/bench_gemv_int4 [K] [N] [M] [iters]
# Default: K=4096, N=4096, M=1, 100 iters
# Tests all 8B model GEMV sizes: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head
# Reports: µs/call, GB/s, % of peak BW, TOPS
```

### PPL validation
```bash
./bench/bench_ppl_int4_8b  # INT4 8B PPL (expected: 21.82 AWQ, 23.52 baseline)
```

### HTTP server
```bash
./server/http_subprocess int4_8b &          # INT4 warp server (~56 t/s)
./server/http_subprocess batched &           # INT4 batched server (~63 t/s)
curl http://localhost:8123/health
curl -X POST http://localhost:8123/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","max_tokens":5}'
```

### Build server
```bash
CUDACXX=/usr/local/cuda-13.3/bin/nvcc nvcc -O3 -std=c++17 -arch=sm_120a \
  server/inference_server_int4.cu build/libblackwell_kernels.a \
  -I include -I /usr/local/cuda-13.3/include \
  -L /usr/local/cuda-13.3/targets/x86_64-linux/lib \
  -o server/inference_server_int4 -lcudart -lpthread -lz
```

---

## 12. Risks & Open Questions

### Implementation risks

1. **Re-quantization cost**: If we switch from scalar float to dp4a, we need INT4→INT8 upcast at runtime. This is a **per-GEMV-call** upcast (activation) + possibly weight pre-processing. Activation upcast is cheap (K/2 bytes → K bytes), but weight upcast would require either:
   - (a) Pre-upcast weights at load time (doubles weight memory: 5.3 GB → 10.6 GB, still fits 16 GB)
   - (b) On-the-fly upcast in kernel (adds back unpack overhead)
   - **Recommended**: (a) pre-upcast at load time, store INT8 weights in GPU memory

2. **PPL regression risk**: Switching from INT4×INT4 dot product to INT8×INT8 (after upcast) should produce **identical results** — the quantized values are the same, just stored in INT8 instead of INT4. The math is: `sum((nib_w - 8) * (nib_x - 8)) * w_sc * x_sc`. With INT8 upcast, this becomes `sum(int8_w * int8_x) * w_sc * x_sc` — same values, same result. **No PPL regression expected.**

   However, if we switch to **dp4a** (INT8×INT8 accumulate), the `__dp4a` instruction does `a*b + c` in INT32. The sum will overflow for large K? Let's check: K=12288, max product per element = 7×7=49, max sum = 12288×49 = 602,112. INT32 max = 2.1 billion. **No overflow risk.** dp4a accumulates in INT32.

3. **AWQ interaction**: AWQ scales are folded into block scales (`w_sc / s[n]`). The dp4a path uses the same scale application (`sumi * w_sc * x_sc`), so AWQ is transparent. **No interaction.**

4. **Weight layout change for dp4a**: Current INT4 stores nibbles packed 2/byte. For dp4a, we need INT8 values. Options:
   - (a) Pre-expand to INT8 at load time: `upload_w4` → `upload_w8` that upcasts. Simple, 2× memory.
   - (b) On-the-fly nibble→int8 in kernel: saves memory but adds unpack back (defeats purpose).
   - (c) Pre-expand to INT8 and store as `.int8_t` files alongside `.int4_t` files.
   - **Recommended**: (a) expand at load time, no file format change.

5. **Block-16 scale resolution**: dp4a accumulates 4 INT8 products at a time. Our block size is 16. After dp4a over 16 elements (4 dp4a calls), multiply by scale. This matches the INT8 kernel pattern exactly. **No scale resolution issue.**

### Open questions for implementation confidence

1. **Should we pre-upcast weights to INT8 at load time, or write a fused unpack-dp4a kernel?**
   - Pre-upcast: simplest, 2× memory, zero unpack overhead in GEMV, reuses existing `gemv_int8_warp_kernel` infrastructure
   - Fused unpack-dp4a: saves memory, but adds nibble→int8 unpack per element in kernel (less overhead than float unpack, but still nonzero)
   - **Evidence**: existing `gemv_int8_warp_kernel` already does exactly what we need (dp4a + block-16 scales). Pre-upcasting weights to INT8 and calling this kernel directly would be the minimal change.

2. **Will the INT8 kernel accept our block-16 scale layout?**
   - INT8 kernel expects `W_int8[N][K]` (1 byte/elem) and `W_scale[N][K/16]` (FP32 per 16 elems)
   - Our INT4 has `W_packed[N][K/2]` and `W_scale[N][K/16]` — **scale layout is identical**
   - Weight layout differs only in packing (2/byte vs 1/byte)
   - **Answer**: Yes, scale layout is compatible. Only weight data needs upcast.

3. **Does the activation quantization path need changes?**
   - `quantize_int4` produces INT4 packed activations + INT4 scales
   - For dp4a INT8 path, need `quantize_int8` activations + INT8 scales
   - `quantize_int8` already exists (production INT8 path uses it)
   - But block-16 scales: INT4 uses `absmax/7`, INT8 uses `absmax/127` — different scale values!
   - **Critical**: If weight scale uses `/7` (INT4) and activation scale uses `/127` (INT8), the product `w_sc * x_sc` still gives correct result because the quantized values compensate. BUT: weight values must be upcast from INT4 range `[-8,7]` to INT8 range `[-8,7]` (NOT re-quantized to `[-127,127]`). The upcast is just `int8_t(nib - 8)`, preserving the original scale.
   - **Answer**: Activation must be quantized to INT4 first (using `quantize_int4`), then upcast to INT8. Or: quantize directly to INT8 using INT4-compatible scale (`absmax/7`). The former is simpler.

4. **Can we just pre-upcast everything to INT8 at startup and use the existing INT8 production path?**
   - INT8 M=1 benchmark: 181.5 t/s (1.7B, no head_norm/RoPE) — but 8B not benchmarked with INT8 M=1 decode
   - INT8 weights: 9.6 GB (vs 5.3 GB INT4) — fits in 16 GB
   - This would give INT8-quality (PPL=18.65 vs INT4 PPL=23.52) at potentially higher M=1 speed
   - **BUT**: INT8 traffic is 2× INT4. Memory-bound ceiling drops from 131 t/s to ~65 t/s.
   - **This doesn't help** — INT8 M=1 would be ~56 t/s (memory-bound), same as current INT4 compute-bound. The whole point of INT4 is less memory traffic.

5. **Blackwell SM_120 specific tensor-core capabilities for INT4?**
   - Blackwell (sm_120) supports mma.sync for INT8 (m16n8k32 or m16n16k16 via WMMA)
   - Native INT4 tensor-core MMA: **unknown** — need to check PTX docs for sm_120
   - If sm_120 supports INT4×INT4 MMA (like some Ampere+ variants), we could use tensor cores directly
   - **Not checked**: search CUDA 13.3 docs for INT4 MMA support on sm_120

6. **CUDA Graph compatibility**: Any new kernel must be graph-safe (no H2D memcpy inside capture). The `_device` and `_pos` variants exist for this. A dp4a INT4 kernel would naturally be graph-safe (no host interaction).

### Escalation triggers

- If sm_120 supports native INT4 MMA → radically different approach (tensor core GEMV, not dp4a)
- If weight upcast at load time exceeds 16 GB GPU memory → need on-the-fly upcast or weight compression
- If PPL regresses after upcast (shouldn't, but verify with `bench_ppl_int4_8b`)

---

## 13. Summary: Likely Path to Close M=1 Gap

### Option A: dp4a INT4→INT8 upcast (recommended, lowest risk)

1. Write nibble→int8 upcast function (trivial: `(b & 0xF) - 8`, `(b >> 4) - 8`)
2. Modify `gemv_int4_batched_kernel` to use dp4a instead of scalar float:
   - Upcast 8 bytes INT4 → 16 int8_t in registers
   - Use 4× `__dp4a` instead of 8× scalar FMA
   - Apply scale: `acc += (float)sumi * w_sc * x_sc`
3. Expected: ~3.4× compute speedup → **100-140 t/s** (memory-bound limited)
4. No weight format change, no PPL regression, no re-quantization

### Option B: Pre-upcast weights to INT8 at load time

1. At weight load (`upload_w4`), expand INT4 nibbles to INT8 bytes
2. Call existing `gemv_int8_warp_kernel` / `gemv_int8_batched` instead of INT4 kernel
3. Memory: 5.3 GB → 10.6 GB (fits 16 GB)
4. Traffic: 2× INT4 → INT8 memory-bound ceiling drops to ~65 t/s
5. **WORSE than Option A** — defeats the purpose of INT4

### Option C: WMMA tensor-core GEMV (high effort, uncertain payoff)

1. Upcast INT4→INT8 at runtime
2. Use WMMA m16n16k16 with M-padded activations (replicate ×16)
3. 16× waste on M dimension, but tensor cores are ~10× faster than dp4a
4. Net: potentially ~1.5-2× over dp4a for large N (gate/up/down/lm_head)
5. Complex: weight layout change, fragment loading, scale application
6. **Highest ceiling but highest risk**

### Option D: dp4a with weight pre-upcast + INT4 storage (hybrid)

1. Pre-upcast weights to INT8 at load time (GPU memory only, no file change)
2. Keep activation quantization as INT4 (for speed)
3. Upcast activation INT4→INT8 in kernel registers (cheap)
4. Use dp4a for INT8×INT8 dot product
5. Weight memory: 10.6 GB, but weight traffic stays INT8 (2× INT4)
6. **Same traffic problem as Option B** for weights

**Recommendation**: Option A. Modify `gemv_int4_batched_kernel` to use dp4a with on-the-fly nibble→int8 upcast. Keeps INT4 memory traffic (5.3 GB), eliminates scalar float overhead, no weight format change. Validate with `bench_gemv_int4` microbenchmark first, then `text_generate_int4_qwen3_8b` end-to-end.
