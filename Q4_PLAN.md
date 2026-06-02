# Q4 Quantization Plan — Blackwell INT8→INT4 Migration

## Overview

Current INT8 M=1 decode is bandwidth-limited: 1 byte/param vs Q4_K_M's ~0.5 byte/param.
INT4 halves the weight data read: 0.5 byte/param (packed 2 values/byte) → potential ~350+ t/s M=1.

## Expected Impact

| Metric | INT8 (current) | INT4 (target) | Q4_K_M (llama.cpp) |
|--------|---------------|---------------|-------------------|
| bytes/param | 1.0 | ~0.62 | ~0.53 |
| M=1 t/s (1.7B) | 181.5 | **~250-290** | 293.4 |
| vs Q4_K_M | 62% | **~85-99%** | 100% |
| M=8 total t/s | 324.6 | **~450+** | — |

INT4 with 16×16 blocks + FP32 scales = 0.75 bytes/param (not as tight as Q4_K_M's 0.53). Using 32×4 blocks brings it to 0.625. Best case with 32×8 blocks: ~0.56. Closes most of the gap.

---

## Phase 1: Weight Format + Conversion Tools

### 1.1 File Format

Current INT8 format (per weight matrix):
```
.int8_t:  [K][N] int8_t values          (K×N bytes)
.scale_t: [K/16][N/16] float scales     (K/16 × N/16 floats + 5-int header)
```

INT4 format (minimal change, same block structure):
```
.int4_t:  [K][N/2] uint8_t packed       (K×N/2 bytes, nibble[even]=val[2i], nibble[odd]=val[2i+1])
.scale_t: [K/16][N/16] float scales     (identical to INT8)
```

Block structure remains 16-element groups with per-16-element FP32 scale.
This gives 0.75 bytes/param (8 bytes values + 4 bytes scale per 16 values).

### 1.2 Conversion Script

File: `scripts/quantize_int4.py`

```python
def quantize_bf16_to_int4(weights_bf16, K, N):
    """Quantize BF16 weight matrix to INT4 block format.
    
    Per-block (16 elements): compute absmax → scale = max/7 → quantize to 4-bit.
    Pack 2 × 4-bit values per byte (little-endian: lower nibble = even index).
    Returns: int4_data (K × N/2 bytes), scales (K/16 × N/16 floats)
    """
```

Steps:
1. Load BF16 weights from HuggingFace safetensors (or use existing FP32 conversion)
2. For each 16×N column block: compute per-16-element absmax
3. For each element: quantize to 4-bit signed (range -7..7, FP4-like)
4. Pack 2 nibbles per byte
5. Save `.int4_t` and `.scale_t` files
6. Save reference FP32 weights for validation

### 1.3 Validation

For each weight matrix, compute:
- Per-block MSE vs FP32
- Max absolute error per block
- PSNR across entire matrix
- Compare against INT8 quantization quality

Target: INT4 PSNR > 40 dB, max error < 5% of INT8 error.

### 1.4 Files to Create
- `scripts/quantize_int4.py` — weight converter
- `scripts/validate_int4.py` — quality validator

---

## Phase 2: INT4 GEMV Kernel

### 2.1 Kernel: `gemv_int4_warp`

File: `src/kernels/gemv_int4.cu`

Architecture (based on `gemv_int8_warp`):

```cuda
// Warp-cooperative INT4 GEMV (1 warp/row, shuffle reduce)
// y[N] = x_scaled[K] @ W_int4[K×N] + ...
//
// Weight layout: W_int4 packed as uint8_t[K][N/2], 2 values/byte
// Block scales: float[K/16][N/16]
//
// Each warp processes 1 output row (N elements).
// 32 lanes × K/16 values per lane = 2K values per warp.
//
// For INT4: 32 lanes × K/16 nibble-pairs = K values per warp
//            Each lane loads K/16 bytes → K/8 int4 values
//            Dequantize nibble → multiply by scale → __dp4a accumulate

__global__ void gemv_int4_warp_kernel(
    float* __restrict__ y,
    const int8_t* __restrict__ x_i4,
    const float* __restrict__ x_scale,
    const uint8_t* __restrict__ W_i4,   // packed INT4 weights
    const float* __restrict__ W_scale,
    int K, int N, cudaStream_t stream);
```

Key design decisions:
- Block size: 16 elements (same as INT8, simplifies scale indexing)
- Packing: 2 int4 values per uint8 byte (lower nibble = even index, upper = odd)
- Dequant: (nibble - 7) × scale (unsigned 4-bit → signed 8-bit)
- Dot product: `__dp4a` on dequantized int8 values
- Registers: 2× INT8 (need to unpack nibbles before __dp4a)
- Throughput: ~2× INT8 memory BW for same compute (halves DRAM reads)

### 2.2 Warp Loop Structure

```
for each 16-element block along K:
    load 2 int4 values from weight (1 byte)
    upper_nibble = byte >> 4
    lower_nibble = byte & 0x0F
    dequant_i4_to_i8: val = (nibble - 7) * scale  (re-center from unsigned to signed)
    pack into int8 array for __dp4a
    after 4 dequantized pairs → 1 __dp4a instruction
    shuffle reduce across 32 lanes
```

### 2.3 Dequantization Cost

For INT8: load 1 int8 = 1 instruction → __dp4a on 4 values = 5 inst/4 values = 1.25 inst/val
For INT4: load 1 byte = 1 inst → unpack 2 nibbles = 2 inst → dequant ×2 = 2 inst → pack to int8 = 1 inst → __dp4a on 4 values

Total: ~7 inst / 4 int4 values ≈ 1.75 inst/val (vs 1.25 for INT8)

But memory reads per value: INT4 = 0.5 bytes/val vs INT8 = 1 byte/val. The extra dequant compute is hidden behind memory latency.

### 2.4 Files to Create
- `src/kernels/gemv_int4.cu` — kernel implementation
- `include/blackwell/kernels.h` — add `gemv_int4_warp` declaration

---

## Phase 3: Pipeline Integration + Benchmark

### 3.1 INT4 Weight Loading

File: `bench/int4_decode_cgraph.cu` (new, based on `decode_int8_cgraph.cu`)

Changes from INT8 version:
- Load `.int4_t` instead of `.int8_t` for all 7 weight matrices per layer
- Load `.scale_t` files (same format as INT8)
- Call `gemv_int4_warp` instead of `gemv_int8_warp` for Q/K/V/Wo/gate/up/down
- Same fused pipeline (14 kernels/layer): pack_fp4, rmsnorm, swiglu_quant, etc. unchanged
- Same CUDA Graph capture (if capture works — H2D seq_pos still a blocker)

### 3.2 Benchmarks

| Benchmark | INT8 | INT4 (target) | Improvement |
|-----------|------|---------------|-------------|
| M=1 per-kernel | 181.5 t/s | ~250 t/s | +38% |
| M=1 CUDA Graph | 181.2 t/s | ~250 t/s | +38% |
| M=8 batched | 324.6 t/s | ~450+ t/s | +39% |

### 3.3 Correctness

Compare INT4 output vs INT8 output:
- Single-layer pipeline: per-element diff
- Full 28-layer pipeline: aggregate metrics
- Target: max diff < 1% of INT8, no layer-by-layer error compounding

---

## Phase 4: Optional — INT4 GEMV Batch Kernel

### 4.1 `gemv_int4_batched`

For batched M=8 path (gate/up/down GEMVs), adapt `gemv_int8_batched` to INT4:
- Each thread block processes N output rows across M sequences
- Load INT4 weights once per K-block, reuse across M sequences
- Same register pressure issue as INT8 batched — M>8 still limited

---

## Files Changed

| File | Change |
|------|--------|
| `include/blackwell/kernels.h` | Add `gemv_int4_warp` declaration |
| `src/kernels/gemv_int4.cu` | New — INT4 GEMV kernel |
| `CMakeLists.txt` | Add gemv_int4.cu to KERNEL_SOURCES |
| `bench/int4_decode_cgraph.cu` | New — INT4 decode benchmark |
| `scripts/quantize_int4.py` | New — BF16→INT4 weight converter |
| `scripts/validate_int4.py` | New — INT4 quality validator |

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| INT4 quality loss | High — model output degrades | Validate PSNR per-layer; compare against INT8 |
| Dequant overhead eats BW savings | Medium — kernel slower than expected | Profile kernel; use LUT for nibble→int8 |
| File format mismatch | Low — same block structure | Reuse INT8 loading code with uint8_t for packed data |
| No BF16 source weights available | Medium — need conversion from HF safetensors | Use `scripts/convert_weights.py` from existing pipeline |
| INT4 GEMV not faster than INT8 | Low — 50% less data should win | Measure; if not, use INT4 for scales-only optimization |

---

## Timeline Estimate

| Phase | Tasks | Est. Time |
|-------|-------|-----------|
| 1 | Format design + conversion + validation | 1 session |
| 2 | Kernel implementation + testing | 1-2 sessions |
| 3 | Pipeline integration + benchmarks | 1 session |
| 4 | Batch kernel + optional | 0.5 session |
| **Total** | | **3-4 sessions** |
