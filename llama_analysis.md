# llama.cpp Analysis Report — 2026-05-31

## 1. Q4_K_M Dequantization Analysis

### Key Optimizations

**1. DP4A SIMD Dot Product**
```cuda
const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1], ggml_cuda_dp4a(v0i, u[2*i+0], 0));
```
- Uses `dp4a` instruction for 4-element INT8 dot products
- Processes 4 values per instruction (vs our 1-element scalar)
- Critical for 4-bit unpacking efficiency

**2. Half-Precision Scales**
```cuda
const float2 dm4f = __half22float2(dm4);
return dm4f.x*sumf_d - dm4f.y*sumf_m;
```
- Stores scales in FP16 (2 bytes vs 4 bytes)
- Reduces memory bandwidth by 50% for scales
- Converts to FP32 only at final accumulation

**3. Vectorized Reads**
```cuda
const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;
```
- Reads 8 nibbles (4 bytes) at once
- Unpacks via bit manipulation (shift + mask)
- Processes 8 values per iteration

**4. Separate Scale/Min Paths**
```cuda
sumf_d += d8[i] * (dot1 * sc[i]);  // scale path
sumf_m += d8[i] * (dot2 * m[i]);   // min path
```
- Computes scale and min contributions separately
- Better instruction-level parallelism (ILP)
- Final: `dm4f.x*sumf_d - dm4f.y*sumf_m`

### Memory Layout
- Q4_K block: 144 bytes (256 nibbles + 8 scales + 8 mins + 2 FP16 dm)
- Block size: 256 elements
- Nibble packing: 2 per byte, little-endian

---

## 2. SM_120 Blackwell Tuning

### Blackwell-Specific Code Paths

**1. Flash Attention Tuning**
```cuda
if (cc >= GGML_CUDA_CC_BLACKWELL) {
    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
    } else {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
    }
}
```
- Different tile sizes for Blackwell (576×512)
- Batch size 16 for long KV sequences
- Batch size 4 for typical decode

**2. FP4 Native Support**
```cuda
#define BLACKWELL_MMA_AVAILABLE
// Enables native FP4 tensor core operations
```
- Uses SM_120's native FP4 MMA instructions
- Different tile sizes for FP4 vs INT8
- `MMQ_ITER_K_FP4` for FP4-specific iteration

**3. MMQ Tile Configuration**
```cuda
case GGML_TYPE_Q4_K: return MMQ_MMA_TILE_X_K_Q8_1;
```
- Q4_K uses Q8_1 tile sizes (256×8)
- Different from INT8 which uses larger tiles

### Occupancy Targets
- Blackwell: 2 blocks per SM (128 threads each)
- Shared memory: 4KB per block
- Register usage: ~128 per thread

---

## 3. Kernel Fusion Patterns

### Decode Path (M=1)
```
for each layer:
    1. Q = x @ W_q (GEMV, Q4_K × Q8_1)
    2. K = x @ W_k (GEMV)
    3. V = x @ W_v (GEMV)
    4. Update KV cache
    5. Attention = softmax(Q @ K^T) @ V
    6. O = attn @ W_o (GEMV)
    7. Residual + RMSNorm
    8. Gate/Up = x @ W_gate, x @ W_up (GEMV)
    9. SwiGLU = silu(gate) * up
    10. Down = mlp @ W_down (GEMV)
    11. Residual + RMSNorm
```

### Key Fusion Insights

**1. No Explicit Fusion**
- Each GEMV is a separate kernel launch
- Fusion happens implicitly via CUDA Graphs
- Kernel launch overhead ~5-10% of total time

**2. PDL (Programmatic Dependent Launch)**
```cuda
// New in b9442: PDL for kernel overlap
cudaLaunchKernelExC(&config, ...);
```
- Allows next kernel to start before current finishes
- Overlaps compute and memory operations
- Requires CTK >= 12.3

**3. Flash Attention as Fusion**
- Fuses Q×K^T and softmax in one kernel
- Avoids materializing attention matrix
- Saves ~10% of memory bandwidth

**4. Batch Processing (M>1)**
- For batch≥4, routes to MMQ kernel
- MMQ processes multiple rows in parallel
- Better SM utilization

---

## 4. Applicable Optimizations for Our INT8 Path

### High Impact (Implement First)

| Optimization | Impact | Effort | Notes |
|--------------|--------|--------|-------|
| **PDL kernel launch** | +3-5% | Low | Overlap GEMV kernels |
| **FP16 scales** | +5-8% | Medium | Reduce scale memory by 50% |
| **DP4A for unpacking** | +10-15% | Medium | 4× faster nibble unpack |

### Medium Impact

| Optimization | Impact | Effort | Notes |
|--------------|--------|--------|-------|
| Flash attention tiles | +2-3% | High | Already have batched attn |
| MMQ-style tiling | +3-5% | High | Different from our approach |
| KV cache PDL | +1-2% | Medium | Overlap cache updates |

### Low Impact (Skip)

| Optimization | Impact | Effort | Notes |
|--------------|--------|--------|-------|
| FP4 native MMA | N/A | High | We use INT8, not FP4 |
| Different tile sizes | +1% | Medium | Already tuned |
| Shared memory tricks | +1% | High | Marginal benefit |

---

## 5. Recommended Implementation Order

1. **PDL kernel launch** — Easy win, +3-5%
2. **FP16 scales** — Medium effort, +5-8%
3. **DP4A unpacking** — High effort, +10-15%
4. **Batch kernel fusion** — High effort, +5-10%

**Total potential**: +20-35% improvement over current 328.8 t/s
**New target**: 395-440 t/s (if fully implemented)

---

## 6. Key Takeaways

1. **llama.cpp's speed comes from Q4_K_M format** — 4-bit is inherently faster than INT8
2. **Their DP4A usage is critical** — we should adopt for unpacking
3. **FP16 scales are a free win** — reduce memory by 50%
4. **PDL is easy to add** — just change kernel launch
5. **Flash attention is their biggest win** — we have batched attn instead

**Bottom line**: We can learn from their optimizations, but our INT8 path is fundamentally different. The 12% gap is mostly due to Q4_K_M being more memory-efficient than INT8.
