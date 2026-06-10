# Blackwell Inference Engine — Architecture

## Overview

Custom CUDA kernels for high-performance INT4 LLM inference on Blackwell GPU (RTX 5060 Ti).

**Key innovations:**
- Warp-cooperative GEMV for INT4 matrix-vector multiplication
- Batched GEMV kernel for M=1-8 sequences
- Fused kernels (RMSNorm + quantize, SwiGLU)
- Device-side seq_pos for CUDA Graph compatibility

## Hardware Target

| Spec | Value |
|------|-------|
| GPU | RTX 5060 Ti 16 GB |
| Architecture | Blackwell (GB206) |
| Compute | SM 12.0 |
| SMs | 36 |
| Memory | 448 GB/s GDDR7 |
| Tensor cores | 2nd gen (FP16/INT8) |

## Kernel Architecture

### INT4 GEMV

Two implementations:

1. **Warp GEMV** (`gemv_int4_warp`)
   - 1 warp per output row
   - dp4a SIMD instructions
   - Shuffle-based reduction
   - ~56 t/s throughput

2. **Batched GEMV** (`gemv_int4_batched`)
   - M sequences × N rows
   - Grid: (N, M)
   - ~63 t/s for M=1 (40% faster than warp for single sequence)

### Attention

**GQA Decode** (`attention_decode_batched_gqa`)
- 32 Q heads, 8 KV heads
- FlashAttention-style tiling
- O(seq × kv_heads) instead of O(seq × q_heads)
- KV cache: [layer][seq][kv_head][head_dim]

### Norm Kernels

**Fused RMSNorm** (`fused_rmsnorm`)
- Warp-cooperative reduction
- Single pass: compute variance + normalize
- ~2μs for H=4096

**Quantize** (`quantize_int4`)
- Block scales (block=16)
- Absmax quantization
- ~1μs for H=4096

## Memory Layout

### Weights (INT4)

```
qkv.weight: [H, Q+K+V] uint4 = [4096, 8192] nibbles
gate.weight: [H, I] uint4 = [4096, 10944] nibbles  
up.weight: [H, I] uint4 = [4096, 10944] nibbles
down.weight: [I, H] uint4 = [10944, 4096] nibbles

Block scale: [N/16] float32
```

### KV Cache

```
d_kc: [NL][max_seq][nkv][hd] float32
d_vc: [NL][max_seq][nkv][hd] float32

NL=36, max_seq=512, nkv=8, hd=128
Total: 36 × 512 × 8 × 128 × 4 = 75 MB per cache
```

### Intermediate Buffers

```
d_x: [H] float32 (hidden state)
d_xi: [H] float32 (normalized)
d_Q/K/V: [Q/K/V] float32 (projections)
d_attn: [Q] float32 (attention output)
d_mlp: [I] float32 (MLP output)
```

## Decode Loop

```
for each token:
    1. Dequantize embedding
    2. For each layer (36):
        a. RMSNorm(input)
        b. Quantize to INT4
        c. GEMV Q, GEMV K, GEMV V
        d. Attention (GQA)
        e. Update KV cache
        f. GEMV output projection
        g. Residual add
        h. GEMV gate, GEMV up
        i. SwiGLU
        j. GEMV down
        k. Residual add
    3. LM head → softmax → sample
```

**Time per token**: ~17ms (58 t/s)

## CUDA Graph

Captures decode loop for reduced kernel launch overhead.

**Status**: Works but no speedup over per-kernel (kernel launches not bottleneck).

**Device-side seq_pos**: Avoids H2D copy in capture.

## Quantization

### INT4 Symmetric

- Block size: 16
- Scale: absmax per block
- Format: nibble = round(x / scale)
- Dequant: x = (nibble - 8) × scale

### AWQ Protection

- Per-output-channel scales
- α=0.6 protection strength
- Scales folded into weight scales

## File Structure

```
src/kernels/
  gemv_int4.cu       — INT4 GEMV (warp, batched)
  decode.cu          — Attention, KV cache, RoPE
  norm.cu            — RMSNorm, quantize
  gemv_int8.cu       — INT8 GEMV (reference)

bench/
  text_generate_int4_batched.cu  — Batched decode benchmark
  profile_decode.cu               — Kernel profiling

server/
  inference_server_int4_batched.cu  — INT4 server
  http_subprocess.cpp               — HTTP wrapper
```

## Performance Breakdown

| Kernel | Time (μs) | % of total |
|--------|-----------|------------|
| GEMV Q/K/V | 4800 | 28% |
| GEMV gate/up/down | 6000 | 35% |
| Attention | 2500 | 15% |
| RMSNorm | 800 | 5% |
| Quantize | 400 | 2% |
| Other | 1500 | 9% |

**Bottleneck**: GEMV (63% of time)

## Optimization Opportunities

1. **Kernel fusion**: Fuse RMSNorm + quantize + GEMV input (saves memory bandwidth)
2. **Warp specialization**: Use specialized warps for different phases
3. **Tensor core GEMM**: For prefill phase (batched token processing)
4. **Continuous batching**: Process multiple requests simultaneously
5. **Speculative decoding**: Draft model for faster token generation

## Testing

```bash
# Quick benchmark
python3 scripts/benchmark_suite.py --quick

# Full benchmark
python3 scripts/benchmark_suite.py

# Throughput test
./bench/text_generate_int4_batched "test" 1 50
```