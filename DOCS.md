# Blackwell INT4 Inference Engine

Custom CUDA kernels for INT4 LLM inference on NVIDIA RTX 5060 Ti (Blackwell GB206, SM 12.0).

**Author:** ronnieops  
**Date:** June 2026  
**Hardware:** RTX 5060 Ti, 16 GB GDDR7, 500 GB/s, 36 SMs  
**CUDA:** 13.3, sm_120a

---

## Architecture

### Quantization Format

INT4 block-16 symmetric quantization:
- Weights: 4-bit offset-binary (nib-8), 2 values per byte, block-16 FP32 scales
- Activations: INT4 same format, quantized per-layer before each GEMV
- Weight format: `[N][K/2]` packed bytes + `[N][K/16]` FP32 scales
- 4.5 GB for 8B model (vs 16 GB FP16 = 3.6× compression)

### Decode Pipeline (per layer)

```
input → RMSNorm → quantize INT4 → GEMV(Q) → GEMV(K) → GEMV(V)
  → head_norm(Q,K) → RoPE(Q,K) → update KV cache → attention
  → quantize INT4 → GEMV(Wo) → residual add
  → RMSNorm → quantize INT4 → GEMV(gate) → GEMV(up)
  → SwiGLU → quantize INT4 → GEMV(down) → residual add → output
```

17 kernel launches per layer × 36 layers = 612 kernels per decode step.

### GEMV Kernel

`gemv_int4_batched` — warp-cooperative INT4 GEMV:
- 1 warp (32 threads) per output row
- Strided K loop: each thread processes K/32 elements
- INT4 unpack: `(nib - 8)` offset-binary, scalar multiply-add
- Warp shuffle reduction for dot product
- Template M=1..16 for batched execution
- ~37 registers/thread, 8 blocks/SM

**Performance:** 421 GB/s effective bandwidth (84% of 500 GB/s peak) for K=N=4096.

### Attention

`attention_decode_batched_gqa` — GQA decode attention:
- 1 block per query head, 128 threads
- Q in shared memory, K/V read from global cache
- Scores computed via warp-cooperative dot product
- Online softmax, weighted sum of V
- Supports M=1..N sequences, GQA ratio up to 4:1

### Prefill

`attention_prefill_v3` — batched causal attention for prompt processing:
- 1 block per (query head, position), 32 threads
- Q in registers, K in shared memory
- Causal mask: Q_m attends to K_0..K_m
- Supports hd=128, M ≤ 16
- Used for prompts ≤ 16 tokens; longer prompts use per-token decode

---

## Performance

### Single Sequence (M=1)

| Config | t/s | ms/tok | vs llama.cpp |
|--------|-----|--------|-------------|
| llama.cpp Q4_K_M | 84 | 11.9 | 1.0× |
| Blackwell INT4 | 56 | 17.9 | 67% |

### Batched (M sequences)

| M | t/s | ms/tok | vs llama.cpp M=1 |
|---|-----|--------|-----------------|
| 1 | 59 | 16.9 | 70% |
| 2 | 75 | 13.3 | 89% |
| 4 | 97 | 10.4 | 115% |
| 8 | 119 | 8.4 | 142% |
| 16 | 138 | 7.3 | 164% |
| 32 | 150 | 6.7 | 179% |
| 48 | 154 | 6.5 | 183% |

### Quality

| Config | PPL (WikiText-2) | vs BF16 |
|--------|-----------------|---------|
| BF16 (llama.cpp Q8_0) | 12.4 | 1.0× |
| INT4 symmetric | 23.52 | 1.9× |
| INT4 + AWQ α=0.6 | 21.82 | 1.76× |

### Runtime Profile (nsys)

| Kernel | % Time | Avg (μs) |
|--------|--------|----------|
| gemv_int4_batched | 92.2% | 54.8 |
| rmsnorm_batched | 3.7% | 7.6 |
| quantize_int4 | 1.3% | 1.3 |
| attn_batched | 0.9% | 3.9 |
| head_norm | 0.5% | 1.1 |

GEMV dominates at 92%. All other kernels combined are <8%.

---

## Server Architecture

```
http_subprocess (C++, httplib)
  │ fork + pipe
  ▼
inference_server_int4 (CUDA, JSON stdio)
  │
  ├── qwen3-8b: 56 t/s, PPL 23.52 ✅
  ├── qwen3-1.7b: 23 t/s, PPL 18.65 ✅
  ├── llama32-1b: 223 t/s ✅
  ├── llama31-8b: garbled (model limitation) ❌
  └── gemma-12b: 24 t/s, needs Python tokenizer ⚠️
```

**Endpoints:**
- `GET /health` — GPU memory, uptime, request count, latency
- `POST /v1/completions` — text completion
- `POST /v1/chat/completions` — chat format
- `POST /v1/batch` — batched completion (M ≤ 8)

**Prefill:** Prompts ≤ 16 tokens processed via batched QKV + per-token attention. Longer prompts use per-token decode.

---

## Abandoned Paths

| Path | Reason | Evidence |
|------|--------|----------|
| INT2 8B | PPL 47B (2B × worse than INT4) | Activation quant accumulation |
| INT4 1.7B | Garbled output | Sub-8-bit quality wall |
| FP4/FP8 | 4.5× slower + worse PPL | No dp4a for FP8 |
| NVFP4 | PPL 24,850 (1000× INT4) | Format mismatch |
| Llama 3.1 INT4 | PPL 273K | Model-specific precision requirement |
| GatedDeltaNet 9B | SSM instability | A_log > 0 → exponential state growth |
| CUDA Graph | 2.1% speedup | GEMV is 92% of runtime |
| Speculative decode | 1.3% acceptance | INT4 draft quality too poor |
| Prefill integration | Cache layout incompatible | Separate cache needed (P0 deferred) |

---

## Key Findings

1. **GEMV is the bottleneck** at 92% of runtime. All other optimizations yield <5%.
2. **INT4 block-16 is the sweet spot** for this hardware. INT2 quality is catastrophic, INT8 is 2× memory.
3. **Batched decode scales well** — M=8 gives 142% of llama.cpp Q4_K_M throughput.
4. **Qwen3 models are quantization-friendly** — survive 36 INT4 layers. Llama 3.1 does not.
5. **TMA is available on GB206** but weight streaming integration is a 2-3 day project.
6. **dp4a SIMD for GEMV**: Warp-cooperative INT4 via dp4a shuffle-reduce. WGMMA (tensor core batched GEMV) is a P2 research direction — not yet implemented.

---

## Build & Run

```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Server
./server/http_subprocess qwen3-8b &
curl -X POST http://localhost:8123/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","max_tokens":30}'

# Benchmarks
./bench/text_generate_int4_qwen3_8b "The capital of France is" 30
./bench/text_generate_int4_batched "The capital of France is" 10 8
./bench/bench_ppl_int4_8b
```

**Docker:**
```bash
docker pull ghcr.io/ronnieops/blackwell-server:v0.10.0
docker run --gpus all -p 8080:8080 \
  -v /path/to/weights_int4_qwen3_8b:/app/weights_int4_qwen3_8b \
  ghcr.io/ronnieops/blackwell-server:v0.10.0 8080 int4_8b
```

---

## Repository Structure

```
src/kernels/
  gemv_int8.cu        — INT4/INT8 GEMV kernels (warp, batched, FP32×INT4)
  decode.cu           — GQA attention, KV cache, RoPE
  attention.cu        — Prefill attention (flash-style, v2, v3)
  fused_rmsnorm.cu    — RMSNorm + quant fusions
  gemm_int8.cu        — WMMA INT8 GEMM (prefill)
  prefill.cu          — Prefill layer orchestration

bench/
  text_generate_int4_qwen3_8b.cu  — INT4 8B text generation
  text_generate_int4_batched.cu    — Batched INT4 decode
  bench_ppl_int4_8b.cu             — PPL benchmark
  bench_gemv_int4.cu               — GEMV throughput benchmark

server/
  inference_server_int4.cu  — INT4 inference daemon
  http_subprocess.cpp       — HTTP wrapper (httplib)
  inference_server_gemma.cu — Gemma 12B server
  inference_server_llama.cu — Llama 3.2 1B server

include/blackwell/
  kernels.h  — Public API (194 kernel symbols)
  bpe_tokenizer.h  — BPE tokenizer
  httplib.h  — HTTP library
```
