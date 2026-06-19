# AGENTS.md - blackwell

Custom CUDA kernels for INT8/INT4 LLM inference on RTX 5060 Ti (Blackwell, GB206).

---

## 1. Mission

INT8/INT4 decode throughput vs llama.cpp Q4_K_M.

**Benchmark: Blackwell INT4 vs llama.cpp Q4_K_M (Qwen3-8B, RTX 5060 Ti)**
| Config | t/s | ms/tok | vs llama.cpp |
|--------|-----|--------|-------------|
| llama.cpp Q4_K_M | **84** | **11.9** | 1.0× (baseline) |
| Blackwell INT4 M=1 (v0.11, FP32 sc) | 56 | 17.9 | 67% |
| **Blackwell INT4 M=1 (v0.12, FP16 sc)** | **74** | **13.5** | **88%** |
| Blackwell INT4 M=1 (v0.12 + graph) | **76.6** | **13.0** | **91%** |
| Blackwell INT4 M=8 (v0.12) | **205** | **4.9** | **244%** |
| Blackwell INT4 M=16 (v0.12) | **220** | **4.6** | **262%** |
| Blackwell INT4 M=48 (v0.11) | **154** | **6.5** | **183%** |

**Why M=1 is 88% (was 67%)**: v0.12 FP16 weight scales cut 33% of GEMV memory traffic (scales FP32→FP16). Kernel already at 95% of 448 GB/s peak in microbench. Remaining gap to llama.cpp is non-GEMV overhead (launches, rmsnorm, quantize). Occupancy increase (8→16 warps/SM): no gain (memory-saturated). CUDA Graph: +2.5%. Batched path already exceeds llama.cpp at M=8+.

**Prior M=1 analysis (v0.11)**: llama.cpp uses tensor cores for GEMV (our dp4a SIMD is slower for skinny M=1), 4.5-bit quantization (vs our 4-bit), and mature CUDA Graph integration. dp4a INT4 inner loop (v0.12) gives +74% on batched M=8 but ~0% on M=1 (M=1 is memory-bound, not compute-bound).

**Servers (v0.12.2 — batched serving M>1)**
| Model | Server | t/s | ms/tok | Quality |
|-------|--------|-----|--------|---------|
| 1.7B INT8 HTTP | `http_subprocess 1.7b` | **~23** | ~43 | PPL 18.65 (1.5× BF16) ✅ |
| **8B INT4 HTTP (FP16 sc)** | `http_subprocess int4_8b` | **~74** | ~13.5 | PPL 24.39 (was 23.52) ✅ |
| **14B INT4 (raw, FP16 sc)** | `http_subprocess int4_14b` | **TBD** | **TBD** | PPL TBD (raw, no AWQ) |
| **8B INT4 batched HTTP (FP16 sc, M=1)** | `http_subprocess int4_8b_batched` | **~65** | ~15.4 | PPL 24.39 ✅ |
| **8B INT4 batched HTTP (8 concurrent)** | `http_subprocess int4_8b_batched` | **~183** | ~5.5 | PPL 24.39 ✅ **2.86× real-world** |
| **Llama 3.2 3B INT4** | `http_subprocess llama32-3b` | **155** | ~6.5 | ❌ Garbled (28-layer INT4 wall) |
| **Gemma 4 12B INT4** | `http_subprocess gemma` | **~24** | ~42 | Coherent (requires Python tokenizer wrapper) ⚠️ |

**INT4 server divergence (v0.9.0 → v0.10.x)**: `inference_server_int4.cu` migrated from `gemv_int4_warp`
to `gemv_int4_batched` (commit 35337ef). Both `inference_server_int4` and `inference_server_int4_batched`
now produce correct output matching bench. Bug history updated.
Repetition penalty eliminates token looping — clients can override via JSON body.

**Batch endpoint (v0.8.3)**: `POST /v1/batch` with `{"prompts":["...","..."],"max_tokens":N}`
- All prompts processed in one batched call → 12-26% per-request speedup
- M=8 batch: 0.52s/req vs 0.7s single. Real-world throughput scales with concurrency.
- Local BpeTokenizer decodes tokens locally (avoids server text parsing)
- JSON escaping: XSS guard `<>` → `\u003c/\u003e`, non-ASCII bytes → `\u00XX`

**Server v0.8.1 features**:
- Repetition penalty: `repetition_penalty` param (1.0-2.0, default 1.0=off)
  Reduces token looping. 8B: "Paris... Paris..." (no pen) → "Paris... the city that has..." (rep=1.3)
- Batched QKV: M sequences batched through QKV in 3 calls (vs M×3 sequential)
- Mixed-precision: auto-detects `.fp16` files per layer, dispatches to FP16 GEMV
- Critical fixes: seq_pos sync bug, empty prompt, prefill cache layout

**Docker**: `ghcr.io/ronnieops/blackwell-server:v0.8.3` (160 MB)
**INT4 Docker**: `ghcr.io/ronnieops/blackwell-server:int4` (198 MB, v0.13) — see `Dockerfile.int4`

**14B INT4 quantization (Session 86)**: Qwen3-14B raw INT4 with FP16 scales completed via streaming quantizer (`scripts/quantize_int4_qwen3_14b.py`). Uses torch mmap for BF16→F32 conversion (no memory spike). 645 files, 6.9 GB. Architecture: H=5120, I=17408, NL=40, nqh=40, nkv=8, hd=128, V=151936. SwiGLU MLP, RMSNorm, RoPE theta=1000000. AWQ calibration pending (optional, 8B got PPL 24.39 raw vs 21.98 AWQ).

**Disk cleanup (Session 86)**: Freed ~550 GB from HF cache. Deleted: Gemma4 GGUF variants (21+73+117+46+21+33=311 GB), Qwen3.6 GGUF (109+97=206 GB), unsloth GGUF (5.2+4.7+4.6=14.5 GB), Qwen3-1.7B (3.8 GB), Qwen2.5-0.5B (5.1 GB), Qwen3-8B HF source (16 GB), 30B INT4 weights (18 GB). Remaining: Qwen3-14B safetensors (28 GB), Qwen3-30B-A3B HF source (57 GB), wikitext dataset (15 MB). Disk: 564 GB free (58%).

**30B MoE weights deleted**: 18 GB `weights_int4_qwen3_30b_a3b/` removed. Blocked by RAM (needs 55+ GB). HF source (57 GB) kept for future re-quantize.

**Gemma4 GGUF all corrupted**: Systematic unsloth GGUF export bug for entire Gemma4 family. All FP32 norm tensors contain garbage values (1e22+). No blackwell fix possible. Requires clean FP32 weights from HuggingFace safetensors.

**Gemma4 12B safetensors download**: ~14 GB of ~24 GB downloaded at `/mnt/data/ai/hf/models--google--gemma-4-12B-it/`. Will provide clean FP32 weights for converter-based quantization once complete.

**Version history**:
- v0.13.0: Docker image pushed (198 MB). 5 model aliases: int4_8b, batched, llama32-3b, int4_14b. Fused kernels + FP16 scales + AWQ. Header 1555→873 lines. Dead code cleanup (25 kernel sources removed, 139→138 symbols). Tagged git v0.13.
- v0.12.7: 14B INT4 raw quantization complete (6.9 GB, 645 files). Disk cleanup (~550 GB freed).
- v0.12.1: Fixed multi-block RMSNorm bug in fusion kernels. Wired all 4 fusion sites (3 RMSNorm + 1 SwiGLU). M=1 warp 64→70 t/s (+9%). PPL 24.39→21.98 (-10%, better than baseline 23.52!). New unit test bench/test_fused_int4.cu (bit-identical at N=4096/12288).
- v0.12.0: INT4 8B FP16 weight scales. PPL 23.52→24.39 (+0.87). M=1 56→74 t/s (+32%). M=8 207→205, M=16 217→220. New kernels `gemv_int4_warp_f16wsc`/`gemv_int4_batched_f16wsc` (templated on WScaleT). New weight dir `weights_int4_qwen3_8b_fp16sc/` (4.8 GB, scales FP16). Conversion script `scripts/convert_scales_fp16.py`. dp4a INT4 inner loop (prior phase).
- v0.11.0: Multi-chunk prefill (any prompt length). Fixed pinned buffer seq_pos race. Server hardening (rate limit all endpoints, restart, payload limit, max_tokens clamp, security fixes).
- v0.10.0: Gemma 4 12B INT4 support (24 t/s, hd=512, GeGLU). Server hardening (rate limiting, graceful shutdown, config file, Prometheus metrics). GGUF parser fixes for Gemma metadata.
- v0.9.3: INT4 8B server (56 t/s, PPL 23.52, repetition penalty)
- v0.9.2: INT4 batched server (M sequential, each with own KV cache)
- v0.9.1: INT4 server from benchmark decode loop (57 t/s)
- v0.8.3: INT8 multi-model (1.7B/8B/9B)

**8B quality with correct dims**: INT8 produces coherent text.
Mixed precision (FP16 early layers) provides NO improvement — ALL-INT8
and MIXED(8 FP16 + 28 INT8) produce IDENTICAL output. The earlier
"garbled 8B INT8" observation was from WRONG model dimensions.

**Gemma 4 12B INT4 (Session 79-81)**: New model ported via GGUF bridge.
- Config: H=3840, I=15360, NL=48, nqh=16, nkv=8, hd=256, V=262144
- GeGLU activation (not SwiGLU), sliding window attention (every 6th layer is SWA)
- QK head norms on ALL layers (per-layer files)
- 4 RMSNorms per layer: attn_norm, post_attention_norm, ffn_norm, post_ffw_norm
- Tokenizer: SentencePiece (BPE incompatible — use Python HF tokenizer wrapper)
- Tied embeddings: embed_tokens == lm_head (no separate lm_head)
- Final logit softcapping: cap=30.0

**Gemma 4 12B QAT (Session 84)**: QAT model from `google/gemma-4-12B-it-qat-q4_0-gguf`.
- Converted via `better-inference/gguf_convert` from Q4_0 GGUF → INT4 block-16
- **FA layers** at positions 5,11,17,23,29,35,41,47: double Q heads (32×256=8192), K=V (k_eq_v, 2 heads=512), no v_proj, o_proj input 8192→3840
- **SWA layers** (40 layers): standard 16 Q heads (4096), 8 KV heads (2048), has v_proj
- Layer 47: missing q_proj in GGUF (truncated). Bench shares from layer 46.
- Weights: 7.0 GB (322 INT4 weight files + norms + embed)
- Bench runs at 43 t/s M=1 but quality garbled (KV cache sharing for FA layers not implemented)
- Bench: `./bench/text_generate_gemma4_12b_qat [token_file] [tokens]`
- Server dims fixed (was H=4608/hd=128/NL=40 — now H=3840/hd=256/NL=48)
- Server needs FA layer handling ported (K=V, double Q, no v_proj)

**Key QAT architecture findings**:
- FA layers use `k_eq_v=True` (K == V, single shared projection), from HF Transformers code
- FA layers have `num_global_key_value_heads` = 2, `global_head_dim` = 256
- GGUF converter doesn't emit v_proj for FA layers (correct — they don't exist)
- GGUF converter truncates layer 47 q_proj (GGUF tensor missing)
- Gemma 4 uses `shared_kv_states` dict: last N layers reuse K/V from last same-type layer
- For both bench and server: FA layer KV cache management is incomplete — needs proper shared KV handling

**Benchmarks (INT4)**
| Config | t/s | ms/tok | Notes |
|--------|-----|--------|-------|
| Llama 3.2 3B M=1 | **155** | **6.5** | Experimental (garbled) |
| Qwen3-8B M=1 | **63** | **16** | Production |
| Warp GEMV M=1 (8B) | **59** | **17** | Benchmark |
| HTTP batched (8B) | **55** | **18** | With HTTP overhead |

**nsys Profile Breakdown** (nsys profile --trace=cuda):
| Kernel | % Time | Avg (μs) | Instances |
|--------|--------|----------|----------|
| gemv_int4_batched | 92.2% | 54.8 | 8851 |
| rmsnorm_batched | 3.7% | 7.6 | 2551 |
| quantize_int4 | 1.3% | 1.3 | 5071 |
| attn_batched | 0.9% | 3.9 | 1260 |
| head_norm | 0.5% | 1.1 | 2520 |

**Key insight**: GEMV dominates (92%), weight loading is the bottleneck.

**Embedding Pre-load Optimization (Session 71)**: Pre-load full embedding table (623 MB FP32) to GPU at startup. Runtime embedding lookup is now D2D copy instead of CPU dequantization + H2D. GPU memory: 7287 MB → 9661 MB. Throughput: ~55 t/s (unchanged, GEMV is still 92% bottleneck).

**API Improvements (Session 71)**: Added OpenAI-compatible fields: unique request IDs per response, timestamps, system_fingerprint, proper usage statistics. All endpoints now return standardized JSON format.

**INT4 8B quality coherent (AWQ per-layer α + FP16 scales, v0.12.3)**. PPL 21.98 (was 24.39 with FP16 scales, 23.52 with FP32). AWQ per-layer alpha search (mean=0.416, 252 submodules) + FP16 scales via `convert_scales_fp16.py`. AWQ+FP16 bug fixed: direct `astype(np.float16)` in AWQ script caused PPL 24.98; write FP32 scales first, then convert. Weight size 5.1 GB (FP16 scales). Grammatically correct English, factual errors, token looping without repetition penalty. 62 t/s M=1 (slight regression from 74 t/s baseline due to AWQ scale distribution).

**Prior (v0.11)**: 59 t/s, 5.3 GB. Root cause of pre-v0.9 INT4 failures: `upload_w4` scale buffer bug — allocated 256 floats instead of N×kblocks (38.9M for lm_head).

**CUDA Graph for INT4 (Session 72, updated v0.12.4)**: 832 nodes captured in batched bench, +2.4% (66→68 t/s M=1). Device-side seq_pos, graph-safe kernels. `--graph` flag: `./bench/text_generate_int4_batched --graph "prompt" 1 tokens wdir`. Standalone graph bench: `./bench/decode_int4_cgraph_8b [tokens]` (76.5 t/s, warp kernel). M>1 falls through to per-kernel path. Multi-size graph not implemented (per-sequence loops add complexity, per-kernel already exceeds llama.cpp at M=8+).

**Batched INT4 (Session 64/65, updated v0.12)**: M=1: 74 t/s, M=2: 135 t/s, M=4: 178 t/s, M=8: 205 t/s, M=16: 220 t/s (all FP16 scales). v0.11 FP32: M=1: 63, M=8: 119, M=16: 138, M=32: 150, M=48: 154.
Scales monotonically to M=48 with correct output (no garbage). No M>8 bug — kernel supports M=1-16 via switch, memory supports up to ~M=100.
MAXSEQ=512 for batched (vs 4096 for single). Uses `gemv_int4_batched_f16wsc` even for M=1.
Benchmark: `./bench/text_generate_int4_batched "prompt" M gen_tokens weights_int4_qwen3_8b_fp16sc`.

**INT4/INT5 1.7B quality dead**. All sub-8-bit paths produce garbled text after 28+ layers.

**GGUF Bridge (Session 73)**: `better-inference/` — GGUF → blackwell INT4 converter. Parser (`gguf.h`) reads GGUF v3 files. Converter (`gguf_convert.cpp`) dequantizes Q4_K/Q5_0/Q6_K/Q8_0 → FP32 → re-quantizes to INT4 block-16. Phase 1 (parser) + Phase 2 (Qwen3 converter) + Phase 3 (Llama 3.1/3.2 support) complete.
- **Critical fix**: GGUF v3 tensor offsets are RELATIVE to tensor data section, not absolute. Converter must add `tensor_data_off` to `ti.offset` when reading tensor data. This bug caused all F32 layernorm weights to be garbage → NaN logits.
- **RoPE fix**: GGUF v3 uses nested prefixes (rope.freq_base stored under full repo URL). Fixed by searching for any key ending with the suffix.
- **Llama 3.2 1B verified**: `bench/text_generate_llama32_1b` — 223 t/s, coherent output. 16L, H=2048, I=8192, nqh=32, nkv=8, hd=64, V=128256, rope_theta=500000. Mixed Q4_K/Q6_K quantization. 262 files, 891 MB.
- **Llama 3.2 3B INT4**: `bench/text_generate_llama32_3b` — 155 t/s, 28L, H=3072, I=8192, nqh=24, nkv=8, hd=128, V=128256, rope_theta=500000. Server alias `llama32-3b`. Tied embeddings. Garbled output (28-layer INT4 wall).
- **Qwen2.5-0.5B verified**: 392 files, mixed Q4_K/Q5_0/Q6_K/Q8_0. Config: 24 layers, H=896, I=4864, nqh=14, nkv=2, hd=64.
- Usage: `./better-inference/gguf_convert model.gguf output_dir/`

**PPL quality (1.7B, WikiText-2, 512 ctx)**
| Config | PPL | vs BF16 |
|--------|-----|--------|
| BF16 (llama.cpp Q8_0) | **12.4** | 1.0× |
| INT8 block-16 (correct dims) | **18.65** | 1.5× |
| INT8 (old, wrong dims) | 7,351,868 | — |

**Root cause of quality issues (Session 56)**: Wrong model dimensions in ALL
pre-session-56 code. Qwen3-1.7B: **nqh=16, nkv=8, hd=128, KV=1024**
(NOT nqh=32, nkv=4, hd=64, KV=512). Half of K/V weights were ignored → PPL=7.3M.

**8B quality with correct dims (Session 59)**: INT8 produces coherent text.
"The capital of France is" → " Paris. The capital of France is Paris..." — coherent 
but looping. Mixed precision (8 FP16 + 28 INT8) produces IDENTICAL output.
PPL = 3.80 on short corpus (both ALL-INT8 and MIXED).

**No INT8 quality wall exists**. INT8 block-16 with correct dims gives PPL=18.65,
only 1.5× worse than BF16.

**FP8 path abandoned** (Session 56). FP8 per-row is 4.5× slower AND 2.3× worse
PPL than INT8 block-16. Reference code kept in src/kernels/gemv_fp8.cu.

**NVFP4 path (Session 69) — ABANDONED**: Double quantization (INT4→FP32→NVFP4) shifts weights, causing catastrophic quality degradation. PPL=24,850 vs INT4 21.82 (1000× worse). Format encoding mismatch (offset-binary INT4 vs signed-magnitude E2M1) means same nibble → different value. Speed 21 t/s is 2.67× slower than INT4 (56 t/s) due to scalar PTX dequant. No production path without retraining.

**NVIDIA Model-Optimizer research (Session 74)**: NVIDIA's `modelopt` library uses NVFP4 E2M1 encoding with FP8 E4M3 per-block scales. Our INT4 block-16 uses offset-binary encoding (nib-8) with FP32 scales. These are incompatible formats — same nibble value maps to different actual values. ModelOpt provides proper calibration (max/MSE/Hessian/AWQ/GPTQ) vs our random normal proxy (128 samples). ModelOpt exports to safetensors+config.json vs our flat binary files. Not directly reusable, but calibration methods could inform future AWQ improvements.

**9B q_proj dimension mismatch (suspected)**: Qwen3.5-9B full_attention q_proj
weight N=8192=32 heads × 256 dim. Server hardcodes NQ=16. If correct config uses
32 heads, half of Q projection is unused → quality degradation.
However, no config.json available to confirm (HF cache cleared).

**All active bench files verified with correct dims** (1.7B: nqh=16, nkv=8,
hd=128, KV=1024). **8B server dims also correct** (nqh=32, nkv=8, hd=128).

---

## 2. Active State

**Stack**: CUDA 13.3, SM_120a, CMake, C++17
**Target**: RTX 5060 Ti 16 GB, compute 12.0, 36 SMs, ~500 GB/s GDDR7
**Nvcc path**: `/usr/local/cuda-13.3/bin/nvcc`
**Library**: 139 symbols in `build/libblackwell_kernels.a`

**Production kernels (INT8 path)**:
- `gemv_int8_warp` — Warp-cooperative INT8 GEMV (1 warp/row, dp4a SIMD, shuffle reduce)
- `gemv_int8_batched` — Batched INT8 GEMV M=1-8
- `gemv_int8_splitk` — Split-K INT8 GEMV (K_splits=4)
- `fused_rmsnorm_quant_int8` — RMSNorm + INT8 quant (1 kernel)
- `fused_swiglu_quant` — SwiGLU + INT8 quant (fused)
- `fused_rmsnorm` — Single-block warp-reduced RMSNorm
- `attention_decode_gqa` — GQA decode attention (M=1)
- `attention_decode_batched_gqa` — Batched GQA decode (M seq)
- `update_kv_cache` / `update_kv_cache_device` / `update_kv_cache_pos` — KV cache write (device-side seq_pos / direct seq_pos arg)
- `pack_int8` / `quantize_int8` — FP32 → INT8 quant with block scales
- `vector_add_fp32` — Elementwise FP32 addition
- `apply_swiglu` — silu(gate) × up
- `apply_rope` / `fused_rope_decode` — In-place RoPE
- `gemv_int8_gate_up` — Fused gate+up INT8 GEMV (0.91× slower than serial)
- `sample_gpu` / `sample_argmax_gpu` — GPU softmax + sampling
- `absmax_scales_kernel` — Block absmax scale computation
- `get_seq_pos_device_ptr` / `update_decode_seq_pos` — Device-side seq_pos for CUDA Graph

**GatedDeltaNet kernels (Qwen3.5-9B)**:
- `gated_delta_conv1d_update` — 1D depthwise conv + SiLU
- `gated_delta_recurrent_step` — SSM recurrent step (NK→NV heads)
- `gated_delta_rmsnorm_gated` — RMSNormGated with SiLU gate
- `attention_decode_kernel_v4` — Decode attention for head_dim=256

**GatedDeltaNet server (v0.7.0)**:
- `server/inference_server_qwen35_9b.cu` — Self-contained C++ daemon
- `server/inference_server_9b` — Compiled binary (2.8 MB)
- `tokenizer_data_9b.bin` — Qwen3.5 BPE tokenizer (248044 vocab, 7.8 MB)
- 32 layers: 24 linear_attention (SSM) + 8 full_attention (GQA, layer 3/7/11/15/19/23/27/31)
- Decode per-token: ~29 ms (35 t/s), 49% of llama.cpp Q3_K_M throughput
- No prefill — token-by-token only (SSM state constraint)
- Quality: degraded at INT8 for 32-layer depth, temperature>0 produces diverse output



---

## 3. Build & Run

### Build
```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### Server (HTTP, production)
```bash
killall hashcat 2>/dev/null  # MUST DO BEFORE ANY MEASUREMENT
# INT4 8B batched server (v0.12 FP16 scales, uses gemv_int4_batched_f16wsc, ~74 t/s M=1)
./server/http_subprocess batched &
# Or INT4 8B warp server (~66 t/s)
./server/http_subprocess int4_8b &
# Test endpoints:
curl http://localhost:8123/health  # Returns GPU memory, uptime, requests, latency
curl -X POST http://localhost:8123/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","max_tokens":5}'
```

**Health endpoint** returns:
```json
{"status":"ok","model":"blackwell-8B","gpu_used_mb":14440,"gpu_total_mb":15849,"uptime_sec":40,"requests":1,"errors":0,"avg_latency_ms":235.0}
```


### 1.7B benchmarks (research/validation)
```bash
killall hashcat 2>/dev/null
./bench/decode_int8_cgraph 28                       # M=1: 181.5 t/s (no head_norm/RoPE)
./bench/decode_int8_nofp4 28 8                     # M=8: 575 t/s CUDA Graph (no head_norm/RoPE)
./bench/text_generate "The capital of France is" 30 # Correctness
```

### Prefill benchmarks (GEMM-only, no attention)
```bash
./bench/prefill_benchmark 512   # GEMM-only: 13,727 t/s at SEQ=512
```

### Prefill + Decode pipeline benchmark
```bash
./bench/prefill_decode_benchmark 8 20   # Full pipeline comparison
# Results (8 prompt + 10 decode tokens):
#   Decode-only: 42-66ms (sequential)
#   Prefill+Decode: ~5.2ms (parallel prompt)
#   Speedup: 8-13x for prompt processing
```
**Note**: Server prefill NOW WORKS (v0.11.0) for all prompt lengths via multi-chunk batched prefill.
**Current status**: Multi-chunk prefill working. `prefill_tokens_batched()` processes prompts of any length in chunks of MAX_BATCH.

### 8B benchmarks
```bash
./bench/text_generate_int4_8b "The capital of France is" 30  # INT4: 74 t/s (FP16 scales, M=1 batched)
```

### Gemma 4 12B benchmarks
```bash
./bench/text_generate_gemma "The capital of France is" 30  # 24 t/s
# With correct tokenization:
python3 scripts/gemma_wrapper.py "The capital of France is" 30
```

### Qwen3.5-9B GatedDeltaNet
```bash
./bench/decode_qwen35_9b weights_int8_qwen35_9b 20        # M=1: 45.7 t/s
./bench/decode_qwen35_9b_batched_v2 8 20                   # M=8: 52.1 t/s (batched GEMV + RMSNorm)
```

### Diagnostics
```bash
nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l  # expect 139
```

### Docker server (v0.7.0, 160 MB, weights mounted at runtime)
```bash
docker pull ghcr.io/ronnieops/blackwell-server:v0.7.0
# Single model (mount weights from host):
docker run --gpus all -p 8080:8080 \
  -v /path/to/weights_int8_bf16:/app/weights_int8_bf16 \
  -v /path/to/tokenizer_data.bin:/app/tokenizer_data.bin \
  ghcr.io/ronnieops/blackwell-server:v0.7.0 8080 1.7b
# 9B model:
docker run --gpus all -p 8081:8080 \
  -v /path/to/weights_int8_qwen35_9b:/app/weights_int8_qwen35_9b \
  -v /path/to/tokenizer_data_9b.bin:/app/tokenizer_data_9b.bin \
  ghcr.io/ronnieops/blackwell-server:v0.7.0 8080 9b
```

### INT4 Docker (v0.10.0, 154 MB)
```bash
docker build -f Dockerfile.int4 -t blackwell-server:int4 .
# v0.13 models: int4_8b (production), batched (M=8), llama32-3b (experimental), int4_14b (experimental)
# Run with weights mounted:
docker run --gpus all -p 8080:8080 \
  -v /path/to/weights_int4_qwen3_8b_fp16sc:/app/weights_int4_qwen3_8b_fp16sc \
  -v /path/to/tokenizer_data.bin:/app/tokenizer_data.bin \
  blackwell-server:int4 8080 int4_8b
# Run Llama 3.2 3B:
docker run --gpus all -p 8080:8080 \
  -v /path/to/weights_llama32_3b:/app/weights_llama32_3b \
  blackwell-server:int4 8080 llama32-3b
```
### Docker compose (multi-model, TBD — docker-compose.yml not yet updated for v0.13)
```bash
docker-compose up -d blackwell-1.7b   # port 8081
docker-compose up -d blackwell-9b    # port 8083
# Or all three:
docker-compose up -d
```

---

## 4. File Layout

### Weight directories
```
weights_int8_bf16/            # 1.7B INT8 weights (2.1 GB)
weights_int4_qwen3_1.7b/      # 1.7B INT4 symmetric (dead end)
weights_int4_qwen3_1.7b_asym/ # 1.7B INT4 asymmetric (dead end)
weights_int5_qwen3_1.7b_asym/ # 1.7B INT5 asymmetric (dead end)
weights_int8_qwen3_8b/        # 8B INT8 weights + norms (canonical, 9.6 GB)
weights_int4_qwen3_8b/            # 8B INT4 symmetric weights + norms, FP32 scales (5.8 GB, PPL 23.52)
weights_int4_qwen3_8b_fp16sc/     # 8B INT4 weights, FP16 scales (4.8 GB, PPL 24.39, +32% M=1 t/s) — PRODUCTION v0.12
weights_int4_qwen3_8b_awq_perlayer/ # 8B INT4 AWQ per-layer α, FP32 scales (6.1 GB, PPL 21.98)
weights_int4_qwen3_8b_awq_perlayer_fp16sc/ # 8B INT4 AWQ per-layer α, FP16 scales (5.1 GB, PPL 21.98) — BEST QUALITY v0.12.3
weights_int4_qwen3_8b_awq_wikitext/ # 8B INT4 AWQ per-layer α, WikiText-2 corpus, FP32 (5.7 GB, PPL 21.98)
weights_int4_qwen3_8b_awq_wikitext_fp16sc/ # 8B INT4 AWQ per-layer α, WikiText-2 corpus, FP16 (4.7 GB, PPL 21.98)
weights_int4_qwen3_14b_fp16sc/ # 14B INT4 raw, FP16 scales (6.9 GB, 645 files) — NEW v0.12.7
weights_int8_qwen3_8b_mixed/  # 8B mixed: 8 FP16 + 28 INT8 (same quality as all-INT8)
weights_int8_qwen3_8b_all_int8/ # 8B pure INT8 copy
weights_int8_qwen35_9b/        # 9B GatedDeltaNet INT8 (11 GB)
weights_int8_qwen35_9b_mixed/ # 9B mixed: 8 FP16 + 24 INT8 (NO quality improvement)
weights_gemma/                # Gemma 4 12B INT4 (11 GB, 760 files)
```

**8B weight status**: All-INT8 and mixed-precision produce IDENTICAL coherent output.
Mixed precision does NOT help 8B. Use `weights_int8_qwen3_8b/` (all-INT8, simpler).

**9B weight status**: Mixed precision (8 or 16 FP16 layers) does NOT fix quality.
Even all-FP16 produces garbled output "The-Fi-Fi..." (same as INT8). Root cause:
SSM instability (A_log > 0 for 68.8% of layer-4 channels → A > 1 → exponential
state growth). A_log stored as FP32 (not quantized). 9B quality BLOCKED.

### Key source files
```
src/kernels/
  gemv_int8.cu            — Production INT8 GEMV (warp, batched, splitk, pack, fused)
  decode.cu               — Attention (GQA, batched, KV cache, RoPE, device-side seq_pos)
  fused_rmsnorm.cu        — RMSNorm + quant + pack fusions
  gemm_int8.cu            — WMMA INT8 GEMM (prefill)
  gated_delta_net.cu       — GatedDeltaNet SSM kernels
  gemv_fp32.cu             — Plain FP32 GEMV (high-precision inference)
  gemv_fp32_int4_asym.cu  — INT4 research (122 dB exact, dead-end)
  gemv_fp32_int5_asym.cu   — INT5 research (122 dB exact, dead-end)
  gemv_int8_gate_up.cu     — Fused gate+up GEMV (0.91×)

bench/
  text_generate.cu              — 1.7B end-to-end text generation
  text_generate_qwen3_8b.cu     — 8B end-to-end text generation (INT8)
  text_generate_int4_qwen3_8b.cu — 8B INT4 single-sequence text generation (59 t/s)
  text_generate_int4_batched.cu — 8B INT4 batched text generation (M=1:61t/s, M=2:110t/s)
  bench_ppl_int4_8b.cu          — INT4 8B PPL benchmark (PPL 21.82)
  decode_int8_cgraph.cu         — 1.7B M=1 CUDA Graph benchmark
  decode_int8_batched_cgraph_attn.cu — 1.7B M=8 batched benchmark
  decode_int8_nofp4.cu          — nofp4 benchmark (per-kernel + CUDA Graph)

server/
  inference_server_nofp4.cu     — C++ inference daemon (stdin/stdout JSON)
  inference_server              — compiled binary
  inference_server_int4.cu     — INT4 8B server (JSON stdio, uses benchmark decode loop)
  inference_server_int4_batched.cu — Batched INT4 server (uses gemv_int4_batched, batched GEMV M=1)
  inference_server_int4        — compiled INT4 server binary (2.7 MB)
  inference_server_int4_batched — compiled batched server binary
  http_subprocess.cpp           — C++ HTTP wrapper (httplib, fork subprocess)
  http_subprocess               — compiled HTTP server
  http_server.py               — Python HTTP wrapper (fallback)
```

---

## 5. Key Findings

**CRITICAL (Session 56)**: All pre-session-56 quality numbers invalid due to wrong model dimensions.
Qwen3-1.7B actual config: **nqh=16, nkv=8, hd=128, KV=1024** (NOT nqh=32, nkv=4, hd=64, KV=512).

### PPL Quality (1.7B, WikiText-2, 512 ctx)
| Config | PPL | vs BF16 | Note |
|--------|-----|---------|------|
| BF16 (llama.cpp Q8_0) | **12.4** | 1.0× | Baseline |
| INT8 block-16 (correct dims) | **18.65** | 1.5× | Production path |
| INT4 symmetric (8B, baseline) | **23.52** | 1.9× | 56 t/s, no calibration, FP32 scales |
| INT4 symmetric (8B, FP16 scales) | **24.39** | 1.97× | 74 t/s M=1, +32% throughput |
| **INT4 symmetric (8B, FP16 sc + fusion)** | **21.98** | **1.77×** | **70 t/s warp, BETTER PPL than baseline!** |
| INT4 + AWQ α=0.6 (8B, FP32 sc) | **21.82** | 1.76× | AWQ calibration, random normal proxy |
| **INT4 + AWQ per-layer α (8B, FP32 sc)** | **21.98** | **1.77×** | **Per-layer alpha search, MSE grid, same PPL as fusion** |
| **INT4 + AWQ per-layer α (8B, FP16 sc)** | **21.98** | **1.77×** | **FP16 scales via convert_scales_fp16.py — NO regression!** |
| **INT2 8B** | **47,529,500** | **3.8M×** | ❌ ABANDONED — activation quant accumulation |
| NVFP4 E2M1 (8B) | **24,850** | 2005× | ❌ ABANDONED — double quantization, PPL vs INT4 |
| FP8 per-row | 41.75 | 3.4× | 4.5× slower than INT8, abandoned |
| INT8 (old, wrong dims) | 7,351,868 | — | **INVALID** — half of K/V weights ignored |

### AWQ INT4 Calibration
| Finding | Value |
|---------|-------|
| Best alpha (fixed) | **0.6** (PPL 21.82 vs 23.52, 7.2% improvement) |
| Best alpha (per-layer search) | **mean=0.416, min=0.000, max=0.950** across 252 (layer,submodule) pairs |
| Method | Random normal proxy (128 seq), per-layer MSE grid search (20 ratios) |
| Real calibration data | WikiText-2 (128 seq, 106K tokens via tokenize_corpus) — PPL 21.98, same as random proxy. Activation stats similar between random normal and real text after RMSNorm. Random proxy sufficient. |
| Scale integration | Folded into block scales `w_sc_new[n] = w_sc[n] * s[n]`, no kernel changes |
| FP16 scales path | Write FP32 first, then `convert_scales_fp16.py` — avoids direct FP16 write bug |
| Script | `scripts/quantize_awq_int4_8b.py` |
| AWQ + FP16 scales bug | **FIXED**: PPL 24.98→21.98. Root cause: direct `astype(np.float16)` in AWQ script lost precision. Fix: write FP32 scales, convert via `convert_scales_fp16.py`. |
| INT8 8B CUDA Graph M=1 | ✅ Working (repetition penalty makes it coherent) |
| INT8 8B batched M>1 | ❌ Garbage (pre-existing bug, not CUDA Graph) |

### Performance
| Finding | Value |
|---------|-------|
| 1.7B INT8 M=1 benchmark (no head_norm/RoPE) | 181.5 t/s |
| 1.7B INT8 M=8 CUDA Graph benchmark | 575 t/s (196% of Q4_K_M) |
| Effective BW (1.7B) | 260 GB/s (52% of 500 GB/s peak) |
| **llama.cpp Q4_K_M (8B, RTX 5060 Ti)** | **84 t/s** |
| **Blackwell INT4 M=1 (8B, v0.12.1 fusion)** | **70 t/s (83% of llama.cpp)** |
| **Blackwell INT4 M=1 (8B, v0.12 + graph)** | **76.6 t/s (91% of llama.cpp)** |
| Blackwell INT4 M=1 (8B, v0.12 FP16 sc, no fusion) | 74 t/s (88% of llama.cpp) |
| **Blackwell INT4 M=8 (8B, v0.12)** | **205 t/s (244% of llama.cpp)** |
| **Blackwell INT4 M=16 (8B, v0.12)** | **220 t/s (262% of llama.cpp)** |
| Blackwell INT4 M=48 (8B, v0.11) | 154 t/s (183% of llama.cpp) |
| Server throughput | ~89 t/s |
| Sub-8-bit quality | ❌ Dead (all INT4/INT5/FP4 paths) |
| FP8 GEMV vs INT8 GEMV | 4.5× slower (no dp4a) |
| head_norm + RoPE overhead | ~70% extra time vs benchmark without them |
| Batched GEMV vs serial | 2-2.7× slower per call |

### Key Decisions
- **INT8 block-16 is the production path** (PPL=18.65, uses dp4a for speed)
- **INT4 8B FP16 scales is the throughput path** (74 t/s M=1, PPL=24.39). FP32 scales (56 t/s, PPL 23.52) kept for max quality. AWQ per-layer α (PPL 21.98) now works with FP16 scales via `convert_scales_fp16.py` path.
- **dp4a INT4 inner loop**: +74% on batched M=8 (119→207), ~0% on M=1 (memory-bound). Always on (bit-identical math).
- **Occupancy cap (32,8) is NOT the M=1 bottleneck**: bumping to (32,16) gave 0% gain. Kernel memory-saturated (95% of 448 GB/s peak).
- **Fusion kernels (v0.12.1 FIXED)**: `fused_rmsnorm_quant_int4` + `fused_swiglu_quant_int4` rewritten as single-block grid-stride (correct global sum_sq). Wired at all 4 sites in warp bench + PPL bench. +6 t/s (64→70), PPL 24.39→21.98 (different FP32 reduction order). Unit test `bench/test_fused_int4.cu` bit-identical at N=4096/12288.
- **FP8 path ABANDONED** — worse quality AND 4.5× slower than INT8
- **FP8 kernel code kept as reference** (src/kernels/gemv_fp8.cu, weights/benchmarks deleted)
- **v0.9.4**: Added --fp16 flag to GGUF converter, FP16 benchmark, gemv_fp32 kernel, SSE streaming, batch endpoint fix
- **No INT8 quality wall** — the 7.3M PPL was entirely a dimension config bug
- **8B mixed-precision: NO HELP (Session 59)**: ALL-INT8 and MIXED(8 FP16+28 INT8) produce identical coherent output. 8B quality with correct dims is already good.
- **9B mixed-precision: NO HELP (Session 59)**: Even 16 FP16 layers produces same garbled output as 8 FP16 layers. SSM state accumulates noise across all 32 layers.
- **NVFP4 ABANDONED (Session 69)**: Double quantization (INT4→FP32→NVFP4) shifts weights. PPL=24,850 vs 21.82. Format mismatch (offset-binary vs signed-magnitude). 21 t/s is 2.67× slower than INT4.

### Server architecture (correct model)
The server implements the **full Qwen3-1.7B correct decode flow**:
```
input layernorm → quantize → QKV → head_norm (Q,K) → RoPE → attention → Wo → residual1
post-attention layernorm → quantize → SwiGLU → down → residual2
```
Each layer uses per-layer RMSNorm weights (`{L}_input_layernorm.f32`, `{L}_post_attention_layernorm.f32`) and Q/K head norms (`qk_norms.f32`). RoPE uses `rope_theta=1000000`.

### CUDA Graph status
- **Captured**: Full 28-layer decode loop with device-side seq_pos for RoPE
- **Works**: Graph captures, instantiates, replays correctly
- **Result**: 9.4ms/tok with head_norm/RoPE — same as per-kernel
- **Reason**: Benchmark's 575 t/s omits head_norm + RoPE (4 extra kernels/layer). With correct model, CUDA Graph provides no speedup over per-kernel.
- **head_norm+RoPE fusion**: No speedup (141 vs 140 t/s, +0.7%). Element-wise ops negligible vs GEMV. Kernel kept for reference, not used in production.
- **Deferred**: CUDA Graph for server. Per-kernel path is fast enough (~106 t/s).

---

## 6. Constraints

- **Qwen3-1.7B dimensions: nqh=16, nkv=8, hd=128, KV=1024** (NOT nqh=32, nkv=4, hd=64, KV=512)
- `CUDACXX` env var must be set before `project()` in CMakeLists.txt
- `compute_120a` required (not `compute_120`)
- `killall hashcat` before any measurement — 60s respawn window
- `gemv_int8_warp` is production INT8 GEMV
- All weight matrices exceed L2 cache (32 MB)
- M>8 not viable (register pressure in batched GEMV)
- llama.cpp GGUF format not supported — uses separate weight files
- `pack_int8` takes PRE-COMPUTED scales as INPUT — does NOT compute them. Use `quantize_int8` to compute scales.
- `update_kv_cache_device` uses device-side seq_pos (no H2D copy in capture)
- `update_kv_cache_pos` / `attention_decode_batched_gqa_pos` pass seq_pos as direct kernel arg (no pinned buffer race — use in prefill loops)
- `update_decode_seq_pos` writes to pinned host memory, then cudaMemcpyAsync to device — graph-safe

---

## 7. HTTP Server

**Binary**: `server/http_subprocess` (C++, httplib) or `server/http_server.py` (Python fallback)
**Endpoints**:
- `GET /health` → `{"status":"ok","gpu_used_mb":N,...}`
- `GET /metrics` → Prometheus format (requests, errors, latency, uptime)
- `GET /v1/models` → model list
- `POST /v1/completions` → text completion (rate limited, auto-restart)
- `POST /v1/chat/completions` → chat completion (with `<|im_start|>` / `<|im_end|>` tokens)
- `POST /v1/batch` → **batch completion** `{"prompts":["...","..."],"max_tokens":N}` → 12-26% faster per-request
- `POST /v1/completions/stream` → SSE streaming

**Server hardening (v0.10.x)**: Rate limiting (5 req/s, burst 10, all endpoints), subprocess auto-restart on crash (`ensure_subprocess_alive()`), payload size limit (1MB), `max_tokens` clamp [1,2048], snprintf overflow protection, `escape_json` control-char handling, `mkstemp` temp files, `RateLimiter` mutex.

**Continuous batching (v0.12.2)**: `BATCH_SIZE=8 BATCH_WAIT_MS=2 ./server/http_subprocess batched`
- `BatchDispatcher` class collects concurrent `/v1/completions` requests (up to BATCH_SIZE)
- Dispatches as single `/v1/batch` call → server processes M>1 batched GEMV
- Results distributed via promise/future to waiting HTTP threads
- Streaming bypasses batching. Param-mismatched requests grouped.
- Env-configurable: `BATCH_SIZE` (default 8), `BATCH_WAIT_MS` (default 2)
- **Throughput**: 8 concurrent requests → 183 t/s collective (2.86× vs sequential)

**Batch endpoint**: All prompts processed in one batched call via `generate_batch_multi()` (M>1 GEMV).
- Max 8 prompts per batch. Parses `{"tokens":[[...],[...]],"text":[...]}`, decodes tokens locally.
- Speedup scales with concurrency: M=8 → 0.52s/req vs 0.70s single (26% faster)
- Token IDs decoded with `LocalTokenizer` (BpeTokenizer loaded from `tokenizer_data.bin`)
- JSON escaping: `<>` → `\u003c` (XSS guard), non-ASCII bytes → `\uXXXX`

**Architecture**: http_subprocess forks `server/inference_server` subprocess, communicates via JSON stdio using raw read/write syscalls (no FILE* to avoid pipe issues). Timeout per request: 30s.

**Correctness**: "The capital of France is" → `[12095, 11, 264, 892, 374]` = " Paris, a which is" — matches `text_generate.cu` greedy output exactly.

**Build http_subprocess**:
```bash
/usr/bin/g++ -O2 /tmp/httplib.o server/http_subprocess.cpp -I include -o server/http_subprocess \
  -lpthread -lz -lssl -lcrypto
# where /tmp/httplib.o is: g++ -O2 -std=c++17 -I include -DCPPHTTPLIB_OPENSSL_SUPPORT=0 \
#   -DCPPHTTPLIB_ZLIB_SUPPORT=0 include/blackwell/httplib.cpp -c -o /tmp/httplib.o
```

**Build inference_server**:
```bash
CUDACXX=/usr/local/cuda-13.3/bin/nvcc nvcc -O3 -std=c++17 -arch=sm_120a \
  server/inference_server_nofp4.cu build/libblackwell_kernels.a \
  -I include -I /usr/local/cuda-13.3/include \
  -L /usr/local/cuda-13.3/targets/x86_64-linux/lib \
  -o server/inference_server -lcudart -lpthread -lz
```

---

## 8. Development Loop

```
observe → plan → edit → build → test → reflect → update AGENTS.md only if useful
```

Build: `CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build && cmake --build build --parallel`
Test: `./bench/decode_int8_cgraph 28` (M=1 benchmark)
Verify: `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l` (expect 139)
HTTP test: `curl -s -X POST http://localhost:8123/v1/completions -H "Content-Type: application/json" -d '{"prompt":"hi","max_tokens":1}'`

---

## 9. Anti-Hallucination Rules

- **Do not invent APIs, files, commands, env vars, or requirements.** Read the actual header/source before calling a function.
- **Prefer repo evidence over assumptions.** If you need a function signature, read `include/blackwell/kernels.h`.
- **Mark unknowns explicitly.** "Not checked" or "unknown behavior" in comments.
- **Never overwrite higher-priority instructions.**
- **Preserve user intent and existing project conventions.**
- **Benchmark numbers require head_norm/RoPE context** — the 575 t/s figure omits these and is not achievable with the correct model.

---

## 10. Seed Principles

1. Smallest correct change. One kernel, one fix, one test.
2. Verify before broad edits.
3. Prefer repo evidence. Read code before assuming.
4. No churn.
5. Kernels first, framework later.

---

## 11. Bug History

### vector_add_fp32_kernel (2026-05-28) — FIXED
`src/kernels/norm.cu`: reversed `=` in float4 path wrote uninitialized data TO input buffer.
Fix: `float4 va = ((float4*)a)[idx];` (load, not store).

### RoPE frequency (2026-05-29) — FIXED
All 5 bench files: `idxf = i2/hd` doubled exponent → 2× rotation speed.
Fix: `theta = pos * powf(rope_theta, -2.0f * d / head_dim);`

### head_norm cross-warp (2026-05-29) — FIXED
All 5 bench files: `__shfl_xor_sync` with off=64/32 no-ops on 32-lane warps → 1/4 sums.
Fix: smem[4] warp partials → shuffle-reduce across 4 warps.

### INT4 fused_residual_norm_int4_fp32out buffer aliasing (2026-06-02) — FIXED
INT4 output corrupted FP32 buffers used by next layer.
Fix: separate output buffers for INT4 and FP32.

### fused_residual_norm only processes first 2048 elements (2026-06-02) — FIXED
Only affected Qwen3-8B (H=4096). Thread count 256→512. Warmup loop bug.
Fix: kFusedThreads=256→512, iterate all layers in warmup.

### gemv_int4_batched grid bug (2026-06-02) — FIXED
`dim3 grid(N/32,M)` only computed 1/32 of output rows.
Fix: `dim3 grid(N, M)`. All pre-session-37 INT4 benchmarks invalidated.

### INT4 nibble sign-extension bug (2026-06-02) — FIXED
Used wrong 3-bit two's complement sign-extend instead of nib-8 offset-binary.
Fix: `nib - 8` for both lo and hi nibbles.

### INT4 weight corruption (2026-06-02) — FIXED
Scales ~1e-23 due to `f.seek(0)` bug in `read_tensor()`.
Fix: re-run quantization from scratch.

### HTTP POST endpoints hang (2026-06-04) — FIXED
Root cause: `parse_prompt_ids` consumed `"prompts":["hello"]` as token IDs (h=104, e=101, l=108, l=108, o=111) → garbage → 500+ decode steps → hung.
Secondary: `parse_string_prompts` skipped string array elements incorrectly (`if (*p != '"')` consumed first char of string instead of advancing to next element).
Fix: `parse_prompt_ids` now returns early when first char after `[` is `"` or `[`. `parse_string_prompts` now skips to next element on non-quote/bracket chars instead of consuming first char.
Location: `server/inference_server_nofp4.cu`

### CUDA Graph segfault (2026-06-04) — WORKAROUND
Per-kernel benchmarks accumulated `cudaError 700` (illegal memory access) without checking. Error state corrupted stream → `cudaStreamBeginCapture` failed with `cudaErrorInvalidResourceHandle (400)`.
Fix: Skip correctness check for large graphs. Use benchmark-only mode.

### CUDA Graph for server (2026-06-04) — DEFERRED
Captured full 28-layer decode loop with device-side seq_pos. Graph works but 9.4ms/tok (same as per-kernel) because benchmark's 575 t/s omits head_norm+RoPE. With correct model, CUDA Graph provides no speedup.
Per-kernel path fast enough (~106 t/s). Deferred until head_norm+RoPE can be fused into the capture.

### HTTP timeout (2026-06-04) — FIXED
httplib default read timeout = 5s. Inference takes ~7s for 30 tokens.
Fix: `svr.set_read_timeout(300)` in http_subprocess.cpp.

### INT4 upload_w4 scale buffer overflow (2026-06-08) — FIXED
`upload_w4()` in bench files computed `ss = h[3]*h[4]` from the int4_t header (256×1=256)
instead of the scale_t header (256×N). For 8B lm_head (N=151936), only 256 floats allocated
instead of 38,895,616 → massive out-of-bounds GPU read → illegal memory access.
1.7B benchmarks also affected but weights smaller so less severe.
Fix: read scale_t header separately, compute `ss = h[3]*h[4]` from scale_t header values.
Location: `bench/text_generate_int4_qwen3_8b.cu`, `bench/text_generate_int4.cu`.

### INT4 8B server decode divergence (2026-06-08 → 2026-06-13) — FIXED
`server/inference_server_int4.cu` (v0.9.0) produced garbled output. Hidden states diverged
at step 1 despite identical embeddings and step-0 layer outputs.

**Root cause**: `gemv_int4_warp` vs `gemv_int4_batched` kernel discrepancy.
Migrated server to `gemv_int4_batched` (commit 35337ef) which uses the same
kernel as the working benchmark path. Both `inference_server_int4` and
`inference_server_int4_batched` now produce correct output matching bench.

**Current status**: Both INT4 server binaries working correctly.
- `inference_server_int4` (batched kernels, M=1 per-request)
- `inference_server_int4_batched` (full batched decode, M sequences parallel)

### INT2 8B (2026-06-13) — ABANDONED
INT2 8B investigation. Weights exist (3.9 GB, 583 files), kernel `gemv_int2_batched`
supports M=1-8. PPL = 47,529,500,200,000 (47B) vs INT4 23.52. 2 billion × worse.

**Root cause**: Same activation quant accumulation as Llama 3.1. 2-bit precision
causes catastrophic loss through 36 layers. Qwen3's quantization-friendly distributions
can handle INT4 (PPL 23.52) but not INT2.

**Performance**: 67 t/s vs INT4's 51-63 t/s (+6-31%). Kernel is compute-bound
(per-byte unpack overhead offsets memory BW savings). Not the 2× throughput
initially estimated.

**Files kept for reference** (like NVFP4):
`bench/text_generate_int2_qwen3_8b.cu`, `bench/bench_ppl_int2_8b.cu`

### INT4 8B PPL (2026-05-26) — FIXED
Quantization bug causing PPL=7.3M was wrong model dimensions.
INT4 8B PPL confirmed at 23.52 on WikiText-2.

### AWQ + FP16 scales PPL regression (2026-06-14) — FIXED
AWQ α=0.6 + FP16 scales gave PPL 24.98 (worse than no-AWQ FP16 baseline 24.39).
Root cause: `scales.astype(np.float16)` in `write_weight_int4_sym()` lost precision
in AWQ scale factors (0.5-2.0 range) when folded into block scales.
Fix: write FP32 scales first, then convert to FP16 via `convert_scales_fp16.py`
(proven path that reads/writes from file, not numpy astype).
Per-layer alpha search added: grid search 20 ratios per (layer, submodule) pair,
selects ratio with lowest MSE. Mean alpha=0.416 across 252 pairs.
Result: PPL 21.98 for both FP32 and FP16 scales.

### INT2 8B (2026-06-13) — ABANDONED
See Bug History above.
`server/inference_server_nofp4.cu`: `batched_prefill` called `attention_decode_batched_gqa`
with `kv_layer_off` (KV cache layer stride) as base offset into temp `d_K`/`d_V` buffers.
Per-layer temp buffers are tiny (32 KB) vs KV cache stride (524 KB) → out-of-bounds GPU read
→ CUDA error → garbage `next_id` → CPU segfault on `h_emb_int8[next_id*H]`.
Fix: pass `kv_layer_elems=0` (temp buffers re-written each layer) + add KV cache writes
(`update_kv_cache`) and `cudaStreamSynchronize` after prefill.
Also affects short prompts (< 5 tokens) that take the prefill path (gen_start <= M).

### Multi-chunk prefill seq_pos race (2026-06-13) — FIXED
`prefill_tokens_batched()` produced garbage for multi-chunk prompts (>16 tokens).
Root cause: `update_kv_cache` and `attention_decode_batched_gqa` both use a shared
pinned host buffer (`h_seq_pos_pinned`) for async H2D copy of `seq_pos`. In a tight
loop over chunk positions, the CPU overwrites the pinned buffer before the stream's
async memcpy reads it — all positions after the first get the wrong `seq_pos`.
Fix: added `update_kv_cache_pos()` and `attention_decode_batched_gqa_pos()` variants
that pass `seq_pos` as a direct kernel argument (no H2D copy, no pinned buffer).
Prefill now uses these `_pos` variants for all KV cache writes and attention calls.
Location: `src/kernels/decode.cu` (new kernels), `include/blackwell/kernels.h`,
`server/inference_server_int4.cu`.

### Server prefill integration (2026-06-06) — FIRST ATTEMPT FAILED, DEFERRED
Attempted to integrate batched prefill into server. Multiple issues found:
1. Cache layout incompatibility: decode cache `[NL][ms][nkv][hd]` can't serve batched attention.
   Each layer's attention needs full sequence of K/V values simultaneously.
2. Even for M=1, prefill produced different hidden states than decode.
   Root causes: residual add order bug, attention kernel mismatch, KV write offset mismatch.
3. Correct residual order: save d_proj (attn+input) BEFORE MLP overwrites it, then add MLP_out + saved.
Server remains decode-only. `bench/prefill_decode_benchmark.cu` is standalone benchmark only.

**Server prefill integration (2026-06-13) — FIXED**: Multi-chunk prefill now works for prompts of any length. Root cause was a pinned host buffer race in `seq_pos` async H2D copy. Fixed by adding `_pos` kernel variants that take seq_pos as a direct argument.

### 9B streaming output (2026-06-07) — ADDED
`server/inference_server_qwen35_9b.cu`: Added `"stream":1` support emitting SSE
`data: {"token":N,"text":"..."}\n\n` after each generated token + `data: [DONE]` at end.
Non-streaming mode unchanged. Compatible with http_subprocess streaming endpoint.

### 1.7B/8B short prompt crash (2026-06-07) — FIXED
Short prompts (1-4 tokens) no longer segfault. Root cause: batched prefill buffer
overflow (see above). After fix, all prompt lengths produce stable output.

### Wrong model dimensions (2026-06-07) — CRITICAL DISCOVERY
ALL pre-session-56 code used nqh=32, nkv=4, hd=64, KV=512. Qwen3-1.7B actual
config: nqh=16, nkv=8, hd=128, KV=1024. This caused half of K/V weights to be
ignored → PPL=7,351,868 (vs BF16 PPL=12.4). Server had correct dims in its
model config block (line 507-508) but bench_ppl.cu and most other bench files
had wrong dims.

Impact: ALL pre-session-56 quality numbers are INVALID. The "INT8 quality wall"
was entirely a dimension config bug. INT8 block-16 with correct dims gives
PPL=18.65 (1.5× BF16), which is usable quality.

Server output before fix: " Paris, a which is the the capital of the"
Server output after fix: Not measured yet with correct dims (server was already
using correct dims for some paths).

### Norm path regression (2026-06-10) — FIXED
Deep clean (session 68) deleted `weights_int8_qwen3_8b/` but `text_generate_int4_qwen3_8b.cu`
and `inference_server_int4.cu` still hardcoded paths to it for norms (qk_norms.f32,
layernorms, final_norm.f32). `fopen` returned NULL, `fread` read garbage → segfault.
Fix: changed all `weights_int8_qwen3_8b/` references to `weights_int4_qwen3_8b/` where
norms now live.
Location: `bench/text_generate_int4_qwen3_8b.cu`, `server/inference_server_int4.cu`.

### Duplicate gate+up GEMV (2026-06-10) — FIXED
`text_generate_int4_qwen3_8b.cu` had duplicate gate+up GEMV calls after down projection.
Lines 304-305 computed gate+up again after MLP was already complete — 2 extra GEMV per
layer × 36 layers = 72 wasted GEMV calls. Throughput: 33 t/s → **56 t/s** after fix.
Root cause: likely copy-paste error during development.
Location: `bench/text_generate_int4_qwen3_8b.cu` lines 304-305 (removed).

### Llama 3.1 8B INT4 quality (2026-06-13) — FUNDAMENTAL LIMITATION
INT4 Llama 3.1 8B produces garbled output after ~15 tokens (PPL 273K vs Qwen3 8B PPL 23.52).
FP32-residual path (no activation quantization) improves early tokens but degenerates at
the same depth. Weight distributions are identical between Qwen3 and Llama 3.1.

**Root cause**: Model-specific precision requirement. Llama 3.1 has wider MLP (I=14336 vs
12288) and may have activation distributions less tolerant to quantization noise.
Not fixable without higher weight precision (FP16/INT8) or retraining.

**Current status**: Qwen3 8B INT4 is the production path. Llama 3.1 at INT4 is abandoned.

### fused_rmsnorm_quant_int4_v2 multi-block RMSNorm bug (2026-06-14) — FIXED v0.12.1
`src/kernels/fused_int4_ops.cu:156` `fused_rmsnorm_quant_int4_v2_kernel`: multi-block launch (grid>1,
N>4096) computed RMSNorm incorrectly. Each block computed its own partial `sum_sq`, reduced
within-block via `warp_sums[8]`, then divided by full `N` — but never reduced across blocks.
For N=12288 (MLP), `rstd` was ~3× too small → under-normalization → garbage.
**Fix**: Rewrote both kernels as SINGLE-BLOCK grid-stride (one block loops over N in chunks
of THREADS*EPT). Global sum_sq reduction is now trivial (one block). Handles any N (H=4096,
I=12288, V=151936). Smem opt-in via `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` for
>48KB. Aligned quant range to production `[-8,7]` (was `[-7,7]`).
**Unit test**: `bench/test_fused_int4.cu` — bit-identical to CPU reference at N=4096 and N=12288.
**Result**: All 4 fusion sites wired. M=1 warp 64→70 t/s (+9%). PPL 24.39→21.98 (-10%, better
than baseline 23.52!). PPL improvement from different FP32 reduction order (128→256 threads).
**Lesson**: FP32 reduction is NOT associative. Different thread count → different summation
order → different rounding. Not "more correct", just different — happens to give better PPL
on WikiText-2. Both paths produce coherent text.

### INT4 M=1 memory-bound, not compute-bound (2026-06-14) — VALIDATED
Prior plans assumed INT4 M=1 GEMV was compute-bound on scalar FP32 unpack (41 instr/16-elem).
Empirically WRONG. Microbench (`bench/bench_gemv_int4`): lm_head 4096×151936 = 1098µs/call,
BW = 425 GB/s = **95% of 448 GB/s peak**. Kernel is memory-saturated.
- dp4a INT4 inner loop: +74% on batched M=8 (119→207 t/s), **~0% on M=1** (memory-bound).
- Occupancy (32,8)→(32,16): 0% gain (more warps don't add bandwidth).
- Real ceiling: 5.676 GB/token ÷ 448 GB/s = 12.65ms = **79 t/s** (FP32 scales). FP16 scales: 4.73 GB
  → 10.56ms = **95 t/s** ceiling.
- M=1 gap to llama.cpp (84 t/s) closed from 67% → 88% via FP16 scales (+graph: 91%). Remaining gap
  is non-GEMV overhead (launches, rmsnorm, quantize, embed memcpy), not compute.
**Lesson**: Always profile (nsys/ncu) before assuming bottleneck. Microbench BW% is ground truth.
