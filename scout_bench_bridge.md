# Code Context: bench/ + GGUF Bridge Scouting Report

## Files Retrieved
1. `bench/` (34 .cu files) — full bench inventory
2. `better-inference/gguf.h` (lines 1-400) — GGUF parser + dequantizers (Q4_K, Q5_0, Q6_K, Q8_0, F16, F32)
3. `better-inference/gguf_convert.cpp` (lines 1-830) — GGUF→INT4 converter, tensor mapper, tokenizer exporter
4. `better-inference/gguf_test.cpp` (lines 1-65) — parser smoke test (metadata dump only)
5. `better-inference/DESIGN.md` (lines 1-65) — bridge architecture, phase plan
6. `scripts/quantize_awq_int4_8b.py` (lines 1-300) — AWQ calibration with real/random proxy modes
7. `scripts/quantize_awq.py` (lines 1-50) — per-channel quant for GDN
8. `scripts/smooth_quant.py` (lines 1-50) — SmoothQuant weight smoothing
9. `scripts/benchmark_suite.py` (lines 1-330) — HTTP-based correctness/throughput/memory suite
10. `include/blackwell/bpe_tokenizer.h` (lines 1-500) — BPE tokenizer (encode/decode/pretokenize)
11. `scripts/gemma_wrapper.py` (lines 1-50) — HF tokenizer wrapper for Gemma
12. `tests/test_attention.cu`, `test_norm.cu`, `test_gemm.cu`, `test_memory.cu` — GTest stubs (all GTEST_SKIP or DISABLED)
13. `bench/run_benchmarks.sh`, `bench/bench_quality.sh` — shell benchmarks
14. `docs/QUALITY.md` (lines 1-80) — PPL summary
15. `CMakeLists.txt` (lines 118-141) — GTest integration (optional, likely never built)

---

## 1. Bench Files: Inventory & Coverage Gaps

### Inventory (34 .cu files)
| Category | Files | What Measured |
|----------|-------|---------------|
| **PPL** | `bench_ppl.cu` (1.7B), `bench_ppl_int4_8b.cu`, `bench_ppl_int2_8b.cu`, `bench_ppl_llama31_8b.cu`, `bench_ppl_llama31_8b_fp32_res.cu`, `bench_ppl_llama32_1b.cu` | Perplexity on hardcoded WikiText-2 sample |
| **E2E text gen** | `text_generate.cu` (1.7B), `text_generate_int4_qwen3_8b.cu`, `text_generate_int4_batched.cu`, `text_generate_int2_qwen3_8b.cu`, `text_generate_qwen3_8b.cu`, `text_generate_gemma.cu`, `text_generate_llama31_8b.cu`, `text_generate_llama31_8b_fp32_res.cu`, `text_generate_llama31_8b_int8.cu`, `text_generate_llama32_1b.cu`, `text_generate_llama32_1b_fp16.cu`, `text_generate_int4_1.7b.cu` | Single-prompt greedy decode + t/s |
| **Decode benchmarks** | `decode_int8_cgraph.cu`, `decode_int4_cgraph_8b.cu`, `decode_int8_nofp4.cu`, `decode_int8_batched_cgraph_attn.cu`, `decode_int8_batched_cgraph_attn_qwen3_8b.cu`, `decode_int8_cgraph_qwen3_8b.cu`, `decode_qwen35_9b.cu`, `decode_qwen35_9b_batched_v2.cu`, `decode_llama32_1b.cu` | Throughput-only (no correctness) |
| **GEMV microbench** | `bench_gemv_int4.cu`, `bench_gemv_int2.cu` | Raw GEMV speed |
| **Prefill** | `prefill_decode_benchmark.cu` | GEMM prefill + decode pipeline |
| **Profiling** | `profile_decode.cu` | nsys/profile harness |
| **Speculative** | `text_generate_speculative.cu` | Speculative decode (research) |
| **Tokenization** | `tokenize_corpus.cu`, `tokenize_text.cu` | Token ID export for AWQ calibration |
| **Accuracy** | `verify_int8_accuracy.py` | INT8 dequant vs BF16 reference |

### Coverage Gaps
- **No PPL for Gemma 4 12B**: No `bench_ppl_gemma`. Quality unverified beyond "coherent output".
- **No PPL for Qwen3.5-9B GDN**: `bench_ppl.cu:6` explicitly says "use bench_ppl_9b.cu instead" — that file **does not exist**.
- **No multi-user latency sim**: Zero concurrent request benchmarks. No TTFT measurement. `grep` for `concurrent|multi.?user|TTFT|latency` in bench/ → 0 hits.
- **No PPL for INT8 8B**: Only INT4 (`bench_ppl_int4_8b.cu`), INT2 (`bench_ppl_int2_8b.cu`). No INT8 8B PPL bench.
- **PPL corpus is tiny**: `bench_ppl.cu:28-30` — hardcoded ~200-word Austria paragraph. Not real WikiText-2 (604K words). All PPL benches reuse same snippet.
- **No batch latency vs M**: `text_generate_int4_batched.cu` measures throughput per M, not p50/p99 latency per request.
- **No regression golden output**: `scripts/benchmark_suite.py:227-255` has loose regression (check "Paris" substring, min length) — NOT exact token comparison.
- **Shell scripts stale**: `bench/run_benchmarks.sh` only tests 1.7B. `bench_quality.sh:18` references `./server/inference_server` (nofp4 binary), not INT4 server.

---

## 2. GGUF Converter: Formats, Error Handling, Validation

### Supported Quant Formats (`gguf.h`)
| Format | Dequant impl | Status |
|--------|-------------|--------|
| Q4_K / Q4_K_M | `dequant_q4_K` (lines 195-230) | ✅ Verified (Qwen3, Llama 3.2) |
| Q5_0 | `dequant_q5_0` (lines 165-185) | ✅ Verified (Qwen2.5) |
| Q6_K | `dequant_q6_K` (lines 240-270) | ✅ Verified (lm_head in Q4_K_M) |
| Q8_0 | `dequant_q8_0` (lines 155-160) | ✅ |
| F32 | memcpy | ✅ |
| F16 | `dequant_f16` (lines 285-310) | ✅ |
| **Q2_K** | File size computed (line 115: `n_super * 84`) | ❌ No dequant impl |
| **Q3_K** | File size computed (line 118: `n_super * 110`) | ❌ No dequant impl |
| **Q5_K** | File size computed (line 121: `n_super * 80`) | ❌ No dequant impl |
| **Q8_K** | Enum exists (15) | ❌ No file size, no dequant |
| **IQ2_XXS/XS, IQ3_XXS/S, IQ1_S/M, IQ4_NL/XS** | Enums 16-24 | ❌ No file size, no dequant |
| Q4_1, Q8_1 | Enums 3, 9 | ❌ No file size, no dequant |
| Q4_0 | Enum 2 | ❌ No file size, no dequant |

**Gap**: 14 GGML quant types defined in enum, only 6 dequantized. Q2_K/Q3_K are common in llama.cpp Q2_K/Q3_K_M models. IQ-series increasingly popular (IQ4_NL, IQ4_XS).

### Missing Dequant Implementations (Priority Order)
1. **Q3_K** — common mid-tier quant, only ~3.5 BPW
2. **Q2_K** — extreme compression, ~2.6 BPW
3. **IQ4_XS** (enum 23) — 4.25 BPW, better quality than Q4_K at same size
4. **IQ4_NL** (enum 20) — non-linear 4-bit, llama.cpp default for some models
5. **Q5_K** — file size known but no dequant

### Error Handling
- `gguf.h:125`: String length sanity check (reject > 100MB strings). ✅
- `gguf.h:155-160`: File size computation falls back to `n_el * 4` for unknown types. ⚠️ Silently wrong for unsupported types.
- `gguf_convert.cpp:330`: Bounds check: `file_offset + ti.file_size > gguf_mem.size` → skip + log. ✅
- `gguf_convert.cpp:105`: `fopen` failure → stderr + return, but converter continues processing. ⚠️ `write_file` silently returns on failure.
- No validation of dequantized values (no NaN/Inf check except Q8_0, `gguf_convert.cpp:317`).
- No checksum/hash verification of output weights.
- No roundtrip validation (dequant → requant → compare).

### Validation Gaps
- `gguf_test.cpp` — only prints metadata + first 20 tensors. No correctness assertions. No dequant verification.
- No comparison against llama.cpp reference dequant output.
- No test GGUF file with known tensor values.

---

## 3. Calibration: AWQ Method Analysis

### AWQ Script (`scripts/quantize_awq_int4_8b.py`)
**Method**: Per-channel activation scaling, alpha=0.5 (AGENTS.md says 0.6 optimal).

**Critical design flaw — layer-0 reuse** (`collect_activation_stats_real:155`, `collect_activation_stats_random:210`):
```python
# Layer-0 only — all layers reuse same activation pattern
for l in range(NL):
    for wn in WEIGHT_NAMES:
        act_stats[f"{l}_{wn}"] = wn_act_mag[wn].copy()  # SAME for all layers
```
Layer-0 activation stats applied to ALL 36 layers. AWQ paper requires per-layer calibration. This is documented but suboptimal — AGENTS.md notes 7.2% PPL improvement (23.52→21.82), but proper per-layer calibration could yield more.

**Calibration data**:
- Real mode: requires `AWQ_CORPUS` env var pointing to `tokenize_corpus` output. Only 35 hardcoded English prompts (`CALIB_PROMPTS`, lines 40-75), not WikiText-2.
- Random mode: `np.random.randn(H)` proxy — crude fallback.

**Scale computation** (`compute_awq_scales:250`):
```python
s = np.clip((act_mag / mean_act) ** alpha, 0.5, 2.0)
```
AWQ paper formula: `s = (s_w * s_a)^alpha / (s_a^alpha + s_w^alpha)` where s_w=weight magnitude, s_a=activation magnitude. This script uses simplified `(act/mean)^alpha` — **not the paper formula**. Missing weight-side term.

**GPTQ**: Not implemented. `grep GPTQ` → only mentioned in `docs/QUALITY.md:57` ("future improvements") and AGENTS.md research notes. No GPTQ code exists.

**SmoothQuant**: `scripts/smooth_quant.py` exists (548 lines), proper alpha-based weight/activation migration into RMSNorm. More theoretically sound than current AWQ. Unknown if tested on 8B — no results in AGENTS.md.

### Calibration Improvements (Ranked)
1. **Per-layer activation collection**: Run forward pass through all 36 layers, collect per-layer stats. Current layer-0 reuse is main quality bottleneck.
2. **Proper AWQ formula**: Use `(s_w * s_a)^alpha / (s_a^alpha + s_w^alpha)` instead of `(act/mean)^alpha`.
3. **Larger calibration corpus**: 35 hardcoded prompts → 512+ WikiText-2 sequences.
4. **GPTQ implementation**: Hessian-based weight adjustment, potentially better than AWQ for INT4.
5. **Per-layer alpha tuning**: Different layers may need different alpha.

---

## 4. Tokenization: Robustness Assessment

### BpeTokenizer (`include/blackwell/bpe_tokenizer.h`)
**ASCII-only pretokenizer** (`pretokenize:380`): simplified GPT-4 regex. `isLetter()` only checks a-z/A-Z — **no Unicode support**:
```cpp
static bool isLetter(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
```
Non-English text (CJK, Cyrillic, Arabic) will produce different tokenization than HF reference. Qwen3 supports multilingual — this is a correctness risk for non-English prompts.

**BPE merge loop** (`encode_bpe:260`): O(n²) per chunk — finds lowest-rank pair each iteration. Acceptable for short prompts, slow for long text. No caching.

**Byte encoder construction** (`build_byte_encoder:95-155`): Multiple abandoned code paths, dead code blocks (`// Actually just hardcode...`, `// Actually, let's rebuild...`). Final result is correct GPT-2 mapping but implementation is messy.

**Special token handling**: Hardcoded per-model:
- Qwen3: `<|im_start|>`=151644, `<|im_end|>`=151645 (`inference_server_nofp4.cu:249-253`)
- Llama 3: `<|begin_of_text|>`=128000, `<|end_of_text|>`=128001, `<|eot_id|>`=128009 (`bpe_tokenizer.h:75-80` in `load_from_data`)
- Not configurable at runtime — hardcoded in C++.

### EOS/BOS Handling
- **Qwen3 bench files**: Hardcode EOS=151643/151645 (`text_generate_int4_qwen3_8b.cu:332`)
- **Llama 3.1**: Hardcode EOS=128001/128009 (`text_generate_llama31_8b_int8.cu:304`)
- **Llama 3.2**: Hardcode EOS=2/3 (`text_generate_llama32_1b_fp16.cu:303`)
- **Gemma**: EOS=1, BOS=2 (`text_generate_gemma.cu:260`)
- **No BOS insertion**: Bench files don't prepend BOS. Llama 3 expects `<|begin_of_text|>` — missing may cause quality drift.

### Gemma SentencePiece Gap
`gemma_wrapper.py` uses Python HF `AutoTokenizer` because BpeTokenizer can't handle SentencePiece. This means:
- Gemma requires Python at runtime (adds dependency)
- `scripts/gemma_wrapper.py:1` subprocess-calls bench binary, parses token IDs from stdout
- Fragile: output format changes break parsing

### Tokenizer Export from GGUF (`gguf_convert.cpp:729-770`)
Exports BPE tokens + merges into `tokenizer_data.bin`. Reads `tokenizer.ggml.tokens/scores/merges` from metadata. **Does not export BOS/EOS IDs** (`gguf_convert.cpp:711-712` reads them but never writes them).

---

## 5. Model Support Matrix

| Model | Convert | Bench | Server | PPL | Quality | Notes |
|-------|---------|-------|--------|-----|---------|-------|
| Qwen3 1.7B | ✅ | ✅ | ✅ | ✅ 18.65 | ✅ Coherent | INT8 block-16 production |
| Qwen3 8B | ✅ | ✅ | ✅ | ✅ 21.82 | ✅ Coherent | INT4 production, AWQ |
| Llama 3.2 1B | ✅ | ✅ | ✅ | ✅ | ✅ 223 t/s | Via GGUF bridge, verified |
| Llama 3.1 8B | ✅ | ✅ | ❌ | ✅ 273K | ❌ Garbled | Abandoned — activation quant |
| Gemma 4 12B | ✅ | ✅ | ✅ | ❌ | ⚠️ Coherent | Needs Python tokenizer |
| Qwen3.5-9B GDN | ✅ | ✅ | ✅ | ❌ | ❌ Garbled | SSM instability, BLOCKED |
| Qwen2.5-0.5B | ✅ | ❌ | ❌ | ❌ | ❓ | Parser verified, no bench |
| **Mistral** | ❌ | ❌ | ❌ | — | — | Same arch as Llama, likely works |
| **Phi-3/3.5** | ❌ | ❌ | ❌ | — | — | Similar to Llama, GeGLU variant |
| **DeepSeek V2/V3** | ❌ | ❌ | ❌ | — | — | MoE, different attention (MLA) |
| **Qwen2.5-Coder** | ❌ | ❌ | ❌ | — | — | Same arch as Qwen, should work |

**Architecture support in converter** (`map_layer_tensor:gguf.h:360`): Only maps standard `attn_q/k/v/output`, `ffn_gate/up/down`, `attn_norm`, `ffn_norm`, `q/k_norm`. No support for:
- MoE expert tensors (`blk.N.ffn_gate_EXP.weight`)
- MLA (multi-latent attention) — DeepSeek
- Gated linear attention (GDN) — handled separately in CUDA kernels, not via GGUF converter
- Sliding window attention metadata (Gemma uses it but converter ignores SWA config)

---

## 6. Testing: Automated Regression

### Current State: **Effectively zero automated correctness tests**

| Test | Status |
|------|--------|
| `tests/test_attention.cu` | `GTEST_SKIP` — "not yet implemented" |
| `tests/test_norm.cu` | `GTEST_SKIP` — "not yet implemented" |
| `tests/test_gemm.cu` | Stub: checks constants only (`EXPECT_EQ(kGEMMTileM, 64)`) |
| `tests/test_memory.cu` | Stub: checks shared mem arithmetic |
| `scripts/benchmark_suite.py` | Loose: checks "Paris" substring, garbage bytes, token count ranges |
| `bench_quality.sh` | Loose: checks `unique` token count > 1 (anti-looping) |
| `better-inference/gguf_test.cpp` | Dump-only: prints metadata, no assertions |
| Golden output comparison | **None**. No stored expected token sequences. |

**CMakeLists.txt:118-141**: GTest integration is optional (`find_package(GTest QUIET)`). All test bodies are `GTEST_SKIP` or `DISABLED_`. Tests have never been functional.

**No CI/CD**: No GitHub Actions, no pre-commit hooks, no automated test runs.

### Regression Test Gaps (Priority)
1. **Golden token sequence comparison**: Fixed prompt → exact token ID sequence (greedy decode). Any kernel/quant change that breaks output would be caught.
2. **Kernel unit tests**: GEMV input→output verification with known small matrices.
3. **GGUF converter roundtrip**: Known FP32 tensor → INT4 → dequant → compare error.
4. **Tokenizer roundtrip**: encode(text)→decode(tokens)→compare.
5. **PPL regression**: Run `bench_ppl_int4_8b` nightly, alert if PPL drifts > 5%.

---

## 7. Weight Management: Redundancy Analysis

### Active Weight Directories (4 dirs, ~22 GB)
| Dir | Size | Status |
|-----|------|--------|
| `weights_int4_qwen3_8b/` | 5.8 GB | ✅ Production (INT4 8B) |
| `weights_gemma/` | 11 GB | ⚠️ Needs Python tokenizer |
| `weights_int2_qwen3_8b/` | 3.9 GB | ❌ Abandoned (PPL 47B) |
| `weights_int4_qwen3_1.7b/` | 1.5 GB | ❌ Dead (sub-8-bit quality dead) |

### AGENTS.md Lists 9 Dirs — Only 4 Exist on Disk
| Listed in AGENTS.md | On Disk? |
|---------------------|----------|
| `weights_int8_bf16/` (1.7B) | ❌ Deleted |
| `weights_int4_qwen3_1.7b_asym/` | ❌ Deleted |
| `weights_int5_qwen3_1.7b_asym/` | ❌ Deleted |
| `weights_int8_qwen3_8b/` | ❌ Deleted |
| `weights_int8_qwen3_8b_mixed/` | ❌ Deleted |
| `weights_int8_qwen3_8b_all_int8/` | ❌ Deleted |
| `weights_int8_qwen35_9b/` | ❌ Deleted |
| `weights_int8_qwen35_9b_mixed/` | ❌ Deleted |
| `weights_int4_qwen3_8b/` | ✅ |
| `weights_gemma/` | ✅ (not listed in AGENTS.md weight section!) |
| `weights_int2_qwen3_8b/` | ✅ (not listed in AGENTS.md weight section!) |
| `weights_int4_qwen3_1.7b/` | ✅ |

### Cleanup Opportunities
1. **Delete `weights_int2_qwen3_8b/`** (3.9 GB) — abandoned, PPL 47B, never used in production.
2. **Delete `weights_int4_qwen3_1.7b/`** (1.5 GB) — AGENTS.md says "dead end". Saves 5.4 GB total.
3. **Update AGENTS.md** — 5 dirs listed don't exist, 2 dirs on disk not listed.
4. **`weights_gemma/`** (11 GB) — functional but quality unverified (no PPL). Consider PPL test before keeping.

---

## Ranked Gaps + Opportunities

### P0 — Correctness Risks
1. **No golden output regression test** — any kernel change silently breaks quality. Most impactful gap. Add fixed-prompt → expected-token-sequence test.
2. **AWQ layer-0 reuse** (`quantize_awq_int4_8b.py:155`) — calibration quality bottleneck. Per-layer forward pass could improve PPL beyond current 21.82.
3. **BpeTokenizer ASCII-only** (`bpe_tokenizer.h:380`) — multilingual prompts produce wrong tokens. Qwen3 is multilingual model.

### P1 — Missing Quant Support
4. **Q3_K dequant missing** (`gguf.h:118`) — common llama.cpp quant. File size known, dequant not implemented.
5. **Q2_K dequant missing** (`gguf.h:115`) — extreme compression tier.
6. **IQ4_XS/NL dequant missing** (enums 20, 23) — newer, better quality-per-bit than Q4_K.
7. **No INT8 8B PPL bench** — INT8 quality claimed "coherent" but no formal PPL number for 8B INT8 (only 1.7B).

### P2 — Coverage Gaps
8. **No Gemma 4 PPL** — 11 GB weights, no quality verification.
9. **No 9B GDN PPL bench** — file referenced (`bench_ppl.cu:6`) doesn't exist.
10. **No multi-user latency test** — production server claim 154 t/s at M=48, no p50/p99 latency data.
11. **PPL corpus too small** — hardcoded ~200 words, not real WikiText-2.
12. **GGUF converter exports no BOS/EOS** (`gguf_convert.cpp:711-712`) — read but discarded.

### P3 — Code Quality / Cleanup
13. **GTest stubs never implemented** (`tests/*.cu`) — all `GTEST_SKIP`. Remove or implement.
14. **AGENTS.md weight dir list stale** — 5/9 dirs deleted, 2 unlisted.
15. **Delete abandoned weight dirs** (5.4 GB savings: INT2 8B + INT4 1.7B).
16. **`bpe_tokenizer.h:95-155`** — dead code in `build_byte_encoder`, 3 abandoned implementations in sequence.
17. **GPTQ not implemented** — mentioned as future work, no code. AWQ is only calibration method.
18. **Mistral/Phi support** — same architecture as Llama, likely works with minor converter changes. Low effort.

---

## Start Here
Open `include/blackwell/bpe_tokenizer.h` — the tokenizer is the single highest-risk component (ASCII-only, dead code, no roundtrip test). Any agent working on quality or correctness should verify tokenizer correctness first, then look at `scripts/quantize_awq_int4_8b.py:155` for calibration improvements.
