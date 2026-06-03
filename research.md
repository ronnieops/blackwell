# Research: INT4 Quantization Quality Problem — blackwell

## Summary
INT4 symmetric per-block (block-16) quantization produces ~14% per-value error that compounds across 28 layers, causing garbled text output. The root causes are: (1) symmetric INT4 has only 15 levels [-7..7] vs INT8's 255 [-127..127], (2) per-block scales with block size 16 amplify outliers, (3) no zero-point/asymmetric range wastes quantization budget. Solutions include asymmetric per-block quantization, per-channel quantization (1 scale per output row), or AWQ-style activation-aware per-channel scaling.

## Findings

### 1. Existing Error Analysis — INT4 symmetric ~14% per-value, INT8 ~1.5%
- **HANDOFF.md §1**: "4-bit symmetric quantization noise ~14% per-value error compounds across 28 layers. First token diverges from INT8 greedy decode." INT8 is ~1.5% per-value.
- **HANDOFF.md §5.2**: "INT4 text_generate quality — garbled after 28 layers. Symmetric 4-bit ~14% error compounds. Needs asymmetric Q or fine-tuning."
- **HANDOFF.md §9**: Single-layer GEMV accuracy shows "avg err 0.11 vs INT8 (expected for 4-bit)". Embedding RMS diff 0.002 vs INT8 — acceptable at single layer but compounds.
- **AGENTS.md §7.21**: "Pipeline structurally correct but 4-bit symmetric quantization noise (~14% per-value error) compounds across 28 layers. First token diverges from INT8 greedy."
- **AGENTS.md §4**: Pipeline SNR measured at **13.9 dB** (constant across 28 layers, no single-layer amplification).

### 2. No asymmetric quantization design exists yet
- **HANDOFF.md §7**: "Implement asymmetric per-block INT4 quantization (separate +ve/-ve scale per block)" — listed as suggested next action, NOT implemented.
- **HANDOFF.md §7**: "Or use per-channel INT4 (one scale per output channel instead of per-block)" — also suggested, NOT implemented.
- No code in `src/kernels/` implements asymmetric or per-channel dequantization.
- Current `gemv_int4_batched_kernel` uses `nib - 8` offset-binary decode with single per-block scale — symmetric by construction.

### 3. Per-channel quantization — NOT implemented, only proposed
- **HANDOFF.md §7** item 3: "per-channel INT4 (one scale per output channel instead of per-block)"
- This would mean 1 scale per output row (N scales total vs N×K/16 currently).
- AWQ (MIT, MLSys 2024) demonstrates that per-channel scaling with activation awareness preserves INT4 quality: "protecting only 1% salient weights can greatly reduce quantization error" by scaling up salient channels before quantization.
- **No AWQ or GPTQ integration exists** in the codebase. All quantization uses custom `quantize_generic.py`.

### 4. `quantize_generic.py` format and the `f.seek(0)` bug
- **Bug location**: `scripts/quantize_generic.py`, function `read_tensor()` (lines 158-170).
- **Exact bug** (line 162-163):
  ```python
  f.seek(8)     # seeks to byte 8
  f.seek(0)     # RESETS to byte 0 — overwrites the seek(8)
  hdr_len_shard = struct.unpack('Q', f.read(8))[0]  # reads header length from start
  ```
- The `f.seek(0)` on line 162 undoes the `f.seek(8)` on line 161. The intent was likely to read the safetensor header length from byte 0, then skip header + data offset. But the redundant `f.seek(8)` before it is a leftover from a different code path.
- **Impact**: When called from `main()` in a loop over layers, the file position from a previous `read_tensor` call may leave `f` at an unexpected offset if the same file handle were reused. But since `read_tensor` opens a fresh file each time (`with open(shard_path, 'rb') as f:`), the `f.seek(0)` is redundant but NOT harmful for single-threaded execution.
- **Weight corruption cause**: The actual corruption (scales ~1e-23) was caused by a different issue — AGENTS.md §6 says "quantize_generic.py read_tensor has f.seek(0) call that corrupts tensor data offset". The real problem: the header is re-read inside `read_tensor` to compute `8 + hdr_len_shard + start`, which is correct for safetensor format. The scales corruption was triggered by running the script after previous runs left stale output, not by the seek bug per se.
- **Current INT4 format** (from `quantize_per_row_int4`):
  - Block size: 16 elements (same as INT8)
  - Symmetric: range [-7..7], scale = absmax/7
  - Nibble packing: `nib = q + 8` (offset-binary, [0..15])
  - Per-block scale: 1 FP32 per 16 elements
  - Bytes/param: 0.5 (weights) + 0.25 (scales) = 0.75

### 5. Q4_PLAN.md next steps
- **Q4_PLAN.md Phase 1.3**: "Compute per-block MSE vs FP32, max absolute error per block, PSNR across entire matrix, compare against INT8. Target: INT4 PSNR > 40 dB."
- **Q4_PLAN.md Phase 3.3**: "Compare INT4 output vs INT8 output: single-layer pipeline per-element diff, full 28-layer aggregate metrics. Target: max diff < 1% of INT8."
- **Q4_PLAN.md Risk**: "INT4 quality loss: HIGH — model output degrades. Mitigation: Validate PSNR per-layer; compare against INT8."
- **Q4_PLAN.md** does NOT propose asymmetric quantization — it assumed symmetric would work. The plan predates the discovery of quality issues.

### 6. SNR measurements
- **AGENTS.md §4**: "Pipeline SNR: 13.9 dB. Constant across 28 layers, no compounding."
- This 13.9 dB was measured for the INT8 pipeline (not INT4). It indicates the overall pipeline is stable — no single layer amplifies noise.
- **No INT4-specific per-layer SNR measurement** exists. HANDOFF.md §7 item 1 suggests: "Verify INT4 vs INT8 at single-layer level (measure per-layer SNR)" — listed as a pending next action.
- **benchmark-results.md**: All INT8 benchmarks show `max_diff: 0.000000 ✅` — INT8 reproduces exactly. INT4 has no equivalent max_diff measurement.

### 7. Available models (from AGENTS.md benchmarks)
- `qwen3-1.7b-base` — Qwen3-1.7B, 28 layers, H=2048, primary development model
- `qwen3-0.6b` — mentioned in benchmark-results.md (28L, H=1024, I=3072)
- `qwen3-8b` — Qwen3-8B, 36 layers, H=4096, I=12288
- `qwen3.5-9b` — MoE model, Qwen3.5-9B
- `qwen3.6-27b` — dense, 16-17 GiB (exceeds VRAM with Q4_K_M)
- `qwen3.6-35b-a3b` — MoE, ~3B active params
- All at `/mnt/data/ai/hf/`

### 8. INT8 vs INT4 error levels
- **INT8**: ~1.5% per-value error, 255 quantization levels, exact reproduction (`max_diff: 0.000000`)
- **INT4**: ~14% per-value error, 15 quantization levels [-7..7]
- **INT4 single-layer GEMV**: avg error 0.11 vs INT8 (from HANDOFF.md §9)
- **INT4 embedding**: RMS diff 0.002 vs INT8 (acceptable at single layer)
- **INT4 full pipeline (28L)**: garbled output — first token diverges from INT8 greedy decode
- **AGENTS.md §7.1**: "symmetric 4-bit ~14% error compounds across 28 layers"
- **Quantization error ratio**: INT4 error is ~9.3× INT8 error per value (14%/1.5%)

### 9. Why Q4_K_M (llama.cpp) works but our INT4 doesn't
- **Q4_K_M uses super-block structure**: group size 256, with FP16 dual scales (super-block scale + sub-block scale). This gives finer granularity than our block-16.
- **Q4_K_M uses asymmetric quantization** with implicit zero points within super-blocks, reducing quantization error for non-symmetric weight distributions.
- **AWQ uses per-channel scaling**: scales salient channels to reduce error on important weights. Group size 128 with separate scale + zero point.
- **Our INT4 uses symmetric block-16 with absmax/7 scale**: only 15 levels, no zero point, coarse granularity. This is the simplest possible INT4 format.

### 10. Literature on INT4 compounding in decoder models
- Research paper "Understanding INT4 Quantization for Transformer Models" (2023): "4-bit decoder models (GPT2, GPT2-medium) show significant drop in perplexity (≥1.5 points) compared to FP32... symmetric quantization shows no degradation for BERT but significant drops for GPT2." This confirms decoder-only models are more sensitive to INT4 quantization.
- AWQ paper (MLSys 2024): "Not all weights equally important. Protecting only 1% salient weights can greatly reduce quantization error." Uses per-channel scaling derived from activation magnitudes.
- GPTQ: Layer-wise optimization with Hessian-based weight adjustment. Uses group size 128 (default) with symmetric per-channel INT4. Supported in vLLM, HF Transformers.
- Key insight from literature: symmetric INT4 works for encoders (BERT) but NOT for decoders without calibration data (GPTQ) or activation-aware scaling (AWQ).

## Sources
- Kept: HANDOFF.md — primary session 37 continuity doc with bug list and next actions
- Kept: AGENTS.md — project state, bug history, known issues, SNR measurements
- Kept: Q4_PLAN.md — original INT4 plan (assumed symmetric would work)
- Kept: REPORT.md — INT8 benchmark report with detailed performance breakdown
- Kept: benchmark-results.md — llama.cpp vs our INT8 comparison
- Kept: scripts/quantize_generic.py — actual quantization code with f.seek(0) bug
- Kept: src/kernels/gemv_int4_batched.cu — INT4 GEMV kernel (nib-8 decode, symmetric)
- Kept: bench/text_generate_int4.cu — INT4 text generation pipeline (17 kernels/layer)
- Kept: AWQ paper (arxiv 2306.00978) — activation-aware per-channel scaling method
- Kept: "Understanding INT4 Quantization for Transformer Models" — decoder-specific INT4 failure analysis
- Dropped: Generic LLM quantization blog posts — no specific technical value
- Dropped: Russian-language Ollama/llama.cpp tutorials — irrelevant to kernel dev

## Gaps
1. **No per-layer INT4 SNR measurement** — only aggregate "garbled output" observation. Need systematic measurement of error at each layer boundary.
2. **No asymmetric quantization implementation** — only proposed. Need design: asymmetric per-block (separate min/max scale + zero point) or per-channel (1 scale per output row).
3. **No AWQ/GPTQ integration** — existing formats use calibration data to find optimal scales. Our quantize_generic.py is naive absmax-based.
4. **llama.cpp Q4_K_M exact format not analyzed in codebase** — known to use super-block 256 with FP16 dual scales, but no detailed comparison against our block-16.
5. **8B INT4 benchmarks stale** — pre-date grid bug fix (HANDOFF.md §5.3). Need re-run with corrected kernel.

## Supervisor coordination
No blocker. All information gathered from file reads and web search. Research complete.
