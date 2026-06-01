# HANDOFF.md — blackwell

Continuity doc. Read before acting. Keep current with AGENTS.md.

---

## 1. Current Objective

INT8 inference engine for RTX 5060 Ti. Production-ready. **141 symbols**.
- Qwen3-1.7B: 326.8 t/s batched M=8 (119% of Q4_K_M 274.4)
- Qwen3-1.7B: 182.8 t/s M=1 (67% of Q4_K_M)
- Qwen3.5-9B: 45.7 t/s (64% of Q4_K_M 71.4) — INT8 weight-bound, no fix possible
- Bottleneck: INT8 reads 2× more data than Q4_K_M. INT4/FP4 dead end on GB206.

---

## 2. Current Status

| Metric | Value |
|--------|-------|
| GPU | RTX 5060 Ti, GB206, SM_120a, 36 SMs, ~500 GB/s GDDR7 |
| CUDA | 13.3.33, driver 580.159.04 |
| Library | **141 symbols** `build/libblackwell_kernels.a` |
| Branch | master @ `2e64aee` |
| Session | **28** |

### Benchmark Results

| Model | Config | Blackwell INT8 | llama.cpp Q4_K_M | Ratio |
|-------|--------|----------------|-------------------|-------|
| Qwen3-1.7B | Batched attn M=8 | **326.8 t/s** | 274.4 t/s | **119%** ✅ |
| Qwen3-1.7B | CUDA Graph M=1 | 182.8 t/s | 274.4 t/s | 67% |
| Qwen3-0.6B | CUDA Graph | 447.4 t/s | — | — |
| Qwen3-8B | CUDA Graph 28L | 57.4 t/s | 78.7 t/s | 73% |
| Qwen3.5-9B | Decode | 45.7 t/s | 71.4 t/s | 64% |
| GEMM prefill | M=128 | **13.0 TFLOPS** | 4.3 TFLOPS old | 3× ✅ |

---

## 3. Recent Decisions

### Session 28 — Qwen3.5-9B + GPU Sampling
- **GatedDeltaNet kernels**: `gated_delta_net.cu` — conv1d_update, recurrent_step, rmsnorm_gated
- **Qwen3.5-9B decode**: 32 layers (24 linear_attn + 8 full_attn), 45.7 t/s
- **Weight quantization**: `scripts/quantize_qwen35.py` — 11GB INT8 weights
- **head_dim=256 attention**: `attention_decode_kernel_v4` — chunked dot product for 256-dim heads
- **GPU sampling**: `sample_gpu.cu` — softmax + top-k + cuRAND, no host fallback
- **CUDA Graph**: Attempted, slower (480+ kernels, graph overhead). Removed.
- **INT4 mixed precision**: Tested, 0.36× slower (35 inst/byte unpack overhead). Dead end.
- **rmsnorm_gated deadlock bug**: `__shfl_xor_sync` inside `if (tid < 8)` — fixed
- **head_norm deadlock bug**: Same pattern — fixed in clean rewrite

### Session 27
- GPU argmax working, WMMA dequant confirmed correct, Docker/API done

---

## 4. Important Constraints

- `export PATH=/usr/local/cuda-13.3/bin:$PATH` before nvcc
- `compute_120a` required (not `compute_120`)
- `gemv_int8_warp` production GEMV — NOT `gemv_int8`
- `gemm_int8_wmma_fast` production GEMM — NOT `gemm_int8_dp4a`
- hashcat auto-restarts. `killall hashcat` before measurement.
- INT8 weight-bound: ~7.9 GB/token for Qwen3.5-9B. Cannot match Q4_K_M (5 GB).
- INT4/FP4 dead end on GB206 — 0.36× slower than INT8

---

## 5. Known Issues / Risks

1. **hashcat**: Auto-restarts, -45% throughput. `killall hashcat` before every measure.
2. **INT8 vs Q4_K_M gap**: Hardware limitation. INT8 reads 2× more data. No fix without sub-byte quant (dead end on GB206).
3. **text_generate repetition**: Greedy decode repeats. Use `-t 0.8 -k 40`.
4. **decode_int8_cgraph mismatch**: Pre-existing (warmup stream inconsistency). 182 t/s works fine.
5. **CUDA Graph not beneficial**: 480+ kernels per step, graph overhead exceeds launch savings.

---

## 6. Pending Tasks

| Priority | Task | Status | Notes |
|----------|------|--------|-------|
| ~~All~~ | ~~Qwen3.5-9B Mamba hybrid~~ | ✅ Done | 45.7 t/s, 64% of llama.cpp |
| ~~All~~ | ~~GPU softmax + top-k~~ | ✅ Done | sample_gpu.cu, all paths working |
| ~~All~~ | ~~Embed scale fix~~ | ✅ Done | Verified correct |
| — | Optimize linear attention kernels | Future | Current 45.7 t/s is weight-bound |
| — | text_generate for Qwen3.5-9B | Future | Needs tokenizer integration |

---

## 7. Suggested Next Actions

| Priority | Task | Rationale |
|----------|------|-----------|
| — | Optimize linear attention | Marginal gains — weight-bound |
| — | text_generate Qwen3.5-9B | End-to-end generation for new model |

---

## 8. Important Files / Commands

### Qwen3.5-9B
```bash
# Quantize weights
python3 scripts/quantize_qwen35.py /mnt/data/ai/hf/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a weights_int8_qwen35_9b

# Decode benchmark
./bench/decode_qwen35_9b weights_int8_qwen35_9b 20
```

### Qwen3-1.7B
```bash
# Decode throughput
./bench/decode_int8_batched_cgraph_attn 28 8   # 326.8 t/s (M=8)
./bench/decode_int8_cgraph 28                   # 182.8 t/s (M=1)
./bench/text_generate "The capital of France is" 15 -t 0.001  # "Paris" correct
```

### Build
```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
killall hashcat 2>/dev/null
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake --build build --parallel
```

### Verify
```bash
nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l  # expect 141
./bench/verify_gemm 128    # 7/7 PASS
```

---

## 9. Validation

| Check | Status | Notes |
|-------|--------|-------|
| Library | ✅ 141 symbols | +7 from GatedDeltaNet kernels |
| INT8 batched attn M=8 | ✅ 326.8 t/s | 119% of Q4_K_M |
| GEMM prefill | ✅ 13.0 TFLOPS | 3× vs old |
| text_generate | ✅ 126 t/s | "Paris" correct, GPU sampling |
| GEMM verify_gemm | ✅ 7/7 PASS | All layer-0 weights |
| Qwen3.5-9B decode | ✅ 45.7 t/s | 64% of llama.cpp (weight-bound) |
| Qwen3.5-9B quantization | ✅ 11GB | 250 INT8 + 105 raw params |

---

## 10. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-06-01 |
| branch | master |
| last_commit | `2e64aee` HANDOFF.md refresh session 27 |
| repo_state | 141 symbols. Qwen3.5-9B decode 45.7 t/s. GPU sampling. GatedDeltaNet. Docker. GEMM prefill 3×. |
| sessions_completed | 28 |

---

## META PROMPT

**Boot sequence**: Read `AGENTS.md` → `HANDOFF.md` → `git log --oneline -3` → `killall hashcat` → `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l` (expect 141) → `./bench/text_generate "The capital of France is" 15 -t 0.001` (expect "Paris").

**Verified state**: 141 symbols. 326.8 t/s batched attn (M=8). GEMM prefill 13.0 TFLOPS. Qwen3.5-9B: 45.7 t/s (64% of llama.cpp). GPU sampling. GatedDeltaNet kernels.

**DO NOT**:
- Use `compute_120` (must be `compute_120a`)
- Use `gemv_int8` in production (use `gemv_int8_warp`)
- Use `gemm_int8_dp4a` for M≥16 (use `gemm_int8_wmma_fast`)
- Benchmark without `killall hashcat`
- Attempt INT4/FP4 quantization (dead end on GB206 — 0.36× slower)
- Run test scripts that overwrite `weights_int8_*/` files in-place (use temp dirs)
- Trust the "simplified dequant" note in AGENTS.md — `gemm_int8_wmma_fast` is correct

**Update discipline**: Update HANDOFF.md only when materially new state. Keep deduplicated with AGENTS.md. Prefer bullets over prose.
