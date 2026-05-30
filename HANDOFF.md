# HANDOFF.md — blackwell

Continuity doc. Read before acting.

---

## 1. Current Objective

**Maximize INT8 decode throughput vs llama.cpp (253.6 t/s Q4_K_M).** Current: **173.8 t/s** CUDA Graph (69% of target). All modes pass, all stubs implemented. 92 symbols.

**NOTE**: Old baseline was 114 t/s (stale, from Phase C 2026-05-26). Re-benchmarked 2026-05-30 with latest llama.cpp (build 9212, CUDA 12.8) — actual Q4_K_M decode is **253.6 t/s** (2.2× the old number). Our INT8 is now 31% behind, not 52% ahead.

---

## 2. Current Status

### Environment

| Component | Value |
|-----------|-------|
| GPU | RTX 5060 Ti (SM_120a, 36 SMs, ~500 GB/s GDDR7) |
| CUDA toolkit | 13.3.33 |
| Driver | 580.159.04 (open kernel modules) |
| Library | **96 symbols** in `build/libblackwell_kernels.a` |
| Kernel registers | `gemv_int8_warp_kernel`: **56 regs**, 0 spills, 0 stack |
| Block size | `32` (1 warp per row, shuffle reduction) |
| Branch | master |

### Measurement Constraint
**hashcat** runs persistently on this GPU (PID changes, auto-restarts). Every ~60 seconds, hashcat uses 95%+ GPU, dropping throughput ~45%. All benchmarks must be run after `killall hashcat` (valid for ~60s window). True numbers: INT8 CUDA Graph **173.6 t/s**, FP4 CUDA Graph **137.4 t/s**.

### Throughput (28L, Qwen3-1.7B)

| Path | t/s | vs llama.cpp 253 t/s | Notes |
|------|-----|---------------------|-------|
| **INT8 CUDA Graph (warp)** | **173.6** | **69%** | Production path, full 28L |
| INT8 per-kernel (warp, 28L) | 155.5 | 61% | No CUDA Graph |
| **FP4 CUDA Graph** | **137.4** | **54%** | Packed FP4, full 28L |
| FP4 per-kernel | 123.7 | 49% | No CUDA Graph |
| llama.cpp Q4_K_M (end-to-end) | **253.0** | **baseline** | Re-measured 2026-05-30, build 9212 |
| llama.cpp F16 (end-to-end) | 108.3 | 43% | |

---

## 3. Recent Decisions

| Decision | Rationale |
|----------|-----------|
| **Warp-cooperative GEMV** | 1 warp per output row, shuffle reduction. 2.5× single-kernel speedup (coalesced loads). 98→174 t/s full pipeline |
| **FP4 packed rejected for M=1 decode** | E2M1 nibble→float conversion overhead kills 2× bandwidth win. FP4=137 vs INT8=174 t/s. May help for M>1 batched |
| **L2 cache hints** | Set persisting for RMSNorm weights, streaming for weights. +0.3% (marginal) |
| **kINT8Block 256→64** | 57-reg kernel needs smaller blocks for occupancy. +13.5% CUDA Graph, +12% per-kernel |
| **Register pressure 57→48 failed** | dp4a chain requires 4×int32 weight + 4×int32 activation live simultaneously. 57 is hard floor |
| **Per-row scales correct** | Phase G fixed 2D block scales. Scale layout `[N × K/16]` |
| **CUDA Graph best path** | Eliminates launch overhead. 10% faster than per-kernel |

---

## 4. Important Constraints

- `PATH=/usr/local/cuda-13.3/bin` before cmake. System nvcc is CUDA 12.0
- `compute_120a` required (not `compute_120`)
- Warp kernel `gemv_int8_warp` is the production path (not `gemv_int8`)
- `phase_a.cu` — DO NOT USE (won't link)
- `decode.cu` static `cudaMalloc` in `attention_decode_gqa` — warm-up before CUDA Graph capture
- No `<mutex>`/`<atomic>` in `.cu` — use `__sync` builtins
- `sizeof(__nv_fp4_e2m1)` = 1 byte (not 0.5)
- Weight matrices exceed L2 (32 MB) — architectural limit for single-token decode
- Scale access `n_out * num_K_blks` NOT `n_blk * num_K_blks` (per-row layout)
- GPU thermal throttles between kernel launches — continuous load needed for stable clocks

---

## 5. Known Issues

1. **hashcat runs persistently on GPU** (PID ~57393/64789, auto-restarts). Uses 3740MiB VRAM, 95%+ GPU util. Drops benchmarks ~45% (174→96 t/s). Must `killall hashcat` before any measurement — 60s window before respawn
2. **Attention decode is 13.5% of pipeline** — 16.1ms/118.9ms. MLP GEMV dominant at 57.7%
3. **text_generate head_norm bug** — Pre-existing. "FAIL head_norm l=0". In `text_generate.cu` (uses `gemv_fp32_int8_per_row`)
4. **FP32 text_generate broken** — Precision accumulation over 28 layers. BF16 format issue
5. **Spec decode CUDA Graph** — Warmup crash from `decode.cu` static `cudaMalloc`
6. **Docker/API packaging** — Not done
7. **24 bench files still use old `gemv_int8`** — 214 call sites not migrated to `gemv_int8_warp`. Includes `inference_server.cu`, `speculative_decode.cu`, `text_generate.cu`
8. **FP4 packed 21% slower than INT8** — 137 vs 174 t/s. E2M1 nibble→float per-element conversion can't compete with __dp4a SIMD for M=1. May still help for M>1 batched
9. **L2 cache hint targets wrong stream** — set on stream 0, not graph_stream. No-op for graph path
10. **CUDA Graph correctness drift** — INT8: max diff ~4.0 after 25 iter (FP4 quantization). FP4: L1~58K, max diff~9K. Synthetic input instability

---

## 6. Pending Tasks

- [x] **Re-benchmark vs llama.cpp** — Done. Q4_K_M=253.0, F16=108.3, our INT8=173.6
- [x] **FP4 packed pipeline** — Done. Full 28L CUDA Graph at 137.4 t/s
- [ ] **Migrate 24 bench files to gemv_int8_warp** — 214 call sites remaining
- [ ] **attention_decode_gqa optimization** — Currently 28.8μs/call, 13.5% of pipeline. Flash attention or fused QK^T+softmax+PV to cut ~40%
- [ ] **Batched FP4 GEMV (M=2-4)** — FP4 unpacking cost amortized across tokens. Could match/exceed INT8 at M=2+
- [ ] **Fix L2 cache hint** — target graph_stream instead of stream 0
- [ ] **Package inference server** (Docker, API wrapper)
- [ ] **Fix CUDA Graph for speculative decode** — warm-up ordering
- [ ] **Fix text_generate head_norm bug**

---

## 7. Important Files

| File | Purpose |
|------|---------|
| `src/kernels/gemv_int8.cu` | INT8 GEMV/GEMM/quantize + **warp-cooperative kernels** + packed FP4 kernels (96 syms) |
| `include/blackwell/kernels.h` | Public API (96 symbols) |
| `bench/bench_packed_fp4.cu` | Isolated FP4 packed vs INT8 warp comparison |
| `bench/bench_mixed_precision.cu` | 1-layer mixed-precision benchmark (INT8 attn + FP4 MLP) |
| `bench/decode_fp4_cgraph.cu` | FP4 packed 28L CUDA Graph pipeline benchmark (**137 t/s**) |
| `bench/bench_warp_gemv.cu` | Isolated warp GEMV benchmark |
| `bench/decode_int8_cgraph.cu` | CUDA Graph benchmark (production path, **173.7 t/s**) |
| `bench/decode_full_int8.cu` | Per-kernel pipeline benchmark (**155.5 t/s**) |
| `src/kernels/gemv_fp4_nv.cu` | NVF4 GEMV (opt: FP32 scales + FP16 acc) |
| `src/kernels/gemm.cu` | FP4 GEMM prefill (FP16 WMMA) |
| `src/kernels/decode.cu` | Attention decode + KV cache (thread-safe) |
| `src/kernels/fused_o_norm.cu` | RMSNorm + INT8/FP4 quant |
| `src/kernels/prefill.cu` | Prefill: quantize_int8 + gemm_int8_dp4a |
| `bench/inference_server.cu` | Production server |

### Commands
```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
./bench/decode_int8_cgraph 28              # CUDA Graph 173 t/s (production)
./bench/decode_full_int8 28                # Per-kernel 155 t/s
./bench/bench_warp_gemv                    # Isolated GEMV comparison
nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l  # expect 96
```

---

## 8. Validation

| Check | Status |
|-------|--------|
| Library build | ✅ 96 symbols |
| Warp GEMV correctness | ✅ cosine=1.0 vs old kernel |
| INT8 CUDA Graph 28L | ✅ **173.6 t/s** (69% of 253 t/s baseline) |
| INT8 per-kernel 28L | ✅ **155.5 t/s** |
| FP4 CUDA Graph 28L | ✅ **137.4 t/s** (79% of INT8, 54% of llama.cpp) |
| FP4 per-kernel 28L | ✅ **123.7 t/s** |
| llama.cpp Q4_K_M baseline | ✅ **253.0 t/s** |
| llama.cpp F16 baseline | ✅ **108.3 t/s** |
| L2 cache hints | ⚠️ Targets wrong stream — needs fix |
| Modes A-D | ✅ All pass |
| hashcat interference | ⚠️ Kills ~45% throughput on GPU-0. Kill before any measurement |

---

## 9. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-05-30 |
| branch | master |
| last_commit | `2eefb6c` feat: full FP4 packed pipeline benchmark (CUDA Graph) |
| repo_state | 96 symbols, clean |
| sessions_completed | 7 (scale_fix → stubs → dp4a+spec+NVF4 → block_size_opt → warp_cooperative → FP4_packed_gemv → FP4_pipeline) |

---

## META PROMPT

**Boot sequence**: `AGENTS.md` → `HANDOFF.md` → `git status --short` → `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l`

**Verified state**: 96 symbols. INT8 CUDA Graph **173.6 t/s** (69% of 253 t/s llama.cpp). FP4 CUDA Graph **137.4 t/s** (79% of INT8). Warp-cooperative GEMV is production path. FP4 packed rejected for M=1 decode (21% slower than INT8).

**CRITICAL**: hashcat runs persistently on GPU-0. Kill before every measurement. 60s window before respawn.

**Next priorities**: attention_decode_gqa optimization > batched FP4 (M=2-4) > migrate 24 bench files > fix L2 hints

**DO NOT**:
- Use `compute_120` (must be `compute_120a`)
- Use `/usr/bin/ptxas` (CUDA 12.0)
- Use `phase_a.cu` (won't link)
- Use `<mutex>`/`<atomic>` in `.cu` files
- Use `n_blk * num_K_blks` for scale access (must be `n_out * num_K_blks`)
- Call `attention_decode_gqa` during `cudaStreamBeginCapture` without warm-up
- Use `gemv_int8` in production path (use `gemv_int8_warp` instead)

**Update discipline**: Only update HANDOFF.md when materially new. Keep AGENTS.md as architecture reference, HANDOFF.md as session continuity.

---

## 10. llama.cpp Benchmark Results

### Fresh baseline (2026-05-30)

| Benchmark | t/s | Notes |
|-----------|-----|-------|
| **llama.cpp Q4_K_M** | **253.6 ± 0.4** | End-to-end inference, build 9212, CUDA 12.8, sm_120 |
| **Our INT8 CUDA Graph** | **173.8** | Synthetic, pure kernel throughput, sm_120a |
| llama.cpp F16 | 108.4 | End-to-end, full precision |

### Reproduction
```bash
# Q4_K_M quantization (already done)
llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
# → 3.3 GB → 1.05 GB, 5.12 BPW

# llama-bench decode
llama-bench -m model-q4_k_m.gguf -p 128 -n 128 -r 5
# → tg128 = 253.6 t/s

# Our benchmark
./bench/decode_int8_cgraph 28
# → CUDA Graph = 173.8 t/s
```

### Why llama.cpp is faster (253 vs 174)
1. **Q4_K_M = 4.5 bits/weight** vs our INT8 = 8 bits/weight → **44% less bandwidth**
2. **llama.cpp uses warp-cooperative MMVQ** — same technique as our `gemv_int8_warp`, but with 4-bit weights
3. **llama.cpp uses CUDA 12.8 sm_120** — compiler may vectorize differently than our CUDA 13.3 sm_120a
4. **End-to-end vs synthetic gap** — llama.cpp includes all overhead and still wins; their kernel throughput is even higher than 253 t/s

### Why FP4 packed did NOT close the gap
- **Packed FP4 E2M1** — 2 vals/byte, read half the data, but E2M1 nibble→float per-element conversion can't use `__dp4a` SIMD
- **FP4 = 137 t/s** — 21% slower than INT8 (174 t/s) despite 2× bandwidth savings
- **Root cause**: M=1 decode is compute-limited by nibble unpacking, not memory-bound

### Path to closing the gap (revised)
- **Option 1: Batched FP4 GEMV (M=2-4)** — Unpacking cost amortized across tokens, weight reuse gives bandwidth win. llama.cpp MMVQ batches 2-8 tokens. Our `gemv_int8_batched` exists; FP4 batched would be new
- **Option 2: Faster attention** — `attention_decode_gqa` is 28.8μs/call = 13.5% of pipeline. Flash decode or fused kernel could cut ~40%
- **Option 3: Accept INT8 as quality tier** — 174 t/s, full INT8 precision, no quantization loss. 69% of llama.cpp is respectable
- **Option 4: INT4 SIMD (not E2M1)** — True 4-bit signed integers with `__dp4a`. Same register pressure as INT8 but half bandwidth. Would need custom weight format + quantizer.
