# HANDOFF.md — blackwell

Continuity doc. Read before acting.

---

## 1. Current Objective

**INT8 production-complete + pre-quantized GEMM + NVF4 optimization.** All modes pass, all stubs implemented. 88 library symbols. Next: deploy (Docker/API) or continued feature work.

---

## 2. Current Status

### Environment

| Component | Value |
|-----------|-------|
| Driver | 580.159.04 (open kernel modules) |
| CUDA toolkit | 13.3.33 |
| GPU | RTX 5060 Ti (SM_120a, 36 SMs) |
| Library | **88 symbols** in libblackwell_kernels.a |
| Thread-safety | Atomic spin-locks for init (decode.cu) |
| Branch | master |

### Throughput (28L, Qwen3-1.7B)

| Path | t/s | Notes |
|------|-----|-------|
| INT8 CUDA Graph | **99** | ✅ Best single-user (with RoPE) |
| INT8 per-kernel | **92-94** | ✅ |
| INT8 pipeline (4L scaled) | **92.5** | ✅ decode_full_int8 |
| Mode D prefill+decode | **87** | ✅ INT8 GEMM, 165ms prefill |
| Batched GEMV M=4 | **61** req/s | ✅ |
| Speculative (M=4) | 227 batch t/s | ✅ 2.24× vs autoregressive (FP4 path) |
| NVF4 GEMV (original) | 98 GB/s | ✅ Scalar FP4, compiler generates mxf4 MMA |
| NVF4 GEMV (optimized) | 122 GB/s | ✅ FP32 scales + FP16 acc, 1.24× |

### vs llama.cpp baseline (114 t/s Q4_K_M)

INT8 CUDA Graph: **87%** of baseline.

text_generate: **Correct** ("Paris").

---

## 3. Recent Decisions

| Decision | Rationale |
|----------|-----------|
| **gemm_int8_dp4a** added | Pre-quantized INT8×INT8 GEMM with __dp4a + 4×4 tiling. 7/7 PASS. 0.6× standalone vs old FP32×INT8 path, but eliminates FP32 activation memory traffic |
| **quantize_int8** added | Fused absmax scales + INT8 pack (1 kernel). Grid=blk/32-threads, warp shuffle reduce |
| **prefill.cu**→pre-quantized | Uses quantize_int8 + gemm_int8_dp4a for Q/K/V/O. Removes on-the-fly quant from old scalar GEMM |
| **Spec decode: FP4 path kept** | INT8-only path 14% slower. Bottleneck is GEMV BW, not kernel launches. FP4→INT8 conversion overlaps with GEMV work |
| **NVF4: FP32 scales+FP16** | ue4m3_to_float eliminated via pre-converted FP32 scales. Inner loop uses __hfma (FP16) for 2× throughput. 1.24× gain, 122 GB/s |
| CUDA Graph still best for single-user | 87→99 t/s (~14%). Would give similar for spec decode but warmup needs decode.cu static alloc ordering |

---

## 4. Important Constraints

- `PATH=/usr/local/cuda-13.3/bin` before cmake. System `/usr/bin/nvcc` is CUDA 12.0
- `compute_120a` required (not `compute_120`) — FP4 MMA needs 12Xa
- `phase_a.cu`: DO NOT USE — won't link (needs `gemv_fp4_v3`, `gemv_fp4_batched`)
- `decode.cu` has static `cudaMalloc` inside `attention_decode_gqa` → warm-up required before CUDA Graph capture
- No `<mutex>`/`<atomic>` in `.cu` files — CUDA 13.3 lacks. Use GCC `__sync` builtins
- `sizeof(__nv_fp4_e2m1)` = 1 byte (not 0.5)
- All weight matrices exceed L2 (32 MB) — architectural limit for single-token decode

---

## 5. Known Issues / Risks

1. **Throughput regression** — 15% drop from re-quantization (correct scales). Cost of correctness.
2. **FP32 text_generate** — Precision accumulation over 28 layers. Not fixable — BF16 format.
3. **gemm_int8_dp4a** — 0.6× standalone vs `gemm_int8`. Pre-quantized format needs extra pass. Only wins where activations already INT8.
4. **Speculative decode CUDA Graph** — Warmup crash from `decode.cu` static `cudaMalloc` during stream capture. Fix: warm-up all kernels before `cudaStreamBeginCapture`.
5. **Docker/API packaging** — Not yet done.
6. **11 bench .cu → 2 won't build** — `phase_a` (dead), `decode_full` (CUDA 13.3 API compat).

---

## 6. Pending Tasks

- [ ] Package inference server (Docker, API wrapper)
- [ ] Wire gemm_int8_dp4a + quantize_int8 into inference_server.cu batched path
- [ ] Fix CUDA Graph capture for speculative decode (warm-up ordering)
- [ ] Write inline PTX mxf4 MMA for NVF4 GEMV (potential 2-4× gain)

---

## 7. Important Files

| File | Purpose |
|------|---------|
| `src/kernels/gemv_int8.cu` | INT8 GEMV + GEMM, gemm_int8_dp4a, quantize_int8 (88 syms) |
| `src/kernels/gemv_fp4_nv.cu` | NVF4 GEMV (original + opt: FP32 scales + FP16 acc) |
| `src/kernels/gemm.cu` | FP4 GEMM via FP16 WMMA (prefill) |
| `src/kernels/decode.cu` | Attention decode + seq_pos (thread-safe) |
| `src/kernels/fused_o_norm.cu` | RMSNorm + INT8/FP4 quant |
| `src/kernels/prefill.cu` | Prefill: uses quantize_int8 + gemm_int8_dp4a |
| `include/blackwell/kernels.h` | Public API (88 symbols) |
| `bench/inference_server.cu` | Production server (CUDA Graph + batched + Mode D) |
| `bench/verify_gemm.cu` | GEMM correctness (7/7) |
| `bench/verify_gemm_dp4a.cu` | __dp4a GEMM correctness (7/7) |
| `bench/bench_gemm_dp4a.cu` | Old vs new GEMM throughput |
| `bench/bench_gemv_fp4.cu` | NVF4 old vs opt throughput |
| `gemv_fp4_nv.ptx` | Compiler-generated PTX with mxf4 MMA instruction |

### Commands
```bash
# Build
export PATH=/usr/local/cuda-13.3/bin:$PATH
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel

# Run
./bench/text_generate "The capital of France is" 30
./bench/inference_server 28 4 20 8
./bench/verify_gemm 128            # 7/7 FP32×INT8 PASS
./bench/verify_gemm_dp4a 128       # 7/7 INT8×INT8 __dp4a PASS
./bench/bench_gemm_dp4a 128        # old vs new GEMM throughput
./bench/bench_gemv_fp4             # NVF4 old vs opt GEMV throughput
./bench/decode_full_int8 4         # 92.5 t/s scaled

# Verification
nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l   # expect 88
```

---

## 8. Validation

| Check | Status |
|-------|--------|
| Library build | ✅ 88 symbols |
| INT8 GEMV | ✅ 775 GB/s kernel |
| INT8 GEMM (7 proj) | ✅ cosine=1.0, max_err<2e-5 |
| INT8×INT8 __dp4a GEMM (7 proj) | ✅ cosine=1.0, max_err<2e-5 |
| INT8 pipeline 28L | ✅ 92.5 t/s |
| INT8 CUDA Graph + RoPE | ✅ 99 t/s |
| text_generate | ✅ Correct ("Paris") |
| Modes A-D | ✅ All throughput targets |
| Speculative decode | ✅ 2.24× (FP4 path) |
| NVF4 opt 1.24× | ✅ benchmarked |
| Thread-safety | ✅ Atomic spin-locks |
| Stub functions | ✅ 8/8 implemented |

---

## 9. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-05-29 |
| branch | master |
| last_commit | `8a46590` feat: NVF4 GEMV optimization |
| repo_state | 88 symbols, clean tree (untracked bench binaries + reports) |
| sessions_completed | 3 (scale fix → stubs/docs → dp4a+spec+NVF4) |

---

## META PROMPT

**Boot sequence**: Read `AGENTS.md` → `HANDOFF.md` → `git status --short` → `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l`.

**Current state**: 88 symbols. INT8 production at 99 t/s (CUDA Graph + RoPE). Pre-quantized GEMM (`gemm_int8_dp4a` + `quantize_int8`) verified 7/7. NVF4 optimized 1.24× (122 GB/s pre). Speculative decode 2.24× via FP4 path. Prefill uses `quantize_int8` + `gemm_int8_dp4a`.

**What to do next**: Docker/API packaging is highest-priority pending task. Or wire `gemm_int8_dp4a` into `inference_server.cu` batched path. Or fix CUDA Graph for speculative decode.

**Critical things to NOT do**:
- Don't use `compute_120` — must be `compute_120a`
- Don't use `/usr/bin/ptxas` — CUDA 12.0
- Don't use `phase_a.cu` — will not link
- Don't use NVF4 MMA for GEMV — scale layout mismatch (use inline PTX for revival)
- Don't use `<mutex>`/`<atomic>` in .cu files
- Don't expect FP32 text_generate to match INT8 — precision accumulation over 28 layers
- Don't use `n_blk * num_K_blks` for scale access — must use `n_out * num_K_blks`
- Don't call functions with internal `cudaMalloc` (e.g., `attention_decode_gqa` first call) during `cudaStreamBeginCapture` — warm-up needed

**Update discipline**: Only update HANDOFF.md when materially new info or decisions. Prefer in-place edits. Keep AGENTS.md as architectural reference, HANDOFF.md as session continuity.
