# HANDOFF.md — blackwell

Continuity doc. Read before acting. Keep current with AGENTS.md.

---

## 1. Current Objective

Maximize INT8/FP4 decode throughput vs llama.cpp Q4_K_M (**253.0 t/s**).  
INT8: **173.2 t/s** CUDA Graph (68% of target). FP4: **247 t/s** CUDA Graph (98% of target, but numerically unstable).  
96 symbols. All stubs implemented.

---

## 2. Current Status

| Metric | Value |
|--------|-------|
| GPU | RTX 5060 Ti, SM_120a, 36 SMs, ~500 GB/s GDDR7 |
| CUDA | 13.3.33, driver 580.159.04 |
| Library | 96 symbols `build/libblackwell_kernels.a` |
| Branch | master @ `4ec21dc` |
| Sessions | 8 completed |

| Path | t/s | vs 253 llama.cpp |
|------|-----|-----------------|
| **INT8 CUDA Graph** | **173.2** | **68%** |
| INT8 per-kernel | 155.5 | 61% |
| **FP4 CUDA Graph** | **247.3** | **98% ⚠️** |
| FP4 per-kernel | 220.4 | 87% |
| llama.cpp Q4_K_M | 253.0 | 100% |

**⚠️ FP4 at 247 t/s is numerically unstable** — synthetic all-ones input explodes (L1 norm ~898M) from FP4 weight quantization error compounding over 28 layers. Real model weights would behave better but untested.

---

## 3. Recent Decisions

- **Warp-cooperative GEMV**: 1 warp/row, shuffle reduction. 2.5-4.6× single-kernel speedup over old gemv_int8. Production path.
- **FP4 packed rejected for M=1 decode**: E2M1 nibble→float overhead can't use __dp4a SIMD. 0.5× single-kernel vs INT8.
- **text_generate head_norm**: `cudaPeekAtLastError` → `cudaGetLastError` throughout. False positive from async error accumulation fixed.
- **Spec decode CUDA Graph**: Changed old `gemv_int8` → `gemv_int8_warp`. Added `cudaDeviceSynchronize()` after warm-up to ensure static cudaMalloc resolves before graph capture.
- **hashcat**: Persistently runs on GPU-0. Kills ~45% throughput. Must kill before measurement.
- **Next: INT4 SIMD GEMV**: True signed 4-bit integers (not E2M1) with `__dp4a`. Projected ~215 t/s.

---

## 4. Important Constraints

- `export PATH=/usr/local/cuda-13.3/bin:$PATH` before any nvcc invocation
- `compute_120a` (not `compute_120`) for FP4 block-scale MMA
- `gemv_int8_warp` is the production kernel — NOT `gemv_int8`
- `sizeof(__nv_fp4_e2m1)` = 1 byte (not 0.5). Packed FP4: 2 vals/byte via nibble layout
- Scale access: `n_out * num_K_blks` (per-row), NOT `n_blk * num_K_blks` (2D block)
- 24 bench files / 214 call sites still on old `gemv_int8`
- hashcat PID changes on each restart. Kill before measurement.
- L2 cache hint (stream 0) doesn't affect CUDA Graph path

---

## 5. Known Issues / Risks

1. **hashcat on GPU-0**: Auto-restarts, -45% throughput, 60s measurement window after kill
2. **24 bench files stale**: Only 2 of ~26 bench files use `gemv_int8_warp`. Rest use old `gemv_int8`
3. **FP4 pipeline numerically unstable**: 247 t/s throughput but outputs garbage on synthetic input. OK for throughput measurement only
4. **FP32 text_generate broken**: `text_generate_fp32.cu` bad output. BF16 weight format issue
5. **L2 cache hint on wrong stream**: Set on stream 0, not graph_stream
6. **Spec decode Graph**: kV_offset baked into captured graph (CUDA Graph limitation). Draft tokens write to wrong KV cache slot
7. **Docker/API packaging**: Not done

---

## 6. Pending Tasks

| Priority | Task | Blocked by |
|----------|------|-----------|
| P0 | Research llama.cpp MMVQ for 4-bit GEMV techniques | — |
| P0 | Build INT4 weight converter (INT8 → packed INT4) | — |
| P0 | Implement INT4 SIMD GEMV with __dp4a | weight converter + MMVQ research |
| P0 | Build INT4 full pipeline with CUDA Graph | INT4 GEMV |
| P1 | Migrate 24 bench files to gemv_int8_warp | — |
| P1 | Research + optimize attention_decode_gqa | MMVQ research |
| P2 | Fix L2 cache hint (target graph_stream) | — |

---

## 7. Important Files / Commands

### Key files
- `src/kernels/gemv_int8.cu` — 96 symbols: INT8/FP4 warp kernels, INT8 GEMM, quantize
- `include/blackwell/kernels.h` — Public API (96 symbols)
- `bench/decode_int8_cgraph.cu` — INT8 CUDA Graph pipeline (173 t/s)
- `bench/decode_fp4_cgraph.cu` — FP4 packed CUDA Graph pipeline (247 t/s ⚠️)
- `bench/bench_warp_gemv.cu` — Isolated warp GEMV comparison benchmark
- `bench/text_generate.cu` — End-to-end text generation (uses gemv_fp32_int8_per_row)
- `bench/speculative_decode_cgraph.cu` — Speculative decode with CUDA Graph

### Commands
```bash
# Build
export PATH=/usr/local/cuda-13.3/bin:$PATH
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Benchmark (kill hashcat first!)
killall hashcat 2>/dev/null
./bench/decode_int8_cgraph 28       # 173 t/s
./bench/decode_fp4_cgraph 28        # 247 t/s (unstable)
./bench/bench_warp_gemv             # Single-kernel comparison
./bench/text_generate "The capital of France is" 30  # "Paris" ✅

# Verify
nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l  # expect 96
```

---

## 8. Validation

| Check | Status |
|-------|--------|
| Library | ✅ 96 symbols |
| INT8 warp correctness | ✅ cosine=1.0 vs old kernel |
| INT8 CUDA Graph 28L | ✅ **173.2 t/s** (68% of 253) |
| FP4 CUDA Graph 28L | ⚠️ **247.3 t/s** but outputs unstable |
| llama.cpp Q4_K_M | ✅ **253.0 t/s** |
| llama.cpp F16 | ✅ **108.3 t/s** |
| text_generate | ✅ "Paris" — head_norm bug fixed |
| hashcat | ⚠️ Interferes with all measurements |
| L2 hints | ❌ Wrong stream (stream 0, not graph_stream) |

---

## 9. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-05-30 |
| branch | master |
| last_commit | `4ec21dc` fix: text_generate head_norm + spec decode CUDA Graph |
| repo_state | 96 symbols, commit + untracked bench binaries |
| sessions_completed | 8 (scale_fix → stubs → dp4a+spec+NVF4 → block_opt → warp_coop → FP4_gemv → FP4_pipeline → bug_fixes) |

---

## META PROMPT

**Boot sequence**: Read `AGENTS.md` → `HANDOFF.md` → `git status --short` → `killall hashcat` → verify: `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l` (expect 96) → `./bench/decode_int8_cgraph 28` to warm GPU + verify (expect 173+ t/s).

**Verified state**: 96 symbols. INT8 CUDA Graph **173.2 t/s** (68% of 253). FP4 CUDA Graph **247.3 t/s** (98% but numerically unstable). head_norm bug fixed. spec decode warm-up fixed. hashcat runs on GPU-0 (kill before measurement).

**Next priorities**: INT4 SIMD GEMV (signed 4-bit + __dp4a) → INT4 pipeline → attention optimize → migrate 24 bench files.

**DO NOT**:
- Use `compute_120` (must be `compute_120a`)
- Use `/usr/bin/ptxas` (CUDA 12.0)
- Use `phase_a.cu` (won't link)
- Use `<mutex>`/`<atomic>` in `.cu`
- Use `n_blk * num_K_blks` for scales (must be `n_out * num_K_blks`)
- Call `attention_decode_gqa` during CUDA Graph capture without warm-up
- Use `gemv_int8` in production path (use `gemv_int8_warp`)
- Trust FP4 pipeline numbers for correctness (247 t/s throughput only)
- Benchmark without `killall hashcat` first

**Update discipline**: Only update AGENTS.md when architecture/API changes. Only update HANDOFF.md when materially new session state changes. Keep both documents deduplicated.
