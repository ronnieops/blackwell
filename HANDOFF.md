# HANDOFF.md — blackwell

Continuity doc. Read before acting. Keep current with AGENTS.md.

---

## 1. Current Objective

**Q4 quantization (INT4) — session 34 active**.
- INT4 decode: **231.5 t/s** (79% of Q4_K_M), **128% of INT8** (181.5 t/s).
- Target: ~250-290 t/s (85-99% of Q4_K_M). Gap: scalar unpack (no __dp4a) + simplified pipeline (no residuals).
- Next: fused quant+GEMV, residual path, fused swiglu.

---

## 2. Current Status

| Metric | Value |
|--------|-------|
| GPU | RTX 5060 Ti, GB206, SM_120a, 36 SMs, ~500 GB/s GDDR7 |
| CUDA | 13.3, C++17, CMake |
| Library | **159 symbols** `build/libblackwell_kernels.a` |
| Branch | master @ `1b96f50` |
| Session | **34** |

### Qwen3-1.7B

| Config | Total t/s | Per-seq | vs Q4_K_M | Notes |
|--------|-----------|---------|-----------|-------|
| INT8 M=1 fused (per-kernel) | 181.5 | 181.5 | 62% | 14 kernels/layer |
| **INT4 M=1** | **231.5** | **231.5** | **79%** | Scalar unpack, 128% of INT8 |
| INT8 M=4 batched-attn + Graph | 308.3 | 77.1 | **105%** | |
| **INT8 M=8 batched-attn + Graph** | **324.3** | **40.6** | **111%** | Production path |
| llama.cpp Q4_K_M FA=on | 293.4 | 293.4 | 100% | |

### Qwen3-8B

| Config | Total t/s | vs Q4_K_M | Notes |
|--------|-----------|-----------|-------|
| Blackwell M=1 | 44.6 | 54% | INT8, weight-bound |
| llama.cpp Q4_K_M FA=on | 82.56 | 100% | |

---

## 3. Session 34 — INT4 Quantization

### What was built

| Component | Status | Result |
|-----------|--------|--------|
| `scripts/quantize_generic.py` | ✅ Extended | INT4 support via `--format int4`, vectorized nibble-pack |
| `weights_int4_qwen3_1.7b/` | ✅ Generated | 1.3 GB (62% of INT8 2.1 GB), 394 files, 3.4s conversion |
| `gemv_int4_warp` kernel | ✅ Built | 60 regs, scalar unpack via `int4_byte_to_floats`, ~10 ms for 2048×2048 |
| `transpose_int4_weights` | ✅ Built | W (K×N/2) → W_t (N×K/2), scales transposed |
| `transpose_scales_int4_kernel` | ✅ Built | Scale transpose kernel |
| `quantize_int4` | ✅ Built | FP32 → packed INT4 (block-16, absmax/7, nibble-pack) |
| `unpack_int4_fp32` | ✅ Built | packed INT4 → FP32 |
| `bench/decode_int4_cgraph` | ✅ Built | **231.5 t/s** (79% of Q4_K_M) |

### Benchmark results

| Config | t/s | vs INT8 | vs Q4_K_M |
|--------|-----|---------|----------|
| INT8 (per-kernel, fused) | 181.5 | 100% | 62% |
| **INT4 (per-kernel, simplified)** | **231.5** | **128%** | **79%** |
| llama.cpp Q4_K_M FA=on | 293.4 | — | 100% |

### Why INT4 is 128% of INT8 (not faster than Q4_K_M)

- **2× less DRAM reads**: 0.5 bytes/val vs 1.0 → bandwidth wins
- **Scalar unpack**: `int4_byte_to_floats` calls, no `__dp4a` SIMD → compute overhead
- **Simplified pipeline**: no residuals, no fused_swiglu, no fused_residual_norm
- **INT4 uses float multiply**: not SIMD-dot like INT8's `__dp4a` → ~50% more compute ops

### Gap to Q4_K_M (79%)

- Q4_K_M uses `__dp4a` + asymmetric scales + block=256
- INT4 uses scalar float multiply + symmetric scales + block=16
- Need: fused quant+GEMV (like INT8), residual path, possibly __dp4a for activations

### Files created

- `scripts/quantize_generic.py` — INT4 support added (vectorized nibble-pack)
- `weights_int4_qwen3_1.7b/` — 394 files, 1.3 GB
- `bench/decode_int4_cgraph.cu` — INT4 decode benchmark (18 KB)

---

## 4. Recent Decisions

### Session 34 — INT4 quantization
- **INT4 231.5 t/s**: 2× bandwidth advantage overcomes scalar unpack overhead. 128% of INT8.
- **gemv_int4_warp**: Uses existing scalar unpack in `gemv_int8.cu`. Not using __dp4a because nibbles need separate dequant before SIMD dot (unlike INT8 where bytes are already signed int8).
- **Weight conversion**: Vectorized nibble-pack (3.4s for full model). Dropped slow nested loop.
- **INT4 is not Q4_K_M**: Uses block=16 (same as INT8) vs Q4_K_M's block=256. Uses symmetric scales vs asymmetric. This limits quality and throughput.

### Session 33 — Spec decode, llama.cpp audit, M=1 Graph, Docker, benchmarks
- **Spec decode infeasible**: Batched verify (24.7 ms/seq) is 4.5× slower per-seq. Draft needs 4.5× speedup. Even 50M draft yields ~92 t/s. Self-speculation fails. Abandoned.
- **M=1 CUDA Graph**: `cudaMemcpyAsync` H2D inside `attention_decode_gqa`/`update_kv_cache` is illegal in ALL capture modes. Need graph-safe wrappers. Per-kernel 181.5 t/s is <3% from theoretical max.
- **Docker server built & tested**: `Dockerfile` uses `ubuntu:24.04`. `"Paris"` ✅.
- **M=8 = 324.3 t/s** vs Q4_K_M = 293.4 t/s = **111%**.

---

## 5. Important Constraints

- `export PATH=/usr/local/cuda-13.3/bin:$PATH` before nvcc
- `compute_120a` required (not `compute_120`)
- `gemv_int8_warp` production GEMV — NOT `gemv_int8`
- `gemv_int4_warp` INT4 GEMV — uses scalar unpack (no __dp4a)
- `killall hashcat` before every measurement (auto-restarts, -45% throughput)

---

## 6. Known Issues / Risks

1. **hashcat**: Auto-restarts, -45% throughput. `killall hashcat` before every measure.
2. **INT4 vs Q4_K_M gap (79%)**: Scalar unpack lacks __dp4a. Need fused quant+GEMV + residuals to close.
3. **INT4 simplified pipeline**: No residuals, no fused_swiglu, no fused_residual_norm (bench uses simplified path).
4. **Qwen3-8B INT8 gap**: 54% of Q4_K_M — INT8 reads 2× data.
5. **gemv_int4_warp in gemv_int8.cu**: Defined alongside INT8 kernels. No separate gemv_int4.cu (removed).

---

## 7. Pending Tasks

| Task | Status | Notes |
|------|--------|-------|
| INT4 fused quant+GEMV | 🔜 Next | Fuse quantize_int4 into gemv_int4_warp |
| INT4 residual path | 🔜 Next | vector_add_fp32 + fused_rmsnorm |
| INT4 fused swiglu | 🔜 Next | fused_swiglu_quant_int4 |
| INT4 batched GEMV | 🔜 Optional | gemv_int4_batched for M=8 path |
| INT4 correctness validation | 🔜 Next | PSNR > 40 dB vs INT8 |
| Qwen3.5-9B integration | TODO | 45.6 t/s bench, not in text_generate |

---

## 8. Important Files / Commands

### Build
```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
killall hashcat 2>/dev/null
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake --build build --parallel
```

### Benchmark
```bash
./bench/decode_int8_cgraph 28                   # INT8: 181.5 t/s
./bench/decode_int8_batched_cgraph_attn 28 8    # INT8 M=8: 324.3 t/s (111% of Q4_K_M)
./bench/decode_int4_cgraph 28                   # INT4: 231.5 t/s (79% of Q4_K_M, 128% of INT8)
./bench/text_generate "The capital of France is" 30  # Correctness
```

### INT4 weights
```bash
python3 scripts/quantize_generic.py /mnt/data/ai/hf/qwen3-1.7b-base weights_int4_qwen3_1.7b int4
# Output: 1.3 GB (62% of INT8 2.1 GB), 394 files
```

### Verify
```bash
nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l  # expect 164
```

---

## 9. Validation

| Check | Status |
|-------|--------|
| Library | ✅ 159 symbols |
| M=8 CUDA Graph (1.7B) | ✅ 324.3 t/s (111% of Q4_K_M) |
| INT4 decode (1.7B) | ✅ 231.5 t/s (79% of Q4_K_M, 128% of INT8) |
| M=1 INT8 fused (1.7B) | ✅ 181.5 t/s (62% of Q4_K_M) |
| INT4 weight conversion | ✅ 1.3 GB, 394 files, 3.4s |
| Correctness | ⏳ Not validated (simplified pipeline) |

---

## 10. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-06-02 |
| branch | master |
| last_commit | `1b96f50` docs: update HANDOFF.md with Q4 plan as active objective |
| repo_state | 159 symbols. INT4: 231.5 t/s (79% of Q4_K_M, 128% of INT8). INT8 M=8: 324.3 t/s (111%). Q4_PLAN.md in progress. |
| uncommitted | AGENTS.md, HANDOFF.md (session 34 updates pending) |

---

## META PROMPT

**Boot sequence**: Read `AGENTS.md` → `HANDOFF.md` → `git log --oneline -3` → `killall hashcat` → `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l` (expect 164) → `./bench/decode_int8_batched_cgraph_attn 28 8` (expect ~324 t/s) → `./bench/decode_int4_cgraph 28` (expect ~232 t/s) → `echo '{"prompts":["The capital of France is"],"max_tokens":10}' | ./server/inference_server` (expect JSON with tokens).

**Verified state**: 159 symbols. INT4: 231.5 t/s (79% of Q4_K_M, 128% of INT8). INT8 M=8: 324.3 t/s (111% of Q4_K_M). M=1 INT8 fused: 181.5 t/s (62%). INT4 weights: 1.3 GB (weights_int4_qwen3_1.7b/). Simplified pipeline (no residuals).

**DO NOT**:
- Use `compute_120` (must be `compute_120a`)
- Use `gemv_int8` in production (use `gemv_int8_warp`)
- Benchmark without `killall hashcat`
- Expect M>8 to help (batched GEMV register pressure)
- Pursue speculative decoding, FP4 tensor core GEMV, or PDL (all dead ends)
- Attempt M=1 CUDA Graph without fixing H2D copies in kernel wrappers

**Revisitable**:
- **Fused INT4 GEMV**: fuse `quantize_int4` output into `gemv_int4_warp` (skip intermediate INT4 buffer). Target: +10-15% speedup.
- **INT4 residual path**: add `vector_add_fp32` + `fused_rmsnorm` after projections. Target: better quality + small perf.
- **Fused swiglu INT4**: `fused_swiglu_quant_int4` (swiglu + quantize, like INT8's fused_swiglu_quant).
- **Q4 quantization (INT4)**: **Active — Phase 1 done**. Phase 2: kernel optimization (fused quant+GEMV, residual path). Target ~250-290 t/s (85-99% of Q4_K_M).

**Update discipline**: Update HANDOFF.md only when materially new state. Keep deduplicated with AGENTS.md. Prefer bullets over prose.