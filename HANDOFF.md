# HANDOFF.md — blackwell

Continuity doc. Read before acting.

---

## 1. Current Objective

**Maximize INT8 decode throughput vs llama.cpp (114 t/s Q4_K_M).** Current: **106.2 t/s** CUDA Graph (93% of target). All modes pass, all stubs implemented. 88 symbols.

---

## 2. Current Status

### Environment

| Component | Value |
|-----------|-------|
| GPU | RTX 5060 Ti (SM_120a, 36 SMs, ~500 GB/s GDDR7) |
| CUDA toolkit | 13.3.33 |
| Driver | 580.159.04 (open kernel modules) |
| Library | **88 symbols** in `build/libblackwell_kernels.a` |
| Kernel registers | `gemv_int8_kernel`: **57 regs**, 0 spills, 0 stack |
| Block size | `kINT8Block = 64` (was 256, improved occupancy +13.5%) |
| Branch | master |

### Throughput (28L, Qwen3-1.7B)

| Path | t/s | vs 114 target |
|------|-----|---------------|
| **INT8 CUDA Graph** | **106.2** | **93%** |
| INT8 per-kernel (28L) | 98.8 | 87% |
| INT8 per-kernel (4L scaled) | 87.9 | 77% |
| Mode D prefill+decode | 87 | 76% |
| Batched GEMV M=4 | 61 req/s | — |
| Speculative (M=4) | 227 batch t/s | 2.24× |
| NVF4 GEMV optimized | 122 GB/s | 1.24× |

---

## 3. Recent Decisions

| Decision | Rationale |
|----------|-----------|
| **kINT8Block 256→64** | 57-reg kernel needs smaller blocks for occupancy. CUDA Graph +13.5% (93.7→106.2), per-kernel +12% (80→88) |
| **Register pressure 57→48 failed** | dp4a chain requires 4×int32 weight + 4×int32 activation live simultaneously. 57 is hard floor. Split chains, unroll 1, maxrregcount all failed |
| **Per-row scales correct** | Phase G fixed 2D block scales (0.945 cosine → 0.999978). Scale layout `[N × K/16]`, access `W_t_scale[n_out * num_K_blks + kb]` |
| **gemm_int8_dp4a slow standalone** | 0.6× vs gemm_int8. Only useful where activations already INT8 |
| **CUDA Graph best path** | Eliminates launch overhead (560 kernels/layer). 7% faster than per-kernel |

---

## 4. Important Constraints

- `PATH=/usr/local/cuda-13.3/bin` before cmake. System nvcc is CUDA 12.0
- `compute_120a` required (not `compute_120`)
- `phase_a.cu` — DO NOT USE (won't link)
- `decode.cu` static `cudaMalloc` in `attention_decode_gqa` — warm-up before CUDA Graph capture
- No `<mutex>`/`<atomic>` in `.cu` — use `__sync` builtins
- `sizeof(__nv_fp4_e2m1)` = 1 byte (not 0.5)
- Weight matrices exceed L2 (32 MB) — architectural limit for single-token decode
- Scale access `n_out * num_K_blks` NOT `n_blk * num_K_blks` (per-row layout)
- GPU thermal throttles between kernel launches — continuous load needed for stable clocks

---

## 5. Known Issues

1. **7% gap to 114 t/s** — Register pressure (57) is hard floor for current kernel design. Would need shared memory tiling, FP16 weights, or tensor core MMA (undocumented SM_120) to close
2. **text_generate head_norm bug** — Pre-existing. "FAIL head_norm l=0". Unrelated to kernel changes
3. **FP32 text_generate** — Precision accumulation over 28 layers. BF16 format issue
4. **Spec decode CUDA Graph** — Warmup crash from `decode.cu` static `cudaMalloc`
5. **Docker/API packaging** — Not done
6. **11 bench .cu → 2 won't build** — `phase_a` (dead), `decode_full` (CUDA 13.3 compat)

---

## 6. Pending Tasks

- [ ] Package inference server (Docker, API wrapper)
- [ ] Wire gemm_int8_dp4a + quantize_int8 into inference_server.cu batched path
- [ ] Fix CUDA Graph capture for speculative decode (warm-up ordering)
- [ ] Fix text_generate head_norm bug
- [ ] Write inline PTX mxf4 MMA for NVF4 GEMV (potential 2-4× gain)

---

## 7. Important Files

| File | Purpose |
|------|---------|
| `src/kernels/gemv_int8.cu` | INT8 GEMV + GEMM + quantize (88 syms, **kINT8Block=64**) |
| `src/kernels/gemv_fp4_nv.cu` | NVF4 GEMV (opt: FP32 scales + FP16 acc) |
| `src/kernels/gemm.cu` | FP4 GEMM prefill (FP16 WMMA) |
| `src/kernels/decode.cu` | Attention decode + KV cache (thread-safe) |
| `src/kernels/fused_o_norm.cu` | RMSNorm + INT8/FP4 quant |
| `src/kernels/prefill.cu` | Prefill: quantize_int8 + gemm_int8_dp4a |
| `include/blackwell/kernels.h` | Public API (88 symbols) |
| `bench/decode_int8_cgraph.cu` | CUDA Graph benchmark (production path) |
| `bench/decode_full_int8.cu` | Per-kernel pipeline benchmark |
| `bench/inference_server.cu` | Production server |

### Commands
```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
./bench/decode_int8_cgraph 28              # CUDA Graph 106 t/s
./bench/decode_full_int8 4                 # Per-kernel 88 t/s (4L scaled)
./bench/inference_server 28 4 20 8         # Batched serving
nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l  # expect 88
```

---

## 8. Validation

| Check | Status |
|-------|--------|
| Library build | ✅ 88 symbols |
| INT8 GEMV kernel BW | ✅ 775 GB/s |
| INT8 GEMM (7 proj) | ✅ cosine=1.0 |
| INT8×INT8 __dp4a GEMM | ✅ cosine=1.0 |
| INT8 CUDA Graph 28L | ✅ 106.2 t/s |
| Per-kernel 28L | ✅ 98.8 t/s |
| Modes A-D | ✅ All pass |
| Speculative decode | ✅ 2.24× |
| NVF4 opt | ✅ 1.24× |
| Thread-safety | ✅ Atomic spin-locks |
| Stubs | ✅ 8/8 implemented |

---

## 9. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-05-30 |
| branch | master |
| last_commit | `b2a0638` docs: refresh HANDOFF.md for session 3 |
| repo_state | 88 symbols, uncommitted: `kINT8Block 256→64`, AGENTS.md/HANDOFF.md updates |
| sessions_completed | 4 (scale fix → stubs → dp4a+spec+NVF4 → block_size_opt) |

---

## META PROMPT

**Boot sequence**: `AGENTS.md` → `HANDOFF.md` → `git status --short` → `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l`

**Verified state**: 88 symbols. CUDA Graph **106.2 t/s** (93% of 114 target). Per-kernel 98.8 t/s. `kINT8Block=64`, `gemv_int8_kernel` uses 57 registers (hard floor). Per-row scales correct. text_generate has pre-existing head_norm bug.

**Next priorities**: Docker packaging > fix head_norm > speculative CUDA Graph > PTX mxf4 MMA

**DO NOT**:
- Use `compute_120` (must be `compute_120a`)
- Use `/usr/bin/ptxas` (CUDA 12.0)
- Use `phase_a.cu` (won't link)
- Use `<mutex>`/`<atomic>` in `.cu` files
- Use `n_blk * num_K_blks` for scale access (must be `n_out * num_K_blks`)
- Call `attention_decode_gqa` during `cudaStreamBeginCapture` without warm-up
- Expect 57→48 register reduction (dp4a chain is hard floor)

**Update discipline**: Only update HANDOFF.md when materially new. Keep AGENTS.md as architecture reference, HANDOFF.md as session continuity.
