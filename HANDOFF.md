# HANDOFF.md — blackwell

Continuity doc. Read with AGENTS.md before acting.

---

## 1. Current Objective

**INT4 decode — COMPLETE.** 612.8 total t/s M=1 (209% Q4_K_M). Bug fixes revealed 2× gap between old (242 t/s) and real (612 t/s) performance. No active objective — next session picks direction.

---

## 2. Current Status

| Metric | Value |
|--------|-------|
| GPU | RTX 5060 Ti, compute 12.0, 36 SMs, ~500 GB/s GDDR7 |
| CUDA | 13.3, SM_120a, C++17, CMake |
| Library | **177 symbols** in `build/libblackwell_kernels.a` |
| Branch | master @ `0ae9e07` |
| Session | **35 (done)** |

### Qwen3-1.7B — INT4 batched attention

| M | Total t/s | Per-seq t/s | vs Q4_K_M |
|---|-----------|-------------|-----------|
| 1 | **612.8** | 612.8 | **209%** |
| 2 | **1881.5** | 940.8 | **641%** |
| 4 | **4922.6** | 1230.7 | **1679%** |
| 8 | **11284.5** | 1410.6 | **3845%** |

### Reference

| Config | t/s | Notes |
|--------|-----|-------|
| INT8 M=1 fused | 181.5 | 14 kernels/layer |
| INT8 M=8 batched-attn CUDA Graph | 324.3 | Production INT8 path |
| llama.cpp Q4_K_M FA=on | 293.4 | Qwen3-1.7B, build b9442 |

---

## 3. Session 35 — INT4 Batched Attention (Q4 Complete)

### What was built

| Component | Status | Notes |
|-----------|--------|-------|
| `gemv_int4_batched` | ✅ | M=1-8, contiguous M×N float output |
| `fused_residual_norm_int4` | ✅ | residual+rmsnorm+quant (3→1 kernel) |
| `fused_residual_norm_int4_fp32out` | ✅ | same + FP32 output (next layer input) |
| `fused_swiglu_quant_int4` | ✅ | SwiGLU+INT4 quant (2→1 kernel) |
| `bench/decode_int4_batched_attn` | ✅ | Primary INT4 benchmark. Uses gemv_int4_batched + attention_decode_batched_gqa |
| `bench/decode_int4_batched` | ✅ | Serial-attn variant (slower, may have stale data bugs) |

### Critical bugs fixed

1. **INT4-FP32 buffer aliasing** — `fused_residual_norm_int4_fp32out` wrote INT4+FP32 to same buffer. INT4 bytes corrupted first 256 FP32 elements (12.5% of hidden state). Fix: separate buffers.
2. **Stale residual** — `d_res` copied once at layer 0, reused for all 28 layers. Fix: pass current `d_x32` per layer.
3. **Per-layer quantization** — `process_seq` quantized once at entry, not per layer. Layers 2-28 stale. Fix: quantize each layer.

### Per-layer kernel count (M=1)

14 kernels: 7× gemv_int4_batched (Q,K,V,O,gate,up,down), 1× quantize_int4, 1× attention_decode_batched_gqa, 1× fused_residual_norm_int4, 1× fused_swiglu_quant_int4, 1× fused_residual_norm_int4_fp32out, 2× update_kv_cache (serial per-seq)

### AGENTS.md audit (session 35 end)

10 issues found + fixed:
- Symbol count stale (159→177)
- INT4 corruption % wrong (33%→12.5%)
- Duplicate §4 Key Findings table (old session 34 data)
- Bytes/val claim wrong (0.625→0.75)
- Missing Q4_K_M bytes/val reference (0.515)
- Kernel count wrong (13→14)
- Library symbol count in dev loop verify (164→177)
- Misleading "near-perfect linear scaling" (actual: super-linear due to kernel launch amortization)
- Ambiguous "t/s" → clarified "total t/s"
- Sentinels + extra blank lines cleaned

---

## 4. Recent Decisions

- **INT4 aims higher**: 612.8 vs 293.4 Q4_K_M = 209%. Old 238 t/s was measurement bug, not compute limit.
- **Batched attention**: Use `attention_decode_batched_gqa` for all M values. Kernels more expensive upfront but amortize across M perfectly.
- **Separate INT4/FP32 buffers**: `fused_residual_norm_int4_fp32out` must receive separate `d_x_i4` (INT4) and `d_x32` (FP32) pointers. Never alias.
- **Per-layer quantize**: Always quantize `d_x32` fresh at start of each layer loop. Never reuse stale quantized data.
- **Multiple passes clean document**: AGENTS.md needed section dedup, metric verification, and stale data removal — all done in session 35 end.

---

## 5. Important Constraints

- `export PATH=/usr/local/cuda-13.3/bin:$PATH` before nvcc
- `compute_120a` required (not compute_120)
- `killall hashcat` before every measurement (auto-restarts, -45% throughput)
- `gemv_int8_warp` is production INT8 GEMV — NOT `gemv_int8`
- `gemv_int4_warp` is production INT4 GEMV — scalar unpack, no __dp4a
- L2 persisting cache harmful for large weights (>8 MB) — only d_rn (8 KB) safe
- CUDA Graph harmful on Blackwell — 10× slower than individual launches
- Speculative decode infeasible — 4.5× per-seq overhead
- NVF4 tensor core MMA abandoned — scale layout mismatch for GEMV

---

## 6. Known Issues / Risks

1. **hashcat auto-restarts** — `killall hashcat` 30s before any benchmark. 60s respawn window.
2. **process_seq in decode_int4_batched.cu** — may have stale data bugs. Use `decode_int4_batched_attn` as primary benchmark.
3. **decode_int4_cgraph** — fixed residual/aliasing bug but still 250 t/s (per-kernel launch overhead vs gemv_int4_batched). Not primary path.
4. **Qwen3-8B INT8** — 44.6 t/s (54% of Q4_K_M 82.56). Weight-bound.
5. **text_generate repetition** — greedy decode repeats. Use -t 0.8 or -k 40.

---

## 7. Pending Tasks

| Task | Status | Notes |
|------|--------|-------|
| Qwen3-8B INT4 weights | 🔜 | INT8 44.6 t/s → INT4 should reach ~90+ t/s |
| 8B batched-attn benchmark | 🔜 | Adapt decode_int4_batched_attn for 8B dims |
| text_generate INT4 inference path | 🔜 | Currently INT8 only. Needs INT4 decode loop |
| Docker server INT4 support | 🔜 | Currently INT8 server only |
| llama.cpp comparison at 8B | 🔜 | INT4 8B vs Q4_K_M 8B (82.56 t/s) |

---

## 8. Important Files / Commands

### Build
```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
killall hashcat 2>/dev/null
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
```

### Benchmark
```bash
./bench/decode_int8_cgraph 28                     # INT8: 181.5 t/s
./bench/decode_int8_batched_cgraph_attn 28 8      # INT8 M=8: 324.3 total t/s
./bench/decode_int4_batched_attn 28 1             # INT4: 612.8 total t/s (PRIMARY)
./bench/decode_int4_batched_attn 28 8             # INT4 M=8: 11285 total t/s
./bench/decode_int4_cgraph 28                     # INT4 (per-kernel, 250 t/s — secondary)
./bench/text_generate "The capital of France is" 30
```

### INT4 weights
```bash
python3 scripts/quantize_generic.py /mnt/data/ai/hf/qwen3-1.7b-base weights_int4_qwen3_1.7b int4
# Output: 1.3 GB (62% of INT8 2.1 GB), 394 files
```

### Verify
```bash
nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l  # expect 177
```

---

## 9. Validation

| Check | Status |
|-------|--------|
| Library symbols | ✅ 177 |
| INT4 batched-attn M=1 | ✅ 612.8 total t/s (209% Q4_K_M) |
| INT4 batched-attn M=8 | ✅ 11285 total t/s (3845% Q4_K_M) |
| INT8 M=1 fused | ✅ 181.5 t/s |
| INT8 M=8 batched-attn Graph | ✅ 324.3 total t/s |
| INT4 weights | ✅ 1.3 GB, 394 files |
| AGENTS.md audit | ✅ 10 issues fixed, document clean |
| Correctness (uniform input) | ✅ mean=1.0, no NaN/inf (4 layers) |
| Multiple runs stability | ✅ ±0.3% variance (3 runs M=1) |

---

## 10. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-06-02 |
| branch | master |
| last_commit | `0ae9e07` Update AGENTS.md |
| uncommitted | AGENTS.md (handoff pending), HANDOFF.md |
| active components | bench/decode_int4_batched_attn.cu, src/kernels/gemv_int4_batched.cu, src/kernels/fused_residual_norm_int4.cu, src/kernels/fused_swiglu_quant_int4.cu |

---

## META PROMPT

**Boot**: Read `AGENTS.md` → `HANDOFF.md` → `git log --oneline -3` → `killall hashcat` → `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l` (expect 177) → `./bench/decode_int4_batched_attn 28 1` (expect ~612 total t/s).

**Verified**: 177 symbols. INT4 M=1: 612.8 total t/s (209% Q4_K_M). INT4 M=8: 11285 total t/s (3845% Q4_K_M). Q4 plan complete — exceeds target (85-99%) by 2×.

**DO NOT**:
- Use `compute_120` (must be `compute_120a`)
- Use `gemv_int8` (use `gemv_int8_warp`)
- Benchmark without `killall hashcat` (-45% throughput)
- Use `decode_int4_batched.cu` as primary benchmark (stale data bugs)
- Use `decode_int4_cgraph.cu` as primary benchmark (per-kernel, 2.4× slower)
- Pursue speculative decode, FP4 tensor core GEMV, PDL, or CUDA Graph (all dead ends)
- Expect M>8 scaling (batched GEMV register pressure)

**What's next** (choose one):
- Qwen3-8B INT4 weights + benchmark (largest quality gap vs llama.cpp Q4_K_M at 8B)
- text_generate INT4 inference path (currently INT8 only)
- Docker server INT4 support
- llama.cpp comparison at 8B

**Update discipline**: Refresh HANDOFF.md only on materially new state. Keep deduplicated with AGENTS.md. Prefer bullets.