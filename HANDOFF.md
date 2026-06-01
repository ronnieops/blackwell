# HANDOFF.md — blackwell

Continuity doc. Read before acting. Keep current with AGENTS.md.

---

## 1. Current Objective

INT8 inference engine for RTX 5060 Ti. Production-ready.
- **M=8 batched: 324 t/s (110% of Q4_K_M)** — competitive path
- M=1 decode: 181 t/s (62% of Q4_K_M) — bandwidth-limited
- Bottleneck: INT8 reads 2× more data than Q4_K_M
- **Spec decode infeasible**: Batched verify 4.5× slower per-seq than sequential. Draft can't bridge gap.

---

## 2. Current Status

| Metric | Value |
|--------|-------|
| GPU | RTX 5060 Ti, GB206, SM_120a, 36 SMs, ~500 GB/s GDDR7 |
| CUDA | 13.3, C++17, CMake |
| Library | **157 symbols** `build/libblackwell_kernels.a` |
| Branch | master @ `4219567` |
| Session | **33** |

### Benchmark Results

| Config | Total t/s | Per-seq | vs Q4_K_M | VRAM |
|--------|-----------|---------|-----------|------|
| M=1 decode | 181 | 181 | 62% | ~3.4 GB |
| **M=8 CUDA Graph** | **324** | **40.5** | **110%** | **~4.4 GB** |
| M=16 batched MLP | 335 | 20.9 | 114% | 9 GB |
| llama.cpp Q4_K_M | 293 | 293 | 100% | 5 GB |
| llama.cpp F16 | 114 | 114 | 39% | 5 GB |

---

## 3. Recent Decisions

### Session 33 — Speculative decode analysis (abandoned)
- **Spec decode infeasible on this hardware**: Batched verify (24.7 ms/seq) is **4.5× slower per-seq** than sequential (5.52 ms/seq). Draft must be 4.5× faster to break even. Even tiny 50M draft gives ~18× speedup but low acceptance kills gains. Best case ~92 t/s vs 181 t/s sequential.
- **Self-speculation (skip layers) won't work**: lm_head needs all 28 layers for meaningful logits. No early-exit head exists. Training one is out of scope.
- **Recommendation**: Abandon spec decode for this hardware. Ship M=8 production server (324 t/s, 110% of Q4_K_M).
- **Docker server ready**: `Dockerfile` + `server/server.py` exist. Not yet built/tested end-to-end.

### Session 32 — M=8 optimization + M>8 discovery
- **`gemv_int8_batched` M>8 bug**: Switch statement only had cases 1-8. M>8 silently returned zero (no kernel launch). All M=16+ measurements before fix were WRONG — MLP GEMVs weren't running.
- **Fixed**: `gemv_int8_batched` now loops over groups of 8. Supports any M.
- **M=16 NOT optimal**: Batched GEMV register pressure for M>8. Each block processes M sequences → 16× activation registers → occupancy drops. Real M=16: 335 t/s (barely better than M=8's 324).
- **M=8 is practical limit**: Serial `gemv_int8_warp` faster than batched for M>8. CUDA Graph with serial MLP has too many nodes (7000+) → overhead exceeds benefit.
- **`gemv_int8_batched` vs `gemv_int8_warp`**: Isolated test shows serial is 1.5-2.7× faster for all GEMV sizes. But in CUDA Graph context, batched is faster (fewer graph nodes).
- **L2 persisting harmful for large weights**: Pinning 12.6 MB gate weights → 28% regression.
- **Fused pack+GEMV kernels**: `fused_pack_gemv_o`, `fused_swiglu_gemv` — correct but 20% slower. Archived, not used.
- **CUDA Graph M=1 abandoned**: `cudaFuncSetAttribute` in `attention_decode_gqa` incompatible with capture.
- **Report**: `REPORT.md` created with full findings.

### Session 31 — Fused kernel exploration
- `fused_pack_gemv_o` + `fused_swiglu_gemv` — numerically correct, 20% slower
- Root cause: two-phase kernels (quant→sync→GEMV) add overhead exceeding launch savings

### Session 30 — CUDA Graph M=1 attempts
- `attention_decode_gqa` wrapper: `cudaFuncSetAttribute` + H2D pinned memcpy poison capture
- Abandoned — per-kernel path is production target for M=1

---

## 4. Important Constraints

- `export PATH=/usr/local/cuda-13.3/bin:$PATH` before nvcc
- `compute_120a` required (not `compute_120`)
- `gemv_int8_warp` production GEMV — NOT `gemv_int8`
- `gemv_int8_batched` supports M>8 now (loop over groups of 8)
- hashcat auto-restarts. `killall hashcat` before every measurement.
- INT8 reads 2× data vs Q4_K_M — fundamental bandwidth limit for M=1

---

## 5. Known Issues / Risks

1. **hashcat**: Auto-restarts, -45% throughput. `killall hashcat` before every measure.
2. **INT8 vs Q4_K_M gap (M=1)**: Hardware limitation. INT8 reads 2× more data. No fix without sub-byte quant.
3. **M=16+ not beneficial**: Batched GEMV register pressure. M=8 is practical limit.
4. **CUDA Graph capture**: `attention_decode_gqa` incompatible with capture (cudaFuncSetAttribute).
5. **Fused pack+GEMV slower**: Two-phase kernel overhead > launch savings. Not used.
6. **Spec decode infeasible**: Batched verify 4.5× slower per-seq than sequential. Draft can't bridge gap.

---

## 6. Pending Tasks

| Task | Status | Notes |
|------|--------|-------|
| Deploy production server | TODO | Docker/API packaging exists, not tested end-to-end |
| Speculative decoding | ❌ Abandoned | Infeasible. Batched verify 4.5× slower per-seq than sequential. Hardware limit. |
| Real Q4 quantization (GPTQ/AWQ) | TODO | +80-100% M=1, needs quantize pipeline |
| text_generate for Qwen3.5-9B | TODO | Needs tokenizer integration |

---

## 7. Suggested Next Actions

| Priority | Task | Rationale |
|----------|------|-----------|
| **High** | **Deploy production server** | 324 t/s beats Q4_K_M by 10%. Ship it. Build Docker, test, deploy. |
| Medium | Real Q4 quantization (GPTQ/AWQ) | Closes M=1 bandwidth gap but complex quantize pipeline |
| Low | Qwen3.5-9B integration | 45.6 t/s (64% of Q4_K_M), needs tokenizer |
| Low | Speculative decoding | ❌ Abandoned. Infeasible on this hardware. |

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
./bench/decode_int8_batched_cgraph_attn 28 8    # M=8: 324 t/s (optimal)
./bench/decode_int8_cgraph 28                   # M=1: 181 t/s
./bench/text_generate "The capital of France is" 30
```

### Docker
```bash
docker build -t blackwell-inference .
docker run --gpus all -p 8080:8080 blackwell-inference
curl -X POST http://localhost:8080/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","max_tokens":30}'
```

### Verify
```bash
nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l  # expect 157
```

---

## 9. Validation

| Check | Status |
|-------|--------|
| Library | ✅ 157 symbols |
| M=8 CUDA Graph | ✅ 324 t/s (110% of Q4_K_M) |
| M=1 fused | ✅ 181 t/s (62% of Q4_K_M) |
| gemv_int8_batched M>8 | ✅ Fixed (loop over groups of 8) |
| Correctness | ✅ Max diff 0.000000 vs serial baseline |
| Docker server | Not yet tested end-to-end |

---

## 10. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-06-01 |
| branch | master |
| last_commit | `4219567` docs: session 33 — spec decode infeasibility analysis, update symbol count, mark server ready |
| repo_state | 157 symbols. gemv_int8_batched M>8 fixed. M=8: 324 t/s (110% of Q4_K_M). Spec decode infeasible. Docker server ready. |
| uncommitted | (none — clean) |

---

## META PROMPT

**Boot sequence**: Read `AGENTS.md` → `HANDOFF.md` → `git log --oneline -3` → `killall hashcat` → `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l` (expect 157) → `./bench/decode_int8_batched_cgraph_attn 28 8` (expect ~324 t/s).

**Verified state**: 157 symbols. M=8 CUDA Graph: 324 t/s (110% of Q4_K_M). M=1: 181 t/s (62%). `gemv_int8_batched` supports M>8. M=8 is practical limit. Fused pack+GEMV archived. CUDA Graph M=1 abandoned. Spec decode infeasible (batched verify 4.5× slower per-seq). Docker server ready.

**DO NOT**:
- Use `compute_120` (must be `compute_120a`)
- Use `gemv_int8` in production (use `gemv_int8_warp`)
- Benchmark without `killall hashcat`
- Expect M>8 to help (batched GEMV register pressure)
- Use fused pack+GEMV kernels (20% slower, archived)
- Attempt CUDA Graph M=1 (attention_decode_gqa incompatible)
- Pursue speculative decoding (batched verify 4.5× slower per-seq than sequential. Infeasible.)

**Update discipline**: Update HANDOFF.md only when materially new state. Keep deduplicated with AGENTS.md. Prefer bullets over prose.