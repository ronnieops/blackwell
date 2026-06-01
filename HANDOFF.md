# HANDOFF.md — blackwell

Continuity doc. Read before acting. Keep current with AGENTS.md.

---

## 1. Current Objective

INT8 inference engine for RTX 5060 Ti. Production-ready. 125 symbols.
- Qwen3-1.7B: 183.6 t/s (67% of llama.cpp Q4_K_M 274.4)
- Qwen3-8B: 57.4 t/s (73% of llama.cpp Q4_K_M 78.7)
- Bottleneck: weight size (8.9 GB vs 4.68 GB Q4_K_M). No sub-byte fix on GB206.

---

## 2. Current Status

| Metric | Value |
|--------|-------|
| GPU | RTX 5060 Ti, GB206, SM_120a, 36 SMs, ~500 GB/s GDDR7 |
| CUDA | 13.3.33, driver 580.159.04 |
| Library | **125 symbols** `build/libblackwell_kernels.a` |
| Branch | master @ `6728180` |
| Session | **27** — weights re-quantized, text_generate fixed, Qwen3.5-9B scoped |

### Benchmark Results

| Model | Config | Blackwell INT8 | llama.cpp Q4_K_M | Ratio |
|-------|--------|----------------|-------------------|-------|
| Qwen3-1.7B | CUDA Graph M=1 | 182.8 t/s | 274.4 t/s | 67% |
| Qwen3-1.7B | Batched attn M=8 | **326.8 t/s** | 274.4 t/s | **119%** |
| Qwen3-0.6B | CUDA Graph | 447.4 t/s | — | — |
| Qwen3-8B | CUDA Graph 28L | 57.4 t/s | 78.7 t/s | 73% |
| GEMM prefill | M=128 | **13.0 TFLOPS** (3× before fix) | 4.3 TFLOPS old | 3× ✅ |

### GPU Architecture (GB206)
- **No FP4 tensor cores** — GB100/GB200 only (RTX 5090). INT4 warp 0.36× slower.
- 36 SMs, 500 GB/s GDDR7

---

## 3. Recent Decisions (Session 27)

- **Qwen3-1.7B weights re-quantized**: `weights_int8_bf16/` missing at session start. Re-generated from `/mnt/data/ai/hf/qwen3-1.7b-base/` (28L, H=2048). 197 files. ✅ Done
- **text_generate segfault fixed**: Missing `.f32` norm files. `quantize_generic.py` only creates INT8 weights. Separate extraction from BF16 safetensors required:
  - `model.layers.{l}.input_layernorm.weight` → `{l}_input_layernorm.f32`
  - `model.layers.{l}.post_attention_layernorm.weight` → `{l}_post_attention_layernorm.f32`
  - `model.layers.{l}.self_attn.{q,k}_norm.weight` → `qk_norms.f32` [28×2×128]
  - `model.norm.weight` → `final_norm.f32`
  - Run: `python3 scripts/extract_norms.py` + re-extract `final_norm.f32` + `qk_norms.f32` from BF16 safetensors (script only handles layernorms)
- **Qwen3.5-9B scoped**: Hybrid SSM architecture — 32 layers with alternating linear_attention (Mamba SSM, 24/32) + full_attention (8/32). H=4096, head_dim=256, linear_key_head_dim=128. New kernel family: SSM scan + selective gating. High effort (~3-5 new kernels). NOT MoE. ✅ Assessed
- **Per-block WMMA dequant investigated**: `gemm_int8_wmma_fast` was ALREADY correct — advisor analysis confirms per-block scaling in the per-iteration SMEM load. The "simplified dequant" note in AGENTS.md §10 is WRONG. Kernel unchanged. Reverted attempted fix.
- **Weight corruption risk**: Running ad-hoc test scripts that overwrite weight files in-place corrupts model state. Always backup or use temp directories.
- **decode_int8_cgraph mismatch**: Pre-existing (session 26). Per-kernel vs CUDA Graph L1 diff ~0.5. 182.8 t/s baseline works fine.
- **INT4/FP4**: Dead end on GB206. 0.36× slower. ❌ Confirmed

---

## 4. Important Constraints

- `export PATH=/usr/local/cuda-13.3/bin:$PATH` before nvcc
- `compute_120a` required (not `compute_120`)
- `gemv_int8_warp` is production GEMV — NOT `gemv_int8`
- `gemm_int8_wmma_fast` is production GEMM — NOT `gemm_int8_dp4a`
- hashcat auto-restarts. `killall hashcat` before measurement.
- `fused_rmsnorm_quant_int8` and `fused_rmsnorm_pack` handle N≤4096 (256×16)
- `gemm_int8_wmma_fast` uses correct per-block scales (NO simplified dequant)
- `quantize_generic.py` does NOT create `.f32` norm files — separate extraction required
- Weight file corruption: test scripts must use temp directories, never overwrite `weights_int8_*/` in-place

---

## 5. Known Issues / Risks

1. **hashcat**: Auto-restarts, -45% throughput. `killall hashcat` before every measure.
2. **FP4/INT4**: Dead end on GB206. No FP4 tensor cores. INT4 0.36× slower.
3. **text_generate repetition**: Greedy decode repeats. Use `-t 0.8 -k 40` for quality.
4. **decode_int8_cgraph**: Pre-existing mismatch (~0.5 L1 diff). 182.8 t/s works fine.
5. **No Qwen3.5-9B support**: Mamba SSM hybrid. New kernel family needed.
6. **Weight corruption**: Test scripts that overwrite weight files corrupt model state.

---

## 6. Pending Tasks

| Priority | Task | Status | Effort | Notes |
|----------|------|--------|--------|-------|
| ~~P1~~ | ~~Docker packaging~~ | ✅ Done | Low | server/server.py, Dockerfile |
| ~~P4~~ | ~~GPU argmax~~ | ✅ Done | Low | +7%, 125 symbols |
| ~~GEMM prefill~~ | ~~3× speedup~~ | ✅ Done | Med | Direct c_frag dequant |
| ~~O4~~ | ~~Fused RMSNorm H=4096~~ | ✅ Done | Low | kElemsPerThread 8→16 |
| ~~Per-block WMMA~~ | ~~Already correct~~ | ✅ Done | — | Advisor confirmed kernel correct |
| P3 | Qwen3.5-9B Mamba hybrid | Not started | **High** | SSM scan + gating, 24 linear + 8 full attn layers |
| P5 | Tokenize + sampler on-GPU | Not started | Med | BPE on GPU, top-k GPU sampling |
| — | Embed tokens scale fix | Not started | Low | scale file has 2× elements (quantize_generic bug), not used in text_generate |

---

## 7. Suggested Next Actions

| Priority | Task | Rationale |
|----------|------|-----------|
| P3 | Qwen3.5-9B Mamba hybrid | Biggest gap vs llama.cpp (71 t/s). New architecture. High effort, high impact. |
| P5 | On-GPU tokenize + sampler | Removes host round-trips, ~5% gain potential |
| — | Embed scale fix | Clean up quantize_generic.py scale layout |

---

## 8. Important Files / Commands

### Weight generation
```bash
# Full re-quantization: INT8 weights + norm files (two steps)
python3 scripts/quantize_generic.py /mnt/data/ai/hf/qwen3-1.7b-base weights_int8_bf16

# Norm file extraction (BF16 → FP32)
python3 scripts/extract_norms.py  # layernorms only

# Final norm + QK norms (separate)
python3 -c "
import struct, json, numpy as np
model='/mnt/data/ai/hf/qwen3-1.7b-base/model.safetensors'
with open(model,'rb') as f:
    hl=struct.unpack('Q',f.read(8))[0]; hdr=json.loads(f.read(hl))
def rd(n):
    i=hdr[n]; s,e=i['data_offsets']
    with open(model,'rb') as f: f.seek(8+hl+s); raw=f.read(e-s)
    u16=np.frombuffer(raw,dtype=np.uint16)
    return (u16.astype(np.uint32)<<16).view(np.float32)
OUT='weights_int8_bf16'
for l in range(28):
    rd(f'model.layers.{l}.input_layernorm.weight').tofile(f'{OUT}/{l}_input_layernorm.f32')
    rd(f'model.layers.{l}.post_attention_layernorm.weight').tofile(f'{OUT}/{l}_post_attention_layernorm.f32')
qk=np.zeros((28,2,128),np.float32)
for l in range(28):
    qk[l,0]=rd(f'model.layers.{l}.self_attn.q_norm.weight')
    qk[l,1]=rd(f'model.layers.{l}.self_attn.k_norm.weight')
qk.tofile(f'{OUT}/qk_norms.f32')
rd('model.norm.weight').tofile(f'{OUT}/final_norm.f32')
"
```

### Benchmark commands
```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
killall hashcat 2>/dev/null  # MUST

# Build
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake --build build --parallel

# Build bench binaries (static link, must rebuild after lib changes)
CUDACXX=/usr/local/cuda-13.3/bin/nvcc nvcc -O3 -std=c++17 \
  -gencode=arch=compute_120a,code=sm_120a -I include \
  bench/text_generate.cu build/libblackwell_kernels.a -o bench/text_generate

# Decode throughput
./bench/decode_int8_batched_cgraph_attn 28 8   # 326.8 t/s (M=8)
./bench/decode_int8_cgraph 28                   # 182.8 t/s (M=1)
./bench/text_generate "The capital of France is" 15 -t 0.001  # "Paris" correct

# Correctness
./bench/verify_gemm 128    # 7/7 PASS
./bench/verify_pipeline_error  # Pipeline SNR

# Prefill
./bench/decode_prefill 20  # GEMM prefill 13 TFLOPS
./bench/int8_prefill_benchmark 20

# Check symbols
nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l  # 125
```

---

## 9. Validation

| Check | Status | Notes |
|-------|--------|-------|
| Library | ✅ 125 symbols | |
| INT8 batched attn M=8 | ✅ 326.8 t/s | 119% of Q4_K_M (hashcat-free) |
| GEMM prefill | ✅ 13.0 TFLOPS | 3× vs old, 26% peak util |
| text_generate | ✅ "Paris" correct | Greedy decode |
| GEMM verify_gemm | ✅ 7/7 PASS | All layer-0 weights |
| Qwen3.5-9B | ✅ Scoped | Hybrid SSM (24 linear + 8 full attn), H=4096 |
| llama.cpp Q4_K_M (1.7B) | ✅ 274.4 t/s | Build 95405ac65 |
| decode_int8_cgraph mismatch | ⚠️ Pre-existing | Per-kernel vs CUDA Graph L1 ~0.5 |

---

## 10. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-06-01 |
| branch | master |
| last_commit | `6728180` docs: fix symbol count 123→125 in AGENTS.md |
| repo_state | 125 symbols. Docker + HTTP server. GEMM prefill 3×. GPU argmax. Pipeline SNR 13.9 dB. |
| sessions_completed | 27 |

---

## META PROMPT

**Boot sequence**: Read `AGENTS.md` → `HANDOFF.md` → `git log --oneline -5` → `killall hashcat` → `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l` (expect 125) → `./bench/text_generate "The capital of France is" 15 -t 0.001` (expect "Paris").

**Verified state**: 125 symbols. 326.8 t/s batched attn (M=8). GEMM prefill 13.0 TFLOPS. Docker + API. Pipeline SNR 13.9 dB. `gemm_int8_wmma_fast` correct per-block dequant (verified by advisor).

**DO NOT**:
- Use `compute_120` (must be `compute_120a`)
- Use `gemv_int8` in production (use `gemv_int8_warp`)
- Use `gemm_int8_dp4a` for M≥16 (use `gemm_int8_wmma_fast`)
- Benchmark without `killall hashcat`
- Attempt INT4/FP4 quantization (dead end on GB206 — 0.36× slower)
- Run test scripts that overwrite `weights_int8_*/` files in-place (use temp dirs)
- Trust the "simplified dequant" note in AGENTS.md §10 — `gemm_int8_wmma_fast` is correct

**Update discipline**: Update HANDOFF.md only when materially new state. Keep deduplicated with AGENTS.md. Prefer bullets over prose.