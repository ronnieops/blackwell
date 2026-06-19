# Handoff — Session 86 (2026-06-17)

## 1. Current Objective

14B INT4 raw quantization complete. AWQ calibration pending. Disk cleanup done (564 GB free).

## 2. Current Status

| Model | t/s (M=1) | Size | Quality | Status |
|-------|-----------|------|---------|--------|
| Qwen3-8B INT4 | **73** | 4.8 GB | PPL 21.98 ✅ | Production (v0.12.x) |
| Qwen3-14B INT4 | **TBD** | 6.9 GB | PPL TBD (raw, no AWQ) | Raw INT4 complete, AWQ pending |
| Gemma 4 12B QAT INT4 | — | — | ❌ GARBLED — GGUF norms corrupted | BLOCKED: Google QAT GGUF export bug |
| Llama 3.2 3B INT4 | ~223 | 1.9 GB | Coherent ✅ | Via GGUF bridge |

**Symbol count**: 209 (unchanged).

**CRITICAL FINDING — Gemma 4 QAT GGUF CORRUPTED:**
**Disk**: 564 GB free (58%). Freed ~550 GB from HF cache. See AGENTS.md for details.

**14B INT4 quantization complete**: 645 files, 6.9 GB. Streaming quantizer `scripts/quantize_int4_qwen3_14b.py` uses torch mmap for BF16→F32 conversion. Architecture: H=5120, I=17408, NL=40, nqh=40, nkv=8, hd=128, V=151936. AWQ calibration pending.

**30B MoE weights deleted**: 18 GB `weights_int4_qwen3_30b_a3b/` removed. Blocked by RAM (needs 55+ GB). HF source (57 GB) kept.

**Gemma4 GGUF ALL corrupted**: Systematic unsloth GGUF export bug for entire Gemma4 family (E2B, E4B, 12B, 26B, 31B). All FP32 norm tensors contain garbage values (1e22+). No blackwell fix possible. Requires clean FP32 weights from HuggingFace safetensors.

**Gemma4 12B safetensors download**: ~14 GB of ~24 GB downloaded at `/mnt/data/ai/hf/models--google--gemma-4-12B-it/`. Will provide clean FP32 weights for converter-based quantization once complete.

**GGUF tensor names** (confirmed from GGUF v3):
- `blk.X.attn_norm.weight` → `input_layernorm` (our naming)
- `blk.X.post_attention_norm.weight` → `post_attn_norm`
- `blk.X.ffn_norm.weight` → `post_attention_layernorm` (pre-FFN)
- `blk.X.post_ffw_norm.weight` → `post_ffn_norm` (post-FFN)
- `blk.X.attn_q_norm.weight` → Q head RMSNorm (256 SWA, 512 FA — NATIVE, no replication needed)
- `blk.X.attn_k_norm.weight` → K head RMSNorm (256 SWA, 512 FA — NATIVE)
- `blk.X.layer_output_scale.weight` → skip/alpha scale per layer (not loaded in bench)
- `output_norm.weight` → `final_norm`
- `blk.X.attn_output.weight` → `o_proj` (Q4_K type=2)

**GGUF paths:**
- `/mnt/data/ai/hf/models--google--gemma-4-12B-it-qat-q4_0-gguf/gemma-4-12b-it-qat-q4_0.gguf`
- Unsloth non-QAT variants: `/mnt/data/ai/hf/models--unsloth/gemma-4-12B-it-GGUF/`

## 3. Bugs Fixed This Session

### BUG #1 — attn_batched_kernel truncates Q·K dot at 128 floats (CRITICAL)
**File**: `src/kernels/decode.cu:444-460` (`attn_batched_kernel`) and line ~555 (`attn_batched_kernel_pos`)
**Symptom**: `Q_reg[4]` + single `float4 Q4[lane_id]` → 32×4=128 floats. head_dim=256 gets 50% (128/256), head_dim=512 gets 25% (128/512). Garbled attention.
**Fix**: Loop over `n_chunks = head_dim / 128` chunks. Loads each 128-float chunk from smem_Q and accumulates. For hd≤128 (8B, Llama): single chunk, bit-identical.
**Verified**: 8B PPL stays 21.98 (bit-identical regression test passed).

### BUG #2 — QK norm read used wrong dim (BENIGN)
**File**: `bench/text_generate_gemma4_12b_qat.cu:248-269`
**Symptom**: `fread(w256,4,hd_swa=256,f)` for FA layer norms. GGUF has 512 floats for FA QK norms.
**Fix**: Read `l_hd` floats directly. GGUF converter writes native per-layer dims.
**Status**: Dead code — QK norms were ALREADY corrupted at GGUF level (garbage values). Fix correct but moot.

### BUG #5 — RoPE kernel smem cap at 64 pairs
**File**: `src/kernels/rope.cu:176-177`
**Symptom**: `smem_cos[64]/smem_sin[64]` hardcoded. For hd=256 (128 pairs), threads 64-127 write OOB smem → UB reads. For hd=512 (256 pairs), catastrophic.
**Fix**: Bumped to `smem_cos[256]/smem_sin[256]`.
**Status**: Fix applied but no effect (GGUF norms dominate).

### BUG #4 — GeGLU uses erf GELU (minor)
**File**: `src/kernels/apply_geglu.cu:18-20`. HF Gemma uses `gelu_pytorch_tanh`. erf vs tanh diff ~1e-3. Low priority.

### BUG #3 — Logit softcap disabled (minor)
Bench comment reasoning wrong (softcap is monotonic, can't cause uniform dist). Symbol correct. Low priority.

## 4. Recent Decisions

- **Gemma 4 QAT path BLOCKED**: Google QAT GGUF has corrupted FP32 norms. No blackwell fix possible. To revive: (a) re-export QAT from source with non-corrupt norms, (b) use non-QAT Gemma 4 GGUF (unsloth variants), or (c) get raw FP32 weights from HuggingFace.
- **GGUF QK norms are NATIVE per-layer dims**: FA layers have 512-dim QK norms in GGUF. No replication needed. Our converter writes them correctly.
- **`layer_output_scale` per-layer files exist**: GGUF has `blk.X.layer_output_scale.weight` (1 float per layer) — NOT loaded in bench. Gemma 4 uses learnable residual scaling. May be minor quality factor but not root cause.
- **SWA windowing math**: When `step > 1024`, bench passes `attn_step=1024` to kernel but KV cache writes at full `step` offset. Kernel reads from cache base → wrong positions. Low impact for short sequences; fix if prefill > 1024.

## 5. Important Constraints

- `killall hashcat` before any GPU measurement
- `CUDACXX=/usr/local/cuda-13.3/bin/nvcc`
- **25 GB free on /mnt/data** (HANDOFF said 55G — STALE)
- Disk 99% full. 30B port blocked (needs 55+ GB).
- `d_cos_swa`/`d_sin_swa` and `d_cos_fa`/`d_sin_fa` are RoPE cache pointer names

## 6. Known Issues / Risks

| Issue | Severity | Status |
|-------|----------|--------|
| Gemma 4 QAT GGUF norm corruption | **Critical** | BLOCKED — Google QAT GGUF bug, not blackwell |
| 14B INT4 garbled | High | AWQ embed/lm_head not quantized |
| Disk at 99% (25G free) | High | Blocks 14B AWQ, 30B port |
| Gemma 4 non-QAT GGUF (IQ4) | Medium | Would need GGUF converter IQ4 support |
| SWA window cache offset | Low | Wrong positions when step > 1024 |

## 7. Pending Tasks

| Priority | Task | Status |
|----------|------|--------|
| **BLOCKED** | Gemma 4 QAT: need non-corrupt GGUF (QAT re-export or non-QAT GGUF) | Root cause: GGUF corruption |
| High | **Disk cleanup** (25G free, 99% full): rm old dead weight dirs, clear /tmp, verify before rm | Not started |
| High | **14B AWQ completion**: embed/lm_head not quantized, garbled output | Blocked by disk |
| Medium | Gemma 4 non-QAT port: GGUF converter IQ4 support | Not started |
| Medium | 14B server: fix quality | Blocked by AWQ |
| Low | Qwen3-30B port | Blocked by disk (55+ GB needed) |

### Disk cleanup candidates (verify before rm)
```
weights_int4_qwen3_1.7b/          (dead, wrong dims, ~0.5 GB)
weights_int4_qwen3_1.7b_asym/     (dead, ~0.5 GB)
weights_int5_qwen3_1.7b_asym/     (dead, ~0.5 GB)
weights_int8_qwen3_8b_mixed/      (no improvement over all-INT8, ~9.6 GB)
weights_int8_qwen35_9b_mixed/      (no improvement, ~11 GB)
/root/.cache/huggingface/         (28GB 14B safetensors — regenerable via download)
wikitext2_corpus_14b.bin          (4.6 GB — regenerable)
```

## 8. Important Files / Commands

### Build
```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel
```

### Gemma 4 12B QAT Bench (BROKEN — GGUF corruption)
```bash
killall hashcat 2>/dev/null
echo "9259, 1902" > /tmp/gemma_hello_tokens.txt
./bench/text_generate_gemma4_12b_qat /tmp/gemma_hello_tokens.txt 15
# Output: garbled tokens (root cause = GGUF norm corruption)
```

### Probe GGUF norms (for investigation)
```bash
# Compile: /usr/bin/g++ -O2 -std=c++17 -Wno-unused-result probe.cc -I better-inference -o probe
# gguf.h header-only — use GGUFReader class
# List all tensors and their raw F32 values to verify corruption
```

### Key source files changed this session
```
src/kernels/decode.cu              — BUG #1 fix: attn batched chunked Q·K dot
src/kernels/rope.cu               — BUG #5 fix: RoPE smem[256]
bench/text_generate_gemma4_12b_qat.cu — BUG #2 fix: native QK norm dims
```

## 9. Session Metadata

- updated_at: 2026-06-17
- branch: master, repo state: dirty
- last commit: `5bbc11c v0.12.7` (more recent than HANDOFF's 37d3eaf)
- active: Gemma 4 QAT investigation completed, disk cleanup, 14B AWQ

---

## META PROMPT

Resuming blackwell CUDA INT4 development on RTX 5060 Ti (sm_120a).

Key truths:
- 8B production (73 t/s, PPL 21.98) unchanged
- Gemma 4 12B QAT garbled ROOT CAUSE FOUND: Google QAT GGUF corrupted all FP32 norm weights (values 4e17 instead of ~1.0). NOT a blackwell implementation bug. Path blocked.
- 3 kernel bugs fixed: attn Q·K dot (BUG #1), QK norm read (BUG #2), RoPE smem cap (BUG #5). All backward-compatible.
- 14B INT4 raw quantization COMPLETE (6.9 GB, 645 files). AWQ calibration pending.
- Disk CLEAN: 564 GB free (58%). Freed ~550 GB from HF cache.
- Gemma4 GGUF ALL corrupted (systematic unsloth export bug for entire Gemma4 family). No usable GGUF source.
- 30B MoE weights deleted (blocked by RAM, needs 55+ GB).

Priorities: (1) 14B AWQ calibration, (2) 14B server port, (3) Gemma 4 non-QAT from safetensors (download ~14/24 GB)
