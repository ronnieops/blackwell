# HANDOFF.md — blackwell

**ALWAYS read `AGENTS.md` AND this file before acting.**

---

## 1. Current Objective

Production INT4 inference across multiple GGUF-converted models. Session 73: GGUF bridge Phase 1-3 complete, Llama 3.2 1B verified (223 t/s), Llama 3.1 8B converted (43 t/s).

---

## 2. Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| INT4 8B HTTP server | ✅ **Production** | **~55 t/s**, PPL **21.82** |
| INT4 8B batched server | ✅ | **~63 t/s** (gemv_int4_batched) |
| CUDA Graph | ✅ | 2.9% speedup (867 nodes). GEMV 92% bottleneck. |
| GGUF Bridge | ✅ **Phase 1-3** | Qwen3 + Llama 3.1 + Llama 3.2 + Qwen3-8B converted. |
| Llama 3.2 1B benchmark | ✅ | **223 t/s**, coherent output |
| Llama 3.1 8B benchmark | ✅ | **43 t/s**, coherent output |
| Qwen3-8B GGUF | ✅ | 4.7 GB → 5.8 GB, valid layernorms |
| Multi-model server | ✅ | `start_servers.sh` script |
| NVFP4 | ❌ **ABANDONED** | PPL=24,850, format mismatch + double quant |
| 9B quality | ❌ **BLOCKED** | SSM instability |
| Llama server binary | ⚠️ **Builds, crashes** | `inference_server_llama` needs debugging |

---

## 3. Recent Decisions (Session 73)

### Critical GGUF Fix
- **GGUF v3 tensor offsets are RELATIVE to tensor data section** — not absolute file offsets
- Converter used `ti.offset` directly → read layernorms from wrong location → garbage weights → NaN logits
- Fix: `file_offset = tensor_data_off + ti.offset` for ALL tensor reads

### GGUF v3 RoPE fix
- Keys stored under nested prefix (e.g., `https://huggingface.co/.../llama.rope.freq_base`)
- Fixed by searching for any key ending with the suffix

### Models Converted
| Model | Config | Speed | Output |
|-------|--------|-------|--------|
| Llama 3.2 1B | 16L, H=2048, I=8192, nqh=32, nkv=8, hd=64 | **223 t/s** | ✅ Coherent |
| Llama 3.1 8B | 32L, H=4096, I=14336, nqh=32, nkv=8, hd=128 | **43 t/s** | ✅ Coherent |
| Qwen3-8B | 36L, H=4096, I=12288, nqh=32, nkv=8, hd=128 | — | ✅ Valid layernorms |

### Benchmarks
- `bench/text_generate_llama32_1b.cu` — Llama 3.2 1B, 223 t/s
- `bench/text_generate_llama31_8b.cu` — Llama 3.1 8B, 43 t/s
- `bench/bench_ppl_llama32_1b.cu` — PPL benchmark (design issue: doesn't propagate hidden states)
- `server/inference_server_llama.cu` — Multi-model server (builds but crashes)

### Multi-model Server
- `start_servers.sh` — Start multiple http_subprocess instances on different ports
- Usage: `./start_servers.sh llama32-1b` or `./start_servers.sh` (all models)
- Ports: 8123 (Llama 3.2 1B), 8124 (Llama 3.1 8B)

---

## 4. Important Constraints

- **Model dims (8B native)**: nqh=32, nkv=8, hd=128, KV=1024, H=4096, I=12288
- **Model dims (1.7B)**: nqh=16, nkv=8, hd=128, KV=1024, H=2048, I=6144
- **Model dims (Llama 3.2 1B)**: nqh=32, nkv=8, hd=64, KV=512, H=2048, I=8192, rope=500000
- **Model dims (Llama 3.1 8B)**: nqh=32, nkv=8, hd=128, KV=1024, H=4096, I=14336, rope=500000
- `compute_120a` required (NOT `compute_120`)
- `killall hashcat` before every GPU measurement
- 186 kernel symbols in `libblackwell_kernels.a`

---

## 5. Known Issues / Risks

| Issue | Severity | Notes |
|-------|----------|-------|
| Llama server crash | MEDIUM | `inference_server_llama` segfaults on startup. Config system + CUDA macros conflict. |
| PPL benchmark | MEDIUM | `bench_ppl_llama32_1b` doesn't propagate hidden states. Each step re-computes from scratch. |
| 9B quality BLOCKED | HIGH | SSM A_log > 0 for 68.8% layer-4 channels |
| Server prefill | MEDIUM | Prompts processed token-by-token |
| GPU non-determinism | LOW | Different outputs on same prompt |

---

## 6. Pending Tasks

| Priority | Task | Notes |
|----------|------|-------|
| HIGH | Fix Llama server crash | Config macros conflict with CUDA kernel params |
| MEDIUM | Fix PPL benchmark design | Propagate hidden states between steps |
| MEDIUM | 8B GGUF quality validation | Compare against native format |
| LOW | AWQ calibration | Add to GGUF converter |
| BLOCKED | 9B SSM fix | Architectural, not quantization |

---

## 7. Suggested Next Actions

1. **Fix Llama server** — Debug segfault. Likely `cfg` pointer not set before `W.resize(NL)` or CUDA macro conflicts in kernel launch configs.
2. **Run PPL on real corpus** — Use llama.cpp to compute reference PPL, compare with converted weights.
3. **Validate Llama 3.1 8B** — Run longer generation, check for repetition or degradation.
4. **AWQ calibration** — Add to GGUF converter for better quality on next conversion.

---

## 8. Important Files / Commands

### GGUF Conversion
```bash
./better-inference/gguf_convert <model.gguf> <output_dir>
# Known working:
./better-inference/gguf_convert /mnt/data/ai/hf/models--unsloth--Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf /mnt/data/ai/models/llama32-1b-int4
./better-inference/gguf_convert /mnt/data/ai/hf/models--unsloth--Llama-3.1-8B-Instruct-GGUF/Llama-3.1-8B-Instruct-Q4_K_M.gguf /mnt/data/ai/models/llama31-8b-int4
```

### Benchmark
```bash
killall hashcat 2>/dev/null
./bench/text_generate_llama32_1b "Hello" 20   # 223 t/s
./bench/text_generate_llama31_8b "Hello" 20     # 43 t/s
./bench/text_generate_int4_qwen3_8b "Hello" 20 # 58 t/s (native)
```

### Multi-model Server
```bash
./start_servers.sh llama32-1b    # Start Llama 3.2 1B on port 8123
./start_servers.sh               # Start all models
curl http://localhost:8123/health
```

### Build
```bash
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### Build server
```bash
CUDACXX=/usr/local/cuda-13.3/bin/nvcc /usr/local/cuda-13.3/bin/nvcc -O3 -std=c++17 -arch=sm_120a \
  server/inference_server_llama.cu build/libblackwell_kernels.a \
  -I include -I /usr/local/cuda-13.3/include \
  -L /usr/local/cuda-13.3/targets/x86_64-linux/lib \
  -o server/inference_server_llama -lcudart -lpthread -lz
```

### Validate
```bash
nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l  # expect 186
```

---

## 9. Validation Status

| Check | Value | Status |
|-------|-------|--------|
| Throughput (native 8B) | **~55 t/s** | ✅ |
| Throughput (Llama 3.2 1B) | **223 t/s** | ✅ |
| Throughput (Llama 3.1 8B) | **43 t/s** | ✅ |
| PPL (native 8B, AWQ α=0.6) | **21.82** | ✅ |
| GPU memory | GPU free | ✅ |
| Kernel symbols | **186** | ✅ |
| Layernorm weights | Valid (range [-0.64, 1.06]) | ✅ |
| GGUF converter | Fixed offset bug | ✅ |
| CUDA Graph | 2.9% speedup | ✅ |

---

## 10. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-06-11 |
| branch | master |
| repo_state | Clean (8 commits ahead of origin) |
| active_components | GGUF bridge, Llama benchmarks, multi-model server |
| key_session | 73 — GGUF Phase 3, Llama models, critical offset fix |
| GPU | RTX 5060 Ti, 0 MiB used (free) |

---

## META PROMPT

**Boot sequence**:
1. Read `AGENTS.md` → `HANDOFF.md`
2. `git status` — repo is clean, 8 commits ahead
3. `killall hashcat 2>/dev/null`
4. `nvidia-smi` — verify GPU free
5. `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l` — expect 186

**Verified facts**:
- GGUF v3 tensor offsets are RELATIVE to tensor data section
- Llama 3.2 1B: **223 t/s**, rope_theta=500000, H=2048, I=8192, NL=16
- Llama 3.1 8B: **43 t/s**, rope_theta=500000, H=4096, I=14336, NL=32
- Qwen3-8B: rope_theta=1000000, H=4096, I=12288, NL=36
- All layernorm weights valid after offset fix
- `inference_server_llama` builds but crashes (config macro conflict)

**DO NOT**:
- Trust pre-session-56 quality numbers (wrong dims)
- Re-dig dead ends: NVFP4, FP8, INT5, 1.7B sub-8-bit, asymmetric INT4
- Assume 9B fixable
- Expect CUDA Graph speedup > 4%
- Ignore GGUF v3 offset format (must add tensor_data_off)

**Current direction**: GGUF bridge Phase 1-3 complete. Next: fix Llama server crash, validate quality, add AWQ calibration.