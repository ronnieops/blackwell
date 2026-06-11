# HANDOFF.md — blackwell

**ALWAYS read `AGENTS.md` AND this file before acting.**

---

## 1. Current Objective

Multi-model INT4 inference on RTX 5060 Ti (GB206, SM_120a). Session 76: fixed
GGUF layout transpose bug, switched Llama to safetensors path (coherent output),
per-model EOS tokens, server config cleanup.

---

## 2. Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Qwen3-8B INT4 server | ✅ **Production** | **56 t/s**, PPL **21.82**, coherent |
| Llama 3.2 1B INT4 (safetensors) | ✅ **Working** | **270 t/s**, coherent output |
| Llama 3.1 8B INT4 (safetensors) | ✅ **Weights ready** | 5.7 GB, not yet tested in server |
| Llama 3.2 1B GGUF path | ⚠️ **Partial fix** | transpose_f32 added, still garbled |
| Llama 3.1 8B GGUF path | ❌ **Broken** | Same GGUF structural issue |
| SSE streaming | ✅ | All 3 models |
| Batch endpoint | ✅ | All 3 models |
| FP16 benchmark | ✅ | `text_generate_llama32_1b_fp16` — 76 t/s, debug |
| gemv_fp32 kernel | ✅ | `gemv_fp32_launch` — high-precision GEMV |
| EOS token | ✅ **Fixed** | Per-model via ModelConfig.eos_id |
| Kernel symbols | ✅ | **189** in `libblackwell_kernels.a` |

---

## 3. Recent Decisions

### Duplicate GEMV Bug (Session 74)
Both Llama benchmarks had duplicate gate+up GEMV after down projection.
Llama 3.2 1B: 223→287 t/s (+28%). Llama 3.1 8B: 43→61 t/s (+42%).

### Llama Server Crash (Session 74, 5 fixes)
1. `static std::vector<LW4> W(NL)` — NL derefed `cfg=NULL` at static init. Fix: empty decl + `W.resize(NL)`
2. Hardcoded `weights_int4_qwen3_8b/` paths — Fix: use `wdir` variable
3. Wrong config paths — Fix: absolute paths in MODELS[]
4. Wrong `strstr` order — Fix: `strstr(wdir, model)` + exact match
5. Invalid warmup token (151643 > 128256) — Fix: dummy(1, 0)

### SSE Streaming (Session 75)
Both `inference_server_llama` and `inference_server_int4` now emit per-token SSE
(`data: {"token":N,"text":"..."}\n\n` + `data: [DONE]`). Streaming endpoint
`POST /v1/completions/stream` was already in http_subprocess.

### GGUF Transpose Fix (Session 76)
GGUF stores weights as [K][N] (input_dim x output_dim) row-major.
GEMV kernels expect [N][K] (output_dim x input_dim).
Converter now transposes dequantized FP32 buffer before requant.
- Applies to Q4_K, Q6_K, Q5_0, Q8_0
- PARTIAL fix: output changed from "adle" to "rites" — still garbled
- Needs dequant verification against llama.cpp internals

### Llama from Safetensors (Session 76) ✅
New scripts/quantize_llama32_1b.py and quantize_llama31_8b.py
- Proven path (same as Qwen3-8B production)
- Llama 3.2 1B: 270 t/s, coherent output
- Llama 3.1 8B: 5.7 GB weights ready, not yet bench-verified
- GGUF path deprecated for production use; keep as fallback

### EOS Token Fix (Session 76)
- Added eos_id to ModelConfig struct
- Llama uses 128001 (was hardcoded to Qwen's 151643)
- inference_server_llama.cu: CFG_EOS_ID() + per-model value
- start_servers.sh: points to safetensors weight paths
- Dockerfile.int4: includes inference_server_llama binary

### QK Norms Fix (Session 75)
GGUF has NO `attn_q_norm`/`attn_k_norm` tensors. Converter now initializes
`qk_norms.f32` to 1.0 (identity). Did NOT fix GGUF output.

---

## 4. Important Constraints

- **Hardware**: RTX 5060 Ti 16 GB, compute 12.0, 36 SMs, ~500 GB/s GDDR7
- **Toolchain**: CUDA 13.3, `compute_120a` (NOT `compute_120`), CMake 3.x
- **Nvcc**: `/usr/local/cuda-13.3/bin/nvcc`
- **Model dims (Llama 3.2 1B)**: NL=16, H=2048, nqh=32, nkv=8, hd=64, rope=500000, V=128256
- **Model dims (Llama 3.1 8B)**: NL=32, H=4096, nqh=32, nkv=8, hd=128, rope=500000, V=128256
- **Model dims (Qwen3-8B)**: NL=36, H=4096, nqh=32, nkv=8, hd=128, rope=1000000, V=151936
- **Pre-measurement**: `killall hashcat 2>/dev/null` before any GPU measurement
- **Build**: `CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build && cmake --build build --parallel`
- **Kernel validation**: `nm build/libblackwell_kernels.a | c++filt | grep " T blackwell" | wc -l` (expect 189)
- Server warmup needs subprocess to print "Ready." before accepting requests

---

## 5. Known Issues / Risks

| Issue | Severity | Notes |
|-------|----------|-------|
| Llama GGUF quality degraded | MEDIUM | Safetensors path works; GGUF converter needs dequant fix |
| Llama 3.1 8B not bench-verified | MEDIUM | Weights quantized, server not tested |
| Server subprocess race on startup | MEDIUM | `ready` flag set before subprocess loads (3s timeout on batch could fail) |
| LocalTokenizer path hardcoded per model | MEDIUM | Need model→path mapping table for multi-model |
| Batch handler uses tempfile for request | LOW | Fragile IO, works in practice |
| 9B SSM stability (A_log > 0) | BLOCKED | Architectural, not quantization |

---

## 7. Pending Tasks

| Priority | Task | Notes |
|----------|------|-------|
| MEDIUM | Verify Llama 3.1 8B bench | Weights ready, run text_generate_llama31_8b |
| LOW | ModelOpt calibration integration | PPL 21.82 is production-ready |
| LOW | NVFP4 format conversion | Format encoding mismatch, abandoned |
| LOW | Server chat template per-model | Hardcoded Qwen format, needs Llama template |
| BLOCKED | 9B SSM fix | Architectural issue |

---

## 8. Important Files / Commands

### Benchmarks
```bash
./bench/text_generate_llama32_1b "Hello" 10     # 287 t/s
./bench/text_generate_llama31_8b "Hello" 10     # 61 t/s
./bench/text_generate_int4_qwen3_8b "Hello" 10  # 56 t/s (coherent)
./bench/text_generate_llama32_1b_fp16 "Hello" 10 # 76 t/s (FP16, garbled)
./bench/bench_ppl_int4_8b                        # PPL 21.82
```

### HTTP Server
```bash
./server/http_subprocess 8123 llama32-1b &    # Llama 3.2 1B on 8123
./server/http_subprocess 8124 llama31-8b &    # Llama 3.1 8B on 8124
./server/http_subprocess 8125 qwen3-8b &      # Qwen3-8B on 8125

# Endpoints
curl http://localhost:8123/health
curl -X POST http://localhost:8123/v1/completions -d '{"prompt":"Hello","max_tokens":5}'
curl -X POST http://localhost:8123/v1/completions/stream -d '{"prompt":"Hello","max_tokens":5,"stream":true}'
curl -X POST http://localhost:8123/v1/batch -d '{"prompts":["Hello","World"],"max_tokens":5}'
```

### Start All Servers
```bash
./start_servers.sh  # Starts all models (Llama 8123, Llama 8B 8124, Qwen3 8125)
```

### Build
```bash
CUDACXX=/usr/local/cuda-13.3/bin/nvcc cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# http_subprocess (g++ only)
/usr/bin/g++ -O2 /tmp/httplib.o server/http_subprocess.cpp -I include -I /usr/local/cuda-13.3/include \
  -L /usr/local/cuda-13.3/targets/x86_64-linux/lib -o server/http_subprocess \
  -lpthread -lz -lssl -lcrypto -lcudart
```

### GGUF Converter
```bash
./better-inference/gguf_convert model.gguf output_dir/        # INT4 (default)
./better-inference/gguf_convert model.gguf output_dir/ --fp16 # FP16 (lossless)
```

---

## 9. Validation Status

| Check | Value | Status |
|-------|-------|--------|
| Kernel symbols | **189** | ✅ |
| Qwen3-8B PPL | **21.82** | ✅ |
| Llama 3.2 1B t/s (safetensors) | **270** | ✅ |
| Llama 3.2 1B PPL | TBD | ⏳ |
| Llama 3.1 8B t/s (safetensors) | TBD | ⏳ |
| Qwen3-8B t/s | **56** | ✅ |
| Server health | **OK** | ✅ |
| SSE streaming | **Working** | ✅ |
| Batch endpoint | **Working** | ✅ |
| Per-model EOS token | **Fixed** | ✅ |

---

## 10. Session Metadata

| Field | Value |
|-------|-------|
| updated_at | 2026-06-11 |
| branch | master |
| repo_state | Clean (11 commits ahead, all pushed) |
| active_binaries | inference_server_llama, inference_server_int4, http_subprocess |
| weight_dirs | llama32-1b-int4-from-safetensors, llama31-8b-int4-from-safetensors, qwen3-8b-int4 |
| GPU | RTX 5060 Ti, free |

---

## META PROMPT

Before acting, read BOTH `AGENTS.md` and `HANDOFF.md` fully. These files contain
all verified operational context for the blackwell project.

Key context:
- **Qwen3-8B INT4** is the production path (56 t/s, PPL 21.82, coherent output)
- **Llama from safetensors** is the working path (270 t/s, scripts/quantize_llama*.py)
- **Llama GGUF** path has partial transpose fix but still broken — needs dequant verification
- **Server** supports completion, SSE streaming, and batch endpoints
- **189 kernel symbols** in the library
- **3 models** in HTTP server: llama32-1b (8123), llama31-8b (8124), qwen3-8b (8125)
- **Per-model EOS** via ModelConfig.eos_id (Llama: 128001, Qwen: 151643)
- All 11 session 76 commits are pushed to `origin master`

Do NOT:
- Re-investigate NVFP4 (abandoned — format encoding mismatch)
- Re-investigate 9B SSM quality (blocked — architectural)
- Use GGUF converter for production Llama weights (use scripts/quantize_llama*.py)
- Duplicate information from AGENTS.md in HANDOFF.md

Do:
- Verify repo state (`git status`) before any edits
- Prefer incremental changes over full re-writes
- Check `nm` count if changing kernel sources
- Confirm `hashcat` is killed before GPU measurements
- Keep HANDOFF.md compact and deduplicated
