# Code Context — Blackwell Repo Scout (2026-06-10)

## 1. Repo State

- **Branch:** `master`
- **Last commit:** `ecf69b7` — "Refresh HANDOFF.md: remove stale task, add batched benchmark file"
- **Git status:** Dirty — 12 modified files + 8 untracked
  - Modified: `AGENTS.md`, `Dockerfile.int4`, `HANDOFF.md`, `bench/text_generate_int4_batched.cu`, `server/inference_server`, `server/inference_server_9b`, `server/inference_server_nofp4.cu`, `server/inference_server_qwen35_9b.cu`, `src/kernels/gemv_int8.cu`
  - Untracked: `bench/diag_9b_ssm.py`, `bench/text_generate_int4_1.7b.cu`, `bench/tokenize_corpus.cu`, `bench/tokenize_text.cu`, `better-inference/`, `scripts/quantize_awq_int4_8b.py`, `server/inference_server_nofp4`, `wiki_corpus.txt`
- **Readme/AGENTS.md** present with full project context (Session 1-65 documented)

## 2. Build System

- **CMakeLists.txt:** Uses CUDA 13.3, `compute_120a`, GCC-12 host compiler. Library target `blackwell_kernels` — 179 symbols confirmed.
- **Compiler:** `/usr/local/cuda-13.3/bin/nvcc`
- **Library:** `build/libblackwell_kernels.a` — 179 exported symbols (matches expected 179)
- **Build dir:** `build/bench/` does NOT exist — bench targets built ad-hoc outside CMake
- **No CMake targets for new bench files** — `text_generate_int4_1.7b.cu`, `text_generate_int4_batched.cu`, `tokenize_*.cu`, `diag_9b_ssm.py` not in CMakeLists.txt

## 3. Key New Files

### `better-inference/` (855 lines total)
- **`gguf.h`** (456 lines) — Standalone GGUF header parser. No llama.cpp dependency.
- **`gguf_convert.cpp`** (333 lines) — GGUF → blackwell INT4 format converter.
- **`gguf_test.cpp`** (66 lines) — Minimal test.
- **`DESIGN.md`** — Phase plan: GGUF parser → Tensor converter → Llama 3.1 8B → AWQ calibration service.
- **Binaries:** `gguf_convert` (69KB, built Jun 10 06:41), `gguf_test` (63KB, built Jun 10 06:12).

### `scripts/quantize_awq_int4_8b.py` (22,699 bytes)
- AWQ INT4 calibration for 8B model.

### `bench/text_generate_int4_1.7b.cu` (17,138 bytes, untracked)
- INT4 text generation for 1.7B model (previously missing).

### `bench/tokenize_corpus.cu` (2,167 bytes, untracked)
- Tokenizes text corpus, writes token IDs to file.

### `bench/tokenize_text.cu` (1,263 bytes, untracked)
- Tokenizes single text string.

### `bench/diag_9b_ssm.py` (4,617 bytes, untracked)
- Diagnose 9B GatedDeltaNet SSM instability.

## 4. Binary Presence

| Binary | Path | Size | Built | Status |
|--------|------|------|-------|--------|
| `bench/bench_ppl_int4_8b` | ✓ | 1.1 MB | Jun 8 13:52 | ✅ |
| `bench/text_generate_int4_batched` | ✓ | 5.2 MB | Jun 9 20:29 | ✅ |
| `bench/text_generate_int4_qwen3_8b` | ✓ | 4.2 MB | Jun 9 16:06 | ✅ |
| `server/inference_server_int4` | ✓ | 4.2 MB | Jun 9 16:27 | ✅ |
| `server/http_subprocess` | ✓ | 1.2 MB | Jun 8 17:49 | ✅ |
| `better-inference/gguf_convert` | ✓ | 69 KB | Jun 10 06:41 | ✅ (new) |
| `server/inference_server_nofp4` | ✓ | untracked | (new) | ✅ |

**No missing critical binaries.** All expected INT4/HTTP binaries present.

## 5. GPU State

- **GPU:** NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM (16311 MiB)
- **Compute Capability:** 12.0
- **Available:** ✅ — GPU online, nvidia-smi functional

## 6. Weight Directories

10 weight dirs present:

| Dir | Size | Purpose |
|-----|------|---------|
| `weights_int4_qwen3_8b/` | **5.8 GB** | ✅ INT4 8B weights (508 files: .int4_t + .scale_t per tensor) |
| `weights_int8_qwen3_8b/` | present | INT8 8B weights |
| `weights_int8_qwen3_8b_all_int8/` | present | Pure INT8 copy |
| `weights_int8_qwen3_8b_mixed/` | present | Mixed FP16+INT8 |
| `weights_int8_bf16/` | present | 1.7B INT8 |
| `weights_int8_qwen35_9b/` | present | 9B GDN INT8 |
| `weights_int8_qwen35_9b_mixed/` | present | 9B mixed |
| `weights_int8_qwen35_9b_all_fp16/` | present | 9B all-FP16 |
| `weights_fp8_bf16/` | present | Dead-end FP8 |
| `weights_int8_per_row/` | present | Experimental |

**INT4 8B weights confirmed** (5.8 GB, 508 files).

## 7. Discrepancies & Issues

1. **Dirty working tree** — Modified source files (`inference_server_nofp4.cu`, `inference_server_qwen35_9b.cu`, `gemv_int8.cu`, `text_generate_int4_batched.cu`) not yet committed. Risk of stale build if rebuild required.

2. **Stale build** — `build/libblackwell_kernels.a` last built unconfirmed. `src/kernels/gemv_int8.cu` modified — **must rebuild** before benchmarking.

3. **Better-inference CMake integration missing** — `gguf_convert.cpp` built ad-hoc via g++, not part of CMake. Needs `add_executable` target.

4. **GoogleTest crash** — `better-inference/gguf_test` built but may fail on machines without GoogleTest installed (CMake test target not investigated).

5. **No new bench files in CMake** — `text_generate_int4_1.7b.cu`, `tokenize_corpus.cu`, `tokenize_text.cu` built manually. Not reproducible via `cmake --build`.

## 8. Summary

Repo is in active development state. All critical binaries present. INT4 8B weights confirmed at 5.8 GB. Build is slightly stale — `gemv_int8.cu` modified since last build. Better-inference GGUF parser is new and promising (855 LOC, no llama.cpp dep). 179 kernel symbols match expectation.

**Start here if building:** Rebuild library first: `cmake -B build && cmake --build build --parallel`

**Start here if adding CMake targets:** Edit `CMakeLists.txt` to add `better-inference/gguf_convert.cpp` and new bench files as executable targets.
