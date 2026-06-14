# Blackwell Enhancement Roadmap

**Scope**: Tracks A (kernel perf) + B (server) + C (quality) + D (testing).
**Risk posture**: Incremental kernel work first (dp4a INT4 + existing fused kernels). Marlin-style rewrite deferred to Phase 4, gated on Phase 2 results.
**Horizon**: Multi-session. Phases ordered by dependency + ROI.
**Source**: Synthesized from scout_kernels.md, scout_servers.md, scout_bench_bridge.md, research_sota.md.

---

## Sequencing Rationale

- **D (testing) goes first**: golden regression is the safety net that makes every later change reviewable. Without it, kernel changes can silently break quality (AGENTS.md has 8+ such bug-history entries).
- **A1 (fused kernels) before A2 (dp4a INT4)**: free launch reduction, zero numerical risk. Verifies the harness before touching the hot kernel.
- **A2 (dp4a INT4) before Marlin**: validates the compute-bound hypothesis cheaply. If dp4a gets us to parity with llama.cpp (84 t/s), Marlin may be unnecessary.
- **B (server)** parallelizable with A — different files, no dependency.
- **C (quality)** depends on D (PPL regression) to measure improvement.

---

## Phase 0 — Testing Harness (Track D) [~1 session]

**Goal**: Catch silent quality regressions. Everything downstream depends on this.

### 0.1 Golden token-sequence regression test
- **What**: Fixed prompts → exact expected greedy-decode token IDs for each production model (Qwen3 8B INT4, Qwen3 1.7B INT8, Llama 3.2 1B, Gemma 12B).
- **Why**: Any kernel/quant/format change that alters output is caught immediately. Highest-leverage single addition.
- **Files**: new `bench/regression_golden.cu`, golden vectors in `tests/golden/`.
- **Acceptance**: `./bench/regression_golden` exits 0 on current binaries; exits non-zero if any kernel mutated. Run before/after every later phase.
- **Effort**: 0.5 day.

### 0.2 PPL regression harness
- **What**: Unify the 6 scattered `bench_ppl_*.cu` into one corpus-driven runner with a real WikiText-2 sample (not the ~200-word Austria snippet).
- **Why**: PPL is the quality signal for calibration work (Phase 5). Current corpus too small + no automation.
- **Files**: new `bench/bench_ppl_unified.cu` reading `tests/data/wikitext-2-raw_sample.txt`, scripts to diff PPL against last-known-good.
- **Acceptance**: Single command runs PPL for any model, logs to `tests/ppl_baseline.json`, fails CI if PPL drifts >5%.
- **Effort**: 0.5 day.

### 0.3 GTest skeleton revival
- **What**: Replace the 4 `GTEST_SKIP` stubs with real kernel unit tests: GEMV known-small-matrix in→out, RMSNorm, quantize/roundtrip, tokenizer encode→decode.
- **Why**: Targeted correctness checks complement golden regression (which is end-to-end).
- **Files**: `tests/test_*.cu` (rewrite bodies), `CMakeLists.txt` (make GTest non-optional when present).
- **Acceptance**: `ctest --test-dir build` passes ≥8 unit tests.
- **Effort**: 1 day (defer if Phase 0.1+0.2 land cleanly — 0.1 covers most risk).

---

## Phase 1 — Server Robustness P0 (Track B) [~1 session]

**Goal**: Fix correctness/security bugs before adding features. Parallelizable with Phase 2.

### 1.1 Streaming endpoint race condition
- **Bug**: `http_subprocess.cpp:760` writes subprocess stdin without `g_engine.lock` → concurrent stream corrupts in-flight non-streaming request.
- **Fix**: Acquire lock in ContentProvider lambda before write.
- **Verify**: Phase 0 golden test still passes; concurrent curl stream + POST returns coherent.
- **Effort**: 0.25 day.

### 1.2 Context-length validation
- **Bug**: `inference_server_int4.cu:480` — prompt + max_tokens can exceed MAXSEQ → silent KV cache overflow / memory corruption.
- **Fix**: Pre-tokenize, assert `input_ids.size() + max_new <= MAXSEQ`, return HTTP 400 if exceeded.
- **Verify**: Request with 2048-token prompt + max_tokens=2048 → 400, not crash.
- **Effort**: 0.25 day.

### 1.3 Graceful shutdown
- **Bug**: `http_subprocess.cpp:815` — no SIGINT/SIGTERM handler → orphaned subprocess holds GPU.
- **Fix**: `signal(SIGTERM/SIGINT, handler)` → `svr.stop()` + `g_engine.stop()` (kill child with SIGTERM first, then SIGKILL after 5s).
- **Verify**: `kill -TERM $(pgrep http_subprocess)` → clean exit, no orphan, GPU freed (`nvidia-smi`).
- **Effort**: 0.25 day.

### 1.4 MAXSEQ unification
- **Bug**: `inference_server_int4.cu:24` MAXSEQ=2048 vs `inference_server_int4_batched.cu:22` MAXSEQ=4096. Same model, different KV allocation.
- **Fix**: Define once in shared header, both servers use MAXSEQ=4096 (RTX 5060 Ti 16GB fits; 8B INT4 weights 5.3GB + 4096 KV cache ~1GB).
- **Verify**: Both servers report same MAXSEQ; KV cache allocation identical.
- **Effort**: 0.25 day.

---

## Phase 2 — Fused Kernel Wiring (Track A1) [~1 session]

**Goal**: Free launch reduction using kernels already built but unused. Zero numerical risk.

### 2.1 Wire `fused_rmsnorm_quant_int4` into server
- **Current**: `inference_server_int4.cu:274-275` calls `fused_rmsnorm` then `quantize_int4_batched` (2 launches).
- **Change**: Single `fused_rmsnorm_quant_int4` call. Same at post-attn norm (line 291-292) and final norm (line 283-284).
- **Saves**: 3 launches/layer × 36 = 108 launches/token + eliminates `d_xi_f` intermediate (16KB read+write × 3/layer).
- **RISK**: scout found `fused_int4_ops.cu:149` multi-block bug (incorrect RMSNorm when N>4096, e.g. MLP dim 12288). **Must verify kernel correctness for I=12288 before wiring** — either fix the multi-block reduction or constrain to single-block dims.
- **Verify**: Phase 0 golden test bit-exact match. nsys shows fewer launches.
- **Effort**: 0.5 day (including multi-block bug fix/verification).

### 2.2 Wire `fused_swiglu_quant_int4`
- **Current**: `inference_server_int4.cu:276-277` — `apply_swiglu` then `quantize_int4_batched`.
- **Change**: Single fused call.
- **Saves**: 1 launch/layer × 36 = 36 launches/token.
- **Verify**: Golden test bit-exact.
- **Effort**: 0.25 day.

### 2.3 Fuse head_norm + RoPE
- **Current**: 4 launches/layer (2× head_norm, 2× apply_rope) at lines 261-264.
- **Change**: Write new fused kernel (head_norm + RoPE per Q/K) — 2 launches.
- **Saves**: 2 launches/layer × 36 = 72 launches/token.
- **Verify**: Golden test bit-exact. nsys confirms 2→ wait, 4→2 launches/layer.
- **Effort**: 0.5 day.

### 2.4 Gate decision: A2 dp4a vs Marlin
- **After 2.1-2.3**: re-benchmark M=1 t/s. If ≥80 t/s (llama.cpp parity) → skip Phase 4 (Marlin), pursue spec decode instead. If still <70 → dp4a (Phase 3) then evaluate Marlin.

---

## Phase 3 — dp4a INT4 GEMV (Track A2) [~2 sessions]

**Goal**: 2-3× GEMV compute speedup. GEMV is 92% of decode time. Keep offset-binary format.

### 3.1 Verify `int4_byte_to_floats` numerical baseline
- **What**: Capture current INT4 GEMV output for known input → reference for dp4a equivalence check.
- **Why**: dp4a path must be bit-identical (or within FP32 rounding) to scalar path. Offset-binary (nib-8) → int8 sign-extension mapping must be proven.
- **Mapping**: `nib_val = nib - 8` gives range [-8..7]. Pack as int8: `(int8_t)((nib << 4) >> 4)` sign-extends. Two nibbles → one int8 pair → one `__dp4a`.
- **Verify**: Unit test (Phase 0.3) comparing scalar vs dp4a GEMV output, max abs diff < 1e-4.
- **Effort**: 0.5 day.

### 3.2 Implement dp4a INT4 inner loop
- **File**: `src/kernels/gemv_int8.cu:998-1067` (`gemv_int4_batched_kernel`).
- **Change**: Replace `int4_byte_to_floats` + scalar FP32 FMAs with int8-packed `__dp4a`. 4 `__dp4a` calls process 16 INT4 values (8 bytes) vs 8 scalar FMAs.
- **Same change**: `gemv_int4_warp_kernel` (line 482-542), `gemv_fp32_int4_warp_kernel` (line 547-601).
- **Verify**: Phase 0 golden bit-exact. nsys GEMV kernel time drops. Re-run nsys profile.
- **Effort**: 1.5 days.

### 3.3 Speculative decoding feasibility (stretch)
- **What**: `bench/text_generate_speculative.cu` exists (research). Evaluate for Qwen3-8B: small draft head (Qwen3-0.5B) generates K candidates, 8B verifies in 1 pass.
- **Why**: Memory-bound decode → verifying K tokens ≈ cost of 1. 1.5-2× speedup at M=1.
- **Gate**: Only if Phase 3.2 doesn't reach llama.cpp parity.
- **Effort**: 2 days (if pursued).

---

## Phase 4 — Server API Compliance (Track B) [~2 sessions]

**Goal**: OpenAI API compatibility for drop-in client support.

### 4.1 OpenAI-compatible streaming
- **Change**: Both `/v1/completions` and `/v1/chat/completions` honor `"stream":true` in body (SSE `data: {...}\n\n`, `data: [DONE]`). Deprecate custom `/v1/completions/stream` path.
- **Verify**: OpenAI Python client `client.completions.create(stream=True)` works unmodified.
- **Effort**: 0.5 day.

### 4.2 Multi-turn chat fix
- **Bug**: `http_subprocess.cpp:414` — `extract_chat_content()` returns first `"content"`, ignores system/assistant/multi-turn.
- **Change**: Parse full messages array, apply Qwen3 chat template (`<|im_start|>role\n...<|im_end|>`), support system + multi-turn.
- **Verify**: 3-turn conversation returns contextually coherent continuation.
- **Effort**: 0.5 day.

### 4.3 Missing sampling/output params
- **Add**: `top_p` (nucleus), `stop` sequences, `seed`, `n>1`, `logprobs`, `echo`.
- **Change**: Extend `sample_gpu` (already has top_k) for top_p; stop-sequence check in decode loop; seed param to RNG.
- **Verify**: Each param exercised in `scripts/benchmark_suite.py`.
- **Effort**: 1 day.

### 4.4 Correct token counting
- **Bug**: `prompt_tokens` hardcoded 0/1 (`http_subprocess.cpp:637,679`).
- **Change**: Pre-tokenize prompt, return real count in `usage`.
- **Verify**: `usage.prompt_tokens + completion_tokens == total_tokens`.
- **Effort**: 0.25 day.

### 4.5 Proper JSON parser
- **Bug**: Hand-rolled `json_string_at`/`json_int_at` (`http_subprocess.cpp:424-460`) — fragile, no nested objects/arrays.
- **Change**: vendor single-header JSON lib (nlohmann/json or rapidjson). Replace all hand-rolled parsing.
- **Constraint**: C++17, no new runtime deps (header-only OK).
- **Verify**: All existing curl tests pass; nested JSON bodies parse.
- **Effort**: 0.75 day.

---

## Phase 5 — Quality & Calibration (Track C) [~2 sessions]

**Goal**: Improve INT4 PPL below 21.82. Depends on Phase 0.2 (PPL harness) for measurement.

### 5.1 Fix AWQ formula + per-layer calibration
- **Bug**: `scripts/quantize_awq_int4_8b.py:155` reuses layer-0 stats for all 36 layers; uses simplified `(act/mean)^α` not paper's `(s_w·s_a)^α / (s_a^α + s_w^α)`.
- **Change**: (1) Real per-layer forward pass collecting activation stats through all 36 layers. (2) Correct AWQ formula. (3) Larger corpus (512+ WikiText-2 sequences, not 35 hardcoded prompts).
- **Verify**: Phase 0.2 PPL harness. Target PPL < 20 (from 21.82).
- **Effort**: 1.5 days.

### 5.2 BpeTokenizer Unicode support
- **Bug**: `bpe_tokenizer.h:380` — `isLetter()` ASCII-only. Non-English (CJK, Cyrillic) produces wrong tokens silently.
- **Change**: UTF-8-aware pretokenizer. Qwen3 is multilingual — correctness risk.
- **Verify**: Roundtrip test (Phase 0.3) with Chinese/Russian prompts; compare token IDs against HF reference.
- **Effort**: 1 day.

### 5.3 Missing PPL benches
- **Add**: INT8 8B, Gemma 12B, 9B GDN PPL to unified harness (Phase 0.2).
- **Why**: INT8 8B "coherent" claim has no PPL number; Gemma 11GB weights quality unverified; 9B PPL bench referenced but file missing.
- **Verify**: PPL logged for all production models in `tests/ppl_baseline.json`.
- **Effort**: 0.5 day.

---

## Phase 6 — Cleanup (Track E) [~0.5 session, opportunistic]

- Delete `weights_int2_qwen3_8b/` (3.9GB, PPL 47B, abandoned) + `weights_int4_qwen3_1.7b/` (1.5GB, dead). Save 5.4GB.
- Sync AGENTS.md weight dir list with disk reality (5 listed deleted, 2 unlisted).
- Update Docker: Dockerfile→v0.11.x, Dockerfile.int4→v0.11.x, docker-compose→add INT4 service + current images. Add `HEALTHCHECK`.
- Decide on 22+ dead kernels: annotate `[[maybe_unused]]` or move to `src/kernels/legacy/` (don't delete — reference value per AGENTS.md convention).

---

## Dependency Graph

```
Phase 0 (testing) ─┬─► Phase 1 (server P0)
                   ├─► Phase 2 (fused kernels) ─► Phase 3 (dp4a) ─┐
                   └─► Phase 5 (quality) ◄─ Phase 0.2 (PPL)       ├─► [gate: Marlin?]
                                                                │
Phase 4 (server API) ◄─ independent (needs Phase 1 done first)  ┘
Phase 6 (cleanup) ◄─ opportunistic, anytime
```

## Gates

1. **After Phase 2**: if M=1 ≥ 80 t/s → skip Phase 4 (Marlin), do spec decode (3.3).
2. **After Phase 3**: if dp4a not bit-exact in 3.1 verification → revert, escalate to Marlin.
3. **Before Phase 5.1**: Phase 0.2 PPL harness must pass (else can't measure improvement).

## Open Questions (resolve during execution)

1. **`fused_rmsnorm_quant_int4` multi-block bug**: fix reduction or constrain to single-block? (Phase 2.1)
2. **dp4a offset-binary mapping**: does `(nib << 4) >> 4` sign-extend correctly for nib=0? (Phase 3.1 — nib=0 → 0, but offset-binary wants -8; need different mapping)
3. **Speculative decode draft head**: no pre-trained EAGLE head for Qwen3-8B — train one or use Qwen3-0.5B as n-gram draft? (Phase 3.3)
4. **JSON parser license**: nlohmann/json (MIT) vs rapidjson (MIT) — both OK, prefer nlohmann for ergonomics. (Phase 4.5)

## Effort Summary

| Phase | Track | Effort | Sessions |
|-------|-------|--------|----------|
| 0 | D | 2 days | 1 |
| 1 | B | 1 day | 1 |
| 2 | A1 | 1.5 days | 1 |
| 3 | A2 | 2 days (+2 spec) | 2 |
| 4 | B | 3 days | 2 |
| 5 | C | 3 days | 2 |
| 6 | E | 0.5 day | 0.5 |
| **Total** | | **~12.5 days** | **~9 sessions** |

Marlin rewrite (deferred): +5-7 days if Phase 2/3 gate fails.
