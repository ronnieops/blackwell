# Research: INT4 Inference Throughput Optimization — Blackwell RTX 5060 Ti

## Summary
GEMV dominates at 92.2% of decode runtime; weight loading is the bottleneck. Most optimizations outside GEMV yield <5% gains. Worth implementing: prefill attention (10× prompt speedup), batch processing (v0.8.3 already has M=1-8), and TMA-based weight streaming if GB206 supports it. Not worth: further CUDA Graph effort, MLA attention, PagedAttention, speculative decoding, MPS/MIG, or additional kernel fusion beyond current state.

## Findings

### 1. CUDA Graph — Diminishing Returns Confirmed
**CUDA Graph captured 867 nodes. Speedup: 64→65 t/s (+2.1%).** This matches expectations when a single kernel type dominates runtime. GEMV at 92.2% means even if kernel launch overhead were 100% eliminated, max gain = 7.8%. vLLM reports similar: CUDA Graph helps most when many small kernel launches exist (e.g. Transformer with 40+ layers × 15 ops = 600 launches). With GEMV at 55 μs average and launch overhead ~3-5 μs on Blackwell, the savings are <10% total. [Source: Efficiently Serving LLMs Part 4 — CUDA Graphs in vLLM](https://www.linkedin.com/pulse/efficiently-serving-llms-part-4-how-cuda-graphs-make-vllm); [vLLM CUDA Graph Design](https://docs.vllm.ai/en/stable/design/cuda_graphs/); [llama.cpp CUDA Graph Issue #6763](https://github.com/ggml-org/llama.cpp/issues/6763)

**Verdict: SKIP further work. 2.1% confirmed ceiling.**

### 2. Flash Attention / MLA — Not Relevant for Decode-Only
MLA (Multi-head Latent Attention) reduces KV cache size via low-rank projection — relevant for memory-bound serving at large batch sizes. At M=1 decode, attention is 0.9% of runtime (3.9 μs avg). Current GQA kernel already handles nqh=32, nkv=8, hd=128 efficiently. Flash Attention optimizes for long-context prefill (softmax tiling), not single-token decode. [Source: TransMLA — arXiv 2502.07864](https://arxiv.org/abs/2502.07864); [DeepSeek MLA Analysis — arXiv 2506.02523](https://arxiv.org/abs/2506.02523)

**Verdict: SKIP. Attention is 0.9% of wall time.**

### 3. PagedAttention / vLLM Patterns — No Decode-Side Benefit
PagedAttention optimizes KV cache memory management for large batches (fragmentation, sharing). For single-sequence M=1 decode, the current fixed KV cache (`MAXSEQ=4096`, pre-allocated per layer) is optimal. Paged memory would add indirection overhead with zero benefit at M=1. vLLM uses it for batching thousands of sequences. [Source: HuggingFace PagedAttention docs](https://huggingface.co/docs/text-generation-inference/en/conceptual/paged_attention)

**Verdict: SKIP. M=1 decode has no fragmentation problem.**

### 4. Blackwell-Specific Optimizations — Potentially Valuable

#### 4a. Tensor Memory Accelerator (TMA) / TMEM
Blackwell (SM_120) introduces Tensor Memory Accelerator (TMA) and Tensor Memory (TMEM). TMA enables async 1D/2D/3D data copies from global→shared without register file usage. **Critical question**: Does GB206 (RTX 5060 Ti) include TMA? Blackwell datacenter (B100/B200/GB202) has TMA. Consumer GB206 may lack tensor memory hardware — this needs verification via sm_120a PTX ISA support. If TMA is available, weight streaming into shared memory via TMA could overlap with tensor core compute, reducing effective GEMV time. [Source: NVIDIA Blackwell Tuning Guide 13](https://docs.nvidia.com/cuda/blackwell-tuning-guide/); [Microbenchmarking Blackwell — arXiv 2512.02189](https://arxiv.org/abs/2512.02189); [Blackwell SM100 TMEM/TMA blog — Jianyu Huang](https://jianyuh.github.io/cuda/2026/04/12/blackwell-tensor-core)

#### 4b. WGMMA / MKI Instructions
Blackwell adds `wgmma.mma_sp` (sparse tensor core) and MKI instructions. These target GEMM workloads with tile sizes ≥ 16×64. Current GEMV (1×N vector × N×M matrix) has tile size [1×128] × [128×4096] — tensor cores are inefficient for such skinny dimensions. WGMMA would require batching multiple queries to form a GEMM. At M=8, batched GEMV already gives 169 t/s (2.7× M=1). WGMMA could accelerate the batched path further. [Source: Semianalysis — Dissecting NVIDIA Blackwell Tensor Cores](https://newsletter.semianalysis.com/p/dissecting-nvidia-blackwell-tensor); [CUTLASS Blackwell Changelog](https://docs.nvidia.com/cutlass/latest/CHANGELOG.html)

#### 4c. CUTLASS 3.x + Blackwell
CUTLASS 3.5+ has Blackwell (SM_120) kernel support. Could generate batched GEMV kernels using CUTLASS instead of hand-written. Gains uncertain — current hand-optimized warp-cooperative GEMV uses dp4a SIMD at near-peak L2 bandwidth. Tensor core GEMV for M=1 likely slower due to setup overhead. [Source: CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/overview.html); [CUTLASS Tutorial: Tensor Memory Accelerator](https://research.colfax-intl.com/cutlass-tutorial-tma/)

**Verdict: INVESTIGATE TMA support on GB206. If present, prototype TMA weight streaming. WGMMA for batched M≥4 is promising but lower priority.**

### 5. Speculative Decoding — NOT VIABLE (tested 2026-06-13)
`bench/text_generate_speculative.cu` — 1.7B INT4 draft + 8B INT4 target.
Acceptance rate: 1.3% (draft and target produce completely different tokens).
Throughput: 30 t/s (slower than target alone at 56 t/s due to sequential execution).

**Root cause**: INT4 1.7B draft quality is too poor to predict 8B target output.
Would need FP16/INT8 draft (higher quality) which uses more GPU memory.
Not viable on 16 GB GPU with both models loaded.

**Verdict: ABANDONED. Speculative decode requires a higher-quality draft model
that doesn't fit in available GPU memory alongside the target.**

### 6. Server Concurrency — MPS/MIG Not Useful
**MPS (Multi-Process Service)**: Shares single GPU context across processes. Reduces context-switch overhead for multiple independent clients. But MPS adds latency and has known bugs with CUDA Graph capture. Not useful for single-GPU decode serving where throughput is CPU-side request parsing bound.
**MIG (Multi-Instance GPU)**: Only available on datacenter Blackwell (B100/B200). RTX 5060 Ti does not support MIG. [Source: NVIDIA MPS Docs](https://docs.nvidia.com/deploy/mps/latest/index.html); [NVIDIA MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/); [Blackwell Architecture vGPU Types](https://docs.nvidia.com/ai-enterprise/release-7/)

**Verdict: SKIP MPS/MIG. Current batch endpoint (M≤8) is correct concurrency approach.**

### 7. Kernel Fusion — Already Mostly Done
Current pipeline per layer: `rmsnorm → quantize → GEMV(QKV) → head_norm → RoPE → attention → GEMV(Wo) → residual add → rmsnorm → quantize → GEMV(gate+up) → SwiGLU → GEMV(down) → residual add`. This is already tight.

**Fusions that exist**: `fused_swiglu_quant` (SwiGLU + quant), `rmsnorm + quant` (separate kernels but sequential). `fused_rmsnorm_quant_int8` combines RMSNorm + quant.

**Fusions worth considering**:
- **quantize + GEMV**: Fuse absmax computation with GEMV read. Current: quantize (1.3%, 1.3 μs) then GEMV (54.8 μs). Could absorb quantize latency by overlapping with GEMV weight load. ~1% gain.
- **head_norm + RoPE**: Already attempted. Fused version gave 141 vs 140 t/s (+0.7%). Negligible.
- **residual add + next RMSNorm**: Both element-wise. Could fuse residual add (0.3 μs estimated) into next layer's RMSNorm. ~0.5% gain.

**Verdict: LOW PRIORITY. Already 97.6% of time in fusions. Remaining 2.4% split across tiny kernels.**

### 8. Prefill Attention — Highest Impact Untapped Optimization
Current: decode-only server. prefill_decode_benchmark shows prefill+decode pipeline at 5.2 ms vs 42-66 ms decode-only (8-13× faster prompt processing). Blocked by cache layout incompatibility: decode cache `[NL][ms][nkv][hd]` can't serve batched prefill attention. [Source: POD-Attention — ASPLOS 2025](https://arxiv.org/abs/2410.18038); [POD-Attention paper PDF](https://akkamath.github.io/files/ASPLOS25_POD.pdf)

**Required changes**:
1. Separate prefill KV cache: `[max_seq][NL][nkv][hd]` layout
2. `attention_prefill_v2` kernel (batched softmax attention)
3. Copy prefill KV into decode KV cache slots after completion
4. Server path: prefill→copy→decode loop

**POD-Attention overlap**: Prefill KV compute can overlap with decode tokens for prior sequence. More complex but gives higher utilization.

**Verdict: IMPLEMENT. Prefill+decode pipeline is the single largest remaining optimization.** Estimated gain: prompt processing 3-5× faster for multi-turn conversations.

## Implementation Recommendations (Priority Order)

| Priority | Optimization | Est. Gain | Effort | Rationale |
|----------|-------------|-----------|--------|-----------|
| **P0** | TMA support investigation | 0-15% | 2-3 days | If available, weight streaming overlaps compute. If not, skip. Check `sm_120a` PTX for TMA instructions. |
| ~~**P0**~~ | ~~Prefill KV cache + attention~~ | ~~2-5× prompt speed~~ | ✅ DONE (2026-06-13) | Multi-chunk prefill working. Fixed pinned buffer race in seq_pos H2D copy. |
| **P1** | Speculative decoding | 1.5-2.5× | 2-3 weeks | High ceiling but requires full pipeline. Draft model fits in 16 GB. |
| **P2** | WGMMA batched GEMV | 10-20% at M≥4 | 1-2 weeks | Only helps batch workloads. M=4+ already fast (148 t/s). |
| **P3** | quantize→GEMV fusion | ~1% | 1 day | Absorb 1.3 μs quantize into GEMV load. Clean code win. |
| **SKIP** | CUDA Graph | ~0% | 0 | 2.1% already demonstrated. Ceiling understood. |
| **SKIP** | MLA/Flash Attention | ~0% | 0 | Attention is 0.9% of runtime for decode. |
| **SKIP** | PagedAttention | ~0% | 0 | No memory fragmentation at M=1. |
| **SKIP** | MPS/MIG | 0% | 0 | MIG not on RTX. MPS adds bugs. Batch endpoint is correct. |
| **SKIP** | Further kernel fusion | <1% | 0 | Remaining kernels are too small to matter. |

## Sources
- **Kept**: [vLLM CUDA Graph Design](https://docs.vllm.ai/en/stable/design/cuda_graphs/) — Reference for CUDA Graph throughput limits
- **Kept**: [Efficiently Serving LLMs Part 4 — CUDA Graphs](https://www.linkedin.com/pulse/efficiently-serving-llms-part-4-how-cuda-graphs-make-vllm) — Confirms graph wins diminish when one kernel dominates
- **Kept**: [NVIDIA Blackwell Tuning Guide 13](https://docs.nvidia.com/cuda/blackwell-tuning-guide/) — Official Blackwell optimization reference
- **Kept**: [Microbenchmarking Blackwell — arXiv 2512.02189](https://arxiv.org/abs/2512.02189) — Blackwell architectural characterization
- **Kept**: [Blackwell TMEM/TMA blog — Jianyu Huang](https://jianyuh.github.io/cuda/2026/04/12/blackwell-tensor-core) — Deep dive into tensor memory architecture
- **Kept**: [POD-Attention — ASPLOS 2025](https://arxiv.org/abs/2410.18038) — Prefill-decode overlap technique
- **Kept**: [Speculative Decoding Survey — arXiv 2402.01528](https://arxiv.org/abs/2402.01528) — Draft model speedup benchmarks
- **Kept**: [Semianalysis — Blackwell Tensor Cores](https://newsletter.semianalysis.com/p/dissecting-nvidia-blackwell-tensor) — WGMMA and MKI instruction analysis
- **Kept**: [NVIDIA MPS Docs](https://docs.nvidia.com/deploy/mps/latest/index.html) — Multi-Process Service limitations
- **Kept**: [CUTLASS Blackwell Changelog](https://docs.nvidia.com/cutlass/latest/CHANGELOG.html) — SM_120 kernel support status
- **Kept**: [llama.cpp CUDA Graph Issue #6763](https://github.com/ggml-org/llama.cpp/issues/6763) — Original CUDA Graph integration discussion
- **Dropped**: Hardware Corner llama.cpp Blackwell blog — Secondary, no technical depth
- **Dropped**: NVIDIA Blackwell Breaks 1000 TPS blog — Datacenter-focused, not relevant to consumer GPU decode

## Gaps
- **GB206 TMA support**: Unknown if RTX 5060 Ti consumer chip includes Tensor Memory Accelerator. Need to verify by inspecting PTX ISA for `cp.async.bulk.tensor` or `tma` instructions on `sm_120a`. If absent, TMA-based optimizations are impossible.
- **Speculative decoding acceptance rates**: Exact acceptance rate for 1.7B→8B on this specific model pair (Qwen3) is unknown. General literature reports 60-80%, but per-domain variance is high.
- **POD-Attention for decode-only server**: Paper focuses on prefill-decode overlap. Our server is decode-only with no prefill at all — implementing basic prefill is the prerequisite.

## Next Steps
1. **Verify TMA support**: Compile `asm("{ cp.async.bulk.tensor ... }")` on `sm_120a` or check CUDA 13 PTX ISA for `sm_120` tensor memory instructions.
2. **Implement prefill cache**: Allocate `[max_seq][NL][nkv][hd]` FP16 cache. Write `attention_prefill_v2` kernel with softmax tiling. Copy prefill KV into decode cache slots.
3. **Profile with prefill+decode**: Measure end-to-end speedup for multi-turn scenarios vs pure decode loop.
4. **If TMA confirmed**: Prototype weight streaming in `gemv_int4_batched` — load weights into shared memory via TMA while previous warp's MMA completes.
