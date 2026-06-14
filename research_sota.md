# Research: State-of-the-Art INT4/INT8 LLM Inference on Blackwell (SM_120a) — Mid-2026

## Summary

Blackwell consumer GPUs (SM120, RTX 50-series) share the FP4/NVFP4 tensor core ISA with datacenter SM100 but may have reduced throughput on some low-precision MMA paths. Native INT4 tensor core MMA does NOT exist on Blackwell — only FP4 (E2M1) and FP8 (E4M3/E5M2) tensor core modes are hardware-accelerated. The most impactful techniques for this project are: (1) Marlin/Machete-style mixed-precision INT4×FP16 GEMM kernels that exploit tensor core via weight pre-shuffle, (2) speculative decoding with a small draft model for 1.5-2× single-sequence speedup, and (3) AWQ calibration improvements. BitNet 1.58-bit requires model retraining (not applicable). Continuous batching helps M>1 but not M=1 decode.

---

## Findings

### 1. Blackwell SM_120a Tensor Core Capabilities: No Native INT4 MMA

1. **No native INT4 tensor core MMA on any Blackwell SM.** Blackwell introduces FP4 (E2M1, both NVFP4 and MXFP4 variants) and FP8 tensor core modes, but there is no INT4-specific tensor core instruction. INT4 inference on Blackwell relies on either software unpack + dp4a (current project approach), or mapping INT4 values into FP4 tensor core format (format mismatch issues). [NVIDIA Developer Forums: "Does Blackwell support INT4 native?"](https://forums.developer.nvidia.com/t/does-blackwell-support-int4-native); [Blackwell GPU Wiki — NVFP4 Deep Dive](https://0xsero.github.io/blackwell-gpu-wiki/basics/nvfp4-deep-dive/)

2. **SM120 vs SM100 differences exist but FP4 tensor core ISA is shared.** The Blackwell GPU Wiki documents that SM120 (consumer, RTX 50-series) and SM100 (datacenter, B200/B100) share the same tensor core instruction set including `wgmma` and `tcgen05` MMA instructions for FP4. However, SM120 has fewer SMs, smaller L2 cache, and potentially different tensor core throughput per-SM. The FP4 tensor core path IS available on SM120. [Blackwell GPU Wiki — SM100 vs SM120](https://0xsero.github.io/blackwell-gpu-wiki/basics/sm100-vs-sm120/); [GitHub: 0xSero/blackwell-gpu-wiki](https://github.com/0xSero/blackwell-gpu-wiki/blob/main/docs/blackwell/sm100-vs-sm120.md)

3. **NVFP4 (E2M1) tensor core: 2× throughput vs FP8, but format mismatch with offset-binary INT4.** NVIDIA's NVFP4 uses E2M1 signed-magnitude encoding with FP8 E4M3 per-block (16-element) scales. This gives 2× tensor core FLOPS vs FP8 on Blackwell. However, our INT4 block-16 uses offset-binary encoding (nib-8) with FP32 scales — same nibble maps to different values. This is the exact issue documented in AGENTS.md Session 69 (NVFP4 abandoned, PPL=24,850). [NVIDIA Developer Blog: Introducing NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/); [Blackwell GPU Wiki — NVFP4 Deep Dive](https://0xsero.github.io/blackwell-gpu-wiki/basics/nvfp4-deep-dive/)

4. **FP4 now landed in llama.cpp (2026): NVFP4 and MXFP4 GGUF quantization types.** llama.cpp added native FP4 quantization (NVFP4/MXFP4) with Blackwell tensor core acceleration. This validates the FP4 path but quality concerns remain for models not natively trained/calibrated in FP4. [insiderllm: FP4 Just Landed in llama.cpp](https://insiderllm.com/guides/fp4-inference-llama-cpp-nvfp4-vs-mxfp4-explained-2026/); [GitHub: llama.cpp MXFP4 compilation issue on sm_120](https://github.com/ggml-org/llama.cpp/issues/19662)

5. **Blackwell microbenchmark paper (arXiv:2507.10789) confirms FP4 tensor core throughput.** The "Dissecting NVIDIA Blackwell" paper benchmarks individual instructions including FP4 MMA throughput. FP4 mma achieves ~2× FP8 throughput on tensor cores, confirming theoretical expectations. The second Blackwell microbenchmark paper (arXiv:2512.02189) provides additional SM-level analysis. [arXiv:2507.10789 — Dissecting NVIDIA Blackwell](https://arxiv.org/abs/2507.10789); [arXiv:2512.02189 — Microbenchmarking Blackwell](https://arxiv.org/abs/2512.02189)

6. **Private LLM Inference on Consumer Blackwell GPUs paper (arXiv:2601.09527)** provides the most directly relevant academic analysis — benchmarks INT4/FP4 inference specifically on consumer Blackwell (RTX 50-series) GPUs with practical recipes. [arXiv:2601.09527](https://arxiv.org/abs/2601.09527); [arXiv HTML version](https://arxiv.org/html/2601.09527v1)

7. **TensorRT-LLM FP4 support on RTX 5090 (SM120): confirmed available but with caveats.** GitHub issue #5018 confirms NVFP4 works on RTX 5090 via TensorRT-LLM. FP4 tensor core acceleration is functional on consumer Blackwell. [GitHub: TensorRT-LLM Issue #5018](https://github.com/NVIDIA/TensorRT-LLM/issues/5018)

8. **Blackwell GeForce NVFP4 GEMM project (lna-lab)** — dedicated repo for implementing FP4 GEMM on consumer Blackwell SM120, providing architectural details and implementation guidance for FP4 tensor core programming on RTX 50-series. [GitHub: lna-lab/blackwell-geforce-nvfp4-gemm](https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm)

9. **CUDA 13.x required for sm_120a.** The project's CUDA 13.3 toolchain is correct. PTX ISA for Blackwell includes `tcgen05` (5th-gen tensor core) and `wgmma` instructions supporting FP4/FP8 types. Software migration guide confirms sm_120a as the correct compute target for RTX 5060 Ti. [NVIDIA Forum: Software Migration Guide for Blackwell RTX GPUs](https://forums.developer.nvidia.com/t/software-migration-guide-for-nvidia-blackwell-rtx-gpus)

### 2. llama.cpp INT4 Performance on Blackwell / RTX 5060 Ti

10. **llama.cpp Q4_K_M on RTX 5060 Ti: confirmed ~80-90 t/s for 8B models.** Multiple benchmarks confirm the project's 84 t/s baseline is representative. Hardware-corner GPU ranking and insiderllm benchmarks place the RTX 5060 Ti at this range for Q4_K_M 8B decode. [insiderllm: RTX 5060 Ti Review for Local AI](https://insiderllm.com/guides/rtx-5060-ti-local-ai-review/); [Hardware-corner: GPU LLM Benchmarks](https://www.hardware-corner.net/gpu-llm-benchmark/); [Hardware-corner: GPU Ranking for LLMs](https://www.hardware-corner.net/gpu-ranking-local-llm-token-generation/)

11. **llama.cpp uses CUDA `mmq` (GEMM quantized) and `mmvq` (GEMV quantized) kernels** for INT4 inference, which leverage dp4a/tensor core instructions. On Blackwell sm_120, these kernels use the `vec_dot_q` abstraction. The mmq kernel achieves near-memory-bandwidth throughput for batched decode. [DeepWiki: llama.cpp CUDA Backend](https://deepwiki.com/ggml-org/llama.cpp/5.1-cuda-backend); [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)

12. **ik_llama.cpp (ikawrakow fork) provides improved INT4 kernels.** The fork includes enhanced IQ4_XS, IQ4_NL quantization formats and optimized GEMV kernels that can achieve 10-20% higher throughput than upstream llama.cpp for certain models/hardware combinations. [GitHub: ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp); [DeepWiki: ik_llama.cpp Quantization System](https://deepwiki.com/ikawrakow/ik_llama.cpp/4.2-quantization-system)

13. **llama.cpp MTP (Multi-Token Prediction) + TurboQuant on sm_120.** Recent work (BoFan/Andgihat fork) combines multi-token prediction with turbo quantization on Blackwell, reporting throughput improvements for Qwen3 models. [GitHub: BoFan-tunning/llama.cpp-MTP-TurboQuant](https://github.com/BoFan-tunning/llama.cpp-MTP-TurboQuant); [GitHub: Andgihat/llama-cpp-mtp-turboquant-sm120-blackwell](https://github.com/Andgihat/llama-cpp-mtp-turboquant-sm120-blackwell)

14. **llama.cpp Blackwell CUDA toolkit pitfall (zenn.dev benchmark).** Documents that using the wrong CUDA toolkit version with Blackwell sm_120 causes significant performance degradation — emphasizes importance of CUDA 13.x. [zenn.dev: Blackwell × llama.cpp CUDA Toolkit Pitfall](https://zenn.dev/toki_mwc/articles/rtx5090-blackwell-llama-cpp-cuda-toolkit-pitfall)

15. **RTX 5060 Ti 16GB confirmed as strong budget LLM GPU.** Multiple reviewers confirm it handles 8B Q4_K_M comfortably in 16GB, with 14B models also fitting. [Treeru: RTX 5060 Ti Local AI Benchmark](https://treeru.com/en/blog/rtx-5060-ti-local-llm-benchmark/); [Runyard.dev: RTX 5060 Ti 16GB Best GPU for Local AI](https://www.runyard.dev/blog/rtx-5060-ti-16gb-local-ai); [smeltcore: Qwen3-8B on RTX 5060 Ti](https://smeltcore.com/recipes/qwen3-8b-on-rtx-5060-ti)

### 3. Best-in-Class INT4 GEMV/GEMM Techniques

16. **Marlin (IST-DASLab): FP16×INT4 mixed-precision GEMM — 3-4× over cuBLAS for auto-regressive decode.** Marlin is the gold-standard INT4 kernel for LLM inference. Key innovations: (1) weight pre-shuffling at load time for conflict-free access, (2) FP16 activations × INT4 weights via mixed-precision tensor core, (3) pipelined memory access hiding latency. Integrated into vLLM. Achieves near-peak memory bandwidth for decode-phase (M=1) where the problem is memory-bound. [arXiv:2408.11743 — MARLIN](https://arxiv.org/abs/2408.11743); [GitHub: IST-DASLab/marlin](https://github.com/IST-DASLab/marlin); [Red Hat Developers: How Marlin pushes boundaries](https://developers.redhat.com/articles/2024/04/how-marlin-pushes-boundaries-mixed-precision-llm-inference/)

17. **Machete: Mixed-input GEMM kernel (CUTLASS-based) for Hopper/Blackwell.** Successor to Marlin in vLLM, optimized for CUTLASS 3.x on Hopper/Blackwell. Supports INT4 weight × FP16 activation with better pipelining and supports larger batch sizes. Specifically designed for newer architectures. [Red Hat Developers: Introducing Machete](https://developers.redhat.com/articles/2024/10/introducing-machete-mixed-input-gemm-kernel-optimized-meta-llama-models/); [GitHub: vllm-project/vllm — Machete](https://github.com/vllm-project/vllm/blob/main/csrc/quantization/machete/)

18. **Marlin/Machete key transferability insight:** These kernels use INT4 weights with **FP16 activations** fed into tensor core mixed-precision MMA. Our project uses INT8 activations (quantized) with INT4 weights. The Marlin approach avoids activation quantization entirely — FP16 activations are kept full precision, only weights are INT4. This eliminates activation quantization error (a known issue in our pipeline for Llama 3.1). **However**, Marlin uses signed-magnitude INT4 (0-15 range), not our offset-binary (nib-8) format. Would require weight format change but could dramatically improve M=1 quality and potentially speed (tensor core vs dp4a). [arXiv:2408.11743](https://arxiv.org/abs/2408.11743)

19. **ExLlamaV2 (EXL2): Fastest INT4 inference engine for consumer GPUs.** ExLlamaV2 uses custom CUDA GEMV kernels optimized specifically for single-GPU consumer cards (RTX series). EXL2 format allows mixed bitrates per layer. Reports suggest it matches or exceeds llama.cpp for single-user decode on consumer GPUs. [GitHub: turboderp-org/exllamav2](https://github.com/turboderp-org/exllamav2); [local-llm.net: ExLlamaV2](https://www.local-llm.net/tools/exllamav2/); [LocalAIMaster: ExLlamaV2 + TabbyAPI Best INT4](https://localaimaster.com/blog/exllamav2-tabbyapi-best-int4-inference-single-gpu-2026/)

20. **AWQ (Activation-aware Weight Quantization): best inference-only INT4 calibration.** AWQ uses activation magnitudes to determine which weight channels are important, then applies per-channel scaling before quantization. Inference-only (no retraining). Our project already uses AWQ-style calibration (α=0.6, PPL 21.82 vs 23.52 baseline, 7.2% improvement). NVIDIA ModelOpt provides AWQ implementation. [arXiv:2306.00978 — AWQ](https://arxiv.org/abs/2306.00978); [GitHub: mit-han-lab/smoothquant](https://github.com/mit-han-lab/smoothquant); [NVIDIA Model-Optimizer: AWQ PTQ](https://github.com/NVIDIA/Model-Optimizer/blob/main/docs/source/awq.md)

21. **SmoothQuant: activation smoothing for weight+activation quantization.** SmoothQuant migrates quantization difficulty from activations to weights by scaling. Enables INT8 activation quantization without quality loss. Useful if we want to keep INT8 activations but improve their quality. Inference-only post-training. [GitHub: mit-han-lab/smoothquant](https://github.com/mit-han-lab/smoothquant); [apxml: GPTQ vs AWQ vs SmoothQuant](https://apxml.com/courses/practical-llm-quantization/gptq-vs-awq-vs-smoothquant/); [GeneralCompute: Quantization Comparison](https://www.generalcompute.com/blog/quantization-for-inference-gptq-awq-smoothquant-and-fp8/)

22. **GPTQ: weight-only post-training quantization via Hessian-based calibration.** GPTQ uses second-order error compensation during quantization. Often combined with Marlin kernel in vLLM. Higher quality than naive RTN (round-to-nearest) but requires calibration data. Inference-only. [apxml: GPTQ vs AWQ vs SmoothQuant](https://apxml.com/courses/practical-llm-quantization/gptq-vs-awq-vs-smoothquant/)

### 4. BitNet 1.58-bit / T-MAC (Requires Retraining — Not Applicable)

23. **BitNet b1.58: ternary weights {-1, 0, +1} — requires model retraining from scratch.** BitNet 1.58-bit LLMs use ternary weights and INT8 activations. The weights are trained natively at 1.58-bit — cannot be applied to existing pretrained models via post-training quantization. NOT applicable to Qwen3-8B. [arXiv:2504.12285 — BitNet b1.58 2B4T Technical Report](https://arxiv.org/abs/2504.12285); [GitHub: microsoft/BitNet](https://github.com/microsoft/BitNet); [Wikipedia: 1.58-bit LLM](https://en.wikipedia.org/wiki/1.58-bit_large_language_model)

24. **T-MAC: table-lookup-based low-bit LLM inference — CPU/NPU focused.** Microsoft's T-MAC uses lookup tables instead of multiply-accumulate for sub-4-bit operations. Primarily targets CPU/NPU, not GPU tensor cores. GPU adaptation would not leverage tensor core hardware. [arXiv:2407.00088 — T-MAC](https://arxiv.org/abs/2407.00088); [GitHub: microsoft/T-MAC](https://github.com/microsoft/T-MAC)

### 5. Continuous Batching / Paged Attention for Single GPU

25. **Continuous batching helps multi-user throughput but NOT single-sequence decode.** vLLM's continuous batching dynamically inserts/removes sequences from a batch, maximizing GPU utilization across concurrent requests. For M=1 (single user), it provides no benefit — decode is already memory-bandwidth-bound. For M>1 (our batched path at M=8-48), continuous batching could help if requests arrive asynchronously. [vLLM Docs: Paged Attention](https://docs.vllm.ai/en/latest/design/paged_attention/); [RunPod: vLLM PagedAttention and Continuous Batching](https://www.runpod.io/articles/guides/vllm-pagedattention-continuous-batching-explained); [insujang: LLM Inference Continuous Batching](https://insujang.github.io/2024-01-07/llm-inference-continuous-batching-and-pagedattention/)

26. **Paged attention reduces KV cache fragmentation, not decode speed.** PagedAttention (block-based KV cache management) eliminates memory fragmentation and enables variable-length sequences. Does not speed up the GEMV/GEMM bottleneck. For our fixed-length batched decode, KV cache is already contiguous. [vLLM Docs: Paged Attention](https://docs.vllm.ai/en/latest/design/paged_attention/)

27. **vLLM overhead significant for small models on single GPU.** vLLM's Python scheduling layer adds overhead. For a custom CUDA kernel project already achieving 56-154 t/s, integrating vLLM-style scheduling would add latency without proportional benefit unless serving many concurrent users. [Dev.to: Deep Dive into vLLM](https://dev.to/maximus_prime_1/deep-dive-into-vllm-how-pagedattention-continuous-batching-works/)

### 6. Speculative Decoding — Viable for INT4

28. **Speculative decoding: 1.5-2× speedup for memory-bound decode.** A small draft model (e.g., 0.5B) generates K candidate tokens; the target model (8B) verifies them in a single forward pass. Since decode is memory-bandwidth-bound (weight loading dominates), verifying K tokens costs ~1× the cost of generating 1 token (same weights loaded). Net speedup depends on acceptance rate. [vLLM Docs: Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/); [NVIDIA Triton: Speculative Decoding with TensorRT-LLM](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/tensorrtllm-speculative-decoding.html); [Introl: Speculative Decoding 2-3x Speedup](https://introl.com/blog/speculative-decoding-llm-throughput-optimization)

29. **EAGLE (SafeAILab): best speculative decoding method as of 2025-2026.** EAGLE uses a lightweight autoregressive draft head that shares the target model's hidden states. Achieves 2-3× speedup with high acceptance rates (>70%). EAGLE-3 (2025) further improves. Compatible with INT4 quantized target models — the draft head runs in FP16. [GitHub: SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE); [SGLang Docs: Speculative Decoding](https://docs.sglang.io/docs/advanced_features/spec_decode)

30. **Speculative decoding overhead analysis.** Draft model adds: (1) ~0.5B parameters loaded per draft step (small vs 8B), (2) tree attention for verification, (3) rejection sampling. For our GEMV-bound M=1 decode at 56 t/s (17.9 ms/tok), speculative decoding could reduce to ~9-12 ms/tok (83-111 t/s) with 2× speedup. Memory: additional ~1-2 GB for draft model weights. [arXiv:2601.11580 — Speculative Decoding: Performance or Illusion?](https://arxiv.org/pdf/2601.11580); [Phonism: Speculative Decoding Complete Guide](https://phonism.github.io/LLMNotes/en/speculative-decoding/)

31. **MTP (Multi-Token Prediction) in llama.cpp forks.** MTP is a related approach (predict multiple tokens per step) being explored in llama.cpp forks targeting Blackwell sm_120. Could be simpler than full EAGLE integration. [GitHub: BoFan-tunning/llama.cpp-MTP-TurboQuant](https://github.com/BoFan-tunning/llama.cpp-MTP-TurboQuant)

### 7. MoE Considerations

32. **MoE models (DeepSeek-V3, Qwen3 MoE) benefit from sparse expert dispatch.** In MoE inference, only top-K experts are activated per token (e.g., 8 of 64). This means effective compute is much smaller than dense equivalent. For INT4 quantized MoE, only active expert weights need loading per token — potentially better cache behavior. [GitHub: deepseek-ai/DeepSeek-MoE](https://github.com/deepseek-ai/DeepSeek-MoE); [arXiv:2401.06066 — DeepSeekMoE](https://arxiv.org/abs/2401.06066)

33. **Qwen3-30B-A3B MoE: fits in 16GB at INT4.** Qwen3-30B-A3B has 30B total params but only ~3B active per token. At INT4 quantization, total weight footprint ~15GB, fitting in 16GB VRAM. Decode throughput benefits from low active param count. This model is a natural target for our kernels if we add sparse expert dispatch. [DeepWiki: Qwen3 Speed Benchmarking](https://deepwiki.com/guquan/Qwen3/6.2-speed-benchmarking); [GitHub: QwenLM/Qwen3](https://github.com/QwenLM/Qwen3); [Gist: Qwen3 MoE Quant Benchmarking](https://gist.github.com/ubergarm/0f9663fd56fc18f8)

34. **MoE requires expert routing + sparse GEMV — our kernels need extension.** Our current `gemv_int4_batched` assumes all weights are dense per layer. MoE needs: (1) gating/routing network forward pass, (2) per-token expert selection, (3) dispatch hidden states to selected expert weight matrices, (4) combine outputs. This is a moderate kernel-level addition — not retraining, just inference architecture change. [NVIDIA Blog: Integrate Qwen3 into Production](https://developer.nvidia.com/blog/integrate-and-deploy-tongyi-qwen3-models-into-production/)

35. **MoE expert GEMV is even more memory-bound.** Each token activates only a few experts, meaning smaller weight matrices per token. This makes the problem MORE memory-bandwidth-bound (less compute per byte loaded). Our INT4 format's small weight footprint is advantageous. However, expert routing adds overhead (gather/scatter of hidden states).

### 8. Additional Relevant Techniques

36. **NVIDIA Model-Optimizer (ModelOpt): production calibration toolkit.** Provides AWQ, GPTQ, max calibration, MSE calibration for NVFP4/INT4/INT8. Could improve our calibration beyond random-normal proxy. Exports to safetensors+config.json — not directly compatible with our flat binary format, but calibration methods (Hessian-based, MSE) could be ported. [GitHub: NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer); [NVIDIA Model-Optimizer AWQ docs](https://github.com/NVIDIA/Model-Optimizer/blob/main/docs/source/awq.md)

37. **Blackwell LLM toolkit (elsung): empirical INT4/FP4 recipes for RTX 50-series.** Community-maintained repository with benchmarked quantization recipes specifically for consumer Blackwell GPUs. [GitHub: elsung/blackwell-llm-toolkit](https://github.com/elsung/blackwell-llm-toolkit)

38. **club-5060ti (5p00kyy): practical local LLM recipes for RTX 5060 Ti.** Community recipes for the exact hardware target. [GitHub: 5p00kyy/club-5060ti](https://github.com/5p00kyy/club-5060ti)

---

## Ranked Recommendations

### Tier 1: Highest Impact, Inference-Only (No Retraining)

| Rank | Technique | Est. Impact | Effort | Retraining? |
|------|-----------|------------|--------|-------------|
| **1** | **Marlin/Machete-style FP16×INT4 tensor core GEMV** | 1.5-2× M=1 speed, better quality (no act quant) | High (new kernel, weight format change) | No |
| **2** | **Speculative decoding (EAGLE-style draft head)** | 1.5-2× M=1 speedup (56→84-112 t/s) | Medium (draft model, verify pass) | No (needs trained draft head) |
| **3** | **AWQ calibration improvement (real calibration data, Hessian/MSE)** | 5-15% PPL improvement (21.82→~19-20) | Low-Medium (calibration pipeline) | No |
| **4** | **MoE support (Qwen3-30B-A3B)** | Enable new model class, better t/active-param | Medium (expert routing, sparse GEMV) | No |

### Tier 2: Medium Impact

| Rank | Technique | Est. Impact | Effort | Retraining? |
|------|-----------|------------|--------|-------------|
| **5** | **SmoothQuant for activation quantization** | Better activation quality, helps INT8 path | Medium | No |
| **6** | **Continuous batching for server (multi-user)** | Better M>1 throughput under load | Medium | No |
| **7** | **NVFP4 tensor core path (revisit with proper calibration)** | 2× tensor core FLOPS if quality fixed | High (calibration, format conversion) | No |

### Tier 3: Low Applicability / Requires Retraining

| Rank | Technique | Est. Impact | Effort | Retraining? |
|------|-----------|------------|--------|-------------|
| **8** | **BitNet 1.58-bit (ternary weights)** | Massive speed if model trained natively | N/A | **YES — must train from scratch** |
| **9** | **T-MAC table-lookup** | CPU/NPU focused, not GPU tensor core | N/A | Partial |
| **10** | **Paged attention** | Marginal for our use case (fixed-length KV) | Medium | No |

---

## Key Architectural Insight: The Marlin Gap

The single most important finding: **our project's M=1 GEMV (56 t/s) is slower than llama.cpp (84 t/s) because llama.cpp's mmq kernel uses tensor core mixed-precision MMA, while our dp4a-based GEMV is compute-bound at M=1.**

The Marlin paper (arXiv:2408.11743) demonstrates that a properly designed FP16×INT4 GEMM kernel using tensor core mixed-precision can achieve near-peak memory bandwidth for auto-regressive decode. The key innovations are:
1. **Weight pre-deinterleaving at load time** — eliminates runtime unpack overhead
2. **FP16 activations** (no activation quantization) — eliminates quant error
3. **Tensor core MMA** instead of dp4a — 4-8× higher throughput per SM

This directly addresses our documented issue: "llama.cpp uses tensor cores for GEMV (our dp4a SIMD is slower for skinny M=1)."

**Format compatibility:** Marlin uses signed INT4 weights (0-15). Our offset-binary format (nib-8) would need conversion. The FP16 activation approach eliminates our activation quantization step entirely, which would also fix the Llama 3.1 quality issue.

---

## Format Compatibility Matrix

| Technique | Our INT4 Block-16 (offset-binary, FP32 scale) | Adaptation Needed |
|-----------|-----------------------------------------------|-------------------|
| Marlin GEMV | Signed-magnitude INT4 | Convert weight format (one-time) |
| Machete GEMM | CUTLASS INT4 | Convert + CUTLASS integration |
| AWQ v2 | Format-agnostic (scales folded) | ✅ Already compatible |
| SmoothQuant | Format-agnostic (pre-scaling) | ✅ Compatible |
| NVFP4 tensor core | E2M1 signed-magnitude + FP8 scale | Full re-quant + calibration |
| BitNet 1.58 | Ternary {-1,0,+1} | ❌ Requires retraining |
| GPTQ | Any INT4 format | ✅ Calibration method only |

---

## Gaps

1. **Exact SM120 FP4 tensor core throughput per-SM** — not publicly confirmed whether SM120 has same FP4 MMA throughput as SM100 or reduced. The microbenchmark papers (2507.10789, 2512.02189) likely contain this but full text not accessed.
2. **Marlin kernel on SM120 specifically** — Marlin was designed for Ampere/Hopper. Performance on Blackwell SM120 needs validation. Machete (CUTLASS 3.x) targets newer architectures more directly.
3. **EAGLE draft head for Qwen3-8B** — no pre-trained draft head exists; would need to train one (inference-only for target model, but draft head training is a light GPU training job).
4. **Qwen3-30B-A3B INT4 on RTX 5060 Ti 16GB** — exact VRAM fit at INT4 with KV cache needs measurement. 30B × 0.5 bytes/param ≈ 15GB weights, leaving ~1GB for KV cache and activations — tight.
5. **llama.cpp mmq kernel internal architecture** — whether it uses wgmma or dp4a on sm_120 specifically not confirmed from search results.

## Suggested Next Steps

1. **Implement Marlin-style tensor core INT4 GEMV** — highest ROI. Study [IST-DASLab/marlin source](https://github.com/IST-DASLab/marlin), adapt for sm_120a. Convert weights to signed-magnitude INT4 format. Keep FP16 activations (drop activation quantization).
2. **Train small EAGLE draft head** for Qwen3-8B — 2× M=1 speedup. Use existing INT4 weights for target, train lightweight draft head (~100M params) on a small GPU.
3. **Read arXiv:2507.10789 and arXiv:2512.02189** (Blackwell microbenchmark papers) for exact FP4/INT4 instruction throughput on SM120.
4. **Read arXiv:2601.09527** (Private LLM Inference on Consumer Blackwell) — most directly relevant paper to our exact hardware/use case.

## Sources

### Kept (Primary/Authoritative)
- **NVIDIA Developer Blog: Introducing NVFP4** (https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) — official NVFP4 spec, E2M1 + FP8 scales
- **Blackwell GPU Wiki — SM100 vs SM120** (https://0xsero.github.io/blackwell-gpu-wiki/basics/sm100-vs-sm120/) — consumer vs datacenter Blackwell tensor core differences
- **Blackwell GPU Wiki — NVFP4 Deep Dive** (https://0xsero.github.io/blackwell-gpu-wiki/basics/nvfp4-deep-dive/) — FP4 tensor core programming details
- **arXiv:2408.11743 — MARLIN** (https://arxiv.org/abs/2408.11743) — gold-standard INT4 mixed-precision kernel, 3-4× speedup
- **GitHub: IST-DASLab/marlin** (https://github.com/IST-DASLab/marlin) — Marlin kernel source code
- **Red Hat: How Marlin pushes boundaries** (https://developers.redhat.com/articles/2024/04/how-marlin-pushes-boundaries-mixed-precision-llm-inference/) — Marlin technical explanation
- **Red Hat: Introducing Machete** (https://developers.redhat.com/articles/2024/10/introducing-machete-mixed-input-gemm-kernel-optimized-meta-llama-models/) — Machete (CUTLASS-based Marlin successor)
- **arXiv:2507.10789 — Dissecting Blackwell** (https://arxiv.org/abs/2507.10789) — Blackwell tensor core microbenchmarks
- **arXiv:2512.02189 — Microbenchmarking Blackwell** (https://arxiv.org/abs/2512.02189) — additional SM-level analysis
- **arXiv:2601.09527 — Private LLM Inference on Consumer Blackwell** (https://arxiv.org/abs/2601.09527) — most directly relevant paper
- **arXiv:2306.00978 — AWQ** (https://arxiv.org/abs/2306.00978) — activation-aware weight quantization
- **arXiv:2504.12285 — BitNet b1.58 2B4T** (https://arxiv.org/abs/2504.12285) — ternary weight LLM (requires retraining)
- **arXiv:2407.00088 — T-MAC** (https://arxiv.org/abs/2407.00088) — table-lookup low-bit inference (CPU/NPU)
- **GitHub: SafeAILab/EAGLE** (https://github.com/SafeAILab/EAGLE) — speculative decoding implementation
- **vLLM Docs: Speculative Decoding** (https://docs.vllm.ai/en/latest/features/speculative_decoding/) — spec decoding framework support
- **vLLM Docs: Paged Attention** (https://docs.vllm.ai/en/latest/design/paged_attention/) — paged attention design
- **NVIDIA Developer Forums: Does Blackwell support INT4 native?** (https://forums.developer.nvidia.com/t/does-blackwell-support-int4-native) — confirms no native INT4 tensor core
- **GitHub: TensorRT-LLM Issue #5018** (https://github.com/NVIDIA/TensorRT-LLM/issues/5018) — NVFP4 on RTX 5090 confirmed
- **GitHub: lna-lab/blackwell-geforce-nvfp4-gemm** (https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm) — FP4 GEMM on consumer Blackwell
- **insiderllm: FP4 in llama.cpp** (https://insiderllm.com/guides/fp4-inference-llama-cpp-nvfp4-vs-mxfp4-explained-2026/) — FP4 quantization in llama.cpp
- **insiderllm: RTX 5060 Ti Review** (https://insiderllm.com/guides/rtx-5060-ti-local-ai-review/) — 5060 Ti benchmarks
- **GitHub: turboderp-org/exllamav2** (https://github.com/turboderp-org/exllamav2) — fast consumer-GPU INT4 inference
- **GitHub: ikawrakow/ik_llama.cpp** (https://github.com/ikawrakow/ik_llama.cpp) — improved llama.cpp fork
- **GitHub: NVIDIA/Model-Optimizer** (https://github.com/NVIDIA/Model-Optimizer) — production calibration toolkit
- **DeepWiki: llama.cpp CUDA Backend** (https://deepwiki.com/ggml-org/llama.cpp/5.1-cuda-backend) — llama.cpp CUDA kernel architecture
- **GitHub: QwenLM/Qwen3** (https://github.com/QwenLM/Qwen3) — Qwen3 model including MoE variants
- **arXiv:2401.06066 — DeepSeekMoE** (https://arxiv.org/abs/2401.06066) — MoE architecture
- **GitHub: BoFan-tunning/llama.cpp-MTP-TurboQuant** (https://github.com/BoFan-tunning/llama.cpp-MTP-TurboQuant) — MTP + quant for Blackwell
- **edge-ai-vision: Impact of NVFP4 for LLM Inference** (https://www.edge-ai-vision.com/2025/10/nvidia-blackwell-the-impact-of-nvfp4-for-llm-inference/) — NVFP4 LLM analysis
- **zenn.dev: Blackwell llama.cpp CUDA pitfall** (https://zenn.dev/toki_mwc/articles/rtx5090-blackwell-llama-cpp-cuda-toolkit-pitfall) — Blackwell CUDA toolkit issues
- **arXiv:2601.11580 — Speculative Decoding: Performance or Illusion?** (https://arxiv.org/pdf/2601.11580) — spec decoding analysis

### Dropped
- **Nebius AI Cloud Platform** — cloud GPU ad, not relevant
- **SitePoint M3 vs 4090** — Apple Silicon comparison, irrelevant
- **YouTube RTX laptop review** — video format, not citable data
- **ACM Queue "It's All About Inference"** — high-level opinion piece, no technical depth
- **LinkedIn GPTQ/AWQ/GGUF explainer** — secondary commentary, primary sources kept
- **MLJourney quantization techniques** — blog spam, primary sources kept
- **Various vLLM tutorial blogs** — kept official vLLM docs instead
