---
name: session28-researcher
package: researcher
description: Session 28 initial context research
systemPromptMode: replace
inheritProjectContext: false
inheritSkills: false
defaultContext: fork
---

Research current state of blackwell CUDA LLM inference project.

Check these files:
1. /mnt/data/dev/projects/blackwell/bench/text_generate.cu - main generation benchmark
2. /mnt/data/dev/projects/blackwell/bench/decode_full_int8.cu - INT8 decode benchmark
3. /mnt/data/dev/projects/blackwell/bench/decode_int8_cgraph.cu - CUDA Graph benchmark
4. /mnt/data/dev/projects/blackwell/bench/decode_int8_batched_cgraph_attn.cu - batched attn benchmark
5. /mnt/data/dev/projects/blackwell/bench/validate_pipeline.cu - correctness test

For each benchmark: what's the current throughput, what does it test, what's the status (working/broken)

Also check:
- What kernels are being used for the decode path
- Is CUDA Graph capture working
- What's the current text generation speed vs llama.cpp baseline (274.4 t/s Q4_K_M)
- What are the pending tasks from HANDOFF.md

Return: current benchmark status, throughput numbers, key kernel implementations, pending tasks
