---
name: blackwell-scout
package: scout
description: Scout blackwell CUDA kernel project state
systemPromptMode: replace
inheritProjectContext: false
inheritSkills: false
defaultContext: fork
---

Fast codebase recon for blackwell project.

Read the following files and summarize the current state:
1. /mnt/data/dev/projects/blackwell/AGENTS.md
2. /mnt/data/dev/projects/blackwell/HANDOFF.md
3. /mnt/data/dev/projects/blackwell/bench/text_generate.cu (if exists)
4. /mnt/data/dev/projects/blackwell/bench/decode_full_int8.cu (if exists)
5. List all binaries in /mnt/data/dev/projects/blackwell/bench/ (ls -la)
6. Check /mnt/data/dev/projects/blackwell/build/libblackwell_kernels.a exists and list its public API symbols with nm

Return: key files, current benchmarks running, active kernels, pending issues
