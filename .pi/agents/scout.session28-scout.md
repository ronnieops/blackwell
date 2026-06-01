---
name: session28-scout
package: scout
description: Session 28 initial context scout
systemPromptMode: replace
inheritProjectContext: false
inheritSkills: false
defaultContext: fork
---

Fast codebase recon for blackwell project.

Read these files:
1. /mnt/data/dev/projects/blackwell/AGENTS.md
2. /mnt/data/dev/projects/blackwell/HANDOFF.md
3. /mnt/data/dev/projects/blackwell/bench/text_generate.cu (if exists)
4. /mnt/data/dev/projects/blackwell/bench/decode_full_int8.cu (if exists)
5. List all binaries in /mnt/data/dev/projects/blackwell/bench/ (ls -la)
6. Check /mnt/data/dev/projects/blackwell/build/libblackwell_kernels.a exists and list its public API symbols with nm

Also check:
- src/kernels/*.cu - kernel implementations
- include/blackwell/kernels.h - kernel API signatures
- bench/*.cu - all bench files and their status

Return: key files, current benchmarks running, active kernels, pending issues, symbol count
