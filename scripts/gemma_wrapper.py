#!/usr/bin/env python3
"""Gemma 4 12B wrapper: uses HF tokenizer for correct encode/decode,
feeds token IDs to the C++ benchmark.

Usage:
  python3 scripts/gemma_wrapper.py "The capital of France is" [max_new] [bench_path]
"""
import subprocess, sys, json, os

HF_MODEL = "google/gemma-4-12b-it"
BENCH   = sys.argv[3] if len(sys.argv) > 3 else "./bench/text_generate_gemma"
PROMPT  = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
MAX_NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 30
WDIR    = "/tmp/gemma_test"

# Load HF tokenizer
print(f"Loading tokenizer from {HF_MODEL}...", file=sys.stderr)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)

# Encode prompt
ids = tok.encode(PROMPT)
print(f"Prompt tokens ({len(ids)}): {ids}", file=sys.stderr)

# Build token string for --tokens argument
token_str = ",".join(str(id) for id in ids)

# Run C++ benchmark
cmd = [BENCH, PROMPT, str(MAX_NEW), WDIR, "--tokens", token_str]
print(f"Running: {' '.join(cmd)}", file=sys.stderr)

proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

# Parse output token IDs from stderr (benchmark now prints token IDs)
# Actually the benchmark prints token IDs to stdout
output = proc.stdout
print(f"Benchmark stdout:\n{output}", file=sys.stderr)

# Extract token IDs from the "── Generating ──\n" section
# The output format is space-separated token IDs
lines = output.split('\n')
decoding = False
gen_tokens = []
for line in lines:
    if "── Generating ──" in line:
        decoding = True
        continue
    if "── Stats ──" in line:
        break
    if decoding:
        for token_str in line.strip().split():
            try:
                gen_tokens.append(int(token_str))
            except ValueError:
                pass

if gen_tokens:
    text = tok.decode(gen_tokens)
    print(f"\nDecoded ({len(gen_tokens)} tokens): {text}")
    print(f"Token IDs: {gen_tokens[:20]}{'...' if len(gen_tokens) > 20 else ''}")
else:
    print("No tokens generated. Full output:", output, file=sys.stderr)
