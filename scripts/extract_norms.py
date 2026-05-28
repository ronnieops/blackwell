#!/usr/bin/env python3
"""Extract per-layer RMSNorm weights + embed_tokens from BF16 safetensors.

Writes:
  weights_int8_bf16/{layer}_input_layernorm.f32
  weights_int8_bf16/{layer}_post_attention_layernorm.f32
  weights_int8_bf16/embed_tokens.f32  (FP32, for host-side lookup)

Run: python3 scripts/extract_norms.py
"""
import struct, json, os
import numpy as np

MODEL = "/mnt/data/ai/hf/qwen3-1.7b-base/model.safetensors"
OUT = "weights_int8_bf16"

with open(MODEL, 'rb') as f:
    hdr_len = struct.unpack('Q', f.read(8))[0]
    hdr = json.loads(f.read(hdr_len))

def read_bf16_tensor(name):
    info = hdr[name]
    start, end = info['data_offsets']
    with open(MODEL, 'rb') as f:
        f.seek(8 + hdr_len + start)
        raw = f.read(end - start)
    u16 = np.frombuffer(raw, dtype=np.uint16)
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    return f32.reshape(info['shape'])

for layer in range(28):
    for norm_type in ["input_layernorm", "post_attention_layernorm"]:
        tname = f"model.layers.{layer}.{norm_type}.weight"
        if tname not in hdr:
            print(f"  SKIP {tname} (not found)")
            continue
        w = read_bf16_tensor(tname)
        assert w.shape == (2048,), f"Expected (2048,), got {w.shape}"
        out_path = f"{OUT}/{layer}_{norm_type}.f32"
        w.astype(np.float32).tofile(out_path)
        print(f"  {out_path}: {w.shape}")

print("Done.")
