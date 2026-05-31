#!/usr/bin/env python3
"""Per-row block-16 INT8 quantization for Qwen3-0.6B draft model.

Same format as qwen3-1.7b: weights_int8_bf16_06b/{name}.int8_t and {name}.scale_t
Format: header[K_in, N_out, 16, K_in/16, N_out] + data
  .int8_t: N_out * K_in int8 bytes (pre-transposed [N_out × K_in])
  .scale_t: N_out * (K_in/16) float32 bytes (per-row scales)

Run: python3 scripts/quantize_per_row_06b.py
"""
import struct, json, os
import numpy as np

MODEL = "/mnt/data/ai/hf/qwen3-0.6b/model.safetensors"
OUT = "weights_int8_bf16_06b"
BLOCK = 16
NL = 28

# Weight matrix suffixes per layer
WEIGHT_NAMES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]

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

def quantize_per_row(W_f32):
    """Quantize FP32 weight matrix with per-row block-16 scales."""
    N, K = W_f32.shape
    assert K % BLOCK == 0, f"K={K} not divisible by block={BLOCK}"
    num_K_blks = K // BLOCK

    W_blk = W_f32.reshape(N, num_K_blks, BLOCK)
    scales = np.max(np.abs(W_blk), axis=2) / 127.0
    scales = np.maximum(scales, 1e-10)

    scale_broadcast = scales[:, :, np.newaxis]
    scale_broadcast = np.repeat(scale_broadcast, BLOCK, axis=2)
    scale_broadcast = scale_broadcast.reshape(N, K)

    q = np.round(W_f32 / scale_broadcast)
    q = np.clip(q, -127, 127).astype(np.int8)

    return q, scales.astype(np.float32)

def write_weight(prefix, int8_data, scales, K_in, N_out):
    """Write .int8_t and .scale_t files with header."""
    num_K_blks = K_in // BLOCK
    header = np.array([K_in, N_out, BLOCK, num_K_blks, N_out], dtype=np.int32)

    path = f"{prefix}.int8_t"
    with open(path, 'wb') as f:
        f.write(header.tobytes())
        f.write(int8_data.tobytes())

    path = f"{prefix}.scale_t"
    with open(path, 'wb') as f:
        f.write(header.tobytes())
        f.write(scales.tobytes())

    mb_data = int8_data.nbytes / (1024*1024)
    mb_sc = scales.nbytes / (1024*1024)
    print(f"  {prefix}: [{N_out}×{K_in}] data={mb_data:.1f}MB scales={mb_sc:.1f}MB")

os.makedirs(OUT, exist_ok=True)

for layer in range(NL):
    for wn in WEIGHT_NAMES:
        tname = f"model.layers.{layer}.{wn}.weight"
        W = read_bf16_tensor(tname)
        N_out, K_in = W.shape
        int8_data, scales = quantize_per_row(W)
        prefix = f"{OUT}/{layer}_{wn}"
        write_weight(prefix, int8_data, scales, K_in, N_out)

print("Processing embed_tokens...")
W_emb = read_bf16_tensor("model.embed_tokens.weight")
N_out, K_in = W_emb.shape
int8_data, scales = quantize_per_row(W_emb)
write_weight(f"{OUT}/embed_tokens", int8_data, scales, K_in, N_out)

print(f"\nDone. {NL} layers × {len(WEIGHT_NAMES)} weights + embed_tokens")
