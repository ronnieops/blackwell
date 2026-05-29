#!/usr/bin/env python3
"""Per-row block-16 INT8 quantization from BF16 safetensors.

Replaces the old 2D block-16 quantization (16×16 tiles share one scale)
with per-row block-16 (each output row has independent scales).

Weight files: weights_int8_bf16/{name}.int8_t and {name}.scale_t
Format: header[K_in, N_out, 16, K_in/16, N_out] + data
  .int8_t: N_out * K_in int8 bytes (pre-transposed [N_out × K_in])
  .scale_t: N_out * (K_in/16) float32 bytes (per-row scales)

Run: python3 scripts/quantize_per_row.py
"""
import struct, json, os
import numpy as np

MODEL = "/mnt/data/ai/hf/qwen3-1.7b-base/model.safetensors"
OUT = "weights_int8_bf16"
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
    """Quantize FP32 weight matrix with per-row block-16 scales.

    W_f32: [N_out, K_in] float32
    Returns: (int8_data [N_out, K_in], scales [N_out, K_in/16])

    GEMV/GEMM kernels use per-row scales: W_t_scale[n_out * num_K_blks + kb].
    """
    N, K = W_f32.shape
    assert K % BLOCK == 0, f"K={K} not divisible by block={BLOCK}"
    num_K_blks = K // BLOCK

    # Per-row block-16 scales: [N, K/16]
    W_blk = W_f32.reshape(N, num_K_blks, BLOCK)
    scales = np.max(np.abs(W_blk), axis=2) / 127.0  # [N, K/16]
    scales = np.maximum(scales, 1e-10)

    # Quantize: broadcast scales to [N, K]
    scale_broadcast = scales[:, :, np.newaxis]  # [N, K/16, 1]
    scale_broadcast = np.repeat(scale_broadcast, BLOCK, axis=2)  # [N, K/16, BLOCK]
    scale_broadcast = scale_broadcast.reshape(N, K)  # [N, K]

    q = np.round(W_f32 / scale_broadcast)
    q = np.clip(q, -127, 127).astype(np.int8)

    return q, scales.astype(np.float32)

def write_weight(prefix, int8_data, scales, K_in, N_out):
    """Write .int8_t and .scale_t files with header.

    Scales are [N_out, K_in/16] — per-row scales for GEMV/GEMM kernels.
    Header: [K_in, N_out, BLOCK, K_in/16, N_out] (reader uses h[3]*h[4] for count)
    """
    num_K_blks = K_in // BLOCK
    header = np.array([K_in, N_out, BLOCK, num_K_blks, N_out], dtype=np.int32)

    # .int8_t: header + N_out*K_in int8 bytes
    path = f"{prefix}.int8_t"
    with open(path, 'wb') as f:
        f.write(header.tobytes())
        f.write(int8_data.tobytes())

    # .scale_t: header + N_out*num_K_blks float32 bytes
    path = f"{prefix}.scale_t"
    with open(path, 'wb') as f:
        f.write(header.tobytes())
        f.write(scales.tobytes())

    mb_data = int8_data.nbytes / (1024*1024)
    mb_sc = scales.nbytes / (1024*1024)
    print(f"  {prefix}: [{N_out}×{K_in}] data={mb_data:.1f}MB scales={mb_sc:.1f}MB")

os.makedirs(OUT, exist_ok=True)

# Per-layer weight matrices
for layer in range(NL):
    for wn in WEIGHT_NAMES:
        tname = f"model.layers.{layer}.{wn}.weight"
        W = read_bf16_tensor(tname)  # [N_out, K_in]
        N_out, K_in = W.shape

        int8_data, scales = quantize_per_row(W)

        # Map weight name to file prefix
        prefix = f"{OUT}/{layer}_{wn}"
        write_weight(prefix, int8_data, scales, K_in, N_out)

# embed_tokens (also used as lm_head since tie_word_embeddings=true)
# Per-block scales [V/16, K/16] — same as weight matrices for GEMV kernel
print("Processing embed_tokens...")
W_emb = read_bf16_tensor("model.embed_tokens.weight")  # [vocab, hidden]
N_out, K_in = W_emb.shape
int8_data, scales = quantize_per_row(W_emb)
write_weight(f"{OUT}/embed_tokens", int8_data, scales, K_in, N_out)

print(f"\nDone. {NL} layers × {len(WEIGHT_NAMES)} weights + embed_tokens")
print("Per-row block-16 scales [N, K/16] for all weights + embeddings.")
