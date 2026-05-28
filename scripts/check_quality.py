#!/usr/bin/env python3
"""Quick quality check: compare per-row INT8 GEMV output vs BF16 reference.
Tests one GEMV call and reports cosine similarity."""

import struct, json, numpy as np

MODEL = "/mnt/data/ai/hf/qwen3-1.7b-base/model.safetensors"
WDIR = "weights_int8_bf16"
B = 16

with open(MODEL, 'rb') as f:
    hl = struct.unpack('Q', f.read(8))[0]
    hdr = json.loads(f.read(hl))

def read_bf16(name):
    info = hdr[name]
    s, e = info['data_offsets']
    with open(MODEL, 'rb') as f:
        f.seek(8 + hl + s)
        raw = f.read(e - s)
    u16 = np.frombuffer(raw, dtype=np.uint16)
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    return f32.reshape(info['shape'])

def load_weight(prefix):
    """Load INT8 weight and per-row scales."""
    with open(f"{prefix}.int8_t", 'rb') as f:
        h = struct.unpack('5i', f.read(20))
        data = np.frombuffer(f.read(h[0]*h[1]), dtype=np.int8).reshape(h[1], h[0])
    with open(f"{prefix}.scale_t", 'rb') as f:
        h2 = struct.unpack('5i', f.read(20))
        scales = np.frombuffer(f.read(h2[3]*h2[4]*4), dtype=np.float32).reshape(h2[4], h2[3])
    return data, scales, h[0], h[1]  # W_t [N,K], scales [N, K/16]

def gemv_per_row(x_fp32, W_t_i8, W_sc, K, N):
    """Per-row INT8 GEMV. x is FP32, quantized inline.
    W_t_i8: [N, K] int8, W_sc: [N, K/16] float32."""
    # Quantize x with per-vector block-16 scales
    x_blk = x_fp32.reshape(-1, B)
    x_amax = np.max(np.abs(x_blk), axis=1)
    x_scales = np.maximum(x_amax / 127.0, 1e-9)
    x_i8 = np.clip(np.round(x_blk / x_scales[:, np.newaxis]), -127, 127).astype(np.int8).reshape(K)

    # GEMV: y[n] = sum_kb( dp4a(W_t[n, kb*16:kb*16+16], x[kb*16:kb*16+16]) * w_sc[n,kb] * x_sc[kb] )
    nb_k = K // B
    x_blk = x_i8.reshape(1, nb_k, B).astype(np.int32)
    W_blk = W_t_i8.reshape(N, nb_k, B).astype(np.int32)
    raw_dot = np.sum(W_blk * x_blk, axis=2).astype(np.float32)  # [N, nb_k]

    # Per-row weight scales: W_sc[n, kb]
    sc = W_sc * x_scales[np.newaxis, :]  # [N, nb_k]
    y = np.sum(raw_dot * sc, axis=1)  # [N]
    return y

def cosim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)

# Test with random input
np.random.seed(42)
x = np.random.randn(2048).astype(np.float32) * 0.5

print("Per-layer GEMV quality (per-row INT8 vs BF16):")
print(f"{'Layer':>5} {'Weight':>20} {'K':>6} {'N':>6} {'cosim':>10} {'max_diff':>12}")
print("-" * 65)

for layer in [0, 5, 10, 15, 20, 27]:
    for wn, wname in [("self_attn.q_proj", "self_attn.q_proj"),
                       ("mlp.gate_proj", "mlp.gate_proj"),
                       ("mlp.down_proj", "mlp.down_proj")]:
        # BF16 reference
        W_bf16 = read_bf16(f"model.layers.{layer}.{wname}.weight")  # [N_out, K_in]
        N_out, K_in = W_bf16.shape

        # Use appropriate input size
        x_layer = np.random.randn(K_in).astype(np.float32) * 0.5
        y_ref = W_bf16 @ x_layer  # BF16→FP32 matmul

        # INT8 per-row
        W_t_i8, W_sc, K, N = load_weight(f"{WDIR}/{layer}_{wn}")
        y_int8 = gemv_per_row(x_layer, W_t_i8, W_sc, K, N)

        cs = cosim(y_ref, y_int8)
        md = np.max(np.abs(y_ref - y_int8))
        print(f"{layer:>5} {wn:>20} {K:>6} {N:>6} {cs:>10.6f} {md:>12.6f}")

# Also check: does 2D-block vs per-row differ?
print("\n\n2D-block vs Per-row comparison (layer 0, gate_proj):")
W_bf16 = read_bf16("model.layers.0.mlp.gate_proj.weight")
N_out, K_in = W_bf16.shape
y_ref = W_bf16 @ x

# Per-row (new)
W_t_i8, W_sc, K, N = load_weight(f"{WDIR}/0_mlp.gate_proj")
y_per_row = gemv_per_row(x, W_t_i8, W_sc, K, N)

# Compute per-row scales SNR
W_blk = W_bf16.reshape(N_out, K_in // B, B)
per_row_scales = np.max(np.abs(W_blk), axis=2) / 127.0
dequant_per_row = (W_t_i8.reshape(N_out, K_in // B, B).astype(np.float32) *
                   per_row_scales[:, :, np.newaxis]).reshape(N_out, K_in)
snr_per_row = 10 * np.log10(np.mean(W_bf16**2) / np.mean((W_bf16 - dequant_per_row)**2 + 1e-30))
print(f"  Per-row block-16 SNR: {snr_per_row:.1f} dB, cosim: {cosim(y_ref, y_per_row):.6f}")
