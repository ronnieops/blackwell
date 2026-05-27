#!/usr/bin/env python3
"""verify_int8_accuracy.py — Compare INT8 quant vs BF16 original.

Reads BF16 from safetensors directly (no numpy BF16 support needed).
Compares INT8 dequant (from weights_int8/) vs original weights.

Usage: python3 bench/verify_int8_accuracy.py
"""
import struct, json, sys
import numpy as np

MODEL = "/mnt/data/ai/hf/qwen3-1.7b-base/model.safetensors"
INT8_DIR = "weights_int8_bf16"

def read_safetensor_bf16(path, tensor_name):
    """Read single tensor from safetensors file, convert BF16→FP32."""
    with open(path, 'rb') as f:
        hdr_len = struct.unpack('Q', f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
    info = hdr[tensor_name]
    start, end = info['data_offsets']
    with open(path, 'rb') as f:
        f.seek(8 + hdr_len + start)
        raw = f.read(end - start)
    u16 = np.frombuffer(raw, dtype=np.uint16)
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    return f32.reshape(info['shape'])

def load_int8_weight(prefix):
    with open(f"{prefix}.int8_t", 'rb') as f:
        h = struct.unpack('5i', f.read(20))
        K, N = h[0], h[1]
        data = np.frombuffer(f.read(K*N), dtype=np.int8).reshape(N, K)
    with open(f"{prefix}.scale_t", 'rb') as f:
        h = struct.unpack('5i', f.read(20))
        nkb, nnb = h[3], h[4]
        scales = np.frombuffer(f.read(nkb*nnb*4), dtype=np.float32).reshape(nnb, nkb)
    return data, scales  # N×K, N/16)×(K/16)

def gemv_int8_cpu(x_raw, W_t, S_t, x_scale, K, N):
    """CPU INT8 GEMV matching GPU gemv_int8.
    
    Mathematically: y[n] = Σ_k q_x[k] * x_sc[kb] * q_W[n,k] * W_sc[nb,kb]
    = Σ_k x_raw[k] * W_dq[n,k] where W_dq = q_W * W_sc and x_sc cancels.
    """
    y = np.zeros(N, dtype=np.float32)
    for n in range(N):
        acc = 0.0
        for k in range(K):
            nb = n // 16
            kb = k // 16
            w_val = float(W_t[n, k]) * S_t[nb, kb]
            acc += w_val * x_raw[k]
        y[n] = acc
    return y

tensors = [
    ("model.layers.0.self_attn.q_proj.weight", "0_self_attn.q_proj", 2048, 2048),
    ("model.layers.0.self_attn.k_proj.weight", "0_self_attn.k_proj", 2048, 1024),
    ("model.layers.0.self_attn.v_proj.weight", "0_self_attn.v_proj", 2048, 1024),
    ("model.layers.0.self_attn.o_proj.weight", "0_self_attn.o_proj", 2048, 2048),
    ("model.layers.0.mlp.gate_proj.weight", "0_mlp.gate_proj", 2048, 6144),
    ("model.layers.0.mlp.up_proj.weight",     "0_mlp.up_proj",    2048, 6144),
    ("model.layers.0.mlp.down_proj.weight",   "0_mlp.down_proj", 6144, 2048),
]

print("=" * 65)
print("INT8 Accuracy vs BF16 Original — Qwen3-1.7B Layer 0")
print("=" * 65)

np.random.seed(42)

for tname, wname, K, N in tensors:
    # Read BF16 original
    w_bf16 = read_safetensor_bf16(MODEL, tname)
    # safetensors shape is [out_features, in_features] = [N, K]
    if w_bf16.shape == (K, N):
        w_bf16 = w_bf16.T  # make [N, K]
    
    # Load INT8 (already transposed N×K)
    i8_data, i8_scales = load_int8_weight(f"{INT8_DIR}/{wname}")
    assert i8_data.shape == (N, K), f"Expected ({N},{K}) got {i8_data.shape}"
    
    # Dequant for comparison
    i8_scales_exp = np.repeat(np.repeat(i8_scales, 16, axis=0), 16, axis=1)
    w_i8_dq = i8_data.astype(np.float32) * i8_scales_exp
    
    # Per-element error
    w_abs = np.max(np.abs(w_bf16))
    err = np.abs(w_bf16 - w_i8_dq) / (w_abs if w_abs > 0 else 1.0)
    
    # GEMV comparison
    x = np.random.randn(K).astype(np.float32) * 0.5
    # Input scales for INT8 GEMV
    nb_x = K // 16
    x_scale = np.full(nb_x, 0.5/127.0)  # x is ~0.5 amplitude
    y_ref = w_bf16[:N, :K] @ x
    y_i8 = gemv_int8_cpu(x, i8_data, i8_scales, x_scale, K, N)
    y_abs = np.max(np.abs(y_ref))
    y_err = np.abs(y_ref - y_i8) / (y_abs if y_abs > 0 else 1.0)
    
    print(f"  {wname:30s} | w_err: max={np.max(err):8.2e} mean={np.mean(err):8.2e} | "
          f"gemv_err: max={np.max(y_err):8.2e} mean={np.mean(y_err):8.2e}")

print("\nDone.")