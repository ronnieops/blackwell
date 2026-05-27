#!/usr/bin/env python3
"""convert_bf16_to_int8.py — Build INT8 weights directly from BF16 original.

This avoids the FP4→INT8 requantization error. Reads BF16 safetensors,
computes per-16×16-block INT8 scales, packs, transposes, writes files.

Usage: python3 bench/convert_bf16_to_int8.py
"""
import struct, json, os
import numpy as np

MODEL = "/mnt/data/ai/hf/qwen3-1.7b-base/model.safetensors"
OUT = "weights_int8_bf16"

os.makedirs(OUT, exist_ok=True)

# Read safetensors header
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

def quant_int8_transposed(w, block=16):
    """Quantize FP32 weight → INT8 transposed.
    w: [N, K] output×input orientation.
    Returns: (int8_t: N×K, scale_t: N/16 × K/16)
    """
    N, K = w.shape
    nnb, nkb = N // block, K // block
    # Compute per-block scales
    scales = np.zeros((nnb, nkb), dtype=np.float32)
    q = np.zeros((N, K), dtype=np.int8)
    for nb in range(nnb):
        for kb in range(nkb):
            blk = w[nb*block:(nb+1)*block, kb*block:(kb+1)*block]
            amax = np.max(np.abs(blk))
            sc = amax / 127.0 if amax > 1e-10 else 1.0/127.0
            scales[nb, kb] = sc
            q[nb*block:(nb+1)*block, kb*block:(kb+1)*block] = \
                np.clip(np.round(blk / sc), -128, 127).astype(np.int8)
    return q, scales

def write_weight(data, scales, K, N, prefix):
    """Write INT8 weight files matching our GPU format.
    data: INT8 N×K (transposed)
    scales: float32 N/16 × K/16 (transposed)
    """
    header = struct.pack('5i', K, N, 16, K//16, N//16)
    with open(f"{prefix}.int8_t", 'wb') as f:
        f.write(header)
        f.write(data.tobytes())
    with open(f"{prefix}.scale_t", 'wb') as f:
        f.write(header)
        f.write(scales.tobytes())
    print(f"  Wrote {prefix}.* (K={K}, N={N}, data={data.nbytes}B, scales={scales.nbytes}B)")

# Map tensor names → output names + dims
# safetensors stores [out_features, in_features] = [N, K]
tensors = [
    ("model.layers.0.self_attn.q_proj.weight", "0_self_attn.q_proj"),
    ("model.layers.0.self_attn.k_proj.weight", "0_self_attn.k_proj"),
    ("model.layers.0.self_attn.v_proj.weight", "0_self_attn.v_proj"),
    ("model.layers.0.self_attn.o_proj.weight", "0_self_attn.o_proj"),
    ("model.layers.0.mlp.gate_proj.weight",    "0_mlp.gate_proj"),
    ("model.layers.0.mlp.up_proj.weight",       "0_mlp.up_proj"),
    ("model.layers.0.mlp.down_proj.weight",    "0_mlp.down_proj"),
    ("model.layers.1.self_attn.q_proj.weight", "1_self_attn.q_proj"),
    ("model.layers.1.self_attn.k_proj.weight", "1_self_attn.k_proj"),
    ("model.layers.1.self_attn.v_proj.weight", "1_self_attn.v_proj"),
    ("model.layers.1.self_attn.o_proj.weight", "1_self_attn.o_proj"),
    ("model.layers.1.mlp.gate_proj.weight",    "1_mlp.gate_proj"),
    ("model.layers.1.mlp.up_proj.weight",       "1_mlp.up_proj"),
    ("model.layers.1.mlp.down_proj.weight",    "1_mlp.down_proj"),
    ("model.layers.2.self_attn.q_proj.weight", "2_self_attn.q_proj"),
    ("model.layers.2.self_attn.k_proj.weight", "2_self_attn.k_proj"),
    ("model.layers.2.self_attn.v_proj.weight", "2_self_attn.v_proj"),
    ("model.layers.2.self_attn.o_proj.weight", "2_self_attn.o_proj"),
    ("model.layers.2.mlp.gate_proj.weight",    "2_mlp.gate_proj"),
    ("model.layers.2.mlp.up_proj.weight",       "2_mlp.up_proj"),
    ("model.layers.2.mlp.down_proj.weight",    "2_mlp.down_proj"),
    ("model.layers.3.self_attn.q_proj.weight", "3_self_attn.q_proj"),
    ("model.layers.3.self_attn.k_proj.weight", "3_self_attn.k_proj"),
    ("model.layers.3.self_attn.v_proj.weight", "3_self_attn.v_proj"),
    ("model.layers.3.self_attn.o_proj.weight", "3_self_attn.o_proj"),
    ("model.layers.3.mlp.gate_proj.weight",    "3_mlp.gate_proj"),
    ("model.layers.3.mlp.up_proj.weight",       "3_mlp.up_proj"),
    ("model.layers.3.mlp.down_proj.weight",    "3_mlp.down_proj"),
]

print(f"Converting BF16 original → INT8 weights → {OUT}/")
print("=" * 60)

for tname, oname in tensors:
    w = read_bf16_tensor(tname)  # [N, K]
    N, K = w.shape
    # Some tensors might be [K, N] (down_proj in safetensors is [K, N])
    # Check: down_proj.weight has shape [2048, 6144] = [K, N]
    # Our format: K=in_features, N=out_features
    # For down_proj: in=6144, out=2048, so we expect [N=2048, K=6144]
    if w.shape[0] != w.shape[1]:  # non-square
        # Check if it matches our expected orientation
        # gate_proj: [6144, 2048] = [N, K]
        # down_proj: [2048, 6144] = [K, N] -> transpose to [N, K]
        expected = oname.split('_')[-1]
        if expected == 'down_proj' and w.shape[0] < w.shape[1]:
            # down_proj: [K, N] = [in, out] = [6144, 2048]
            # We want [N, K] = [out, in] = [2048, 6144]
            K_act, N_act = w.shape
            w = w.T  # now [N, K]
            N, K = w.shape
        elif w.shape[0] > w.shape[1]:
            # gate/up: [N, K] = [6144, 2048] — correct
            N, K = w.shape
    
    q, scales = quant_int8_transposed(w)
    write_weight(q, scales, K, N, f"{OUT}/{oname}")

print("\nDone.")