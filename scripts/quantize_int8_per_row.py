#!/usr/bin/env python3
"""Per-row INT8 quantize 1.7B weights.

Usage: python3 scripts/quantize_int8_per_row.py <model_path> <output_dir>

Per-row scaling: 1 scale per output row instead of per 16-element block.
Scale file header: [K, N, 1, 1, N] → then N floats (one per row).
"""
import struct, json, os, sys, glob
import numpy as np

BLOCK = 16  # same block size for weight data

def find_config(model_path):
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        snap = os.path.join(model_path, "snapshots")
        if os.path.isdir(snap):
            for s in sorted(os.listdir(snap)):
                cp = os.path.join(snap, s, "config.json")
                if os.path.exists(cp):
                    config_path = cp; break
    return config_path

def find_shards(model_path):
    shards = []
    for pat in ["model*.safetensors", "*.safetensors"]:
        shards.extend(sorted(glob.glob(os.path.join(model_path, pat))))
        for d in ["snapshots"]:
            sd = os.path.join(model_path, d)
            if os.path.isdir(sd):
                for s in sorted(os.listdir(sd)):
                    shards.extend(sorted(glob.glob(os.path.join(sd, s, pat))))
    # Deduplicate
    seen = set()
    return [s for s in shards if s not in seen and not seen.add(s)]

def load_safetensor_headers(shard_paths):
    import json
    tensor_map = {}
    for sp in shard_paths:
        with open(sp, 'rb') as f:
            hlen = struct.unpack('Q', f.read(8))[0]
            hdr = json.loads(f.read(hlen).decode('utf-8'))
            for k, v in hdr.items():
                if isinstance(v, dict) and 'dtype' in v:
                    tensor_map[k] = (sp, v)
    return tensor_map

def read_tensor(tensor_map, shard_paths, name):
    if name not in tensor_map:
        return None
    sp, info = tensor_map[name]
    dtype_str = info['dtype']
    shape = info['shape']
    start, end = info['data_offsets']
    with open(sp, 'rb') as f:
        # Read header length (8 bytes at start)
        hdr_len = struct.unpack('Q', f.read(8))[0]
        # Absolute offset to data = 8 (header_len) + hdr_len (json) + start (relative)
        abs_off = 8 + hdr_len + start
        f.seek(abs_off)
        size = end - start
        raw = f.read(size)
    if dtype_str.upper() in ('BFLOAT16', 'BF16'):
        u16 = np.frombuffer(raw, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        return u32.view(np.float32).reshape(shape)
    return np.frombuffer(raw, dtype=dtype_str).reshape(shape)

def write_f32(path, data):
    data.tofile(path + ".f32")

def quantize_per_row(W_f32):
    """INT8 per-row quantization. Returns (int8_data, scales_per_row)."""
    N, K = W_f32.shape
    scales = np.max(np.abs(W_f32), axis=1) / 127.0
    scales = np.maximum(scales, 1e-10)
    q = np.round(W_f32 / scales[:, np.newaxis])
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, scales.astype(np.float32)

def write_int8_per_row(prefix, int8_data, scales, K_in, N_out):
    hdr = np.array([K_in, N_out, BLOCK, K_in // BLOCK, N_out], dtype=np.int32)
    with open(f"{prefix}.int8_t", 'wb') as f:
        f.write(hdr.tobytes())
        f.write(int8_data.tobytes())
    # Per-row scale header: [K, N, 1, 1, N]
    hdr_sc = np.array([K_in, N_out, 1, 1, N_out], dtype=np.int32)
    with open(f"{prefix}.scale_t", 'wb') as f:
        f.write(hdr_sc.tobytes())
        f.write(scales.tobytes())
    mb = (int8_data.nbytes + scales.nbytes) / (1024*1024)
    print(f"  {os.path.basename(prefix)}: [{N_out}×{K_in}] {mb:.1f}MB per-row")

def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    MODEL = sys.argv[1]
    OUT = sys.argv[2]
    os.makedirs(OUT, exist_ok=True)

    # Load config
    config_path = find_config(MODEL)
    with open(config_path) as f:
        config = json.load(f)
    NL = config.get("num_hidden_layers", 28)
    H = config.get("hidden_size", 2048)
    V = config.get("vocab_size", 151936)
    print(f"Model: {MODEL}")
    print(f"  {NL} layers, H={H}, V={V}")

    shard_paths = find_shards(MODEL)
    if not shard_paths:
        print("ERROR: no safetensor files")
        sys.exit(1)
    tensor_map = load_safetensor_headers(shard_paths)
    print(f"  {len(tensor_map)} tensors across {len(shard_paths)} shards")

    # Embed tokens
    tname = "model.embed_tokens.weight"
    if tname in tensor_map:
        W = read_tensor(tensor_map, shard_paths, tname)
        int8_data, scales = quantize_per_row(W)
        write_int8_per_row(f"{OUT}/embed_tokens", int8_data, scales, *W.shape[::-1])

    # Layer weights
    for l in range(NL):
        print(f"\nLayer {l}:")
        # Q/K/V/O
        for wn in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            tname = f"model.layers.{l}.self_attn.{wn}.weight"
            if tname in tensor_map:
                W = read_tensor(tensor_map, shard_paths, tname)
                int8_data, scales = quantize_per_row(W)
                write_int8_per_row(f"{OUT}/{l}_self_attn.{wn}", int8_data, scales, *W.shape[::-1])
        # MLP
        for wn in ["gate_proj", "up_proj", "down_proj"]:
            tname = f"model.layers.{l}.mlp.{wn}.weight"
            if tname in tensor_map:
                W = read_tensor(tensor_map, shard_paths, tname)
                int8_data, scales = quantize_per_row(W)
                write_int8_per_row(f"{OUT}/{l}_mlp.{wn}", int8_data, scales, *W.shape[::-1])
        # Layer norms
        for ln in ["input_layernorm", "post_attention_layernorm"]:
            tname = f"model.layers.{l}.{ln}.weight"
            if tname in tensor_map:
                w = read_tensor(tensor_map, shard_paths, tname)
                write_f32(f"{OUT}/{l}_{ln}", w)

    # Final norm
    tname = "model.norm.weight"
    if tname in tensor_map:
        w = read_tensor(tensor_map, shard_paths, tname)
        write_f32(f"{OUT}/final_norm", w)

    # LM head
    tname = "lm_head.weight"
    if tname in tensor_map:
        W = read_tensor(tensor_map, shard_paths, tname)
        int8_data, scales = quantize_per_row(W)
        write_int8_per_row(f"{OUT}/lm_head", int8_data, scales, *W.shape[::-1])

    # QK norms
    qk = []
    for l in range(NL):
        for nn in ["q_norm", "k_norm"]:
            tname = f"model.layers.{l}.self_attn.{nn}.weight"
            if tname in tensor_map:
                w = read_tensor(tensor_map, shard_paths, tname)
                qk.extend(w.tolist())
    if qk:
        np.array(qk, dtype=np.float32).tofile(f"{OUT}/qk_norms.f32")

    print(f"\nDone. Output in {OUT}/")

if __name__ == '__main__':
    main()
