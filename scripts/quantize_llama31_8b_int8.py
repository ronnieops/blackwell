#!/usr/bin/env python3
"""Quantize Llama 3.1 8B from safetensors to Blackwell INT8 format.

Uses block-16 INT8 symmetric quantization (range [-127..127]).
Each weight stored as int8_t, scales per block-16 as float32.
For multi-shard models, reads from all shards.

Usage:
    python3 scripts/quantize_llama31_8b_int8.py [output_dir]

Expected: 9.6 GB, ~4 t/s on RTX 5060 Ti (INT8 GEMV).
"""
import struct, json, os, sys, re, math
import numpy as np

HF_PATH = "/mnt/data/ai/models/llama31-8b-safetensors"
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/ai/models/llama31-8b-int8-from-safetensors"
BLOCK = 16

# Llama 3.1 8B config
H = 4096
Q = 4096
KV = 1024
I = 14336
NL = 32
nqh = 32
nkv = 8
hd = 128
V = 128256
eps = 1e-5
rope_theta = 500000.0


def find_shards(model_dir):
    files = sorted([
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.startswith("model-") and f.endswith(".safetensors")
    ])
    return files


def load_tensor_map(shard_paths):
    tensor_map = {}
    for shard_idx, shard_path in enumerate(shard_paths):
        with open(shard_path, 'rb') as f:
            hdr_len = struct.unpack('Q', f.read(8))[0]
            hdr = json.loads(f.read(hdr_len))
        for name, info in hdr.items():
            if name != '__metadata__':
                tensor_map[name] = (shard_idx, info)
    return tensor_map


def read_tensor(tensor_map, shard_paths, name):
    shard_idx, info = tensor_map[name]
    shard_path = shard_paths[shard_idx]
    start, end = info['data_offsets']
    with open(shard_path, 'rb') as f:
        f.seek(0)
        hdr_len = struct.unpack('Q', f.read(8))[0]
        f.seek(8 + hdr_len + start)
        raw = f.read(end - start)
    if info['dtype'] == 'BF16':
        arr = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif info['dtype'] == 'F32':
        arr = np.frombuffer(raw, dtype=np.float32)
    else:
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    return arr.reshape(info['shape'])


def quantize_int8_sym(W, block=16):
    """INT8 symmetric block quantization. Range [-127..127].

    INT8 weights are stored as raw int8_t (not packed like INT4).
    Scale layout: [N][num_k_blks] float32.

    Returns:
        q_int8: ndarray int8 [N, K]
        scales: ndarray float32 [N, num_k_blks]
    """
    N, K = W.shape
    assert K % block == 0
    num_blks = K // block
    W_blk = W.reshape(N, num_blks, block)
    blk_abs = np.max(np.abs(W_blk), axis=2)
    scales = np.maximum(blk_abs, 1e-10) / 127.0
    q = np.round(W_blk / (blk_abs / 127.0)[:, :, np.newaxis])
    q = np.clip(q, -127, 127).astype(np.int8)
    q_int8 = q.reshape(N, K)
    scales = (blk_abs / 127.0).astype(np.float32)
    return q_int8, scales


def write_weight_int8(prefix, q_int8, scales, K_in, N_out):
    """Write INT8 weight file: header + raw int8 data + scale header + scale data."""
    num_kb = K_in // BLOCK
    # int8_t file
    with open(f"{prefix}.int8_t", 'wb') as f:
        header = np.array([K_in, N_out, BLOCK, num_kb, 1], dtype=np.int32)
        f.write(header.tobytes())
        f.write(q_int8.tobytes())
    # scale file
    with open(f"{prefix}.scale_t", 'wb') as f:
        header_sc = np.array([0, 0, 0, num_kb, N_out], dtype=np.int32)
        f.write(header_sc.tobytes())
        f.write(scales.tobytes())
    mb = (q_int8.nbytes + scales.nbytes) / (1024*1024)
    print(f"  {os.path.basename(prefix)}: {N_out}x{K_in} {mb:.1f}MB")


def write_f32(prefix, data):
    with open(f"{prefix}.f32", 'wb') as f:
        f.write(data.astype(np.float32).tobytes())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    shard_paths = find_shards(HF_PATH)
    print(f"Found {len(shard_paths)} shards in {HF_PATH}")
    tensor_map = load_tensor_map(shard_paths)

    weight_names = []
    for l in range(NL):
        weight_names += [
            (f"{l}_self_attn.q_proj", f"model.layers.{l}.self_attn.q_proj.weight", H, Q),
            (f"{l}_self_attn.k_proj", f"model.layers.{l}.self_attn.k_proj.weight", H, KV),
            (f"{l}_self_attn.v_proj", f"model.layers.{l}.self_attn.v_proj.weight", H, KV),
            (f"{l}_self_attn.o_proj", f"model.layers.{l}.self_attn.o_proj.weight", Q, H),
            (f"{l}_mlp.gate_proj", f"model.layers.{l}.mlp.gate_proj.weight", H, I),
            (f"{l}_mlp.up_proj", f"model.layers.{l}.mlp.up_proj.weight", H, I),
            (f"{l}_mlp.down_proj", f"model.layers.{l}.mlp.down_proj.weight", I, H),
        ]

    norm_names = []
    for l in range(NL):
        norm_names += [
            (f"{l}_input_layernorm", f"model.layers.{l}.input_layernorm.weight", H),
            (f"{l}_post_attention_layernorm", f"model.layers.{l}.post_attention_layernorm.weight", H),
        ]

    embed_t_name = "model.embed_tokens.weight"
    lm_head_name = "model.lm_head.weight"
    final_norm_name = "model.norm.weight"

    print("Quantizing weights...")
    for local_name, hf_name, K_in, N_out in weight_names:
        W = read_tensor(tensor_map, shard_paths, hf_name).astype(np.float32)
        N_actual, K_actual = W.shape
        print(f"  {local_name}: {N_actual}x{K_actual} (K_in={K_in}, N_out={N_out})")
        q_int8, scales = quantize_int8_sym(W, BLOCK)
        write_weight_int8(f"{OUT_DIR}/{local_name}", q_int8, scales, K_in, N_out)
        del W, q_int8, scales

    print("Quantizing norms...")
    for local_name, hf_name, dim in norm_names:
        w = read_tensor(tensor_map, shard_paths, hf_name).astype(np.float32)
        write_f32(f"{OUT_DIR}/{local_name}", w)
        del w

    print("Quantizing embed_tokens...")
    W_emb = read_tensor(tensor_map, shard_paths, embed_t_name).astype(np.float32)
    N_emb, K_emb = W_emb.shape
    print(f"  embed_tokens: {N_emb}x{K_emb}")
    q_int8, scales = quantize_int8_sym(W_emb, BLOCK)
    write_weight_int8(f"{OUT_DIR}/embed_tokens", q_int8, scales, K_emb, N_emb)
    del W_emb, q_int8, scales

    print("Quantizing lm_head (tied to embed)...")
    # Llama 3.1 8B has tied lm_head — copy embed weights
    import shutil
    shutil.copy(f"{OUT_DIR}/embed_tokens.int8_t", f"{OUT_DIR}/lm_head.int8_t")
    shutil.copy(f"{OUT_DIR}/embed_tokens.scale_t", f"{OUT_DIR}/lm_head.scale_t")

    print("Writing final_norm...")
    w_fn = read_tensor(tensor_map, shard_paths, final_norm_name).astype(np.float32)
    write_f32(f"{OUT_DIR}/final_norm", w_fn)
    del w_fn

    # QK norms: identity for Llama (no separate QK norms)
    qk_norms = np.ones((NL, 2, hd), dtype=np.float32)
    write_f32(f"{OUT_DIR}/qk_norms", qk_norms)

    # RoPE config
    rope_cfg = np.array([rope_theta, eps], dtype=np.float32)
    write_f32(f"{OUT_DIR}/rope_config", rope_cfg)

    print(f"\nDone. Output: {OUT_DIR}")
    total = 0
    for f in os.listdir(OUT_DIR):
        fp = os.path.join(OUT_DIR, f)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp)
            total += sz
    print(f"Total: {total/1024/1024/1024:.1f} GB")


if __name__ == "__main__":
    main()
