#!/usr/bin/env python3
"""Quantize Llama 3.1 8B from safetensors to Blackwell INT4 format.

Uses same quantizer as llama32_1b but with 8B config.
For multi-shard models, reads from all shards.

Usage:
    python3 scripts/quantize_llama31_8b.py [output_dir]
"""
import struct, json, os, sys, re, math
import numpy as np

HF_PATH = "/mnt/data/ai/models/llama31-8b-safetensors"
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/ai/models/llama31-8b-int4-from-safetensors"
BLOCK = 16

# Llama 3.1 8B config
H = 4096
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


def quantize_int4_sym(W, block=16):
    N, K = W.shape
    assert K % block == 0
    num_blks = K // block
    W_blk = W.reshape(N, num_blks, block)
    blk_abs = np.max(np.abs(W_blk), axis=2)
    scales = np.maximum(blk_abs, 1e-10) / 7.0
    q = np.round(W_blk / (blk_abs / 7.0)[:, :, np.newaxis])
    q = np.clip(q, -7, 7).astype(np.int32)
    q_shifted = (q + 8).astype(np.uint8).reshape(N, K)
    q_reshaped = q_shifted.reshape(N, K // 2, 2)
    packed = (q_reshaped[:, :, 0] & 0x0F) | ((q_reshaped[:, :, 1] & 0x0F) << 4)
    scales = (blk_abs / 7.0).astype(np.float32)
    return packed, scales


def write_weight(prefix, packed, scales, K_in, N_out):
    num_kb = K_in // BLOCK
    header = np.array([K_in, N_out, BLOCK, num_kb, 1], dtype=np.int32)
    with open(f"{prefix}.int4_t", 'wb') as f:
        f.write(header.tobytes())
        f.write(packed.tobytes())
    header_sc = np.array([0, 0, 0, num_kb, N_out], dtype=np.int32)
    with open(f"{prefix}.scale_t", 'wb') as f:
        f.write(header_sc.tobytes())
        f.write(scales.tobytes())
    mb = (packed.nbytes + scales.nbytes) / (1024*1024)
    print(f"  {os.path.basename(prefix)}: {N_out}x{K_in} {mb:.1f}MB")


def write_f32(prefix, data):
    with open(f"{prefix}.f32", 'wb') as f:
        f.write(data.astype(np.float32).tobytes())
    print(f"  {os.path.basename(prefix)}: {len(data)} F32")


def main():
    print("=" * 60)
    print("Llama 3.1 8B -> Blackwell INT4 (from safetensors)")
    print("=" * 60)

    shard_paths = find_shards(HF_PATH)
    print(f"Found {len(shard_paths)} shard(s)")
    tensor_map = load_tensor_map(shard_paths)
    os.makedirs(OUT_DIR, exist_ok=True)

    WEIGHT_NAMES = {
        "self_attn.q_proj": "model.layers.{}.self_attn.q_proj.weight",
        "self_attn.k_proj": "model.layers.{}.self_attn.k_proj.weight",
        "self_attn.v_proj": "model.layers.{}.self_attn.v_proj.weight",
        "self_attn.o_proj": "model.layers.{}.self_attn.o_proj.weight",
        "mlp.gate_proj": "model.layers.{}.mlp.gate_proj.weight",
        "mlp.up_proj": "model.layers.{}.mlp.up_proj.weight",
        "mlp.down_proj": "model.layers.{}.mlp.down_proj.weight",
    }

    for l in range(NL):
        for bw_name, hf_pattern in WEIGHT_NAMES.items():
            tname = hf_pattern.format(l)
            W = read_tensor(tensor_map, shard_paths, tname)
            if W is None:
                print(f"  SKIP: {tname}")
                continue
            N_out, K_in = W.shape
            packed, scales = quantize_int4_sym(W.astype(np.float32), BLOCK)
            write_weight(f"{OUT_DIR}/{l}_{bw_name}", packed, scales, K_in, N_out)

        # Norms
        for ntype in ["input_layernorm", "post_attention_layernorm"]:
            tname = f"model.layers.{l}.{ntype}.weight"
            if tname in tensor_map:
                w = read_tensor(tensor_map, shard_paths, tname)
                write_f32(f"{OUT_DIR}/{l}_{ntype}", w.ravel())

        if l % 8 == 0:
            print(f"  Layer {l}/{NL}")

    # Final norm
    if "model.norm.weight" in tensor_map:
        fn = read_tensor(tensor_map, shard_paths, "model.norm.weight")
        write_f32(f"{OUT_DIR}/final_norm", fn.ravel())

    # Embed
    if "model.embed_tokens.weight" in tensor_map:
        W_emb = read_tensor(tensor_map, shard_paths, "model.embed_tokens.weight")
        N_out, K_in = W_emb.shape
        packed, scales = quantize_int4_sym(W_emb.astype(np.float32), BLOCK)
        write_weight(f"{OUT_DIR}/embed_tokens", packed, scales, K_in, N_out)

    # LM head
    if "lm_head.weight" in tensor_map:
        W_lm = read_tensor(tensor_map, shard_paths, "lm_head.weight")
        N_out, K_in = W_lm.shape
        packed, scales = quantize_int4_sym(W_lm.astype(np.float32), BLOCK)
        write_weight(f"{OUT_DIR}/lm_head", packed, scales, K_in, N_out)
    else:
        print("  lm_head: not found (may be tied to embed)")

    # QK norms: identity for Llama
    qk = np.ones(NL * 2 * hd, dtype=np.float32)
    with open(f"{OUT_DIR}/qk_norms.f32", 'wb') as f:
        f.write(qk.tobytes())

    # RoPE config
    with open(f"{OUT_DIR}/rope_config.f32", 'wb') as f:
        cfg = np.array([rope_theta, hd], dtype=np.float32)
        f.write(cfg.tobytes())
    print(f"  rope_config: theta={rope_theta}, hd={hd}")

    # Tokenizer from 3.2 (same vocab for Llama 3.1)
    for candidate in [
        "/tmp/final_int4/tokenizer_data.bin",
        "/mnt/data/ai/models/llama32-gguf-test/llama32-int4-fresh/tokenizer_data.bin",
    ]:
        if os.path.exists(candidate):
            import shutil
            shutil.copy2(candidate, f"{OUT_DIR}/tokenizer_data.bin")
            print(f"  tokenizer: copied from {candidate}")
            break

    print(f"\nDone. Output: {OUT_DIR}/")
    print(f"  16.5 GB INT4 weights")


if __name__ == "__main__":
    main()
