#!/usr/bin/env python3
"""Quantize Qwen3.5-9B weights to INT8 for blackwell inference.

Handles mixed layer types: 24 linear_attention (GatedDeltaNet) + 8 full_attention (GQA).

Usage:
    python3 scripts/quantize_qwen35.py /mnt/data/ai/hf/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a weights_int8_qwen35_9b

Output structure:
    weights_int8_qwen35_9b/
    ├── 0_linear_attn.in_proj_qkv.int8_t / .scale_t
    ├── 0_linear_attn.in_proj_a.int8_t / .scale_t
    ├── 0_linear_attn.in_proj_b.int8_t / .scale_t
    ├── 0_linear_attn.in_proj_z.int8_t / .scale_t
    ├── 0_linear_attn.out_proj.int8_t / .scale_t
    ├── 0_linear_attn.conv1d.weight.f16        (kept as BF16)
    ├── 0_linear_attn.A_log.f32                (kept as F32)
    ├── 0_linear_attn.dt_bias.f16              (kept as BF16)
    ├── 0_linear_attn.norm.f32                 (kept as F32)
    ├── 0_input_layernorm.f32
    ├── 0_mlp.gate_proj.int8_t / .scale_t
    ├── 0_mlp.up_proj.int8_t / .scale_t
    ├── 0_mlp.down_proj.int8_t / .scale_t
    ├── 3_self_attn.q_proj.int8_t / .scale_t
    ├── 3_self_attn.k_proj.int8_t / .scale_t
    ├── 3_self_attn.v_proj.int8_t / .scale_t
    ├── 3_self_attn.o_proj.int8_t / .scale_t
    ├── ...
    ├── embed_tokens.int8_t / .scale_t
    ├── final_norm.f32
    └── qk_norms.f32
"""
import struct, json, os, sys
import numpy as np

BLOCK = 16

# Qwen3.5-9B layer types (from config.json)
LAYER_TYPES = [
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
    "linear_attention", "linear_attention", "linear_attention", "full_attention",
]

def quantize_per_row(W_f32):
    """INT8 block-16 per-row quantization. Returns (int8_data, scales)."""
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

def write_int8_weight(prefix, int8_data, scales, K_in, N_out):
    """Write INT8 weight + scale files."""
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
    mb = (int8_data.nbytes + scales.nbytes) / (1024*1024)
    print(f"  {os.path.basename(prefix)}: [{N_out}×{K_in}] {mb:.1f}MB")

def write_raw(prefix, data, dtype_str):
    """Write raw data file (BF16 or F32)."""
    path = f"{prefix}.{dtype_str}"
    with open(path, 'wb') as f:
        f.write(data.tobytes())
    mb = data.nbytes / (1024*1024)
    print(f"  {os.path.basename(path)}: {mb:.2f}MB")

def find_model_shards(model_dir):
    """Find all safetensor shards."""
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        return [single]
    shard_files = sorted([
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.startswith("model.safetensors-") and f.endswith(".safetensors")
    ])
    if shard_files:
        return shard_files
    # Also try model- prefix
    shard_files = sorted([
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.startswith("model-") and f.endswith(".safetensors")
    ])
    if shard_files:
        return shard_files
    # Snapshot search
    snapshot_dir = os.path.join(model_dir, "snapshots")
    if os.path.isdir(snapshot_dir):
        for snap in sorted(os.listdir(snapshot_dir)):
            snap_path = os.path.join(snapshot_dir, snap)
            if os.path.isdir(snap_path):
                single = os.path.join(snap_path, "model.safetensors")
                if os.path.exists(single):
                    return [single]
                shards = sorted([
                    os.path.join(snap_path, f)
                    for f in os.listdir(snap_path)
                    if f.startswith("model-") and f.endswith(".safetensors")
                ])
                if shards:
                    return shards
    return []

def load_safetensor_headers(shard_paths):
    """Load headers from all shards. Returns dict: tensor_name -> (shard_idx, info)."""
    tensor_map = {}
    for shard_idx, shard_path in enumerate(shard_paths):
        with open(shard_path, 'rb') as f:
            hdr_len = struct.unpack('Q', f.read(8))[0]
            hdr = json.loads(f.read(hdr_len))
        for name, info in hdr.items():
            if name == '__metadata__':
                continue
            tensor_map[name] = (shard_idx, info)
    return tensor_map

def read_tensor(tensor_map, shard_paths, name):
    """Read a tensor from safetensors."""
    shard_idx, info = tensor_map[name]
    shard_path = shard_paths[shard_idx]
    start, end = info['data_offsets']
    with open(shard_path, 'rb') as f:
        f.seek(0)
        hdr_len_shard = struct.unpack('Q', f.read(8))[0]
        f.seek(8 + hdr_len_shard + start)
        raw = f.read(end - start)

    dtype_str = info.get('dtype', 'BF16')
    if dtype_str == 'BF16':
        u16 = np.frombuffer(raw, dtype=np.uint16).copy()
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        return f32.reshape(info['shape'])
    elif dtype_str == 'F32':
        return np.frombuffer(raw, dtype=np.float32).reshape(info['shape'])
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 quantize_qwen35.py <model_path> <output_dir>")
        sys.exit(1)

    MODEL = sys.argv[1]
    OUT = sys.argv[2]

    # Load config
    config_path = os.path.join(MODEL, "config.json")
    if not os.path.exists(config_path):
        snap = os.path.join(MODEL, "snapshots")
        if os.path.isdir(snap):
            for s in sorted(os.listdir(snap)):
                cp = os.path.join(snap, s, "config.json")
                if os.path.exists(cp):
                    config_path = cp
                    break

    with open(config_path) as f:
        config = json.load(f)

    text_config = config.get("text_config", config)
    NL = text_config["num_hidden_layers"]
    H = text_config["hidden_size"]
    I = text_config.get("intermediate_size", H * 4)
    V = text_config.get("vocab_size", 248320)
    NK = text_config.get("linear_num_key_heads", 16)
    NV = text_config.get("linear_num_value_heads", 32)
    HD = text_config.get("linear_key_head_dim", 128)

    print(f"Model: {MODEL}")
    print(f"Config: {NL} layers, H={H}, I={I}, V={V}")
    print(f"  Linear attn: {NK} key heads × {HD} dim, {NV} value heads × {HD} dim")
    print(f"  Layer types: {sum(1 for t in LAYER_TYPES if t=='linear_attention')} linear + "
          f"{sum(1 for t in LAYER_TYPES if t=='full_attention')} full")
    print(f"Output: {OUT}/")

    shard_paths = find_model_shards(MODEL)
    if not shard_paths:
        print("ERROR: No safetensor files found")
        sys.exit(1)
    print(f"Using {len(shard_paths)} safetensor shard(s)")

    tensor_map = load_safetensor_headers(shard_paths)
    print(f"Loaded {len(tensor_map)} tensors from headers")

    os.makedirs(OUT, exist_ok=True)

    # ── Process each layer ──────────────────────────────────────────────
    for layer in range(NL):
        ltype = LAYER_TYPES[layer]
        is_linear = (ltype == "linear_attention")
        print(f"\nLayer {layer} ({ltype}):")

        # ── Attention weights ───────────────────────────────────────────
        if is_linear:
            # Linear attention (GatedDeltaNet)
            for wn in ["in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj"]:
                tname = f"model.language_model.layers.{layer}.linear_attn.{wn}.weight"
                if tname not in tensor_map:
                    print(f"  WARNING: {tname} not found")
                    continue
                W = read_tensor(tensor_map, shard_paths, tname)
                N_out, K_in = W.shape
                int8_data, scales = quantize_per_row(W)
                prefix = f"{OUT}/{layer}_linear_attn.{wn}"
                write_int8_weight(prefix, int8_data, scales, K_in, N_out)

            # Conv1d weight (keep as BF16, tiny)
            tname = f"model.language_model.layers.{layer}.linear_attn.conv1d.weight"
            if tname in tensor_map:
                W = read_tensor(tensor_map, shard_paths, tname)
                write_raw(f"{OUT}/{layer}_linear_attn.conv1d.weight", W.astype(np.float16), "f16")

            # A_log (F32)
            tname = f"model.language_model.layers.{layer}.linear_attn.A_log"
            if tname in tensor_map:
                W = read_tensor(tensor_map, shard_paths, tname)
                write_raw(f"{OUT}/{layer}_linear_attn.A_log", W, "f32")

            # dt_bias (BF16 → F32)
            tname = f"model.language_model.layers.{layer}.linear_attn.dt_bias"
            if tname in tensor_map:
                W = read_tensor(tensor_map, shard_paths, tname)
                write_raw(f"{OUT}/{layer}_linear_attn.dt_bias", W, "f32")

            # Norm weight (F32)
            tname = f"model.language_model.layers.{layer}.linear_attn.norm.weight"
            if tname in tensor_map:
                W = read_tensor(tensor_map, shard_paths, tname)
                write_raw(f"{OUT}/{layer}_linear_attn.norm", W, "f32")

        else:
            # Full attention (GQA)
            for wn in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                tname = f"model.language_model.layers.{layer}.self_attn.{wn}.weight"
                if tname not in tensor_map:
                    print(f"  WARNING: {tname} not found")
                    continue
                W = read_tensor(tensor_map, shard_paths, tname)
                N_out, K_in = W.shape
                int8_data, scales = quantize_per_row(W)
                prefix = f"{OUT}/{layer}_self_attn.{wn}"
                write_int8_weight(prefix, int8_data, scales, K_in, N_out)

            # Q/K norm weights (BF16 → F32, tiny)
            for norm_name in ["q_norm", "k_norm"]:
                tname = f"model.language_model.layers.{layer}.self_attn.{norm_name}.weight"
                if tname in tensor_map:
                    W = read_tensor(tensor_map, shard_paths, tname)
                    write_raw(f"{OUT}/{layer}_self_attn.{norm_name}", W, "f32")

        # ── MLP weights (all layers) ───────────────────────────────────
        for wn in ["gate_proj", "up_proj", "down_proj"]:
            tname = f"model.language_model.layers.{layer}.mlp.{wn}.weight"
            if tname not in tensor_map:
                print(f"  WARNING: {tname} not found")
                continue
            W = read_tensor(tensor_map, shard_paths, tname)
            N_out, K_in = W.shape
            int8_data, scales = quantize_per_row(W)
            prefix = f"{OUT}/{layer}_mlp.{wn}"
            write_int8_weight(prefix, int8_data, scales, K_in, N_out)

        # ── LayerNorms (all layers) ────────────────────────────────────
        for ln_name in ["input_layernorm", "post_attention_layernorm"]:
            tname = f"model.language_model.layers.{layer}.{ln_name}.weight"
            if tname in tensor_map:
                W = read_tensor(tensor_map, shard_paths, tname)
                write_raw(f"{OUT}/{layer}_{ln_name}", W, "f32")

    # ── Global weights ──────────────────────────────────────────────────
    print(f"\nGlobal:")

    # Embed tokens
    tname = "model.language_model.embed_tokens.weight"
    if tname in tensor_map:
        W = read_tensor(tensor_map, shard_paths, tname)
        N_out, K_in = W.shape
        int8_data, scales = quantize_per_row(W)
        write_int8_weight(f"{OUT}/embed_tokens", int8_data, scales, K_in, N_out)

    # lm_head (tied with embed_tokens in some models, separate in others)
    tname = "lm_head.weight"
    if tname in tensor_map:
        W = read_tensor(tensor_map, shard_paths, tname)
        N_out, K_in = W.shape
        int8_data, scales = quantize_per_row(W)
        write_int8_weight(f"{OUT}/lm_head", int8_data, scales, K_in, N_out)

    # Final norm
    tname = "model.language_model.norm.weight"
    if tname in tensor_map:
        W = read_tensor(tensor_map, shard_paths, tname)
        write_raw(f"{OUT}/final_norm", W, "f32")

    # ── Summary ─────────────────────────────────────────────────────────
    total_int8 = 0
    total_raw = 0
    for f in os.listdir(OUT):
        fp = os.path.join(OUT, f)
        sz = os.path.getsize(fp)
        if f.endswith(('.int8_t', '.scale_t')):
            total_int8 += sz
        else:
            total_raw += sz
    print(f"\nDone. INT8: {total_int8/(1024**3):.2f} GB, Raw: {total_raw/(1024**3):.2f} GB, "
          f"Total: {(total_int8+total_raw)/(1024**3):.2f} GB")

if __name__ == "__main__":
    main()
