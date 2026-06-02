#!/usr/bin/env python3
"""Generic block-16 quantization for Qwen3 models.

Supports INT8 and INT4 quantization.
Handles single-file and multi-shard safetensors.

Usage:
    python3 scripts/quantize_generic.py <model_path> <output_dir> [int8|int4]

Example:
    python3 scripts/quantize_generic.py /mnt/data/ai/hf/qwen3-1.7b-base weights_int8_bf16 int8
    python3 scripts/quantize_generic.py /mnt/data/ai/hf/qwen3-1.7b-base weights_int4_qwen3_1.7b int4
"""
import struct, json, os, sys
import numpy as np

BLOCK = 16

def quantize_per_row_int8(W_f32):
    """INT8: symmetric block-16, range [-127..127]."""
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

def quantize_per_row_int4(W_f32):
    """INT4: symmetric block-16, 4-bit signed [-7..7], nibble-packed 2 values/byte.

    Returns:
        packed: uint8 array [N * K/2] — lower nibble = even idx, upper nibble = odd idx
        scales: float32 array [N * num_K_blks] — per-16-element block scales
    """
    N, K = W_f32.shape
    assert K % BLOCK == 0, f"K={K} not divisible by block={BLOCK}"
    num_K_blks = K // BLOCK

    # Compute per-block scales (absmax / 7 for 4-bit range)
    W_blk = W_f32.reshape(N, num_K_blks, BLOCK)
    block_max = np.max(np.abs(W_blk), axis=2)  # [N, num_K_blks]
    scales = block_max / 7.0
    scales = np.maximum(scales, 1e-10)  # prevent div-by-zero

    # Quantize to [-7..7], symmetric
    sc_bc = scales[:, :, np.newaxis]  # [N, num_K_blks, 1]
    sc_bc = np.repeat(sc_bc, BLOCK, axis=2)  # [N, num_K_blks, BLOCK]
    sc_bc = sc_bc.reshape(N, K)  # [N, K]
    q = np.round(W_f32 / sc_bc)
    q = np.clip(q, -7, 7).astype(np.int8)  # [N, K]

    # Vectorized nibble-pack: reshape to [N, K/2, 2], pack even/odd into bytes
    q_reshaped = q.reshape(N, K // 2, 2)
    nib0 = q_reshaped[:, :, 0]  # even indices
    nib1 = q_reshaped[:, :, 1]  # odd indices
    packed = ((nib0 + 8) & 0x0F) | (((nib1 + 8) & 0x0F) << 4)

    return packed, scales.astype(np.float32)

def write_weight_int8(prefix, int8_data, scales, K_in, N_out):
    """Write INT8 weight file with same header format as existing."""
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
    print(f"  {prefix}: [{N_out}×{K_in}] INT8 data={mb_data:.1f}MB scales={mb_sc:.1f}MB")

def write_weight_int4(prefix, int4_packed, scales, K_in, N_out):
    """Write INT4 weight file — same header as INT8, data = K×N/2 bytes."""
    num_K_blks = K_in // BLOCK
    header = np.array([K_in, N_out, BLOCK, num_K_blks, N_out], dtype=np.int32)
    path = f"{prefix}.int4_t"
    with open(path, 'wb') as f:
        f.write(header.tobytes())
        f.write(int4_packed.tobytes())
    path = f"{prefix}.scale_t"
    with open(path, 'wb') as f:
        f.write(header.tobytes())
        f.write(scales.tobytes())
    mb_data = int4_packed.nbytes / (1024*1024)
    mb_sc = scales.nbytes / (1024*1024)
    print(f"  {prefix}: [{N_out}×{K_in}] INT4 data={mb_data:.1f}MB scales={mb_sc:.1f}MB")

def find_model_shards(model_dir):
    """Find all safetensor shards in model directory (handles single and multi-shard)."""
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        return [single]

    shard_files = sorted([
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.startswith("model-") and f.endswith(".safetensors")
    ])
    if shard_files:
        return shard_files

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
    """Load headers from all shards. Returns dict: tensor_name -> (shard_idx, header_info)."""
    if len(shard_paths) == 1:
        with open(shard_paths[0], 'rb') as f:
            hdr_len = struct.unpack('Q', f.read(8))[0]
            hdr = json.loads(f.read(hdr_len))
        tensor_map = {}
        for name, info in hdr.items():
            if name == '__metadata__':
                continue
            tensor_map[name] = (0, info)
        return tensor_map

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
    shard_idx, info = tensor_map[name]
    shard_path = shard_paths[shard_idx]
    start, end = info['data_offsets']
    with open(shard_path, 'rb') as f:
        f.seek(8)
        f.seek(0)
        hdr_len_shard = struct.unpack('Q', f.read(8))[0]
        f.seek(8 + hdr_len_shard + start)
        raw = f.read(end - start)

    dtype_map = {
        'BF16': (np.uint16, 2),
        'F16':  (np.uint16, 2),
        'F32':  (np.float32, 4),
        'F64':  (np.float64, 8),
        'U8':   (np.uint8, 1),
        'I8':   (np.int8, 1),
        'U16':  (np.uint16, 2),
        'I16':  (np.int16, 2),
        'U32':  (np.uint32, 2),
        'I32':  (np.int32, 4),
        'U64':  (np.uint64, 8),
        'I64':  (np.int64, 8),
    }
    dtype_str = info.get('dtype', 'BF16')
    elem_size = dtype_map.get(dtype_str, (np.uint16, 2))[1]
    shape = info['shape']

    if dtype_str == 'BF16':
        u16 = np.frombuffer(raw, dtype=np.uint16).copy()
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        return f32.reshape(shape)
    else:
        dtype_np, _ = dtype_map.get(dtype_str, (np.float32, 4))
        return np.frombuffer(raw, dtype=dtype_np).reshape(shape)

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/quantize_generic.py <model_path> <output_dir> [int8|int4]")
        print("Example: python3 scripts/quantize_generic.py /mnt/data/ai/hf/qwen3-1.7b-base weights_int8_bf16 int8")
        print("         python3 scripts/quantize_generic.py /mnt/data/ai/hf/qwen3-1.7b-base weights_int4_qwen3_1.7b int4")
        sys.exit(1)

    MODEL = sys.argv[1]
    OUT = sys.argv[2]
    fmt = sys.argv[3].lower() if len(sys.argv) > 3 else 'int8'
    if fmt not in ('int8', 'int4'):
        print("Format must be 'int8' or 'int4'")
        sys.exit(1)
    print(f"Quantization format: {fmt.upper()}")

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

    NL = config["num_hidden_layers"]
    H = config["hidden_size"]
    I = config.get("intermediate_size", H * 4)
    V = config.get("vocab_size", 151936)

    print(f"Model: {MODEL}")
    print(f"Config: {NL} layers, H={H}, I={I}, V={V}")
    print(f"Output: {OUT}/")

    shard_paths = find_model_shards(MODEL)
    if not shard_paths:
        print("ERROR: No safetensor files found")
        sys.exit(1)
    print(f"Using {len(shard_paths)} safetensor shard(s)")

    tensor_map = load_safetensor_headers(shard_paths)
    print(f"Loaded {len(tensor_map)} tensors from headers")

    WEIGHT_NAMES = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]

    os.makedirs(OUT, exist_ok=True)

    for layer in range(NL):
        for wn in WEIGHT_NAMES:
            tname = f"model.layers.{layer}.{wn}.weight"
            if tname not in tensor_map:
                print(f"  WARNING: {tname} not found, skipping")
                continue
            W = read_tensor(tensor_map, shard_paths, tname)
            N_out, K_in = W.shape
            if fmt == 'int4':
                int4_packed, scales = quantize_per_row_int4(W)
                prefix = f"{OUT}/{layer}_{wn}"
                write_weight_int4(prefix, int4_packed, scales, K_in, N_out)
            else:
                int8_data, scales = quantize_per_row_int8(W)
                prefix = f"{OUT}/{layer}_{wn}"
                write_weight_int8(prefix, int8_data, scales, K_in, N_out)

    print(f"\nProcessing embed_tokens...")
    W_emb = read_tensor(tensor_map, shard_paths, "model.embed_tokens.weight")
    N_out, K_in = W_emb.shape
    if fmt == 'int4':
        int4_packed, scales = quantize_per_row_int4(W_emb)
        write_weight_int4(f"{OUT}/embed_tokens", int4_packed, scales, K_in, N_out)
    else:
        int8_data, scales = quantize_per_row_int8(W_emb)
        write_weight_int8(f"{OUT}/embed_tokens", int8_data, scales, K_in, N_out)

    print(f"\nDone. {NL} layers × {len(WEIGHT_NAMES)} weights + embed_tokens ({fmt.upper()})")

if __name__ == "__main__":
    main()