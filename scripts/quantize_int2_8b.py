#!/usr/bin/env python3
"""INT2 block-16 symmetric quantization for Qwen3-8B.

INT2 encoding: 2 bits per weight, 4 levels {-2, -1, 0, 1} with block scale.
Block size: 16 weights per FP32 scale.

Storage:
  .int2_t: header [rows, cols, cols/4, BLOCK, 0] + packed bytes (4 weights/byte)
  .scale_t: header [rows, cols/BLOCK, 1, cols/BLOCK, rows] + FP32 scales
"""
import struct, os, sys, json, glob
import numpy as np

HF_PATH = "/mnt/data/ai/hf/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "weights_int2_qwen3_8b"
BLOCK = 16

os.makedirs(OUT_DIR, exist_ok=True)

def build_tensor_map(hf_path):
    """Build tensor name → (shard_idx, {dtype, shape, data_offsets}) map."""
    shard_paths = sorted(glob.glob(os.path.join(hf_path, "model*.safetensors")))
    tensor_map = {}
    for idx, sp in enumerate(shard_paths):
        with open(sp, 'rb') as f:
            hdr_len = struct.unpack('Q', f.read(8))[0]
            hdr = json.loads(f.read(hdr_len))
        for name, info in hdr.items():
            if name == '__metadata__': continue
            tensor_map[name] = (idx, info)
    return tensor_map, shard_paths

def read_tensor(tensor_map, shard_paths, name):
    """Read tensor from safetensors, handling BF16."""
    shard_idx, info = tensor_map[name]
    start, end = info['data_offsets']
    with open(shard_paths[shard_idx], 'rb') as f:
        hdr_len = struct.unpack('Q', f.read(8))[0]
        f.seek(8 + hdr_len + start)
        raw = f.read(end - start)
    if info['dtype'] == 'BF16':
        return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(info['shape'])
    if info['dtype'] == 'F32':
        return np.frombuffer(raw, dtype=np.float32).reshape(info['shape'])
    return np.frombuffer(raw, dtype=np.float16).reshape(info['shape']).astype(np.float32)

def quantize_int2_matrix(W, block=BLOCK):
    """Vectorized INT2 quantization for full matrix. Returns (packed, scales)."""
    rows, cols = W.shape
    n_blocks = cols // block

    # Reshape into blocks: [rows, n_blocks, block]
    W_blk = W.reshape(rows, n_blocks, block)
    amax = np.max(np.abs(W_blk), axis=2)  # [rows, n_blocks]
    sc = np.maximum(amax / 2.0, 1e-10).astype(np.float32)

    # Quantize: q = clip(round(val/sc), -2, 1)
    q = np.clip(np.round(W_blk / sc[:, :, np.newaxis]), -2, 1).astype(np.int8)

    # Pack: offset by 2 → unsigned [0,3], then 4 per byte
    uq = (q + 2).astype(np.uint8)  # [rows, n_blocks, block]
    # Reshape to [rows, n_blocks, block/4, 4]
    uq4 = uq.reshape(rows, n_blocks, block // 4, 4)
    packed = (uq4[:,:,0] & 0x3) | ((uq4[:,:,1] & 0x3) << 2) | \
             ((uq4[:,:,2] & 0x3) << 4) | ((uq4[:,:,3] & 0x3) << 6)
    # packed shape: [rows, n_blocks, block//4]
    packed = packed.reshape(rows, cols // 4)

    return packed, sc

def quantize_and_save(tensor_map, shard_paths, name, out_prefix):
    """Load weight, quantize to INT2, save."""
    int2_path = os.path.join(OUT_DIR, f"{out_prefix}.int2_t")
    if os.path.exists(int2_path):
        print(f"  Skip {out_prefix} (exists)")
        return

    W = read_tensor(tensor_map, shard_paths, name).astype(np.float32)
    if W.ndim == 1:
        print(f"  Skip {out_prefix} (1D norm, copying)")
        # Copy FP32 norms directly
        import shutil
        int4_src = os.path.join("/mnt/data/ai/models/qwen3-8b-int4", f"{out_prefix}.f32")
        if os.path.exists(int4_src):
            shutil.copy2(int4_src, os.path.join(OUT_DIR, f"{out_prefix}.f32"))
        return

    rows, cols = W.shape
    print(f"  Quantizing {name} ({rows}x{cols})...", end=" ", flush=True)

    all_packed, all_scales = quantize_int2_matrix(W)

    with open(int2_path, 'wb') as f:
        f.write(struct.pack('5i', rows, cols, cols//4, BLOCK, 0))
        f.write(all_packed.tobytes())

    sc_path = os.path.join(OUT_DIR, f"{out_prefix}.scale_t")
    with open(sc_path, 'wb') as f:
        f.write(struct.pack('5i', rows, cols//BLOCK, 1, cols//BLOCK, rows))
        f.write(all_scales.tobytes())

    print(f"{os.path.getsize(int2_path)/1e6:.1f} MB")

if __name__ == "__main__":
    print("Building tensor map...")
    tensor_map, shard_paths = build_tensor_map(HF_PATH)
    print(f"Found {len(tensor_map)} tensors in {len(shard_paths)} shards")

    # Layer weight names
    layer_weights = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    for layer in range(36):
        for wn in layer_weights:
            name = f"model.layers.{layer}.self_attn.{wn}.weight" if wn in ["q_proj","k_proj","v_proj","o_proj"] else f"model.layers.{layer}.mlp.{wn}.weight"
            prefix = f"{layer}_{wn.replace('_proj','') if 'proj' in wn else wn}"
            # Fix prefix naming
            if wn in ["q_proj","k_proj","v_proj"]:
                prefix = f"{layer}_self_attn.{wn}"
            elif wn == "o_proj":
                prefix = f"{layer}_self_attn.o_proj"
            else:
                prefix = f"{layer}_mlp.{wn}"

            if name in tensor_map:
                quantize_and_save(tensor_map, shard_paths, name, prefix)

    # Embed + lm_head
    quantize_and_save(tensor_map, shard_paths, "model.embed_tokens.weight", "embed_tokens")
    quantize_and_save(tensor_map, shard_paths, "lm_head.weight", "lm_head")

    # Copy norms from INT4 directory
    import shutil
    int4_dir = "/mnt/data/ai/models/qwen3-8b-int4"
    for f in os.listdir(int4_dir):
        if f.endswith('.f32'):
            src = os.path.join(int4_dir, f)
            dst = os.path.join(OUT_DIR, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"  Copied {f}")

    total = sum(os.path.getsize(os.path.join(OUT_DIR, f)) for f in os.listdir(OUT_DIR))
    print(f"\nDone. Total: {total/1e9:.2f} GB ({len(os.listdir(OUT_DIR))} files)")
