#!/usr/bin/env python3
"""INT4 quantization for Qwen3-30B-A3B MoE model.

Streaming approach: load one safetensor shard at a time, quantize all weights,
write INT4 files, free memory. This avoids OOM from loading entire 57 GB model.

Architecture: 48L, H=2048, nqh=32, nkv=4, hd=128
  MoE: 128 experts × moe_intermediate_size=768, top-8
  No shared dense MLP — only MoE experts

Usage:
    python3 scripts/quantize_int4_moe_30b.py [output_dir]
"""
import struct, json, os, sys
import numpy as np

HF_PATH = "/mnt/data/huggingface/Qwen3-30B-A3B"
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "weights_int4_qwen3_30b_a3b"
BLOCK = 16

def get_shard_paths(model_dir):
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        return [single]
    return sorted([
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.startswith("model-") and f.endswith(".safetensors")
    ])

def load_shard_headers(shard_paths):
    """Load tensor → (shard_idx, dtype, shape, start, end) for all shards"""
    tensor_info = {}
    for shard_idx, sp in enumerate(shard_paths):
        with open(sp, 'rb') as f:
            hdr_len = struct.unpack('Q', f.read(8))[0]
            hdr = json.loads(f.read(hdr_len))
        for name, info in hdr.items():
            if name == '__metadata__':
                continue
            s, e = info['data_offsets']
            tensor_info[name] = (shard_idx, info['dtype'], info['shape'], s, e)
    return tensor_info

def load_shard(shard_path, tensor_names):
    """Load specific tensors from a shard, return {name: np.array (float32)}"""
    result = {}
    with open(shard_path, 'rb') as f:
        hdr_len = struct.unpack('Q', f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
        for name in tensor_names:
            if name not in hdr:
                continue
            info = hdr[name]
            s, e = info['data_offsets']
            dtype_str = info['dtype']
            f.seek(s, os.SEEK_SET)
            raw = f.read(e - s)
            if dtype_str == 'BF16':
                u16 = np.frombuffer(raw, dtype=np.uint16).copy()
                result[name] = (u16.astype(np.uint32) << 16).view(np.float32).reshape(info['shape'])
            else:
                result[name] = np.frombuffer(raw, dtype=dtype_str).reshape(info['shape'])
    return result

def quantize_int4_sym(data):
    """Vectorized symmetric INT4 quantization"""
    N, K = data.shape
    num_kb = K // BLOCK
    # Reshape to [N, num_kb, BLOCK], compute absmax per block
    blocks = data.reshape(N, num_kb, BLOCK)
    absmax = np.max(np.abs(blocks), axis=2)  # [N, num_kb]
    scales = np.maximum(absmax, 1e-10) / 7.0  # [N, num_kb]
    scales[~np.isfinite(scales)] = 1.0
    # Quantize: qi = round(clip(w / scale, -7, 7))
    qi = np.clip(np.round(blocks / scales[:, :, None]), -7, 7).astype(np.int32)
    # Pack: even indices -> lo nibble, odd -> hi nibble
    qi_even = qi[:, :, 0::2]  # [N, num_kb, 8]
    qi_odd  = qi[:, :, 1::2]  # [N, num_kb, 8]
    packed = ((qi_odd + 8) << 4) | (qi_even + 8)  # [N, num_kb, 8]
    packed = packed.astype(np.uint8).reshape(N, K // 2)
    return packed, scales

def write_weight(prefix, packed, scales, K, N):
    num_kb = K // BLOCK
    with open(f"{prefix}.int4_t", "wb") as f:
        f.write(struct.pack('iiiii', K, N, BLOCK, num_kb, 1))
        f.write(packed.tobytes())
    with open(f"{prefix}.scale_t", "wb") as f:
        f.write(struct.pack('iiiii', 0, 0, 0, num_kb, N))
        # Clamp scales to FP16 max (65504) to prevent overflow to inf
        clipped = np.clip(scales.flatten(), -65504.0, 65504.0)
        hscales = clipped.astype(np.float16)
        f.write(hscales.tobytes())
    mb = (packed.nbytes + hscales.nbytes) / (1024*1024)
    print(f"  {prefix.split('/')[-1]}: {N}×{K} INT4 {mb:.1f}MB")

def write_f32(prefix, data):
    with open(f"{prefix}.f32", "wb") as f:
        f.write(data.astype(np.float32).tobytes())
    print(f"  {prefix.split('/')[-1]}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    with open(os.path.join(HF_PATH, "config.json")) as f:
        cfg = json.load(f)
    NL = cfg['num_hidden_layers']
    H = cfg['hidden_size']
    nqh = cfg['num_attention_heads']
    nkv = cfg['num_key_value_heads']
    hd = cfg['head_dim']
    V = cfg['vocab_size']
    moe_im = cfg['moe_intermediate_size']
    num_experts = cfg['num_experts']
    num_experts_per_tok = cfg['num_experts_per_tok']
    print(f"Config: {NL}L, H={H}, nqh={nqh}, nkv={nkv}, hd={hd}, V={V}")
    print(f"  MoE: {num_experts} experts, top-{num_experts_per_tok}, intermediate={moe_im}")

    shard_paths = get_shard_paths(HF_PATH)
    print(f"Shards: {len(shard_paths)}")
    tensor_info = load_shard_headers(shard_paths)
    print(f"Total tensors: {len(tensor_info)}")

    # Group tensors by shard for efficient loading
    by_shard = {}
    for name, (shard_idx, _, _, _, _) in tensor_info.items():
        by_shard.setdefault(shard_idx, []).append(name)

    # Track writes to avoid over-writing when same tensor name appears
    # (shouldn't happen but be safe)
    written = set()

    for s_idx, sp in enumerate(shard_paths):
        print(f"\n── Shard {s_idx+1}/{len(shard_paths)}: {os.path.basename(sp)} ──")
        names_in_shard = by_shard.get(s_idx, [])
        if not names_in_shard:
            continue
        tensors = load_shard(sp, names_in_shard)
        print(f"  Loaded {len(tensors)} tensors")

        for name, W in tensors.items():
            if name in written:
                continue

            # Parse the tensor name to determine output path
            # model.layers.{l}.{component}.weight
            parts = name.split('.')
            if len(parts) >= 3 and parts[0] == 'model' and parts[1] == 'layers':
                l = int(parts[2])
                rest = '.'.join(parts[3:])
                # rest is like: self_attn.q_proj.weight, mlp.experts.0.gate_proj.weight, etc.
                # Strip .weight
                if rest.endswith('.weight'):
                    rest = rest[:-7]  # remove .weight

                # Map to blackwell naming
                if rest == 'input_layernorm':
                    write_f32(f"{OUT_DIR}/{l}_input_layernorm", W)
                elif rest == 'post_attention_layernorm':
                    write_f32(f"{OUT_DIR}/{l}_post_attention_layernorm", W)
                elif rest.startswith('self_attn.'):
                    sub = rest[len('self_attn.'):]
                    if sub in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                        packed, scales = quantize_int4_sym(W)
                        write_weight(f"{OUT_DIR}/{l}_self_attn.{sub}", packed, scales, W.shape[1], W.shape[0])
                    elif sub in ('q_norm', 'k_norm'):
                        write_f32(f"{OUT_DIR}/{l}_self_attn.{sub}", W)
                elif rest.startswith('mlp.'):
                    # mlp.gate -> router, mlp.experts.{e}.{proj} -> expert weights
                    if rest == 'mlp.gate':
                        packed, scales = quantize_int4_sym(W)
                        write_weight(f"{OUT_DIR}/{l}_mlp.gate", packed, scales, W.shape[1], W.shape[0])
                    elif 'experts' in rest:
                        # mlp.experts.{e}.{proj}
                        e_parts = rest.split('.')
                        e = int(e_parts[2])
                        proj = e_parts[3]  # gate_proj, up_proj, down_proj
                        packed, scales = quantize_int4_sym(W)
                        write_weight(f"{OUT_DIR}/{l}_mlp.experts.{e}.{proj}", packed, scales, W.shape[1], W.shape[0])

                written.add(name)
            
            elif name == 'model.norm.weight':
                write_f32(f"{OUT_DIR}/final_norm", W)
                written.add(name)
            elif name == 'model.embed_tokens.weight':
                packed, scales = quantize_int4_sym(W)
                write_weight(f"{OUT_DIR}/embed_tokens", packed, scales, W.shape[1], W.shape[0])
                written.add(name)
            elif name == 'lm_head.weight':
                packed, scales = quantize_int4_sym(W)
                write_weight(f"{OUT_DIR}/lm_head", packed, scales, W.shape[1], W.shape[0])
                written.add(name)
            else:
                print(f"  SKIP: {name}")

        # Free tensors
        del tensors

    # Count results
    int4_files = [f for f in os.listdir(OUT_DIR) if f.endswith('.int4_t')]
    scale_files = [f for f in os.listdir(OUT_DIR) if f.endswith('.scale_t')]
    f32_files = [f for f in os.listdir(OUT_DIR) if f.endswith('.f32')]
    total_size = sum(
        os.path.getsize(os.path.join(OUT_DIR, f))
        for f in os.listdir(OUT_DIR)
    )
    print(f"\nDone. {len(int4_files)} weight files, {len(f32_files)} norm files")
    print(f"  Total size: {total_size / 1024**3:.1f} GB")
    print(f"  Output: {OUT_DIR}/")

if __name__ == "__main__":
    main()
