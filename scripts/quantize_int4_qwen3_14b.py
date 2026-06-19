#!/usr/bin/env python3
"""Streaming INT4 quantization for Qwen3-14B.

Uses torch (mmap) to read BF16 safetensor shards correctly, then
converts each tensor to float32 and quantizes to INT4 block-16 symmetric.

Architecture (Qwen3-14B):
  H=5120, I=17408, NL=40, nqh=40, nkv=8, hd=128, V=151936
  SwiGLU MLP, RMSNorm, RoPE theta=1000000

Usage:
    python3 scripts/quantize_int4_qwen3_14b.py [output_dir]
"""
import struct, json, os, sys, gc
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────
HF_PATH = "/mnt/data/ai/hf/models--Qwen--Qwen3-14B"
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "weights_int4_qwen3_14b_fp16sc"
FP16_SCALES = True
BLOCK = 16

# ── Helpers ───────────────────────────────────────────────────────────────

def quantize_weight(w, block=16):
    """Quantize FP32 weight matrix [N_out, K_in] to INT4 block-16 symmetric.
    Returns (packed_uint8, scales_float16)."""
    N, K = w.shape
    num_kb = K // block
    assert K % block == 0, f"K={K} not divisible by {block}"

    w = w.astype(np.float32)
    w_blocks = w.reshape(N, num_kb, block)
    scales = np.abs(w_blocks).max(axis=2)  # [N, num_kb]
    scales[scales == 0] = 1e-8
    w_blocks_q = np.round(w_blocks / scales[:, :, None]).astype(np.int32)
    w_blocks_q = np.clip(w_blocks_q, -7, 7)

    # Per-block scales [N, num_kb] — kernel expects this format
    block_scales = scales.astype(np.float16)

    # Offset-binary encode and pack 2 int4→1 uint8 (nibble-8)
    q_shifted = (w_blocks_q + 8).astype(np.uint8).reshape(N, K)  # [N, K]
    q_reshaped = q_shifted.reshape(N, K // 2, 2)  # [N, K/2, 2]
    packed = (q_reshaped[:, :, 0] & 0x0F) | ((q_reshaped[:, :, 1] & 0x0F) << 4)  # [N, K/2]
    # packed.shape = (N, K/2)

    return packed, block_scales


def write_int4(prefix, packed, scales, K, N):
    """Write INT4 weight + FP16 scale files."""
    num_kb = K // BLOCK
    header = np.array([K, N, BLOCK, num_kb, 1], dtype=np.int32)
    with open(f"{prefix}.int4_t", 'wb') as f:
        f.write(header.tobytes())
        f.write(packed.tobytes())
    hsc = np.array([0, 0, 0, num_kb, N], dtype=np.int32)
    with open(f"{prefix}.scale_t", 'wb') as f:
        f.write(hsc.tobytes())
        f.write(scales.astype(np.float16).tobytes())


def write_f32_norm(prefix, w):
    """Write FP32 RMSNorm weight (1D)."""
    with open(f"{prefix}.f32", 'wb') as f:
        f.write(w.astype(np.float32).tobytes())


# ── Name mapping ───────────────────────────────────────────────────────────
# safetensor key → our weight file base name
WEIGHT_MAP = {
    "model.embed_tokens.weight": ("embed_tokens", "int4"),
    "lm_head.weight": ("lm_head", "int4"),
    "model.norm.weight": ("final_norm", "f32"),
}
LAYER_PREFIXES = [
    ("input_layernorm", "ln1"),
    ("post_attention_layernorm", "ln2"),
    ("self_attn.q_proj", "q_proj"),
    ("self_attn.k_proj", "k_proj"),
    ("self_attn.v_proj", "v_proj"),
    ("self_attn.o_proj", "o_proj"),
    ("mlp.gate_proj", "gate_proj"),
    ("mlp.up_proj", "up_proj"),
    ("mlp.down_proj", "down_proj"),
    ("self_attn.q_norm", "q_norm"),
    ("self_attn.k_norm", "k_norm"),
]
LAYER_TEMPLATE = "model.layers.{}.{}"


def build_layer_shard_map(shards):
    """Build {layer_idx: {weight_name: shard_path}} mapping."""
    result = {}  # {li: {short_name: fp}}
    for fp in shards:
        from safetensors.torch import load_file
        tensors = load_file(fp)  # mmap, no memory spike
        for key in tensors:
            if not key.startswith("model.layers."): continue
            parts = key.split(".")
            li = int(parts[2])
            weight_part = ".".join(parts[3:])
            # Strip .weight suffix
            if weight_part.endswith(".weight"):
                weight_part = weight_part[:-7]
            for prefix, short in LAYER_PREFIXES:
                if weight_part == prefix:
                    result.setdefault(li, {})[short] = fp
                    break
        del tensors
    return result



# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Streaming INT4 Quantization — Qwen3-14B (torch mmap)")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Collect shard paths
    shards = sorted([
        os.path.join(HF_PATH, f)
        for f in os.listdir(HF_PATH)
        if f.startswith("model-") and f.endswith(".safetensors")
    ])
    print(f"Shards: {len(shards)}")

    # Build layer → shard lookup (torch mmap each shard once)
    print("Building layer→shard map...")
    layer_map = build_layer_shard_map(shards)
    for li in range(40):
        n = len(layer_map.get(li, {}))
        if n != 11:
            print(f"  WARNING: layer {li} has {n} weights (expected 11)")

    # ── Process global weights ──
    print("\n=== Global weights ===")
    from safetensors.torch import load_file
    # Search all shards for global weights
    global_tensors = {}
    for fp in shards:
        t = load_file(fp)
        for key in WEIGHT_MAP:
            if key in t:
                global_tensors[key] = t[key]
        del t
    for key, (base, wtype) in WEIGHT_MAP.items():
        if key not in global_tensors:
            print(f"  SKIP {key} (not found)"); continue
        t = global_tensors[key].float()  # BF16 → F32
        w = t.cpu().numpy()
        del t
        print(f"  {base}: {w.shape}")
        if wtype == "int4":
            K, N = w.shape[1], w.shape[0]
            packed, scales = quantize_weight(w)
            write_int4(os.path.join(OUT_DIR, base), packed, scales, K, N)
            del packed, scales
        else:
            write_f32_norm(os.path.join(OUT_DIR, base), w)
        del w
        gc.collect()
    del global_tensors
    gc.collect()

    # ── Process layers ──
    print("\n=== Layers ===")
    for li in range(40):
        layer_shards = layer_map.get(li, {})
        needed = {short for _, short in LAYER_PREFIXES}
        found = set(layer_shards.keys())
        missing = needed - found
        if missing:
            print(f"Layer {li:2d}: MISSING {missing}")
            continue

        print(f"Layer {li:2d}: ", end="", flush=True)
        # Load needed shards for this layer (most layers are in 1 shard, some split)
        shard_fps = set(layer_shards.values())
        tensors = {}
        for fp in shard_fps:
            tensors.update(load_file(fp))

        for prefix, short in LAYER_PREFIXES:
            key = f"model.layers.{li}.{prefix}.weight"
            t = tensors[key].float()
            w = t.cpu().numpy()
            del t
            if w.ndim == 1:
                # Norm weight (1D) — write as FP32
                write_f32_norm(os.path.join(OUT_DIR, f"{li}_{short}"), w)
                print(f"{short}(norm) ", end="", flush=True)
            else:
                K, N = w.shape[1], w.shape[0]
                packed, scales = quantize_weight(w)
                write_int4(os.path.join(OUT_DIR, f"{li}_{short}"), packed, scales, K, N)
                del packed, scales
                print(f"{short} ", end="", flush=True)
            del w

        print("done")
        del tensors
        gc.collect()

    # ── Extract QK head norms ──
    print("\n=== QK head norms ===")
    # Qwen3-14B has per-layer Q/K head norms (hd=128 each)
    # Write as single qk_norms.f32: [40][2*128] = [40][256] floats
    NL_qk = 40; hd_qk = 128
    qk_norms = np.zeros((NL_qk, 2 * hd_qk), dtype=np.float32)
    for li in range(NL_qk):
        layer_shards = layer_map.get(li, {})
        if not layer_shards:
            print(f"  SKIP layer {li}: no shards")
            continue
        shard_fps = set(layer_shards.values())
        tensors = {}
        for fp in shard_fps:
            tensors.update(load_file(fp))
        q_key = f"model.layers.{li}.self_attn.q_norm.weight"
        k_key = f"model.layers.{li}.self_attn.k_norm.weight"
        if q_key in tensors:
            qn = tensors[q_key].float().cpu().numpy()
            qk_norms[li, :hd_qk] = qn
        if k_key in tensors:
            kn = tensors[k_key].float().cpu().numpy()
            qk_norms[li, hd_qk:] = kn
        del tensors
    qk_norms.tofile(os.path.join(OUT_DIR, "qk_norms.f32"))
    print(f"  qk_norms.f32: {qk_norms.shape} ({qk_norms.nbytes/1024:.0f} KB)")
    del qk_norms
    gc.collect()

    # ── Summary ──
    total_size = sum(os.path.getsize(os.path.join(OUT_DIR, f))
                     for f in os.listdir(OUT_DIR))
    print(f"\nDone. Output: {OUT_DIR}/ ({total_size/1e9:.1f} GB)")


if __name__ == "__main__":
    main()
