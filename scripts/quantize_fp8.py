#!/usr/bin/env python3
"""FP8 E4M3 quantization for Qwen3 models.

Per-element FP8 E4M3 quantization (no block scales needed).
Handles single-file and multi-shard safetensors.

Usage:
    python3 scripts/quantize_fp8.py <model_path> <output_dir>

Example:
    python3 scripts/quantize_fp8.py /mnt/data/ai/hf/qwen3-1.7b-base weights_fp8_bf16
"""
import struct, json, os, sys
import numpy as np


def float_to_fp8_e4m3(x):
    """Convert float32 array to FP8 E4M3 (uint8).

    E4M3: sign(1) + exp(4) + mantissa(3), bias=7
    Normal range: [-448, 448], no Inf/NaN
    Max value: 2^8 * (1 + 7/8) = 448
    """
    x = np.asarray(x, dtype=np.float32)
    sign = np.signbit(x).astype(np.uint8)
    ax = np.abs(x)
    result = np.zeros(x.shape, dtype=np.uint8)

    nonzero = ax > 0
    if not np.any(nonzero):
        return result

    ax_nz = ax[nonzero].astype(np.float64)  # float64 for precision

    # Clamp to FP8 E4M3 max
    ax_nz = np.minimum(ax_nz, 448.0)

    # Compute biased exponent: floor(log2(x)) + 7
    exp_unbiased = np.floor(np.log2(ax_nz)).astype(np.int32)
    exp_biased = exp_unbiased + 7

    # Compute mantissa
    # Normal (e > 0): value = 2^(e-7) * (1 + m/8), m = round(x / 2^(e-7) * 8 - 8)
    # Subnormal (e = 0): value = 2^(-6) * (m/8), m = round(x * 512)
    is_normal = exp_biased > 0

    mantissa = np.zeros_like(ax_nz, dtype=np.int32)
    if np.any(is_normal):
        scale_n = np.power(2.0, (7 - exp_biased[is_normal]).astype(np.float64))
        mantissa[is_normal] = np.round(ax_nz[is_normal] * scale_n * 8.0 - 8.0).astype(np.int32)
    if np.any(~is_normal):
        # Subnormal: m = round(x * 2^9) = round(x * 512)
        mantissa[~is_normal] = np.clip(np.round(ax_nz[~is_normal] * 512.0).astype(np.int32), 0, 7)
        exp_biased[~is_normal] = 0

    # Handle overflow (mantissa rounded to 8)
    overflow = mantissa >= 8
    if np.any(overflow):
        exp_biased[overflow] += 1
        mantissa[overflow] = 0

    # Clamp
    exp_biased = np.clip(exp_biased, 0, 15)
    mantissa = np.clip(mantissa, 0, 7)

    # Pack: sign(7) | exp(3) | mantissa(0)
    packed = (sign[nonzero].astype(np.uint8) << 7) | \
             (exp_biased.astype(np.uint8) << 3) | \
             mantissa.astype(np.uint8)
    result[nonzero] = packed
    return result


def fp8_e4m3_to_float(fp8_bytes):
    """Dequantize FP8 E4M3 bytes back to float32 (for verification)."""
    b = fp8_bytes.astype(np.uint32)
    sign = (b >> 7) & 1
    exp = (b >> 3) & 0xF
    mant = b & 0x7

    is_normal = exp > 0
    value = np.where(
        is_normal,
        (1.0 + mant.astype(np.float64) / 8.0) * np.power(2.0, exp.astype(np.float64) - 7.0),
        (mant.astype(np.float64) / 8.0) * np.power(2.0, -6.0)
    )

    return np.where(sign, -value, value).astype(np.float32)


def quantize_per_row_fp8(W_f32):
    """FP8 E4M3 per-row scaling. One scale per output row."""
    N, K = W_f32.shape
    # Per-row scale: max abs / 448
    row_max = np.max(np.abs(W_f32), axis=1, keepdims=True)
    row_max = np.maximum(row_max, 1e-30)
    scales = (row_max / 448.0).squeeze().astype(np.float32)  # [N] floats

    # Scale to full FP8 range
    scale_bc = row_max / 448.0  # [N, 1]
    W_scaled = W_f32 / scale_bc
    fp8_data = float_to_fp8_e4m3(W_scaled)

    return fp8_data, scales


def write_weight_fp8(prefix, fp8_data, scales, K_in, N_out):
    """Write FP8 weight file + per-row scale file."""
    header = np.array([K_in, N_out, 1, N_out, N_out], dtype=np.int32)
    path = f"{prefix}.fp8_t"
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'wb') as f:
        f.write(header.tobytes())
        f.write(fp8_data.tobytes())
    path = f"{prefix}.scale_t"
    with open(path, 'wb') as f:
        f.write(header.tobytes())
        f.write(scales.tobytes())
    mb_data = fp8_data.nbytes / (1024 * 1024)
    mb_sc = scales.nbytes / (1024 * 1024)
    print(f"  {prefix}: [{N_out}×{K_in}] FP8 data={mb_data:.1f}MB scales={mb_sc:.1f}MB")


def find_model_shards(model_dir):
    """Find safetensor shards in model directory."""
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
    """Load headers from all shards."""
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
    """Read a tensor from safetensors, returning float32."""
    shard_idx, info = tensor_map[name]
    shard_path = shard_paths[shard_idx]
    start, end = info['data_offsets']
    with open(shard_path, 'rb') as f:
        hdr_len = struct.unpack('Q', f.read(8))[0]
        f.seek(8 + hdr_len + start)
        raw = f.read(end - start)

    dtype_str = info.get('dtype', 'BF16')
    shape = info['shape']

    if dtype_str == 'BF16':
        u16 = np.frombuffer(raw, dtype=np.uint16).copy()
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        return f32.reshape(shape)
    elif dtype_str == 'F16':
        import torch  # fallback for F16
        u16 = np.frombuffer(raw, dtype=np.uint16).copy()
        return u16.reshape(shape).astype(np.float32)
    else:
        np_map = {'F32': np.float32, 'F64': np.float64, 'U8': np.uint8,
                   'I8': np.int8, 'I32': np.int32, 'I64': np.int64}
        dt = np_map.get(dtype_str, np.float32)
        return np.frombuffer(raw, dtype=dt).reshape(shape).copy()


def verify_quantization(W_f32, fp8_data, scales, name):
    """Spot-check quantization error with per-row scaling."""
    N, K = W_f32.shape
    scale_bc = scales[:, np.newaxis]  # [N, 1]
    W_deq = fp8_e4m3_to_float(fp8_data).reshape(N, K) * scale_bc
    diff = np.abs(W_f32 - W_deq)
    rel_err = diff / (np.abs(W_f32) + 1e-10)
    psnr = 10 * np.log10(np.max(np.abs(W_f32))**2 / (np.mean(diff**2) + 1e-30))
    print(f"    {name}: max_err={np.max(diff):.6f}, mean_rel_err={np.mean(rel_err):.4f}, PSNR={psnr:.1f}dB")
    return psnr


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/quantize_fp8.py <model_path> <output_dir>")
        print("Example: python3 scripts/quantize_fp8.py /mnt/data/ai/hf/qwen3-1.7b-base weights_fp8_bf16")
        sys.exit(1)

    MODEL = sys.argv[1]
    OUT = sys.argv[2]

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

    total_psnr = 0.0
    total_count = 0

    for layer in range(NL):
        for wn in WEIGHT_NAMES:
            tname = f"model.layers.{layer}.{wn}.weight"
            if tname not in tensor_map:
                print(f"  WARNING: {tname} not found, skipping")
                continue
            W = read_tensor(tensor_map, shard_paths, tname)
            N_out, K_in = W.shape
            fp8_data, scales = quantize_per_row_fp8(W)
            prefix = f"{OUT}/{layer}_{wn}"
            write_weight_fp8(prefix, fp8_data, scales, K_in, N_out)
            # Verify first layer only
            if layer == 0:
                psnr = verify_quantization(W, fp8_data, scales, wn)
                total_psnr += psnr
                total_count += 1
            del W, fp8_data, scales

    # Embed tokens
    print(f"\nProcessing embed_tokens...")
    W_emb = read_tensor(tensor_map, shard_paths, "model.embed_tokens.weight")
    N_out, K_in = W_emb.shape
    fp8_data, scales = quantize_per_row_fp8(W_emb)
    write_weight_fp8(f"{OUT}/embed_tokens", fp8_data, scales, K_in, N_out)
    psnr = verify_quantization(W_emb, fp8_data, scales, "embed_tokens")
    del W_emb, fp8_data, scales

    # lm_head (if separate)
    lm_name = "lm_head.weight"
    if lm_name in tensor_map:
        print(f"\nProcessing lm_head...")
        W_lm = read_tensor(tensor_map, shard_paths, lm_name)
        N_out, K_in = W_lm.shape
        fp8_data, scales = quantize_per_row_fp8(W_lm)
        write_weight_fp8(f"{OUT}/lm_head", fp8_data, scales, K_in, N_out)
        verify_quantization(W_lm, fp8_data, scales, "lm_head")
        del W_lm, fp8_data, scales

    # Copy norm files (stays FP32)
    print(f"\nCopying norm files...")
    for name in ["input_layernorm", "post_attention_layernorm"]:
        for layer in range(NL):
            tname = f"model.layers.{layer}.{name}.weight"
            if tname not in tensor_map:
                continue
            W = read_tensor(tensor_map, shard_paths, tname)
            prefix = f"{OUT}/{layer}_{name}"
            with open(f"{prefix}.f32", 'wb') as f:
                f.write(W.astype(np.float32).tobytes())
            del W

    # Final norm
    fn_name = "model.norm.weight"
    if fn_name in tensor_map:
        W = read_tensor(tensor_map, shard_paths, fn_name)
        with open(f"{OUT}/final_norm.f32", 'wb') as f:
            f.write(W.astype(np.float32).tobytes())
        del W

    print(f"\nDone. {NL} layers × {len(WEIGHT_NAMES)} weights + embed_tokens (FP8 E4M3)")
    if total_count > 0:
        print(f"Average PSNR (layer 0): {total_psnr / total_count:.1f} dB")


if __name__ == "__main__":
    main()
