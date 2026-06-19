#!/usr/bin/env python3
"""AWQ-style calibration for Qwen3-14B INT4 symmetric quantization.

Fork of quantize_awq_int4_8b.py adapted for 14B architecture:
  H=5120, I=17408, NL=40, nqh=40, nkv=8, hd=128, V=151936

Output: weights_int4_qwen3_14b_awq/ with FP32 scales (then convert to FP16
  via scripts/convert_scales_fp16.py for throughput path).

Usage:
    python3 scripts/quantize_awq_int4_14b.py [output_dir] [n_calib]
"""
import struct, json, os, sys, re, math, copy
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────
HF_PATH = "/mnt/data/ai/hf/models--Qwen--Qwen3-14B"
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "weights_int4_qwen3_14b_awq"
N_CALIB = int(sys.argv[2]) if len(sys.argv) > 2 else 128  # calibration prompts
ALPHA = 0.3        # AWQ scaling strength (0 = off, 0.5 = standard, 1.0 = strong)
BLOCK = 16         # quantization block size (must match kernel)

# Source weight dir for norm files (already quantized)
SRC_WEIGHT_DIR = "weights_int4_qwen3_14b_fp16sc"

# Calibration prompts
CALIB_PROMPTS = [
    "The capital of France is",
    "The theory of relativity was developed by",
    "In the beginning, God created the heavens and the earth",
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "The Roman Empire fell in the year 476 AD",
    "Water is composed of two hydrogen atoms and one oxygen atom",
    "The Industrial Revolution began in Great Britain",
    "Quantum mechanics describes the behavior of particles at the atomic scale",
    "The human genome project was completed in 2003",
    "The Great Wall of China is over 13,000 miles long",
    "DNA is the hereditary material in humans and almost all other organisms",
    "The speed of light in vacuum is approximately 299,792,458 meters per second",
    "Photosynthesis is the process by which plants convert light into energy",
    "The Amazon rainforest produces approximately twenty percent of the world's oxygen",
    "The Pythagorean theorem states that a squared plus b squared equals c squared",
    "The French Revolution began in 1789 with the storming of the Bastille",
    "Electricity is the flow of electric charge through a conductor",
    "The solar system consists of the Sun and eight planets",
    "Charles Darwin proposed the theory of evolution by natural selection",
    "The United Nations was established in 1945 after World War Two",
    "The periodic table organizes chemical elements by atomic number",
    "Shakespeare wrote thirty seven plays and over one hundred and fifty sonnets",
    "The internet is a global network of interconnected computers",
    "Plate tectonics explains the movement of Earth's lithospheric plates",
    "The Renaissance was a period of cultural revival in Europe",
    "Gravity is the force that attracts objects with mass toward each other",
    "The mitochondria is often called the powerhouse of the cell",
    "Cryptography is the practice of secure communication",
    "The Alamo is a historic Spanish mission in San Antonio Texas",
    "Neural networks are computing systems inspired by biological brains",
    "The Cold War was a period of geopolitical tension between the United States and the Soviet Union",
    "Algebra is a branch of mathematics dealing with symbols and equations",
    "The human brain contains approximately eighty six billion neurons",
    "The Eiffel Tower was completed in 1889 as the entrance to the World's Fair",
]

def find_model_shards(model_dir):
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
            cp = os.path.join(snapshot_dir, snap, "model.safetensors")
            if os.path.exists(cp):
                return [cp]
    # Try the canonical symlink
    link_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(link_path):
        print(f"  Found index: {link_path}")
        with open(link_path) as f:
            idx = json.load(f)
        weight_map = idx.get("weight_map", {})
        dir_name = os.path.dirname(link_path)
        shards = sorted(set(os.path.join(dir_name, v) for v in weight_map.values()))
        return shards
    return []

def load_safetensor_headers(shard_paths):
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
        f.seek(0)
        hdr_len_shard = struct.unpack('Q', f.read(8))[0]
        f.seek(8 + hdr_len_shard + start)
        raw = f.read(end - start)
    if info['dtype'] == 'BF16':
        return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(info['shape'])
    return np.frombuffer(raw, dtype=np.float32 if info['dtype'] == 'F32' else np.float16).reshape(info['shape'])

# ── Activation collector (random normal proxy) ────────────────────────────
def collect_activation_stats_random(tensor_map, shard_paths, config, n_seqs):
    """Random normal proxy activation stats. Layer-0 only, replicated to all layers."""
    H = config["hidden_size"]
    I = config.get("intermediate_size", H * 4)
    NL = config["num_hidden_layers"]
    V = config.get("vocab_size", 151936)

    WEIGHT_NAMES = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]

    print(f"  Loading layer-0 weights...")
    l0_weights = {}
    for wn in WEIGHT_NAMES:
        tname = f"model.layers.0.{wn}.weight"
        if tname in tensor_map:
            l0_weights[wn] = read_tensor(tensor_map, shard_paths, tname).astype(np.float32)

    print(f"  Running {n_seqs} random normal inputs...")
    np.random.seed(42)

    wn_act_mag = {}
    for wn in WEIGHT_NAMES:
        if wn in l0_weights:
            N_out, _ = l0_weights[wn].shape
            wn_act_mag[wn] = np.zeros(N_out, dtype=np.float64)

    for seq_idx in range(n_seqs):
        x = np.random.randn(H).astype(np.float32)
        for wn in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]:
            vals = x @ l0_weights[wn].T
            wn_act_mag[wn] += np.abs(vals)
        for wn in ["mlp.gate_proj", "mlp.up_proj"]:
            vals = x @ l0_weights[wn].T
            wn_act_mag[wn] += np.abs(vals)
        x_inter = np.random.randn(I).astype(np.float32)
        vals = x_inter @ l0_weights["mlp.down_proj"].T
        wn_act_mag["mlp.down_proj"] += np.abs(vals)

    for wn in wn_act_mag:
        wn_act_mag[wn] = wn_act_mag[wn] / n_seqs

    act_stats = {}
    for l in range(NL):
        for wn in WEIGHT_NAMES:
            if wn in wn_act_mag:
                act_stats[f"{l}_{wn}"] = wn_act_mag[wn].copy()

    # lm_head
    lm_tname = "model.language_model.output.weight"
    if lm_tname not in tensor_map:
        lm_tname = "lm_head.weight"
    if lm_tname in tensor_map:
        N_out, _ = tensor_map[lm_tname][1]['shape']
        act_stats["lm_head"] = np.ones(N_out, dtype=np.float64)

    return act_stats, [], H, I, V, NL

# ── AWQ scale computation ─────────────────────────────────────────────────
def compute_awq_scales(W_f32, act_mag, alpha=0.5):
    N, K = W_f32.shape
    if act_mag is None or len(act_mag) != N:
        return np.ones(N, dtype=np.float32)
    mean_act = act_mag.mean()
    if mean_act < 1e-10:
        return np.ones(N, dtype=np.float32)
    act_mag_f64 = act_mag.astype(np.float64)
    mean_act_f64 = float(mean_act)
    s_f64 = np.clip((act_mag_f64 / mean_act_f64) ** alpha, 0.5, 2.0)
    return s_f64.astype(np.float32)

def pseudo_quantize_int4_sym(W_f32, block=16):
    N, K = W_f32.shape
    assert K % block == 0
    num_blks = K // block
    W_blk = W_f32.reshape(N, num_blks, block)
    blk_abs = np.max(np.abs(W_blk), axis=2)
    blk_abs = np.maximum(blk_abs, 1e-10)
    q = np.round(W_blk / (blk_abs / 7.0)[:, :, np.newaxis])
    q = np.clip(q, -7, 7)
    W_q = q * (blk_abs / 7.0)[:, :, np.newaxis]
    return W_q.reshape(N, K)

def search_best_alpha(W_f32, act_mag, x_max_pow=None, n_grid=20):
    N, K = W_f32.shape
    if act_mag is None or len(act_mag) != N:
        return 0.0, compute_awq_scales(W_f32, act_mag, 0.0)
    x_max = act_mag.astype(np.float64)
    mean_act = float(x_max.mean())
    if mean_act < 1e-10:
        return 0.0, compute_awq_scales(W_f32, act_mag, 0.0)
    best_ratio = 0.0
    best_mse = float('inf')
    best_scales = None
    for ri in range(n_grid):
        ratio = ri / n_grid
        scales_f64 = np.power(x_max / mean_act, ratio)
        scales_f64 = np.clip(scales_f64, 1e-4, None)
        scales_f64 = scales_f64 / np.sqrt(scales_f64.max() * scales_f64.min())
        scales = scales_f64.astype(np.float32)[:, np.newaxis]
        W_scaled = W_f32.astype(np.float64) / scales
        W_q = pseudo_quantize_int4_sym(W_scaled.astype(np.float32), 16)
        W_recon = W_q.astype(np.float64) * scales
        mse = np.mean((W_f32.astype(np.float64) - W_recon) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_ratio = ratio
            best_scales = scales_f64.astype(np.float32)
    s = np.clip(best_scales, 0.5, 2.0)
    return best_ratio, s

# ── INT4 symmetric quantization (with AWQ scales) ─────────────────────────
def quantize_int4_sym_awq(W_f32, awq_scale, block=16):
    """INT4 symmetric block quantization with AWQ per-channel pre-scaling."""
    N, K = W_f32.shape
    assert K % block == 0
    num_blks = K // block
    W_scaled = W_f32 / awq_scale[:, np.newaxis]
    W_blk = W_scaled.reshape(N, num_blks, block)
    blk_abs = np.max(np.abs(W_blk), axis=2)
    scales = np.maximum(blk_abs, 1e-10) / 7.0  # [N, num_blks]
    # Fold AWQ scale into block scales
    scales = scales * awq_scale[:, np.newaxis]
    q = np.round(W_blk / (blk_abs / 7.0)[:, :, np.newaxis])
    q = np.clip(q, -7, 7).astype(np.int32)
    scales = (blk_abs / 7.0 * awq_scale[:, np.newaxis]).astype(np.float32)
    q_shifted = (q + 8).astype(np.uint8).reshape(N, K)
    q_reshaped = q_shifted.reshape(N, K // 2, 2)
    packed = (q_reshaped[:, :, 0] & 0x0F) | ((q_reshaped[:, :, 1] & 0x0F) << 4)
    return packed, scales

def write_weight_int4_sym(prefix, packed, scales, K_in, N_out):
    """Write INT4 symmetric weights in kernel-compatible format (FP16 scales)."""
    num_kb = K_in // BLOCK
    header = np.array([K_in, N_out, BLOCK, num_kb, 1], dtype=np.int32)
    with open(f"{prefix}.int4_t", 'wb') as f:
        f.write(header.tobytes())
        f.write(packed.tobytes())
    header_sc = np.array([0, 0, 0, num_kb, N_out], dtype=np.int32)
    with open(f"{prefix}.scale_t", 'wb') as f:
        f.write(header_sc.tobytes())
        f.write(scales.astype(np.float16).tobytes())
    mb = (packed.nbytes + scales.nbytes) / (1024*1024)
    short = prefix.split('/')[-1]
    print(f"  AWQ: {short}: {N_out}×{K_in} {mb:.1f}MB")

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    global ALPHA, N_CALIB, OUT_DIR

    print("=" * 60)
    print("AWQ INT4 Calibration — Qwen3-14B")
    print("=" * 60)

    with open(os.path.join(HF_PATH, "config.json")) as f:
        config = json.load(f)
    NL_cfg = config['num_hidden_layers']
    H_cfg = config['hidden_size']
    print(f"Config: {NL_cfg}L, H={H_cfg}, I={config.get('intermediate_size', '?')}")

    # 1. Find shards
    shard_paths = find_model_shards(HF_PATH)
    if not shard_paths:
        print("ERROR: No safetensor files found")
        sys.exit(1)
    print(f"Found {len(shard_paths)} shard(s)")

    tensor_map = load_safetensor_headers(shard_paths)

    # 2. Collect activation stats (random normal proxy)
    act_stats, key_names, H, I, V, NL = \
        collect_activation_stats_random(tensor_map, shard_paths, config, N_CALIB)

    print(f"\nActivation stats collected for {len(act_stats)} weight tensors")
    for k, v in list(act_stats.items())[:5]:
        if v is not None:
            print(f"  {k}: mean_act={v.mean():.4f}, max_act={v.max():.4f}")

    # 3. Create output dir
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\nOutput: {OUT_DIR}/")

    # 4. Process each layer's weights
    # Strategy: group weights by shard, read one shard at a time, process all its layers
    WEIGHT_NAMES = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]
    alpha_log = []

    # Build {shard_idx: {tensor_name: info}} reverse map
    shard_tensors = {i: {} for i in range(len(shard_paths))}
    for tname, (si, info) in tensor_map.items():
        shard_tensors[si][tname] = info

    print("  Processing layers by shard (streaming)...")
    for si, shard_path in enumerate(shard_paths):
        tensors_in_shard = shard_tensors[si]
        if not tensors_in_shard:
            continue
        
        # Read each tensor from the file directly (no mmap)
        # Group by layer first
        layer_weights = {}  # {li: {short_wn: W}}
        
        for tname, info in tensors_in_shard.items():
            if not tname.startswith("model.layers."):
                continue
            parts = tname.split(".")
            li = int(parts[2])
            weight_part = ".".join(parts[3:]).replace(".weight", "")
            for prefix, short_wn in [(wn, wn.split('.')[1] if '.' in wn else wn) for wn in WEIGHT_NAMES]:
                if weight_part == prefix:
                    # Read this single tensor using raw file seek+read
                    start, end = info['data_offsets']
                    f = open(shard_path, 'rb')
                    hdr_len = struct.unpack('Q', f.read(8))[0]
                    f.seek(8 + hdr_len + start)
                    raw = f.read(end - start)
                    f.close()
                    dtype = info['dtype']
                    shape = info['shape']
                    if dtype == 'BF16':
                        W = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
                    elif dtype == 'F32':
                        W = np.frombuffer(raw, dtype=np.float32).reshape(shape)
                    else:
                        W = np.frombuffer(raw, dtype=np.float16).reshape(shape)
                    layer_weights.setdefault(li, {})[short_wn] = W
                    break

        for li in sorted(layer_weights.keys()):
            for wn in WEIGHT_NAMES:
                short_wn = wn.split('.')[1] if '.' in wn else wn
                if short_wn not in layer_weights[li]:
                    continue
                W = layer_weights[li][short_wn]
                N_out, K_in = W.shape
                k = f"{li}_{wn}"
                act_mag = act_stats.get(k, None)

                if act_mag is not None and N_CALIB > 0:
                    awq_sc = compute_awq_scales(W, act_mag, 0.4)
                    best_ratio = 0.4
                else:
                    awq_sc = compute_awq_scales(W, act_mag, 0.4) if act_mag is not None else np.ones(N_out)
                    best_ratio = ALPHA

                packed, scales = quantize_int4_sym_awq(W, awq_sc, BLOCK)
                prefix = f"{OUT_DIR}/{li}_{short_wn}"
                write_weight_int4_sym(prefix, packed, scales, K_in, N_out)
                alpha_log.append((li, short_wn, best_ratio))
                del W

        import gc; gc.collect()
        print(f"  Shard {si+1}/{len(shard_paths)} done ({len(layer_weights)} layers)")

    # Print alpha distribution
    ratios = [a[2] for a in alpha_log]
    if ratios:
        print(f"\nAlpha distribution across {len(ratios)} (layer, submodule) pairs:")
        print(f"  mean={np.mean(ratios):.3f}, min={np.min(ratios):.3f}, max={np.max(ratios):.3f}")

    # 5. Embed tokens
    W_emb = read_tensor(tensor_map, shard_paths, "model.embed_tokens.weight").astype(np.float32)
    N_out, K_in = W_emb.shape
    act_mag = act_stats.get("embed_tokens", None)
    if act_mag is not None and N_CALIB > 0:
        awq_sc = compute_awq_scales(W_emb, act_mag, 0.3)
        print(f"  embed_tokens: alpha=0.3")
    else:
        awq_sc = compute_awq_scales(W_emb, act_mag, ALPHA) if act_mag is not None else np.ones(N_out)
    packed, scales = quantize_int4_sym_awq(W_emb, awq_sc, BLOCK)
    write_weight_int4_sym(f"{OUT_DIR}/embed_tokens", packed, scales, K_in, N_out)
    print(f"  embed_tokens: {N_out}×{K_in}")

    # 6. LM head
    lm_tname = "lm_head.weight"
    if lm_tname in tensor_map:
        W_lm = read_tensor(tensor_map, shard_paths, lm_tname).astype(np.float32)
        N_out, K_in = W_lm.shape
        act_mag = act_stats.get("lm_head", None)
        if act_mag is not None and N_CALIB > 0:
            awq_sc = compute_awq_scales(W_lm, act_mag, 0.3)
            print(f"  lm_head: alpha=0.3")
        else:
            awq_sc = compute_awq_scales(W_lm, act_mag, ALPHA) if act_mag is not None else np.ones(N_out)
        packed, scales = quantize_int4_sym_awq(W_lm, awq_sc, BLOCK)
        write_weight_int4_sym(f"{OUT_DIR}/lm_head", packed, scales, K_in, N_out)
        print(f"  lm_head: {N_out}×{K_in}")
    else:
        print(f"  SKIP lm_head (not found in tensor_map)")

    # 7. Copy norm files from existing quantized weight dir
    print(f"\nCopying norm files from {SRC_WEIGHT_DIR}/...")
    norm_count = 0
    for fn in os.listdir(SRC_WEIGHT_DIR):
        if fn.endswith('.f32') or fn == 'qk_norms.f32':
            src = os.path.join(SRC_WEIGHT_DIR, fn)
            dst = os.path.join(OUT_DIR, fn)
            if not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)
                norm_count += 1
    print(f"  Copied {norm_count} norm files")

    print(f"\nDone. AWQ-calibrated INT4 weights in {OUT_DIR}/")
    print(f"  N_calib={N_CALIB}, alpha={ALPHA}")

    print(f"\nTo convert to FP16 scales (for throughput):")
    print(f"  python3 scripts/convert_scales_fp16.py {OUT_DIR} {OUT_DIR}_fp16sc")

    print(f"\nTo benchmark:")
    print(f"  ./bench/text_generate_int4_qwen3_14b -w {OUT_DIR} 'prompt' 30")


if __name__ == "__main__":
    main()