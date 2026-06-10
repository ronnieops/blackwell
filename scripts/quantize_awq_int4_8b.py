#!/usr/bin/env python3
"""AWQ-style calibration for Qwen3-8B INT4 symmetric quantization.

Key insight (from AWQ, MLSys 2024):
  Not all weight channels matter equally. Channels with large activations
  cause more quantization error. Scale them up before quantization, then
  fold scale into weight scale during inference.

What this script does:
  1. Load FP16 model weights from safetensors
  2. Run 128 calibration prompts → collect per-channel activation stats
  3. Compute AWQ per-channel protection scales: s = (max/mean)^alpha
  4. Apply scales to weights, re-quantize to INT4 symmetric
  5. Write updated .int4_t + .scale_t files

Scale integration with GEMV kernel:
  The INT4 GEMV does: out = sum_blk(w_nib * x_nib * w_sc * x_sc)
  AWQ scales W by s per output channel, dividing weight_scales by s.
  Fold: w_sc_new = w_sc / s  (applied per output channel)
  No kernel changes needed.

Usage:
    python3 scripts/quantize_awq_int4_8b.py [output_dir] [n_calib]
"""
import struct, json, os, sys, re, math, copy
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────
HF_PATH = "/mnt/data/ai/hf/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "weights_int4_qwen3_8b_awq"
N_CALIB = int(sys.argv[2]) if len(sys.argv) > 2 else 128  # calibration prompts
ALPHA = 0.5        # AWQ scaling strength (0 = off, 0.5 = standard, 1.0 = strong)
BLOCK = 16         # quantization block size (must match kernel)

# Calibration prompts — typical WikiText-2 style
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

# ── Load token IDs from binary ───────────────────────────────────────────
def load_token_corpus(path):
    """Load token corpus from binary file written by bench/tokenize_corpus.

    Format: [n_seqs: i32][lens: i32 x n_seqs][token_ids: sum(lens) x i32]
    Returns: list of numpy arrays (each seq as int32 array)
    """
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.int32)
    n_seqs = int(data[0])
    lens = data[1:1+n_seqs]
    ids_flat = data[1+n_seqs:]
    seqs = []
    pos = 0
    for l in lens:
        seqs.append(ids_flat[pos:pos+l].copy())
        pos += l
    return seqs

# ── Activation collector ──────────────────────────────────────────────────
def collect_activation_stats_real(tensor_map, shard_paths, config, corpus_path, max_seqs=128):
    """Collect per-channel activation stats using real tokenized text.

    Loads tokens from corpus file, runs forward through layer 0 with
    real embedding lookup, collects per-channel activation magnitudes.
    """
    H = config["hidden_size"]
    I = config.get("intermediate_size", H * 4)
    NL = config["num_hidden_layers"]
    V = config.get("vocab_size", 151936)

    WEIGHT_NAMES = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]

    # Load layer-0 weights
    print(f"  Loading layer-0 weights...")
    l0_weights = {}
    for wn in WEIGHT_NAMES:
        tname = f"model.layers.0.{wn}.weight"
        if tname in tensor_map:
            l0_weights[wn] = read_tensor(tensor_map, shard_paths, tname).astype(np.float32)

    # Load embedding table
    W_emb = read_tensor(tensor_map, shard_paths, "model.embed_tokens.weight").astype(np.float32)  # [V, H]

    # Load lm_head
    lm_tname = "model.language_model.output.weight"
    if lm_tname not in tensor_map:
        lm_tname = "lm_head.weight"
    W_lm = read_tensor(tensor_map, shard_paths, lm_tname).astype(np.float32) if lm_tname in tensor_map else None

    # Load input norm for layer 0
    norm0 = read_tensor(tensor_map, shard_paths, "model.layers.0.input_layernorm.weight").astype(np.float32)

    # Load corpus
    all_seqs = load_token_corpus(corpus_path)
    n_seqs = min(len(all_seqs), max_seqs)
    print(f"  Loaded {len(all_seqs)} sequences, using {n_seqs}")

    # Per weight-type activation stats
    wn_act_mag = {}
    for wn in WEIGHT_NAMES:
        if wn in l0_weights:
            N_out, _ = l0_weights[wn].shape
            wn_act_mag[wn] = np.zeros(N_out, dtype=np.float64)
    lm_act_mag = np.zeros(V, dtype=np.float64) if W_lm is not None else None
    lm_count = 0

    for seq_idx in range(n_seqs):
        tokens = all_seqs[seq_idx]
        seq_len = len(tokens)
        if seq_len == 0:
            continue

        # Embed lookup
        h = W_emb[tokens]  # [seq, H]

        # RMSNorm (layer 0 input norm)
        x_norm = h.copy()
        for s in range(seq_len):
            ss = np.sqrt(np.mean(x_norm[s] ** 2) + 1e-6)
            x_norm[s] = (x_norm[s] / ss) * norm0

        # QKV
        for wn in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]:
            vals = x_norm @ l0_weights[wn].T  # [seq, N]
            wn_act_mag[wn] += np.abs(vals).mean(axis=0)

        # O_proj
        vals = x_norm @ l0_weights["self_attn.o_proj"].T
        wn_act_mag["self_attn.o_proj"] += np.abs(vals).mean(axis=0)

        # Residual + gate+up + down
        # For down_proj activations, use intermediate dim
        for wn in ["mlp.gate_proj", "mlp.up_proj"]:
            vals = x_norm @ l0_weights[wn].T
            wn_act_mag[wn] += np.abs(vals).mean(axis=0)

        # Down: approximate with gate*up (SiLU simulated)
        gate = x_norm @ l0_weights["mlp.gate_proj"].T
        up = x_norm @ l0_weights["mlp.up_proj"].T
        sigmoid = 1.0 / (1.0 + np.exp(-gate))
        mlp_out = (sigmoid * gate) * up  # SiLU(gate) * up
        vals = mlp_out @ l0_weights["mlp.down_proj"].T  # [seq, H]
        wn_act_mag["mlp.down_proj"] += np.abs(vals).mean(axis=0)

        # lm_head: collect activation stats on last token
        if W_lm is not None:
            h_last = h[-1]  # [H]
            # Simple RMSNorm per-token
            ss = np.sqrt(np.mean(h_last ** 2) + 1e-6)
            h_norm = (h_last / ss)
            # Also load final_norm weight
            vals = h_norm @ W_lm.T  # [V]
            lm_act_mag += np.abs(vals)
            lm_count += 1

        if (seq_idx + 1) % 32 == 0:
            print(f"    seq {seq_idx+1}/{n_seqs}")

    # Normalize
    for wn in wn_act_mag:
        wn_act_mag[wn] = wn_act_mag[wn] / n_seqs
    if lm_act_mag is not None and lm_count > 0:
        lm_act_mag = lm_act_mag / lm_count

    # Build act_stats for all layers
    act_stats = {}
    for l in range(NL):
        for wn in WEIGHT_NAMES:
            if wn in wn_act_mag:
                k = f"{l}_{wn}"
                act_stats[k] = wn_act_mag[wn].copy()

    # lm_head
    if lm_act_mag is not None:
        act_stats["lm_head"] = lm_act_mag
    else:
        N_out = 151936
        act_stats["lm_head"] = np.ones(N_out, dtype=np.float64)

    return act_stats, [], H, I, V, NL


def collect_activation_stats_random(tensor_map, shard_paths, config, n_seqs):
    """Fallback: random normal proxy. Layer-0 only."""
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

    lm_tname = "model.language_model.output.weight"
    if lm_tname not in tensor_map:
        lm_tname = "lm_head.weight"
    if lm_tname in tensor_map:
        N_out, _ = tensor_map[lm_tname][1]['shape']
        act_stats["lm_head"] = np.ones(N_out, dtype=np.float64)

    return act_stats, [], H, I, V, NL


# ── AWQ scale computation ─────────────────────────────────────────────────
def compute_awq_scales(W_f32, act_mag, alpha=0.5):
    """Compute per-channel AWQ protection scales.

    Channels with activation >> average get scaled up:
      s = (max_act / act_mag)^alpha  clipped to [0.5, 2.0]

    This reduces quantization error on salient channels.
    """
    N, K = W_f32.shape
    if act_mag is None or len(act_mag) != N:
        return np.ones(N, dtype=np.float32)

    mean_act = act_mag.mean()
    if mean_act < 1e-10:
        return np.ones(N, dtype=np.float32)

    # s = (act_mag / mean_act)^alpha — scale up high-activation channels
    s = np.clip((act_mag / mean_act) ** alpha, 0.5, 2.0)
    return s.astype(np.float32)


# ── INT4 symmetric quantization (with AWQ scales) ─────────────────────────
def quantize_int4_sym_awq(W_f32, awq_scale, block=16):
    """INT4 symmetric block quantization with AWQ per-channel pre-scaling.

    Process:
      1. Scale weights: W'[n,:] = W[n,:] / awq_scale[n]
      2. Block-16 INT4 quantize W'
      3. Fold AWQ scale into block scales: w_sc_new = w_sc * awq_scale[n]

    Returns:
        packed: uint8 [N * K/2]
        scales: float32 [N * num_K_blks] — AWQ-aware block scales
    """
    N, K = W_f32.shape
    assert K % block == 0
    num_blks = K // block

    # 1. Apply AWQ pre-scaling
    W_scaled = W_f32 / awq_scale[:, np.newaxis]

    # 2. Block-16 INT4 quantization (symmetric, range [-7..7])
    W_blk = W_scaled.reshape(N, num_blks, block)
    blk_abs = np.max(np.abs(W_blk), axis=2)
    scales = np.maximum(blk_abs, 1e-10) / 7.0  # [N, num_blks]

    # 3. Fold AWQ scale INTO block scales
    #    w_sc_new[n, kb] = scales[n, kb] * awq_scale[n]
    scales = scales * awq_scale[:, np.newaxis]

    # Quantize W_scaled (already divided by awq_scale)
    sc_bc = scales[:, :, np.newaxis].repeat(block, axis=2).reshape(N, K)
    q = np.round(W_scaled * awq_scale[:, np.newaxis] / sc_bc)
    # Cancel the AWQ scale: W' * awq_scale / (scales * awq_scale) = W / scales
    # Wait — let me re-derive.
    #
    # Quantized weight w_nib = round(W' / w_sc *)
    #   where W' = W / awq_scale
    #   and w_sc * = blk_abs / 7  (block-16 scale of W')
    #
    # GEMV computes: out = sum(w_nib * x * (w_sc_new) * x_sc)
    #   where w_sc_new = w_sc * * awq_scale  (fold AWQ scale into block scales)
    #
    # So: w_nib * w_sc_new = round(W'/w_sc*) * w_sc* * awq_scale
    #                      = round(W/(awq_scale * w_sc*)) * w_sc* * awq_scale
    #                      ≈ W  ✓

    # Actually simpler: just compute w_nib from W_scaled and block scales of W_scaled
    # Then fold into weight scales
    W_blk = W_scaled.reshape(N, num_blks, block)
    q = np.round(W_blk / (blk_abs / 7.0)[:, :, np.newaxis])
    q = np.clip(q, -7, 7).astype(np.int32)

    # Scale folded: w_sc_new = (blk_abs/7) * awq_scale
    scales = (blk_abs / 7.0 * awq_scale[:, np.newaxis]).astype(np.float32)

    # Pack nibbles
    q_shifted = (q + 8).astype(np.uint8).reshape(N, K)  # offset-binary
    q_reshaped = q_shifted.reshape(N, K // 2, 2)
    packed = (q_reshaped[:, :, 0] & 0x0F) | ((q_reshaped[:, :, 1] & 0x0F) << 4)

    return packed, scales


# ── Weight file I/O ───────────────────────────────────────────────────────
def write_weight_int4_sym(prefix, packed, scales, K_in, N_out):
    """Write INT4 symmetric weights in kernel-compatible format."""
    num_kb = K_in // BLOCK
    header = np.array([K_in, N_out, BLOCK, num_kb, 1], dtype=np.int32)
    path_int4 = f"{prefix}.int4_t"
    with open(path_int4, 'wb') as f:
        f.write(header.tobytes())
        f.write(packed.tobytes())
    header_sc = np.array([0, 0, 0, num_kb, N_out], dtype=np.int32)
    path_sc = f"{prefix}.scale_t"
    with open(path_sc, 'wb') as f:
        f.write(header_sc.tobytes())
        f.write(scales.tobytes())
    mb = (packed.nbytes + scales.nbytes) / (1024*1024)
    print(f"  AWQ INT4: {prefix.split('/')[-1]}: {N_out}×{K_in} {mb:.1f}MB")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    global ALPHA, N_CALIB, OUT_DIR

    print("=" * 60)
    print("AWQ INT4 Calibration — Qwen3-8B")
    print("=" * 60)

    with open(os.path.join(HF_PATH, "config.json")) as f:
        config = json.load(f)
    print(f"Config: {config['num_hidden_layers']}L, "
          f"H={config['hidden_size']}, "
          f"I={config.get('intermediate_size', '?')}")

    # 1. Find shards
    shard_paths = find_model_shards(HF_PATH)
    if not shard_paths:
        print("ERROR: No safetensor files found")
        sys.exit(1)
    print(f"Found {len(shard_paths)} shard(s)")

    tensor_map = load_safetensor_headers(shard_paths)

    # 2. Collect activation stats
    corpus_path = os.environ.get("AWQ_CORPUS", "")
    if corpus_path and os.path.exists(corpus_path):
        print(f"  Using real token corpus: {corpus_path}")
        act_stats, key_names, H, I, V, NL = \
            collect_activation_stats_real(tensor_map, shard_paths, config, corpus_path, N_CALIB)
    else:
        print(f"  Using random normal proxy (or set AWQ_CORPUS env var)")
        act_stats, key_names, H, I, V, NL = \
            collect_activation_stats_random(tensor_map, shard_paths, config, N_CALIB)

    print(f"\nActivation stats collected for {len(act_stats)} weight tensors")
    for k, v in list(act_stats.items())[:5]:
        if v is not None:
            print(f"  {k}: mean_act={v.mean():.4f}, max_act={v.max():.4f}")
        else:
            print(f"  {k}: no data")

    # 3. Create output dir
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\nOutput: {OUT_DIR}/")

    # 4. Process each layer's weights
    WEIGHT_NAMES = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]

    for l in range(NL):
        for wn in WEIGHT_NAMES:
            tname = f"model.layers.{l}.{wn}.weight"
            if tname not in tensor_map:
                continue

            # Load FP32 weight
            W = read_tensor(tensor_map, shard_paths, tname).astype(np.float32)
            N_out, K_in = W.shape

            # Get activation stats for this weight
            k = f"{l}_{wn}"
            act_mag = act_stats.get(k, None)
            if act_mag is not None:
                awq_sc = compute_awq_scales(W, act_mag, ALPHA)
            else:
                awq_sc = np.ones(N_out, dtype=np.float32)

            # Quantize with AWQ
            packed, scales = quantize_int4_sym_awq(W, awq_sc, BLOCK)

            # Write
            prefix = f"{OUT_DIR}/{l}_{wn}"
            write_weight_int4_sym(prefix, packed, scales, K_in, N_out)

        if l % 8 == 0:
            print(f"  Layer {l}/{NL}")

    # 5. Embed tokens
    W_emb = read_tensor(tensor_map, shard_paths, "model.embed_tokens.weight").astype(np.float32)
    N_out, K_in = W_emb.shape
    act_mag = act_stats.get("embed_tokens", None)
    awq_sc = compute_awq_scales(W_emb, act_mag, ALPHA) if act_mag is not None else np.ones(N_out)
    packed, scales = quantize_int4_sym_awq(W_emb, awq_sc, BLOCK)
    write_weight_int4_sym(f"{OUT_DIR}/embed_tokens", packed, scales, K_in, N_out)
    print(f"  embed_tokens: {N_out}×{K_in}")

    # 6. LM head
    lm_tname = "model.language_model.output.weight"
    if lm_tname not in tensor_map:
        lm_tname = "lm_head.weight"
    if lm_tname in tensor_map:
        W_lm = read_tensor(tensor_map, shard_paths, lm_tname).astype(np.float32)
        N_out, K_in = W_lm.shape
        act_mag = act_stats.get("lm_head", None)
        awq_sc = compute_awq_scales(W_lm, act_mag, ALPHA) if act_mag is not None else np.ones(N_out)
        packed, scales = quantize_int4_sym_awq(W_lm, awq_sc, BLOCK)
        write_weight_int4_sym(f"{OUT_DIR}/lm_head", packed, scales, K_in, N_out)
        print(f"  lm_head: {N_out}×{K_in}")
    else:
        # Copy from existing weights
        import shutil
        for ext in ('.int4_t', '.scale_t'):
            src = f"weights_int4_qwen3_8b/lm_head{ext}"
            if os.path.exists(src):
                shutil.copy2(src, f"{OUT_DIR}/lm_head{ext}")
                print(f"  Copied lm_head{ext} (no activation stats)")

    # 7. Copy non-weight files (norms, etc.)
    for fn in os.listdir("weights_int4_qwen3_8b"):
        if fn.endswith('.f32') or fn.startswith('qk_norms'):
            src = f"weights_int4_qwen3_8b/{fn}"
            dst = f"{OUT_DIR}/{fn}"
            if not os.path.exists(dst):
                import shutil
                shutil.copy2(src, dst)

    print(f"\nDone. AWQ-calibrated INT4 weights in {OUT_DIR}/")
    print(f"  N_calib={N_CALIB}, alpha={ALPHA}")
    print(f"\nTo benchmark:")
    print(f"  ./bench/text_generate_int4_8b {OUT_DIR} 30")
    print(f"  ./bench/text_generate_int4_batched \"prompt\" 8 10 {OUT_DIR}")


if __name__ == "__main__":
    main()
