#!/usr/bin/env python3
"""AWQ-style calibration for Llama 3.1 8B INT4 symmetric quantization.

Reuses core logic from quantize_awq_int4_8b.py with Llama-specific config
and tensor name mappings.

Key insight (from AWQ, MLSys 2024):
  Not all weight channels matter equally. Channels with large activations
  cause more quantization error. Scale them up before quantization, then
  fold scale into weight scale during inference.

Usage:
    python3 scripts/quantize_awq_llama31_8b.py [output_dir] [n_calib]
"""
import struct, json, os, sys, re, math, copy
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────
HF_PATH = "/mnt/data/ai/models/llama31-8b-safetensors"
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/ai/models/llama31-8b-int4-awq"
N_CALIB = int(sys.argv[2]) if len(sys.argv) > 2 else 128
ALPHA = 0.5
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

# ── Safetensor I/O ───────────────────────────────────────────────────────
def find_shards(model_dir):
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        return [single]
    return sorted([os.path.join(model_dir, f) for f in os.listdir(model_dir)
                   if f.startswith("model-") and f.endswith(".safetensors")])

def load_tensor_map(shard_paths):
    tm = {}
    for si, sp in enumerate(shard_paths):
        with open(sp, 'rb') as f:
            hdr_len = struct.unpack('Q', f.read(8))[0]
            hdr = json.loads(f.read(hdr_len))
        for name, info in hdr.items():
            if name != '__metadata__':
                tm[name] = (si, info)
    return tm

def read_tensor(tm, shards, name):
    si, info = tm[name]
    sp = shards[si]
    start, end = info['data_offsets']
    with open(sp, 'rb') as f:
        f.seek(0)
        hdr_len = struct.unpack('Q', f.read(8))[0]
        f.seek(8 + hdr_len + start)
        raw = f.read(end - start)
    if info['dtype'] == 'BF16':
        return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(info['shape'])
    return np.frombuffer(raw, dtype=np.float32 if info['dtype'] == 'F32' else np.float16).reshape(info['shape'])

# ── Tokenizer ─────────────────────────────────────────────────────────────
def encode_prompt(text, tok_data):
    """BPE encode using tokenizer_data.bin format (same as BpeTokenizer)."""
    # Simple GPT-2 BPE encoder reusing the tokenizer file
    with open(tok_data, 'rb') as f:
        nv = struct.unpack('I', f.read(4))[0]
        nm = struct.unpack('I', f.read(4))[0]
        na = struct.unpack('I', f.read(4))[0]
        f.read(256 * 4)  # byte encoder

        # Read vocab
        vocab = {}
        for _ in range(nv):
            tid = struct.unpack('I', f.read(4))[0]
            length = struct.unpack('H', f.read(2))[0]
            token_str = f.read(length).decode('utf-8', errors='replace')
            vocab[token_str] = tid

        # Read merges
        merges = []
        for _ in range(nm):
            ll = struct.unpack('H', f.read(2))[0]
            left = f.read(ll).decode('utf-8', errors='replace')
            rl = struct.unpack('H', f.read(2))[0]
            right = f.read(rl).decode('utf-8', errors='replace')
            merges.append((left, right))

    # Build merge rank map
    merge_rank = {f"{a} {b}": i for i, (a, b) in enumerate(merges)}

    # Check for special tokens
    special_tokens = {
        "<|begin_of_text|>": 128000,
        "<|end_of_text|>": 128001,
        "<|eot_id|>": 128009,
    }

    # Simple encode: check special tokens first, then BPE
    ids = []
    pos = 0
    while pos < len(text):
        matched = False
        for tok_str, tok_id in sorted(special_tokens.items(), key=lambda x: -len(x[0])):
            if text[pos:pos+len(tok_str)] == tok_str:
                ids.append(tok_id)
                pos += len(tok_str)
                matched = True
                break
        if matched:
            continue
        # BPE encode the rest up to next special token
        next_special = len(text)
        for tok_str in special_tokens:
            idx = text.find(tok_str, pos)
            if idx != -1 and idx < next_special:
                next_special = idx

        segment = text[pos:next_special]
        # Byte-level BPE encode
        byte_enc = {}
        for i in range(256):
            if i < 33 or i == 127 or i == 173 or (128 <= i <= 160):
                byte_enc[i] = chr(256 + (i if i < 33 else i - 128 + 256 if 128 <= i <= 160 else 259 if i == 127 else 321))
            else:
                byte_enc[i] = chr(i)

        # Split into words (simplified pre-tokenize)
        words = re.findall(r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\S+", segment)
        for word in words:
            bpe = [byte_enc[b] for b in word.encode('utf-8')]
            # Merge
            while len(bpe) > 1:
                best_pair = None
                best_rank = 1 << 30
                for j in range(len(bpe)-1):
                    key = f"{bpe[j]} {bpe[j+1]}"
                    if key in merge_rank and merge_rank[key] < best_rank:
                        best_rank = merge_rank[key]
                        best_pair = j
                if best_pair is None:
                    break
                bpe[best_pair] = bpe[best_pair] + bpe[best_pair+1]
                del bpe[best_pair+1]
            for token in bpe:
                if token in vocab:
                    ids.append(vocab[token])
        pos = next_special

    return ids


# ── Activation collector ──────────────────────────────────────────────────
def collect_activation_stats(tm, shards, config, n_seqs, tok_data):
    """Collect per-channel activation stats using layer-0 forward pass.

    For Llama: loads layer-0 weights, runs calibration prompts through
    layer 0 (embed → RMSNorm → QKV → O_proj → residual → gate/up/down),
    collects activation magnitude per output channel.
    """
    WEIGHT_NAMES = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]

    print(f"  Loading layer-0 weights...")
    l0_w = {}
    for wn in WEIGHT_NAMES:
        tname = f"model.layers.0.{wn}.weight"
        l0_w[wn] = read_tensor(tm, shards, tname).astype(np.float32)

    W_emb = read_tensor(tm, shards, "model.embed_tokens.weight").astype(np.float32)
    norm0 = read_tensor(tm, shards, "model.layers.0.input_layernorm.weight").astype(np.float32)

    wn_act = {wn: np.zeros(l0_w[wn].shape[0], dtype=np.float64) for wn in WEIGHT_NAMES}

    # Use calibration prompts directly (encode each, use first token's embedding)
    n_procs = 0
    for prompt in CALIB_PROMPTS[:n_seqs]:
        ids = encode_prompt(prompt, tok_data)
        if not ids:
            continue
        tok = ids[0]

        # Embed lookup for first token
        h = W_emb[tok].copy()

        # RMSNorm
        ss = np.sqrt(np.mean(h ** 2) + 1e-6)
        h_norm = (h / ss) * norm0

        # QKV
        for wn in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]:
            wn_act[wn] += np.abs(h_norm @ l0_w[wn].T)

        # O_proj
        wn_act["self_attn.o_proj"] += np.abs(h_norm @ l0_w["self_attn.o_proj"].T)

        # Gate + up
        for wn in ["mlp.gate_proj", "mlp.up_proj"]:
            wn_act[wn] += np.abs(h_norm @ l0_w[wn].T)

        # Down (approximate with gate*up and SiLU)
        gate = h_norm @ l0_w["mlp.gate_proj"].T
        up = h_norm @ l0_w["mlp.up_proj"].T
        sig = 1.0 / (1.0 + np.exp(-gate))
        mlp = (sig * gate) * up
        wn_act["mlp.down_proj"] += np.abs(mlp @ l0_w["mlp.down_proj"].T)

        n_procs += 1

    # Normalize
    for wn in wn_act:
        wn_act[wn] /= max(n_procs, 1)

    # Replicate to all layers
    act_stats = {}
    for l in range(NL):
        for wn in WEIGHT_NAMES:
            act_stats[f"{l}_{wn}"] = wn_act[wn].copy()

    # lm_head: also collect activation stats for last-token prediction
    # For calibration, use last activation after final norm
    # (approximate as same distribution as o_proj)
    act_stats["lm_head"] = wn_act["self_attn.o_proj"].copy()

    return act_stats, H, I, V, NL


# ── AWQ scale computation ────────────────────────────────────────────────
def compute_awq_scales(W, act_mag, alpha=0.5):
    N, K = W.shape
    if act_mag is None or len(act_mag) != N:
        return np.ones(N, dtype=np.float32)
    mean_act = act_mag.mean()
    if mean_act < 1e-10:
        return np.ones(N, dtype=np.float32)
    s = np.clip((act_mag / mean_act) ** alpha, 0.5, 2.0)
    return s.astype(np.float32)


def quantize_int4_sym_awq(W, awq_scale, block=16):
    N, K = W.shape
    assert K % block == 0
    num_blks = K // block

    W_scaled = W / awq_scale[:, np.newaxis]
    W_blk = W_scaled.reshape(N, num_blks, block)
    blk_abs = np.max(np.abs(W_blk), axis=2)
    scales = np.maximum(blk_abs, 1e-10) / 7.0
    scales = scales * awq_scale[:, np.newaxis]

    W_blk = W_scaled.reshape(N, num_blks, block)
    q = np.round(W_blk / (blk_abs / 7.0)[:, :, np.newaxis])
    q = np.clip(q, -7, 7).astype(np.int32)

    scales = (blk_abs / 7.0 * awq_scale[:, np.newaxis]).astype(np.float32)

    q_shifted = (q + 8).astype(np.uint8).reshape(N, K)
    q_reshaped = q_shifted.reshape(N, K // 2, 2)
    packed = (q_reshaped[:, :, 0] & 0x0F) | ((q_reshaped[:, :, 1] & 0x0F) << 4)

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
    print(f"  AWQ INT4: {os.path.basename(prefix)}: {N_out}x{K_in} {mb:.1f}MB")


def write_f32(prefix, data):
    with open(f"{prefix}.f32", 'wb') as f:
        f.write(data.astype(np.float32).tobytes())


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    global ALPHA, N_CALIB, OUT_DIR

    print("=" * 60)
    print("AWQ INT4 Calibration — Llama 3.1 8B")
    print("=" * 60)

    shards = find_shards(HF_PATH)
    if not shards:
        print("ERROR: No safetensor files found")
        sys.exit(1)
    print(f"Found {len(shards)} shard(s)")

    tm = load_tensor_map(shards)

    # Find tokenizer_data.bin
    tok_data = None
    for cand in [
        "/mnt/data/ai/models/llama31-8b-int4-from-safetensors/tokenizer_data.bin",
        "/tmp/final_int4/tokenizer_data.bin",
    ]:
        if os.path.exists(cand):
            tok_data = cand
            break
    if not tok_data:
        print("ERROR: no tokenizer_data.bin found")
        sys.exit(1)

    # Collect activation stats
    print(f"  Collecting activation stats ({N_CALIB} sequences)...")
    act_stats, H, I, V, NL = collect_activation_stats(tm, shards, None, N_CALIB, tok_data)
    print(f"\n  Activation stats collected for {len(act_stats)} weight tensors")
    for k, v in list(act_stats.items())[:5]:
        print(f"  {k}: mean_act={v.mean():.4f}, max_act={v.max():.4f}")

    # Create output dir
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\nOutput: {OUT_DIR}/")

    # Process each layer
    WEIGHT_NAMES = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]

    for l in range(NL):
        for wn in WEIGHT_NAMES:
            tname = f"model.layers.{l}.{wn}.weight"
            if tname not in tm:
                continue
            W = read_tensor(tm, shards, tname).astype(np.float32)
            N_out, K_in = W.shape
            k = f"{l}_{wn}"
            act_mag = act_stats.get(k, None)
            awq_sc = compute_awq_scales(W, act_mag, ALPHA) if act_mag is not None else np.ones(N_out)
            packed, scales = quantize_int4_sym_awq(W, awq_sc, BLOCK)
            write_weight(f"{OUT_DIR}/{l}_{wn}", packed, scales, K_in, N_out)

        # Norms
        for ntype in ["input_layernorm", "post_attention_layernorm"]:
            tname = f"model.layers.{l}.{ntype}.weight"
            if tname in tm:
                w = read_tensor(tm, shards, tname)
                write_f32(f"{OUT_DIR}/{l}_{ntype}", w.ravel())

        if l % 8 == 0:
            print(f"  Layer {l}/{NL}")

    # Final norm
    if "model.norm.weight" in tm:
        fn = read_tensor(tm, shards, "model.norm.weight")
        write_f32(f"{OUT_DIR}/final_norm", fn.ravel())

    # Embed tokens
    W_emb = read_tensor(tm, shards, "model.embed_tokens.weight").astype(np.float32)
    N_out, K_in = W_emb.shape
    awq_sc = np.ones(N_out, dtype=np.float32)  # No AWQ for embed (different role)
    packed, scales = quantize_int4_sym_awq(W_emb, awq_sc, BLOCK)
    write_weight(f"{OUT_DIR}/embed_tokens", packed, scales, K_in, N_out)

    # LM head (may be tied to embed for Llama)
    lm_tname = "lm_head.weight"
    if lm_tname in tm:
        W_lm = read_tensor(tm, shards, lm_tname).astype(np.float32)
        N_out, K_in = W_lm.shape
        act_mag = act_stats.get("lm_head", None)
        awq_sc = compute_awq_scales(W_lm, act_mag, ALPHA) if act_mag is not None else np.ones(N_out)
        packed, scales = quantize_int4_sym_awq(W_lm, awq_sc, BLOCK)
        write_weight(f"{OUT_DIR}/lm_head", packed, scales, K_in, N_out)
    else:
        print("  lm_head: not found (tied to embed)")

    # QK norms: identity for Llama
    qk = np.ones(NL * 2 * hd, dtype=np.float32)
    write_f32(f"{OUT_DIR}/qk_norms", qk)

    # RoPE config
    with open(f"{OUT_DIR}/rope_config.f32", 'wb') as f:
        cfg = np.array([rope_theta, hd], dtype=np.float32)
        f.write(cfg.tobytes())
    print(f"  rope_config: theta={rope_theta}, hd={hd}")

    # Tokenizer
    import shutil
    shutil.copy2(tok_data, f"{OUT_DIR}/tokenizer_data.bin")
    print(f"  tokenizer: copied from {tok_data}")

    print(f"\nDone. AWQ-calibrated INT4 weights in {OUT_DIR}/")
    print(f"  N_calib={N_CALIB}, alpha={ALPHA}")
    print(f"  To test: ./bench/text_generate_llama31_8b -w {OUT_DIR} 'Hello' 20")
    print(f"  To PPL: ./bench/bench_ppl_llama31_8b {OUT_DIR}")


if __name__ == "__main__":
    main()
