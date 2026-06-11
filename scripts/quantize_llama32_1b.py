#!/usr/bin/env python3
"""Quantize Llama 3.2 1B from safetensors to Blackwell INT4 format.

Loads FP16 safetensors → dequant to BF16 → INT4 symmetric quantization
→ writes .int4_t + .scale_t files for the bench.

Usage:
    python3 scripts/quantize_llama32_1b.py [output_dir]
"""
import struct, json, os, sys, re, math
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────
HF_PATH = "/mnt/data/ai/models/llama32-1b-safetensors"
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/ai/models/llama32-1b-int4-from-safetensors"
BLOCK = 16  # quantization block size (must match kernel)

# Llama 3.2 1B config
H = 2048
I = 8192
NL = 16
nqh = 32
nkv = 8
hd = 64
V = 128256
eps = 1e-5
rope_theta = 500000.0
tie_embeddings = True


def read_tensor(safetensors_path, name):
    """Read a single tensor from safetensors by name."""
    with open(safetensors_path, 'rb') as f:
        hdr_len = struct.unpack('Q', f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
        
    if name not in hdr:
        return None
    
    info = hdr[name]
    start, end = info['data_offsets']
    
    with open(safetensors_path, 'rb') as f:
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
    """Block-16 INT4 symmetric quantization (no AWQ)."""
    N, K = W.shape
    assert K % block == 0
    num_blks = K // block
    
    W_blk = W.reshape(N, num_blks, block)
    blk_abs = np.max(np.abs(W_blk), axis=2)
    scales = np.maximum(blk_abs, 1e-10) / 7.0  # [N, num_blks]
    
    q = np.round(W_blk / (blk_abs / 7.0)[:, :, np.newaxis])
    q = np.clip(q, -7, 7).astype(np.int32)
    
    q_shifted = (q + 8).astype(np.uint8).reshape(N, K)
    q_reshaped = q_shifted.reshape(N, K // 2, 2)
    packed = (q_reshaped[:, :, 0] & 0x0F) | ((q_reshaped[:, :, 1] & 0x0F) << 4)
    
    scales = (blk_abs / 7.0).astype(np.float32)
    return packed, scales


def write_weight(prefix, packed, scales, K_in, N_out):
    """Write INT4 weight in kernel format."""
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
    print(f"  {prefix.split('/')[-1]}: {N_out}×{K_in} {mb:.1f}MB")


def write_f32(prefix, data):
    """Write F32 norm weight."""
    with open(f"{prefix}.f32", 'wb') as f:
        f.write(data.astype(np.float32).tobytes())
    print(f"  {prefix.split('/')[-1]}: {len(data)} F32")


def main():
    print("=" * 60)
    print("Llama 3.2 1B → Blackwell INT4 (from safetensors)")
    print("=" * 60)
    
    model_path = os.path.join(HF_PATH, "model.safetensors")
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found")
        return 1
    
    print(f"Model: {model_path}")
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
    
    NORM_NAMES = {
        "input_layernorm": "model.layers.{}.input_layernorm.weight",
        "post_attention_layernorm": "model.layers.{}.post_attention_layernorm.weight",
    }
    
    # Process layers
    for l in range(NL):
        for bw_name, hf_pattern in WEIGHT_NAMES.items():
            tname = hf_pattern.format(l)
            W = read_tensor(model_path, tname)
            if W is None:
                print(f"  SKIP: {tname} not found")
                continue
            
            N_out, K_in = W.shape
            packed, scales = quantize_int4_sym(W.astype(np.float32), BLOCK)
            prefix = f"{OUT_DIR}/{l}_{bw_name}"
            write_weight(prefix, packed, scales, K_in, N_out)
        
        # Norms
        for bw_name, hf_pattern in NORM_NAMES.items():
            tname = hf_pattern.format(l)
            w = read_tensor(model_path, tname)
            if w is None:
                print(f"  SKIP: {tname} not found")
                continue
            write_f32(f"{OUT_DIR}/{l}_{bw_name}", w.ravel())
        
        if l % 4 == 0:
            print(f"  Layer {l}/{NL}")
    
    # Final norm
    fn = read_tensor(model_path, "model.norm.weight")
    if fn is not None:
        write_f32(f"{OUT_DIR}/final_norm", fn.ravel())
    
    # Embedding
    W_emb = read_tensor(model_path, "model.embed_tokens.weight")
    if W_emb is not None:
        N_out, K_in = W_emb.shape
        packed, scales = quantize_int4_sym(W_emb.astype(np.float32), BLOCK)
        write_weight(f"{OUT_DIR}/embed_tokens", packed, scales, K_in, N_out)
    
    # LM head (often tied to embedding for Llama)
    W_lm = read_tensor(model_path, "lm_head.weight")
    if W_lm is None:
        print("  lm_head: tied to embed (no separate file)")
    else:
        N_out, K_in = W_lm.shape
        packed, scales = quantize_int4_sym(W_lm.astype(np.float32), BLOCK)
        write_weight(f"{OUT_DIR}/lm_head", packed, scales, K_in, N_out)
    
    # QK norms: Llama 3.2 has NO QK norms → write identity
    qk = np.ones(NL * 2 * hd, dtype=np.float32)
    with open(f"{OUT_DIR}/qk_norms.f32", 'wb') as f:
        f.write(qk.tobytes())
    print(f"  qk_norms: {NL}x2x{hd} F32 (identity)")
    
    # RoPE config
    with open(f"{OUT_DIR}/rope_config.f32", 'wb') as f:
        cfg = np.array([rope_theta, hd], dtype=np.float32)
        f.write(cfg.tobytes())
    print(f"  rope_config: theta={rope_theta}, hd={hd}")
    
    # Copy tokenizer from GGUF-converted (already have tokenizer_data.bin)
    tok_src = os.path.join(HF_PATH, "tokenizer.json")
    if os.path.exists(tok_src):
        # Try to find an existing tokenizer_data.bin from converter output
        for candidate in [
            "/mnt/data/ai/models/llama32-gguf-test/llama32-int4-fresh/tokenizer_data.bin",
            "/tmp/final_int4/tokenizer_data.bin",
        ]:
            if os.path.exists(candidate):
                import shutil
                shutil.copy2(candidate, f"{OUT_DIR}/tokenizer_data.bin")
                print(f"  tokenizer: copied from {candidate}")
                break
        else:
            print("  WARNING: no tokenizer_data.bin found, bench may crash")
    else:
        print("  WARNING: no tokenizer.json found")
    
    print(f"\nDone. Output: {OUT_DIR}/")
    print(f"  To test: ./bench/text_generate_llama32_1b -w {OUT_DIR} 'Hello' 20")


if __name__ == "__main__":
    main()
