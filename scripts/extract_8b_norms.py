#!/usr/bin/env python3
"""Extract FP32 norms + lm_head from Qwen3-8B safetensors."""
import json, struct, sys, os
import numpy as np

MODEL = "/mnt/data/ai/hf/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
OUT = "weights_int8_qwen3_8b"
os.makedirs(OUT, exist_ok=True)

# Find all shards
shards = sorted([os.path.join(MODEL, f) for f in os.listdir(MODEL) if f.endswith(".safetensors")])

# Read all headers
tensor_map = {}
for path in shards:
    with open(path, 'rb') as f:
        hdr_len = struct.unpack('Q', f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
    for name, info in hdr.items():
        if name == '__metadata__': continue
        info['shard'] = path
        tensor_map[name] = info

def read_tensor(name, n=None):
    info = tensor_map[name]
    start, end = info['data_offsets']
    with open(info['shard'], 'rb') as f:
        f.seek(8)
        f.seek(0)
        hdr_len = struct.unpack('Q', f.read(8))[0]
        f.seek(8 + hdr_len + start)
        raw = f.read(end - start)
    if info.get('dtype', 'BF16') == 'BF16':
        u16 = np.frombuffer(raw, dtype=np.uint16).copy()
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        return f32.reshape(info['shape'])
    return np.frombuffer(raw, dtype=np.float32).reshape(info['shape'])

config_path = os.path.join(MODEL, "config.json")
with open(config_path) as f:
    cfg = json.load(f)
tc = cfg
NL = tc['num_hidden_layers']
H = tc['hidden_size']
V = tc.get('vocab_size', 152064)
HD = tc.get('head_dim', 128)
NQ = tc['num_attention_heads']
NKV = tc['num_key_value_heads']

print(f"Layers: {NL}, H={H}, V={V}, NQ={NQ}, NKV={NKV}, HD={HD}")

# Extract norms
for l in range(NL):
    for name in [f"model.layers.{l}.input_layernorm.weight",
                 f"model.layers.{l}.post_attention_layernorm.weight"]:
        w = read_tensor(name)
        out_name = f"{l}_" + name.split('.')[-2] + ".f32"
        w.tofile(os.path.join(OUT, out_name))
        sys.stdout.write(f"\r  Norm {l}/36") 
        sys.stdout.flush()

# Final norm
fn = read_tensor("model.norm.weight")
fn.tofile(os.path.join(OUT, "final_norm.f32"))

# QK norms  
qk = []
for l in range(NL):
    qn = read_tensor(f"model.layers.{l}.self_attn.q_norm.weight")
    kn = read_tensor(f"model.layers.{l}.self_attn.k_norm.weight")
    qk.extend(qn.tolist())
    qk.extend(kn.tolist())
qk_a = np.array(qk, dtype=np.float32)
qk_a.tofile(os.path.join(OUT, "qk_norms.f32"))

# lm_head (V × H) INT8 quant
print("\n\nExtracting lm_head...")
lm = read_tensor("lm_head.weight")  # [V, H] BF16
N_out, K_in = lm.shape
BLOCK = 16
num_K_blks = K_in // BLOCK

lm_blk = lm.reshape(N_out, num_K_blks, BLOCK)
blk_max = np.max(np.abs(lm_blk), axis=2)
scales = blk_max / 127.0
scales = np.maximum(scales, 1e-10)
lm_q = np.clip(np.round(lm / scales[:, :, np.newaxis].repeat(BLOCK, axis=2).reshape(N_out, K_in)), -128, 127).astype(np.int8)

hdr = np.array([K_in, N_out, BLOCK, num_K_blks, N_out], dtype=np.int32)
with open(os.path.join(OUT, "lm_head.int8_t"), 'wb') as f:
    f.write(hdr.tobytes())
    f.write(lm_q.tobytes())
with open(os.path.join(OUT, "lm_head.scale_t"), 'wb') as f:
    f.write(hdr.tobytes())
    f.write(scales.tobytes())

print(f"lm_head: [{N_out}×{K_in}] INT8 = {lm_q.nbytes/1e6:.1f} MB")

# Also verify embed_tokens exists
embed_path = os.path.join(OUT, "embed_tokens.int8_t")
if os.path.exists(embed_path):
    print(f"embed_tokens: exists ✅")
else:
    print("embed_tokens: MISSING — run quantize_generic.py first")
    sys.exit(1)

print(f"\nDone. All support files extracted to {OUT}/")
print(f"  {NL*2} layernorms + final_norm + qk_norms + lm_head")