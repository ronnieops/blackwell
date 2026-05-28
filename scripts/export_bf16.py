#!/usr/bin/env python3
"""Export BF16 weights from safetensors to raw binary files for CUDA bench.

Each weight file: header [N_out, K_in] as int32 + N_out*K_in BF16 bytes.
Weight layout: [N_out, K_in] row-major (same as safetensors, NOT transposed).
GEMV kernel handles the transpose implicitly or weights are stored transposed.

Run: python3 scripts/export_bf16.py
Output: weights_bf16/*.bf16 (one file per weight matrix)
"""
import struct, json, os
import numpy as np

MODEL = "/mnt/data/ai/hf/qwen3-1.7b-base/model.safetensors"
OUT = "weights_fp16"
NL = 28

WEIGHT_NAMES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]

with open(MODEL, 'rb') as f:
    hdr_len = struct.unpack('Q', f.read(8))[0]
    hdr = json.loads(f.read(hdr_len))

def read_bf16_tensor(name):
    info = hdr[name]
    start, end = info['data_offsets']
    with open(MODEL, 'rb') as f:
        f.seek(8 + hdr_len + start)
        raw = f.read(end - start)
    u16 = np.frombuffer(raw, dtype=np.uint16)
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    return f32.reshape(info['shape'])

os.makedirs(OUT, exist_ok=True)

def write_fp16_weight(path, W_f32):
    """Write FP32 → FP16 binary file. Header: [N_out, K_in] int32."""
    N_out, K_in = W_f32.shape
    fp16 = W_f32.astype(np.float16)
    with open(path, 'wb') as f:
        f.write(struct.pack('ii', N_out, K_in))
        f.write(fp16.tobytes())
    mb = fp16.nbytes / (1024*1024)
    print(f"  {path}: [{N_out}×{K_in}] {mb:.1f}MB")

# Per-layer weights
for layer in range(NL):
    for wn in WEIGHT_NAMES:
        tname = f"model.layers.{layer}.{wn}.weight"
        W = read_bf16_tensor(tname)  # [N_out, K_in]
        fname = f"{layer}_{wn}.fp16"
        write_fp16_weight(f"{OUT}/{fname}", W)

# embed_tokens
print("Processing embed_tokens...")
W_emb = read_bf16_tensor("model.embed_tokens.weight")  # [vocab, hidden]
write_fp16_weight(f"{OUT}/embed_tokens.fp16", W_emb)

# RMSNorm weights (FP32, small)
for layer in range(NL):
    for nt in ["input_layernorm", "post_attention_layernorm"]:
        tname = f"model.layers.{layer}.{nt}.weight"
        W = read_bf16_tensor(tname)
        W.astype(np.float32).tofile(f"{OUT}/{layer}_{nt}.f32")

# Final norm
W = read_bf16_tensor("model.norm.weight")
W.astype(np.float32).tofile(f"{OUT}/final_norm.f32")

# Q/K norms
qk_data = np.zeros(NL * 2 * 128, dtype=np.float32)
for l in range(NL):
    qn = read_bf16_tensor(f"model.layers.{l}.self_attn.q_norm.weight")
    kn = read_bf16_tensor(f"model.layers.{l}.self_attn.k_norm.weight")
    qk_data[l*2*128 : l*2*128+128] = qn.astype(np.float32)
    qk_data[l*2*128+128 : l*2*128+256] = kn.astype(np.float32)
qk_data.tofile(f"{OUT}/qk_norms.f32")

print(f"\nDone. {NL} layers × {len(WEIGHT_NAMES)} weights + embed + norms")
