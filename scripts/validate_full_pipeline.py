#!/usr/bin/env python3
"""Full 28-layer pipeline validation: INT8 per-row vs BF16 reference.
Pure numpy — no PyTorch needed."""

import struct, json, numpy as np

MODEL = "/mnt/data/ai/hf/qwen3-1.7b-base/model.safetensors"
WDIR = "weights_int8_bf16"
B = 16
H = 2048; QD = 2048; KV = 1024; ID = 6144
nqh = 16; nkv = 8; hd = 128; NL = 28
VOCAB = 151936; eps = 1e-6

with open(MODEL, 'rb') as f:
    hl = struct.unpack('Q', f.read(8))[0]
    hdr = json.loads(f.read(hl))

def read_bf16(name):
    info = hdr[name]
    s, e = info['data_offsets']
    with open(MODEL, 'rb') as f:
        f.seek(8 + hl + s); raw = f.read(e - s)
    u16 = np.frombuffer(raw, dtype=np.uint16)
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    return f32.reshape(info['shape'])

def load_i8(prefix):
    with open(f"{prefix}.int8_t", 'rb') as f:
        h = struct.unpack('5i', f.read(20))
        data = np.frombuffer(f.read(h[0]*h[1]), dtype=np.int8).reshape(h[1], h[0])
    with open(f"{prefix}.scale_t", 'rb') as f:
        h2 = struct.unpack('5i', f.read(20))
        scales = np.frombuffer(f.read(h2[3]*h2[4]*4), dtype=np.float32).reshape(h2[4], h2[3])
    return data, scales  # W_t [N,K], scales [N, K/16]

def rmsnorm(x, w):
    ss = np.mean(x.astype(np.float64)**2)
    return x * w / np.sqrt(ss + eps)

def quant_int8(x):
    """Per-vector block-16 INT8 quantization."""
    x_blk = x.reshape(-1, B)
    amax = np.max(np.abs(x_blk), axis=1)
    sc = np.maximum(amax / 127.0, 1e-9)
    q = np.clip(np.round(x_blk / sc[:, np.newaxis]), -127, 127).astype(np.int8)
    return q.reshape(-1), sc

def gemv_per_row(x_i8, x_sc, W_t_i8, W_sc):
    """Per-row INT8 GEMV. Returns [N] float32."""
    N, K = W_t_i8.shape
    nb = K // B
    x_blk = x_i8.reshape(1, nb, B).astype(np.int32)
    W_blk = W_t_i8.reshape(N, nb, B).astype(np.int32)
    dot = np.sum(W_blk * x_blk, axis=2).astype(np.float32)
    sc = W_sc * x_sc[np.newaxis, :]
    return np.sum(dot * sc, axis=1)

def gemv_bf16(x, W):
    """BF16 reference GEMV: y = W @ x"""
    return W.astype(np.float32) @ x.astype(np.float32)

def cosim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return np.dot(a, b) / (na * nb)

# ── Load weights ──
print("Loading weights...")
norms_in, norms_post = [], []
for l in range(NL):
    norms_in.append(read_bf16(f"model.layers.{l}.input_layernorm.weight"))
    norms_post.append(read_bf16(f"model.layers.{l}.post_attention_layernorm.weight"))

qk_norms = []
for l in range(NL):
    qn = read_bf16(f"model.layers.{l}.self_attn.q_norm.weight")
    kn = read_bf16(f"model.layers.{l}.self_attn.k_norm.weight")
    qk_norms.append((qn, kn))

final_norm = read_bf16("model.norm.weight")

# INT8 per-row weights
W_i8 = {}
for l in range(NL):
    for wn in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
               "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
        W_i8[(l, wn)] = load_i8(f"{WDIR}/{l}_{wn}")

W_i8_emb = load_i8(f"{WDIR}/embed_tokens")

# BF16 weights
W_bf16 = {}
for l in range(NL):
    for wn in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
               "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
        W_bf16[(l, wn)] = read_bf16(f"model.layers.{l}.{wn}.weight")
W_bf16_emb = read_bf16("model.embed_tokens.weight")

print(f"Loaded {len(W_i8)} INT8 weights + BF16 references")

# ── Single-token pipeline (step 0) ──
# Use token for "The" (first token of "The capital of France is")
# For a proper test, use a fixed random input
np.random.seed(42)
x = np.random.randn(H).astype(np.float32) * 0.3  # simulate post-embedding activation

print(f"\n{'Layer':>5} {'INT8 cosim':>11} {'BF16 ref norm':>13} {'INT8 norm':>10}")
print("-" * 45)

# INT8 pipeline
x_i8 = x.copy()
x_bf16 = x.copy()

for l in range(NL):
    # ── INT8 path ──
    x_normed_i8 = rmsnorm(x_i8, norms_in[l])
    xi_q, xi_sc = quant_int8(x_normed_i8)
    Q_i8 = gemv_per_row(xi_q, xi_sc, *W_i8[(l, "self_attn.q_proj")])
    K_i8 = gemv_per_row(xi_q, xi_sc, *W_i8[(l, "self_attn.k_proj")])
    V_i8 = gemv_per_row(xi_q, xi_sc, *W_i8[(l, "self_attn.v_proj")])
    # Simplified attention (skip head norms, RoPE for this test — just add residual)
    attn_out_i8 = np.zeros(QD, dtype=np.float32)  # placeholder
    # For full test, skip attention details — just test MLP path quality
    o_i8 = gemv_per_row(*quant_int8(attn_out_i8), *W_i8[(l, "self_attn.o_proj")])
    x_res1_i8 = x_i8 + o_i8  # residual

    x_normed2_i8 = rmsnorm(x_res1_i8, norms_post[l])
    xi2_q, xi2_sc = quant_int8(x_normed2_i8)
    gate_i8 = gemv_per_row(xi2_q, xi2_sc, *W_i8[(l, "mlp.gate_proj")])
    up_i8 = gemv_per_row(xi2_q, xi2_sc, *W_i8[(l, "mlp.up_proj")])
    mlp_i8 = gate_i8 * (1 / (1 + np.exp(-gate_i8))) * up_i8  # SwiGLU
    down_i8 = gemv_per_row(*quant_int8(mlp_i8), *W_i8[(l, "mlp.down_proj")])
    x_i8 = x_res1_i8 + down_i8

    # ── BF16 path ──
    x_normed_bf16 = rmsnorm(x_bf16, norms_in[l])
    Q_bf16 = gemv_bf16(x_normed_bf16, W_bf16[(l, "self_attn.q_proj")])
    K_bf16 = gemv_bf16(x_normed_bf16, W_bf16[(l, "self_attn.k_proj")])
    V_bf16 = gemv_bf16(x_normed_bf16, W_bf16[(l, "self_attn.v_proj")])
    attn_out_bf16 = np.zeros(QD, dtype=np.float32)
    o_bf16 = gemv_bf16(attn_out_bf16, W_bf16[(l, "self_attn.o_proj")])
    x_res1_bf16 = x_bf16 + o_bf16

    x_normed2_bf16 = rmsnorm(x_res1_bf16, norms_post[l])
    gate_bf16 = gemv_bf16(x_normed2_bf16, W_bf16[(l, "mlp.gate_proj")])
    up_bf16 = gemv_bf16(x_normed2_bf16, W_bf16[(l, "mlp.up_proj")])
    mlp_bf16 = gate_bf16 * (1 / (1 + np.exp(-gate_bf16))) * up_bf16
    down_bf16 = gemv_bf16(mlp_bf16, W_bf16[(l, "mlp.down_proj")])
    x_bf16 = x_res1_bf16 + down_bf16

    cs = cosim(x_i8, x_bf16)
    if l % 4 == 0 or l == NL - 1:
        print(f"{l:>5} {cs:>11.6f} {np.linalg.norm(x_bf16):>13.4f} {np.linalg.norm(x_i8):>10.4f}")

print(f"\nFinal 28L cosim (MLP-only, no attention): {cosim(x_i8, x_bf16):.6f}")
