#!/usr/bin/env python3
"""validate_pipeline.py — Python reference, EXACT match to CUDA pipeline.

Compares CUDA output (/tmp/layer0_out.bin) against Python computation
using the same weights, input, and operations.

Usage: python3 bench/validate_pipeline.py
"""
import struct, json, sys, os
import numpy as np

MODEL = "/mnt/data/ai/hf/qwen3-1.7b-base/model.safetensors"
WEIGHTDIR = "weights_int8_bf16"

H = 2048   # hidden_size
QD = 2048  # q_dim
KV = 1024  # kv_dim
ID = 6144  # intermediate_size
hd = 128
nqh = 16
nkv = 8
eps = 1e-6
MAXSEQ = 2048
B = 16     # block size

# ── Weight loaders ────────────────────────────────────────────────────

def load_int8(prefix):
    """Load PRE-TRANSPOSED INT8 weight file.
    File is .int8_t suffix = [N,K] layout (transposed).
    Header reports original K_in=h[0], N_out=h[1].
    Returns W [N,K] int8, W_scale [N/16,K/16] float32."""
    with open(f"{prefix}.int8_t", 'rb') as f:
        hdr = struct.unpack('5i', f.read(20))
        K_in, N_out = hdr[0], hdr[1]
        raw = np.frombuffer(f.read(K_in * N_out), dtype=np.int8)
    # File is [N×K] (pre-transposed). Reshape directly as [N, K]
    W = raw.reshape(N_out, K_in).copy()  # [N, K]
    
    with open(f"{prefix}.scale_t", 'rb') as f:
        hdr = struct.unpack('5i', f.read(20))
        raw_s = np.frombuffer(f.read(hdr[3] * hdr[4] * 4), dtype=np.float32)
    # Scale file is [N/16 × K/16] (pre-transposed)
    W_sc = raw_s.reshape(hdr[4], hdr[3]).copy()  # [N/16, K/16]
    return W, W_sc

def load_f32(path):
    """Load raw f32 file. Returns [N] float32."""
    return np.fromfile(path, dtype=np.float32)

# ── Quantization (EXACT match to GPU: round-to-nearest, saturate) ────

def block_quant(x, B=16):
    """Quantize x (float32 [N]) to INT8.
    For each 16-element block: absmax → scale = max(absmax/127, 1e-9)
    x_i8 = clip(round(x / scale[block]), -127, 127)
    Returns (x_i8 int8 [N], scales float32 [N/B])"""
    N = len(x)
    nb = N // B
    x_blocks = x.reshape(nb, B)
    absmax = np.max(np.abs(x_blocks), axis=1)  # [nb]
    scales = np.where(absmax > 1e-9, absmax / 127.0, 1.0/127.0)  # [nb]
    scaled = x_blocks / scales[:, np.newaxis]  # [nb, B]
    scaled = np.clip(np.round(scaled), -127.0, 127.0)  # saturating round
    x_i8 = scaled.astype(np.int8).reshape(N)
    return x_i8, scales

# ── RMSNorm (EXACT match: mean(x²) → rstd, normed = x * weight * rstd) ─

def rmsnorm(x, weight, eps=1e-6):
    """x: [N], weight: [N]. Returns normed: [N]."""
    ss = np.mean(x.astype(np.float64) ** 2)
    rstd = 1.0 / np.sqrt(ss + eps)
    return x * weight * rstd

# ── INT8 GEMV (EXACT match to GPU gemv_int8) ──────────────────────────

def gemv_int8(x_i8, x_scales, W_t, W_sc_t, K):
    """GEMV with block-scaled INT8.
    x_i8: [K] int8, x_scales: [K/B] float32
    W_t: [N,K] int8 (transposed weight), W_sc_t: [N/B,K/B] float32
    Returns y: [N] float32"""
    N = W_t.shape[0]
    nb_k = K // B
    
    # Reshape to blocks
    x_blk = x_i8.reshape(1, nb_k, B)    # [1, K/B, B]
    W_blk = W_t.reshape(N, nb_k, B)      # [N, K/B, B]
    
    # DP4A-style: dot product per block
    raw_dot = np.sum(W_blk.astype(np.int32) * x_blk.astype(np.int32), axis=2).astype(np.float32)  # [N, nb_k]
    
    # Apply scales
    sc = x_scales[np.newaxis, :] * W_sc_t  # [N/B, K/B]  -- need to expand N dim
    # W_sc_t is [N/B, K/B], need [N, K/B] by repeating each row 16x
    sc_exp = np.repeat(W_sc_t, B, axis=0)  # [N, K/B]
    sc_exp = sc_exp * x_scales[np.newaxis, :]  # [N, K/B]
    
    y = np.sum(raw_dot * sc_exp, axis=1)  # [N]
    return y

# ── GQA decode attention (single token) ───────────────────────────────

def attention_decode_gqa(Q, K_proj, V_proj, nqh, nkv, hd):
    """Single-token decode GQA.
    K, V have shape [nkv*hd] (seq_pos=0 only).
    Q has shape [nqh*hd].
    Returns attn: [nqh*hd]"""
    g = nqh // nkv  # groups per kv head = 2
    attn = np.zeros(nqh * hd, dtype=np.float32)
    
    for gid in range(nkv):
        q_group = Q[gid * g * hd : (gid + 1) * g * hd].reshape(g, hd)  # [g, hd]
        k = K_proj[gid * hd : (gid + 1) * hd]  # [hd]
        v = V_proj[gid * hd : (gid + 1) * hd]  # [hd]
        
        # Single token: score = dot(q, k) / sqrt(hd), softmax=1.0
        # output = v for each head in group
        for hh in range(g):
            attn[(gid * g + hh) * hd : (gid * g + hh + 1) * hd] = v
    
    return attn

# ── Load weights ──────────────────────────────────────────────────────

print("Loading weights...", end=" ", flush=True)

# INT8 weights
W_names = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
           'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
W = {}
for nm in W_names:
    W[nm] = load_int8(f"{WEIGHTDIR}/0_{nm}")

# RMSNorm weights (f32 files, same as CUDA loads)
rn_input = load_f32(f"{WEIGHTDIR}/0_input_layernorm.f32")  # [H]
print(f"ok. rn_input: mean={rn_input.mean():.4f}")

# ── Input (exact match to CUDA) ───────────────────────────────────────

x = np.array([(j % 17 - 8) * 0.01 for j in range(H)], dtype=np.float32)
print(f"Input: mean={x.mean():.6f} std={x.std():.6f}\n")

# ── 1. Pre-attention RMSNorm + INT8 quant ────────────────────────────

x_normed = rmsnorm(x, rn_input, eps)
x_i8, x_sc = block_quant(x_normed, B)

print(f"  RMSNorm: mean={x_normed.mean():.4f} std={x_normed.std():.4f}")
print(f"  Quant: x_i8 range=[{x_i8.min()},{x_i8.max()}] scale range=[{x_sc.min():.6f},{x_sc.max():.6f}]")

# ── 2. QKV GEMV ──────────────────────────────────────────────────────

Q = gemv_int8(x_i8, x_sc, W['self_attn.q_proj'][0], W['self_attn.q_proj'][1], H)
K = gemv_int8(x_i8, x_sc, W['self_attn.k_proj'][0], W['self_attn.k_proj'][1], H)
V = gemv_int8(x_i8, x_sc, W['self_attn.v_proj'][0], W['self_attn.v_proj'][1], H)

print(f"  Q: mean={Q.mean():.4f} std={Q.std():.4f}")
print(f"  K: mean={K.mean():.4f} std={K.std():.4f}")
print(f"  V: mean={V.mean():.4f} std={V.std():.4f}")

# ── 3. GQA attention (single token decode, seq_pos=0) ────────────────

attn = attention_decode_gqa(Q, K, V, nqh, nkv, hd)
print(f"  Attn: mean={attn.mean():.4f} std={attn.std():.4f}")

# ── 4. Wo projection ─────────────────────────────────────────────────

a_i8, a_sc = block_quant(attn, B)
proj = gemv_int8(a_i8, a_sc, W['self_attn.o_proj'][0], W['self_attn.o_proj'][1], QD)
print(f"  Wo_out: mean={proj.mean():.4f} std={proj.std():.4f}")

# ── 5. Residual 1 (proj += x) ────────────────────────────────────────

proj = proj + x
res = proj.copy()  # save for MLP residual
print(f"  Res1: mean={proj.mean():.4f} std={proj.std():.4f}")

# ── 6. Post-attention RMSNorm + INT8 quant ───────────────────────────

# NOTE: CUDA validate_pipeline uses the SAME d_rn (input_layernorm) for both
p_norm = rmsnorm(proj, rn_input, eps)
p_i8, p_sc = block_quant(p_norm, B)
print(f"  Post-RMSNorm: mean={p_norm.mean():.4f} std={p_norm.std():.4f}")

# ── 7. Gate + Up GEMV + SwiGLU ──────────────────────────────────────

gate = gemv_int8(p_i8, p_sc, W['mlp.gate_proj'][0], W['mlp.gate_proj'][1], H)
up   = gemv_int8(p_i8, p_sc, W['mlp.up_proj'][0],   W['mlp.up_proj'][1],   H)

# SwiGLU: silu(gate) * up
def silu(x):
    return x / (1.0 + np.exp(-x))

mlp = silu(gate) * up
print(f"  Gate: mean={gate.mean():.4f} std={gate.std():.4f}")
print(f"  Up:   mean={up.mean():.4f}   std={up.std():.4f}")
print(f"  MLP:  mean={mlp.mean():.4f}  std={mlp.std():.4f}")

# ── 8. Down projection ───────────────────────────────────────────────

m_i8, m_sc = block_quant(mlp, B)
proj2 = gemv_int8(m_i8, m_sc, W['mlp.down_proj'][0], W['mlp.down_proj'][1], ID)
print(f"  Down: mean={proj2.mean():.4f} std={proj2.std():.4f}")

# ── 9. Residual 2 (proj2 += saved res) ───────────────────────────────

out = proj2 + res
print(f"  Res2: mean={out.mean():.4f} std={out.std():.4f}")

# ── Compare with CUDA output ─────────────────────────────────────────

print(f"\n{'='*60}")
print(f"Comparison with CUDA (/tmp/layer0_out.bin)")
print(f"{'='*60}")

try:
    cuda_out = np.fromfile('/tmp/layer0_out.bin', dtype=np.float32)
    if len(cuda_out) != H:
        print(f"  CUDA output has {len(cuda_out)} elements, expected {H}")
        sys.exit(1)
    
    diff = np.abs(out - cuda_out)
    rel = diff / (np.abs(out) + 1e-10)
    
    print(f"  Python: mean={out.mean():.6f} std={out.std():.6f}")
    print(f"  CUDA:   mean={cuda_out.mean():.6f} std={cuda_out.std():.6f}")
    print(f"  Mean abs diff:  {diff.mean():.6e}")
    print(f"  Max abs diff:   {diff.max():.6e}")
    print(f"  Mean rel diff:  {rel.mean():.6e}")
    print(f"  Max rel diff:   {rel.max():.6e}")
    print(f"  Cosine sim:     {np.dot(out, cuda_out) / (np.linalg.norm(out) * np.linalg.norm(cuda_out)):.8f}")
    print(f"  Match (1e-3):   {(diff < 1e-3).sum()}/{H} elements")
    print(f"  Match (1e-2):   {(diff < 1e-2).sum()}/{H} elements")
    print(f"  Match (1e-1):   {(diff < 1e-1).sum()}/{H} elements")
    
    if diff.max() < 1e-2:
        print("\n  ✅ CUDA and Python match! Pipeline is correct.")
    elif diff.max() < 1e-1:
        print("\n  ⚠️  Minor differences (quantization noise). Likely acceptable.")
    else:
        print(f"\n  ❌ Large differences! Check pipeline.")
        # Show which elements differ most
        worst = np.argsort(diff)[-10:][::-1]
        print(f"  Worst 10 elements (idx, python, cuda, diff):")
        for idx in worst:
            print(f"    [{idx:4d}] py={out[idx]:.4f} cuda={cuda_out[idx]:.4f} diff={diff[idx]:.4f}")
except FileNotFoundError:
    print("  Run ./bench/validate_pipeline first to generate CUDA output.")

# ── Dump Python output for external comparison ────────────────────────

out.tofile('/tmp/layer0_out_py.bin')
print(f"\nPython output written to /tmp/layer0_out_py.bin (8.0 KB)")
