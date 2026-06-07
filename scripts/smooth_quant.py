#!/usr/bin/env python3
"""SmoothQuant: offline weight smoothing for INT8 LLM inference.

Migrates quantization difficulty from activations → weights by scaling
weight columns and absorbing into preceding RMSNorm. The matmul output
is mathematically identical in FP32; the benefit is smaller channel-wise
variation in activations entering the block-16 quantizer.

Usage: python3 scripts/smooth_quant.py <model_path> <weights_dir> <output_dir> [alpha]

Model path: HF model directory (safetensors + config.json)
Weights dir: existing INT8 quantized weights (from quantize_generic.py)
Output dir: new weights with SmoothQuant applied
Alpha (default 0.5): migration strength. 0=all to weights, 1=all to activations
"""
import struct, json, os, sys, glob, math
import numpy as np

BLOCK = 16
EPS = 1e-10

# ── Safetensor loading ─────────────────────────────────────────────────
def find_config(model_path):
    for path in [os.path.join(model_path, "config.json")]:
        if os.path.exists(path): return path
    snap = os.path.join(model_path, "snapshots")
    if os.path.isdir(snap):
        for s in sorted(os.listdir(snap)):
            cp = os.path.join(snap, s, "config.json")
            if os.path.exists(cp): return cp
    return None

def find_shards(model_path):
    shards = []
    seen = set()
    for base in [model_path]:
        for pat in ["model*.safetensors", "*.safetensors"]:
            for f in glob.glob(os.path.join(base, pat)):
                if f not in seen: shards.append(f); seen.add(f)
        sd = os.path.join(base, "snapshots")
        if os.path.isdir(sd):
            for s in sorted(os.listdir(sd)):
                for f in glob.glob(os.path.join(sd, s, "model*.safetensors")):
                    if f not in seen: shards.append(f); seen.add(f)
    return sorted(shards)

def load_headers(shard_paths):
    tensor_map = {}
    for sp in shard_paths:
        with open(sp, 'rb') as f:
            hlen = struct.unpack('Q', f.read(8))[0]
            hdr = json.loads(f.read(hlen).decode('utf-8'))
            for k, v in hdr.items():
                if isinstance(v, dict) and 'dtype' in v:
                    tensor_map[k] = (sp, v)
    return tensor_map

def read_tensor(tensor_map, name):
    """Read BF16 → FP32 tensor from safetensors."""
    sp, info = tensor_map[name]
    start, end = info['data_offsets']
    with open(sp, 'rb') as f:
        hlen = struct.unpack('Q', f.read(8))[0]
        f.seek(8 + hlen + start)
        raw = f.read(end - start)
    if info.get('dtype', 'BF16') in ('BF16', 'BFLOAT16', 'bfloat16'):
        u16 = np.frombuffer(raw, dtype=np.uint16).copy()
        return (u16.astype(np.uint32) << 16).view(np.float32).reshape(info['shape'])
    return np.frombuffer(raw, dtype=np.float32).reshape(info['shape'])

# ── INT8 weight loading ────────────────────────────────────────────────
def load_int8_weight(prefix):
    """Load INT8 weight, return fp32 array [N_out, K_in]."""
    p = f"{prefix}.int8_t"
    with open(p, 'rb') as f:
        h = np.frombuffer(f.read(20), dtype=np.int32)
    K_in, N_out, block = h[0], h[1], h[2]
    sz = K_in * N_out
    with open(p, 'rb') as f:
        f.read(20)
        i8 = np.frombuffer(f.read(sz), dtype=np.int8).reshape(N_out, K_in)
    sp = f"{prefix}.scale_t"
    with open(sp, 'rb') as f:
        sh = np.frombuffer(f.read(20), dtype=np.int32)
    nblks = sh[3]  # num_K_blks
    with open(sp, 'rb') as f:
        f.read(20)
        scales = np.frombuffer(f.read(nblks * sh[4] * 4), dtype=np.float32).reshape(N_out, nblks)
    # Dequant
    fp32 = np.zeros((N_out, K_in), dtype=np.float32)
    for r in range(N_out):
        for b in range(nblks):
            base = b * block
            end = min(base + block, K_in)
            fp32[r, base:end] = i8[r, base:end].astype(np.float32) * scales[r, b]
    return fp32

def load_int8_weight_with_scales(prefix):
    """Like load_int8_weight but returns fp32, scales, header info."""
    p = f"{prefix}.int8_t"
    with open(p, 'rb') as f:
        h = np.frombuffer(f.read(20), dtype=np.int32)
    K_in, N_out, block = int(h[0]), int(h[1]), int(h[2])
    nblks = K_in // block
    sz = K_in * N_out
    with open(p, 'rb') as f:
        f.read(20)
        i8 = np.frombuffer(f.read(sz), dtype=np.int8).reshape(N_out, K_in)
    # Load scales
    sp = f"{prefix}.scale_t"
    with open(sp, 'rb') as f:
        f.read(20)  # skip header
        scales = np.frombuffer(f.read(nblks * N_out * 4), dtype=np.float32).reshape(N_out, nblks)
    return i8, scales, K_in, N_out, block, nblks

# ── RMSNorm ────────────────────────────────────────────────────────────
def rmsnorm(x, w, eps=1e-6):
    """Apply RMSNorm. x: [N], w: [N]."""
    rms = np.sqrt(np.mean(x.astype(np.float64)**2) + eps)
    return (x / rms).astype(np.float32) * w

# ── GEMM helper ────────────────────────────────────────────────────────
def gemv(x, W):
    """Matrix-vector multiply: y = x @ W^T. x: [K], W: [N, K]."""
    return (x.astype(np.float64) @ W.astype(np.float64).T).astype(np.float32)

def swiglu(g, u):
    """SiLU(gate) * up."""
    return (1.0 / (1.0 + np.exp(-g.astype(np.float64))) * u.astype(np.float64)).astype(np.float32)

# ── RoPE ───────────────────────────────────────────────────────────────
def apply_rope(q, k, pos, head_dim, rope_theta=1000000.0):
    """Apply rotary position embeddings in-place to fp32 arrays."""
    nq = q.shape[0] // head_dim
    nk = k.shape[0] // head_dim
    for h in range(nq):
        for d in range(head_dim // 2):
            th = pos * rope_theta ** (-2.0 * d / head_dim)
            c, s = math.cos(th), math.sin(th)
            i = h * head_dim + d * 2
            x, y = q[i], q[i+1]
            q[i], q[i+1] = x * c - y * s, x * s + y * c
    for h in range(nk):
        for d in range(head_dim // 2):
            th = pos * rope_theta ** (-2.0 * d / head_dim)
            c, s = math.cos(th), math.sin(th)
            i = h * head_dim + d * 2
            x, y = k[i], k[i+1]
            k[i], k[i+1] = x * c - y * s, x * s + y * c

def head_norm(d, nh, hd, w):
    """Apply head norm. d: [nh*hd], w: [hd] (shared across heads)."""
    out = d.copy()
    for h in range(nh):
        sl = d[h*hd:(h+1)*hd]
        rms = np.sqrt(np.mean(sl.astype(np.float64)**2) + 1e-6)
        out[h*hd:(h+1)*hd] = (sl / rms * w).astype(np.float32)
    return out

# ── Calibration text ───────────────────────────────────────────────────
CALIB_TEXT = (
    "The Republic of Austria is a federal republic in Central Europe . "
    "It is bordered by Germany to the northwest , the Czech Republic to the north , "
    "Slovakia to the northeast , Hungary to the east , Slovenia and Italy to the south ."
)

# ── Main ───────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_path> <weights_dir> <output_dir> [alpha]")
        sys.exit(1)
    model_path = sys.argv[1]
    weights_dir = sys.argv[2]
    out_dir = sys.argv[3]
    alpha = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

    # Load config
    cf = find_config(model_path)
    with open(cf) as f:
        cfg = json.load(f)
    NL = cfg['num_hidden_layers']
    H = cfg['hidden_size']
    NQ = cfg['num_attention_heads']
    NKV = cfg.get('num_key_value_heads', NQ)
    HD = cfg.get('head_dim', 128)
    ID = cfg['intermediate_size']
    V = cfg.get('vocab_size', 151936)
    Q = NQ * HD
    KV = NKV * HD
    rope_theta = cfg.get('rope_theta', 10000.0)
    print(f"Model: {NL}L {H}H {NQ}q {NKV}k {HD}hd {ID}i {V}v")
    print(f"Alpha: {alpha}")

    os.makedirs(out_dir, exist_ok=True)

    # Load safetensor headers for reading raw BF16 weights
    shards = find_shards(model_path)
    print(f"Shards: {len(shards)}")
    tensor_map = load_headers(shards)

    # ── Per-layer calibration ──────────────────────────────────────
    # We run forward through the model on calibration text and record
    # per-channel activation max after each RMSNorm.
    # Store: act_max_attn[l][j] for layer l, channel j (attention input)
    #        act_max_mlp[l][j]  for layer l, channel j (MLP input)
    
    print("\n=== Phase 1: Calibration (forward pass) ===")
    
    # Load embed tokens for calibration
    print("Loading embed tokens...")
    emb_i8, emb_sc, emb_K, emb_N, _, emb_nblks = load_int8_weight_with_scales(f"{weights_dir}/embed_tokens")
    # emb has shape [N, K] where N=V, K=H and header says [K, N] but data is [N, K]
    print(f"  embed tokens: {emb_N}×{emb_K}")
    
    # Pre-dequant embed: [V, H]
    emb_fp32 = np.zeros((emb_N, emb_K), dtype=np.float32)
    for r in range(emb_N):
        block = 16
        for b in range(emb_K // block):
            base = b * block
            emb_fp32[r, base:base+block] = emb_i8[r, base:base+block].astype(np.float32) * emb_sc[r, b]
    
    # Load RMSNorm weights
    print("Loading RMSNorm weights...")
    rn_in = [None] * NL
    rn_post = [None] * NL
    for l in range(NL):
        p = f"{weights_dir}/{l}_input_layernorm.f32"
        rn_in[l] = np.frombuffer(open(p, 'rb').read(), dtype=np.float32).copy()
        p = f"{weights_dir}/{l}_post_attention_layernorm.f32"
        rn_post[l] = np.frombuffer(open(p, 'rb').read(), dtype=np.float32).copy()
    
    # Load final norm
    fn_path = f"{weights_dir}/final_norm.f32"
    fn = np.frombuffer(open(fn_path, 'rb').read(), dtype=np.float32).copy()
    
    # Load QK norms
    qkn = np.frombuffer(open(f"{weights_dir}/qk_norms.f32", 'rb').read(), dtype=np.float32).copy()
    
    # Load lm_head if exists
    lm_head = None
    if os.path.exists(f"{weights_dir}/lm_head.int8_t"):
        print("Loading lm_head...")
        lm_i8, lm_sc, lm_K, lm_N, _, lm_nblks = load_int8_weight_with_scales(f"{weights_dir}/lm_head")
        # Dequant
        lm_fp32 = np.zeros((lm_N, lm_K), dtype=np.float32)
        for r in range(lm_N):
            block = 16
            for b in range(lm_K // block):
                base = b * block
                lm_fp32[r, base:base+block] = lm_i8[r, base:base+block].astype(np.float32) * lm_sc[r, b]
        lm_head = lm_fp32  # shape [V, H]
    
    # Process calibration text
    # Use simple BPE-free tokenization: just byte values for rough calibration
    calib_ids = [ord(c) % V for c in CALIB_TEXT.replace(" ", "Ġ")[:100]]
    calib_ids = [min(max(t, 0), V-1) for t in calib_ids[:20]]
    print(f"Calibration tokens: {len(calib_ids)}")
    
    # Per-channel activation max storage
    act_max_attn = [np.zeros(H, dtype=np.float64) for _ in range(NL)]
    act_max_mlp = [np.zeros(H, dtype=np.float64) for _ in range(NL)]
    n_act_attn = [0 for _ in range(NL)]
    n_act_mlp = [0 for _ in range(NL)]
    
    # Load first layer weights for calibration (load more if needed)
    # For calibration, we need weights for ALL layers since we do full forward pass
    # Dequant all weights to FP32 for CPU forward pass
    # This is memory-heavy but practical for 1.7B (~8 GB FP32)
    
    print("Loading all INT8 weights and dequantizing to FP32...")
    # We store weight matrices to run forward pass
    W_q = [None]*NL; W_k = [None]*NL; W_v = [None]*NL
    W_o = [None]*NL; W_g = [None]*NL; W_u = [None]*NL; W_d = [None]*NL
    
    total_mb = 0
    for l in range(NL):
        pfx = f"{weights_dir}/{l}"
        W_q[l] = load_int8_weight(f"{pfx}_self_attn.q_proj")
        W_k[l] = load_int8_weight(f"{pfx}_self_attn.k_proj")
        W_v[l] = load_int8_weight(f"{pfx}_self_attn.v_proj")
        W_o[l] = load_int8_weight(f"{pfx}_self_attn.o_proj")
        W_g[l] = load_int8_weight(f"{pfx}_mlp.gate_proj")
        W_u[l] = load_int8_weight(f"{pfx}_mlp.up_proj")
        W_d[l] = load_int8_weight(f"{pfx}_mlp.down_proj")
        mb = sum(w.nbytes for w in [W_q[l],W_k[l],W_v[l],W_o[l],W_g[l],W_u[l],W_d[l]]) / 1e6
        total_mb += mb
        if l % 7 == 0:
            print(f"  layer {l}/{NL} ({mb:.0f} MB)...")
    print(f"  Total: {total_mb:.0f} MB FP32")
    
    # Forward pass for calibration
    print("\nForward pass for calibration...")
    KV_SZ = 2048  # max seq len for KV cache
    k_cache = np.zeros((NL, NKV * HD * KV_SZ), dtype=np.float32)
    v_cache = np.zeros((NL, NKV * HD * KV_SZ), dtype=np.float32)
    
    for s in range(len(calib_ids) - 1):
        tok = calib_ids[s]
        if s % 5 == 0:
            print(f"  token {s}/{len(calib_ids)-1}...")
        
        # Embed (column access: emb_fp32[H, V], so h = emb_fp32[:, tok])
        h = emb_fp32[:, tok].copy()  # [H]
        
        for l in range(NL):
            # ── Input RMSNorm → activation max for attention
            a_attn = rmsnorm(h, rn_in[l], 1e-6)
            act_max_attn[l] = np.maximum(act_max_attn[l], np.abs(a_attn).astype(np.float64))
            n_act_attn[l] += 1
            
            # QKV
            q = gemv(a_attn, W_q[l])  # [Q]
            k = gemv(a_attn, W_k[l])  # [KV]
            v = gemv(a_attn, W_v[l])  # [KV]
            
            # Head norm
            qkn_l_q = qkn[l*2*HD:l*2*HD+HD]
            qkn_l_k = qkn[l*2*HD+HD:l*2*HD+2*HD]
            q = head_norm(q, NQ, HD, qkn_l_q)
            k = head_norm(k, NKV, HD, qkn_l_k)
            
            # RoPE
            apply_rope(q, k, s, HD, rope_theta)
            
            # KV cache update (simplified)
            # k_cache[l] has NKV * HD * KV_SZ elements. No kv_off needed (layer is [l]).
            for h in range(NKV):
                base = h * HD
                pos_start = h * KV_SZ * HD + s * HD
                pos_end = h * KV_SZ * HD + (s+1) * HD
                k_cache[l, pos_start:pos_end] = k[base:base+HD]
                v_cache[l, pos_start:pos_end] = v[base:base+HD]
            
            # Attention (simplified GQA: MQA-like)
            attn_out = np.zeros(NQ * HD, dtype=np.float32)
            for h_q in range(NQ):
                h_kv = h_q // (NQ // NKV)
                q_slice = q[h_q*HD:(h_q+1)*HD]
                max_seq = min(s + 1, KV_SZ)
                scores = np.zeros(max_seq, dtype=np.float64)
                for t in range(max_seq):
                    k_slice = k_cache[l, h_kv * KV_SZ * HD + t * HD:h_kv * KV_SZ * HD + (t+1) * HD]
                    scores[t] = np.dot(q_slice.astype(np.float64), k_slice.astype(np.float64)) / math.sqrt(HD)
                # Softmax
                scores_exp = np.exp(scores - np.max(scores))
                scores_sm = scores_exp / np.sum(scores_exp)
                # Weighted sum
                out_slice = np.zeros(HD, dtype=np.float64)
                for t in range(max_seq):
                    v_slice = v_cache[l, h_kv * KV_SZ * HD + t * HD:h_kv * KV_SZ * HD + (t+1) * HD]
                    out_slice += scores_sm[t] * v_slice.astype(np.float64)
                attn_out[h_q*HD:(h_q+1)*HD] = out_slice.astype(np.float32)
            
            # Output projection
            proj = gemv(attn_out, W_o[l])  # [H]
            h = h + proj  # residual
            
            # ── Post-attention RMSNorm → activation max for MLP
            a_mlp = rmsnorm(h, rn_post[l], 1e-6)
            act_max_mlp[l] = np.maximum(act_max_mlp[l], np.abs(a_mlp).astype(np.float64))
            n_act_mlp[l] += 1
            
            # MLP
            gate = gemv(a_mlp, W_g[l])  # [ID]
            up = gemv(a_mlp, W_u[l])  # [ID]
            mlp = swiglu(gate, up)
            down = gemv(mlp, W_d[l])  # [H]
            h = h + down  # residual
        
        # Final RMSNorm
        h = rmsnorm(h, fn, 1e-6)
        
        # lm_head (optional, not needed for calibration)
        # logits = gemv(h, lm_head) if lm_head is not None else gemv(h, emb_fp32)
    
    # Average activation max across observed tokens
    for l in range(NL):
        if n_act_attn[l] > 0:
            act_max_attn[l] = act_max_attn[l]  # already max across tokens
        if n_act_mlp[l] > 0:
            act_max_mlp[l] = act_max_mlp[l]
    
    # Print some stats
    print("\nCalibration stats:")
    for l in range(min(5, NL)):
        print(f"  Layer {l}: attn max per-channel: min={np.min(act_max_attn[l]):.4f} max={np.max(act_max_attn[l]):.4f} mean={np.mean(act_max_attn[l]):.4f}")
        print(f"            mlp  max per-channel: min={np.min(act_max_mlp[l]):.4f}  max={np.max(act_max_mlp[l]):.4f}  mean={np.mean(act_max_mlp[l]):.4f}")
    
    # ── Phase 2: Compute smoothing factors ─────────────────────────
    print("\n=== Phase 2: Computing smoothing factors ===")
    
    s_attn_list = []
    s_mlp_list = []
    
    for l in range(NL):
        # Attention block: Q, K, V share the same input
        # Per-channel weight max across Q, K, V columns
        Wq = W_q[l]  # [Q, H]
        Wk = W_k[l]  # [KV, H]
        Wv = W_v[l]  # [KV, H]
        col_max_W = np.zeros(H, dtype=np.float64)
        for j in range(H):
            col_max_W[j] = max(
                np.max(np.abs(Wq[:, j].astype(np.float64))),
                np.max(np.abs(Wk[:, j].astype(np.float64))),
                np.max(np.abs(Wv[:, j].astype(np.float64)))
            )
        col_max_W = np.maximum(col_max_W, EPS)
        
        # Activation max (per channel)
        act_max = np.maximum(act_max_attn[l], EPS)
        
        # Smoothing factor
        s_attn = (act_max ** alpha) / (col_max_W ** (1 - alpha))
        # Normalize to avoid extreme values
        s_attn_mean = np.mean(s_attn)
        s_attn = s_attn / s_attn_mean  # Keep mean=1
        
        s_attn_list.append(s_attn)
        
        # MLP block: gate, up, down share the same post-attention input
        Wg = W_g[l]  # [ID, H]
        Wu = W_u[l]  # [ID, H]
        Wd = W_d[l]  # [H, ID] — but smoothing is on the INPUT side (H dim), not ID
        # Actually Wd has shape [H, ID], input is [ID], output is [H]
        # Smoothing s_mlp applies to the input of this sub-block
        # The input of the MLP block (after RMSNorm) goes to gate and up.
        # Gate and up have shape [ID, H] — input dimension H = columns
        # down has shape [H, ID] — we don't smooth on the ID side (output of gate/up)
        
        col_max_W_mlp = np.zeros(H, dtype=np.float64)
        for j in range(H):
            col_max_W_mlp[j] = max(
                np.max(np.abs(Wg[:, j].astype(np.float64))),
                np.max(np.abs(Wu[:, j].astype(np.float64)))
            )
        col_max_W_mlp = np.maximum(col_max_W_mlp, EPS)
        
        act_max_mlp_l = np.maximum(act_max_mlp[l], EPS)
        
        s_mlp = (act_max_mlp_l ** alpha) / (col_max_W_mlp ** (1 - alpha))
        s_mlp_mean = np.mean(s_mlp)
        s_mlp = s_mlp / s_mlp_mean
        
        s_mlp_list.append(s_mlp)
    
    # Print smoothing factor stats
    for l in range(min(3, NL)):
        print(f"  Layer {l}: s_attn min={np.min(s_attn_list[l]):.4f} max={np.max(s_attn_list[l]):.4f} mean={np.mean(s_attn_list[l]):.4f}")
        print(f"           s_mlp  min={np.min(s_mlp_list[l]):.4f}  max={np.max(s_mlp_list[l]):.4f}  mean={np.mean(s_mlp_list[l]):.4f}")
    
    # ── Phase 3: Apply smoothing to all layers ──────────────────────
    print("\n=== Phase 3: Applying smoothing ===")
    
    def quantize_int8_block(W_fp32):
        """Quantize FP32 to INT8 with per-row block-16 scaling.
        W_fp32: [N_out, K_in]
        Returns: i8 [N_out, K_in], scales [N_out, nblks]
        """
        N_out, K_in = W_fp32.shape
        nblks = K_in // BLOCK
        W_blk = W_fp32.reshape(N_out, nblks, BLOCK)
        scales = np.max(np.abs(W_blk), axis=2) / 127.0
        scales = np.maximum(scales, 1e-10)
        i8 = np.clip(np.round(W_fp32 / scales[:, :, np.newaxis].repeat(BLOCK, axis=2).reshape(N_out, K_in)), -128, 127).astype(np.int8)
        return i8, scales
    
    def write_new_weight(prefix, i8_data, scales, K_in, N_out):
        """Write INT8 weight in block-16 format."""
        nblks = K_in // BLOCK
        header = np.array([K_in, N_out, BLOCK, nblks, N_out], dtype=np.int32)
        with open(f"{prefix}.int8_t", 'wb') as f:
            f.write(header.tobytes())
            f.write(i8_data.tobytes())
        with open(f"{prefix}.scale_t", 'wb') as f:
            f.write(header.tobytes())
            f.write(scales.tobytes())
    
    for l in range(NL):
        # ── Attention block: apply s_attn ──
        s_a = s_attn_list[l].astype(np.float32)  # [H]
        
        # Q, K, V weights: multiply columns by s_a
        W_q_s = (W_q[l].astype(np.float64) * s_a[np.newaxis, :]).astype(np.float32)
        W_k_s = (W_k[l].astype(np.float64) * s_a[np.newaxis, :]).astype(np.float32)
        W_v_s = (W_v[l].astype(np.float64) * s_a[np.newaxis, :]).astype(np.float32)
        
        # Re-quantize
        i8_q, sc_q = quantize_int8_block(W_q_s)
        i8_k, sc_k = quantize_int8_block(W_k_s)
        i8_v, sc_v = quantize_int8_block(W_v_s)
        
        pfx = f"{out_dir}/{l}"
        write_new_weight(f"{pfx}_self_attn.q_proj", i8_q, sc_q, H, Q)
        write_new_weight(f"{pfx}_self_attn.k_proj", i8_k, sc_k, H, KV)
        write_new_weight(f"{pfx}_self_attn.v_proj", i8_v, sc_v, H, KV)
        
        # O projection: input Q dim, output H dim
        # The smoothing was applied to the activation input of Q/K/V, not O
        # O's input is the attention output which was already smoothed.
        # Re-quantize O with original weights (no column scaling needed for O)
        # Actually O takes [Q] as input and outputs [H]. The Q input is NOT affected
        # by smoothing (which targets the RMSNorm output before Q projection).
        # The attention output goes back to O projection without an intervening RMSNorm.
        # So O needs no special handling.
        # Actually wait — O takes attention output as input. The attention output
        # is computed from values which were projected from the same smoothed input.
        # But the smoothing factor was on the INPUT to K/V projection, not the OUTPUT.
        # So V values are already smoothed (divided by s_a in each channel), but the
        # V projection amplifies them back (multiplies by s_a in the weight).
        # The attention output is in the original scale because V_proj is smoothed.
        # So O's input is NOT in the smoothed domain.
        # Hmm, actually O's input is the attention output which is computed from
        # smoothed V values with smoothed V weights. So the attention output is in
        # the ORIGINAL domain (V_smoothed * W_v_smoothed = V_original * W_v_original).
        # So O needs no column scaling. Just re-quantize.
        W_o_s = W_o[l]  # no column scaling
        i8_o, sc_o = quantize_int8_block(W_o_s)
        write_new_weight(f"{pfx}_self_attn.o_proj", i8_o, sc_o, Q, H)
        
        # Input RMSNorm: divide weight by s_a
        rn_in_s = (rn_in[l].astype(np.float64) / (s_a + EPS)).astype(np.float32)
        rn_in[l] = rn_in_s
        rn_in[l].tofile(f"{out_dir}/{l}_input_layernorm.f32")
        
        # ── MLP block: apply s_mlp ──
        s_m = s_mlp_list[l].astype(np.float32)  # [H]
        
        # Gate, up weights: multiply columns by s_m
        W_g_s = (W_g[l].astype(np.float64) * s_m[np.newaxis, :]).astype(np.float32)
        W_u_s = (W_u[l].astype(np.float64) * s_m[np.newaxis, :]).astype(np.float32)
        
        # Down weight: input is ID dim, output is H dim
        # The smoothing was applied to the activation input of gate/up (H dim)
        # Down takes the gated output (ID dim) as input. The ID dim is not smoothed.
        # But the output of down goes into the H-dim residual, which was smoothed
        # by s_m. Wait no — the residual is in the ORIGINAL domain.
        # Actually, after down projection: y = W_d @ x where x is MLP output (ID dim)
        # The output y goes into the residual. The residual path is UNSMOOTHED.
        # So W_d needs no special handling from our smoothing.
        # However, our smoothing of the INPUT to gate/up means those outputs are
        # weighted differently. After swiglu(gate*up) and down projection, the
        # output should be in the original domain.
        # Actually, gate and up outputs are: W_g_s * a_mlp / s_m = W_g * s_m * a_mlp / s_m = W_g * a_mlp
        # So gate and up outputs are unchanged! And swiglu is unchanged!
        # Down output is: W_d @ swiglu = unchanged.
        
        W_d_s = W_d[l]  # no column scaling needed
        
        i8_g, sc_g = quantize_int8_block(W_g_s)
        i8_u, sc_u = quantize_int8_block(W_u_s)
        i8_d, sc_d = quantize_int8_block(W_d_s)
        
        write_new_weight(f"{pfx}_mlp.gate_proj", i8_g, sc_g, H, ID)
        write_new_weight(f"{pfx}_mlp.up_proj", i8_u, sc_u, H, ID)
        write_new_weight(f"{pfx}_mlp.down_proj", i8_d, sc_d, ID, H)
        
        # Post-attention RMSNorm: divide weight by s_m
        rn_post_s = (rn_post[l].astype(np.float64) / (s_m + EPS)).astype(np.float32)
        rn_post[l] = rn_post_s
        rn_post[l].tofile(f"{out_dir}/{l}_post_attention_layernorm.f32")
        
        if l % 7 == 0:
            print(f"  Layer {l}/{NL} smoothed and re-quantized")
    
    # ── Copy unchanged files ──────────────────────────────────────────
    import shutil
    
    # qk norms
    shutil.copy2(f"{weights_dir}/qk_norms.f32", f"{out_dir}/qk_norms.f32")
    print("  Copied qk_norms.f32")
    
    # final norm
    shutil.copy2(f"{weights_dir}/final_norm.f32", f"{out_dir}/final_norm.f32")
    print("  Copied final_norm.f32")
    
    # embed tokens (unchanged)
    for ext in ['.int8_t', '.scale_t']:
        src = f"{weights_dir}/embed_tokens{ext}"
        if os.path.exists(src):
            shutil.copy2(src, f"{out_dir}/embed_tokens{ext}")
    print("  Copied embed_tokens")
    
    # lm_head (unchanged)
    for ext in ['.int8_t', '.scale_t']:
        src = f"{weights_dir}/lm_head{ext}"
        if os.path.exists(src):
            shutil.copy2(src, f"{out_dir}/lm_head{ext}")
            print("  Copied lm_head")
    
    print(f"\nDone! Smoothed weights in {out_dir}/")
    print(f"  {NL} layers with α={alpha}")
    print("  To test: ./server/http_subprocess <out_dir> <port>")

if __name__ == "__main__":
    main()
