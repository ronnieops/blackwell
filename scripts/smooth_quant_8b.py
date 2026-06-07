"""Weight-only SmoothQuant for 8B model. No calibration needed.
Processes one layer at a time to fit in memory.

Usage: python3 scripts/smooth_quant_8b.py <weights_dir> <output_dir> [alpha]
"""
import numpy as np, os, sys, shutil

BLOCK = 16

def load_int8_weight(prefix, K_in=None, N_out=None):
    """Load INT8 weight, return fp32 array. Shape from file unless overridden."""
    p = f"{prefix}.int8_t"
    with open(p, 'rb') as f:
        h = np.frombuffer(f.read(20), dtype=np.int32)
    if K_in is None: K_in = int(h[0])
    if N_out is None: N_out = int(h[1])
    block = int(h[2])
    nblks = K_in // block
    sz = K_in * N_out
    with open(p, 'rb') as f:
        f.read(20)
        i8 = np.frombuffer(f.read(sz), dtype=np.int8).reshape(N_out, K_in)
    sp = f"{prefix}.scale_t"
    with open(sp, 'rb') as f:
        f.read(20)
        scales = np.frombuffer(f.read(nblks * N_out * 4), dtype=np.float32).reshape(N_out, nblks)
    fp32 = np.zeros((N_out, K_in), dtype=np.float32)
    for r in range(N_out):
        for b in range(nblks):
            base = b * block
            fp32[r, base:base+block] = i8[r, base:base+block].astype(np.float32) * scales[r, b]
    return fp32

def quantize_int8_block(W_fp32):
    N_out, K_in = W_fp32.shape
    nblks = K_in // BLOCK
    W_blk = W_fp32.reshape(N_out, nblks, BLOCK)
    scales = np.max(np.abs(W_blk), axis=2) / 127.0
    scales = np.maximum(scales, 1e-10)
    i8 = np.clip(np.round(W_fp32 / scales[:, :, np.newaxis].repeat(BLOCK, axis=2).reshape(N_out, K_in)), -128, 127).astype(np.int8)
    return i8, scales

def write_weight(prefix, i8_data, scales, K_in, N_out):
    nblks = K_in // BLOCK
    header = np.array([K_in, N_out, BLOCK, nblks, N_out], dtype=np.int32)
    with open(f"{prefix}.int8_t", 'wb') as f:
        f.write(header.tobytes()); f.write(i8_data.tobytes())
    with open(f"{prefix}.scale_t", 'wb') as f:
        f.write(header.tobytes()); f.write(scales.tobytes())

def compute_smoothing_factors(W_list, H, alpha):
    """Compute weight-only smoothing factors from column max of weight matrices.
    W_list: list of [N_out, H] matrices. Returns [H] smoothing factors.
    s_j = 1 / max_col_W_j^(1-alpha)
    """
    col_max = np.zeros(H, dtype=np.float64)
    for j in range(H):
        m = 0.0
        for W in W_list:
            m = max(m, np.max(np.abs(W[:, j].astype(np.float64))))
        col_max[j] = m
    col_max = np.maximum(col_max, 1e-10)
    s = 1.0 / (col_max ** (1.0 - alpha))
    s = s / np.mean(s)
    return s.astype(np.float32)

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <weights_dir> <output_dir> [alpha=0.3]")
        sys.exit(1)
    wdir = sys.argv[1]
    outdir = sys.argv[2]
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    
    H, Q, KV, ID = 4096, 4096, 1024, 12288
    NL = 36
    
    os.makedirs(outdir, exist_ok=True)
    print(f"8B Weight-only SmoothQuant α={alpha}")
    
    for l in range(NL):
        pfx = f"{wdir}/{l}"
        opfx = f"{outdir}/{l}"
        
        # ── Attention block ──
        Wq = load_int8_weight(f"{pfx}_self_attn.q_proj", H, Q)
        Wk = load_int8_weight(f"{pfx}_self_attn.k_proj", H, KV)
        Wv = load_int8_weight(f"{pfx}_self_attn.v_proj", H, KV)
        
        s_attn = compute_smoothing_factors([Wq, Wk, Wv], H, alpha)
        
        # Apply smoothing
        Wq_s = (Wq.astype(np.float64) * s_attn[np.newaxis, :]).astype(np.float32); del Wq
        Wk_s = (Wk.astype(np.float64) * s_attn[np.newaxis, :]).astype(np.float32); del Wk
        Wv_s = (Wv.astype(np.float64) * s_attn[np.newaxis, :]).astype(np.float32); del Wv
        
        i8_q, sc_q = quantize_int8_block(Wq_s); del Wq_s
        i8_k, sc_k = quantize_int8_block(Wk_s); del Wk_s
        i8_v, sc_v = quantize_int8_block(Wv_s); del Wv_s
        write_weight(f"{opfx}_self_attn.q_proj", i8_q, sc_q, H, Q)
        write_weight(f"{opfx}_self_attn.k_proj", i8_k, sc_k, H, KV)
        write_weight(f"{opfx}_self_attn.v_proj", i8_v, sc_v, H, KV)
        del i8_q, i8_k, i8_v, sc_q, sc_k, sc_v
        
        # O projection (no smoothing)
        Wo = load_int8_weight(f"{pfx}_self_attn.o_proj", Q, H)
        i8_o, sc_o = quantize_int8_block(Wo); del Wo
        write_weight(f"{opfx}_self_attn.o_proj", i8_o, sc_o, Q, H)
        del i8_o, sc_o
        
        # Input RMSNorm: divide by s_attn
        rn = np.frombuffer(open(f"{wdir}/{l}_input_layernorm.f32", 'rb').read(), dtype=np.float32).copy()
        (rn.astype(np.float64) / s_attn.astype(np.float64)).astype(np.float32).tofile(f"{outdir}/{l}_input_layernorm.f32")
        del rn, s_attn
        
        # ── MLP block ──
        Wg = load_int8_weight(f"{pfx}_mlp.gate_proj", H, ID)
        Wu = load_int8_weight(f"{pfx}_mlp.up_proj", H, ID)
        
        s_mlp = compute_smoothing_factors([Wg, Wu], H, alpha)
        
        Wg_s = (Wg.astype(np.float64) * s_mlp[np.newaxis, :]).astype(np.float32); del Wg
        Wu_s = (Wu.astype(np.float64) * s_mlp[np.newaxis, :]).astype(np.float32); del Wu
        
        i8_g, sc_g = quantize_int8_block(Wg_s); del Wg_s
        i8_u, sc_u = quantize_int8_block(Wu_s); del Wu_s
        write_weight(f"{opfx}_mlp.gate_proj", i8_g, sc_g, H, ID)
        write_weight(f"{opfx}_mlp.up_proj", i8_u, sc_u, H, ID)
        del i8_g, i8_u, sc_g, sc_u
        
        # Down projection (no smoothing)
        Wd = load_int8_weight(f"{pfx}_mlp.down_proj", ID, H)
        i8_d, sc_d = quantize_int8_block(Wd); del Wd
        write_weight(f"{opfx}_mlp.down_proj", i8_d, sc_d, ID, H)
        del i8_d, sc_d
        
        # Post-attention RMSNorm: divide by s_mlp
        rn_p = np.frombuffer(open(f"{wdir}/{l}_post_attention_layernorm.f32", 'rb').read(), dtype=np.float32).copy()
        (rn_p.astype(np.float64) / s_mlp.astype(np.float64)).astype(np.float32).tofile(f"{outdir}/{l}_post_attention_layernorm.f32")
        del rn_p, s_mlp
        
        if l % 6 == 0:
            print(f"  Layer {l}/{NL}")
    
    # Copy unchanged files
    for f in ['qk_norms.f32', 'final_norm.f32',
              'embed_tokens.int8_t', 'embed_tokens.scale_t',
              'lm_head.int8_t', 'lm_head.scale_t']:
        src = f"{wdir}/{f}"
        if os.path.exists(src):
            shutil.copy2(src, f"{outdir}/{f}")
    
    print(f"\nDone! 8B weight-only SmoothQuant (α={alpha}) in {outdir}")

if __name__ == "__main__":
    main()
