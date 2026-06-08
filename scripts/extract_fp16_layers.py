#!/usr/bin/env python3
"""Extract FP16 weights for first K layers of any model.
Auto-detects weight files (self_attn.*, linear_attn.*, mlp.*, etc.)

Usage: python3 scripts/extract_fp16_layers.py <int8_dir> <output_dir> <first_k>

Layers 0..K-1: FP16 weights (converted from INT8)
Layers K..NL-1: INT8 (copied)
"""
import struct, os, sys, shutil, re
import numpy as np

BLOCK = 16

def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <int8_dir> <output_dir> <first_k>")
        sys.exit(1)
    int8_dir = sys.argv[1]
    out_dir = sys.argv[2]
    K_fp16 = int(sys.argv[3])

    # Discover all INT8 weight files, infer NL and weight names
    i8_set = set()
    for fn in os.listdir(int8_dir):
        if fn.endswith('.int8_t') and not fn.startswith('embed_') and not fn.startswith('lm_head'):
            # Parse: {layer}_{name}.int8_t
            m = re.match(r'^(\d+)_(.+)\.int8_t$', fn)
            if m:
                i8_set.add((int(m.group(1)), m.group(2)))

    # Group by layer
    layers = sorted(set(l for l, _ in i8_set))
    NL = len(layers)
    if NL == 0:
        print("FAIL: no INT8 weight files found")
        sys.exit(1)

    # Group by weight name
    wnames = sorted(set(w for _, w in i8_set))
    
    # Get dimensions from first layer's q_proj or first available weight
    first_i8 = next((fn for fn in os.listdir(int8_dir) 
                     if fn.endswith('.int8_t') and re.match(r'^\d+_self_attn\.q_proj\.int8_t$', fn)
                     or fn.endswith('.int8_t') and re.match(r'^\d+_linear_attn\.in_proj_qkv\.int8_t$', fn)), None)
    if first_i8:
        with open(os.path.join(int8_dir, first_i8), 'rb') as f:
            h = np.frombuffer(f.read(20), dtype=np.int32)
        H = int(h[0])
        print(f"Detected: {NL} layers, H={H}, weight types: {wnames}")
    else:
        print(f"Detected: {NL} layers, weight types: {wnames}")

    print(f"FP16: first {K_fp16} layers, INT8: layers {K_fp16}..{NL-1}")
    
    os.makedirs(out_dir, exist_ok=True)

    for l in layers:
        for wname in wnames:
            src_i8 = f"{int8_dir}/{l}_{wname}.int8_t"
            src_sc = f"{int8_dir}/{l}_{wname}.scale_t"
            
            if not os.path.exists(src_i8):
                continue
            
            if l < K_fp16:
                # Convert to FP16
                with open(src_i8, 'rb') as f:
                    h = np.frombuffer(f.read(20), dtype=np.int32)
                KK, NN, block = int(h[0]), int(h[1]), int(h[2])
                nblks = KK // block
                
                with open(src_i8, 'rb') as f:
                    f.read(20)
                    i8 = np.frombuffer(f.read(KK*NN), dtype=np.int8).reshape(NN, KK)
                with open(src_sc, 'rb') as f:
                    f.read(20)
                    scales = np.frombuffer(f.read(nblks * NN * 4), dtype=np.float32).reshape(NN, nblks)
                
                # Dequant INT8 → FP32 → FP16
                fp32 = np.zeros((NN, KK), dtype=np.float32)
                for r in range(NN):
                    for b in range(nblks):
                        base = b * block
                        fp32[r, base:base+block] = i8[r, base:base+block].astype(np.float32) * scales[r, b]
                
                fp16_u16 = fp32.astype(np.float16).view(np.uint16)
                header = np.array([KK, NN], dtype=np.int32)
                with open(f"{out_dir}/{l}_{wname}.fp16", 'wb') as f:
                    f.write(header.tobytes())
                    f.write(fp16_u16.tobytes())
            else:
                # Copy INT8 files
                shutil.copy2(src_i8, f"{out_dir}/{l}_{wname}.int8_t")
                shutil.copy2(src_sc, f"{out_dir}/{l}_{wname}.scale_t")
        
        if l % 8 == 0 or l == NL - 1:
            print(f"  Layer {l}/{NL}")
    
    # Copy non-weight files (norms, etc.)
    for fname in os.listdir(int8_dir):
        src = os.path.join(int8_dir, fname)
        if os.path.isfile(src):
            # Skip embed/lm_head INT8 files (handled separately)
            if fname.endswith('.int8_t') and not fname.startswith('embed_') and not fname.startswith('lm_head'):
                continue
            if fname.endswith('.scale_t') and not fname.startswith('embed_') and not fname.startswith('lm_head'):
                continue
            dst = os.path.join(out_dir, fname)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
    
    # Copy embed and lm_head
    for fn in ['embed_tokens.int8_t', 'embed_tokens.scale_t', 'lm_head.int8_t', 'lm_head.scale_t']:
        src = os.path.join(int8_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, fn))
    
    print(f"\nDone! Mixed-precision weights in {out_dir}")

if __name__ == "__main__":
    main()
