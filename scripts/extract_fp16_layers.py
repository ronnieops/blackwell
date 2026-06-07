#!/usr/bin/env python3
"""Extract FP16 weights for first K layers of a model.
Converts INT8 → FP32 → FP16 (lossless for dequantized values).

Usage: python3 scripts/extract_fp16_layers.py <int8_dir> <output_dir> <first_k>

Layers 0..K-1: FP16 weights
Layers K..NL-1: INT8 (copied from int8_dir)
"""
import struct, os, sys, shutil
import numpy as np

BLOCK = 16

def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <int8_dir> <output_dir> <first_k>")
        sys.exit(1)
    int8_dir = sys.argv[1]
    out_dir = sys.argv[2]
    K_fp16 = int(sys.argv[3])

    # Determine dimensions from weight headers
    test_path = os.path.join(int8_dir, "0_self_attn.q_proj.int8_t")
    with open(test_path, 'rb') as f:
        h = np.frombuffer(f.read(20), dtype=np.int32)
    H, Q = int(h[0]), int(h[1])
    
    with open(os.path.join(int8_dir, "0_self_attn.k_proj.int8_t"), 'rb') as f:
        KV = int(np.frombuffer(f.read(20), dtype=np.int32)[1])
    with open(os.path.join(int8_dir, "0_mlp.gate_proj.int8_t"), 'rb') as f:
        ID = int(np.frombuffer(f.read(20), dtype=np.int32)[1])
    
    NL = sum(1 for f in os.listdir(int8_dir) if f.endswith('_self_attn.q_proj.int8_t'))
    
    print(f"Detected: {NL}L H={H} Q={Q} KV={KV} ID={ID}")
    print(f"FP16: first {K_fp16} layers, INT8: layers {K_fp16}..{NL-1}")
    
    os.makedirs(out_dir, exist_ok=True)
    
    weight_names = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]
    
    for l in range(NL):
        for wname in weight_names:
            src_i8 = f"{int8_dir}/{l}_{wname}.int8_t"
            src_sc = f"{int8_dir}/{l}_{wname}.scale_t"
            
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
        
        if l % 7 == 0:
            print(f"  Layer {l}/{NL}")
    
    # Copy non-weight files
    for fname in os.listdir(int8_dir):
        src = os.path.join(int8_dir, fname)
        if os.path.isfile(src) and not any(fname.startswith(f"{l}_") for l in range(NL)):
            shutil.copy2(src, os.path.join(out_dir, fname))
    
    print(f"\nDone! Mixed-precision weights in {out_dir}")

if __name__ == "__main__":
    main()
