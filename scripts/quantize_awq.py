#!/usr/bin/env python3
"""Per-channel quantization for GatedDeltaNet models.

Uses 1 scale per output channel (BLOCK=K), instead of block-16 (BLOCK=16).
Smoother scaling may help SSM recurrent state stability.

Usage:
    python3 scripts/quantize_awq.py <input_int8_dir> <output_dir> [n_fp16_layers]
"""
import struct, json, os, sys, shutil, re
import numpy as np

def dequantize_w(int8_path, scale_path):
    """Load INT8 weight + scales, return FP32 weight matrix."""
    i8_data = open(int8_path, 'rb').read()
    sc_data = open(scale_path, 'rb').read()
    
    i8_h = np.frombuffer(i8_data[:20], dtype=np.int32)
    sc_h = np.frombuffer(sc_data[:20], dtype=np.int32)
    K, N = int(i8_h[0]), int(i8_h[1])
    BLOCK_w = int(i8_h[2])
    nblks = K // BLOCK_w
    
    i8 = np.frombuffer(i8_data[20:], dtype=np.int8).reshape(N, K)
    scales = np.frombuffer(sc_data[20:], dtype=np.float32).reshape(N, nblks)
    
    W_f32 = (i8.astype(np.float32).reshape(N, nblks, BLOCK_w) * scales[:, :, np.newaxis]).reshape(N, K)
    return W_f32

def quantize_per_channel(W_f32):
    """Per-channel quantization: 1 scale per output channel, BLOCK=K."""
    N, K = W_f32.shape
    scales = np.max(np.abs(W_f32), axis=1) / 127.0  # [N]
    scales = np.maximum(scales, 1e-10)
    
    # Broadcast
    sc_bc = np.repeat(scales[:, np.newaxis], K, axis=1)  # [N, K]
    q = np.round(W_f32 / sc_bc)
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, scales

def process_weight(int8_path, scale_path, out_dir, prefix):
    """Quantize one weight matrix with per-channel."""
    W_f32 = dequantize_w(int8_path, scale_path)
    K, N = W_f32.shape
    
    q, scales = quantize_per_channel(W_f32)
    
    # Write INT8
    int8_out = os.path.join(out_dir, f"{prefix}.int8_t")
    header = np.array([K, N, K], dtype=np.int32)  # BLOCK=K for per-channel
    with open(int8_out, 'wb') as f:
        f.write(header.tobytes())
        f.write(q.tobytes())
    
    # Write scales
    sc_out = os.path.join(out_dir, f"{prefix}.scale_t")
    sc_header = np.array([K, N, K, N, 1], dtype=np.int32)
    with open(sc_out, 'wb') as f:
        f.write(sc_header.tobytes())
        f.write(scales.astype(np.float32).tobytes())
    
    mb = (q.nbytes + scales.nbytes) / (1024*1024)
    print(f"  {os.path.basename(prefix)}: [{N}×{K}] {mb:.2f}MB scales={scales.shape}")

def main():
    if len(sys.argv) < 3:
        print("Usage: quantize_awq.py <input_dir> <output_dir> [n_fp16_layers]")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    n_fp16 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all quantized weight files
    int8_set = set()
    for fn in os.listdir(input_dir):
        if fn.endswith('.int8_t'):
            m = re.match(r'^(\d+)_(.+)\.int8_t$', fn)
            if m:
                layer = int(m.group(1))
                wname = m.group(2)
                if wname.startswith('embed_') or wname.startswith('lm_head'):
                    continue
                int8_set.add((layer, wname))
    
    layers = sorted(set(l for l, _ in int8_set))
    wnames = sorted(set(w for _, w in int8_set))
    NL = len(layers)
    
    print(f"Per-channel quantization: {NL} layers, {len(wnames)} weights/layer")
    print(f"FP16 first {n_fp16} layers, per-channel INT8 for rest")
    
    for l in layers:
        for wname in wnames:
            int8_path = f"{input_dir}/{l}_{wname}.int8_t"
            scale_path = f"{input_dir}/{l}_{wname}.scale_t"
            
            if not os.path.exists(int8_path):
                continue
            
            if l < n_fp16:
                shutil.copy2(int8_path, f"{output_dir}/{l}_{wname}.int8_t")
                shutil.copy2(scale_path, f"{output_dir}/{l}_{wname}.scale_t")
            else:
                process_weight(int8_path, scale_path, output_dir, f"{l}_{wname}")
        
        if l % 4 == 0 or l == NL - 1:
            print(f"  Layer {l}/{NL}")
    
    # Copy non-weight files
    for fn in os.listdir(input_dir):
        if fn.startswith('embed_') or fn.startswith('lm_head') or fn.endswith('.f32'):
            src = os.path.join(input_dir, fn)
            dst = os.path.join(output_dir, fn)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
    
    print(f"Done. Per-channel weights in {output_dir}/")

if __name__ == '__main__':
    main()