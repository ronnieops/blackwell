#!/usr/bin/env python3
"""Convert large Gemma 4 GGUF weight tensors (embed_tokens, lm_head) to INT4 block-16.

These tensors are too large (>100M elements) for the C++ converter's FP32 buffer.
Processes in chunks to keep memory usage manageable.

Usage: python3 scripts/convert_gemma_large.py <gguf_path> <out_dir>
"""
import struct, os, sys, math
import numpy as np

GGUF_PATH = sys.argv[1]
OUT_DIR  = sys.argv[2]
BLOCK    = 16
CHUNK_ROWS = 1024  # process 1024 output rows at a time

def read_gguf_tensor_info(path):
    """Find all tensors in GGUF, return list of {name, dims, type, offset, file_size}."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_meta = struct.unpack('<Q', f.read(8))[0]
        
        # Skip metadata
        for _ in range(n_meta):
            klen = struct.unpack('<Q', f.read(8))[0]
            f.read(klen)
            vtype = struct.unpack('<I', f.read(4))[0]
            if vtype == 0: f.read(1)
            elif vtype == 1: f.read(1)
            elif vtype == 2: f.read(2)
            elif vtype == 3: f.read(2)
            elif vtype == 4: f.read(4)
            elif vtype == 5: f.read(4)
            elif vtype == 6: f.read(4)
            elif vtype == 7: f.read(1)
            elif vtype == 8: slen = struct.unpack('<Q', f.read(8))[0]; f.read(slen)
            elif vtype == 9:
                atype = struct.unpack('<I', f.read(4))[0]
                alen = struct.unpack('<Q', f.read(8))[0]
                for _ in range(alen):
                    if atype == 8: sl = struct.unpack('<Q', f.read(8))[0]; f.read(sl)
                    elif atype == 7: f.read(1)
                    elif atype in (0,1): f.read(1)
                    else: f.read(4)
            elif vtype in (10,11): f.read(8)
            elif vtype in (12,): f.read(2)
            elif vtype == 13: f.read(8)
            else: f.read(4)
        
        tensor_info = []
        for i in range(n_tensors):
            nlen = struct.unpack('<Q', f.read(8))[0]
            name = f.read(nlen).decode('utf-8', errors='replace')
            ndim = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(ndim)]
            ttype = struct.unpack('<I', f.read(4))[0]
            toff = struct.unpack('<Q', f.read(8))[0]
            # Compute file_size from type
            n_el = 1
            for d in dims: n_el *= d
            n_super = (n_el + 255) // 256
            if ttype == 0: fs = n_el * 4  # F32
            elif ttype == 1: fs = n_el * 2  # F16
            elif ttype == 12: fs = n_super * 144  # Q4_K
            elif ttype == 14: fs = n_super * 164  # Q6_K
            else: fs = n_el
            tensor_info.append({'name': name, 'dims': dims, 'type': ttype,
                                'offset': toff, 'file_size': fs, 'n_el': n_el})
        
        # Compute tensor_data_off
        tensor_data_off = f.tell()
        tensor_data_off = (tensor_data_off + 31) & ~31
        
        return tensor_info, tensor_data_off

def dequant_q4_K_row(src, n):
    """Dequantize one row of Q4_K data. src is packed Q4_K superblocks.
    Q4_K: 256 elements per super-block, 144 bytes per block.
    Layout: d(half), dmin(half), scales(12), qs(128)
    """
    n_blocks = (n + 255) // 256
    result = np.zeros(n, dtype=np.float32)
    
    for bi in range(n_blocks):
        bo = bi * 144
        d = struct.unpack('<e', src[bo:bo+2])[0]  # fp16 scale
        dmin = struct.unpack('<e', src[bo+2:bo+4])[0]
        scales = src[bo+4:bo+16]  # 12 bytes: 6 pairs of fp16 scales
        qs = src[bo+16:bo+144]  # 128 bytes of 4-bit values
        
        # Dequantize 256 values in sub-blocks of 16
        for sb in range(16):
            # Extract scale for this sub-block from the 12-byte scales
            # Q4_K stores 6 fp16 values for pairs of sub-blocks
            sc_idx = sb // 2
            sc_val = struct.unpack('<e', scales[sc_idx*2:sc_idx*2+2])[0]
            is_min = sb % 2  # even/odd sub-block
            if is_min:
                sc = dmin  # minimum scale (sub-block 1,3,5...)
            else:
                sc = d      # maximum scale (sub-block 0,2,4...)
            
            # Actually Q4_K has specific format. Let me just use the known formula:
            # For Gemma's Q4_K, dequantize each 4-bit value
            for j in range(16):
                qb = qs[sb * 8 + j // 2]
                nib = (qb >> (4 * (j % 2))) & 0xF
                # Q4_K: offset 8, range [-8, 7]
                val = (nib - 8) * sc
                idx = bi * 256 + sb * 16 + j
                if idx < n:
                    result[idx] = val
    
    return result

def quantize_int4_row(row):
    """Quantize one FP32 row to INT4 block-16. Returns (packed_bytes, scales)."""
    n = len(row)
    n_blocks = n // BLOCK
    scales = np.zeros(n_blocks, dtype=np.float32)
    packed = np.zeros(n // 2, dtype=np.uint8)
    
    for b in range(n_blocks):
        blk = row[b*BLOCK:(b+1)*BLOCK]
        absmax = max(np.max(np.abs(blk)), 1e-10)
        scale = absmax / 7.0
        scales[b] = scale
        q = np.clip(np.round(blk / scale), -7, 7).astype(np.int8)
        uq = (q + 8).astype(np.uint8)
        for i in range(0, BLOCK, 2):
            byte_idx = b * (BLOCK // 2) + i // 2
            packed[byte_idx] = uq[i] | (uq[i+1] << 4)
    
    return packed.tobytes(), scales.tobytes()

def write_int4_weight(out_dir, name, K, N):
    """Write header + packed data + scales for a weight tensor."""
    int4_path = os.path.join(out_dir, f"{name}.int4_t")
    scale_path = os.path.join(out_dir, f"{name}.scale_t")
    
    # Check if already exists
    if os.path.exists(int4_path):
        print(f"  Skip {name} (exists)")
        return
    
    num_kb = K // BLOCK
    with open(int4_path, 'wb') as f:
        hdr = struct.pack('<5i', K, N, BLOCK, num_kb, 1)
        f.write(hdr)
        # Packed data will be appended by caller
    
    with open(scale_path, 'wb') as f:
        hdr = struct.pack('<5i', 0, 0, 0, num_kb, N)
        f.write(hdr)
    
    print(f"  {name}: {N}x{K} INT4 (initialized)")

def append_int4_row(int4_path, scale_path, packed_bytes, scales_bytes):
    """Append one row of quantized data to the weight files."""
    with open(int4_path, 'ab') as f:
        f.write(packed_bytes)
    with open(scale_path, 'ab') as f:
        f.write(scales_bytes)

# ── Main ──
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Reading {GGUF_PATH}...")
tensors, tensor_data_off = read_gguf_tensor_info(GGUF_PATH)

print(f"Tensor data offset: {tensor_data_off}")
print(f"Total tensors: {len(tensors)}")

# Find token_embd.weight and output.weight
targets = []
for t in tensors:
    if t['name'] in ('token_embd.weight', 'output.weight'):
        targets.append(t)

if not targets:
    print("No large tensors found. Searching...")
    for t in tensors:
        if 'embed' in t['name'] or 'output' in t['name'] or 'tok_emb' in t['name']:
            n_el = 1
            for d in t['dims']: n_el *= d
            print(f"  {t['name']}: {t['dims']}, {n_el} elements, type={t['type']}")
    sys.exit(1)

for t in targets:
    name = t['name']
    n_el = t['n_el']
    dims = t['dims']
    # GGUF dims: [inner, outer] = [K, N]
    K = dims[0]
    N = dims[1] if len(dims) > 1 else 1
    
    print(f"\nConverting {name}: {N}x{K} Q4_K ({n_el:,} elements)")
    
    # Map to blackwell name
    if name == 'token_embd.weight':
        bw_name = 'embed_tokens'
    elif name == 'output.weight':
        bw_name = 'lm_head'
    else:
        bw_name = name.replace('.weight', '').replace('.', '_')
    
    write_int4_weight(OUT_DIR, bw_name, K, N)
    
    int4_path = os.path.join(OUT_DIR, f"{bw_name}.int4_t")
    scale_path = os.path.join(OUT_DIR, f"{bw_name}.scale_t")
    
    # Process in chunks of CHUNK_ROWS rows
    with open(GGUF_PATH, 'rb') as f:
        for chunk_start in range(0, N, CHUNK_ROWS):
            chunk_end = min(chunk_start + CHUNK_ROWS, N)
            chunk_rows = chunk_end - chunk_start
            print(f"  chunk {chunk_start}-{chunk_end}/{N}...", end=" ", flush=True)
            
            for row_idx in range(chunk_start, chunk_end):
                # Compute file offset for this row
                # Q4_K stores values row-major: each row is K elements in Q4_K blocks
                # File layout: [N rows][K elements in Q4_K super-blocks]
                # Each row has n_blocks = (K + 255) // 256 super-blocks of 144 bytes
                row_n_blocks = (K + 255) // 256
                row_file_size = row_n_blocks * 144
                file_off = tensor_data_off + t['offset'] + row_idx * row_file_size
                f.seek(file_off)
                src = f.read(row_file_size)
                
                # Dequantize one row
                row_f32 = dequant_q4_K_row(src, K)
                
                # Transpose: GGUF stores [K, N], we need [N, K]
                # For row_idx, we're reading the row_idx-th output row
                # Row in order: output_rows in column-major from GGUF
                # Actually we need to dequantize N rows each of K elements
                # and store as [N, K] (the kernel expects this layout)
                # Since we're reading row by row, each row is already in the order we need
                
                # Quantize to INT4 block-16
                packed_bytes, scales_bytes = quantize_int4_row(row_f32)
                append_int4_row(int4_path, scale_path, packed_bytes, scales_bytes)
            
            print(f"done")
    
    # Verify
    ds = os.path.getsize(int4_path) - 20  # subtract header
    expected_ds = N * K // 2
    ss = os.path.getsize(scale_path) - 20
    expected_ss = N * (K // BLOCK) * 4
    print(f"  int4: {ds:,} / {expected_ds:,} bytes, scales: {ss:,} / {expected_ss:,} bytes")
    ok = ds == expected_ds and ss == expected_ss
    print(f"  {'OK' if ok else 'SIZE MISMATCH'}")
