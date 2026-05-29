#!/usr/bin/env python3
"""Convert FP4/block-16/FP32-scale weights to NVF4 format.

NVF4 uses same block=16 FP4 E2M1 data but UE4M3 scales (1 byte vs 4 bytes).
This script reads existing FP4 weights and converts the FP32 scales to UE4M3.

UE4M3 format (unsigned E4M3):
  - 4-bit exponent (biased by 7), 3-bit mantissa
  - Max value: 448 (per NVIDIA spec)
  - Values: 2^(e-7) * (1 + m/8) for e in [0,15], m in [0,7]
  - Special: e=0,m=0 → 0 (zero)
  - Denorms: e=0,m≠0 → 2^(-6) * (m/8)

Scale conversion: FP32 → UE4M3
  - Clamp to [0, 448]
  - Round to nearest representable UE4M3 value
"""

import struct
import numpy as np
import sys
import os


def float_to_ue4m3(val: float) -> int:
    """Convert a non-negative float to UE4M3 (uint8)."""
    if val <= 0:
        return 0
    
    # Clamp to max
    val = min(val, 448.0)
    
    # Find exponent (biased)
    # val = 2^(e-7) * (1 + m/8)
    # log2(val) = e - 7 + log2(1 + m/8)
    import math
    log2_val = math.log2(val)
    
    # Unbiased exponent: floor(log2(val))
    exp_unbiased = int(math.floor(log2_val))
    
    # Biased exponent (4 bits)
    exp_biased = exp_unbiased + 7
    
    # Handle denorms (exp_biased == 0)
    if exp_biased <= 0:
        # Denormalized: 2^(-6) * (m/8)
        # m = val / 2^(-6) * 8 = val * 512
        mantissa = int(round(val * 512))
        mantissa = max(0, min(7, mantissa))
        return (0 << 3) | mantissa
    
    # Handle overflow (exp_biased >= 15)
    if exp_biased >= 15:
        # Max value: 2^8 * (1 + 7/8) = 480, but spec says max is 448
        # So exp_biased=15 is special: only m up to 6 (448 = 2^8 * 1.75)
        if exp_biased > 15:
            return (15 << 3) | 6  # 448
        # exp_biased == 15: check if mantissa fits
        mantissa_float = val / (2 ** exp_unbiased) - 1.0
        mantissa = int(round(mantissa_float * 8))
        if mantissa > 6:  # 448 = 1.75 * 256
            mantissa = 6
        return (exp_biased << 3) | mantissa
    
    # Normal case
    mantissa_float = val / (2 ** exp_unbiased) - 1.0
    mantissa = int(round(mantissa_float * 8))
    mantissa = max(0, min(7, mantissa))
    
    return (exp_biased << 3) | mantissa


def ue4m3_to_float(val: int) -> float:
    """Convert UE4M3 (uint8) to float."""
    if val == 0:
        return 0.0
    
    exp_biased = (val >> 3) & 0xF
    mantissa = val & 0x7
    
    if exp_biased == 0:
        # Denormalized
        return (mantissa / 8.0) * (2 ** (-6))
    else:
        # Normal
        return (1.0 + mantissa / 8.0) * (2 ** (exp_biased - 7))


def convert_scales_to_ue4m3(scales: np.ndarray) -> np.ndarray:
    """Convert FP32 scales to UE4M3 (uint8)."""
    flat = scales.flatten()
    ue4m3_vals = np.array([float_to_ue4m3(float(v)) for v in flat], dtype=np.uint8)
    return ue4m3_vals.reshape(scales.shape)


def verify_ue4m3_conversion(scales: np.ndarray, ue4m3: np.ndarray):
    """Verify UE4M3 conversion accuracy."""
    flat_s = scales.flatten()
    flat_u = ue4m3.flatten()
    
    reconstructed = np.array([ue4m3_to_float(int(v)) for v in flat_u], dtype=np.float32)
    
    # Relative error (avoid div by zero)
    mask = flat_s > 1e-10
    rel_err = np.abs(reconstructed[mask] - flat_s[mask]) / flat_s[mask]
    
    print(f"  UE4M3 conversion stats:")
    print(f"    Scale range: [{flat_s.min():.6f}, {flat_s.max():.6f}]")
    print(f"    Reconstructed range: [{reconstructed.min():.6f}, {reconstructed.max():.6f}]")
    print(f"    Max relative error: {rel_err.max():.6f}")
    print(f"    Mean relative error: {rel_err.mean():.6f}")
    
    # Check how many unique values
    unique_ue4m3 = len(np.unique(flat_u))
    print(f"    Unique UE4M3 values: {unique_ue4m3}/256")


def load_fp4_weight(path: str):
    """Load FP4 weight file (header + FP4 data + FP32 scales)."""
    with open(path, 'rb') as f:
        header = struct.unpack('5i', f.read(20))
        K, N, _, scales_h, scales_w = header
        
        # FP4 data (1 byte per element, 2 FP4 values packed)
        fp4_data = np.frombuffer(f.read(K * N), dtype=np.uint8)
        
        # FP32 scales
        num_scales = scales_h * scales_w
        scales = np.frombuffer(f.read(num_scales * 4), dtype=np.float32)
        scales = scales.reshape(scales_h, scales_w)
    
    return K, N, scales_h, scales_w, fp4_data, scales


def save_nvfp4_weight(path: str, K: int, N: int, scales_h: int, scales_w: int,
                       fp4_data: np.ndarray, ue4m3_scales: np.ndarray):
    """Save NVF4 weight file (header + FP4 data + UE4M3 scales)."""
    with open(path, 'wb') as f:
        # Header: K, N, 16 (block size), scales_h, scales_w
        header = struct.pack('5i', K, N, 16, scales_h, scales_w)
        f.write(header)
        
        # FP4 data (unchanged)
        f.write(fp4_data.tobytes())
        
        # UE4M3 scales (1 byte each)
        f.write(ue4m3_scales.astype(np.uint8).tobytes())


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input_dir> [output_dir]")
        print(f"  input_dir: directory containing .fp4 weight files")
        print(f"  output_dir: directory for .nvfp4 files (default: input_dir)")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else input_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all FP4 weight files
    fp4_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.fp4')])
    
    if not fp4_files:
        print(f"No .fp4 files found in {input_dir}")
        sys.exit(1)
    
    print(f"Converting {len(fp4_files)} FP4 weights to NVF4 format")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print()
    
    total_fp4_bytes = 0
    total_nvfp4_bytes = 0
    
    for fname in fp4_files:
        in_path = os.path.join(input_dir, fname)
        out_name = fname.replace('.fp4', '.nvfp4')
        out_path = os.path.join(output_dir, out_name)
        
        K, N, sh, sw, fp4_data, scales = load_fp4_weight(in_path)
        
        print(f"{fname}: K={K}, N={N}, scales=[{sh}x{sw}]")
        
        # Verify UE4M3 conversion
        ue4m3_scales = convert_scales_to_ue4m3(scales)
        verify_ue4m3_conversion(scales, ue4m3_scales)
        
        # Save NVF4 weight
        save_nvfp4_weight(out_path, K, N, sh, sw, fp4_data, ue4m3_scales)
        
        fp4_bytes = 20 + K * N + sh * sw * 4  # original
        nvfp4_bytes = 20 + K * N + sh * sw    # new
        total_fp4_bytes += fp4_bytes
        total_nvfp4_bytes += nvfp4_bytes
        
        ratio = fp4_bytes / nvfp4_bytes
        print(f"  Saved: {out_name} ({nvfp4_bytes / 1024:.1f} KB, {ratio:.2f}x smaller)")
        print()
    
    print(f"Summary:")
    print(f"  Total original: {total_fp4_bytes / 1024 / 1024:.2f} MB")
    print(f"  Total NVF4:     {total_nvfp4_bytes / 1024 / 1024:.2f} MB")
    print(f"  Compression:    {total_fp4_bytes / total_nvfp4_bytes:.2f}x")


if __name__ == '__main__':
    main()
