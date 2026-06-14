#!/usr/bin/env python3
"""Convert INT4 weight *scales* from FP32 to FP16.

Reads  weights_int4_qwen3_8b/*.scale_t   (FP32, 20-byte header + N*num_K_blks float32)
Writes weights_int4_qwen3_8b_fp16sc/*.scale_t (FP16, same header + N*num_K_blks float16)

Packed .int4_t and norm .f32 files are hardlinked (unchanged, zero disk cost).
The FP16 dir is ~0.94 GB smaller (scales 1.89GB -> 0.94GB).

Block-16 absmax scales have typical magnitude 0.01-0.5; FP16 (11-bit mantissa)
introduces ~0.05% relative error per scale. Re-verify PPL after conversion
(bench/bench_ppl_int4_8b). Expected PPL drift +0.1-0.5 from 23.52 baseline.
"""
import os, sys, struct, shutil
import numpy as np

SRC = "weights_int4_qwen3_8b"
DST = "weights_int4_qwen3_8b_fp16sc"

def main():
    if not os.path.isdir(SRC):
        sys.exit(f"source dir not found: {SRC}")
    os.makedirs(DST, exist_ok=True)

    files = sorted(os.listdir(SRC))
    n_scale = n_link = 0
    for fn in files:
        sp = os.path.join(SRC, fn)
        dp = os.path.join(DST, fn)
        if fn.endswith(".scale_t"):
            with open(sp, "rb") as f:
                header = f.read(20)
                data = np.frombuffer(f.read(), dtype=np.float32)
            # header h[3]*h[4] = scale count; sanity check
            h = struct.unpack("<5I", header)
            count = h[3] * h[4]
            assert data.size == count, f"{fn}: {data.size} != h3*h4 {count}"
            data16 = data.astype(np.float16)
            with open(dp, "wb") as f:
                f.write(header)          # header unchanged (count semantics identical)
                f.write(data16.tobytes())
            n_scale += 1
        else:
            # .int4_t (packed weights) and .f32 (layernorms/final_norm): byte-identical
            if os.path.lexists(dp):
                os.remove(dp)
            os.link(sp, dp)              # hardlink: zero disk cost
            n_link += 1
    sz_src = sum(os.path.getsize(os.path.join(SRC,f)) for f in os.listdir(SRC))
    sz_dst = sum(os.path.getsize(os.path.join(DST,f)) for f in os.listdir(DST))
    print(f"Converted {n_scale} scale_t files (FP32->FP16), hardlinked {n_link} files.")
    print(f"Size: {sz_src/1e9:.2f} GB -> {sz_dst/1e9:.2f} GB (saved {(sz_src-sz_dst)/1e9:.2f} GB)")

if __name__ == "__main__":
    main()
