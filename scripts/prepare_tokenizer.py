#!/usr/bin/env python3
"""Prepare binary tokenizer data from tokenizer.json for C++ BPE tokenizer.

Usage:
    python3 scripts/prepare_tokenizer.py [tokenizer.json_path] [output_dir]

Default: reads /mnt/data/ai/hf/qwen3-1.7b-base/tokenizer.json
         writes tokenizer_data.bin in current directory
"""
import json, struct, sys, os

def gpt2_byte_encoder():
    """Build GPT-2 byte-level encoder: byte → unicode ordinal."""
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

def gpt2_byte_decoder():
    """Inverse: unicode char → byte value."""
    enc = gpt2_byte_encoder()
    return {v: k for k, v in enc.items()}

def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/data/ai/hf/qwen3-1.7b-base/tokenizer.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "tokenizer_data.bin"

    print(f"Reading {json_path}...")
    with open(json_path) as f:
        data = json.load(f)

    vocab = data["model"]["vocab"]          # str → id
    merges = data["model"]["merges"]        # list of "str1 str2"
    added_tokens = data.get("added_tokens", [])

    num_vocab = len(vocab)
    num_merges = len(merges)
    num_added = len(added_tokens)

    byte_enc = gpt2_byte_encoder()          # byte_value → unicode_char

    print(f"Vocab: {num_vocab}, Merges: {num_merges}, Added: {num_added}")

    with open(out_path, "wb") as f:
        # Header
        f.write(struct.pack("<III", num_vocab, num_merges, num_added))

        # Byte encoder: 256 uint32 values (byte → unicode codepoint)
        for b in range(256):
            ch = byte_enc[b]
            f.write(struct.pack("<I", ord(ch)))

        # Vocab entries: id, len, bytes
        for token_str, token_id in vocab.items():
            tb = token_str.encode("utf-8")
            f.write(struct.pack("<IH", token_id, len(tb)))
            f.write(tb)

        # Added tokens: id, len, bytes, is_special
        for at in added_tokens:
            cb = at["content"].encode("utf-8")
            f.write(struct.pack("<IHB", at["id"], len(cb), 1 if at.get("special") else 0))
            f.write(cb)

        # Merges: left_len, left_bytes, right_len, right_bytes
        for merge_str in merges:
            parts = merge_str.split(" ", 1)
            if len(parts) != 2:
                print(f"WARNING: bad merge: {merge_str!r}")
                continue
            left, right = parts
            lb = left.encode("utf-8")
            rb = right.encode("utf-8")
            f.write(struct.pack("<H", len(lb)))
            f.write(lb)
            f.write(struct.pack("<H", len(rb)))
            f.write(rb)

    sz = os.path.getsize(out_path)
    print(f"Wrote {out_path}: {sz} bytes ({sz/1024/1024:.1f} MB)")

if __name__ == "__main__":
    main()
