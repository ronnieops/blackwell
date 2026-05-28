#pragma once
// bpe_tokenizer.h — Minimal BPE tokenizer for Qwen (tiktoken-compatible).
// Loads from binary data prepared by scripts/prepare_tokenizer.py.
//
// Supports: encode(text) → token IDs, decode(token_id) → UTF-8 string.
// Pre-tokenizer: simplified ASCII regex (handles English text).
// Byte-level encoding: GPT-2 style.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstdint>

namespace blackwell {

struct BpeTokenizer {
    // Byte-level encoder: raw byte → unicode char
    char byte_enc[256];
    // Byte-level decoder: unicode char → raw byte
    std::unordered_map<char32_t, uint8_t> byte_dec;

    // Token string → token ID
    std::unordered_map<std::string, uint32_t> vocab;
    // Token ID → token string (for decode)
    std::unordered_map<uint32_t, std::string> id_to_str;

    // Merge rank: "left right" → rank (0 = highest priority)
    std::unordered_map<std::string, int> merge_rank;

    // Special tokens
    std::unordered_map<std::string, uint32_t> special_tokens;
    std::unordered_map<uint32_t, std::string> special_id_to_str;

    int load(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "FAIL open %s\n", path); return -1; }

        uint32_t num_vocab, num_merges, num_added;
        fread(&num_vocab, 4, 1, f);
        fread(&num_merges, 4, 1, f);
        fread(&num_added, 4, 1, f);

        // Byte encoder
        for (int i = 0; i < 256; i++) {
            uint32_t cp;
            fread(&cp, 4, 1, f);
            // Convert unicode codepoint to UTF-8 char
            if (cp < 0x80) {
                byte_enc[i] = (char)cp;
            } else if (cp < 0x800) {
                // 2-byte UTF-8 — we only store the multi-byte string
                // For simplicity, store as a 2-char string in a separate map
                // Actually, let's handle this properly
            }
            // For the byte-level encoder, we need byte→string mapping
            // Let's build it differently
        }

        // Actually, let's rebuild the byte encoder properly.
        // The GPT-2 byte encoder maps each byte 0-255 to a unique unicode char.
        // Most are single-byte ASCII, but bytes 0-32, 127-160, 173 map to 2-byte UTF-8.
        // We need byte→string (potentially multi-byte UTF-8).
        rewind(f);
        // Re-read header
        fread(&num_vocab, 4, 1, f);
        fread(&num_merges, 4, 1, f);
        fread(&num_added, 4, 1, f);

        // Build byte encoder: byte → UTF-8 string
        std::string byte_to_str[256];
        for (int i = 0; i < 256; i++) {
            uint32_t cp;
            fread(&cp, 4, 1, f);
            char buf[4];
            int len = 0;
            if (cp < 0x80) {
                buf[0] = (char)cp; len = 1;
            } else if (cp < 0x800) {
                buf[0] = (char)(0xC0 | (cp >> 6));
                buf[1] = (char)(0x80 | (cp & 0x3F));
                len = 2;
            } else if (cp < 0x10000) {
                buf[0] = (char)(0xE0 | (cp >> 12));
                buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
                buf[2] = (char)(0x80 | (cp & 0x3F));
                len = 3;
            }
            byte_to_str[i] = std::string(buf, len);
        }

        // Build byte decoder: UTF-8 string → byte value
        for (int i = 0; i < 256; i++) {
            // For decode, we need to map unicode chars back to bytes
            // Store as char32_t → byte
            uint32_t cp;
            // Re-derive: already have byte_to_str, need reverse
            // Use the codepoint directly
        }
        // Actually for decode we need string→byte. Let's use the UTF-8 strings.
        std::unordered_map<std::string, uint8_t> str_to_byte;
        for (int i = 0; i < 256; i++) {
            str_to_byte[byte_to_str[i]] = (uint8_t)i;
        }

        // Vocab entries
        vocab.reserve(num_vocab);
        id_to_str.reserve(num_vocab);
        for (uint32_t i = 0; i < num_vocab; i++) {
            uint32_t id;
            uint16_t len;
            fread(&id, 4, 1, f);
            fread(&len, 2, 1, f);
            std::string s(len, '\0');
            fread(&s[0], 1, len, f);
            vocab[s] = id;
            id_to_str[id] = s;
        }

        // Added tokens
        for (uint32_t i = 0; i < num_added; i++) {
            uint32_t id;
            uint16_t len;
            uint8_t is_special;
            fread(&id, 4, 1, f);
            fread(&len, 2, 1, f);
            fread(&is_special, 1, 1, f);
            std::string s(len, '\0');
            fread(&s[0], 1, len, f);
            if (is_special) {
                special_tokens[s] = id;
                special_id_to_str[id] = s;
            }
            // Also add to main vocab/id maps
            vocab[s] = id;
            id_to_str[id] = s;
        }

        // Merges
        for (uint32_t i = 0; i < num_merges; i++) {
            uint16_t ll, rl;
            fread(&ll, 2, 1, f);
            std::string left(ll, '\0');
            fread(&left[0], 1, ll, f);
            fread(&rl, 2, 1, f);
            std::string right(rl, '\0');
            fread(&right[0], 1, rl, f);
            std::string key = left + " " + right;
            merge_rank[key] = (int)i;
        }

        fclose(f);

        // Store byte mappings for encoding/decoding
        // We need byte_to_str and str_to_byte accessible in encode/decode
        // Store as member vectors
        for (int i = 0; i < 256; i++) byte_enc_str_[i] = byte_to_str[i];
        byte_dec_map_ = str_to_byte;

        fprintf(stderr, "[tokenizer] Loaded: %u vocab, %u merges, %u special tokens\n",
                num_vocab, num_merges, (unsigned)special_tokens.size());
        return 0;
    }

    // Encode a text string to token IDs.
    std::vector<uint32_t> encode(const std::string& text) const {
        std::vector<uint32_t> ids;

        // 1. Pre-tokenize: split into chunks using simplified GPT-4 regex
        // For ASCII: match words, numbers, punctuation, whitespace
        std::vector<std::string> chunks = pretokenize(text);

        // 2. For each chunk: byte-level encode → BPE
        for (const auto& chunk : chunks) {
            // Byte-level encode: each byte → its unicode string
            std::string byte_encoded;
            for (unsigned char c : chunk) {
                byte_encoded += byte_enc_str_[c];
            }

            // Check if the whole thing is a known token (fast path)
            auto it = vocab.find(byte_encoded);
            if (it != vocab.end()) {
                ids.push_back(it->second);
                continue;
            }

            // Split into individual characters (as strings)
            std::vector<std::string> symbols;
            // UTF-8 aware split: each codepoint as a string
            for (size_t i = 0; i < byte_encoded.size(); ) {
                unsigned char c = byte_encoded[i];
                int clen = 1;
                if (c >= 0xC0 && c < 0xE0) clen = 2;
                else if (c >= 0xE0 && c < 0xF0) clen = 3;
                else if (c >= 0xF0) clen = 4;
                symbols.emplace_back(byte_encoded.substr(i, clen));
                i += clen;
            }

            // BPE merge loop
            while (symbols.size() > 1) {
                // Find the pair with lowest merge rank
                int best_rank = INT32_MAX;
                int best_idx = -1;
                for (size_t j = 0; j + 1 < symbols.size(); j++) {
                    std::string key = symbols[j] + " " + symbols[j + 1];
                    auto it = merge_rank.find(key);
                    if (it != merge_rank.end() && it->second < best_rank) {
                        best_rank = it->second;
                        best_idx = (int)j;
                    }
                }
                if (best_idx < 0) break;  // no more merges

                // Merge the pair
                symbols[best_idx] = symbols[best_idx] + symbols[best_idx + 1];
                symbols.erase(symbols.begin() + best_idx + 1);
            }

            // Look up each symbol in vocab
            for (const auto& sym : symbols) {
                auto it = vocab.find(sym);
                if (it != vocab.end()) {
                    ids.push_back(it->second);
                } else {
                    // Unknown symbol — encode byte by byte
                    for (unsigned char c : sym) {
                        auto sit = vocab.find(byte_enc_str_[c]);
                        if (sit != vocab.end()) {
                            ids.push_back(sit->second);
                        }
                    }
                }
            }
        }

        return ids;
    }

    // Decode a token ID to a UTF-8 string.
    std::string decode(uint32_t token_id) const {
        auto it = id_to_str.find(token_id);
        if (it == id_to_str.end()) return "";
        const std::string& token_str = it->second;

        // Byte-level decode: map unicode chars back to bytes
        std::string result;
        for (size_t i = 0; i < token_str.size(); ) {
            // Try matching longest byte-decoded string
            bool found = false;
            for (int len = 3; len >= 1; len--) {
                if (i + len <= token_str.size()) {
                    auto dit = byte_dec_map_.find(token_str.substr(i, len));
                    if (dit != byte_dec_map_.end()) {
                        result += (char)dit->second;
                        i += len;
                        found = true;
                        break;
                    }
                }
            }
            if (!found) {
                // Pass through as-is (shouldn't happen with valid tokens)
                result += token_str[i];
                i++;
            }
        }
        return result;
    }

    // Decode multiple token IDs to a single UTF-8 string.
    std::string decode(const std::vector<uint32_t>& ids) const {
        std::string result;
        for (uint32_t id : ids) {
            result += decode(id);
        }
        return result;
    }

private:
    std::string byte_enc_str_[256];
    std::unordered_map<std::string, uint8_t> byte_dec_map_;

    // Simplified GPT-4 pre-tokenizer for ASCII text.
    // Matches: contractions, optional-space+word, digits, punctuation, newlines+spaces, trailing spaces.
    // Key: leading space groups with following word ("Hello world" → ["Hello", " world"])
    std::vector<std::string> pretokenize(const std::string& text) const {
        std::vector<std::string> chunks;
        size_t i = 0, n = text.size();

        while (i < n) {
            unsigned char c = text[i];

            // 1. Contractions: 's 't 're 've 'm 'll 'd (case insensitive)
            if (c == '\'' && i + 1 < n) {
                char nx = text[i+1];
                char lx = nx | 0x20;  // lowercase
                if (lx == 's' || lx == 't' || lx == 'm' || lx == 'd') {
                    chunks.push_back(text.substr(i, 2));
                    i += 2; continue;
                }
                if (lx == 'r' && i+2 < n && (text[i+2]|0x20) == 'e') {
                    chunks.push_back(text.substr(i, 3));
                    i += 3; continue;
                }
                if (lx == 'v' && i+2 < n && (text[i+2]|0x20) == 'e') {
                    chunks.push_back(text.substr(i, 3));
                    i += 3; continue;
                }
                if (lx == 'l' && i+2 < n && (text[i+2]|0x20) == 'l') {
                    chunks.push_back(text.substr(i, 3));
                    i += 3; continue;
                }
            }

            // 2. Optional non-alpha-non-digit + letters: [ ^\r\n\p{L}\p{N}]?\p{L}+
            //    This handles " word", "word", "!word", etc.
            //    For ASCII: space + letters, or just letters
            {
                size_t start = i;
                bool has_space = false;
                if (c == ' ' && i+1 < n && isLetter(text[i+1])) {
                    has_space = true;
                    i++;
                }
                if (isLetter(text[i])) {
                    while (i < n && isLetter(text[i])) i++;
                    chunks.push_back(text.substr(start, i - start));
                    continue;
                }
                i = start; // reset
            }

            // 3. Digits: \p{N}+
            if (c >= '0' && c <= '9') {
                size_t start = i;
                while (i < n && text[i] >= '0' && text[i] <= '9') i++;
                chunks.push_back(text.substr(start, i - start));
                continue;
            }

            // 4. Punctuation with optional leading space: " ?[^\s\p{L}\p{N}]+[\r\n]*"
            {
                size_t start = i;
                if (c == ' ' && i+1 < n && isPunct(text[i+1])) {
                    i++;
                }
                if (i < n && isPunct(text[i])) {
                    while (i < n && isPunct(text[i])) i++;
                    while (i < n && (text[i] == '\r' || text[i] == '\n')) i++;
                    chunks.push_back(text.substr(start, i - start));
                    continue;
                }
                i = start;
            }

            // 5. Newlines with trailing whitespace: "\s*[\r\n]+"
            if (c == '\n' || c == '\r') {
                size_t start = i;
                while (i < n && (text[i] == '\n' || text[i] == '\r')) i++;
                // Include trailing whitespace
                while (i < n && (text[i] == ' ' || text[i] == '\t')) i++;
                chunks.push_back(text.substr(start, i - start));
                continue;
            }

            // 6. Spaces (not followed by word/digit/punct — those are handled above)
            //    "\s+(?!\S)" | "\s+" — trailing spaces or spaces before non-space
            if (c == ' ' || c == '\t') {
                size_t start = i;
                while (i < n && (text[i] == ' ' || text[i] == '\t')) i++;
                chunks.push_back(text.substr(start, i - start));
                continue;
            }

            // 7. Fallback: single char
            chunks.push_back(text.substr(i, 1));
            i++;
        }
        return chunks;
    }

    static bool isLetter(unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    }
    static bool isPunct(unsigned char c) {
        // Not letter, not digit, not whitespace
        return c > 0 && c != ' ' && c != '\t' && c != '\n' && c != '\r' &&
               !isLetter(c) && !(c >= '0' && c <= '9');
    }
};

} // namespace blackwell
