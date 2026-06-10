// better-inference: GGUF format parser
// Header-only C++ library, no dependencies beyond C++17 + cstdint.
//
// Usage:
//   GGUFReader reader("model.gguf");
//   auto tensors = reader.tensors();
//   auto meta = reader.metadata();
//   for (auto& t : tensors) load_tensor_data(reader, t);

#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>

// ── GGUF type constants ──────────────────────────────────────────────
enum GGMLType : uint32_t {
    GGML_TYPE_F32      = 0,
    GGML_TYPE_F16      = 1,
    GGML_TYPE_Q4_0     = 2,
    GGML_TYPE_Q4_1     = 3,
    GGML_TYPE_Q5_0     = 6,
    GGML_TYPE_Q5_1     = 7,
    GGML_TYPE_Q8_0     = 8,
    GGML_TYPE_Q8_1     = 9,
    GGML_TYPE_Q2_K     = 10,
    GGML_TYPE_Q3_K     = 11,
    GGML_TYPE_Q4_K     = 12,
    GGML_TYPE_Q5_K     = 13,
    GGML_TYPE_Q6_K     = 14,
    GGML_TYPE_Q8_K     = 15,
    GGML_TYPE_IQ2_XXS  = 16,
    GGML_TYPE_IQ2_XS   = 17,
    GGML_TYPE_IQ3_XXS  = 18,
    GGML_TYPE_IQ1_S    = 19,
    GGML_TYPE_IQ4_NL   = 20,
    GGML_TYPE_IQ3_S    = 21,
    GGML_TYPE_IQ2_S    = 22,
    GGML_TYPE_IQ4_XS   = 23,
    GGML_TYPE_IQ1_M    = 24,
    GGML_TYPE_Q4_K_M   = 26,  // NOT official GGML enum — our alias for Q4_K
    GGML_TYPE_COUNT
};

// Per-type element sizes (bytes per element after dequant)
static int ggml_type_size(GGMLType t) {
    switch (t) {
        case GGML_TYPE_F32: return 4;
        case GGML_TYPE_F16: return 2;
        case GGML_TYPE_Q8_0: return 1;  // 1 byte per element (after block quant)
        case GGML_TYPE_Q4_K: return 1;  // effective: 1 byte per element
        default: return 4;
    }
}

// Block size for quantized types
static int ggml_block_size(GGMLType t) {
    switch (t) {
        case GGML_TYPE_Q8_0: return 32;
        case GGML_TYPE_Q4_K: return 256;  // Q4_K: 256 elements per super-block
        default: return 1;
    }
}

// Bytes per block in file
static int ggml_block_bytes(GGMLType t) {
    switch (t) {
        case GGML_TYPE_Q8_0: return sizeof(uint16_t) + 32;  // d(half) + q[32](int8)
        case GGML_TYPE_Q4_K: return 144;  // Q4_K super-block: d(half)+dmin(half)+scales(12)+qs(128)
        default: return 4;
    }
}

// ── Metadata value ───────────────────────────────────────────────────
using GGUFValue = std::variant<
    bool, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t,
    float, double, uint64_t, int64_t,
    std::string,
    std::vector<std::string>,
    std::vector<int32_t>,
    std::vector<float>,
    std::vector<uint32_t>
>;

// ── Tensor info ──────────────────────────────────────────────────────
struct GGUFTensorInfo {
    std::string name;
    GGMLType type;
    std::vector<uint64_t> shape;  // dims: [0]=outer, [N-1]=inner (row-major storage)
    uint64_t offset;              // byte offset in file
    uint64_t file_size;           // bytes in file for this tensor

    uint64_t nelements() const {
        uint64_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
};

// ── Reader ───────────────────────────────────────────────────────────
class GGUFReader {
public:
    GGUFReader(const char* path) {
        f_ = fopen(path, "rb");
        if (!f_) { valid_ = false; return; }
        valid_ = read_header();
    }

    ~GGUFReader() { if (f_) fclose(f_); }

    bool valid() const { return valid_; }
    uint32_t version() const { return version_; }
    const std::unordered_map<std::string, GGUFValue>& metadata() const { return metadata_; }
    const std::vector<GGUFTensorInfo>& tensors() const { return tensors_; }

    // Get metadata as typed value with default
    template<typename T>
    T meta(const char* key, T default_val) const {
        auto it = metadata_.find(key);
        if (it == metadata_.end()) return default_val;
        if (auto* v = std::get_if<T>(&it->second)) return *v;
        return default_val;
    }
    // Specialization for int32 with no default
    int32_t meta_int(const char* key, int32_t def = 0) const {
        auto it = metadata_.find(key);
        if (it == metadata_.end()) return def;
        if (auto* v = std::get_if<int32_t>(&it->second)) return *v;
        if (auto* v = std::get_if<uint32_t>(&it->second)) return (int32_t)*v;
        return def;
    }

    // Read tensor data into pre-allocated buffer
    bool read_tensor_data(const GGUFTensorInfo& t, void* dst) const {
        if (!valid_ || !f_) return false;
        // Check if file position needs seeking
        if (last_seek_pos_ != t.offset) {
            if (fseek(f_, t.offset, SEEK_SET) != 0) return false;
            last_seek_pos_ = t.offset;
        }
        size_t n = fread(dst, 1, t.file_size, f_);
        last_seek_pos_ = t.offset + n;
        return n == t.file_size;
    }

    // Read entire file into memory (for mmap-free path)
    uint8_t* read_all() const {
        if (!valid_ || !f_) return nullptr;
        fseek(f_, 0, SEEK_END);
        long sz = ftell(f_);
        fseek(f_, 0, SEEK_SET);
        uint8_t* buf = (uint8_t*)malloc(sz);
        if (!buf) return nullptr;
        fread(buf, 1, sz, f_);
        return buf;
    }

private:
    FILE* f_ = nullptr;
    bool valid_ = false;
    uint32_t version_ = 0;
    std::unordered_map<std::string, GGUFValue> metadata_;
    std::vector<GGUFTensorInfo> tensors_;
    mutable uint64_t last_seek_pos_ = 0;

    template<typename T>
    T read_scalar() {
        T v;
        if (fread(&v, sizeof(T), 1, f_) != 1) return T{};
        return v;
    }

    std::string read_string() {
        uint64_t len = read_scalar<uint64_t>();
        std::string s(len, '\0');
        if (len > 0) fread(&s[0], 1, len, f_);
        return s;
    }

    bool read_header() {
        // GGUF magic: "GGUF" at offset 0
        char magic[4];
        if (fread(magic, 1, 4, f_) != 4) return false;
        if (memcmp(magic, "GGUF", 4) != 0) return false;

        version_ = read_scalar<uint32_t>();
        if (version_ < 1 || version_ > 3) return false;

        uint64_t tensor_count = read_scalar<uint64_t>();
        uint64_t metadata_kv_count = read_scalar<uint64_t>();

        // Read metadata
        for (uint64_t i = 0; i < metadata_kv_count; i++) {
            auto key = read_string();
            uint32_t val_type = read_scalar<uint32_t>();
            auto val = read_value(val_type);
            metadata_[key] = val;
        }

        // Read tensor info
        tensors_.reserve(tensor_count);
        for (uint64_t i = 0; i < tensor_count; i++) {
            GGUFTensorInfo ti;
            ti.name = read_string();
            uint32_t n_dims = read_scalar<uint32_t>();
            ti.shape.resize(n_dims);
            for (uint32_t d = 0; d < n_dims; d++) {
                ti.shape[d] = read_scalar<uint64_t>();
            }
            ti.type = (GGMLType)read_scalar<uint32_t>();
            ti.offset = read_scalar<uint64_t>();
            
            // Compute file size
            ti.file_size = compute_tensor_file_size(ti);
            tensors_.push_back(ti);
        }

        return true;
    }

    GGUFValue read_value(uint32_t type) {
        switch (type) {
            case 0: return (uint64_t)read_scalar<uint8_t>();
            case 1: return (int32_t)(int)read_scalar<int8_t>();
            case 2: return (uint32_t)read_scalar<uint16_t>();
            case 3: return (int32_t)read_scalar<int16_t>();
            case 4: return read_scalar<uint32_t>();
            case 5: return read_scalar<int32_t>();
            case 6: return read_scalar<float>();
            case 7: return (bool)read_scalar<uint8_t>();
            case 8: return read_string();
            case 9: return read_array();
            case 10: return read_scalar<uint64_t>();
            case 11: return read_scalar<int64_t>();
            case 12: return (double)read_scalar<uint16_t>() / 16384.0; // fp16 approx
            case 13: return (double)read_scalar<double>();
            default: {
                // Skip unknown type
                fseek(f_, 4, SEEK_CUR);
                return std::string("<unknown>");
            }
        }
    }

    GGUFValue read_array() {
        uint32_t arr_type = read_scalar<uint32_t>();
        uint64_t arr_len = read_scalar<uint64_t>();
        
        if (arr_type == 8) {  // string array
            std::vector<std::string> arr;
            arr.reserve(arr_len);
            for (uint64_t i = 0; i < arr_len; i++) arr.push_back(read_string());
            return arr;
        } else if (arr_type == 5) {  // int32 array
            std::vector<int32_t> arr(arr_len);
            fread(arr.data(), sizeof(int32_t), arr_len, f_);
            return arr;
        } else if (arr_type == 4) {  // uint32 array
            std::vector<uint32_t> arr(arr_len);
            fread(arr.data(), sizeof(uint32_t), arr_len, f_);
            return arr;
        } else if (arr_type == 6) {  // float32 array
            std::vector<float> arr(arr_len);
            fread(arr.data(), sizeof(float), arr_len, f_);
            return arr;
        } else if (arr_type == 1) {  // int8 array
            std::vector<int32_t> arr(arr_len);
            for (uint64_t i = 0; i < arr_len; i++) arr[i] = read_scalar<int8_t>();
            return arr;
        } else {
            // Skip unknown array type
            fseek(f_, arr_len * 4, SEEK_CUR);
            return std::vector<std::string>();
        }
    }

    static uint64_t compute_tensor_file_size(const GGUFTensorInfo& t) {
        uint64_t n_el = t.nelements();
        uint64_t n_super = (n_el + 255) / 256;
        switch (t.type) {
            case GGML_TYPE_F32: return n_el * 4;
            case GGML_TYPE_F16: return n_el * 2;
            case GGML_TYPE_Q8_0: {
                uint64_t n_blocks = (n_el + 31) / 32;
                return n_blocks * (2 + 32);  // fp16 scale + 32 int8
            }
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q4_K_M:
                return n_super * 144;  // d(2)+dmin(2)+scales(12)+qs(128)
            case GGML_TYPE_Q2_K:
                return n_super * 84;   // block_q2_K: d(2)+dmin(2)+scales(16)+qs(64)
            case GGML_TYPE_Q3_K:
                return n_super * 110;  // block_q3_K: d(2)+hmask(32)+qs(64)+scales(12)
            case GGML_TYPE_Q5_K:
                return n_super * 80;   // block_q5_K: d(2)+dmin(2)+scales(12)+qh(32)+qs(32)
            case GGML_TYPE_Q6_K:
                return n_super * 210;  // block_q6_K: d(2)+ql(128)+qh(64)+scales(16)
            default: return n_el * 4;  // fallback
        }
    }
};

// ── Dequant helpers ──────────────────────────────────────────────
static void dequant_f16(const uint8_t* src, float* dst) {
    // IEEE 754-2008 binary16 → float32
    uint16_t raw; memcpy(&raw, src, 2);
    uint32_t sign = (uint32_t)((raw >> 15) & 1) << 31;
    uint32_t exp = (raw >> 10) & 0x1f;
    uint32_t mant = raw & 0x3ff;
    if (exp == 0) {
        // Subnormal/zero: (mant / 2^10) * 2^(-14)
        uint32_t m = mant;
        if (m == 0) { *(uint32_t*)dst = sign; return; }
        while ((m & 0x400) == 0) { m <<= 1; exp--; }
        exp++;
        m &= 0x3ff;
        exp = exp + 127 - 15;
        *(uint32_t*)dst = sign | (exp << 23) | (m << 13);
    } else if (exp == 31) {
        // Inf/NaN
        *(uint32_t*)dst = sign | 0x7f800000 | (mant << 13);
    } else {
        // Normal: (1 + mant/2^10) * 2^(exp-15)
        *(uint32_t*)dst = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
}

// Dequant one Q8_0 block (32 elements) into dst (32 floats)
static void dequant_q8_0_block(const uint8_t* src, float* dst) {
    float d;
    dequant_f16(src, &d);
    const int8_t* q = (const int8_t*)(src + 2);
    for (int i = 0; i < 32; i++) dst[i] = q[i] * d;
}

// Dequant entire Q8_0 tensor: src (raw file data) → dst (n_el floats)
static void dequant_q8_0(const uint8_t* src, float* dst, uint64_t n_el) {
    uint64_t n_blocks = (n_el + 31) / 32;
    for (uint64_t b = 0; b < n_blocks; b++)
        dequant_q8_0_block(src + b * 34, dst + b * 32);
}

// ── Q4_K (k-quant) dequant ───────────────────────────────────────
// Q4_K super-block: 256 elements, 144 bytes total
// Struct mirrors block_q4_K from llama.cpp ggml-quants.h
//
// Source: llama.cpp ggml-quants.c `get_scale_min_k4` + `dequantize_row_q4_K`
//
// Super-block layout:
//   d:     fp16 super-block scale
//   dmin:  fp16 super-block minimum (stored as negative in ggml: dmin = -|actual_dmin|)
//   scales[12]: 8 pairs of (6-bit scale, 6-bit min), cleverly packed
//   qs[128]: 256 x 4-bit nibbles
//
// 8 sub-blocks of 32 elements each.
// For j=0..7, get scale & min from scales[12]:
//   if j < 4:  sc = scales[j] & 63,          min = scales[j+4] & 63
//   if j >= 4: sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4),
//               min = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
//
// Dequant formula (from ggml): y = d * sc * q - dmin * min * (15-q)
// Where q is nibble [0..15], sc/min are 6-bit values [0..63]
// ggml-computed: dl = d * sc, ml = dmin * min, y = dl * q - ml * (15-q)
// = (dl + ml) * q - ml * 15

struct block_q4_K {
    uint16_t d;       // super-block scale (fp16)
    uint16_t dmin;    // super-block min (fp16, negative for ggml optimization)
    uint8_t  scales[12]; // 8 x (6-bit scale + 6-bit min) packed
    uint8_t  qs[128]; // 256 x 4-bit nibbles
};

// Dequantize one Q4_K super-block (256 elements) -> 256 floats
// Matches llama.cpp dequantize_row_q4_K (scalar path, no SIMD)
static void dequant_q4_K_block(const uint8_t* src, float* dst) {
    const block_q4_K* x = (const block_q4_K*)src;
    float d, dmin;
    dequant_f16((const uint8_t*)&x->d, &d);
    dequant_f16((const uint8_t*)&x->dmin, &dmin);

    for (int j = 0; j < 8; j++) {
        uint8_t sc, mn;
        // get_scale_min_k4 from ggml-quants.c
        if (j < 4) {
            sc = x->scales[j] & 63;
            mn = x->scales[j + 4] & 63;
        } else {
            sc = (x->scales[j + 4] & 0xF) | ((x->scales[j - 4] >> 6) << 4);
            mn = (x->scales[j + 4] >> 4)   | ((x->scales[j] >> 6) << 4);
        }

        float dl = d * (float)sc;
        float ml = dmin * (float)mn;

        // Sub-block j: 32 nibbles packed in qs[16*j .. 16*j+15]
        // y = dl * q - ml
        // where q is 4-bit nibble [0..15], sc and mn are 6-bit [0..63]
        // dmin is typically negative, making -ml = |dmin|*mn the minimum offset
        for (int i = 0; i < 16; i++) {
            uint8_t byte = x->qs[j * 16 + i];
            uint8_t n0 = byte & 0x0F;
            uint8_t n1 = byte >> 4;

            dst[j * 32 + i * 2 + 0] = dl * (float)n0 - ml;
            dst[j * 32 + i * 2 + 1] = dl * (float)n1 - ml;
        }
    }
}

// Dequant entire Q4_K tensor: src (raw file data) -> dst (n_el floats)
static void dequant_q4_K(const uint8_t* src, float* dst, uint64_t n_el) {
    uint64_t n_super = (n_el + 255) / 256;
    for (uint64_t b = 0; b < n_super; b++)
        dequant_q4_K_block(src + b * 144, dst + b * 256);
}

// ── Q6_K dequant (used for lm_head in Q4_K_M GGUF files) ─────────
// Q6_K: 16 sub-blocks of 16 elements. Each has an int8 scale.
// ql[128] has lower 4 bits, qh[64] has upper 2 bits = 6 bits total
// Formula: y = d * sc * (q - 32)
struct block_q6_K {
    uint8_t ql[128];     // lower 4 bits of 6-bit quant
    uint8_t qh[64];      // upper 2 bits of 6-bit quant
    int8_t  scales[16];  // per-sub-block scale (16 x 16 = 256)
    uint16_t d;           // super-block scale (fp16)
};

static void dequant_q6_K_block(const uint8_t* src, float* dst) {
    const block_q6_K* x = (const block_q6_K*)src;
    float d;
    dequant_f16((const uint8_t*)&x->d, &d);

    // Q6_K: per llama.cpp convention, scales are int8_t (not divided by anything)
    // Dequant: val = d * scale * (q - 32)
    // where q is the 6-bit quant [0..63], scale is int8
    for (int j = 0; j < 16; j++) {
        float fsc = d * (float)x->scales[j];
        for (int i = 0; i < 16; i++) {
            int idx = j * 16 + i;
            // ql[idx/2] has 2 nibbles: low nibble for even idx, high for odd
            uint8_t ql_byte = x->ql[idx / 2];
            int nib_low = (idx % 2 == 0) ? (ql_byte & 0x0F) : (ql_byte >> 4);
            // qh[idx/4] has 4 x 2-bit fields
            uint8_t qh_byte = x->qh[idx / 4];
            int qh_shift = (idx % 4) * 2;
            int nib_high = (qh_byte >> qh_shift) & 0x03;
            int q = nib_low | (nib_high << 4);   // 6-bit value [0..63]
            int8_t q_centered = (int8_t)(q - 32); // centered at 0
            dst[idx] = fsc * (float)q_centered;
        }
    }
}

static void dequant_q6_K(const uint8_t* src, float* dst, uint64_t n_el) {
    uint64_t n_super = (n_el + 255) / 256;
    for (uint64_t b = 0; b < n_super; b++)
        dequant_q6_K_block(src + b * sizeof(block_q6_K), dst + b * 256);
}

// ── INT4 block-16 requantizer ────────────────────────────────────
// Requantize FP32 tensor to INT4 symmetric block-16
// Returns packed uint8 data and float scales
struct Int4Weight {
    std::vector<uint8_t> packed;  // [N * K/2]
    std::vector<float> scales;    // [N * K/16]
    int K, N;
};

static Int4Weight requant_int4(const float* data, int N, int K) {
    constexpr int INT4_BLOCK = 16;
    assert(K % INT4_BLOCK == 0);
    int num_blks = K / INT4_BLOCK;
    Int4Weight w;
    w.K = K; w.N = N;
    w.packed.resize((size_t)N * K / 2);
    w.scales.resize((size_t)N * num_blks);

    for (int n = 0; n < N; n++) {
        const float* row = data + (size_t)n * K;
        for (int kb = 0; kb < num_blks; kb++) {
            // Find absmax in block
            float absmax = 0.0f;
            for (int i = 0; i < INT4_BLOCK; i++)
                absmax = fmaxf(absmax, fabsf(row[kb * INT4_BLOCK + i]));
            float scale = fmaxf(absmax, 1e-10f) / 7.0f;
            // Guard against inf/nan from corrupt input
            if (!std::isfinite(scale)) scale = 1.0f;
            w.scales[(size_t)n * num_blks + kb] = scale;

            // Quantize and pack
            for (int i = 0; i < INT4_BLOCK; i += 2) {
                int v0 = (int)roundf(row[kb * INT4_BLOCK + i] / scale);
                int v1 = (int)roundf(row[kb * INT4_BLOCK + i + 1] / scale);
                v0 = v0 < -7 ? -7 : (v0 > 7 ? 7 : v0);
                v1 = v1 < -7 ? -7 : (v1 > 7 ? 7 : v1);
                uint8_t packed_byte = ((uint8_t)(v0 + 8) & 0x0F) | (((uint8_t)(v1 + 8) & 0x0F) << 4);
                w.packed[(size_t)n * K / 2 + (size_t)kb * INT4_BLOCK / 2 + i / 2] = packed_byte;
            }
        }
    }
    return w;
}

// ── Tensor name mapper (Qwen3 + Llama 3.1) ───────────────────────
// GGUF name → blackwell weight file name.
// Both Qwen3 and Llama 3.1 use the same tensor naming convention:
//   blk.{l}.attn_q.weight  → layer l self_attn.q_proj
//   blk.{l}.ffn_gate.weight → layer l mlp.gate_proj
//   token_embd.weight → embed_tokens
//
// Key differences:
//   - Llama 3: rope_freqs (float array) vs Qwen3: rope_theta (scalar)
//   - Llama 3: tiktoken vocab; Qwen3: BPE vocab
//   - Llama 2: no attn_q_norm / attn_k_norm (optional; skipped if absent)
//   - Qwen3: has attn_q_norm / attn_k_norm (required)

// Non-layer tensor mapping (shared Qwen3 + Llama)
static const char* map_tensor_name_common(const char* gguf_name) {
    if (strcmp(gguf_name, "token_embd.weight") == 0) return "embed_tokens";
    if (strcmp(gguf_name, "output_norm.weight") == 0) return "final_norm";
    if (strcmp(gguf_name, "output.weight") == 0) return "lm_head";
    return nullptr;
}

// Extract layer number from "blk.{N}.xxx"
static int extract_blk_layer(const char* name) {
    if (strncmp(name, "blk.", 4) != 0) return -1;
    const char* p = name + 4;
    int layer = 0;
    while (*p >= '0' && *p <= '9') { layer = layer * 10 + (*p - '0'); p++; }
    return (*p == '.') ? layer : -1;
}

// Internal layer tensor mapper.
// suf: pointer to tensor suffix after "blk.{l}."
// Returns true if matched, writes "{l}_{bw_name}" to buf.
static bool map_layer_tensor(const char* suf, int layer, char* buf, int buf_size) {
    static const struct NameMap { const char* gguf; const char* bw; } maps[] = {
        {"attn_norm.weight",       "input_layernorm"},
        {"attn_q.weight",          "self_attn.q_proj"},
        {"attn_k.weight",          "self_attn.k_proj"},
        {"attn_v.weight",          "self_attn.v_proj"},
        {"attn_output.weight",      "self_attn.o_proj"},
        {"attn_q_norm.weight",      "q_norm"},   // Llama 3 + Qwen3
        {"attn_k_norm.weight",      "k_norm"},   // Llama 3 + Qwen3
        {"ffn_norm.weight",         "post_attention_layernorm"},
        {"ffn_gate.weight",         "mlp.gate_proj"},
        {"ffn_up.weight",            "mlp.up_proj"},
        {"ffn_down.weight",         "mlp.down_proj"},
    };
    for (auto& m : maps) {
        if (strcmp(suf, m.gguf) == 0) {
            snprintf(buf, buf_size, "%d_%s", layer, m.bw);
            return true;
        }
    }
    return false;
}

// Map GGUF tensor name → blackwell weight file name.
// Works for both Qwen3 and Llama 3.1 (same naming convention).
// Returns true if mapped, writes to buf (128 bytes).
static bool map_tensor_name(const char* gguf_name, char* buf, int buf_size) {
    // Non-layer tensors
    auto* common = map_tensor_name_common(gguf_name);
    if (common) { snprintf(buf, buf_size, "%s", common); return true; }

    // Layer tensor
    int l = extract_blk_layer(gguf_name);
    if (l < 0) return false;

    const char* suf = gguf_name + 4;
    while (*suf >= '0' && *suf <= '9') suf++;
    if (*suf == '.') suf++; else return false;

    return map_layer_tensor(suf, l, buf, buf_size);
}
