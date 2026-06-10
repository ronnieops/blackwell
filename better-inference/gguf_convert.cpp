// better-inference: GGUF → Blackwell weight format converter
// Reads GGUF, dequantizes, requantizes to INT4 block-16, writes files.
//
// Usage: ./gguf_convert model.gguf output_dir/
//
// Output format matches blackwell's existing weight format:
//   {layer}_{name}.int4_t + .scale_t
//   {layer}_input_layernorm.f32
//   {layer}_post_attention_layernorm.f32
//   final_norm.f32
//   embed_tokens.int4_t + .scale_t
//   lm_head.int4_t + .scale_t
//   qk_norms.f32  (combined Q/K head norms)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include <string>
#include <filesystem>
#include "gguf.h"

struct GGUFFile {
    uint8_t* data;
    size_t size;
};

static GGUFFile load_whole_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return {nullptr, 0};
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* buf = (uint8_t*)malloc(sz);
    if (!buf) { fclose(f); return {nullptr, 0}; }
    fread(buf, 1, sz, f);
    fclose(f);
    return {buf, (size_t)sz};
}

static void write_file(const char* path, const void* data, size_t size) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "FAIL: can't write %s\n", path); return; }
    fwrite(data, 1, size, f);
    fclose(f);
}

// Write INT4 sym weight in blackwell format
static void write_int4_weight(const char* out_dir, const char* name,
                               const uint8_t* packed, const float* scales,
                               int K, int N) {
    char path[256];
    int num_kb = K / 16;

    // .int4_t: header [K, N, 16, num_kb, 1] + packed data
    snprintf(path, 256, "%s/%s.int4_t", out_dir, name);
    FILE* f = fopen(path, "wb");
    if (!f) return;
    int hdr[5] = {K, N, 16, num_kb, 1};
    fwrite(hdr, 4, 5, f);
    fwrite(packed, 1, (size_t)N * K / 2, f);
    fclose(f);

    // .scale_t: header [0, 0, 0, num_kb, N] + scales
    snprintf(path, 256, "%s/%s.scale_t", out_dir, name);
    f = fopen(path, "wb");
    if (!f) return;
    int hdr_sc[5] = {0, 0, 0, num_kb, N};
    fwrite(hdr_sc, 4, 5, f);
    fwrite(scales, 4, (size_t)N * num_kb, f);
    fclose(f);

    double mb = ((double)N * K / 2 + (double)N * num_kb * 4) / (1024 * 1024);
    printf("  %s: %dx%d INT4 %.1fMB\n", name, N, K, mb);
}

// Write F32 norm weight
static void write_f32(const char* out_dir, const char* name, const float* data, int n) {
    char path[256];
    snprintf(path, 256, "%s/%s.f32", out_dir, name);
    write_file(path, data, n * 4);
    printf("  %s: %d F32\n", name, n);
}

// Write combined Q/K head norms
static void write_qk_norms(const char* out_dir, int NL, int hd,
                           const float* qnorms, const float* knorms) {
    // Layout: [l][2][hd] — q_norm then k_norm per layer
    std::vector<float> buf((size_t)NL * 2 * hd);
    for (int l = 0; l < NL; l++) {
        memcpy(&buf[(size_t)l * 2 * hd], &qnorms[(size_t)l * hd], hd * 4);
        memcpy(&buf[(size_t)l * 2 * hd + hd], &knorms[(size_t)l * hd], hd * 4);
    }
    char path[256];
    snprintf(path, 256, "%s/qk_norms.f32", out_dir);
    write_file(path, buf.data(), buf.size() * 4);
    printf("  qk_norms: %dx2x%d F32\n", NL, hd);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s model.gguf output_dir/\n", argv[0]);
        return 1;
    }

    const char* gguf_path = argv[1];
    const char* out_dir = argv[2];

    // Create output directory
    std::filesystem::create_directories(out_dir);

    // Load GGUF
    GGUFFile gf = load_whole_file(gguf_path);
    if (!gf.data) { fprintf(stderr, "FAIL: can't read %s\n", gguf_path); return 1; }

    // Parse header manually (since GGUFReader uses FILE*, not memory)
    // We'll use the reader approach — read from memory
    // For now, use GGUFReader on the file
    GGUFReader reader(gguf_path);
    if (!reader.valid()) { fprintf(stderr, "FAIL: can't parse GGUF\n"); return 1; }

    // Extract config
    auto arch = reader.meta<std::string>("general.architecture", "qwen3");
    printf("Architecture: %s\n", arch.c_str());

    // Handle different arch prefixes for metadata keys
    const char* prefix = "llama";
    if (arch == "qwen3" || arch == "qwen2") prefix = "qwen3";
    else if (arch == "llama") prefix = "llama";
    bool is_llama = (arch == "llama");

    auto get_meta = [&](const char* key_suffix, int default_val) -> int {
        char full_key[128];
        snprintf(full_key, 128, "%s.%s", prefix, key_suffix);
        return reader.meta_int(full_key, default_val);
    };

    int NL = get_meta("block_count", 0);
    int H = get_meta("embedding_length", 0);
    int I = get_meta("feed_forward_length", 0);
    int nqh = get_meta("attention.head_count", 0);
    int nkv = get_meta("attention.head_count_kv", 0);
    int hd = (nqh > 0) ? H / nqh : 0;
    // Get vocab size from tokens array count in metadata
    auto it = reader.metadata().find("tokenizer.ggml.tokens");
    int V = 151936;
    if (it != reader.metadata().end()) {
        if (auto* sa = std::get_if<std::vector<std::string>>(&it->second))
            V = (int)sa->size();
    }
    printf("V = %d\n", V);

    // Get RoPE config
    // Llama 3: rope_freqs (float array). Qwen3: rope_theta (scalar).
    auto get_meta_f = [&](const char* key_suffix, float default_val) -> float {
        char full_key[128];
        snprintf(full_key, 128, "%s.%s", prefix, key_suffix);
        auto it = reader.metadata().find(full_key);
        if (it != reader.metadata().end()) {
            if (auto* v = std::get_if<float>(&it->second)) return *v;
        }
        return default_val;
    };
    float rope_theta = get_meta_f("rope.freq_base", 1000000.0f);

    // Llama 3: rope_freqs array overrides rope_theta.
    // rope_freqs[i] = theta^{ -2*(i % 2) / head_dim } for position i.
    // We store rope_theta and let server RoPE kernel compute freq from pos.
    // If rope_freqs exists, derive rope_theta from freq[0].
    {
        char rope_freqs_key[128];
        snprintf(rope_freqs_key, 128, "%s.rope_freqs", prefix);
        auto rf_it = reader.metadata().find(rope_freqs_key);
        if (rf_it != reader.metadata().end()) {
            auto* rf = std::get_if<std::vector<float>>(&rf_it->second);
            if (rf && !rf->empty()) {
                // freq[0] = theta^{-2/head_dim}. Extract theta: theta = freq[0]^{ -head_dim/2 }
                float f0 = (*rf)[0];
                if (f0 > 0) {
                    rope_theta = powf(f0, -(float)hd / 2.0f);
                    printf("  Llama 3 rope: rope_freqs[0]=%.6f → rope_theta=%.0f\n", f0, rope_theta);
                }
            }
        }
    }

    printf("Config: %d layers, H=%d, I=%d, nqh=%d, nkv=%d, hd=%d, V=%d, rope_theta=%.0f\n",
           NL, H, I, nqh, nkv, hd, V, rope_theta);

    if (NL == 0 || H == 0) { fprintf(stderr, "FAIL: bad config\n"); return 1; }

    // Load all GGUF data into memory
    GGUFFile gguf_mem = load_whole_file(gguf_path);

    // Compute tensor data section offset: after header + metadata + tensor infos (aligned)
    // Read from memory: magic(4) + version(4) + tensor_count(8) + meta_count(8) = 24
    uint64_t total_tensors = *(const uint64_t*)(gguf_mem.data + 8);
    uint64_t meta_count = *(const uint64_t*)(gguf_mem.data + 16);

    uint64_t tensor_data_off = 24;  // skip header
    {
        const uint8_t* pp = gguf_mem.data + 24;
        // Skip metadata
        for (uint64_t i = 0; i < meta_count; i++) {
            uint64_t klen = *(const uint64_t*)pp; pp += 8;
            pp += klen;
            uint32_t vtype = *(const uint32_t*)pp; pp += 4;
            if (vtype == 8) { uint64_t slen = *(const uint64_t*)pp; pp += 8; pp += slen; }
            else if (vtype == 9) {
                uint32_t atype = *(const uint32_t*)pp; pp += 4;
                uint64_t alen = *(const uint64_t*)pp; pp += 8;
                if (atype == 8) { for (uint64_t j = 0; j < alen; j++) { uint64_t sl = *(const uint64_t*)pp; pp += 8; pp += sl; } }
                else { pp += alen * 4; }
            } else if (vtype <= 7 || vtype == 10 || vtype == 11) {
                int sz = (vtype == 0 || vtype == 1 || vtype == 7) ? 1 :
                         (vtype == 2 || vtype == 3 || vtype == 12) ? 2 :
                         (vtype == 4 || vtype == 5 || vtype == 6) ? 4 :
                         (vtype == 10 || vtype == 11 || vtype == 13) ? 8 : 4;
                pp += sz;
            } else { pp += 4; }
        }
        // Skip tensor info entries
        for (uint64_t i = 0; i < total_tensors; i++) {
            uint64_t nlen = *(const uint64_t*)pp; pp += 8;
            pp += nlen;
            uint32_t ndims = *(const uint32_t*)pp; pp += 4;
            pp += ndims * 8;
            pp += 4;  // type
            pp += 8;  // offset
        }
        // Align to 32 bytes
        tensor_data_off = (uint64_t)(pp - gguf_mem.data);
        tensor_data_off = (tensor_data_off + 31) & ~31;
        printf("Tensor data offset: %llu\n", (unsigned long long)tensor_data_off);
    }

    // Process all tensors from GGUF reader
    auto& tensors = reader.tensors();

    // Buffers for norms (collected per layer)
    std::vector<float> input_norms((size_t)NL * H);
    std::vector<float> post_norms((size_t)NL * H);
    std::vector<float> q_norms((size_t)NL * hd);
    std::vector<float> k_norms((size_t)NL * hd);

    // Process each tensor
    for (auto& ti : tensors) {
        
        char bw_name[128];
        if (!map_tensor_name(ti.name.c_str(), bw_name, sizeof(bw_name))) {
            printf("  SKIP: %s (unmapped)\n", ti.name.c_str());
            continue;
        }

        // Get file data pointer — GGUF offset is relative to tensor data section start
        uint64_t file_offset = tensor_data_off + ti.offset;
        if (file_offset + ti.file_size > gguf_mem.size) {
            fprintf(stderr, "  ERROR: %s offset out of bounds (file_offset=%llu + size=%llu > %zu)\n",
                    ti.name.c_str(), (unsigned long long)file_offset,
                    (unsigned long long)ti.file_size, gguf_mem.size);
            continue;
        }
        const uint8_t* src = gguf_mem.data + file_offset;

        int l = extract_blk_layer(ti.name.c_str());

        if (ti.type == GGML_TYPE_F32 || ti.type == GGML_TYPE_F16) {
            // Handle norm weights (F32 or F16)
            uint64_t n_el = ti.nelements();
            std::vector<float> f32_buf(n_el);
            if (ti.type == GGML_TYPE_F16) {
                for (uint64_t i = 0; i < n_el; i++)
                    dequant_f16(src + i * 2, &f32_buf[i]);
            } else {
                memcpy(f32_buf.data(), src, n_el * 4);
            }

            // Check which norm this is
            const char* suf = strstr(ti.name.c_str(), "attn_norm.weight");
            if (suf) {
                memcpy(&input_norms[(size_t)l * H], f32_buf.data(), H * 4);
                continue;  // Will write at end
            }
            suf = strstr(ti.name.c_str(), "ffn_norm.weight");
            if (suf) {
                memcpy(&post_norms[(size_t)l * H], f32_buf.data(), H * 4);
                continue;
            }
            suf = strstr(ti.name.c_str(), "attn_q_norm.weight");
            if (suf) {
                memcpy(&q_norms[(size_t)l * hd], f32_buf.data(), hd * 4);
                continue;
            }
            suf = strstr(ti.name.c_str(), "attn_k_norm.weight");
            if (suf) {
                memcpy(&k_norms[(size_t)l * hd], f32_buf.data(), hd * 4);
                continue;
            }
            suf = strstr(ti.name.c_str(), "output_norm.weight");
            if (suf) {
                write_f32(out_dir, "final_norm", f32_buf.data(), n_el);
                continue;
            }

            // Other F32 — treat as weight to requant
            // (unlikely for Qwen3)
            printf("  F32 FALLTHRU: %s\n", ti.name.c_str());
        }

        // Handle Q8_0 quantized tensors
        if (ti.type == GGML_TYPE_Q8_0) {
            uint64_t n_el = ti.nelements();
            // GGUF stores weights transposed from HF: [K, N] where K=input_dim, N=output_dim
            // Same as our format
            uint64_t K = ti.shape[0];  // input dim
            uint64_t N = ti.shape.size() > 1 ? ti.shape[1] : 1;  // output dim

            printf("  Converting %s: [%llu x %llu] Q8_0\n",
                   ti.name.c_str(), (unsigned long long)N,
                   (unsigned long long)K);

            // Skip very small tensors (like norms in Q8_0 — shouldn't happen)
            if (n_el < 1024) {
                printf("    -> too small, skipping\n");
                continue;
            }

            // Dequantize to FP32
            // Check for inf/nan in Q8_0 blocks first
            int n_blocks = (int)((n_el + 31) / 32);
            int inf_blocks = 0;
            for (int bi = 0; bi < n_blocks; bi++) {
                float d;
                dequant_f16(src + bi * 34, &d);
                if (std::isinf(d) || std::isnan(d)) inf_blocks++;
            }
            if (inf_blocks > 0) {
                fprintf(stderr, "    WARNING: %d/%d Q8_0 blocks have inf/nan scale\n", inf_blocks, n_blocks);
            }
            std::vector<float> f32_buf(n_el);
            dequant_q8_0(src, f32_buf.data(), n_el);

            // Requantize to INT4
            // f32_buf is stored in row-major [N, K] — each row is K elements
            // requant_int4 expects data in [N, K] layout (N rows, K columns)
            auto i4 = requant_int4(f32_buf.data(), (int)N, (int)K);

            // Write
            write_int4_weight(out_dir, bw_name, i4.packed.data(), i4.scales.data(), i4.K, i4.N);
        }

        // Handle Q4_K quantized tensors (GGUF Q4_K_M)
        if (ti.type == GGML_TYPE_Q4_K || ti.type == GGML_TYPE_Q4_K_M) {
            uint64_t n_el = ti.nelements();
            uint64_t K = ti.shape[0];
            uint64_t N = ti.shape.size() > 1 ? ti.shape[1] : 1;

            printf("  Converting %s: [%llu x %llu] Q4_K\n",
                   ti.name.c_str(), (unsigned long long)N,
                   (unsigned long long)K);

            if (n_el < 1024) {
                printf("    -> too small, skipping\n");
                continue;
            }

            // Dequantize Q4_K -> FP32
            std::vector<float> f32_buf(n_el);
            dequant_q4_K(src, f32_buf.data(), n_el);

            // Requantize to INT4 block-16 (symmetric)
            auto i4 = requant_int4(f32_buf.data(), (int)N, (int)K);

            // Write
            write_int4_weight(out_dir, bw_name, i4.packed.data(), i4.scales.data(), i4.K, i4.N);
        }

        // Handle Q6_K quantized tensors (used for lm_head + some ffn_down in Q4_K_M)
        if (ti.type == GGML_TYPE_Q6_K) {
            uint64_t n_el = ti.nelements();
            uint64_t K = ti.shape[0];
            uint64_t N = ti.shape.size() > 1 ? ti.shape[1] : 1;

            printf("  Converting %s: [%llu x %llu] Q6_K\n",
                   ti.name.c_str(), (unsigned long long)N,
                   (unsigned long long)K);

            if (n_el < 1024) {
                printf("    -> too small, skipping\n");
                continue;
            }

            std::vector<float> f32_buf(n_el);
            dequant_q6_K(src, f32_buf.data(), n_el);

            auto i4 = requant_int4(f32_buf.data(), (int)N, (int)K);
            write_int4_weight(out_dir, bw_name, i4.packed.data(), i4.scales.data(), i4.K, i4.N);
        }
    }

    // Write norm files
    for (int l = 0; l < NL; l++) {
        char name[64];
        snprintf(name, 64, "%d_input_layernorm", l);
        write_f32(out_dir, name, &input_norms[(size_t)l * H], H);
        snprintf(name, 64, "%d_post_attention_layernorm", l);
        write_f32(out_dir, name, &post_norms[(size_t)l * H], H);
    }

    // Write Q/K head norms
    write_qk_norms(out_dir, NL, hd, q_norms.data(), k_norms.data());

    // Write RoPE config (used by server at runtime)
    {
        char path[256];
        snprintf(path, 256, "%s/rope_config.f32", out_dir);
        float rope_cfg[2] = {rope_theta, (float)hd};
        write_file(path, rope_cfg, 8);
        printf("  rope_config: theta=%.0f, hd=%d\n", rope_theta, hd);
    }

    // Export tokenizer from GGUF metadata (BPE format)
    {
        // Read GGUF tokenizer metadata
        auto tokens_it = reader.metadata().find("tokenizer.ggml.tokens");
        auto scores_it = reader.metadata().find("tokenizer.ggml.scores");
        auto merges_it = reader.metadata().find("tokenizer.ggml.merges");
        auto bos_it = reader.metadata().find("tokenizer.ggml.bos_token_id");
        auto eos_it = reader.metadata().find("tokenizer.ggml.eos_token_id");

        int num_tokens = 0;
        const std::vector<std::string>* tokens = nullptr;
        const std::vector<float>* scores = nullptr;
        const std::vector<std::string>* merges = nullptr;

        if (tokens_it != reader.metadata().end())
            tokens = std::get_if<std::vector<std::string>>(&tokens_it->second);
        if (scores_it != reader.metadata().end())
            scores = std::get_if<std::vector<float>>(&scores_it->second);
        if (merges_it != reader.metadata().end())
            merges = std::get_if<std::vector<std::string>>(&merges_it->second);

        if (tokens) num_tokens = (int)tokens->size();

        if (num_tokens > 0) {
            // Export tokenizer in BpeTokenizer::load() binary format
            char path[256];
            snprintf(path, 256, "%s/tokenizer_data.bin", out_dir);
            FILE* f = fopen(path, "wb");
            if (f) {
                // Header: num_vocab, num_merges, num_added
                uint32_t num_vocab = (uint32_t)num_tokens;
                uint32_t num_merges_u = merges ? (uint32_t)merges->size() : 0;
                uint32_t num_added_u = 0;
                fwrite(&num_vocab, 4, 1, f);
                fwrite(&num_merges_u, 4, 1, f);
                fwrite(&num_added_u, 4, 1, f);

                // Byte encoder: standard GPT-2 mapping (bytes → unicode codepoints)
                for (int i = 0; i < 256; i++) {
                    uint32_t cp;
                    if (i < 33) cp = 256 + i;
                    else if (i == 127) cp = 259;
                    else if (i >= 128 && i <= 160) cp = 288 + (i - 128);
                    else if (i == 173) cp = 321;
                    else cp = (uint32_t)i;
                    fwrite(&cp, 4, 1, f);
                }

                // Vocab entries: [id(uint32), len(uint16), string]
                for (uint32_t id = 0; id < num_vocab; id++) {
                    const std::string& s = (*tokens)[id];
                    uint16_t len = (uint16_t)s.size();
                    fwrite(&id, 4, 1, f);
                    fwrite(&len, 2, 1, f);
                    fwrite(s.data(), 1, len, f);
                }

                // No added tokens

                // Merges: [left_len(uint16), left_str, right_len(uint16), right_str]
                if (merges) {
                    for (const auto& merge_str : *merges) {
                        // Merge format: "left right"
                        size_t space = merge_str.find(' ');
                        std::string left = merge_str.substr(0, space);
                        std::string right = merge_str.substr(space + 1);
                        uint16_t ll = (uint16_t)left.size();
                        uint16_t rl = (uint16_t)right.size();
                        fwrite(&ll, 2, 1, f);
                        fwrite(left.data(), 1, ll, f);
                        fwrite(&rl, 2, 1, f);
                        fwrite(right.data(), 1, rl, f);
                    }
                }

                fclose(f);
                printf("  tokenizer: %d tokens, %u merges -> %s\n",
                       num_tokens, num_merges_u, path);
            }
        } else {
            printf("  tokenizer: not found in GGUF metadata\n");
        }
    }

    printf("\nDone. Output: %s\n", out_dir);
    return 0;
}
