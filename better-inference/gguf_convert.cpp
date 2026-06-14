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

// Transpose FP32 buffer from GGUF [K][N] layout to kernel [N][K] layout.
// GGUF stores row-major: data[k * N + n] = weight[input_k][output_n]
// Kernel expects:       data[n * K + k] = weight[output_n][input_k]
static void transpose_f32(float* buf, int K, int N) {
    std::vector<float> tmp((size_t)K * N);
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++)
            tmp[(size_t)n * K + k] = buf[(size_t)k * N + n];
    memcpy(buf, tmp.data(), (size_t)K * N * 4);
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

// Write FP16 weight (lossless, for high quality)
static void write_fp16_weight(const char* out_dir, const char* name, const float* data, int K, int N) {
    char path[256];
    snprintf(path, 256, "%s/%s.fp16", out_dir, name);
    FILE* f = fopen(path, "wb");
    if (!f) return;
    // Header: [K, N]
    int hdr[2] = {K, N};
    fwrite(hdr, 4, 2, f);
    // FP16 data (transpose KxN -> NxK for row-major GEMV)
    std::vector<uint16_t> fp16((size_t)N * K);
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            float v = data[n * (size_t)K + k];  // [N, K] -> row n
            // Float32 to float16
            uint32_t f = *(uint32_t*)&v;
            uint16_t h;
            int exp = (f >> 23) & 0xFF;
            if (exp == 0) { h = 0; }
            else if (exp == 255) { h = (f & 0x80000000) ? 0xFC00 : 0x7C00; }
            else {
                int sign = (f >> 16) & 0x8000;
                exp -= 127;
                if (exp < -14) exp = -14;
                else if (exp > 15) exp = 15;
                h = sign | ((exp + 15) << 10) | ((f >> 13) & 0x3FF);
            }
            fp16[n * (size_t)K + k] = h;
        }
    }
    fwrite(fp16.data(), 2, (size_t)N * K, f);
    fclose(f);
    double mb = (double)N * K * 2 / (1024 * 1024);
    printf("  %s: %dx%d FP16 %.1fMB\n", name, N, K, mb);
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
        fprintf(stderr, "Usage: %s model.gguf output_dir/ [--fp16]\n", argv[0]);
        fprintf(stderr, "  --fp16: output FP16 instead of INT4 (lossless, larger)\n");
        return 1;
    }

    const char* gguf_path = argv[1];
    const char* out_dir = argv[2];
    bool fp16_mode = (argc >= 4 && strcmp(argv[3], "--fp16") == 0);
    if (fp16_mode) printf("FP16 mode: preserving original precision (larger files)\n");

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
    // Prefix matches architecture name for metadata keys.
    // Qwen2 → qwen2, Qwen3 → qwen3, Llama → llama
    std::string prefix = arch;
    // Normalize: lowercase
    for (auto& c : prefix) if (c >= 'A' && c <= 'Z') c += 32;
    bool is_llama = (arch == "llama");

    auto get_meta = [&](const char* key_suffix, int default_val) -> int {
        char full_key[128];
        snprintf(full_key, 128, "%s.%s", prefix.c_str(), key_suffix);
        return reader.meta_int(full_key, default_val);
    };

    int NL = get_meta("block_count", 0);
    int H = get_meta("embedding_length", 0);
    int I = get_meta("feed_forward_length", 0);
    int nqh = get_meta("attention.head_count", 0);
    int nkv = get_meta("attention.head_count_kv", 0);
    // Gemma 4 stores head_count_kv as a per-layer array (all values equal)
    // If nkv == 0 from the array, try extracting the first element
    if (nkv == 0) {
        const auto& meta = reader.metadata();
        auto it = meta.find("gemma4.attention.head_count_kv");
        if (it != meta.end()) {
            if (auto* v = std::get_if<std::vector<int32_t>>(&it->second)) {
                if (!v->empty()) nkv = (*v)[0];
            }
        }
    }
    int hd = get_meta("attention.key_length", 0);
    if (hd == 0) hd = (nqh > 0) ? H / nqh : 0;
    // Get vocab size from tokens array count in metadata
    auto it = reader.metadata().find("tokenizer.ggml.tokens");
    int V = 151936;
    if (it != reader.metadata().end()) {
        if (auto* sa = std::get_if<std::vector<std::string>>(&it->second))
            V = (int)sa->size();
    }
    printf("V = %d\n", V);

    // Get RoPE config
    // GGUF v3 uses nested prefixes. Keys like "rope.freq_base" are stored as
    // "general.repo_url.key" (e.g., "https://huggingface.co/.../llama.rope.freq_base").
    // We need to search for any key ending with the suffix.
    auto get_meta_f = [&](const char* key_suffix, float default_val) -> float {
        char full_key[128];
        snprintf(full_key, 128, "%s.%s", prefix, key_suffix);
        auto it = reader.metadata().find(full_key);
        if (it != reader.metadata().end()) {
            if (auto* v = std::get_if<float>(&it->second)) return *v;
        }
        // Fallback: search for any key ending with key_suffix
        for (const auto& kv : reader.metadata()) {
            if (kv.first.length() > strlen(key_suffix) &&
                kv.first.compare(kv.first.length() - strlen(key_suffix), strlen(key_suffix), key_suffix) == 0) {
                if (auto* v = std::get_if<float>(&kv.second)) return *v;
            }
        }
        return default_val;
    };
    float rope_theta = get_meta_f("rope.freq_base", 0.0f);
    // Fallback: if rope_theta is 0, search file for klen(20)+"rope.freq_base" pattern
    if (rope_theta == 0.0f) {
        FILE* f = fopen(gguf_path, "rb");
        if (f) {
            char buf[8192];
            size_t file_pos = 0;
            while (fread(buf, 1, 8192, f) == 8192) {
                for (size_t i = 0; i + 22 < 8192; i++) {  // 8 (klen) + 14 (key) <= 8192
                    uint64_t klen = *(uint64_t*)(buf + i);
                    if (klen == 20 && memcmp(buf + i + 8, "rope.freq_base", 14) == 0) {
                        // Found klen=20 + "rope.freq_base". Value is at i + 8 + 20 + 4 = i + 32
                        float v = *(float*)(buf + i + 32);
                        if (v > 0) {
                            rope_theta = v;
                            printf("  Fallback read rope.freq_base = %.0f\n", rope_theta);
                            break;
                        }
                    }
                }
                file_pos += 8192 - 22;
                if (file_pos > 100000) break;  // metadata is at start of file
                fseek(f, file_pos, SEEK_SET);
            }
            fclose(f);
        }
        if (rope_theta == 0.0f) rope_theta = 1000000.0f;  // default
    }

    // Llama 3: rope_freqs array overrides rope_theta.
    // rope_freqs[i] = theta^{ -2*(i % 2) / head_dim } for position i.
    // We store rope_theta and let server RoPE kernel compute freq from pos.
    // If rope_freqs exists, derive rope_theta from freq[0].
    {
        char rope_freqs_key[128];
        snprintf(rope_freqs_key, 128, "%s.rope_freqs", prefix.c_str());
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

    // Tensor data offset: after header + metadata + tensor infos (aligned to 32)
    // Compute from the last tensor's offset + file_size in GGUFReader data.
    // The reader holds the raw file data, we need the start of the data section.
    // Simplest: compute from the reader's tensor info — last tensor offset + size
    // gives data section end, which equals data section start + data section size.
    // Actually the reader doesn't have data section start readily. Compute manually:
    uint64_t tensor_data_off = 24;  // skip header
    {
        // Re-read header values
        uint64_t meta_count = *(const uint64_t*)(gguf_mem.data + 16);
        uint64_t total_tensors = *(const uint64_t*)(gguf_mem.data + 8);
        // Use GGUFReader's internal position to get tensor data offset
        // The reader already parsed metadata + tensor info. We need the offset
        // AFTER tensor info, aligned to 32 bytes.
        // Compute from the file by finding the last tensor offset and its file_size.
        auto& tensors = reader.tensors();
        if (!tensors.empty()) {
            uint64_t max_end = 0;
            for (auto& t : tensors) {
                uint64_t end = t.offset + t.file_size;
                if (end > max_end) max_end = end;
            }
            // max_end is relative to data section start. The actual data section
            // end in file = tensor_data_off + max_end. But we don't know tensor_data_off.
            // Instead, compute from the first tensor offset.
            // The first tensor's offset is 0, so tensor_data_off = first_tensor_file_pos.
            // We can't get this from the reader since it stores relative offsets.
            // Fallback: walk raw memory
        }
        // Walk raw memory to compute tensor_data_off
        const uint8_t* pp = gguf_mem.data + 24;
        fprintf(stderr, "DEBUG: gguf_mem.size=%zu, pp-offset=%llu\n", gguf_mem.size, (unsigned long long)(pp - gguf_mem.data));
        for (uint64_t i = 0; i < meta_count; i++) {
            if ((uint64_t)(pp - gguf_mem.data) + 12 > gguf_mem.size) {
                fprintf(stderr, "META WALK: hit end of file at metadata %llu\n", (unsigned long long)i);
                break;
            }
            uint64_t klen = *(const uint64_t*)pp; pp += 8;
            pp += klen;
            uint32_t raw_type = *(const uint32_t*)pp; pp += 4;
            if (raw_type == 0) { pp += 1; }
            else if (raw_type == 1) { pp += 1; }
            else if (raw_type == 2) { pp += 2; }
            else if (raw_type == 3) { pp += 2; }
            else if (raw_type == 4) { pp += 4; }
            else if (raw_type == 5) { pp += 4; }
            else if (raw_type == 6) { pp += 4; }
            else if (raw_type == 7) { pp += 1; }
            else if (raw_type == 8) { uint64_t slen = *(const uint64_t*)pp; pp += 8; pp += slen; }
            else if (raw_type == 9) {
                uint32_t atype = *(const uint32_t*)pp; pp += 4;
                uint64_t alen = *(const uint64_t*)pp; pp += 8;
                if (atype == 8) { while (alen-- > 0) { uint64_t sl = *(const uint64_t*)pp; pp += 8; pp += sl; } }
                else if (atype == 7 || atype == 0 || atype == 1) { pp += alen; }
                else if (atype == 2 || atype == 3) { pp += alen * 2; }
                else { pp += alen * 4; }
            }
            else if (raw_type == 10 || raw_type == 11) { pp += 8; }
            else if (raw_type == 12) { pp += 2; }
            else if (raw_type == 13) { pp += 8; }
            else { pp += 4; }
        }
        // Skip tensor info entries
        uint64_t total_tensors_from_header = *(const uint64_t*)(gguf_mem.data + 8);
        for (uint64_t i = 0; i < total_tensors_from_header; i++) {
            uint64_t nlen = *(const uint64_t*)pp; pp += 8;
            pp += nlen;
            uint32_t ndims = *(const uint32_t*)pp; pp += 4;
            pp += ndims * 8;
            pp += 4;  // type
            pp += 8;  // offset
        }
        tensor_data_off = (uint64_t)(pp - gguf_mem.data);
        tensor_data_off = (tensor_data_off + 31) & ~31;
        printf("Tensor data offset: %llu\n", (unsigned long long)tensor_data_off);
        fflush(stdout);
    }

    // Process all tensors from GGUF reader
    auto& tensors = reader.tensors();
    fprintf(stderr, "Processing %zu tensors...\n", tensors.size());
    fprintf(stderr, "DEBUG: tensor_data_off=%llu, first tensor '%s' offset=%llu file_size=%llu\n",
        (unsigned long long)tensor_data_off, tensors[0].name.c_str(),
        (unsigned long long)tensors[0].offset,
        (unsigned long long)tensors[0].file_size);

    // Buffers for norms (collected per layer)
    std::vector<float> input_norms((size_t)NL * H);
    std::vector<float> post_norms((size_t)NL * H);
    std::vector<float> q_norms((size_t)NL * hd, 1.0f);  // init to 1.0 (identity)
    std::vector<float> k_norms((size_t)NL * hd, 1.0f);  // init to 1.0 (identity)

    // Process each tensor
    for (auto& ti : tensors) {
        
        char bw_name[128];
        if (!map_tensor_name(ti.name.c_str(), bw_name, sizeof(bw_name))) {
            printf("  SKIP: %s (unmapped)\n", ti.name.c_str());
            continue;
        }

        // Get file data pointer — GGUF v3 tensor offset is RELATIVE to tensor data section
        uint64_t file_offset = tensor_data_off + ti.offset;
        if (file_offset + ti.file_size > gguf_mem.size) {
            fprintf(stderr, "  ERROR: %s offset out of bounds (data_off=%llu + ti.off=%llu + size=%llu > %zu)\n",
                    ti.name.c_str(), (unsigned long long)tensor_data_off,
                    (unsigned long long)ti.offset,
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

        // Handle Q5_0 quantized tensors
        if (ti.type == GGML_TYPE_Q5_0) {
            uint64_t n_el = ti.nelements();
            uint64_t K = ti.shape[0];
            uint64_t N = ti.shape.size() > 1 ? ti.shape[1] : 1;

            printf("  Converting %s: [%llu x %llu] Q5_0\n",
                   ti.name.c_str(), (unsigned long long)N,
                   (unsigned long long)K);

            if (n_el < 1024) { printf("    -> too small, skipping\n"); continue; }

            std::vector<float> f32_buf(n_el);
            dequant_q5_0(src, f32_buf.data(), n_el);
            transpose_f32(f32_buf.data(), (int)K, (int)N);

            if (fp16_mode) {
                write_fp16_weight(out_dir, bw_name, f32_buf.data(), (int)K, (int)N);
            } else {
                auto i4 = requant_int4(f32_buf.data(), (int)N, (int)K);
                write_int4_weight(out_dir, bw_name, i4.packed.data(), i4.scales.data(), i4.K, i4.N);
            }
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
            transpose_f32(f32_buf.data(), (int)K, (int)N);

            if (fp16_mode) {
                write_fp16_weight(out_dir, bw_name, f32_buf.data(), (int)K, (int)N);
            } else {
                // Requantize to INT4
                auto i4 = requant_int4(f32_buf.data(), (int)N, (int)K);
                write_int4_weight(out_dir, bw_name, i4.packed.data(), i4.scales.data(), i4.K, i4.N);
            }
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
            // Large tensors (embed_tokens, lm_head > 100M elements) — process in chunks
            if (n_el > 100 * 1024 * 1024) {
                printf("    -> too large (%llu elements), chunked conversion\n",
                       (unsigned long long)n_el);
                
                // Create INT4 header files first
                char int4_path[256], scale_path[256];
                snprintf(int4_path, 256, "%s/%s.int4_t", out_dir, bw_name);
                snprintf(scale_path, 256, "%s/%s.scale_t", out_dir, bw_name);
                
                if (!std::filesystem::exists(int4_path)) {
                    // Write headers
                    int num_kb = (int)K / 16;
                    {
                        FILE* hf = fopen(int4_path, "wb");
                        int hdr[5] = {(int)K, (int)N, 16, num_kb, 1};
                        fwrite(hdr, 4, 5, hf);
                        fclose(hf);
                    }
                    {
                        FILE* hf = fopen(scale_path, "wb");
                        int hdr[5] = {0, 0, 0, num_kb, (int)N};
                        fwrite(hdr, 4, 5, hf);
                        fclose(hf);
                    }
                }
                
                // Process N rows in chunks
                const int ROW_CHUNK = 256;
                for (int ch = 0; ch < (int)N; ch += ROW_CHUNK) {
                    int ch_end = ch + ROW_CHUNK;
                    if (ch_end > (int)N) ch_end = (int)N;
                    int ch_rows = ch_end - ch;
                    
                    // Allocate FP32 buffer for this chunk
                    uint64_t chunk_el = (uint64_t)K * ch_rows;
                    std::vector<float> chunk_f32(chunk_el);
                    
                    // Dequant each row
                    // GGUF stores row-major: data[k * N + n] but Q4_K uses blocks
                    // Q4_K row stride in file
                    uint64_t row_stride = ((K + 255) / 256) * 144;
                    for (int r = 0; r < ch_rows; r++) {
                        int src_row = ch + r;
                        const uint8_t* row_src = src + (uint64_t)src_row * row_stride;
                        dequant_q4_K(row_src, chunk_f32.data() + (uint64_t)r * K, K);
                    }
                    
                    // Transpose: GGUF [K,N] -> kernel [N,K]
                    // For chunk rows, we need [ch_rows, K] format
                    // requant_int4 expects [N, K] where N is output dimension
                    // Already have [ch_rows, K] from dequant
                    
                    // Quantize each row to INT4
                    for (int r = 0; r < ch_rows; r++) {
                        float* row = chunk_f32.data() + (uint64_t)r * K;
                        
                        // INT4 block-16 quant
                        int num_kb = K / 16;
                        std::vector<uint8_t> packed(K / 2);
                        std::vector<float> scales(num_kb);
                        
                        for (int kb = 0; kb < num_kb; kb++) {
                            float absmax = 0.0f;
                            for (int i = 0; i < 16; i++)
                                absmax = fmaxf(absmax, fabsf(row[kb * 16 + i]));
                            float scale = fmaxf(absmax, 1e-10f) / 7.0f;
                            if (!std::isfinite(scale)) scale = 1.0f;
                            scales[kb] = scale;
                            for (int i = 0; i < 16; i += 2) {
                                int v0 = (int)roundf(row[kb * 16 + i] / scale);
                                int v1 = (int)roundf(row[kb * 16 + i + 1] / scale);
                                v0 = v0 < -7 ? -7 : (v0 > 7 ? 7 : v0);
                                v1 = v1 < -7 ? -7 : (v1 > 7 ? 7 : v1);
                                uint8_t packed_byte = ((uint8_t)(v0 + 8) & 0x0F) |
                                    (((uint8_t)(v1 + 8) & 0x0F) << 4);
                                packed[(uint64_t)kb * 8 + i / 2] = packed_byte;
                            }
                        }
                        
                        // Append to file
                        {
                            FILE* pf = fopen(int4_path, "ab");
                            fwrite(packed.data(), 1, K / 2, pf);
                            fclose(pf);
                        }
                        {
                            FILE* sf = fopen(scale_path, "ab");
                            fwrite(scales.data(), 4, num_kb, sf);
                            fclose(sf);
                        }
                    }
                    
                    printf("    chunk %d-%d/%d\n", ch, ch_end, (int)N);
                }
                
                double mb = ((double)N * K / 2 + (double)N * (K / 16) * 4) / (1024 * 1024);
                printf("  %s: %dx%d INT4 %.1fMB (chunked)\n", bw_name, N, K, mb);
                continue;
            }

            // Dequantize Q4_K -> FP32
            std::vector<float> f32_buf(n_el);
            dequant_q4_K(src, f32_buf.data(), n_el);
            transpose_f32(f32_buf.data(), (int)K, (int)N);

            if (fp16_mode) {
                // Write FP16 directly (lossless)
                write_fp16_weight(out_dir, bw_name, f32_buf.data(), (int)K, (int)N);
            } else {
                // Requantize to INT4 block-16 (symmetric)
                auto i4 = requant_int4(f32_buf.data(), (int)N, (int)K);
                write_int4_weight(out_dir, bw_name, i4.packed.data(), i4.scales.data(), i4.K, i4.N);
            }
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
            if (n_el > 100 * 1024 * 1024) {
                printf("    -> too large (%llu elements), skipping INT4 conversion\n",
                       (unsigned long long)n_el);
                continue;
            }

            std::vector<float> f32_buf(n_el);
            dequant_q6_K(src, f32_buf.data(), n_el);
            transpose_f32(f32_buf.data(), (int)K, (int)N);

            if (fp16_mode) {
                write_fp16_weight(out_dir, bw_name, f32_buf.data(), (int)K, (int)N);
            } else {
                auto i4 = requant_int4(f32_buf.data(), (int)N, (int)K);
                write_int4_weight(out_dir, bw_name, i4.packed.data(), i4.scales.data(), i4.K, i4.N);
            }
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
                // 68 non-printable bytes: 0-32 (33), 127 (1), 128-160 (33), 173 (1)
                // → codepoints 256-323
                for (int i = 0; i < 256; i++) {
                    uint32_t cp;
                    if (i < 33) cp = 256 + i;                          // 0-32 → U+0100-0120
                    else if (i == 127) cp = 289;                        // 127 → U+0121
                    else if (i >= 128 && i <= 160) cp = 290 + (i-128); // 128-160 → U+0122-0142
                    else if (i == 173) cp = 323;                        // 173 → U+0143
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
