// tools/extract_weights.cpp — Extract Qwen3-1.7B BF16 weights, quantize to FP4
// Compile: g++-12 -O3 -std=c++17 tools/extract_weights.cpp -o tools/extract_weights
// Run: ./tools/extract_weights [--layers N]

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <fstream>

// BF16 -> float
inline float bf16_to_float(uint16_t bf) {
    uint32_t bits = (uint32_t)bf << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// FP4 LUT values (E2M1)
constexpr float fp4_lut[8] = {0.25f, 0.5f, 1.0f, 2.0f, -0.25f, -0.5f, -1.0f, -2.0f};

// Quantize a float to FP4 E2M1, given block scale
// Returns: FP4 value packed into low nibble of return byte, scale set via scale_out
// Called once per N values (block size)
void quantize_block(const float* data, int n, uint8_t* out_fp4, float* scale_out) {
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) {
        float a = fabsf(data[i]);
        if (a > amax) amax = a;
    }
    float scale = amax / 3.0f + 1e-9f;
    *scale_out = scale;
    
    for (int i = 0; i < n; ++i) {
        float sv = data[i] / scale;
        if (sv < -3.0f) sv = -3.0f;
        if (sv > 3.0f) sv = 3.0f;
        
        // Find nearest FP4 value
        int best = 0;
        float best_err = 1e9;
        for (int j = 0; j < 8; ++j) {
            float err = fabsf(sv - fp4_lut[j]);
            if (err < best_err) { best_err = err; best = j; }
        }
        out_fp4[i] = (uint8_t)best;
    }
}

int main(int argc, char** argv) {
    int num_layers = 4;  // default: 4 layers
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            num_layers = atoi(argv[i+1]);
        }
    }
    
    const char* path = "/mnt/data/ai/hf/qwen3-1.7b-base/model.safetensors";
    const int hidden = 2048;
    const int q_dim = 2048;   // 16 heads * 128
    const int kv_dim = 1024;  // 8 KV heads * 128
    const int block = 16;
    
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "FATAL: cannot open %s\n", path); return 1; }
    
    // Read header length
    uint64_t header_len;
    if (fread(&header_len, 8, 1, f) != 1) { fprintf(stderr, "FATAL: read error\n"); return 1; }
    
    std::string header(header_len, '\0');
    if (fread(header.data(), header_len, 1, f) != 1) { fprintf(stderr, "FATAL: read header error\n"); return 1; }
    
    uint64_t data_offset = 8 + header_len;
    
    // Simple JSON parsing: find tensor names and their data_offsets
    // Format: ...,"model.layers.0.self_attn.q_proj.weight":{"dtype":"BF16","shape":[2048,2048],"data_offsets":[X,Y]},...
    
    struct WeightInfo {
        std::string name;
        uint64_t start, end;
    };
    std::vector<WeightInfo> weights;
    
    size_t pos = 0;
    while (pos < header.size()) {
        // Find next quoted tensor name
        size_t q1 = header.find('"', pos);
        if (q1 == std::string::npos) break;
        size_t q2 = header.find('"', q1 + 1);
        if (q2 == std::string::npos) break;
        
        std::string name = header.substr(q1 + 1, q2 - q1 - 1);
        size_t rest = q2 + 1;
        
        // Skip to "data_offsets"
        size_t do_key = header.find("data_offsets", rest);
        if (do_key == std::string::npos) { pos = rest; continue; }
        size_t bracket_open = header.find('[', do_key);
        size_t comma = header.find(',', bracket_open);
        size_t bracket_close = header.find(']', comma);
        
        if (bracket_open != std::string::npos && comma != std::string::npos && bracket_close != std::string::npos) {
            uint64_t s = strtoull(header.c_str() + bracket_open + 1, nullptr, 10);
            uint64_t e = strtoull(header.c_str() + comma + 1, nullptr, 10);
            weights.push_back({name, s, e});
        }
        
        pos = bracket_close != std::string::npos ? bracket_close + 1 : rest;
    }
    
    printf("Found %zu tensors. Data offset: %lu\n", weights.size(), (unsigned long)data_offset);
    
    // Filter to weight matrices (skip norms, embeddings, etc.)
    // We need per layer: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    printf("\nExtracting weights for %d layers...\n", num_layers);
    
    // Create output directory
    system("mkdir -p /mnt/data/dev/projects/blackwell/weights");
    
    std::vector<float> buf32;
    
    for (auto& w : weights) {
        // Check if this tensor is for an attention/MLP weight within our layer range
        int layer = -1;
        bool is_weight = false;
        std::string short_name;
        
        // Parse: model.layers.N.self_attn.q_proj.weight
        if (sscanf(w.name.c_str(), "model.layers.%d.", &layer) == 1) {
            if (layer >= num_layers) continue;
            
            // Extract short type name
            for (const char* type : {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", 
                                      "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}) {
                if (w.name.find(type) != std::string::npos) {
                    short_name = std::to_string(layer) + "_" + type;
                    is_weight = true;
                    break;
                }
            }
        }
        
        if (!is_weight) continue;
        
        uint64_t tensor_size = w.end - w.start;
        uint64_t num_elements = tensor_size / 2;  // BF16 = 2 bytes each
        printf("  Reading %s (%lu elements, %lu bytes)... ", short_name.c_str(), 
               (unsigned long)num_elements, (unsigned long)tensor_size);
        fflush(stdout);
        
        // Read BF16 data
        std::vector<uint16_t> bf16_data(num_elements);
        fseek(f, data_offset + w.start, SEEK_SET);
        if (fread(bf16_data.data(), 2, num_elements, f) != num_elements) {
            printf("READ ERROR\n");
            continue;
        }
        
        // Convert to float
        buf32.resize(num_elements);
        for (uint64_t i = 0; i < num_elements; ++i) {
            buf32[i] = bf16_to_float(bf16_data[i]);
        }
        
        // Determine shape (rows x cols) for weight layout
        // For q_proj: [2048, 2048] — row-major: rows=2048, cols=2048
        // For k_proj: [1024, 2048] — rows=1024, cols=2048 (but it's transposed view)
        // The weight W is (out_features × in_features) in row-major
        // Our GEMV expects W[k][n] layout: in_features × out_features
        // But we stored packed by flattening — need to match GEMV kernel's access pattern
        //
        // GEMV kernel does: W[k * N + n_out] where k loops over in_features and
        // n_out is the per-thread output index.
        // So W layout is [in_features × out_features] = [K × N] in row-major
        //
        // From safetensors: q_proj.weight has shape [2048, 2048] where
        // first dim = out_features = q_dim, second dim = in_features = hidden
        // So it's stored as [q_dim × hidden] — NOT what our kernel expects.
        //
        // Our GEMV expects [in_features × out_features] = [hidden × q_dim]
        // So we need to transpose!
        
        // Transpose: new[i * rows + j] = old[j * cols + i]
        // where old shape = [out_features, in_features]
        // and desired shape = [in_features, out_features]
        
        // Determine old dimensions from tensor name
        int dim0, dim1;  // dim0 = out_features, dim1 = in_features (BF16 layout)
        if (short_name.find("q_proj") != std::string::npos || short_name.find("o_proj") != std::string::npos) {
            dim0 = q_dim; dim1 = hidden;
        } else if (short_name.find("k_proj") != std::string::npos || short_name.find("v_proj") != std::string::npos) {
            dim0 = kv_dim; dim1 = hidden;
        } else if (short_name.find("gate_proj") != std::string::npos || short_name.find("up_proj") != std::string::npos) {
            dim0 = 6144; dim1 = hidden;
        } else if (short_name.find("down_proj") != std::string::npos) {
            dim0 = hidden; dim1 = 6144;
        } else {
            printf("SKIP (unknown dims)\n");
            continue;
        }
        
        // Verify size
        if ((uint64_t)dim0 * dim1 != num_elements) {
            printf("SKIP (size mismatch: %d*%d=%d != %lu)\n", dim0, dim1, dim0*dim1, (unsigned long)num_elements);
            continue;
        }
        
        // Transpose: result[i * dim0 + j] = buf32[j * dim1 + i]
        // But this is huge and expensive. For our GEMV kernel, we can 
        // adjust the load pattern instead — tell the kernel that W is
        // [out_features × in_features] and adjust access.
        //
        // Actually easier: just store as-is, note the layout in output file.
        // Our GEMV does W[k * N + n_out]. If we store W as [out × in], 
        // then W[k * N + n_out] reads row k, column n_out — but row is out_features,
        // so k=0 means first output row, k=1 means second output row, etc.
        // That's WRONG. We need k to index in_features.
        //
        // Fix: store transposed. Do the transpose here.
        std::vector<float> transposed(num_elements);
        for (int i = 0; i < dim1; ++i) {      // in_features
            for (int j = 0; j < dim0; ++j) {  // out_features
                transposed[i * dim0 + j] = buf32[j * dim1 + i];
            }
        }
        
        // Quantize to FP4 in blocks of 16 along K dimension
        // Block scaling: groups of 16 elements along K (in_features), 
        // each group shares one scale.
        // Layout: FP4 data = [num_blocks_K * 16 × out_dim] 
        //         scales = [num_blocks_K × out_blocks]
        
        int num_K_blocks = (dim1 + block - 1) / block;
        int num_N_blocks = (dim0 + block - 1) / block;
        
        std::vector<uint8_t> fp4_data(num_elements);  // 1 byte per value (packed)
        std::vector<float> scales(num_K_blocks * num_N_blocks);
        
        for (int kb = 0; kb < num_K_blocks; ++kb) {
            for (int nb = 0; nb < num_N_blocks; ++nb) {
                int k_start = kb * block;
                int n_start = nb * block;
                
                // Gather block of 16×16 elements
                float block_vals[256];
                int idx = 0;
                for (int kk = 0; kk < block && k_start + kk < dim1; ++kk) {
                    for (int nn = 0; nn < block && n_start + nn < dim0; ++nn) {
                        block_vals[idx++] = transposed[(k_start + kk) * dim0 + (n_start + nn)];
                    }
                }
                
                // Compute scale for this 16×16 block
                float amax = 0.0f;
                for (int i = 0; i < idx; ++i) {
                    float a = fabsf(block_vals[i]);
                    if (a > amax) amax = a;
                }
                float scale = amax / 3.0f + 1e-9f;
                scales[kb * num_N_blocks + nb] = scale;
                
                // Quantize each element
                for (int kk = 0; kk < block && k_start + kk < dim1; ++kk) {
                    for (int nn = 0; nn < block && n_start + nn < dim0; ++nn) {
                        float v = transposed[(k_start + kk) * dim0 + (n_start + nn)] / scale;
                        if (v < -3.0f) v = -3.0f;
                        if (v > 3.0f) v = 3.0f;
                        int best = 0;
                        float best_err = 1e9;
                        for (int j = 0; j < 8; ++j) {
                            float err = fabsf(v - fp4_lut[j]);
                            if (err < best_err) { best_err = err; best = j; }
                        }
                        fp4_data[(k_start + kk) * dim0 + (n_start + nn)] = (uint8_t)best;
                    }
                }
            }
        }
        
        // Write output file
        std::string out_path = "/mnt/data/dev/projects/blackwell/weights/" + short_name + ".fp4";
        FILE* out = fopen(out_path.c_str(), "wb");
        if (!out) { printf("WRITE ERROR\n"); continue; }
        
        // Write header: dim1, dim0, block, num_K_blocks, num_N_blocks
        int header_vals[5] = {dim1, dim0, block, num_K_blocks, num_N_blocks};
        fwrite(header_vals, 4, 5, out);
        
        // Write FP4 data
        fwrite(fp4_data.data(), 1, num_elements, out);
        
        // Write scales
        fwrite(scales.data(), 4, num_K_blocks * num_N_blocks, out);
        
        fclose(out);
        
        // Check quantization error
        double mse = 0.0;
        double max_err = 0.0;
        for (uint64_t i = 0; i < num_elements; ++i) {
            float rec = fp4_lut[fp4_data[i]] * scales[i / dim0 / block * num_N_blocks + (i % dim0) / block];
            float err = fabsf(transposed[i] - rec);
            mse += err * err;
            if (err > max_err) max_err = err;
        }
        mse /= num_elements;
        
        printf("done (max_err=%.4f, MSE=%.6f)\n", max_err, mse);
    }
    
    fclose(f);
    
    printf("\nDone. Weights written to /mnt/data/dev/projects/blackwell/weights/\n");
    return 0;
}
