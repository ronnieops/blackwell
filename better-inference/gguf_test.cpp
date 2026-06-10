// Test GGUF parser
#include <cstdio>
#include "gguf.h"

int main(int argc, char** argv) {
    const char* path = argc >= 2 ? argv[1] 
        : "/mnt/data/ai/hf/models--Qwen--Qwen3-1.7B-GGUF/blobs/"
          "061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a";

    GGUFReader r(path);
    if (!r.valid()) {
        fprintf(stderr, "FAIL: can't read %s\n", path);
        return 1;
    }

    printf("GGUF v%u\n", r.version());
    printf("Metadata:\n");
    for (auto& [k, v] : r.metadata()) {
        printf("  %s: ", k.c_str());
        if (auto* s = std::get_if<std::string>(&v)) {
            if (s->size() > 80) printf("<%zu chars>", s->size());
            else printf("\"%s\"", s->c_str());
        } else if (auto* b = std::get_if<bool>(&v)) {
            printf("%s", *b ? "true" : "false");
        } else if (auto* i32 = std::get_if<int32_t>(&v)) {
            printf("%d", *i32);
        } else if (auto* u64 = std::get_if<uint64_t>(&v)) {
            printf("%llu", (unsigned long long)*u64);
        } else if (auto* f = std::get_if<float>(&v)) {
            printf("%f", *f);
        } else if (auto* d = std::get_if<double>(&v)) {
            printf("%f", *d);
        } else if (auto* sa = std::get_if<std::vector<std::string>>(&v)) {
            printf("[%zu strings]", sa->size());
        } else if (auto* ia = std::get_if<std::vector<int32_t>>(&v)) {
            printf("[%zu ints]", ia->size());
        } else {
            printf("<?>");
        }
        printf("\n");
    }

    auto tensors = r.tensors();
    printf("\nTensors (%zu):\n", tensors.size());
    for (size_t i = 0; i < tensors.size() && i < 20; i++) {
        auto& t = tensors[i];
        printf("  [%zu] %s: ", i, t.name.c_str());
        for (auto s : t.shape) printf("%llu ", (unsigned long long)s);
        printf("type=%u offset=%llu size=%llu\n",
            (unsigned)t.type, (unsigned long long)t.offset,
            (unsigned long long)t.file_size);
    }
    if (tensors.size() > 20)
        printf("  ... %zu more\n", tensors.size() - 20);

    // Extract model config
    auto arch = r.meta<std::string>("general.architecture", "");
    printf("\nArchitecture: %s\n", arch.c_str());
    printf("Block count: %d\n", r.meta<int32_t>("llama.block_count", 0));
    printf("Head count: %d\n", r.meta<int32_t>("llama.attention.head_count", 0));
    printf("Head count KV: %d\n", r.meta<int32_t>("llama.attention.head_count_kv", 0));
    printf("Embedding length: %d\n", r.meta<int32_t>("llama.embedding_length", 0));
    printf("Feed forward length: %d\n", r.meta<int32_t>("llama.feed_forward_length", 0));

    return 0;
}
