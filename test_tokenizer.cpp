// test_tokenizer.cu — Quick test for BPE tokenizer
// Build: g++ -std=c++17 -I include test_tokenizer.cu -o test_tokenizer
// (No CUDA needed, pure C++)

#include <iostream>
#include <string>
#include <vector>
#include "blackwell/bpe_tokenizer.h"

int main() {
    blackwell::BpeTokenizer tok;
    if (tok.load("tokenizer_data.bin") != 0) return 1;

    // Test encode
    const char* tests[] = {
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "Hello, world!",
        "   spaces   ",
        "12345",
        "def foo(x):",
    };

    for (const char* text : tests) {
        auto ids = tok.encode(text);
        std::cout << "encode(\"" << text << "\") → [";
        for (size_t i = 0; i < ids.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << ids[i];
        }
        std::cout << "]" << std::endl;

        // Decode back
        std::string decoded = tok.decode(ids);
        std::cout << "decode → \"" << decoded << "\"" << std::endl;

        if (decoded != text) {
            std::cout << "  WARNING: roundtrip mismatch!" << std::endl;
        }
        std::cout << std::endl;
    }

    // Verify against known Qwen tokenization
    // "Hello" should tokenize to a small number of tokens
    auto ids = tok.encode("Hello");
    std::cout << "\"Hello\" → " << ids.size() << " tokens: [";
    for (auto id : ids) std::cout << id << " ";
    std::cout << "]" << std::endl;

    // Test special token decode
    std::cout << "EOS (151643) → \"" << tok.decode(151643) << "\"" << std::endl;
    std::cout << "<|im_start|> (151644) → \"" << tok.decode(151644) << "\"" << std::endl;

    return 0;
}
