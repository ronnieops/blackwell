// bench/tokenize_text.cu — Tokenize text via BpeTokenizer, dump token IDs to binary
// Usage: ./bench/tokenize_text "text to tokenize" [output_file.bin]
// Default output: /tmp/token_ids.bin
// Output format: [num_tokens: i32][token_ids: i32 x num_tokens]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "blackwell/bpe_tokenizer.h"

int main(int argc, char** argv) {
    const char* text = argc >= 2 ? argv[1] : "The capital of France is";
    const char* out_path = argc >= 3 ? argv[2] : "/tmp/token_ids.bin";
    
    blackwell::BpeTokenizer tokenizer;
    if (tokenizer.load("tokenizer_data.bin") != 0) {
        fprintf(stderr, "FAIL: no tokenizer_data.bin\n");
        return 1;
    }
    
    auto ids = tokenizer.encode(text);
    
    FILE* f = fopen(out_path, "wb");
    if (!f) { fprintf(stderr, "FAIL: can't write %s\n", out_path); return 1; }
    
    int n = (int)ids.size();
    fwrite(&n, sizeof(int), 1, f);
    fwrite(ids.data(), sizeof(int), n, f);
    fclose(f);
    
    printf("Tokenized: \"%s\" → %d tokens → %s\n", text, n, out_path);
    printf("Token IDs:");
    for (int i = 0; i < n && i < 20; ++i) printf(" %d", ids[i]);
    if (n > 20) printf(" ...");
    printf("\n");
    
    return 0;
}
