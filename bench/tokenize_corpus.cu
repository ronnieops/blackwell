// bench/tokenize_corpus.cu — Tokenize multi-line corpus, dump all token IDs
// Reads lines from stdin (max 1024 chars each), outputs to binary file.
//
// Usage: python3 -c "
//   with open('/tmp/wiki_corpus.txt','w') as f:
//     for s in open('wiki_sentences.txt'): f.write(s)
// " && cat /tmp/wiki_corpus.txt | ./bench/tokenize_corpus /tmp/corpus_tokens.bin
//
// Output format: [num_seqs: i32][seq_lens: i32 x num_seqs][all_token_ids: i32 x sum(lens)]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "blackwell/bpe_tokenizer.h"

int main(int argc, char** argv) {
    const char* out_path = argc >= 2 ? argv[1] : "/tmp/corpus_tokens.bin";
    
    blackwell::BpeTokenizer tokenizer;
    if (tokenizer.load("tokenizer_data.bin") != 0) {
        fprintf(stderr, "FAIL: no tokenizer_data.bin\n");
        return 1;
    }
    
    std::vector<std::vector<uint32_t>> all_seqs;
    char line[1024];
    
    while (fgets(line, sizeof(line), stdin)) {
        // Strip newline
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') line[len-1] = '\0';
        if (len > 0 && line[len-1] == '\r') line[len-1] = '\0';
        if (strlen(line) == 0) continue;
        
        auto ids = tokenizer.encode(line);
        if (ids.size() > 0) {
            all_seqs.push_back(ids);
        }
    }
    
    int n_seqs = (int)all_seqs.size();
    printf("Tokenized %d sequences\n", n_seqs);
    
    // Write format: [n_seqs: i32][lens: i32 x n_seqs][token_ids: flattened]
    FILE* f = fopen(out_path, "wb");
    if (!f) { fprintf(stderr, "FAIL: can't write %s\n", out_path); return 1; }
    
    fwrite(&n_seqs, sizeof(int), 1, f);
    for (const auto& seq : all_seqs) {
        int len = (int)seq.size();
        fwrite(&len, sizeof(int), 1, f);
    }
    for (const auto& seq : all_seqs) {
        fwrite(seq.data(), sizeof(int), seq.size(), f);
    }
    fclose(f);
    
    // Stats
    size_t total_tokens = 0;
    for (const auto& seq : all_seqs) total_tokens += seq.size();
    printf("Total tokens: %zu\n", total_tokens);
    printf("Avg len: %.1f\n", (double)total_tokens / n_seqs);
    
    return 0;
}
