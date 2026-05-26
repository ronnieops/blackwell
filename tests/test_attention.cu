// tests/test_attention.cu — Attention / RoPE / KV-cache tests
#include <gtest/gtest.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace {

using namespace blackwell;

TEST(AttentionTest, PrefillAttentionStubCheck) {
    // TODO(#7): attention_fp4 test — Q/K/V known values, verify softmax math.
    // Use small batch=1, seq_len=4, num_heads=2, head_dim=8.
    GTEST_SKIP() << "Prefill attention not yet implemented";
}

TEST(AttentionTest, DISABLED_KVCacheUpdate // TODO(#6): load_kv_cache_qkgv + update_kv_cache roundtrip test.
std::vector<float> k_new(num_heads * head_dim, 0.5f);
std::vector<float> v_new(num_heads * head_dim, 0.25f);
update_kv_cache(k_cache, v_cache, k_new.data(), v_new.data(),
                 batch_idx, seq_pos, num_heads, head_dim, max_seq_len, stream);
std::vector<float> k_loaded(num_heads * head_dim);
std::vector<float> v_loaded(num_heads * head_dim);
load_kv_cache_qkgv(nullptr, k_loaded.data(), v_loaded.data(),
                   k_cache, v_cache, batch_idx, seq_pos, num_heads,
                   head_dim, max_seq_len, stream);
EXPECT_NEAR(k_loaded[0], 0.5f, 1e-3);
EXPECT_NEAR(v_loaded[0], 0.25f, 1e-3);
GTEST_SKIP() << "KV cache not yet implemented";
}

TEST(RoPETest, RoPEStubCheck) {
    // TODO(#5): fused_rope test — cos/sin precomputed, apply rotation.
    GTEST_SKIP() << "RoPE not yet implemented";
}

} // namespace
