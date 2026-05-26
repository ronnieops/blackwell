// tests/test_memory.cu — Memory packing / coalescing tests
#include <gtest/gtest.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace {

using namespace blackwell;

TEST(MemoryTest, SharedMemLimitRespect) {
    // 99 KB per block.  Tile 128×128×2 bytes FP16 = 32,768 bytes — fits.
    constexpr int tile_fp16_bytes = 128 * 128 * 2;
    EXPECT_LT(tile_fp16_bytes, kMaxSharedMemBytesPerBlock);
}

TEST(MemoryTest, DISABLED_PackFP4) {
    // TODO(#3): pack_fp4 test — roundtrip FP32 → FP4 → FP32.
    // std::vector<float> in(1024, 1.0f);
    // std::vector<__nv_fp4_e2m1> packed(512);
    // std::vector<float> scale(64);
    // pack_fp4(packed.data(), in.data(), scale.data(), in.size(), stream);
    // unpack_fp4(out.data(), packed.data(), scale.data(), out.size(), stream);
    // EXPECT_NEAR(out[0], 1.0f, 1e-2);
    GTEST_SKIP() << "FP4 pack not yet implemented";
}

TEST(MemoryTest, DISABLED_CoalescedCopy) {
    // TODO(#3): coalesced_copy — verify alignment and throughput.
    GTEST_SKIP() << "Coalesced copy not yet implemented";
}

} // namespace
