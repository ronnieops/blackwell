// tests/test_norm.cu — Norm / activation tests
#include <gtest/gtest.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"
#include <vector>

namespace {

using namespace blackwell;

TEST(NormTest, RMSNormStubCheck) {
    // TODO(#5): real RMSNorm test once implemented
    // Verify: rmsnorm(x, w, N) = x * w / sqrt(sum(x²)/N + eps)
    // Pass known x=[1,2,3], w=[1,1,1], check output.
    GTEST_SKIP() << "RMSNorm not yet implemented";
}

TEST(NormTest, SwiGLUStubCheck) {
    // TODO(#5): SwiGLU test: out = silu(gate) * up
    GTEST_SKIP() << "SwiGLU not yet implemented";
}

} // namespace
