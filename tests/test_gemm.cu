// tests/test_gemm.cu — GEMM / GEMV kernel tests
#include <gtest/gtest.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace {

using namespace blackwell;

TEST(GEMMTest, PlaceholderCompileTest) {
    // Stub: verify kernel linkage, kernel enum, and config constants.
    EXPECT_EQ(kGEMMTileM, 64);
    EXPECT_EQ(kGEMMTileN, 64);
    EXPECT_EQ(kGEMMTileK, 64);
    EXPECT_EQ(kFP4BlockSize, 16);
}

TEST(GEMMTest, DISABLED_FP4GEMMCorrectness) {
    // TODO(#2): replace with real FP4 GEMM correctness test once implemented.
    // Validate against FP32 baseline once MMA is wired up.
    // Set scale=1.0, zero input, expect all-zero output.
    // Expected: |C_fp4 - C_fp32| < 1e-3 per element for identity inputs.
    GTEST_SKIP() << "FP4 GEMM not yet implemented";
}

TEST(GEMVTest, PlaceholderCompileTest) {
    // TODO(#6): real GEMV test once implemented — A=M=1, large N, known weights.
    EXPECT_EQ(kGEMVTileM, 8);
    EXPECT_EQ(kGEMVTileN, 64);
    EXPECT_EQ(kGEMVTileK, 64);
}

TEST(GEMVTest, DISABLED_FP4GEMVCorrectness) {
    // TODO(#6): Decoder-only path: single input vector per call.
    // EXPECT_EQ(in_features % kFP4BlockSize, 0);
    // EXPECT_EQ(out_features % kFP4BlockSize, 0);
    // float y_expected[out_features];
    // gemv_fp4(y, x_fp4, x_scale, W_fp4, W_scale, in_features, out_features, stream);
    // Compare against FP32 gemv result — max absolute error < 5e-3.
    GTEST_SKIP() << "FP4 GEMV not yet implemented";
}

TEST(GEMMTest, DispatchPrefill) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    EXPECT_EQ(dispatch_matmul(nullptr, nullptr, nullptr, nullptr, nullptr,
                              128, 64, 128, KernelMode::Prefill, stream),
              cudaErrorNotReady);
    cudaStreamDestroy(stream);
}

TEST(GEMMTest, DispatchDecode) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    EXPECT_EQ(dispatch_matmul(nullptr, nullptr, nullptr, nullptr, nullptr,
                              1, 1024, 4096, KernelMode::Decode, stream),
              cudaErrorNotReady);
    cudaStreamDestroy(stream);
}

} // namespace
