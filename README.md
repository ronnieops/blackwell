# Blackwell Inference — Custom CUDA Kernels for RTX 5060 Ti

Target: **RTX 5060 Ti 16 GB** (compute capability 12.0 / sm_120)

## Performance

| Method | t/s | vs llama.cpp b9442 |
|--------|-----|-------------------|
| llama.cpp Q4_K_M | 292.52 | 100% |
| Blackwell INT8 batched attn M=8 + CUDA Graph | **326.8** | **112%** ✅ |

**Key optimizations**:
- INT8 WMMA FAST GEMM: 4.3-5.0K GFLOPS
- Block GEMV unrolling: +9-45% speedup
- Batched attention: +10% over serial
- CUDA Graph: +3% over batched attention

## Key tuning priorities (per doc.md)

1. **FP4 block-scaled GEMM** — weight quantized to FP4 E2M1, scale factors per block
2. **Memory coalescing** — Q/K/V/weights laid out for 128-bit bus efficiency
3. **Shared-memory tiling** — respect 99 KB/block limit, 48 warps/SM occupancy
4. **Fused epilogues** — RMSNorm, RoPE, dequant, bias fused into matmul
5. **KV-cache decode path** — compact layout, bandwidth-first for long-context
6. **Separate prefill kernels** — GEMM/attention-tile heavy vs decode GEMV
7. **CUDA Graphs** — reduce per-token decode launch overhead
8. **Nsight Compute profiling** — validate on actual hardware

## Project structure

```
blackwell/
├── CMakeLists.txt
├── pyproject.toml
├── README.md
├── include/
│   └── blackwell/         # public header-only config + device math
├── src/
│   └── kernels/           # .cu implementations
├── tests/
│   └── test_*.cu          # GoogleTest unit tests
└── python/
    └── blackwell_pybind.cpp  # pybind11 bindings + smoke test
```

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build
```

## Compiler requirements

- **CUDA 12.8+** (12.8 adds SM_120 support)
- **nvcc** with compute_120 + sm_120 targets
- **GCC/Clang** for host code
- **pybind11** (for Python bindings)

## Constraints (RTX 5060 Ti / CC 12.0)

- 128-bit GDDR7 memory interface → bandwidth is the bottleneck, not just tensor throughput
- 128 KB shared memory/SM total, **99 KB max per thread block**
- **48 concurrent warps/SM**, 64K 32-bit registers/SM, 255 registers/thread max
- SM_120 has 5th-gen Tensor Cores → FP4 capability present
