cmake_minimum_required(VERSION 3.26)

# ---------------------------------------------------------------------------
# CUDA 13.3 for Blackwell (SM_120a) support — must be BEFORE project()
# ---------------------------------------------------------------------------
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.3/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 120a)

# Ensure nvcc finds its own ptxas, not the system one
set(ENV{PATH} "/usr/local/cuda-13.3/bin:$ENV{PATH}")

project(blackwell_inference VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Force CUDA 12.8 nvcc before ANY CMake detection runs.
# CMakeDetermineCUDACompiler checks CUDACXX env var first, so this must be set
# before project().  CMake also needs -allow-unsupported-compiler during its
# CUDA compiler-ID probe (which uses the system g++-13; GCC-13 is technically
# supported by CUDA 12.8 since the check is "< 14").
set(ENV{CUDACXX} "/usr/local/cuda-13.3/bin/nvcc")

# GCC-13.3 is technically supported by CUDA 12.8 (check is < 14).  This means
# CMAKE_CUDA_FLAGS override via -cccdin is not strictly needed here, but
# keeping -allow-unsupported-compiler as belt-and-suspenders.
list(APPEND CMAKE_CUDA_FLAGS -allow-unsupported-compiler)

# ---------------------------------------------------------------------------
# Host C++ compiler for actual compilation (not CMake ID probes)
# ---------------------------------------------------------------------------
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12 CACHE PATH "Host C++ compiler for nvcc")

# Find CUDA
find_package(CUDAToolkit REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

# ---------------------------------------------------------------------------
# Kernel sources
# ---------------------------------------------------------------------------
set(KERNEL_SOURCES
    src/kernels/quantize.cu
    src/kernels/memory.cu
    src/kernels/norm.cu
    src/kernels/apply_geglu.cu
    src/kernels/rope.cu
    src/kernels/attention.cu
    src/kernels/decode.cu
    src/kernels/cuda_graphs.cu
    src/kernels/gemv_int8.cu
    src/kernels/gemv_fp32.cu
    src/kernels/gemv_int8_fp16sc.cu
    src/kernels/sample_gpu.cu
    src/kernels/fused_swiglu_quant.cu
    src/kernels/fused_int4_ops.cu
    src/kernels/fused_gate_up_int4.cu
    src/kernels/fused_qkv_int4.cu
)

# ---------------------------------------------------------------------------
# Shared include (public)
# ---------------------------------------------------------------------------
add_library(blackwell_config INTERFACE)
target_include_directories(blackwell_config INTERFACE include)
target_compile_options(blackwell_config INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120a,code=sm_120a>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120a,code=compute_120a>
)

# ---------------------------------------------------------------------------
# CUDA library
# ---------------------------------------------------------------------------
add_library(blackwell_kernels STATIC ${KERNEL_SOURCES})
target_link_libraries(blackwell_kernels PRIVATE
    CUDA::cudart
    CUDA::curand
    blackwell_config
)
target_compile_options(blackwell_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120a,code=sm_120a>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120a,code=compute_120a>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v,-O3>
)

# ---------------------------------------------------------------------------
# Python bindings (pybind11) — optional, quiet if not found
# ---------------------------------------------------------------------------
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 CONFIG)
    if(pybind11_FOUND)
        pybind11_add_module(blackwell_pybind
            python/blackwell_pybind.cpp
        )
        target_sources(blackwell_pybind PRIVATE ${KERNEL_SOURCES})
        target_include_directories(blackwell_pybind PRIVATE include)
        target_link_libraries(blackwell_pybind PRIVATE CUDA::cudart)
    else()
        message(STATUS "pybind11 not found — Python bindings disabled (pip install pybind11).")
    endif()
endif()

# ---------------------------------------------------------------------------
# better-inference: GGUF bridge tools (no CUDA dependency)
# ---------------------------------------------------------------------------
add_executable(gguf_convert
    better-inference/gguf_convert.cpp
)
target_include_directories(gguf_convert PRIVATE better-inference)
target_compile_options(gguf_convert PRIVATE -O2 -std=c++17)
target_link_libraries(gguf_convert PRIVATE stdc++fs)
set_target_properties(gguf_convert PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../better-inference")

add_executable(gguf_test
    better-inference/gguf_test.cpp
)
target_include_directories(gguf_test PRIVATE better-inference)
target_compile_options(gguf_test PRIVATE -O2 -std=c++17)
set_target_properties(gguf_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../better-inference")

# ---------------------------------------------------------------------------
# Benchmarks (CUDA, link against kernel library)
# ---------------------------------------------------------------------------
function(add_blackwell_bench NAME SOURCE)
    add_executable(${NAME} ${SOURCE})
    target_link_libraries(${NAME} PRIVATE
        blackwell_kernels
        CUDA::cudart
        CUDA::curand
    )
    target_compile_options(${NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120a,code=sm_120a>
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120a,code=compute_120a>
    )
    target_include_directories(${NAME} PRIVATE include)
    set_target_properties(${NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bench")
endfunction()

# Active INT4 benches (Qwen3-8B production path)
add_blackwell_bench(text_generate_int4_batched bench/text_generate_int4_batched.cu)
add_blackwell_bench(bench_ppl_int4_8b bench/bench_ppl_int4_8b.cu)
add_blackwell_bench(decode_int4_cgraph_8b bench/decode_int4_cgraph_8b.cu)
add_blackwell_bench(bench_gemv_int4 bench/bench_gemv_int4.cu)
add_blackwell_bench(bench_gemv_2warp bench/bench_gemv_2warp.cu)
add_blackwell_bench(test_fused_int4 bench/test_fused_int4.cu)
add_blackwell_bench(bench_kld_int4_8b bench/bench_kld_int4_8b.cu)
add_blackwell_bench(text_generate_llama32_3b bench/text_generate_llama32_3b.cu)
add_blackwell_bench(text_generate_llama32_1b bench/text_generate_llama32_1b.cu)
add_blackwell_bench(text_generate_qwen25_0.5b bench/text_generate_qwen25_0.5b.cu)
add_blackwell_bench(text_generate_gemma4_12b_qat bench/text_generate_gemma4_12b_qat.cu)
add_blackwell_bench(text_generate_int4_qwen3_14b bench/text_generate_int4_qwen3_14b.cu)
add_blackwell_bench(bench_ppl_int4_14b bench/bench_ppl_int4_14b.cu)
