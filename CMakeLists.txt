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
    src/kernels/gemm.cu
    src/kernels/gemm_int8_mma.cu
    src/kernels/gemm_int8_wmma.cu
    src/kernels/gemm_int8_wmma_fast.cu
    src/kernels/quantize.cu
    src/kernels/memory.cu
    src/kernels/norm.cu
    src/kernels/rope.cu
    src/kernels/attention.cu
    src/kernels/prefill.cu
    src/kernels/decode.cu
    src/kernels/cuda_graphs.cu
    src/kernels/fused_decode.cu
    src/kernels/fused_o_norm.cu
    src/kernels/gemv_v2.cu
    src/kernels/gemv_int8.cu
    src/kernels/gemv_bf16.cu
    src/kernels/gemv_fp4_nv.cu
    src/kernels/fused_mlp.cu
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
# Unit tests (GoogleTest) — optional, quiet if not found
# ---------------------------------------------------------------------------
enable_testing()
find_package(GTest CONFIG)
if(GTest_FOUND)
    add_executable(blackwell_tests
        tests/test_gemm.cu
        tests/test_memory.cu
        tests/test_norm.cu
        tests/test_attention.cu
    )
    target_link_libraries(blackwell_tests PRIVATE
        blackwell_kernels
        GTest::gtest
        CUDA::cudart
    )
    target_compile_options(blackwell_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120a,code=sm_120a>
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120a,code=compute_120a>
    )
    target_include_directories(blackwell_tests PRIVATE include)
else()
    message(STATUS "GTest not found — tests disabled (set GModelTest_DIR or install libgtest-dev).")
endif()

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
