cmake_minimum_required(VERSION 3.26)
project(blackwell_inference VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 5060 Ti = compute_120, sm_120
set(CMAKE_CUDA_ARCHITECTURES 120)

# ---------------------------------------------------------------------------
# CUDA 12.8 for Blackwell (SM_120) support
# ---------------------------------------------------------------------------
# CMake always runs a host-compiler compile test to determine compiler ID.
# CUDA 12.8 only supports GCC <= 14, but the system has GCC-13.3 which fails
# the check.  We must use --allow-unsupported-compiler for the CMake ID test,
# and set a supported host compiler (g++-12) for actual builds.
# NOTE: CMakeDetermineCUDACompiler.cmake invokes nvcc -cccdin --compiler-bindir
# <host> which is what we need to override.  Using CMAKE_CUDA_HOST_COMPILER
# here does NOT help the CMake ID step (that happens before project() reads it).
# Instead, add the override in CMAKE_CUDA_FLAGS so it applies to the ID probe.
# ---------------------------------------------------------------------------
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")

# Force CUDA 12.8 nvcc before ANY CMake detection runs.
# CMakeDetermineCUDACompiler checks CUDACXX env var first, so this must be set
# before project().  CMake also needs -allow-unsupported-compiler during its
# CUDA compiler-ID probe (which uses the system g++-13; GCC-13 is technically
# supported by CUDA 12.8 since the check is "< 14").
set(ENV{CUDACXX} "/usr/local/cuda-12.8/bin/nvcc")

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
    src/kernels/quantize.cu
    src/kernels/memory.cu
    src/kernels/norm.cu
    src/kernels/rope.cu
    src/kernels/attention.cu
    src/kernels/prefill.cu
    src/kernels/decode.cu
    src/kernels/cuda_graphs.cu
)

# ---------------------------------------------------------------------------
# Shared include (public)
# ---------------------------------------------------------------------------
add_library(blackwell_config INTERFACE)
target_include_directories(blackwell_config INTERFACE include)
target_compile_options(blackwell_config INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120,code=sm_120>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120,code=compute_120>
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
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120,code=sm_120>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120,code=compute_120>
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
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120,code=sm_120>
        $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120,code=compute_120>
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
