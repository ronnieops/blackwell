// python/blackwell_pybind.cpp — pybind11 bindings + smoke test
#include <pybind11/pybind11.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace py = pybind11;
using namespace blackwell;
using namespace blackwell::kernels;

PYBIND11_MODULE(blackwell_pybind, m) {
    m.doc() = "Custom CUDA kernels for LLM inference on RTX 5060 Ti (SM_120)";

    // Config constants
    m.attr("SM_ARCH")         = kSMArchitecture;
    m.attr("FP4_BLOCK_SIZE")  = kFP4BlockSize;
    m.attr("GEMM_TILE_M")     = kGEMMTileM;
    m.attr("GEMM_TILE_N")     = kGEMMTileN;
    m.attr("GEMM_TILE_K")     = kGEMMTileK;
    m.attr("MAX_SHARED_MEM_PER_BLOCK") = kMaxSharedMemBytesPerBlock;
    m.attr("MAX_WARPS_PER_SM") = kMaxWarpsPerSM;

    // Kernel enums
    py::enum_<KernelMode>(m, "KernelMode")
        .value("Prefill", KernelMode::Prefill)
        .value("Decode",  KernelMode::Decode);

    // GEMM
    m.def("gemm_fp4", &gemm_fp4_block_scaled, py::arg("C"), py::arg("A_fp4"),
          py::arg("A_scale"), py::arg("B_fp4"), py::arg("B_scale"),
          py::arg("M"), py::arg("N"), py::arg("K"), py::arg("stream") = 0);
    m.def("gemv_fp4", &gemv_fp4, py::arg("y"), py::arg("x_fp4"),
          py::arg("x_scale"), py::arg("W_fp4"), py::arg("W_scale"),
          py::arg("in_features"), py::arg("out_features"),
          py::arg("stream") = 0);
    m.def("dispatch_matmul", &dispatch_matmul, py::arg("C"), py::arg("A"),
          py::arg("B"), py::arg("A_scale"), py::arg("B_scale"),
          py::arg("M"), py::arg("N"), py::arg("K"), py::arg("mode"),
          py::arg("stream") = 0);

    // Quantization
    m.def("pack_fp4",     &pack_fp4,  "Pack FP32 → FP4 with block scales");
    m.def("unpack_fp4",   &unpack_fp4, "Unpack FP4 → FP32 with block scales");
    m.def("coalesced_copy", &coalesced_copy, "Coalesced FP32 copy");

    // Norms
    m.def("fused_rmsnorm", &fused_rmsnorm, py::arg("out"), py::arg("inp"),
          py::arg("weight"), py::arg("num_elements"), py::arg("eps") = 1e-5f,
          py::arg("stream") = 0);
    m.def("apply_swiglu", &apply_swiglu);
    m.def("fused_rope",   &fused_rope);

    // Attention
    m.def("attention_fp4",  &attention_fp4, "Flash-style attention for prefill");
    m.def("update_kv_cache",    &update_kv_cache,    "Write new K/V to cache");
    m.def("load_kv_cache_qkgv", &load_kv_cache_qkgv, "Load K/V for current position");

    // Prefill
    m.def("run_prefill_layer", &run_prefill_layer);

    // CUDA Graphs
    m.def("capture_decode_graph",   &capture_decode_graph,   py::arg("graph_out"),
          py::arg("node_out"), py::arg("graph_exec_out"),
          py::arg("d_temp_storage"), py::arg("temp_storage_bytes"),
          py::arg("stream") = 0);
    m.def("launch_decode_graph", &launch_decode_graph);
    m.def("destroy_decode_graph", &destroy_decode_graph);

    // Smoke test — verify CUDA device visible
    m.def("smoke_test", []() {
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        return py::dict(
            "device"_a = device,
            "name"_a = prop.name,
            "total_mem"_a = prop.totalGlobalMem,
            "sm"_a = prop.major * 10 + prop.minor
        );
    });
}
