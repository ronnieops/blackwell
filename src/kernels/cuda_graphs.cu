// src/kernels/cuda_graphs.cu — CUDA Graphs for decode launch overhead reduction
#include <cuda_runtime.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace blackwell {
namespace kernels {

// TODO(#8): Capture decode graph for token-by-token inference.
//   CUDA 12.8 adds conditional execution in CUDA Graphs — useful for
//   dynamic sequence lengths without re-capturing.
//   Steps: cudaGraphCreate → cudaStreamBeginCapture → launch decode kernels
//          → cudaStreamEndCapture → cudaGraphInstantiate
//   On each token: cudaGraphExecKernelNodeSetParams (update KV cache pointers)
//                  → cudaGraphLaunch
//   Key: graph capture captures the entire decode kernel sequence (GEMV + attention + norm).

cudaError_t capture_decode_graph(
    void** graph_out, void** node_out, void* graph_exec_out,
    float* d_temp_storage, size_t temp_storage_bytes,
    cudaStream_t stream) {
    (void)graph_out; (void)node_out; (void)graph_exec_out;
    (void)d_temp_storage; (void)temp_storage_bytes; (void)stream;
    return cudaErrorNotReady;
}

cudaError_t launch_decode_graph(void* graph_exec, cudaStream_t stream) {
    (void)graph_exec; (void)stream;
    return cudaErrorNotReady;
}

cudaError_t destroy_decode_graph(void* graph_exec, void* graph) {
    (void)graph_exec; (void)graph;
    return cudaErrorNotReady;
}

} // namespace kernels
} // namespace blackwell
