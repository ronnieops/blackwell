// src/kernels/cuda_graphs.cu — CUDA Graphs for decode step (stub — capture done in benchmark)
#include <cuda_runtime.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace blackwell {
namespace kernels {

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
