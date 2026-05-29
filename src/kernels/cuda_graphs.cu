// src/kernels/cuda_graphs.cu — CUDA Graph management API
//
// Wraps cudaGraph/cudaGraphExec lifecycle for decode step.
// The capture logic lives in benchmark code (inference_server.cu) because
// it needs access to model-specific buffers. These API functions provide
// launch and destroy for graphs created externally.

#include <cuda_runtime.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"

namespace blackwell {
namespace kernels {

// Capture is done in benchmark code. This stub exists for API completeness.
// Returns not-ready to signal caller should use direct capture.
cudaError_t capture_decode_graph(
    void** graph_out, void** node_out, void* graph_exec_out,
    float* d_temp_storage, size_t temp_storage_bytes,
    cudaStream_t stream) {
    (void)graph_out; (void)node_out; (void)graph_exec_out;
    (void)d_temp_storage; (void)temp_storage_bytes; (void)stream;
    return cudaErrorNotReady;
}

// Launch a previously captured and instantiated CUDA Graph.
// graph_exec must be a valid cudaGraphExec_t cast to void*.
cudaError_t launch_decode_graph(void* graph_exec, cudaStream_t stream) {
    if (!graph_exec) return cudaErrorInvalidValue;
    cudaError_t e = cudaGraphLaunch(static_cast<cudaGraphExec_t>(graph_exec), stream);
    if (e != cudaSuccess) return e;
    return cudaPeekAtLastError();
}

// Destroy a CUDA Graph and its executable.
cudaError_t destroy_decode_graph(void* graph_exec, void* graph) {
    cudaError_t e1 = cudaSuccess, e2 = cudaSuccess;
    if (graph_exec) {
        e1 = cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph_exec));
    }
    if (graph) {
        e2 = cudaGraphDestroy(static_cast<cudaGraph_t>(graph));
    }
    return (e1 != cudaSuccess) ? e1 : e2;
}

} // namespace kernels
} // namespace blackwell
