// src/kernels/cuda_graphs.cu — CUDA Graphs for decode step
//
// Captures full decode step (all layers) as a single CUDA Graph.
// On each invocation, updates seq_pos and launches graph.
//
// Kernel sequence per layer (captured):
//   1. gemv_fp4 Q
//   2. gemv_fp4 K
//   3. gemv_fp4 V
//   4. update_kv_cache
//   5. attention_decode
//   6. pack_fp4 (attn_out)
//   7. gemv_fp4 O
//   8. fused_rmsnorm
//   9. pack_fp4 (x)
//
// The graph is captured once, then replayed with updated parameters.
// seq_pos is updated in the graph via cudaGraphExecKernelNodeSetParams.

#include <cuda_runtime.h>
#include "blackwell/kernels.h"
#include "blackwell/config.h"
#include <vector>

namespace blackwell {
namespace kernels {

// Describes one captured decode graph instance
struct DecodeGraph {
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    int num_nodes;               // total kernel nodes
    std::vector<cudaGraphNode_t> kernel_nodes;  // all kernel nodes (for param updates)
    std::vector<int> seq_pos_node_indices;       // node indices that take seq_pos
    int num_layers;
    bool captured;
};

// Single global decode graph instance
static DecodeGraph g_decode_graph = {};

// ===========================================================================
// capture_decode_graph: Capture all kernels for one decode step
//
// Parameters:
//   All device pointers must be valid. The graph is captured by launching
//   each kernel with cudaStreamBeginCapture/EndCapture.
//   After capture, kernel node params can be updated per-token.
// ===========================================================================
cudaError_t capture_decode_graph(
    void** graph_out, void** node_out, void* graph_exec_out,
    float* d_temp_storage, size_t temp_storage_bytes,
    cudaStream_t stream) {

    (void)d_temp_storage; (void)temp_storage_bytes;
    (void)graph_out; (void)node_out; (void)graph_exec_out;

    if (g_decode_graph.captured) {
        // Already captured, return existing
        if (graph_out) *graph_out = &g_decode_graph.graph;
        if (graph_exec_out) *graph_exec_out = &g_decode_graph.graph_exec;
        return cudaSuccess;
    }

    // We capture by calling the actual decode kernels in a stream
    // capture session. But this requires all device pointers to be known
    // and valid. For now, return not-ready — the decode benchmark uses
    // direct kernel launches.
    //
    // True graph capture requires the outer caller (benchmark or runtime)
    // to own the pointers and call cudaStreamBeginCapture itself.
    return cudaErrorNotReady;
}

// ===========================================================================
// launch_decode_graph: Execute captured decode graph
// ===========================================================================
cudaError_t launch_decode_graph(void* graph_exec, cudaStream_t stream) {
    if (!graph_exec) return cudaErrorInvalidValue;
    cudaGraphExec_t exec = *static_cast<cudaGraphExec_t*>(graph_exec);
    return cudaGraphLaunch(exec, stream);
}

// ===========================================================================
// destroy_decode_graph: Free captured graph resources
// ===========================================================================
cudaError_t destroy_decode_graph(void* graph_exec, void* graph) {
    if (graph_exec) {
        cudaGraphExecDestroy(*static_cast<cudaGraphExec_t*>(graph_exec));
    }
    if (graph) {
        cudaGraphDestroy(*static_cast<cudaGraph_t*>(graph));
    }
    g_decode_graph = {};
    return cudaSuccess;
}

} // namespace kernels
} // namespace blackwell
