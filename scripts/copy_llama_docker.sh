#!/bin/bash
# scripts/copy_llama_docker.sh — Bundle llama.cpp server + model for Docker build
#
# Copies the llama-server binary, its shared libraries, and the GGUF model
# into a local directory that Docker can COPY (since COPY cannot follow
# symlinks or access paths outside the build context).
#
# Run this before docker build:
#   ./scripts/copy_llama_docker.sh
#   docker build -t llama-server -f Dockerfile.llama .
#
# Source paths (host):
#   /mnt/data/ai/llama.cpp/build-cuda13.2-opt/bin/llama-server
#   /mnt/data/ai/llama.cpp/build-cuda13.2-opt/bin/lib*.so*
#   /mnt/data/ai/hf/qwen3-1.7b-base/model-q4_k_m.gguf

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DST="$PROJECT_DIR/llama-docker"

echo "Copying llama.cpp server + model to $DST..."
mkdir -p "$DST"

# llama-server binary
cp /mnt/data/ai/llama.cpp/build-cuda13.2-opt/bin/llama-server "$DST/"
echo "  llama-server"

# Shared libraries
for lib in /mnt/data/ai/llama.cpp/build-cuda13.2-opt/bin/lib*.so*; do
  cp "$lib" "$DST/"
done
echo "  shared libraries ($(ls "$DST"/*.so* 2>/dev/null | wc -l) files)"

# NCCL (needed by CUDA runtime in container)
for lib in libnccl.so.2 libnccl.so; do
  src=$(find /usr/lib/x86_64-linux-gnu -name "$lib" 2>/dev/null | head -1)
  if [ -n "$src" ]; then
    cp "$src" "$DST/"
    for link in $(find /usr/lib/x86_64-linux-gnu -name "${lib%.so}*" 2>/dev/null); do
      [ -f "$link" ] && cp "$link" "$DST/" 2>/dev/null ||:
    done 2>/dev/null
    echo "  $lib"
  fi
done
for lib in libcudart.so.13 libcublas.so.13 libcublasLt.so.13; do
  src=$(find /usr/local/cuda-13.3/targets/x86_64-linux/lib -name "$lib" 2>/dev/null | head -1)
  if [ -n "$src" ]; then
    cp "$src" "$DST/"
    # Also copy symlinks
    for link in $(find /usr/local/cuda-13.3/targets/x86_64-linux/lib -name "${lib%.so*}*" 2>/dev/null); do
      [ -f "$link" ] && cp "$link" "$DST/" ||:
    done
    echo "  $lib"
  fi
done

# GGUF model (1.03 GB)
cp /mnt/data/ai/hf/qwen3-1.7b-base/model-q4_k_m.gguf "$DST/model.gguf"
echo "  model.gguf ($(du -h "$DST/model.gguf" | cut -f1))"

echo ""
echo "Done. Build with:"
echo "  docker build -t llama-server -f Dockerfile.llama $PROJECT_DIR"
echo ""
echo "Or bench directly:"
echo "  docker run --gpus all -p 8082:8082 llama-server"