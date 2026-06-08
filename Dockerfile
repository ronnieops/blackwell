# Dockerfile — Blackwell INT8 Inference Server
# Multi-model: 1.7B / 8B (Transformer) + 9B (GatedDeltaNet)
# Weights mounted as volume at runtime (22 GB on disk, too large for image)
#
# Build:
#   docker build -t blackwell-server .
#
# Run (requires nvidia-container-toolkit):
#   # Mount one weight dir (smallest image)
#   docker run --gpus all -p 8080:8080 \
#     -v /mnt/data/dev/projects/blackwell/weights_int8_bf16:/app/weights_int8_bf16 \
#     blackwell-server
#
#   # Mount all weights (multi-model)
#   docker run --gpus all -p 8080:8080 \
#     -v /mnt/data/dev/projects/blackwell/weights_int8_bf16:/app/weights_int8_bf16 \
#     -v /mnt/data/dev/projects/blackwell/weights_int8_qwen3_8b:/app/weights_int8_qwen3_8b \
#     -v /mnt/data/dev/projects/blackwell/weights_int8_qwen35_9b:/app/weights_int8_qwen35_9b \
#     blackwell-server 8080 9b
#
# Test:
#   curl http://localhost:8080/v1/models
#   curl -X POST http://localhost:8080/v1/completions \
#     -H 'Content-Type: application/json' \
#     -d '{"prompt":"Hello","max_tokens":10}'

FROM ubuntu:24.04

LABEL description="Blackwell INT8 — RTX 5060 Ti, multi-model (1.7B/8B/9B GDN)"
LABEL version="0.8.1"

# v0.8.1 features:
# - Repetition penalty (repetition_penalty param, 1.0-2.0)
# - Batched QKV optimization (reduced kernel launches for M>1)
# - Mixed-precision auto-detection (.fp16 files per layer)
# - Critical bug fixes (seq_pos sync, empty prompt, prefill cache)
# - 1.7B: ~23 t/s, 8B: ~3.7 t/s

RUN apt-get update \
    -o Acquire::Check-Valid-Until=false \
    -o Acquire::Check-Date=false \
    -o Acquire::AllowInsecureRepositories=true \
    -o Acquire::AllowDowngradeToInsecureRepositories=true && \
    apt-get install -y --allow-unauthenticated --no-install-recommends \
    ca-certificates curl && rm -rf /var/lib/apt/lists/*

# Copy CUDA runtime library from host
COPY cuda-libs/libcudart.so.13* /usr/local/lib/
RUN ln -sf /usr/local/lib/libcudart.so.13 /usr/local/lib/libcudart.so

WORKDIR /app

# Binaries + tokenizer data (weights mounted from host at runtime)
COPY server/http_subprocess server/inference_server server/inference_server_9b /app/server/
COPY tokenizer_data.bin tokenizer_data_9b.bin /app/

RUN ldconfig || true

EXPOSE 8080
ENV CUDA_VISIBLE_DEVICES=0

ENTRYPOINT ["/app/server/http_subprocess"]
# Default: port 8080, model 1.7B. Override by passing args: <port> <model>
#   docker run ... blackwell-server 8081 9b
CMD ["8080", "1.7b"]
