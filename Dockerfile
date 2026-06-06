# Dockerfile — Blackwell INT8 NOFP4 Inference Server
#
# C++ only: http_subprocess + inference_server binary
# Correct model: per-layer RMSNorm, Q/K head norms, RoPE
#
# Build:
#   docker build -t blackwell-server .
#
# Run (requires nvidia-container-toolkit):
#   docker run --gpus all -p 8080:8080 blackwell-server
#
# Test:
#   curl -X POST http://localhost:8080/v1/completions \
#     -H 'Content-Type: application/json' \
#     -d '{"prompt":"The capital of France is","max_tokens":10}'

FROM ubuntu:24.04

LABEL description="Blackwell INT8 NOFP4 — RTX 5060 Ti, multi-model"
LABEL version="0.6.0"

RUN apt-get update \
    -o Acquire::Check-Valid-Until=false \
    -o Acquire::Check-Date=false \
    -o Acquire::AllowInsecureRepositories=true \
    -o Acquire::AllowDowngradeToInsecureRepositories=true && \
    apt-get install -y --allow-unauthenticated --no-install-recommends \
    ca-certificates curl && rm -rf /var/lib/apt/lists/*

# Copy CUDA runtime library from host
# nvidia-container-toolkit provides driver but NOT cudart
COPY cuda-libs/libcudart.so.13* /usr/local/lib/
RUN ln -sf /usr/local/lib/libcudart.so.13 /usr/local/lib/libcudart.so

WORKDIR /app

COPY tokenizer_data.bin /app/
COPY weights_int8_bf16 /app/weights_int8_bf16
COPY weights_int8_qwen3_8b /app/weights_int8_qwen3_8b
COPY server/http_subprocess /app/bin/
COPY server/inference_server /app/bin/

RUN ldconfig || true

EXPOSE 8080

ENV CUDA_VISIBLE_DEVICES=0

# Default: 1.7B model. Pass "8b" as last arg for 8B.
CMD ["/app/bin/http_subprocess", "8080", "1.7b"]
