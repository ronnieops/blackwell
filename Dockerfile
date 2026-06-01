# Dockerfile — Blackwell INT8 Inference Server
#
# text_generate statically links CUDA 13.3 (cudart + curand) via the .a archive.
# No CUDA runtime needed at container level — only nvidia-container-toolkit
# providing the kernel-mode driver (libcuda.so).
#
# Build:
#   docker build -t blackwell-inference .
#
# Run (requires nvidia-container-toolkit on host):
#   docker run --gpus all -p 8080:8080 blackwell-inference
#
# Test:
#   curl -X POST http://localhost:8080/generate \
#     -H 'Content-Type: application/json' \
#     -d '{"prompt":"The capital of France is","max_tokens":30}'

FROM ubuntu:24.04

LABEL description="Blackwell INT8 Inference — RTX 5060 Ti"
LABEL version="0.1.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY bench/text_generate /app/bin/
COPY weights_int8_bf16 /app/weights_int8_bf16
COPY server/server.py /app/server.py

RUN pip3 install --no-cache-dir flask gunicorn

EXPOSE 8080

ENV CUDA_VISIBLE_DEVICES=0

CMD ["python3", "/app/server.py"]