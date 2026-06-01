# Dockerfile — Blackwell INT8 Inference Server
#
# Build:
#   docker build -t blackwell-inference .
#
# Run (with CUDA GPU):
#   docker run --gpus all -v /mnt/data/ai/hf:/models \
#     -p 8080:8080 blackwell-inference
#
# Test:
#   curl -X POST http://localhost:8080/generate \
#     -H 'Content-Type: application/json' \
#     -d '{"prompt":"The capital of France is","max_tokens":30}'

FROM nvidia/cuda:13.3-runtime-ubuntu24.04 AS runtime

LABEL description="Blackwell INT8 Inference — RTX 5060 Ti"
LABEL version="0.1.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy library and inference binary
COPY build/libblackwell_kernels.a /app/lib/
COPY bench/text_generate /app/bin/
COPY include/ /app/include/
COPY weights_int8_bf16 /app/weights_int8_bf16

# Copy Python API server
COPY server/server.py /app/server.py

# Install Python deps
RUN pip3 install --no-cache-dir flask gunicorn

EXPOSE 8080

ENV CUDA_VISIBLE_DEVICES=0
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/app/lib

CMD ["python3", "/app/server.py"]