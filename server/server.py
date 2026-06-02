#!/usr/bin/env python3
"""server.py — Blackwell INT8 inference HTTP API server with continuous batching.

Provides /generate endpoint for text generation.
Spawns a persistent C++ inference server process that handles M=8 batched decode.

Run:
    python3 server.py

Test:
    curl -X POST http://localhost:8080/generate \
        -H 'Content-Type: application/json' \
        -d '{"prompt":"The capital of France is","max_tokens":30}'
"""

import subprocess
import json
import os
import sys
import threading
import queue
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

BIN = os.path.join(os.path.dirname(__file__), "inference_server")

# ── C++ inference server process (persistent, batched) ────────────────────
class BatchServer:
    """Manages a persistent C++ inference server process.
    Accumulates requests and dispatches them in batches up to M=8."""

    def __init__(self, bin_path, timeout=30):
        self.bin_path = bin_path
        self.timeout = timeout
        self.proc = None
        self.lock = threading.Lock()
        self.pending = {}  # request_id → queue entry
        self.req_counter = 0
        self._start()

    def _start(self):
        if not os.path.exists(self.bin_path):
            alt = os.path.join(os.path.dirname(__file__), "..", "server", "inference_server")
            if os.path.exists(alt):
                self.bin_path = alt
            else:
                raise RuntimeError(f"inference_server binary not found at {self.bin_path}")
        self.proc = subprocess.Popen(
            [self.bin_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    def generate(self, token_ids, max_tokens, temperature, top_k):
        """Send a single request to the batch server. Returns generated token IDs."""
        req_id = self.req_counter
        self.req_counter += 1

        request = {
            "prompts": [token_ids],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k
        }

        with self.lock:
            self.proc.stdin.write(json.dumps(request) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()

        try:
            result = json.loads(line)
            tokens = result.get("tokens", [[]])[0]
            return tokens
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            raise RuntimeError(f"batch server error: {e}, response: {line[:200]}")

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/generate":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            req = json.loads(body)
        except:
            self.send_json({"error": "invalid json"}, 400)
            return

        prompt = req.get("prompt", "Once upon a time")
        max_tokens = req.get("max_tokens", 30)
        temperature = req.get("temperature", 1.0)
        top_k = req.get("top_k", 40)

        try:
            # Tokenize prompt using the BPE tokenizer
            prompt_tokens = self.server.tokenizer.encode(prompt)
            # Generate via batch server
            tokens = self.server.batch_server.generate(
                prompt_tokens, max_tokens, temperature, top_k)
            # Decode tokens to text
            output = ""
            for tid in tokens:
                txt = self.server.tokenizer.decode(tid)
                output += txt

            self.send_json({"generated": output, "prompt": prompt})
        except Exception as e:
            self.send_json({"error": str(e)}, 500)

    def do_GET(self):
        if self.path == "/health":
            self.send_json({"status": "ok", "model": "Qwen3-1.7B INT8"})
        elif self.path == "/":
            self.send_json({
                "name": "Blackwell INT8 Inference",
                "model": "Qwen3-1.7B",
                "quant": "INT8",
                "version": "0.2.0",
                "endpoints": {
                    "GET /health": "health check",
                    "GET /": "this info",
                    "POST /generate": "generate text (single seq)"
                }
            })
        else:
            self.send_error(404)

    def send_json(self, data, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())


class Server:
    def __init__(self):
        # Load tokenizer
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from blackwell.bpe_tokenizer import BpeTokenizer
        tokenizer_path = os.path.join(os.path.dirname(__file__), "..", "tokenizer_data.bin")
        self.tokenizer = BpeTokenizer()
        self.tokenizer.load(tokenizer_path)

        # Start batch server
        bin_path = os.path.join(os.path.dirname(__file__), "inference_server")
        self.batch_server = BatchServer(bin_path)

    def run(self, port=8080):
        server = HTTPServer(("0.0.0.0", port), Handler)
        Handler.server = self
        print(f"Blackwell INT8 Inference Server v0.2.0")
        print(f"Listening on http://0.0.0.0:{port}")
        print(f"Endpoints: GET /health, GET /, POST /generate")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
            server.shutdown()
        finally:
            self.batch_server.close()

if __name__ == "__main__":
    s = Server()
    s.run()