#!/usr/bin/env python3
"""server.py — Blackwell INT8 inference HTTP API server.

Provides /generate endpoint for text generation.
Calls text_generate binary as subprocess.

Run:
    python3 server.py

Test:
    curl -X POST http://localhost:8080/generate \
        -H 'Content-Type: application/json' \
        -d '{"prompt":"The capital of France is","max_tokens":30}'
"""

import subprocess
import json
import re
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

BIN = "/app/bin/text_generate"
BIN_DEV = os.path.join(os.path.dirname(__file__), "..", "bench", "text_generate")

def find_bin():
    if os.path.exists(BIN):
        return BIN
    if os.path.exists(BIN_DEV):
        return BIN_DEV
    # Search PATH
    for p in os.environ.get("PATH", "").split(":"):
        fp = os.path.join(p, "text_generate")
        if os.path.exists(fp):
            return fp
    return None

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
        temperature = req.get("temperature", 0.8)
        top_k = req.get("top_k", 40)
        
        try:
            output = self.generate(prompt, max_tokens, temperature, top_k)
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
                "version": "0.1.0",
                "endpoints": {
                    "GET /health": "health check",
                    "GET /": "this info",
                    "POST /generate": "generate text"
                }
            })
        else:
            self.send_error(404)
    
    def generate(self, prompt, max_tokens, temperature, top_k):
        bin_path = find_bin()
        if not bin_path:
            raise RuntimeError("text_generate binary not found")
        
        cmd = [bin_path, prompt, str(max_tokens), "-t", str(temperature), "-k", str(top_k)]
        
        # Kill hashcat before generation
        subprocess.run(["killall", "hashcat"], capture_output=True, timeout=5)
        
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=min(120, max_tokens * 5))
        
        if result.returncode != 0:
            raise RuntimeError(f"text_generate failed: {result.stderr[:500]}")
        
        # Parse output: extract text after ── Generating ──
        lines = result.stdout.split("\n")
        in_gen = False
        for line in lines:
            if "── Generating ──" in line:
                in_gen = True
                continue
            if in_gen and line.strip() and "── Stats ──" not in line:
                text = re.sub(r'\s*\[tok#\d+=?\d*\]', '', line).strip()
                if text:
                    return text
        
        return ""
    
    def send_json(self, data, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

def main():
    bin_path = find_bin()
    if not bin_path:
        print("ERROR: text_generate binary not found", file=sys.stderr)
        print(f"  Expected at: {BIN} or {BIN_DEV}", file=sys.stderr)
        sys.exit(1)
    print(f"Blackwell INT8 Inference Server v0.1.0")
    print(f"Binary: {bin_path}")
    port = int(os.environ.get("PORT", 8080))
    print(f"Listening on http://0.0.0.0:{port}")
    print(f"Endpoints: GET /health, GET /, POST /generate")
    server = HTTPServer(("0.0.0.0", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()

if __name__ == "__main__":
    main()