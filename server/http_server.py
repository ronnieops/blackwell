#!/usr/bin/env python3
"""Blackwell HTTP Server - Python wrapper for C++ inference_server"""

import subprocess
import json
import signal
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time

MODEL = sys.argv[1] if len(sys.argv) > 1 else "weights_int8_bf16"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8123

class InferenceEngine:
    def __init__(self, model):
        self.proc = subprocess.Popen(
            ["./server/inference_server", model],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0)
        time.sleep(7)  # wait for server ready
        self.lock = threading.Lock()

    def generate(self, prompt, max_tokens=30, temperature=0.7, top_k=40):
        with self.lock:
            req = json.dumps({"prompts": [prompt], "max_tokens": max_tokens,
                            "temperature": temperature, "top_k": top_k}) + "\n"
            self.proc.stdin.write(req.encode())
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if not line:
                return [], ""
            resp = json.loads(line.decode())
            return resp.get("tokens", [[]])[0], resp.get("text", [""])[0]

    def stop(self):
        self.proc.terminate()
        self.proc.wait()

engine = InferenceEngine(MODEL)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        elif self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {"object":"list","data":[{"id":"blackwell-1.7B","object":"model","created":0,"owned_by":"blackwell"}]}
            self.wfile.write(json.dumps(resp).encode())
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode()
        try:
            data = json.loads(body)
        except:
            self.send_error(400)
            return

        if self.path == "/v1/completions":
            prompt = data.get("prompt", "")
            max_tokens = data.get("max_tokens", 30)
            temperature = data.get("temperature", 0.7)
            top_k = data.get("top_k", 40)
            tokens, text = engine.generate(prompt, max_tokens, temperature, top_k)
            resp = {"id":"cmpl-0","object":"text_completion","created":0,"model":"blackwell-1.7B",
                   "choices":[{"text": text, "index": 0, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 0, "completion_tokens": len(tokens), "total_tokens": len(tokens)}}
        elif self.path == "/v1/chat/completions":
            content = ""
            for msg in data.get("messages", []):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
            prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n"
            max_tokens = data.get("max_tokens", 30)
            temperature = data.get("temperature", 0.7)
            top_k = data.get("top_k", 40)
            tokens, text = engine.generate(prompt, max_tokens, temperature, top_k)
            resp = {"id":"chatcmpl-0","object":"chat.completion","created":0,"model":"blackwell-1.7B",
                   "choices":[{"index":0,"message":{"role":"assistant","content": text},"finish_reason":"stop"}],
                   "usage": {"prompt_tokens":0,"completion_tokens":len(tokens),"total_tokens":len(tokens)}}
        else:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode())

    def log_message(self, fmt, *args):
        pass

try:
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Blackwell HTTP Server on port {PORT}")
    server.serve_forever()
finally:
    engine.stop()
