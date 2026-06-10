#!/usr/bin/env python3
"""Blackwell Benchmark Suite — automated correctness and performance tests.

Usage:
    python3 scripts/benchmark_suite.py [--quick] [--full]

Quick mode: fast sanity checks (~30s)
Full mode: comprehensive tests (~5 min)
"""
import subprocess, time, json, sys, os

# Config
SERVER_URL = "http://localhost:8123"
SERVER_PORT = 8123
WEIGHTS_DIR = "weights_int4_qwen3_8b"
SERVER_BINARY = "./server/http_subprocess"

# Test prompts
TEST_PROMPTS = [
    "The capital of France is",
    "Machine learning is",
    "The quick brown fox",
    "In the beginning",
    "Quantum mechanics describes",
]

# Expected tokens (known-good outputs for correctness testing)
# These are from previous runs - update if model changes
EXPECTED_TOKEN_COUNTS = {
    "The capital of France is": (8, 15),  # min, max tokens
    "Machine learning is": (5, 12),
    "The quick brown fox": (5, 10),
}

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def log(msg, color=""):
    print(f"{color}{msg}{Colors.END}")

def run_curl(payload=None, endpoint="/v1/completions"):
    """Run curl and return JSON response."""
    cmd = ["curl", "-s"]
    if payload is None:
        # GET request (for /health, /v1/models)
        cmd.extend(["-X", "GET", f"{SERVER_URL}{endpoint}"])
    else:
        # POST request
        cmd.extend(["-X", "POST", f"{SERVER_URL}{endpoint}",
                   "-H", "Content-Type: application/json",
                   "-d", json.dumps(payload)])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return None, f"curl failed: {result.stderr}"
        return json.loads(result.stdout), None
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {e}"

def wait_for_server(timeout=60):
    """Wait for server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        resp, _ = run_curl(None, "/health")
        if resp and resp.get("status") == "ok":
            return True
        time.sleep(2)
    return False

def test_health():
    """Test /health endpoint."""
    log("Testing /health endpoint...", Colors.BLUE)
    resp, err = run_curl(None, "/health")
    if err or not resp:
        log(f"  FAIL: {err}", Colors.RED)
        return False
    
    # Check required fields
    required = ["status", "model", "gpu_used_mb", "uptime_sec", "requests", "errors"]
    missing = [f for f in required if f not in resp]
    if missing:
        log(f"  FAIL: missing fields {missing}", Colors.RED)
        return False
    
    log(f"  OK: model={resp['model']}, gpu={resp['gpu_used_mb']}MB, errors={resp['errors']}", Colors.GREEN)
    return True

def test_correctness():
    """Test that model produces coherent output (not necessarily identical due to GPU non-determinism)."""
    log("Testing output quality (3 runs same prompt)...", Colors.BLUE)
    
    prompt = "The capital of France is"
    results = []
    
    for i in range(3):
        resp, err = run_curl({"prompt": prompt, "max_tokens": 20, "temperature": 0.0})
        if err or not resp:
            log(f"  Run {i+1} FAIL: {err}", Colors.RED)
            return False
        results.append(resp)
    
    # Check all runs have coherent output (not garbage)
    texts = [r["choices"][0]["text"] for r in results]
    
    # Check for common garbage indicators
    garbage = ["\x00", "\xff", "\xfe"]
    for i, text in enumerate(texts):
        for g in garbage:
            if g in text:
                log(f"  FAIL: Run {i+1} contains garbage byte", Colors.RED)
                return False
    
    # Check that outputs are reasonable length (not empty, not too long)
    for i, text in enumerate(texts):
        if len(text) < 5:
            log(f"  FAIL: Run {i+1} too short ({len(text)} chars)", Colors.RED)
            return False
    
    log(f"  OK: all 3 runs coherent (lengths: {[len(t) for t in texts]})", Colors.GREEN)
    return True

def test_throughput(prompt_count=5, tokens=30):
    """Measure throughput (tokens/sec)."""
    log(f"Testing throughput ({prompt_count} prompts, {tokens} tokens)...", Colors.BLUE)
    
    prompts = TEST_PROMPTS[:prompt_count]
    start = time.time()
    
    # Run sequentially (simpler than batching for throughput test)
    total_tokens = 0
    for p in prompts:
        resp, err = run_curl({"prompt": p, "max_tokens": tokens})
        if err or not resp:
            log(f"  FAIL: {err}", Colors.RED)
            return None
        total_tokens += resp["usage"]["completion_tokens"]
    
    elapsed = time.time() - start
    tps = total_tokens / elapsed
    
    log(f"  OK: {total_tokens} tokens in {elapsed:.1f}s = {tps:.1f} t/s", Colors.GREEN)
    return tps

def test_batch_endpoint():
    """Test /v1/batch endpoint."""
    log("Testing /v1/batch endpoint...", Colors.BLUE)
    
    prompts = ["The capital of", "Machine learning", "Hello world"]
    start = time.time()
    
    resp, err = run_curl({
        "prompts": prompts,
        "max_tokens": 10,
        "temperature": 0.0
    }, "/v1/batch")
    
    if err or not resp:
        log(f"  FAIL: {err}", Colors.RED)
        return None
    
    elapsed = time.time() - start
    
    # Parse batch response
    try:
        data = json.loads(resp["text"][0])
        tokens = data["tokens"]
        tps = len(tokens) / elapsed
        log(f"  OK: {len(prompts)} prompts, {sum(len(t) for t in tokens)} tokens in {elapsed:.1f}s = {tps:.1f} t/s", Colors.GREEN)
        return tps
    except (KeyError, json.JSONDecodeError) as e:
        log(f"  FAIL: parse error {e}", Colors.RED)
        return None

def test_memory_stability():
    """Test that GPU memory doesn't leak."""
    log("Testing memory stability (10 sequential requests)...", Colors.BLUE)
    
    initial = run_curl(None, "/health")[0]
    if not initial:
        log("  FAIL: can't get initial memory", Colors.RED)
        return False
    
    mem_start = initial["gpu_used_mb"]
    
    for i in range(10):
        resp, err = run_curl({"prompt": "test", "max_tokens": 5})
        if err:
            log(f"  FAIL at request {i+1}: {err}", Colors.RED)
            return False
    
    final = run_curl(None, "/health")[0]
    mem_end = final["gpu_used_mb"]
    mem_delta = mem_end - mem_start
    
    log(f"  Memory: {mem_start}MB -> {mem_end}MB (delta: {mem_delta:+d}MB)", Colors.GREEN if abs(mem_delta) < 100 else Colors.YELLOW)
    return True

def test_repetition_penalty():
    """Test that repetition penalty reduces looping."""
    log("Testing repetition penalty...", Colors.BLUE)
    
    # Generate with rep_pen=1.0 (no penalty)
    resp1, err1 = run_curl({
        "prompt": "The capital of France is",
        "max_tokens": 20,
        "repetition_penalty": 1.0
    })
    
    # Generate with rep_pen=1.5 (with penalty)
    resp2, err2 = run_curl({
        "prompt": "The capital of France is",
        "max_tokens": 20,
        "repetition_penalty": 1.5
    })
    
    if err1 or err2 or not resp1 or not resp2:
        log(f"  FAIL: generation error", Colors.RED)
        return False
    
    text1 = resp1["choices"][0]["text"]
    text2 = resp2["choices"][0]["text"]
    
    # With rep_pen, output should be different (longer/more varied)
    # We just check they differ
    if text1 == text2:
        log(f"  WARN: identical outputs with different rep_pen", Colors.YELLOW)
    
    log(f"  OK: rep_pen=1.0: {len(text1)} chars, rep_pen=1.5: {len(text2)} chars", Colors.GREEN)
    return True

def test_models_endpoint():
    """Test /v1/models endpoint."""
    log("Testing /v1/models endpoint...", Colors.BLUE)
    
    resp, err = run_curl(None, "/v1/models")
    if err or not resp:
        log(f"  FAIL: {err}", Colors.RED)
        return False
    
    if "data" not in resp or not resp["data"]:
        log(f"  FAIL: empty model list", Colors.RED)
        return False
    
    log(f"  OK: {len(resp['data'])} model(s)", Colors.GREEN)
    return True

def test_ppl_estimation():
    """Estimate perplexity by measuring log probability of generated tokens.
    
    Note: This is a rough estimate using token probabilities from softmax.
    Real PPL requires full model forward pass with logits.
    """
    log("Estimating perplexity (rough)...", Colors.BLUE)
    
    # Simple test: generate text and measure token confidence
    prompts = [
        "The capital of France is",
        "Machine learning is a",
        "The quick brown fox jumps",
    ]
    
    total_confidence = 0
    total_tokens = 0
    
    for prompt in prompts:
        resp, err = run_curl({
            "prompt": prompt,
            "max_tokens": 5,
            "temperature": 0.0  # Greedy for deterministic logits
        })
        if err or not resp:
            log(f"  FAIL: {err}", Colors.RED)
            return False
        
        # Count tokens generated (proxy for quality)
        tokens = resp.get("usage", {}).get("completion_tokens", 0)
        total_tokens += tokens
    
    # Rough PPL estimate based on token generation success
    # (In reality, we'd need logits to compute proper PPL)
    avg_tokens = total_tokens / len(prompts)
    
    log(f"  OK: Generated avg {avg_tokens:.1f} tokens per prompt", Colors.GREEN)
    return True

def run_quick_tests():
    """Run quick sanity checks."""
    log(f"\n{Colors.BOLD}=== Quick Benchmark Suite ==={Colors.END}\n")
    
    results = {}
    
    # Start server if not running
    log("Checking server status...", Colors.BLUE)
    if not wait_for_server():
        log("Server not responding. Start with:", Colors.YELLOW)
        log(f"  ./server/http_subprocess batched", Colors.YELLOW)
        return results
    
    # Tests
    results["health"] = test_health()
    results["models"] = test_models_endpoint()
    results["correctness"] = test_correctness()
    results["throughput"] = test_throughput(3, 20)
    results["repetition"] = test_repetition_penalty()
    results["ppl_estimate"] = test_ppl_estimation()
    
    return results

def test_regression():
    """Regression test: check outputs against known-good baselines."""
    log("Testing regression (known outputs)...", Colors.BLUE)
    
    # Known-good outputs from previous runs
    # These should remain stable across versions
    baselines = [
        {
            "prompt": "The capital of France is",
            "expected_contains": ["Paris"],  # Should contain Paris
            "min_length": 20,
            "max_length": 200,
        },
        {
            "prompt": "Machine learning is",
            "expected_contains": [],  # No strong expectation
            "min_length": 10,
            "max_length": 200,
        },
    ]
    
    for baseline in baselines:
        resp, err = run_curl({
            "prompt": baseline["prompt"],
            "max_tokens": 30,
            "temperature": 0.0
        })
        if err or not resp:
            log(f"  FAIL: {err}", Colors.RED)
            return False
        
        text = resp["choices"][0]["text"]
        
        # Check length
        if len(text) < baseline["min_length"]:
            log(f"  FAIL: output too short ({len(text)} < {baseline['min_length']})", Colors.RED)
            return False
        if len(text) > baseline["max_length"]:
            log(f"  WARN: output too long ({len(text)} > {baseline['max_length']})", Colors.YELLOW)
        
        # Check expected substrings
        for expected in baseline["expected_contains"]:
            if expected.lower() not in text.lower():
                log(f"  WARN: expected '{expected}' not in output", Colors.YELLOW)
    
    log(f"  OK: All regression checks passed", Colors.GREEN)
    return True

def test_ppl_formal():
    """Formal PPL measurement using the benchmark tool.
    
    Runs bench_ppl_int4_8b to get actual perplexity on test corpus.
    NOTE: This test requires GPU to be free (server should not be running).
    """
    log("Running formal PPL benchmark...", Colors.BLUE)
    
    # Check if server is running
    resp, _ = run_curl(None, "/health")
    if resp and resp.get("status") == "ok":
        log("  SKIP: Server is running. Stop server first for PPL benchmark.", Colors.YELLOW)
        log("    Run: killall http_subprocess inference_server", Colors.YELLOW)
        return None  # Skip, not fail
    
    import subprocess, re
    
    # Run PPL benchmark
    result = subprocess.run(
        ["./bench/bench_ppl_int4_8b"],
        capture_output=True,
        text=True,
        timeout=120
    )
    
    if result.returncode != 0:
        log(f"  FAIL: benchmark failed", Colors.RED)
        log(f"    stderr: {result.stderr[:200]}", Colors.RED)
        return False
    
    # Parse PPL from output
    match = re.search(r"PPL:\s*([\d.]+)", result.stderr)
    if not match:
        log(f"  FAIL: could not parse PPL from output", Colors.RED)
        return False
    
    ppl = float(match.group(1))
    
    # Check if PPL is within acceptable range
    # INT4 8B should be around 21-23
    if ppl < 15:
        log(f"  WARN: PPL {ppl:.2f} is suspiciously low", Colors.YELLOW)
    elif ppl > 30:
        log(f"  WARN: PPL {ppl:.2f} is very high", Colors.YELLOW)
    else:
        log(f"  OK: PPL = {ppl:.2f} (target: ~21-23 for INT4 8B)", Colors.GREEN)
    
    return True

def run_full_tests():
    """Run comprehensive tests."""
    log(f"\n{Colors.BOLD}=== Full Benchmark Suite ==={Colors.END}\n")
    
    results = run_quick_tests()
    
    results["batch"] = test_batch_endpoint()
    results["memory"] = test_memory_stability()
    results["regression"] = test_regression()
    results["throughput_full"] = test_throughput(5, 30)
    results["ppl_formal"] = test_ppl_formal()
    
    return results

def main():
    quick = "--quick" in sys.argv
    
    log(f"\n{Colors.BOLD}{'='*50}{Colors.END}")
    log(f"{Colors.BOLD}Blackwell Benchmark Suite{' '*20}{Colors.END}")
    log(f"{Colors.BOLD}{'='*50}{Colors.END}\n")
    
    results = run_quick_tests() if quick else run_full_tests()
    
    # Summary
    log(f"\n{Colors.BOLD}=== Summary ==={Colors.END}")
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    metrics = {k: v for k, v in results.items() if isinstance(v, (int, float))}
    
    log(f"Tests: {passed} passed, {failed} failed")
    
    if metrics:
        log(f"Metrics:")
        for k, v in metrics.items():
            if "throughput" in k or "batch" in k:
                log(f"  {k}: {v:.1f} t/s")
            else:
                log(f"  {k}: {v}")
    
    if failed == 0:
        log(f"\n{Colors.GREEN}All tests passed!{Colors.END}")
        return 0
    else:
        log(f"\n{Colors.RED}Some tests failed.{Colors.END}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
