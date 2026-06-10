#!/usr/bin/env python3
"""Diagnose SSM state explosion in Qwen3.5-9B GatedDeltaNet.
Loads quantized INT8 weights, runs SSM recurrence, traces state growth.
"""
import numpy as np, os

WT = "weights_int8_qwen35_9b"

def dequant(name, K, N):
    """Dequantize INT8 weight. Returns [N, K] float32."""
    h = np.frombuffer(open(f"{WT}/{name}.int8_t", 'rb').read(20), dtype=np.int32)
    BLOCK = h[2]
    nblks = K // BLOCK
    i8 = np.frombuffer(open(f"{WT}/{name}.int8_t", 'rb').read()[20:], dtype=np.int8).reshape(N, K)
    sc = np.frombuffer(open(f"{WT}/{name}.scale_t", 'rb').read()[20:], dtype=np.float32).reshape(N, nblks)
    W = np.zeros((N, K), dtype=np.float32)
    for i in range(nblks):
        W[:, i*BLOCK:(i+1)*BLOCK] = i8[:, i*BLOCK:(i+1)*BLOCK].astype(np.float32) * sc[:, i:i+1]
    return W  # [N, K]

def load_f32(name, n):
    return np.frombuffer(open(f"{WT}/{name}", 'rb').read(), dtype=np.float32)

# Load layer 1 (has highest A_log values up to +3.7)
L = 1
print(f"=== Layer {L} GatedDeltaNet SSM Diagnostic ===\n")

QKV_W = dequant(f"{L}_linear_attn.in_proj_qkv", 4096, 8192)  # [8192, 4096]
A_W = dequant(f"{L}_linear_attn.in_proj_a", 4096, 32)        # [32, 4096]
B_W = dequant(f"{L}_linear_attn.in_proj_b", 4096, 32)        # [32, 4096]
A_log = load_f32(f"{L}_linear_attn.A_log.f32", 32)
dt_bias = load_f32(f"{L}_linear_attn.dt_bias.f32", 32)

print(f"A_log: mean={A_log.mean():+.4f} range=[{A_log.min():+.4f}, {A_log.max():+.4f}]")
print(f"  exp(A_log): [{np.exp(A_log).min():.4f}, {np.exp(A_log).max():.4f}]")
print(f"  A_log > 0: {(A_log > 0).sum()}/32 channels")
print(f"dt_bias: mean={dt_bias.mean():+.4f} range=[{dt_bias.min():+.4f}, {dt_bias.max():+.4f}]")

# Simulate with actual input (random, since no tokenizer pipeline)
np.random.seed(42)
state = np.zeros((32, 128, 128), dtype=np.float32)
num_steps = 20
print(f"\nRunning {num_steps} SSM steps...")
print(f"{'step':>5} {'state_norm':>12} {'state_max':>12} {'out_norm':>12} {'g_min':>10} {'g_max':>10} {'g_mean':>10} {'|g|>1':>8}")
print("-" * 80)

for step in range(num_steps):
    x = np.random.randn(4096).astype(np.float32) * 0.01  # ~layernorm output
    
    # QKV: x . W^T = [8192]
    qkv = x @ QKV_W.T  # [4096] @ [4096, 8192] = [8192]
    q = qkv[:16*128].reshape(16, 128)
    k = qkv[16*128:32*128].reshape(16, 128)
    v = qkv[32*128:].reshape(32, 128)
    
    # Broadcast NK→NV
    q_bc = q.repeat(2, axis=0)  # [32, 128]
    k_bc = k.repeat(2, axis=0)
    
    # Gates
    a_gate = x @ A_W.T  # [4096] @ [4096, 32] = [32]
    beta = x @ B_W.T    # [32]
    
    # g = -exp(A_log) * softplus(a + dt_bias)
    sp = np.log(1.0 + np.exp(a_gate + dt_bias))
    g = -np.exp(A_log) * sp
    
    # Recurrent step per head
    for h in range(32):
        # state[h] *= g[h]  (decay)
        state[h] *= g[h]
        # kv_mem = state[h]^T @ k_bc[h]
        kv_mem = state[h].T @ k_bc[h]
        # delta = (v[h] - kv_mem) * beta[h]
        delta = (v[h] - kv_mem) * beta[h]
        # state[h] += outer(k_bc[h], delta)
        state[h] += np.outer(k_bc[h], delta)
    
    # Output
    o = np.zeros(32*128)
    for h in range(32):
        o[h*128:(h+1)*128] = state[h].T @ q_bc[h]
    
    g_abs_gt1 = (np.abs(g) > 1.0).sum()
    print(f"{step:5d} {np.linalg.norm(state):12.2e} {np.abs(state).max():12.2e} "
          f"{np.linalg.norm(o):12.2e} {g.min():10.4f} {g.max():10.4f} "
          f"{g.mean():10.4f} {g_abs_gt1:8d}")

# Same test with clamped A_log
print(f"\n\n=== Same test with A_log clamped to ≤ 0 ===\n")
state2 = np.zeros((32, 128, 128), dtype=np.float32)
al_clamped = np.minimum(A_log, 0.0)
print(f"{'step':>5} {'state_norm':>12} {'state_max':>12} {'out_norm':>12} {'g_min':>10} {'g_max':>10} {'g_mean':>10}")
print("-" * 70)

for step in range(num_steps):
    x = np.random.randn(4096).astype(np.float32) * 0.01
    qkv = x @ QKV_W.T
    q = qkv[:16*128].reshape(16, 128)
    k = qkv[16*128:32*128].reshape(16, 128)
    v = qkv[32*128:].reshape(32, 128)
    q_bc = q.repeat(2, axis=0)
    k_bc = k.repeat(2, axis=0)
    a_gate = x @ A_W.T
    beta = x @ B_W.T
    sp = np.log(1.0 + np.exp(a_gate + dt_bias))
    g = -np.exp(al_clamped) * sp
    
    for h in range(32):
        state2[h] *= g[h]
        kv_mem = state2[h].T @ k_bc[h]
        delta = (v[h] - kv_mem) * beta[h]
        state2[h] += np.outer(k_bc[h], delta)
    
    o = np.zeros(32*128)
    for h in range(32):
        o[h*128:(h+1)*128] = state2[h].T @ q_bc[h]
    
    print(f"{step:5d} {np.linalg.norm(state2):12.2e} {np.abs(state2).max():12.2e} "
          f"{np.linalg.norm(o):12.2e} {g.min():10.4f} {g.max():10.4f} {g.mean():10.4f}")
