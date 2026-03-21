"""
test_attention.py — PyTorch validation for attention_v4_multihead

Usage:
    python3 build.py          # compile once
    python3 test_attention.py # run validation
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import attention_cuda  # built by build.py

PASS_THRESH      = 1e-3
NUMERICAL_THRESH = 1e-2

def classify(label, err, square):
    if err < PASS_THRESH:
        verdict = "PASS"
    elif err < NUMERICAL_THRESH:
        verdict = "FAIL (numerical inaccuracy)"
    else:
        verdict = "FAIL (square test)" if square else "FAIL (non-square test)"
    print(f"  {label}: {verdict}  max_err={err:.2e}")

def run_test(label, batch_size, num_heads, seq_q, seq_k, d_head, square):
    scale = d_head ** -0.5
    print(f"\n=== {label}: batch={batch_size} heads={num_heads} "
          f"seq_q={seq_q} seq_k={seq_k} d_head={d_head} ===")

    torch.manual_seed(42)
    Q = torch.randn(batch_size, num_heads, seq_q, d_head, device="cuda")
    K = torch.randn(batch_size, num_heads, seq_k, d_head, device="cuda")
    V = torch.randn(batch_size, num_heads, seq_k, d_head, device="cuda")

    with torch.no_grad():
        O_kernel = attention_cuda.forward(Q, K, V, scale)
        O_ref    = F.scaled_dot_product_attention(Q, K, V,
                       attn_mask=None, dropout_p=0.0, scale=scale)

    err = (O_ref - O_kernel).abs().max().item()
    classify("vs SDPA", err, square)

    if err >= NUMERICAL_THRESH:
        print("  per-head breakdown:")
        for b in range(batch_size):
            for h in range(num_heads):
                e = (O_ref[b,h] - O_kernel[b,h]).abs().max().item()
                print(f"    batch={b} head={h}  max_err={e:.2e}  {'✗' if e >= PASS_THRESH else '✓'}")

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"

    print("\n==================== SQUARE TESTS (seq_q == seq_k) ====================")
    run_test("Square 1", batch_size=1, num_heads=4, seq_q=64, seq_k=64,  d_head=32, square=True)
    run_test("Square 2", batch_size=2, num_heads=4, seq_q=48, seq_k=48,  d_head=32, square=True)
    run_test("Square 3", batch_size=3, num_heads=8, seq_q=96, seq_k=96,  d_head=64, square=True)

    print("\n==================== NON-SQUARE TESTS (seq_q != seq_k) ====================")
    run_test("Non-square 1", batch_size=1, num_heads=4, seq_q=64, seq_k=128, d_head=32, square=False)
    run_test("Non-square 2", batch_size=2, num_heads=4, seq_q=48, seq_k=80,  d_head=32, square=False)
    run_test("Non-square 3", batch_size=3, num_heads=8, seq_q=96, seq_k=160, d_head=64, square=False)