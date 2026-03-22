"""
test_paged_attention.py

Strategy: trivial page table (page_table[b][lp] = b*pages_per_batch + lp)
With this table, paged access is logically identical to contiguous access.
Output must match F.scaled_dot_product_attention.

Usage:
    python3 build_paged.py
    python3 test_paged_attention.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import paged_attention_cuda

PAGE_SIZE   = 64
PASS_THRESH = 1e-3
FAIL_THRESH = 1e-2


def make_paged_inputs(Q, K, V):
    batch, heads, seq_k, d_head = K.shape
    pages_per_batch = (seq_k + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages     = batch * pages_per_batch

    K_paged = torch.zeros(total_pages, PAGE_SIZE, heads, d_head,
                          dtype=K.dtype, device=K.device)
    V_paged = torch.zeros(total_pages, PAGE_SIZE, heads, d_head,
                          dtype=V.dtype, device=V.device)

    for b in range(batch):
        for t in range(seq_k):
            lp  = t // PAGE_SIZE
            tip = t % PAGE_SIZE
            pp  = b * pages_per_batch + lp
            K_paged[pp, tip, :, :] = K[b, :, t, :]
            V_paged[pp, tip, :, :] = V[b, :, t, :]

    page_table = torch.arange(total_pages, dtype=torch.int32, device=K.device)
    page_table = page_table.reshape(batch, pages_per_batch)

    K_paged = K_paged.reshape(-1)
    V_paged = V_paged.reshape(-1)

    return K_paged, V_paged, page_table, seq_k


def run_test(label, batch, heads, seq_q, seq_k, d_head):
    scale = d_head ** -0.5
    print(f"\n=== {label}: b={batch} h={heads} sq={seq_q} sk={seq_k} d={d_head} ===")

    torch.manual_seed(42)
    Q = torch.randn(batch, heads, seq_q, d_head, device="cuda")
    K = torch.randn(batch, heads, seq_k, d_head, device="cuda")
    V = torch.randn(batch, heads, seq_k, d_head, device="cuda")

    K_paged, V_paged, page_table, seq_k_actual = make_paged_inputs(Q, K, V)

    with torch.no_grad():
        O_paged = paged_attention_cuda.forward(Q, K_paged, V_paged, page_table, seq_k_actual, scale)
        O_ref   = F.scaled_dot_product_attention(Q, K, V,
                      attn_mask=None, dropout_p=0.0, scale=scale)

    err = (O_ref - O_paged).abs().max().item()

    if err < PASS_THRESH:
        verdict = "PASS"
    elif err < FAIL_THRESH:
        verdict = "FAIL (numerical)"
    else:
        verdict = "FAIL"

    print(f"  vs SDPA: {verdict}  max_err={err:.2e}")

    if err >= FAIL_THRESH:
        print("  per-head breakdown:")
        for b_ in range(batch):
            for h_ in range(heads):
                e = (O_ref[b_, h_] - O_paged[b_, h_]).abs().max().item()
                print(f"    batch={b_} head={h_}  max_err={e:.2e}  {'✗' if e >= PASS_THRESH else '✓'}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"

    print("\n==================== SQUARE (seq_q == seq_k) ====================")
    run_test("Square 1", batch=1, heads=4, seq_q=64,  seq_k=64,  d_head=32)
    run_test("Square 2", batch=2, heads=4, seq_q=48,  seq_k=48,  d_head=32)

    print("\n==================== sk > sq ====================")
    run_test("sk>sq 1",  batch=1, heads=4, seq_q=64,  seq_k=128, d_head=32)
    run_test("sk>sq 2",  batch=2, heads=8, seq_q=96,  seq_k=192, d_head=64)

    print("\n==================== sq > sk ====================")
    run_test("sq>sk 1",  batch=1, heads=4, seq_q=128, seq_k=64,  d_head=32)

    print("\n==================== PAGE BOUNDARY CASES ====================")
    run_test("sk=1 page",  batch=1, heads=4, seq_q=32, seq_k=64,  d_head=32)
    run_test("sk=2 pages", batch=1, heads=4, seq_q=32, seq_k=128, d_head=32)
    run_test("sk odd",     batch=1, heads=4, seq_q=32, seq_k=80,  d_head=32)