"""
test_mla_attention.py

Strategy: trivial page table (page_table[lp] = lp).
With this table, paged access is logically identical to contiguous access.

Validates two outputs:
  - output  must match manual MLA reference (q_nope @ ckv^T + q_pe @ kpe^T)
  - lse     must match logsumexp of raw MLA attention scores
        shape: [num_tokens, num_qo_heads]
        layout: token outermost, head innermost  (matches competition spec)

Usage:
    python3 build_paged.py
    python3 test_mla_attention.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import paged_attention_cuda

PAGE_SIZE   = 64
PASS_THRESH = 1e-3
FAIL_THRESH = 1e-2


def make_paged_cache(contiguous, feat_dim, num_kv_tokens):
    pages_per_seq = (num_kv_tokens + PAGE_SIZE - 1) // PAGE_SIZE
    paged = torch.zeros(pages_per_seq, PAGE_SIZE, feat_dim,
                        dtype=contiguous.dtype, device=contiguous.device)
    for t in range(num_kv_tokens):
        lp  = t // PAGE_SIZE
        tip = t % PAGE_SIZE
        paged[lp, tip, :] = contiguous[t, :]
    page_table = torch.arange(pages_per_seq, dtype=torch.int32, device=contiguous.device)
    return paged.reshape(-1), page_table


def mla_reference(q_nope, q_pe, ckv, kpe, scale):
    # q_nope: [num_tokens, num_heads, head_dim_ckv]
    # permute to [num_heads, num_tokens, head_dim_ckv] for batched matmul
    qn = q_nope.permute(1, 0, 2)
    qp = q_pe.permute(1, 0, 2)

    # raw scores: [num_heads, num_tokens, num_kv_tokens]
    raw_scores = (qn @ ckv.T + qp @ kpe.T) * scale

    # lse: logsumexp over kv dimension -> [num_heads, num_tokens]
    # then permute to [num_tokens, num_heads] to match spec layout
    lse = torch.logsumexp(raw_scores, dim=-1).permute(1, 0)

    # output: [num_heads, num_tokens, head_dim_ckv] -> [num_tokens, num_heads, head_dim_ckv]
    weights = F.softmax(raw_scores, dim=-1)
    output  = (weights @ ckv).permute(1, 0, 2)

    return output, lse


def run_test(label, num_tokens, num_heads, num_kv_tokens, head_dim_ckv, head_dim_kpe):
    scale = (head_dim_ckv + head_dim_kpe) ** -0.5
    print(f"\n=== {label}: tok={num_tokens} h={num_heads} kv={num_kv_tokens} ckv={head_dim_ckv} kpe={head_dim_kpe} ===")

    torch.manual_seed(42)
    q_nope = torch.randn(num_tokens, num_heads, head_dim_ckv, device="cuda")
    q_pe   = torch.randn(num_tokens, num_heads, head_dim_kpe, device="cuda")
    ckv    = torch.randn(num_kv_tokens, head_dim_ckv,         device="cuda")
    kpe    = torch.randn(num_kv_tokens, head_dim_kpe,         device="cuda")

    ckv_paged, page_table = make_paged_cache(ckv, head_dim_ckv, num_kv_tokens)
    kpe_paged, _          = make_paged_cache(kpe, head_dim_kpe, num_kv_tokens)

    with torch.no_grad():
        O_kernel, lse_kernel = paged_attention_cuda.forward_mla(
            q_nope, q_pe, ckv_paged, kpe_paged,
            page_table, num_kv_tokens, scale
        )
        O_ref, lse_ref = mla_reference(q_nope, q_pe, ckv, kpe, scale)

    err_O   = (O_ref   - O_kernel).abs().max().item()
    err_lse = (lse_ref - lse_kernel).abs().max().item()

    verdict_O   = "PASS" if err_O   < PASS_THRESH else ("FAIL (numerical)" if err_O   < FAIL_THRESH else "FAIL")
    verdict_lse = "PASS" if err_lse < PASS_THRESH else ("FAIL (numerical)" if err_lse < FAIL_THRESH else "FAIL")

    print(f"  output vs reference: {verdict_O:<18}  max_err={err_O:.2e}")
    print(f"  lse    vs reference: {verdict_lse:<18}  max_err={err_lse:.2e}")

    if err_O >= FAIL_THRESH:
        print("  per-head output breakdown:")
        for h in range(num_heads):
            e = (O_ref[:, h, :] - O_kernel[:, h, :]).abs().max().item()
            print(f"    head={h}  max_err={e:.2e}  {'✗' if e >= PASS_THRESH else '✓'}")

    if err_lse >= FAIL_THRESH:
        print("  per-head lse breakdown:")
        for h in range(num_heads):
            e = (lse_ref[:, h] - lse_kernel[:, h]).abs().max().item()
            print(f"    head={h}  max_err={e:.2e}  {'✗' if e >= PASS_THRESH else '✓'}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"

    print("\n" + "=" * 60)
    print("  KERNEL: attention_mla_v1")
    print("  Reference: manual MLA (q_nope @ ckv^T + q_pe @ kpe^T)")
    print("  LSE layout: [num_tokens, num_qo_heads] — matches spec")
    print("=" * 60)

    print("\n==================== SMALL DIMS (correctness first) ====================")
    run_test("square",  num_tokens=32,  num_heads=4,  num_kv_tokens=64,  head_dim_ckv=32,  head_dim_kpe=16)
    run_test("square",  num_tokens=48,  num_heads=4,  num_kv_tokens=48,  head_dim_ckv=32,  head_dim_kpe=16)

    print("\n==================== kv > tok ====================")
    run_test("kv>tok",  num_tokens=64,  num_heads=4,  num_kv_tokens=128, head_dim_ckv=32,  head_dim_kpe=16)
    run_test("kv>tok",  num_tokens=96,  num_heads=8,  num_kv_tokens=192, head_dim_ckv=64,  head_dim_kpe=32)

    print("\n==================== tok > kv ====================")
    run_test("tok>kv",  num_tokens=128, num_heads=4,  num_kv_tokens=64,  head_dim_ckv=32,  head_dim_kpe=16)

    print("\n==================== PAGE BOUNDARY CASES ====================")
    run_test("kv=1pg",  num_tokens=32,  num_heads=4,  num_kv_tokens=64,  head_dim_ckv=32,  head_dim_kpe=16)
    run_test("kv=2pg",  num_tokens=32,  num_heads=4,  num_kv_tokens=128, head_dim_ckv=32,  head_dim_kpe=16)
    run_test("kv odd",  num_tokens=32,  num_heads=4,  num_kv_tokens=80,  head_dim_ckv=32,  head_dim_kpe=16)

    print("\n==================== SPEC DIMENSIONS ====================")
    run_test("spec",    num_tokens=32,  num_heads=16, num_kv_tokens=64,  head_dim_ckv=512, head_dim_kpe=64)
    run_test("spec",    num_tokens=64,  num_heads=16, num_kv_tokens=128, head_dim_ckv=512, head_dim_kpe=64)