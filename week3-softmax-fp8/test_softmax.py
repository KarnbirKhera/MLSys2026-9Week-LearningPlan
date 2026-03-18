import torch
import torch.nn.functional as F
from build import softmax_cuda

# Test 1: normal random input
x = torch.randn(8, 1024, device="cuda", dtype=torch.float32)
ref = F.softmax(x, dim=-1)
out = softmax_cuda.forward(x)

print(f"\n\n\n\n")

max_err = (ref - out).abs().max().item()
print(f"Normal input  — max absolute error: {max_err:.2e}  {'PASS' if max_err < 1e-5 else 'FAIL'}")

# Test 2: the NaN trap — large values that overflow expf()
x_large = torch.full((1, 8), 1000.0, device="cuda", dtype=torch.float32)
out_large = softmax_cuda.forward(x_large)
print(f"Large input   — output: {out_large}")
print(f"Contains NaN: {out_large.isnan().any().item()}")

# Test 3: rows should sum to 1.0
row_sums = out.sum(dim=-1)
print(f"Row sums (should all be ~1.0):\n{row_sums}")
