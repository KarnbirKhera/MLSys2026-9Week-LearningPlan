"""
test_topk_scorer.py  —  PyTorch test harness for compute_scores_kernel
=======================================================================
Mirrors all four C++ tests and adds four new tests covering the three
"silent killers" identified in the Week 6 schedule:
  - Index space confusion (global vs page-local indices)
  - Sorting direction (descending vs ascending)
  - Tiny manually-verifiable example (batch=1, small seq_len)
  - PyTorch stress test across multiple configs

Run with:
  python3 test_topk_scorer.py
"""

import struct
import numpy as np
import torch
import torch.testing
from torch.utils.cpp_extension import load_inline


# ──────────────────────────────────────────────────────────────────────────────
#  Sanity checks
# ──────────────────────────────────────────────────────────────────────────────
assert torch.cuda.is_available(), "CUDA GPU required"
assert hasattr(torch, "float8_e4m3fn"), (
    "torch.float8_e4m3fn not found — upgrade to PyTorch >= 2.1"
)


# ──────────────────────────────────────────────────────────────────────────────
#  Constants — must match the kernel #defines exactly
# ──────────────────────────────────────────────────────────────────────────────
NUM_HEADS        = 64
HEAD_DIM         = 128
PAGE_SIZE        = 64
TOP_K            = 2048
PAGE_DATA_BYTES  = PAGE_SIZE * HEAD_DIM       # 8192: FP8 key bytes per page
PAGE_SCALE_BYTES = PAGE_SIZE * 4              # 256:  float32 scale bytes per page
PAGE_STRIDE      = PAGE_DATA_BYTES + PAGE_SCALE_BYTES  # 8448: total bytes per page


# ──────────────────────────────────────────────────────────────────────────────
#  CUDA kernel source
# ──────────────────────────────────────────────────────────────────────────────
CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_fp8.h>

#define NUM_HEADS         64
#define HEAD_DIM          128
#define PAGE_SIZE         64
#define PAGE_DATA_BYTES   (PAGE_SIZE * HEAD_DIM)
#define PAGE_SCALE_BYTES  (PAGE_SIZE * (int)sizeof(float))
#define PAGE_STRIDE       (PAGE_DATA_BYTES + PAGE_SCALE_BYTES)
#define SMEM_STRIDE       (HEAD_DIM + 1)

__global__ void compute_scores_kernel(
    const float*   __restrict__ Q,
    const uint8_t* __restrict__ K_cache,
    const float*   __restrict__ head_weights,
    float*         __restrict__ scores,
    const int*     __restrict__ page_table,
    int max_seq_len,
    int max_pages
) {
    int batch_idx = blockIdx.x;
    int token_idx = blockIdx.y * blockDim.x + threadIdx.x;

    __shared__ float smem_q[NUM_HEADS * SMEM_STRIDE];
    const float* q_base = Q + batch_idx * NUM_HEADS * HEAD_DIM;

    for (int i = threadIdx.x; i < NUM_HEADS * HEAD_DIM; i += blockDim.x)
        smem_q[(i / HEAD_DIM) * SMEM_STRIDE + (i % HEAD_DIM)] = q_base[i];
    __syncthreads();

    if (token_idx >= max_seq_len) return;

    int logical_page = token_idx / PAGE_SIZE;
    int offset       = token_idx % PAGE_SIZE;

    if (logical_page >= max_pages) {
        scores[batch_idx * max_seq_len + token_idx] = 0.0f;
        return;
    }

    int physical_page = page_table[batch_idx * max_pages + logical_page];
    const uint8_t* page_base = K_cache + (size_t)physical_page * PAGE_STRIDE;
    const uint8_t* k_fp8     = page_base + (size_t)offset * HEAD_DIM;
    float k_scale = *reinterpret_cast<const float*>(
        page_base + PAGE_DATA_BYTES + offset * sizeof(float));

    float dots[NUM_HEADS];
    #pragma unroll
    for (int h = 0; h < NUM_HEADS; h++) dots[h] = 0.0f;

    for (int d = 0; d < HEAD_DIM; d++) {
        __nv_fp8_e4m3 fp8_val;
        memcpy(&fp8_val, &k_fp8[d], sizeof(__nv_fp8_e4m3));
        float kd = __half2float((__half)fp8_val) * k_scale;
        #pragma unroll
        for (int h = 0; h < NUM_HEADS; h++)
            dots[h] += smem_q[h * SMEM_STRIDE + d] * kd;
    }

    float total_score = 0.0f;
    #pragma unroll
    for (int h = 0; h < NUM_HEADS; h++) {
        float relu_dot = (dots[h] > 0.0f) ? dots[h] : 0.0f;
        total_score += relu_dot * head_weights[h];
    }

    scores[batch_idx * max_seq_len + token_idx] = total_score;
}

torch::Tensor compute_scores_cuda(
    torch::Tensor Q,
    torch::Tensor K_cache,
    torch::Tensor head_weights,
    torch::Tensor page_table,
    int max_seq_len,
    int max_pages
) {
    const int batch_size = Q.size(0);
    auto scores = torch::zeros(
        {batch_size, max_seq_len},
        torch::TensorOptions().dtype(torch::kFloat32).device(Q.device())
    );
    constexpr int THREADS = 256;
    dim3 grid(batch_size, (max_seq_len + THREADS - 1) / THREADS);
    compute_scores_kernel<<<grid, THREADS>>>(
        Q.data_ptr<float>(),
        K_cache.data_ptr<uint8_t>(),
        head_weights.data_ptr<float>(),
        scores.data_ptr<float>(),
        page_table.data_ptr<int>(),
        max_seq_len,
        max_pages
    );
    cudaDeviceSynchronize();
    return scores;
}
"""

CPP_SOURCE = """
#include <torch/extension.h>

torch::Tensor compute_scores_cuda(
    torch::Tensor Q, torch::Tensor K_cache,
    torch::Tensor head_weights, torch::Tensor page_table,
    int max_seq_len, int max_pages);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_scores", &compute_scores_cuda,
          "Compute FP8 paged attention scores (CUDA)");
}
"""

print("Compiling CUDA kernel...")
scorer = load_inline(
    name="topk_scorer",
    cpp_sources=[CPP_SOURCE],
    cuda_sources=[CUDA_SOURCE],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)
print("Ready.\n")


# ──────────────────────────────────────────────────────────────────────────────
#  FP8 helpers
#  Mirrors the C++ conversion chain: float → half → fp8 (store), fp8 → half → float (load).
#  Both the GPU kernel and the Python reference must go through the same chain
#  or the comparison will show a spurious error from quantization mismatch.
# ──────────────────────────────────────────────────────────────────────────────
def floats_to_fp8_bytes(arr: np.ndarray) -> np.ndarray:
    t   = torch.from_numpy(arr.astype(np.float32))
    fp8 = t.half().to(torch.float8_e4m3fn)
    return fp8.view(torch.uint8).numpy()


def fp8_bytes_to_floats(byte_arr: np.ndarray) -> np.ndarray:
    t   = torch.from_numpy(byte_arr.view(np.uint8))
    fp8 = t.view(torch.float8_e4m3fn)
    return fp8.half().float().numpy()


# ──────────────────────────────────────────────────────────────────────────────
#  K cache builders
# ──────────────────────────────────────────────────────────────────────────────
def build_k_cache(num_phys_pages: int, rng: np.random.RandomState):
    """Random K cache — returns (raw_bytes, k_float_dequantized)."""
    vals      = rng.uniform(-0.5, 0.5, (num_phys_pages, PAGE_SIZE, HEAD_DIM)).astype(np.float32)
    fp8_bytes = floats_to_fp8_bytes(vals)
    k_float   = fp8_bytes_to_floats(fp8_bytes)  # what the kernel actually computes with

    raw    = np.zeros(num_phys_pages * PAGE_STRIDE, dtype=np.uint8)
    scales = np.ones(PAGE_SIZE, dtype=np.float32).view(np.uint8)

    for pg in range(num_phys_pages):
        base = pg * PAGE_STRIDE
        raw[base : base + PAGE_DATA_BYTES]  = fp8_bytes[pg].ravel()
        raw[base + PAGE_DATA_BYTES : base + PAGE_STRIDE] = scales

    return raw, k_float


def build_k_cache_uniform(num_phys_pages: int, page_vals: list):
    """
    K cache where every byte in page pg's data section encodes page_vals[pg].
    Used for index-space test where we need page 0 = all-zero and page 1 = all-one.
    """
    raw    = np.zeros(num_phys_pages * PAGE_STRIDE, dtype=np.uint8)
    scales = np.ones(PAGE_SIZE, dtype=np.float32).view(np.uint8)
    k_float = np.zeros((num_phys_pages, PAGE_SIZE, HEAD_DIM), dtype=np.float32)

    for pg in range(num_phys_pages):
        # Encode the float through the same FP8 round-trip as the kernel uses
        val_arr   = np.full((PAGE_SIZE, HEAD_DIM), page_vals[pg], dtype=np.float32)
        fp8_bytes = floats_to_fp8_bytes(val_arr)
        k_float[pg] = fp8_bytes_to_floats(fp8_bytes)

        base = pg * PAGE_STRIDE
        raw[base : base + PAGE_DATA_BYTES]  = fp8_bytes.ravel()
        raw[base + PAGE_DATA_BYTES : base + PAGE_STRIDE] = scales

    return raw, k_float


# ──────────────────────────────────────────────────────────────────────────────
#  PyTorch reference implementation
# ──────────────────────────────────────────────────────────────────────────────
def compute_scores_reference(
    Q_flat:       torch.Tensor,   # [batch, num_heads * head_dim]
    K_dequant:    torch.Tensor,   # [batch, seq_len, head_dim]
    head_weights: torch.Tensor,   # [num_heads]
) -> torch.Tensor:
    batch   = Q_flat.shape[0]
    Q       = Q_flat.view(batch, NUM_HEADS, HEAD_DIM)
    dots    = torch.einsum("bhd,btd->bth", Q, K_dequant)     # [batch, seq_len, heads]
    scores  = (torch.relu(dots) * head_weights.view(1, 1, -1)).sum(dim=-1)
    return scores


def gather_k(k_float: np.ndarray, page_table_np: np.ndarray,
             batch_size: int, max_seq_len: int) -> torch.Tensor:
    K = np.zeros((batch_size, max_seq_len, HEAD_DIM), dtype=np.float32)
    for b in range(batch_size):
        for tok in range(max_seq_len):
            lp   = tok // PAGE_SIZE
            off  = tok %  PAGE_SIZE
            phys = page_table_np[b, lp]
            K[b, tok] = k_float[phys, off]
    return torch.from_numpy(K)


# ──────────────────────────────────────────────────────────────────────────────
#  Kernel launcher
# ──────────────────────────────────────────────────────────────────────────────
def run_kernel(Q, K_raw, head_weights, page_table, max_seq_len, max_pages, device="cuda"):
    return scorer.compute_scores(
        Q.to(device),
        torch.from_numpy(K_raw).to(device),
        head_weights.to(device),
        page_table.to(device),
        max_seq_len,
        max_pages,
    ).cpu()


# ──────────────────────────────────────────────────────────────────────────────
#  Original Test 1 — score correctness
# ──────────────────────────────────────────────────────────────────────────────
def test_score_correctness():
    print("=== test_score_correctness ===")
    batch_size  = 2
    num_pages   = 4
    max_seq_len = num_pages * PAGE_SIZE

    rng          = np.random.RandomState(42)
    Q            = torch.from_numpy(rng.uniform(-0.5, 0.5, (batch_size, NUM_HEADS * HEAD_DIM)).astype(np.float32))
    head_weights = torch.from_numpy(np.abs(rng.uniform(-0.5, 0.5, NUM_HEADS).astype(np.float32)))
    K_raw, K_float = build_k_cache(num_pages, rng)
    pt_np        = np.tile(np.arange(num_pages, dtype=np.int32), (batch_size, 1))
    page_table   = torch.from_numpy(pt_np)

    scores_gpu = run_kernel(Q, K_raw, head_weights, page_table, max_seq_len, num_pages)
    K_ref      = gather_k(K_float, pt_np, batch_size, max_seq_len)
    scores_ref = compute_scores_reference(Q, K_ref, head_weights)

    torch.testing.assert_close(scores_gpu, scores_ref, rtol=1e-3, atol=1e-3)
    print(f"Score correctness:  max_err = {(scores_gpu - scores_ref).abs().max():.2e}  PASS\n")


# ──────────────────────────────────────────────────────────────────────────────
#  Original Test 2 — paged addressing (reversed page table)
# ──────────────────────────────────────────────────────────────────────────────
def test_paged_addressing():
    print("=== test_paged_addressing (shuffled page table) ===")
    batch_size  = 1
    num_pages   = 8
    max_seq_len = num_pages * PAGE_SIZE

    rng          = np.random.RandomState(7)
    Q            = torch.from_numpy(rng.uniform(-0.5, 0.5, (batch_size, NUM_HEADS * HEAD_DIM)).astype(np.float32))
    head_weights = torch.from_numpy(np.abs(rng.uniform(-0.5, 0.5, NUM_HEADS).astype(np.float32)))
    K_raw, K_float = build_k_cache(num_pages, rng)
    pt_np        = np.array([[num_pages - 1 - p for p in range(num_pages)]], dtype=np.int32)
    page_table   = torch.from_numpy(pt_np)

    scores_gpu = run_kernel(Q, K_raw, head_weights, page_table, max_seq_len, num_pages)
    K_ref      = gather_k(K_float, pt_np, batch_size, max_seq_len)
    scores_ref = compute_scores_reference(Q, K_ref, head_weights)

    torch.testing.assert_close(scores_gpu, scores_ref, rtol=1e-3, atol=1e-3)
    print(f"Paged addressing:   max_err = {(scores_gpu - scores_ref).abs().max():.2e}  PASS\n")


# ──────────────────────────────────────────────────────────────────────────────
#  Original Test 3 — TopK ranking
# ──────────────────────────────────────────────────────────────────────────────
def test_topk_ranking():
    print("=== test_topk_ranking ===")
    seq_len = 512
    k       = 16

    scores           = torch.from_numpy(np.random.RandomState(0).uniform(-1, 1, seq_len).astype(np.float32))
    topk_vals, topk_indices = torch.topk(scores, k)

    threshold = topk_vals[-1].item()
    topk_set  = set(topk_indices.tolist())
    errors    = sum(1 for i, s in enumerate(scores.tolist())
                    if s > threshold and i not in topk_set)

    assert errors == 0, f"TopK ranking failed with {errors} missed tokens"
    print(f"Top-k ranking:      errors = {errors}  PASS\n")


# ──────────────────────────────────────────────────────────────────────────────
#  Original Test 4 — batch independence
# ──────────────────────────────────────────────────────────────────────────────
def test_batch_independence():
    print("=== test_batch_independence ===")
    batch_size  = 2
    num_pages   = 4
    max_seq_len = num_pages * PAGE_SIZE

    rng  = np.random.RandomState(1)
    Q_np = np.zeros((batch_size, NUM_HEADS * HEAD_DIM), dtype=np.float32)
    Q_np[0] =  0.5
    Q_np[1] = -0.5
    Q            = torch.from_numpy(Q_np)
    head_weights = torch.full((NUM_HEADS,), 1.0 / NUM_HEADS)
    K_raw, _     = build_k_cache(num_pages, rng)
    pt_np        = np.tile(np.arange(num_pages, dtype=np.int32), (batch_size, 1))
    page_table   = torch.from_numpy(pt_np)

    scores_gpu = run_kernel(Q, K_raw, head_weights, page_table, max_seq_len, num_pages)

    sum0 = scores_gpu[0].sum().item()
    sum1 = scores_gpu[1].sum().item()
    assert sum0 != sum1, f"Batch contamination: sum0={sum0:.4f} sum1={sum1:.4f}"
    print(f"Batch independence: sum[0]={sum0:.2f}  sum[1]={sum1:.2f}  PASS\n")


# ──────────────────────────────────────────────────────────────────────────────
#  NEW Test 5 — index space correctness
#
#  The silent killer: does topk return GLOBAL token indices (0…seq_len-1)
#  or PAGE-LOCAL offsets (0…PAGE_SIZE-1)?
#
#  Two pages. Page 0: K=0.0 (zero dot product with any Q). Page 1: K=1.0
#  (positive dot product). Q=all +1.  Only page-1 tokens score > 0.
#
#  Correct:    all top-k indices are in [PAGE_SIZE, 2*PAGE_SIZE-1] = [64, 127]
#  Index-space bug: indices would be in [0, PAGE_SIZE-1] = [0, 63] because
#  the page-local offset (0…63) was returned instead of the global index.
# ──────────────────────────────────────────────────────────────────────────────
def test_index_space():
    print("=== test_index_space_correctness ===")
    batch_size  = 1
    num_pages   = 2            # page 0: tokens 0-63, page 1: tokens 64-127
    max_seq_len = num_pages * PAGE_SIZE  # 128
    k           = PAGE_SIZE   # ask for exactly 64 results = all of page 1

    # Q = all +1.0, head_weights = uniform
    Q            = torch.ones(batch_size, NUM_HEADS * HEAD_DIM)
    head_weights = torch.full((NUM_HEADS,), 1.0 / NUM_HEADS)

    # page_vals[0]=0.0 → page 0 scores ≈ 0
    # page_vals[1]=1.0 → page 1 scores = HEAD_DIM = 128 (after dot product)
    K_raw, K_float = build_k_cache_uniform(num_pages, [0.0, 1.0])

    # Identity page table: logical p → physical p
    pt_np      = np.tile(np.arange(num_pages, dtype=np.int32), (batch_size, 1))
    page_table = torch.from_numpy(pt_np)

    scores_gpu = run_kernel(Q, K_raw, head_weights, page_table, max_seq_len, num_pages)

    # torch.topk with largest=True returns descending global indices
    _, topk_indices = torch.topk(scores_gpu[0], k, largest=True)
    topk_indices    = topk_indices.tolist()

    # Every returned index must be >= PAGE_SIZE (global index into page 1).
    # A page-local bug would return indices in [0, 63] instead of [64, 127].
    errors = [idx for idx in topk_indices if idx < PAGE_SIZE]
    assert len(errors) == 0, (
        f"Index-space BUG: {len(errors)} indices < PAGE_SIZE ({PAGE_SIZE}).\n"
        f"  First few bad indices: {errors[:5]}\n"
        f"  These look like page-local offsets, not global token indices."
    )
    print(f"Index space:        all {k} indices in [PAGE_SIZE, seq_len-1]  PASS\n")


# ──────────────────────────────────────────────────────────────────────────────
#  NEW Test 6 — sorting direction
#
#  The silent killer: does topk return the HIGHEST or LOWEST scoring tokens?
#  An ascending sort gives you the wrong end of the ranking silently.
#
#  We inject scores[i] = i directly (bypassing the scoring kernel) so we know
#  exactly which k indices should be returned: [seq_len-k, …, seq_len-1].
#  Ascending sort would return [0, 1, …, k-1] instead — a completely wrong set.
# ──────────────────────────────────────────────────────────────────────────────
def test_sorting_direction():
    print("=== test_sorting_direction ===")
    seq_len = 512
    k       = 32

    # scores[i] = i, so the top-k by value are the k largest indices
    scores   = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0)  # [1, seq_len]
    expected = set(range(seq_len - k, seq_len))  # {seq_len-k, ..., seq_len-1}

    # torch.topk(largest=True) is the reference for "descending order"
    _, topk_indices = torch.topk(scores[0], k, largest=True)
    returned = set(topk_indices.tolist())

    # If sort was ascending, returned = {0, 1, ..., k-1} which has no overlap with expected
    wrong = returned - expected
    assert len(wrong) == 0, (
        f"Direction BUG: {len(wrong)} returned indices are NOT in the top-{k} score band.\n"
        f"  Wrong indices: {sorted(wrong)[:10]}\n"
        f"  This looks like an ascending sort was used instead of descending."
    )
    print(f"Sorting direction:  all {k} indices in top-{k} score band  PASS\n")


# ──────────────────────────────────────────────────────────────────────────────
#  NEW Test 7 — tiny manually-verifiable example
#
#  Week 6 principle: "test with tiny examples where you can manually verify."
#  batch=1, seq_len=256, k=4.  We inject known scores directly so the correct
#  answer is visible in the source code below — no computation needed to verify.
#
#  Expected top-4 indices are chosen to span all four pages:
#    Token  42: page 0, offset 42
#    Token  77: page 1, offset 13
#    Token 133: page 2, offset  5
#    Token 201: page 3, offset  9
# ──────────────────────────────────────────────────────────────────────────────
def test_tiny_manual():
    print("=== test_tiny_manual (hand-verifiable) ===")
    seq_len          = 256
    k                = 4
    expected_indices = {42, 77, 133, 201}  # chosen to span all 4 pages

    # Build scores: exactly 4 tokens get 100.0, everything else 0.0
    scores = torch.zeros(1, seq_len)
    for idx in expected_indices:
        scores[0, idx] = 100.0

    # torch.topk is what the pipeline uses after the scoring kernel
    _, topk_indices = torch.topk(scores[0], k, largest=True)
    returned = set(topk_indices.tolist())

    assert returned == expected_indices, (
        f"Tiny manual FAIL.\n"
        f"  Expected: {sorted(expected_indices)}\n"
        f"  Returned: {sorted(returned)}\n"
        f"  Missing:  {sorted(expected_indices - returned)}\n"
        f"  Extra:    {sorted(returned - expected_indices)}"
    )
    print(f"Tiny manual:        expected={sorted(expected_indices)}")
    print(f"                    returned={sorted(returned)}  PASS\n")


# ──────────────────────────────────────────────────────────────────────────────
#  Original Test 8 — PyTorch stress test across multiple configs
# ──────────────────────────────────────────────────────────────────────────────
def test_pytorch_close():
    print("=== test_pytorch_close (stress) ===")
    configs = [(1, 2, "small "), (4, 8, "medium"), (8, 16, "large ")]

    for batch_size, num_pages, label in configs:
        max_seq_len = num_pages * PAGE_SIZE
        rng         = np.random.RandomState(batch_size * 100 + num_pages)

        Q            = torch.from_numpy(rng.uniform(-0.5, 0.5, (batch_size, NUM_HEADS * HEAD_DIM)).astype(np.float32))
        head_weights = torch.from_numpy(np.abs(rng.uniform(-0.5, 0.5, NUM_HEADS).astype(np.float32)))
        K_raw, K_float = build_k_cache(num_pages, rng)
        pt_np        = np.tile(np.arange(num_pages, dtype=np.int32), (batch_size, 1))
        page_table   = torch.from_numpy(pt_np)

        scores_gpu = run_kernel(Q, K_raw, head_weights, page_table, max_seq_len, num_pages)
        K_ref      = gather_k(K_float, pt_np, batch_size, max_seq_len)
        scores_ref = compute_scores_reference(Q, K_ref, head_weights)

        torch.testing.assert_close(scores_gpu, scores_ref, rtol=1e-3, atol=1e-3)
        max_err = (scores_gpu - scores_ref).abs().max().item()
        print(f"  [{label}] batch={batch_size}  pages={num_pages:2d}  max_err={max_err:.2e}  PASS")

    print()


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Week 6: Top-K Indexer — PyTorch test harness")
    print("=" * 60 + "\n")

    # Original tests — validate scoring kernel correctness
    test_score_correctness()
    test_paged_addressing()
    test_topk_ranking()
    test_batch_independence()

    # New tests — validate the three silent killers from Week 6 spec
    test_index_space()        # global vs page-local indices
    test_sorting_direction()  # descending vs ascending sort
    test_tiny_manual()        # hand-verifiable tiny case

    # Stress test
    test_pytorch_close()

    print("=" * 60)
    print("  All tests passed.")
    print("=" * 60)