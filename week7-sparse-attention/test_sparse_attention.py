"""
test_sparse_attention.py  —  PyTorch test harness for sparse_attention_kernel
==============================================================================
Mirrors both C++ tests and adds:
  - A pure PyTorch reference implementation to validate against
  - LSE output verification
  - Batch independence check

Run with:
  python3 test_sparse_attention.py
"""

import numpy as np
import torch
import torch.testing
from torch.utils.cpp_extension import load_inline


# ──────────────────────────────────────────────────────────────────────────────
#  Sanity checks
# ──────────────────────────────────────────────────────────────────────────────
assert torch.cuda.is_available(), "CUDA GPU required"
assert hasattr(torch, "float8_e4m3fn"), "Upgrade to PyTorch >= 2.1 for float8 support"


# ──────────────────────────────────────────────────────────────────────────────
#  Constants — must match the kernel #defines exactly
# ──────────────────────────────────────────────────────────────────────────────
NUM_HEADS        = 64
HEAD_DIM         = 128
PAGE_SIZE        = 64
TOP_K            = 2048
TILE_K           = 64
PAGE_DATA_BYTES  = PAGE_SIZE * HEAD_DIM       # 8192 bytes: FP8 key/value data
PAGE_SCALE_BYTES = PAGE_SIZE * 4              # 256 bytes: float32 scale factors
PAGE_STRIDE      = PAGE_DATA_BYTES + PAGE_SCALE_BYTES  # 8448 bytes total per page
QK_SCALE         = 1.0 / (HEAD_DIM ** 0.5)   # 1/sqrt(128) ≈ 0.0884


# ──────────────────────────────────────────────────────────────────────────────
#  CUDA kernel source — identical to the competition kernel.
#  The wrapper function translates PyTorch tensors into raw pointers so the
#  kernel sees exactly the same types it expects from the C++ test harness.
# ──────────────────────────────────────────────────────────────────────────────
CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <float.h>

#define NUM_HEADS         64
#define HEAD_DIM          128
#define PAGE_SIZE         64
#define TOP_K             2048
#define TILE_K            64
#define PAGE_DATA_BYTES   (PAGE_SIZE * HEAD_DIM)
#define PAGE_SCALE_BYTES  (PAGE_SIZE * (int)sizeof(float))
#define PAGE_STRIDE       (PAGE_DATA_BYTES + PAGE_SCALE_BYTES)

__global__ void sparse_attention_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const uint8_t*       __restrict__ K_cache,
    const uint8_t*       __restrict__ V_cache,
    const int*           __restrict__ page_table,
    const int*           __restrict__ topk_indices,
    __nv_bfloat16*       __restrict__ out,
    float*               __restrict__ lse_out,
    int   max_pages,
    float qk_scale
) {
    int batch = blockIdx.x;
    int head  = blockIdx.y;
    int tid   = threadIdx.x;

    __shared__ float smem_q[HEAD_DIM];
    __shared__ float smem_kv[TILE_K][HEAD_DIM + 1];
    __shared__ float smem_dot[TILE_K];
    __shared__ float smem_reduce[HEAD_DIM];

    // Load this head's Q vector from bfloat16 global memory into float32 smem.
    // Three-term address: batch × (NUM_HEADS × HEAD_DIM) + head × HEAD_DIM + tid
    smem_q[tid] = __bfloat162float(
        Q[batch * NUM_HEADS * HEAD_DIM + head * HEAD_DIM + tid]
    );
    __syncthreads();

    float running_max   = -FLT_MAX;
    float running_denom = 0.0f;
    float out_acc       = 0.0f;

    int num_tiles = (TOP_K + TILE_K - 1) / TILE_K;

    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_base = tile * TILE_K;
        int tile_len  = min(TILE_K, TOP_K - tile_base);

        // Load tile of K vectors from paged FP8 cache into smem_kv
        for (int tok = 0; tok < tile_len; tok++) {
            int global_tok = topk_indices[batch * TOP_K + tile_base + tok];
            int log_page   = global_tok / PAGE_SIZE;
            int page_off   = global_tok % PAGE_SIZE;
            int phys_page  = page_table[batch * max_pages + log_page];

            const uint8_t* k_page = K_cache + (size_t)phys_page * PAGE_STRIDE;
            float k_scale = *reinterpret_cast<const float*>(
                k_page + PAGE_DATA_BYTES + page_off * sizeof(float));

            __nv_fp8_e4m3 fp8_k;
            memcpy(&fp8_k, k_page + page_off * HEAD_DIM + tid, 1);
            smem_kv[tok][tid] = __half2float((__half)fp8_k) * k_scale;
        }
        __syncthreads();

        // Compute Q·K dot products, one per token in the tile.
        // Tree reduction over HEAD_DIM collapses that dimension into one scalar.
        for (int tok = 0; tok < tile_len; tok++) {
            smem_reduce[tid] = smem_q[tid] * smem_kv[tok][tid];
            __syncthreads();

            for (int s = HEAD_DIM / 2; s > 0; s >>= 1) {
                if (tid < s) smem_reduce[tid] += smem_reduce[tid + s];
                __syncthreads();
            }

            if (tid == 0) smem_dot[tok] = smem_reduce[0] * qk_scale;
            __syncthreads();
        }

        // Find tile max for numerically stable softmax
        float tile_max = -FLT_MAX;
        if (tid == 0) {
            for (int tok = 0; tok < tile_len; tok++)
                tile_max = fmaxf(tile_max, smem_dot[tok]);
            smem_reduce[0] = tile_max;
        }
        __syncthreads();
        tile_max = smem_reduce[0];

        float new_max = fmaxf(running_max, tile_max);
        float rescale = expf(running_max - new_max);

        // Accumulate tile denominator and rescale running stats
        float tile_denom = 0.0f;
        if (tid == 0) {
            for (int tok = 0; tok < tile_len; tok++)
                tile_denom += expf(smem_dot[tok] - new_max);
            smem_reduce[0] = tile_denom;
        }
        __syncthreads();
        tile_denom = smem_reduce[0];

        out_acc        *= rescale;
        running_denom   = running_denom * rescale + tile_denom;
        running_max     = new_max;

        // Load tile of V vectors and accumulate weighted output
        for (int tok = 0; tok < tile_len; tok++) {
            int global_tok = topk_indices[batch * TOP_K + tile_base + tok];
            int log_page   = global_tok / PAGE_SIZE;
            int page_off   = global_tok % PAGE_SIZE;
            int phys_page  = page_table[batch * max_pages + log_page];

            const uint8_t* v_page = V_cache + (size_t)phys_page * PAGE_STRIDE;
            float v_scale = *reinterpret_cast<const float*>(
                v_page + PAGE_DATA_BYTES + page_off * sizeof(float));

            __nv_fp8_e4m3 fp8_v;
            memcpy(&fp8_v, v_page + page_off * HEAD_DIM + tid, 1);
            smem_kv[tok][tid] = __half2float((__half)fp8_v) * v_scale;
        }
        __syncthreads();

        for (int tok = 0; tok < tile_len; tok++) {
            float w = expf(smem_dot[tok] - running_max);
            out_acc += w * smem_kv[tok][tid];
        }
        __syncthreads();
    }

    // Normalize and write output
    out_acc /= running_denom;
    out[batch * NUM_HEADS * HEAD_DIM + head * HEAD_DIM + tid] =
        __float2bfloat16(out_acc);

    // Thread 0 writes log-sum-exp for this (batch, head)
    if (tid == 0)
        lse_out[batch * NUM_HEADS + head] = running_max + logf(running_denom);
}

// ── PyTorch-facing wrapper ───────────────────────────────────────────────────
std::vector<torch::Tensor> sparse_attention_cuda(
    torch::Tensor Q,            // [batch, num_heads, head_dim]  bfloat16
    torch::Tensor K_cache,      // raw byte buffer               uint8
    torch::Tensor V_cache,      // raw byte buffer               uint8
    torch::Tensor page_table,   // [batch, max_pages]            int32
    torch::Tensor topk_indices, // [batch, TOP_K]                int32
    int max_pages,
    float qk_scale
) {
    const int batch_size = Q.size(0);
    auto out = torch::zeros_like(Q);
    auto lse = torch::zeros(
        {batch_size, NUM_HEADS},
        torch::TensorOptions().dtype(torch::kFloat32).device(Q.device())
    );

    dim3 grid(batch_size, NUM_HEADS);
    dim3 block(HEAD_DIM);

    sparse_attention_kernel<<<grid, block>>>(
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
        K_cache.data_ptr<uint8_t>(),
        V_cache.data_ptr<uint8_t>(),
        page_table.data_ptr<int>(),
        topk_indices.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        lse.data_ptr<float>(),
        max_pages,
        qk_scale
    );
    cudaDeviceSynchronize();
    return {out, lse};
}
"""

CPP_SOURCE = """
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> sparse_attention_cuda(
    torch::Tensor Q,
    torch::Tensor K_cache,
    torch::Tensor V_cache,
    torch::Tensor page_table,
    torch::Tensor topk_indices,
    int max_pages,
    float qk_scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_attention", &sparse_attention_cuda,
          "Sparse attention with paged FP8 KV cache (CUDA)");
}
"""

print("Compiling CUDA kernel...")
attn = load_inline(
    name="sparse_attn",
    cpp_sources=[CPP_SOURCE],
    cuda_sources=[CUDA_SOURCE],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
    verbose=False,
)
print("Ready.\n")


# ──────────────────────────────────────────────────────────────────────────────
#  FP8 helpers
#  The kernel stores K/V as FP8 E4M3 and dequantizes via:
#    fp8 → half → float32
#  Both the GPU kernel and the Python reference must go through the same chain
#  or numerical comparison will show a spurious error from quantization mismatch.
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
#  KV cache builder
#  Memory layout per physical page (PAGE_STRIDE bytes total):
#    [FP8 data:   PAGE_SIZE × HEAD_DIM bytes ]  ← PAGE_DATA_BYTES
#    [float scales: PAGE_SIZE × 4  bytes     ]  ← PAGE_SCALE_BYTES
#  Returns (raw_bytes, kv_float_dequantized) so both the GPU and the reference
#  implementation work from the same quantized values.
# ──────────────────────────────────────────────────────────────────────────────
def build_kv_cache(num_phys_pages: int, rng: np.random.RandomState):
    vals      = rng.uniform(-0.1, 0.1, (num_phys_pages, PAGE_SIZE, HEAD_DIM)).astype(np.float32)
    fp8_bytes = floats_to_fp8_bytes(vals)
    kv_float  = fp8_bytes_to_floats(fp8_bytes)   # what the kernel actually computes with

    raw    = np.zeros(num_phys_pages * PAGE_STRIDE, dtype=np.uint8)
    scales = np.ones(PAGE_SIZE, dtype=np.float32).view(np.uint8)  # all scales = 1.0

    for pg in range(num_phys_pages):
        base = pg * PAGE_STRIDE
        raw[base : base + PAGE_DATA_BYTES]  = fp8_bytes[pg].ravel()
        raw[base + PAGE_DATA_BYTES : base + PAGE_STRIDE] = scales

    return raw, kv_float


# ──────────────────────────────────────────────────────────────────────────────
#  PyTorch reference implementation
#  Computes the same sparse attention as the kernel:
#    1. Gather the top-k K and V vectors using the page table
#    2. Compute scaled dot products: Q · K^T
#    3. Online softmax over the top-k scores
#    4. Weighted sum of V vectors
#    5. Return output and log-sum-exp
# ──────────────────────────────────────────────────────────────────────────────
def gather_kv(kv_float: np.ndarray, page_table_np: np.ndarray,
              topk_np: np.ndarray, batch_size: int) -> torch.Tensor:
    """
    Gather the top-k K or V vectors for each (batch) using the page table.
    Returns a tensor of shape [batch, TOP_K, HEAD_DIM].
    """
    result = np.zeros((batch_size, TOP_K, HEAD_DIM), dtype=np.float32)
    for b in range(batch_size):
        for ki in range(TOP_K):
            global_tok = topk_np[b, ki]
            lp         = global_tok // PAGE_SIZE
            off        = global_tok %  PAGE_SIZE
            phys       = page_table_np[b, lp]
            result[b, ki] = kv_float[phys, off]
    return torch.from_numpy(result)


def sparse_attention_reference(
    Q_fp32:       torch.Tensor,   # [batch, num_heads, head_dim]  float32
    K_gathered:   torch.Tensor,   # [batch, top_k,     head_dim]  float32
    V_gathered:   torch.Tensor,   # [batch, top_k,     head_dim]  float32
    qk_scale:     float,
):
    """
    Pure PyTorch sparse attention — equivalent to the CUDA kernel but readable.
    Q has shape [batch, heads, dim]. K and V have shape [batch, top_k, dim]
    (already gathered from the page table so all top_k tokens are contiguous).
    The attention is computed independently per head, which matches the kernel
    where each (batch, head) block computes one output row.
    """
    batch, heads, dim = Q_fp32.shape

    # dots[b, h, ki] = Q[b,h,:] · K[b,ki,:] * qk_scale
    # einsum: for each batch and head, dot against every top-k key vector
    dots = torch.einsum("bhd,bkd->bhk", Q_fp32, K_gathered) * qk_scale  # [batch, heads, top_k]

    # Numerically stable softmax: subtract row max before exp
    dots_max  = dots.max(dim=-1, keepdim=True).values
    exp_dots  = torch.exp(dots - dots_max)
    denom     = exp_dots.sum(dim=-1, keepdim=True)
    weights   = exp_dots / denom                    # [batch, heads, top_k]

    # Weighted sum of V: out[b,h,d] = Σ_k weights[b,h,k] * V[b,k,d]
    out = torch.einsum("bhk,bkd->bhd", weights, V_gathered)  # [batch, heads, dim]

    # Log-sum-exp: log(Σ exp(dot - max)) + max = log(denom) + max
    lse = torch.log(denom.squeeze(-1)) + dots_max.squeeze(-1)  # [batch, heads]

    return out, lse


# ──────────────────────────────────────────────────────────────────────────────
#  Kernel launcher — moves data to GPU, runs kernel, returns CPU tensors
# ──────────────────────────────────────────────────────────────────────────────
def run_kernel(Q_bf16, K_raw, V_raw, page_table, topk_indices, max_pages, device="cuda"):
    out_bf16, lse = attn.sparse_attention(
        Q_bf16.to(device),
        torch.from_numpy(K_raw).to(device),
        torch.from_numpy(V_raw).to(device),
        page_table.to(device),
        topk_indices.to(device),
        max_pages,
        QK_SCALE,
    )
    # Convert bfloat16 output back to float32 for comparison
    return out_bf16.float().cpu(), lse.cpu()


# ──────────────────────────────────────────────────────────────────────────────
#  Test 1 — basic correctness
#  Mirrors test_sparse_attention_correctness from the C++ harness.
#  Identity page table, sequential top-k indices (0…TOP_K-1), random Q/K/V.
# ──────────────────────────────────────────────────────────────────────────────
def test_correctness():
    print("=== test_sparse_attention_correctness ===")
    batch_size = 2
    num_pages  = (TOP_K * 2) // PAGE_SIZE   # 64 pages — more than needed so bounds are exercised

    rng = np.random.RandomState(42)

    Q_fp32 = torch.from_numpy(rng.uniform(-0.1, 0.1, (batch_size, NUM_HEADS, HEAD_DIM)).astype(np.float32))
    Q_bf16 = Q_fp32.to(torch.bfloat16)

    K_raw, K_float = build_kv_cache(num_pages, rng)
    V_raw, V_float = build_kv_cache(num_pages, rng)

    # Identity page table: logical page p → physical page p
    pt_np      = np.tile(np.arange(num_pages, dtype=np.int32), (batch_size, 1))
    page_table = torch.from_numpy(pt_np)

    # Sequential top-k: token 0, 1, 2, …, TOP_K-1
    topk_np      = np.tile(np.arange(TOP_K, dtype=np.int32), (batch_size, 1))
    topk_indices = torch.from_numpy(topk_np)

    # GPU kernel output
    out_gpu, lse_gpu = run_kernel(Q_bf16, K_raw, V_raw, page_table, topk_indices, num_pages)

    # PyTorch reference using the same dequantized values the kernel sees
    K_ref = gather_kv(K_float, pt_np, topk_np, batch_size)
    V_ref = gather_kv(V_float, pt_np, topk_np, batch_size)
    out_ref, lse_ref = sparse_attention_reference(Q_fp32, K_ref, V_ref, QK_SCALE)

    out_err = (out_gpu - out_ref).abs().max().item()
    lse_err = (lse_gpu - lse_ref).abs().max().item()

    # Tolerance is 1e-2 for output (bfloat16 has ~7 bits of mantissa, so some
    # rounding error relative to float32 reference is expected) and 1e-3 for LSE
    # which is computed in float32 throughout.
    assert out_err < 1e-2, f"Output error too large: {out_err:.2e}"
    assert lse_err < 1e-3, f"LSE error too large: {lse_err:.2e}"
    print(f"Output error: max_err = {out_err:.2e}  PASS")
    print(f"LSE error:    max_err = {lse_err:.2e}  PASS\n")


# ──────────────────────────────────────────────────────────────────────────────
#  Test 2 — paged scatter (shuffled page table)
#  Mirrors test_paged_scatter from the C++ harness.
#  Reversed page table so logical page p maps to physical page (num_pages-1-p).
#  This verifies the page table indirection is being applied correctly.
# ──────────────────────────────────────────────────────────────────────────────
def test_paged_scatter():
    print("=== test_paged_scatter (shuffled page table) ===")
    batch_size = 1
    num_pages  = TOP_K // PAGE_SIZE   # 32 pages — exactly covers TOP_K tokens

    rng = np.random.RandomState(7)

    Q_fp32 = torch.from_numpy(rng.uniform(-0.1, 0.1, (batch_size, NUM_HEADS, HEAD_DIM)).astype(np.float32))
    Q_bf16 = Q_fp32.to(torch.bfloat16)

    K_raw, K_float = build_kv_cache(num_pages, rng)
    V_raw, V_float = build_kv_cache(num_pages, rng)

    # Reversed page table — the silent killer: wrong page table = silently wrong output
    pt_np      = np.array([[num_pages - 1 - p for p in range(num_pages)]], dtype=np.int32)
    page_table = torch.from_numpy(pt_np)

    topk_np      = np.tile(np.arange(TOP_K, dtype=np.int32), (batch_size, 1))
    topk_indices = torch.from_numpy(topk_np)

    out_gpu, lse_gpu = run_kernel(Q_bf16, K_raw, V_raw, page_table, topk_indices, num_pages)

    K_ref = gather_kv(K_float, pt_np, topk_np, batch_size)
    V_ref = gather_kv(V_float, pt_np, topk_np, batch_size)
    out_ref, lse_ref = sparse_attention_reference(Q_fp32, K_ref, V_ref, QK_SCALE)

    out_err = (out_gpu - out_ref).abs().max().item()
    lse_err = (lse_gpu - lse_ref).abs().max().item()

    assert out_err < 1e-2, f"Output error too large: {out_err:.2e}"
    assert lse_err < 1e-3, f"LSE error too large: {lse_err:.2e}"
    print(f"Output error: max_err = {out_err:.2e}  PASS")
    print(f"LSE error:    max_err = {lse_err:.2e}  PASS\n")


# ──────────────────────────────────────────────────────────────────────────────
#  Test 3 — batch independence
#  Two batches with opposite-sign Q vectors attend to the same K and V cache.
#  Their outputs must differ — if they're identical, the batches are sharing
#  state (e.g., a smem variable that wasn't properly scoped per block).
# ──────────────────────────────────────────────────────────────────────────────
def test_batch_independence():
    print("=== test_batch_independence ===")
    batch_size = 2
    num_pages  = TOP_K // PAGE_SIZE

    rng = np.random.RandomState(1)

    # Batch 0: Q = +0.1 everywhere, Batch 1: Q = -0.1 everywhere
    Q_np       = np.zeros((batch_size, NUM_HEADS, HEAD_DIM), dtype=np.float32)
    Q_np[0]    =  0.1
    Q_np[1]    = -0.1
    Q_fp32     = torch.from_numpy(Q_np)
    Q_bf16     = Q_fp32.to(torch.bfloat16)

    K_raw, _ = build_kv_cache(num_pages, rng)
    V_raw, _ = build_kv_cache(num_pages, rng)

    pt_np      = np.tile(np.arange(num_pages, dtype=np.int32), (batch_size, 1))
    topk_np    = np.tile(np.arange(TOP_K,     dtype=np.int32), (batch_size, 1))
    page_table = torch.from_numpy(pt_np)
    topk_idx   = torch.from_numpy(topk_np)

    out_gpu, _ = run_kernel(Q_bf16, K_raw, V_raw, page_table, topk_idx, num_pages)

    sum0 = out_gpu[0].sum().item()
    sum1 = out_gpu[1].sum().item()
    assert sum0 != sum1, f"Batch contamination detected: sum0={sum0:.4f}  sum1={sum1:.4f}"
    print(f"Batch independence: sum[0]={sum0:.4f}  sum[1]={sum1:.4f}  PASS\n")


# ──────────────────────────────────────────────────────────────────────────────
#  Test 4 — LSE numerical sanity
#  The log-sum-exp value must satisfy a basic mathematical property:
#    exp(lse) == Σ exp(dot_i)  for all i in the top-k set
#  This is independent of the output values and catches bugs where running_max
#  or running_denom are computed correctly but lse_out is written incorrectly
#  (e.g., missing the + logf(running_denom) term).
# ──────────────────────────────────────────────────────────────────────────────
def test_lse_sanity():
    print("=== test_lse_sanity ===")
    batch_size = 1
    num_pages  = TOP_K // PAGE_SIZE

    rng = np.random.RandomState(3)

    Q_fp32 = torch.from_numpy(rng.uniform(-0.1, 0.1, (batch_size, NUM_HEADS, HEAD_DIM)).astype(np.float32))
    Q_bf16 = Q_fp32.to(torch.bfloat16)

    K_raw, K_float = build_kv_cache(num_pages, rng)
    V_raw, V_float = build_kv_cache(num_pages, rng)

    pt_np      = np.tile(np.arange(num_pages, dtype=np.int32), (batch_size, 1))
    topk_np    = np.tile(np.arange(TOP_K,     dtype=np.int32), (batch_size, 1))
    page_table = torch.from_numpy(pt_np)
    topk_idx   = torch.from_numpy(topk_np)

    _, lse_gpu = run_kernel(Q_bf16, K_raw, V_raw, page_table, topk_idx, num_pages)

    # Compute what lse SHOULD be from the reference
    K_ref = gather_kv(K_float, pt_np, topk_np, batch_size)
    V_ref = gather_kv(V_float, pt_np, topk_np, batch_size)
    _, lse_ref = sparse_attention_reference(Q_fp32, K_ref, V_ref, QK_SCALE)

    lse_err = (lse_gpu - lse_ref).abs().max().item()
    assert lse_err < 1e-3, f"LSE sanity failed: max_err = {lse_err:.2e}"
    print(f"LSE sanity:         max_err = {lse_err:.2e}  PASS\n")


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Week 7: Sparse Attention — PyTorch test harness")
    print("=" * 60 + "\n")

    test_correctness()
    test_paged_scatter()
    test_batch_independence()
    test_lse_sanity()

    print("=" * 60)
    print("  All tests passed.")
    print("=" * 60)