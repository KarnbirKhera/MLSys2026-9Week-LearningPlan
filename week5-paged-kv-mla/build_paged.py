import os
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))

paged_attention_cuda = load(
    name="paged_attention_cuda",
    sources=[os.path.join(HERE, "paged-kv-mla.cu")],
    extra_cuda_cflags=["-O2", "-lineinfo", "-DWITH_TORCH"],
    build_directory=HERE,
    verbose=True,
)
print("Build complete — run: python3 test_paged_attention.py")