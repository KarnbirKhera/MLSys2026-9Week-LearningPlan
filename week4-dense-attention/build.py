import os
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))

attention_cuda = load(
    name="attention_cuda",
    sources=[os.path.join(HERE, "attention_kernels.cu")],
    extra_cuda_cflags=["-O2", "-lineinfo", "-DWITH_TORCH"],
    build_directory=HERE,   # put attention_cuda.so right here
    verbose=True,
)
print("Build complete — run: python3 test_attention.py")