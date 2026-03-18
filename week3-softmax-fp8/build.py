from torch.utils.cpp_extension import load

softmax_cuda = load(
    name="softmax_cuda",
    sources=["softmax.cu"],
    extra_cuda_cflags=["-O2"],
    verbose=True
)
