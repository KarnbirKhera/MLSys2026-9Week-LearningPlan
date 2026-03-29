NVCC  = nvcc
FLAGS = -O2 -lineinfo -arch=sm_89

# ─────────────────────────────────────────────────────────────────────────────
#  Build targets
# ─────────────────────────────────────────────────────────────────────────────

all: sparse_attention

# cuda_bf16.h and cuda_fp8.h require -std=c++17 for some host-side helpers.
sparse_attention: sparse_attention.cu
	$(NVCC) $(FLAGS) -std=c++17 -o sparse_attention sparse_attention.cu


# ─────────────────────────────────────────────────────────────────────────────
#  Test targets
# ─────────────────────────────────────────────────────────────────────────────

# Run the C++ test harness (correctness + paged scatter)
run: sparse_attention
	./sparse_attention

# Run the PyTorch test harness (load_inline compiles the kernel at import time —
# no separate build step needed for the Python side)
pytorch:
	python3 test_sparse_attention.py

# Run both test harnesses back to back
test: run pytorch


# ─────────────────────────────────────────────────────────────────────────────
#  Profile targets
# ─────────────────────────────────────────────────────────────────────────────

profile: sparse_attention
	ncu --set full --import-source yes --force-overwrite -o sparse_profile ./sparse_attention
	ncu-ui sparse_profile.ncu-rep &

# ─────────────────────────────────────────────────────────────────────────────
#  Utility
# ─────────────────────────────────────────────────────────────────────────────

build: clean all

clean:
	rm -f sparse_attention sparse_profile.ncu-rep

.PHONY: all run pytorch test profile build clean
