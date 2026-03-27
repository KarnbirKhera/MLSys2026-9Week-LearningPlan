NVCC  = nvcc
FLAGS = -O2 -lineinfo -arch=sm_89

# ─────────────────────────────────────────────────────────────────────────────
#  Build targets
# ─────────────────────────────────────────────────────────────────────────────

all: topk_indexer

# Thrust is header-only and ships with the CUDA toolkit — no extra link flags needed.
# -std=c++17 is required for <algorithm> and thrust in the same translation unit.
topk_indexer: topk_indexer.cu
	$(NVCC) $(FLAGS) -std=c++17 -o topk_indexer topk_indexer.cu


# ─────────────────────────────────────────────────────────────────────────────
#  Test targets
# ─────────────────────────────────────────────────────────────────────────────

# Run the C++ test harness (original 4 tests + 3 new silent-killer tests)
run: topk_indexer
	./topk_indexer

# Run the PyTorch test harness (load_inline compiles the kernel at import time —
# no separate build step needed for the Python side)
pytorch:
	python3 test_topk_scorer.py

# Run both test harnesses back to back
test: run pytest


# ─────────────────────────────────────────────────────────────────────────────
#  Profile targets
# ─────────────────────────────────────────────────────────────────────────────

profile: topk_indexer
	ncu --set full --import-source yes --force-overwrite -o topk_profile ./topk_indexer
	ncu-ui topk_profile.ncu-rep &

# ─────────────────────────────────────────────────────────────────────────────
#  Utility
# ─────────────────────────────────────────────────────────────────────────────

build: clean all

clean:
	rm -f topk_indexer topk_profile.ncu-rep

.PHONY: all run pytest test profile build clean