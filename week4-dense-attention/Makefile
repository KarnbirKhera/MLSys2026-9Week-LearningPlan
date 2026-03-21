NVCC   = nvcc
FLAGS  = -O2 -lineinfo
PYTHON = $(shell python3 -c "import torch" 2>/dev/null && echo python3 || \
         find /usr /opt $${HOME} -name "python3" 2>/dev/null | \
         xargs -I{} sh -c '{} -c "import torch" 2>/dev/null && echo {}' | head -1)

all: attention

attention: attention_kernels.cu
	$(NVCC) $(FLAGS) -o attention attention_kernels.cu

run: attention
	./attention

pytorch: attention_kernels.cu
	$(PYTHON) build.py && $(PYTHON) test_attention.py

profile: attention
	ncu --set full --import-source yes --force-overwrite -o attention_profile ./attention
	ncu-ui attention_profile.ncu-rep &

build: clean all

clean:
	rm -f attention attention_profile.ncu-rep

.PHONY: all run pytorch profile build clean