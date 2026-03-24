NVCC   = nvcc
FLAGS  = -O2 -lineinfo
PYTHON = $(shell python3 -c "import torch" 2>/dev/null && echo python3 || \
         find /usr /opt $${HOME} -name "python3" 2>/dev/null | \
         xargs -I{} sh -c '{} -c "import torch" 2>/dev/null && echo {}' | head -1)

all: paged-kv-mla

paged-kv-mla: paged-kv-mla.cu
	$(NVCC) $(FLAGS) -o paged-kv-mla paged-kv-mla.cu

run: paged-kv-mla
	./paged-kv-mla

pytorch: paged-kv-mla.cu
	$(PYTHON) build_paged.py && $(PYTHON) test_paged_attention.py

pytorch-mla: paged-kv-mla.cu
	$(PYTHON) build_paged.py && $(PYTHON) test_mla_attention.py

profile: paged-kv-mla
	ncu --set full --import-source yes --force-overwrite -o paged_profile ./paged-kv-mla
	ncu-ui paged_profile.ncu-rep &

clean:
	rm -f paged-kv-mla paged_profile.ncu-rep mla_profile.ncu-rep

.PHONY: all run pytorch pytorch-mla profile-paged profile-mla clean

pytorch-all: paged-kv-mla.cu
	$(PYTHON) build_paged.py && $(PYTHON) test_paged_attention.py && $(PYTHON) test_mla_attention.py