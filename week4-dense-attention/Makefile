NVCC = nvcc
FLAGS = -O2 -lineinfo

all: attention

attention: qkv_attention_standalone_1.cu
	$(NVCC) $(FLAGS) -o attention qkv_attention_standalone_1.cu

run: attention
	./attention

profile: attention
	ncu --set full --import-source yes --force-overwrite -o attention_profile ./attention
	ncu-ui attention_profile.ncu-rep &

build: clean all

clean:
	rm -f attention attention_profile.ncu-rep