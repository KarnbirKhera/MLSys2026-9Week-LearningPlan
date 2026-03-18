NVCC  = nvcc
FLAGS = -O2 -lineinfo
 
all: softmax
 
softmax: softmax.cu
	$(NVCC) $(FLAGS) -o softmax softmax.cu
 
run: softmax
	./softmax
 
profile: softmax
	ncu --set full --import-source yes --force-overwrite -o softmax_profile ./softmax
	ncu-ui softmax_profile.ncu-rep &
 
build: clean all
 
clean:
	rm -f softmax softmax_profile.ncu-rep