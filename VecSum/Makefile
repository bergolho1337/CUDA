CCUDA = /usr/local/cuda-6.0/bin/nvcc
CCUDA_FLAGS = -g -G

vecsum: vecsum.cu
        $(CCUDA) $(CCUDA_FLAGS) $^ -o $@

clear:
      	rm vecsum
