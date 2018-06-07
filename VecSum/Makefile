CCUDA = nvcc
CCUDA_FLAGS = -g -G

vecsum: vecsum.cu
        $(CCUDA) $(CCUDA_FLAGS) $^ -o $@

clear:
      	rm vecsum
