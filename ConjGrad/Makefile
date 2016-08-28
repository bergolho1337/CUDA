CCUDA = /usr/local/cuda-6.0/bin/nvcc
CCUDA_FLAGS = -g -G -arch=sm_13
PROG_NAME = conjgrad

$(PROG_NAME): $(PROG_NAME).cu
	$(CCUDA) $(CCUDA_FLAGS) $^ -o $@

clear:
	rm $(PROG_NAME)
