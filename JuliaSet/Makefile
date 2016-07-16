CCUDA = /usr/local/cuda-6.0/bin/nvcc
CCUDA_FLAGS = -g -G

julia: julia.cu
	$(CCUDA) $(CCUDA_FLAGS) $^ -o $@

clear:
	rm julia
