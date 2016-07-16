# Caminho do compilador no cluster do PGMC
CCUDA = /usr/local/cuda-7.5/bin/nvcc
CCUDA_FLAGS = -g -G

hello: hello.cu
        $(CCUDA) $(CCUDA_FLAGS) $^ -o $@

clear:
      	rm hello

				
