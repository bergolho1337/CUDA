CCUDA = nvcc
CCUDA_FLAGS = -g
GEN_CODE = arch=compute_30,code=sm_30
PROGRAM_NAME = helloGPU

$(PROGRAM_NAME): $(PROGRAM_NAME).cu
	$(CCUDA) -gencode=$(GEN_CODE) $(CCUDA_FLAGS) -o $@ $<

clear:
	rm $(PROGRAM_NAME)


# nvcc -gencode=arch=compute_30,code=sm_30 hello.cu
