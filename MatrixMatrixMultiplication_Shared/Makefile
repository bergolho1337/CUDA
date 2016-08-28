CCUDA = /usr/local/cuda-6.0/bin/nvcc
CCUDA_FLAGS = -g -G
DEFINE_DEBUG = 1

matvec_Shared: matvec_Shared.cu
ifeq ($(DEFINE_DEBUG),1)
	$(CCUDA) $(CCUDA_FLAGS) -D DEBUG $^ -o $@
else
	$(CCUDA) $(CCUDA_FLAGS) $^ -o $@
endif

clear:
	rm matvec_Shared
