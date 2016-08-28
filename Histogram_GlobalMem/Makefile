CCUDA = /usr/local/cuda-6.0/bin/nvcc
CCUDA_FLAGS = -arch=sm_12 -g -G
DEFINE_DEBUG = 1

histogram: histogram.cu
ifeq ($(DEFINE_DEBUG),1)
	$(CCUDA) $(CCUDA_FLAGS) -D DEBUG $^ -o $@
else
	$(CCUDA) $(CCUDA_FLAGS) $^ -o $@
endif

clear:
	rm histogram
