CCUDA = nvcc
CCUDA_FLAGS = -g -G
DEFINE_DEBUG = 1

matvec: matvec.cu
ifeq ($(DEFINE_DEBUG),1)
	$(CCUDA) $(CCUDA_FLAGS) -D DEBUG $^ -o $@
else
	$(CCUDA) $(CCUDA_FLAGS) $^ -o $@
endif

clear:
	rm matvec
