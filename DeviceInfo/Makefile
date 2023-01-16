CCUDA = nvcc
CCUDA_FLAGS = -g -G

deviceInfo: deviceInfo.cu
	$(CCUDA) $(CCUDA_FLAGS) $^ -o $@

clear:
	rm deviceInfo
