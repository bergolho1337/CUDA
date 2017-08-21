NVCC = nvcc
NVCC_FLAGS = -O2

all: atomics

atomics: atomics.cu
	$(NVCC) -o atomics atomics.cu

clean:
	rm -f atomics
