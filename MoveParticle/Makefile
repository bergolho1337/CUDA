OBJECTS = main.o particle.o v3.o
CCUDA = /usr/local/cuda-6.0/bin/nvcc
CCUDA_FLAGS = -g -G -arch=sm_20

all: $(OBJECTS)
	$(CCUDA) $(CCUDA_FLAGS) $(OBJECTS) -o app

%.o: %.cpp
	$(CCUDA) -x cu -arch=sm_20 -I. -dc $< -o $@

clean:
	rm -f *.o app


# -dc = Pede para o 'nvcc' gerar codigo do Device primeiro para depois linkasr
# -x cu = Diz ao 'nvcc' para tratar todos os arquivos de entrada como .cu
