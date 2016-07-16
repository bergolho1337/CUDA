// Exemplo do Hello World em CUDA
// Compilar: make
// Executar: qsub job (cluster)

#include <stdio.h>
#include <stdlib.h>

// Funcao executada na GPU, tambem eh chamada de kernel
__global__ void kernel ()
{
  // No caso eh um kernel que vai para GPU e nao faz nada
}

int main ()
{
  // Informamos ao codigo da CPU que queremos executar a funcao kernel na GPU
  kernel<<<1,1>>>();
  // Voltamos a execucao na CPU e printamos a mensagem de Hello World.
  printf("Hello World!\n");
  return 0;
}
