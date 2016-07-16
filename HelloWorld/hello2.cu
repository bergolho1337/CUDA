// Exemplo de soma entre dois valores em CUDA
// Compilar: make
// Executar: qsub job (cluster)

#include <stdio.h>
#include <stdlib.h>
#include "book.h"                     // Biblioteca com MACROS do livro "Cuda by Example"

// Kernel que soma dois valores 'a' e 'b' e armazena o resultado em 'c' na GPU
__global__ void add (int a, int b, int *c)
{
  *c = a + b;
}

int main ()
{
  int c;            // Variavel da CPU
  int *dc;          // Referencia para variavel da GPU (device pointer)

  // Aloca memoria na GPU
  // Usa-se ponteiro duplo para que o valor da variavel possa ser alterado quando passarmos de um dispositivo para outro
  // Envolvemos a chamada de funcao da GPU com um tratador de erros implementado na biblioteca "book.h"
  HANDLE_ERROR(cudaMalloc((void**)&dc,sizeof(int)));

  // Executamos a funcao kernel na GPU
  add<<<1,1>>>(2,7,dc);

  HANDLE_ERROR(cudaMemcpy(&c,dc,sizeof(int),cudaMemcpyDeviceToHost));

  // Confere o resultado do calculo da GPU e da CPU (tem que ser igual)
  printf("CPU\tGPU\n");
  printf("$d\t%d\n",2+7,c);

  // Libera memoria da GPU
  cudaFree(dc);

  return 0;
}
