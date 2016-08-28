// Programa que calcula o produto interno entre dois vetores usando GPU

#include <stdio.h>
#include <stdlib.h>
#include "book.h"

#define imin(a,b) (a < b ? a : b) // Funcao que descobre o minimo entre dois valores

const int N 33792;                   // N = 33 * 1024
const int THREAD_PER_BLOCK 256;      // Numero de threads por bloco

// Descobre o menor multiplo maior que N
// Eh necessario fazer isso para que o tamanho do vetor da CPU fique pequeno
const int BLOCKS_PER_GRID = imin(32, (N+THREAD_PER_BLOCK-1) / THREAD_PER_BLOCK);

// Kernel da GPU
__global__ dot (double *a, double *b, double *c)
{
  // Reserva para cada bloco um vetor compartilhado
  // Sera usado para escrever o resultado parcial do produto de cada thread
  __shared__ double cache[THREAD_PER_BLOCK];
  int tid = blockDim.x*blockIdx.x + threadIdx.x;
  int cacheIndex = threadIdx.x;
  // Computa o produto interno dos indices da minha thread
  double sum = 0;
  while (tid < N)
  {
    sum += a[tid]*b[tid];
    tid += gridDim.x*blockDim.x;
  }
  // Salva a soma parcial na 'cache' da GPU
  cache[cacheIndex] = sum;
  // Sincroniza as threads para garantir que todo mundo ja escreveu na 'cache'
  __syncthreads();
  // Realiza uma redução de soma do vetor da cache de maneira inteligente
  int i = blockDim.x / 2;
  while (i != 0)
  {
    // Diminui o numero de somas que tem de se fazer
    if (cacheIndex < i)
      cache[cacheIndex] += cache[cacheIndex + i];
    // Garante que todas as threads fizeram sua parte
    __syncthreads();
    i /= 2;
  }
  // A soma fica armazenada na thread 0 de cada bloco
  if (cacheIndex == 0)
    c[blockId.x] = cache[0];
}

int main ()
{
  int i, size;
  double *a, *b, c, *partial_c;
  double *da, *db, *dpartial_c;

  // Aloca memoria na CPU
  a = (double*)malloc(sizeof(double)*N);
  b = (double*)malloc(sizeof(double)*N);
  partial_c = (double*)malloc(sizeof(double)*BLOCKS_PER_GRID);

  // Aloca a memoria para a GPU
  HANDLE_ERROR(cudaMalloc((void**)&da,N*sizeof(double)));
  HANDLE_ERROR(cudaMalloc((void**)&db,N*sizeof(double)));
  HANDLE_ERROR(cudaMalloc((void**)&dpartial_c,BLOCKS_PER_GRID*sizeof(double)));

  // Preenche os vetores 'a' e 'b'
  for (i = 0; i < N; i++)
  {
    a[i] = i;
    b[i] = 2*i;
  }

  // Copia os vetores 'a' e 'b' para a GPU
  HANDLE_ERROR(cudaMemcpy(da,a,N*sizeof(double),cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(db,b,N*sizeof(double),cudaMemcpyHostToDevice));

  // Chama o kernel
  dot<<<BLOCKS_PER_GRID,THREAD_PER_BLOCK>>>(da,db,dpartial_c);

  // Copia o array 'dc' que esta na GPU para a CPU
  HANDLE_ERROR(cudaMemcpy(partial_c,dpartial_c,BLOCKS_PER_GRID*sizeof(double),cudaMemcpyDeviceToHost));

  // Soma os valores na CPU e o resultado sera o produto interno
  c = 0;
  for (i = 0; i < BLOCKS_PER_GRID; i++)
    c += partial_c[i];

  // Para o produto interno desse exemplo o resultado deve ser a soma de quadrados
  #define sum_squares(x) (x*(x+1)*(2*x+1)/6);
  printf("Does GPU value: %.6g = %.6g?\n",c,(double)(sum_squares(N-1)));

  // Libera memoria
  cudaFree(da);
  cudaFree(db);
  cudaFree(dpartial_c);
  free(a);
  free(b);
  free(partial_c);

  return 0;
}
