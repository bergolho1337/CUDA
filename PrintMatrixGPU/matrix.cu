// Multiplicacao matriz-vetor em CUDA
// Considerando 1 bloco por linha
// A*b = c

#include <stdio.h>
#include <stdlib.h>
#include "book.h"

const int BLOCK_SIZE = 4;
const int THREAD_PER_BLOCK = 4;

__global__ void PrintMatrix (int *a, const int pitch)
{
  int tid;
  tid = blockDim.x*blockIdx.x + threadIdx.x;

  int c1_value = *((int*)((char *)a + pitch * 0) + tid);
  int c2_value = *((int*)((char *)a + pitch * 1) + tid);
  int c3_value = *((int*)((char *)a + pitch * 2) + tid);
  int c4_value = *((int*)((char *)a + pitch * 3) + tid);

  // Printing each line of the matrix
  if (tid < BLOCK_SIZE)
  	printf("Thread %d -> %d %d %d %d\n",tid,c1_value,c2_value,c3_value,c4_value);
  
}

__global__ void PrintKernel ()
{
  printf("CHUPA\n");
}

__global__ void FillMatrix (int *a, const int pitch)
{
  int tid;
  tid = blockDim.x*blockIdx.x + threadIdx.x;
  
  // Filling each line of the matrix
  if (tid < BLOCK_SIZE)
  {
	*((int*)((char *)a + pitch * 0) + tid) = 0;
	*((int*)((char *)a + pitch * 1) + tid) = 1;
	*((int*)((char *)a + pitch * 2) + tid) = 2;
	*((int*)((char *)a + pitch * 3) + tid) = 3; 
  }
  
}

int main ()
{
  size_t pitch;
  int *a;
  int *da;


  // Aloca memoria na CPU
  a = (int*)malloc(sizeof(int)*BLOCK_SIZE*BLOCK_SIZE);
  // Aloca memoria na GPU.
  // A funcao cudaMallocPitch() aloca um vetor 2d contiguamente na memoria e retorna um parametro adicional pitch
  // Este parametro pitch deve ser utilizado para acessar os elementos do array bidimensional
  HANDLE_ERROR(cudaMallocPitch((void **) &(da), &pitch, sizeof(int)*BLOCK_SIZE, (size_t)BLOCK_SIZE));

  // Chama o kernel
  FillMatrix<<<BLOCK_SIZE,THREAD_PER_BLOCK>>>(da,pitch);
  PrintMatrix<<<BLOCK_SIZE,THREAD_PER_BLOCK>>>(da,pitch);

  PrintKernel<<<BLOCK_SIZE,THREAD_PER_BLOCK>>>();
  
  cudaFree(da);
  free(a);

  return 0;
}
