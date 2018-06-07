// Multiplicacao matriz-vetor em CUDA
// Considerando 1 bloco por linha
// A*b = c

#include <stdio.h>
#include <stdlib.h>
#include "book.h"

const int BLOCK_SIZE = 4;
const int THREAD_PER_BLOCK = 4;

__global__ void MatVecMult (int *a, int *b, int *c)
{
  __shared__ int cache[THREAD_PER_BLOCK];
  int tid, sum;
  tid = blockDim.x*blockIdx.x + threadIdx.x;
  sum = 0;
  sum += a[tid]*b[threadIdx.x];
  cache[threadIdx.x] = sum;
  __syncthreads();

  int i = blockDim.x / 2;
  while (i != 0)
  {
    if (threadIdx.x < i)
      cache[threadIdx.x] += cache[threadIdx.x + i];
    __syncthreads();
    i /= 2;
  }
  if (threadIdx.x == 0)
    c[blockIdx.x] = cache[0];
}

int main ()
{
  size_t pitch, size;
  int i, j;
  int *a, *b, *c;
  int *da, *dap, *db, *dc;


  // Aloca memoria
  a = (int*)malloc(sizeof(int)*BLOCK_SIZE*BLOCK_SIZE);
  b = (int*)malloc(sizeof(int)*BLOCK_SIZE);
  c = (int*)malloc(sizeof(int)*BLOCK_SIZE);
  HANDLE_ERROR(cudaMalloc((void**)&da,sizeof(int)*BLOCK_SIZE*BLOCK_SIZE));
//  HANDLE_ERROR(cudaMallocPitch((void **) &(dap), &pitch, size, (size_t )BLOCK_SIZE));
  HANDLE_ERROR(cudaMalloc((void**)&db,sizeof(int)*BLOCK_SIZE));
  HANDLE_ERROR(cudaMalloc((void**)&dc,sizeof(int)*BLOCK_SIZE));


  // Ler a matriz e o vetor
  for (i = 0; i < BLOCK_SIZE; i++)
  {
    for (j = 0; j < BLOCK_SIZE; j++)
      a[i*BLOCK_SIZE+j] = j+1;
    b[i] = 1;
  }

  // Copia os dados para a GPU
  HANDLE_ERROR(cudaMemcpy(da,a,sizeof(int)*BLOCK_SIZE*BLOCK_SIZE,cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(db,b,sizeof(int)*BLOCK_SIZE,cudaMemcpyHostToDevice));

  // Chama o kernel
  MatVecMult<<<BLOCK_SIZE,THREAD_PER_BLOCK>>>(da,db,dc);

  // Copia o resultado para CPU
  HANDLE_ERROR(cudaMemcpy(c,dc,sizeof(int)*BLOCK_SIZE,cudaMemcpyDeviceToHost));

  // Checar o resultado
  int sum, check;
  check = 0;
  for (i = 0; i < BLOCK_SIZE; i++)
  {
    sum = 0;
    for (j = 0; j < BLOCK_SIZE; j++)
      sum += a[i*BLOCK_SIZE+j];
    if (sum != c[i])
    {
      check = 1;
      break;
    }
  }
  if (check)
    printf("[-] ERROR! The result is not the same!\n");
  else
    printf("[+] SUCESS! The result is the same!\n");
  return 0;
}
