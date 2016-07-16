#include <stdio.h>
#include <stdlib.h>
#include "book.h"

#define N 10                  // MAX = 65535 (maior numero de blocos permitidos)
#define DEBUG 1

__global__ void add (int *a, int *b, int *c)
{
  int tid;
  tid = blockIdx.x;         // Handle the data of this index
  if (tid < N)
    c[tid] = a[tid] + b[tid];
}

int main ()
{
  int i, size;
  int a[N], b[N], c[N];           // CPU pointers
  int *da, *db, *dc;              // GPU pointers

  // Size of memory for each vector
  size = sizeof(int)*N;

  // Allocate memory GPU
  HANDLE_ERROR(cudaMalloc((void**)&da,size));
  HANDLE_ERROR(cudaMalloc((void**)&db,size));
  HANDLE_ERROR(cudaMalloc((void**)&dc,size));

  // Fill the array 'a' and 'b'
  for (i = 0; i < N; i++)
  {
    a[i] = -i;
    b[i] = i*i;
  }

  // Copy the array 'a' and 'b' to the GPU
  HANDLE_ERROR(cudaMemcpy(da,a,size,cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(db,b,size,cudaMemcpyHostToDevice));

  // Call the kernel
  add<<<N,1>>>(da,db,dc);

  // Copy the array 'dc' that has the result back to 'c' on the CPU
  HANDLE_ERROR(cudaMemcpy(c,dc,size,cudaMemcpyDeviceToHost));

  #ifdef DEBUG
  // Display the result
  for (i = 0; i < N; i++)
  {
    printf("%d + %d = %d\n",a[i],b[i],c[i]);
  }
  #endif

  // Free memory
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);

  return 0;
}
