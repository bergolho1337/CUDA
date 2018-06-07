#include <stdio.h>
#include <stdlib.h>

#define N 128                       // 2^7
#define THREAD_PER_BLOCK 32         // 2^5

// GPU function for adding two vectors 'a' and 'b'
__global__ void add (int *a, int *b, int *c)
{
  // Calculate the index of the current thread 
  // of the current block
  int index = threadIdx.x + blockDim.x*blockIdx.x;
  c[index] = a[index] + b[index];
}

// Gera um vetor aleatorio
void random_ints (int *v)
{
  for (int i = 0; i < N; i++)
    v[i] = rand() % 10;
}

void checkSolution (int *a, int *b, int *c)
{
  printf("===== RESULT OF THE SUM =====\n");
  printf("CPU\tGPU\n");
  for (int i = 0; i < N; i++)
    printf("%d\t%d\n",a[i]+b[i],c[i]);
}

int main ()
{
  int *a, *b, *c;                   // CPU pointers
  int *da, *db, *dc;                // GPU pointers
  int size;
  size = sizeof(int)*N;

  // Alloc memory space for GPU
  cudaMalloc((void**)&da,size);
  cudaMalloc((void**)&db,size);
  cudaMalloc((void**)&dc,size);

  // Alloc space for the CPU
  a = (int*)malloc(size); random_ints(a);
  b = (int*)malloc(size); random_ints(b);
  c = (int*)malloc(size);

  // Copy the inputs to the device
  cudaMemcpy(da,a,size,cudaMemcpyHostToDevice);
  cudaMemcpy(db,b,size,cudaMemcpyHostToDevice);

  // Lauch the kernel on the GPU
  add<<<N/THREAD_PER_BLOCK,N>>>(da,db,dc);

  // Copy the result back to the CPU
  cudaMemcpy(c,dc,size,cudaMemcpyDeviceToHost);

  // Check the solution
  checkSolution(a,b,c);

  // Cleanup
  free(a); free(b); free(c);
  cudaFree(da); cudaFree(db); cudaFree(dc);

  return 0;
}
