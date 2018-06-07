// This example demonstrate how use the printf() function inside a kernel.
// In order to do that the code must be generate to architetures with compute capability greater than 2.0
// Compile:
// 		nvcc -gencode=arch=compute_30,code=sm_30 -g -o helloGPU helloGPU.cu

#include <stdio.h>

__global__ void helloCUDA(float f)
{
  printf("Hello world from thread %d -- My value of f = %f\n", threadIdx.x, f);
}

int main()
{
  helloCUDA<<<1, 5>>>(1.2345f);
  cudaDeviceReset();
  return 0;
}
