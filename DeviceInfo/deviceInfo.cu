#include <stdio.h>
#include <stdlib.h>

int main ()
{
  cudaDeviceProp prop;
  int count, i;
  count = 0;
  // Descobre o numero de "devices" rodando CUDA
  cudaGetDeviceCount(&count);
  printf("Number of devices running CUDA = %d\n",count);
  // Iterar entre todos os devices e descobrir suas propriedades
  for (i = 0; i < count; i++)
  {
    printf("======================= Device %d ========================================\n",i+1);
    // Essa funcao retorna uma struct. Varaiveis disponiveis em "CUDA_by_Example.pdf"
    cudaGetDeviceProperties(&prop,i);
    printf("Device Name = %s\n",prop.name);
    printf("Compute capability = %d - %d\n",prop.minor,prop.major);
    printf("Clock rate = %d\n",prop.clockRate);
    printf("Total global memory (bytes) = %d\n",prop.totalGlobalMem);
    printf("Total shared memory a block can use (bytes)  = %d\n",prop.sharedMemPerBlock);
    printf("Number of 32-bit registers available per block = %d\n",prop.regsPerBlock);
    printf("Maximum number of threads a block can contain = %d\n",prop.maxThreadsPerBlock);
    printf("Maximum number of threads allowed along each dimension of a block\n");
    printf("\tx = %d\ty = %d\tz = %d\n",prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
    printf("Maximum number of blocks allowed along each dimension of a grid\n");
    printf("\tx = %d\ty = %d\tz = %d\n",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
    printf("Amount of available constant memory (bytes) = %d\n",prop.totalConstMem);
    printf("Integrate GPU = %d\n",prop.integrated);
    printf("==========================================================================\n");
  }
  return 0;
}
