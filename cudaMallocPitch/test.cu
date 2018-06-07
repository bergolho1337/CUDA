#include<stdio.h>
#include <stdlib.h>

#define Nrows 3
#define Ncols 5


__global__ void fillMatrix (float *devPtr, size_t pitch)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < Ncols)
    {
        *((float * )((char *) devPtr + pitch * 0) + tid) = 1.0f;
        *((float * )((char *) devPtr + pitch * 1) + tid) = 3.0f;
        *((float * )((char *) devPtr + pitch * 2) + tid) = 5.0f;
    }
        
}

/********/
/* MAIN */
/********/
int main()
{
   float *hostPtr;
   float *devPtr;
   size_t pitch;

   hostPtr = (float*)malloc(sizeof(float)*Nrows*Ncols);
   cudaMallocPitch(&devPtr, &pitch, Ncols * sizeof(float), Nrows);

   fillMatrix<<<1,64>>>(devPtr,pitch);

   cudaMemcpy2D(hostPtr, Ncols * sizeof(float), devPtr, pitch, Ncols * sizeof(float), Nrows, cudaMemcpyDeviceToHost);

   for (int i = 0; i < Nrows; i++)
        for (int j = 0; j < Ncols; j++)
            printf("row %i column %i value %f \n", i, j, hostPtr[i*Ncols+j]);
    
    return 0;
}

