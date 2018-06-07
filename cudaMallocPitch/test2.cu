#include<stdio.h>
#include <stdlib.h>

#define Nrows 3
#define Ncols 5
#define Nmatrix 4


__global__ void fillMatrix (float *devPtr, size_t pitch, int matrix_type)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < Ncols)
    {
        switch (matrix_type)
        {
            case 0: {
                        *((float * )((char *) devPtr + pitch * 0) + tid) = 1.0f;
                        *((float * )((char *) devPtr + pitch * 1) + tid) = 3.0f;
                        *((float * )((char *) devPtr + pitch * 2) + tid) = 5.0f;
                        break;
                    }
            case 1: {
                        *((float * )((char *) devPtr + pitch * 0) + tid) = 2.0f;
                        *((float * )((char *) devPtr + pitch * 1) + tid) = 4.0f;
                        *((float * )((char *) devPtr + pitch * 2) + tid) = 6.0f;
                        break;
                    }
            case 2: {
                        *((float * )((char *) devPtr + pitch * 0) + tid) = 5.0f;
                        *((float * )((char *) devPtr + pitch * 1) + tid) = 15.0f;
                        *((float * )((char *) devPtr + pitch * 2) + tid) = 30.0f;
                        break;
                    }
            case 3: {
                        *((float * )((char *) devPtr + pitch * 0) + tid) = 10.0f;
                        *((float * )((char *) devPtr + pitch * 1) + tid) = 30.0f;
                        *((float * )((char *) devPtr + pitch * 2) + tid) = 90.0f;
                        break;
                    }
        }
        
    }
        
}

/********/
/* MAIN */
/********/
int main()
{
float **hostPtr;
float **devPtr;
size_t *pitch;

hostPtr = (float**)malloc(sizeof(float*)*Nmatrix);
for (int i = 0; i < Nmatrix; i++)
    hostPtr[i] = (float*)malloc(sizeof(float)*Ncols*Nrows);

pitch = (size_t*)malloc(sizeof(size_t)*Nmatrix);
devPtr = (float**)malloc(sizeof(float*)*Nmatrix);
for (int i = 0; i < Nmatrix; i++)
    cudaMallocPitch(&devPtr[i], &pitch[i], Ncols * sizeof(float), Nrows);

for (int i = 0; i < Nmatrix; i++)
    fillMatrix<<<1,64>>>(devPtr[i],pitch[i],i);

for (int i = 0; i < Nmatrix; i++)
    cudaMemcpy2D(hostPtr[i], Ncols * sizeof(float), devPtr[i], pitch[i], Ncols * sizeof(float), Nrows, cudaMemcpyDeviceToHost);

for (int k = 0; k < Nmatrix; k++)
{
    printf("=========================================================================\n");
    printf("Printing matrix %d\n",k);
    printf("Pitch = %d\n",pitch[k]);
    for (int i = 0; i < Nrows; i++)
        for (int j = 0; j < Ncols; j++)
            printf("row %i column %i value %f \n", i, j, hostPtr[k][i*Ncols+j]);
    printf("=========================================================================\n");
}

for (int i = 0; i < Nmatrix; i++)
{
    free(hostPtr[i]);
    cudaFree(devPtr[i]);
}
free(hostPtr);
free(devPtr);

return 0;
}

