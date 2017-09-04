#include <stdio.h>
#include <string.h>

const int NUMCOLS = 8;
const int BLOCKSIZE = 2;
const int GRIDSIZE = 4;

__global__ void kernel (int *v)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = row * NUMCOLS + col;

    if (blockIdx.x == 0 && blockIdx.y == 0)
        v[tid] = 0;
    if (blockIdx.x == 0 && blockIdx.y == 1)
        v[tid] = 1;
    if (blockIdx.x == 0 && blockIdx.y == 2)
        v[tid] = 2;
    if (blockIdx.x == 0 && blockIdx.y == 3)
        v[tid] = 3;
    if (blockIdx.x == 1 && blockIdx.y == 0)
        v[tid] = 4;
    if (blockIdx.x == 1 && blockIdx.y == 1)
        v[tid] = 5;
    if (blockIdx.x == 1 && blockIdx.y == 2)
        v[tid] = 6;
    if (blockIdx.x == 1 && blockIdx.y == 3)
        v[tid] = 7;
    if (blockIdx.x == 2 && blockIdx.y == 0)
        v[tid] = 8;
    if (blockIdx.x == 2 && blockIdx.y == 1)
        v[tid] = 9;
    if (blockIdx.x == 2 && blockIdx.y == 2)
        v[tid] = 10;
    if (blockIdx.x == 2 && blockIdx.y == 3)
        v[tid] = 11;

}

void print (int *v)
{
    for (int i = 0; i < NUMCOLS; i++)
    {
        for (int j = 0; j < NUMCOLS; j++)
            printf("%4d ",v[i*NUMCOLS+j]);
        printf("\n");
    }
    printf("\n");
}

int main ()
{
    int *v, *u;
    int *d_v;
    size_t size = NUMCOLS*NUMCOLS*sizeof(int);
    v = (int*)malloc(size); memset(v,0,size);
    u = (int*)malloc(size);
    cudaMalloc(&d_v,size);
    cudaMemcpy(d_v,v,size,cudaMemcpyHostToDevice);

    dim3 gridSize(GRIDSIZE,GRIDSIZE);
    dim3 blockSize(BLOCKSIZE,BLOCKSIZE);

    print (v);

    kernel<<<gridSize,blockSize>>>(d_v);
    cudaMemcpy(u,d_v,size,cudaMemcpyDeviceToHost);
    
    print(u);

    free(v); free(u);
    cudaFree(d_v);

    return 0;
}