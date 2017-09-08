#include <stdio.h>
#include <string.h>

const int N = 8;
const int BLOCKSIZE = 2;
const int GRIDSIZE = 4;

const int MAXINT = 9999;
const int MININT = -1;

// ---------------------------------------------- KERNELS ---------------------------------------------------------------
__global__ void gpu_reduce_sum (int *v, int *out)
{
    extern __shared__ int cache[];
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int gid = row * N + col;
    int tid = threadIdx.y * BLOCKSIZE + threadIdx.x;
    int bid = blockIdx.x * GRIDSIZE + blockIdx.y;

    if (gid < N*N)
        cache[tid] = v[gid];
    else
        cache[tid] = 0;
    __syncthreads();

    for (int s = BLOCKSIZE*BLOCKSIZE/2; s > 0; s >>= 1)
    {
        if (tid < s)
            cache[tid] += cache[tid + s];
        __syncthreads();
    }

    if (tid == 0)
    {
        out[bid] = cache[0];
        //printf("Block (%d,%d) - %d = %d\n",blockIdx.x,blockIdx.y,bid,out[bid]);
    }
}

__global__ void gpu_reduce_min (int *v, int *out)
{
    extern __shared__ int cache[];
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int gid = row * N + col;
    int tid = threadIdx.y * BLOCKSIZE + threadIdx.x;
    int bid = blockIdx.x * GRIDSIZE + blockIdx.y;

    if (gid < N*N)
        cache[tid] = v[gid];
    else
        cache[tid] = MAXINT;
    __syncthreads();

    for (int s = BLOCKSIZE*BLOCKSIZE/2; s > 0; s >>= 1)
    {
        if (tid < s)
            cache[tid] = min(cache[tid],cache[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
    {
        out[bid] = cache[0];
        //printf("Block (%d,%d) - %d = %d\n",blockIdx.x,blockIdx.y,bid,out[bid]);
    }
}

__global__ void gpu_reduce_max (int *v, int *out)
{
    extern __shared__ int cache[];
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int gid = row * N + col;
    int tid = threadIdx.y * BLOCKSIZE + threadIdx.x;
    int bid = blockIdx.x * GRIDSIZE + blockIdx.y;

    if (gid < N*N)
        cache[tid] = v[gid];
    else
        cache[tid] = MININT;
    __syncthreads();

    for (int s = BLOCKSIZE*BLOCKSIZE/2; s > 0; s >>= 1)
    {
        if (tid < s)
            cache[tid] = max(cache[tid],cache[tid + s]);
        __syncthreads();
    }

    if (tid == 0)
    {
        out[bid] = cache[0];
        //printf("Block (%d,%d) - %d = %d\n",blockIdx.x,blockIdx.y,bid,out[bid]);
    }
}
// ---------------------------------------------------------------------------------------------------------------------------
// -------------------------------------------------- CPU Functions ----------------------------------------------------------

int cpu_sum (int *out)
{
    int resp = 0;
    for (int i = 0; i < GRIDSIZE*GRIDSIZE; i++)
        resp += out[i];
    return resp;
}

int cpu_min (int *out)
{
    int resp = out[0];
    for (int i = 0; i < GRIDSIZE*GRIDSIZE; i++)
        resp = min(resp,out[i]);
    return resp;
}

int cpu_max (int *out)
{
    int resp = out[0];
    for (int i = 0; i < GRIDSIZE*GRIDSIZE; i++)
        resp = max(resp,out[i]);
    return resp;
}

void print (int *v)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            printf("%4d ",v[i*N+j]);
        printf("\n");
    }
    printf("\n");
}

void generate (int *v)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            v[i*N+j] = i*N+j; 
} 

void Usage (char pName[])
{
    printf("============================================\n");
    printf("Usage:> %s <op>\n",pName);
    printf("<op> = Type of reduction operation\n");
    printf("\t1 - Sum\n");
    printf("\t2 - Min\n");
    printf("\t3 - Max\n");
    printf("============================================\n");
}

// ---------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------ MAIN FUNCTION ------------------------------------------------------------

int main (int argc, char *argv[])
{
    if (argc-1 != 1)
    {
        Usage(argv[0]);
        exit(1);
    }

    // Type of the reduction operation
    int op = atoi(argv[1]);
    // Declare and allocate memory for the host and device structures
    int *h_v, *h_out;
    int *d_v, *d_out;
    size_t sizeIn = N*N*sizeof(int);
    size_t sizeOut = GRIDSIZE*GRIDSIZE*sizeof(int);
    h_v = (int*)malloc(sizeIn); generate(h_v); print(h_v);
    cudaMalloc(&d_v,sizeIn);
    cudaMemcpy(d_v,h_v,sizeIn,cudaMemcpyHostToDevice);

    h_out = (int*)malloc(sizeOut);
    cudaMalloc(&d_out,sizeOut);

    dim3 gridSize(GRIDSIZE,GRIDSIZE);
    dim3 blockSize(BLOCKSIZE,BLOCKSIZE);
    size_t sharedMem = sizeof(int)*BLOCKSIZE*BLOCKSIZE;

    // Call reduce kernel
    if (op == 1)
        gpu_reduce_sum<<<gridSize,blockSize,sharedMem>>>(d_v,d_out);
    else if (op == 2)
        gpu_reduce_min<<<gridSize,blockSize,sharedMem>>>(d_v,d_out);
    else if (op == 3)
        gpu_reduce_max<<<gridSize,blockSize,sharedMem>>>(d_v,d_out);

    cudaMemcpy(h_out,d_out,sizeOut,cudaMemcpyDeviceToHost);

    // Complete the reduction on CPU
    int result;
    if (op == 1)
        result = cpu_sum(h_out);
    else if (op == 2)
        result = cpu_min(h_out);
    else if (op == 3)
        result = cpu_max(h_out);

    printf("Result = %d\n",result);

    free(h_v); free(h_out);
    cudaFree(d_v); cudaFree(d_out);

    return 0;
}