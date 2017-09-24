#include <stdio.h>
#include <string.h>

const int N = 8;
const int BLOCKSIZE = 8;
const int GRIDSIZE = 1;

// ---------------------------------------------- KERNELS ---------------------------------------------------------------
// Hillis Steele Scan - Inclusive Scan
__global__ void gpu_blelloch_scan (int *in, int *out)
{
    extern __shared__ int cache[];
    int myId = threadIdx.x;
    // Copy the array to shared memory
    cache[myId] = in[myId];
    __syncthreads();

    for (int d = 1; d < N; d <<= 1)
    {
        if (myId >= d)
            cache[myId] += cache[myId - d];
        __syncthreads();
    }

    // Write results to output
    // -- Inclusive
    //out[myId] = cache[myId];       

    // -- Exclusive
    if (myId == 0) out[0] = 0;
    else out[myId] = cache[myId-1];

}
// ---------------------------------------------------------------------------------------------------------------------------
// -------------------------------------------------- CPU Functions ----------------------------------------------------------

void print (int *v)
{
    for (int i = 0; i < N; i++)
        printf("%d ",v[i]);
    printf("\n\n");
}

void generate (int *v)
{
    for (int i = 0; i < N; i++)
        v[i] = i+1;
    /*
    v[0] = 13;
    v[1] = 7;
    v[2] = 16;
    v[3] = 21;
    v[4] = 8;
    v[5] = 20;
    v[6] = 13;
    v[7] = 12;
    */
} 

void Usage (char pName[])
{
    printf("============================================\n");
    printf("Usage:> %s \n",pName);
    printf("============================================\n");
}

// ---------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------ MAIN FUNCTION ------------------------------------------------------------

int main (int argc, char *argv[])
{
    if (argc-1 != 0)
    {
        Usage(argv[0]);
        exit(1);
    }

    // Declare and allocate memory for the host and device structures
    int *h_in, *h_out;
    int *d_in, *d_out;
    size_t sizeIn = N*sizeof(int);
    size_t sizeOut = N*sizeof(int);
    h_in = (int*)malloc(sizeIn); generate(h_in); print(h_in);
    cudaMalloc(&d_in,sizeIn);
    cudaMemcpy(d_in,h_in,sizeIn,cudaMemcpyHostToDevice);

    h_out = (int*)malloc(sizeOut);
    cudaMalloc(&d_out,sizeOut);

    dim3 gridSize(1,1);
    dim3 blockSize(BLOCKSIZE,1);
    size_t sharedMem = sizeof(int)*BLOCKSIZE*2;

    // Call reduce kernel
    gpu_blelloch_scan<<<gridSize,blockSize,sharedMem>>>(d_in,d_out);
    cudaMemcpy(h_out,d_out,sizeOut,cudaMemcpyDeviceToHost);

    // Print the result
    print(h_out);

    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);

    return 0;
}