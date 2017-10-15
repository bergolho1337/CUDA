#include <iostream>
#include <cstdio>
#include <cstring>
#include <thrust/device_vector.h>

// ---------------------------------------------------------------------------------------------------------------------------
// -------------------------------------------------- Kernels ----------------------------------------------------------------
// Compute an histogram of how many 0's and 1's are set using the current mask
__global__ void Gpu_Histogram (unsigned int *in, unsigned int *bins, const unsigned int mask, const int i)
{
    int tid = threadIdx.x;
    unsigned int bin = (in[tid] & mask) >> i;
    atomicAdd(&bins[bin],1);
}

// Compute an exclusive scan on the input array and store the result on the output array
__global__ void Gpu_ExScan (unsigned int *in, unsigned int *out, const size_t n)    
{
    extern __shared__ unsigned int cache2[];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int offset = 1;
    // Load the input into shared memory
    cache2[2*tid] = in[2*tid]; 
    cache2[2*tid+1] = in[2*tid+1];
    __syncthreads();

    // Build sum in place up the tree
    for (int d = n >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (tid < d)
        {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;

            cache2[bi] += cache2[ai];
        }
        offset *= 2;
    }

    // Clear the last element
    if (tid == 0)
    {
        cache2[n-1] = 0;
    }

    // Transverse down and build scan
    for (int d = 1; d < n; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if (tid < d)
        {
            int ai = offset*(2*tid+1)-1;
            int bi = offset*(2*tid+2)-1;

            // Avoiding read-write bugs
            int aux = cache2[ai];
            cache2[ai] = cache2[bi];
            cache2[bi] += aux;
        }
    }
    __syncthreads();

    // Write results to output
    out[2*tid] = cache2[2*tid];
    out[2*tid+1] = cache2[2*tid+1];
    offset = out[1];

    //__syncthreads();
    //if (tid == 0)
    //    printf("Binscan 0 = %u || Binscan 1 = %u\n",out[0],out[1]);
}

// Calculate the predicate of each element on the input
__global__ void Gpu_Predicate (unsigned int *in, unsigned int *out, unsigned int *nout, const int mask, const int i)
{
    int tid = threadIdx.x;

    // Predicate format = Check if the ith bit of the element is set
    unsigned int bit = (in[tid] & mask) >> i;
    
    out[tid] = bit;
    nout[tid] = !bit;

}   

// Map the correct position of each element on input based on the scan array of the predicate
__global__ void Gpu_Mapping (unsigned int *in, unsigned int *out, unsigned int *bscan, unsigned int *scan, unsigned int *nscan, const int mask, const int i)
{
    int tid = threadIdx.x;

    unsigned int bit = (in[tid] & mask) >> i;
    // If bit is 1, than pred[tid] is active
    if (bit)
    {
        int offset = bscan[1];
        out[scan[tid] + offset] = in[tid];
    }   
    // Bit 0 is active in npred[tid] 
    else
    {
        out[nscan[tid]] = in[tid];
    }
}

// ---------------------------------------------------------------------------------------------------------------------------
// -------------------------------------------------- CPU Functions ----------------------------------------------------------

void Print (unsigned int *v, unsigned int N)
{
    for (int i = 0; i < N; i++)
        printf("%4u\n",v[i]);
    printf("\n");
}

void GetInput (unsigned int *v, unsigned int N)
{
    for (int i = 0; i < N; i++)
        scanf("%u",&v[i]); 
} 

void Swap (unsigned int* &a, unsigned int* &b)
{
  unsigned int *temp = a;
  a = b;
  b = temp;
}

void RadixSort (unsigned int *in, unsigned int *out, const unsigned int N)
{
    // Define kernel dimension
    dim3 gridSize(1,1);
    dim3 blockSize(N,1);

    const int numBits = 1;
    const int numBins = 2;
    // Shared memory sizes
    size_t sizeBins = sizeof(unsigned int)*numBins;
    size_t sizeN = sizeof(unsigned int)*N;

    // Allocate GPU memory
    unsigned int *binHistogram, *binScan, *pred, *npred, *scan, *nscan;
    cudaMalloc(&binHistogram,sizeof(unsigned int)*numBins);
    cudaMalloc(&binScan,sizeof(unsigned int)*numBins);
    cudaMalloc(&pred,sizeof(unsigned int)*N);
    cudaMalloc(&npred,sizeof(unsigned int)*N);
    cudaMalloc(&scan,sizeof(unsigned int)*N);
    cudaMalloc(&nscan,sizeof(unsigned int)*N);

    for (int i = 0; i < 8 * sizeof(unsigned int); i += numBits)
    {
        unsigned int mask = numBits << i;

        // Reset the bins
        cudaMemset(binHistogram,0,sizeof(unsigned int)*numBins);
        cudaMemset(binScan,0,sizeof(unsigned int)*numBins);

        // Call kernel to compute histogram
        Gpu_Histogram<<<gridSize,blockSize>>>(in,binHistogram,mask,i);

        // Call kernel to compute the exclusive scan on the histogram
        Gpu_ExScan<<<1,numBins,2*sizeBins>>>(binHistogram,binScan,numBins);

        // Call kernel that calculate the predicate
        Gpu_Predicate<<<gridSize,blockSize>>>(in,pred,npred,mask,i);

        // Call kernel to compute the exclusive scan on the predicates
        Gpu_ExScan<<<gridSize,blockSize,sizeN>>>(pred,scan,N);
        Gpu_ExScan<<<gridSize,blockSize,sizeN>>>(npred,nscan,N);

        // Call kernel to do the final mapping using the scan array
        Gpu_Mapping<<<gridSize,blockSize>>>(in,out,binScan,scan,nscan,mask,i);

        Swap(out,in);
    }

    cudaMemcpy(out,in,sizeof(unsigned int)*N,cudaMemcpyDeviceToDevice);

    // Free memory
    cudaFree(binHistogram);
    cudaFree(binScan);
    cudaFree(pred);
    cudaFree(npred);
    cudaFree(scan);
    cudaFree(nscan);
}

void Usage (char pName[])
{
    printf("============================================\n");
    printf("Usage:> %s < 'input_filename'\n",pName);
    printf("Example:> %s < input\n",pName);
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

    // Read size of the input
    unsigned int N;
    scanf("%u",&N);

    // Declare and allocate memory for the host
    unsigned int *in, *out;
    unsigned int *d_in, *d_out;
    in = new unsigned int[N]();
    out = new unsigned int[N]();
    cudaMalloc(&d_in,sizeof(unsigned int)*N);
    cudaMalloc(&d_out,sizeof(unsigned int)*N);

    // Get or Generate the array to sort
    GetInput(in,N);
    Print(in,N);

    // Copy the array to the GPU
    cudaMemcpy(d_in,in,sizeof(unsigned int)*N,cudaMemcpyHostToDevice);

    // Initialize output array
    cudaMemset(d_out,0,sizeof(unsigned int)*N);

    // Sort the array using RadixSort
    RadixSort(d_in,d_out,N);
    
    // Print the result
    cudaMemcpy(out,d_out,sizeof(unsigned int)*N,cudaMemcpyDeviceToHost);
    Print(out,N);

    delete [] in;
    delete [] out;
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}