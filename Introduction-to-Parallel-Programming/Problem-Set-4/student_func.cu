//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

void Swap (unsigned int* &a, unsigned int* &b)
{
  unsigned int *temp = a;
  a = b;
  b = temp;
}

// ---------------------------------------------------------------------------------------------------------------------------
// -------------------------------------------------- Kernels ----------------------------------------------------------------
// Compute an histogram of how many 0's and 1's are set using the current mask
__global__ void Gpu_Histogram (unsigned int *in, unsigned int *bins, const unsigned int mask, const int i, const size_t numElems)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numElems)
    {
      unsigned int bin = (in[tid] & mask) >> i;
      atomicAdd(&bins[bin],1);
    }

    //__syncthreads();
    //if (tid == 0)
    //  printf("Bin 0 = %u || Bin 1 = %u\n",bins[0],bins[1]);
}

// Do a exclusive scan on the input and generate two output. 
// One considering the original array (d_output) and another using the negated version (d_noutput)
// TICK: This exclusive scan only work with an array size multiple of a power of 2, so in order to process all the elements
//       we call this kernel a number of times until we reach the number of elements 
__global__ void Gpu_Exclusive_Scan (unsigned int pass, unsigned int const *d_inputVals,
                                      unsigned int *d_output, unsigned int *d_noutput,
                                      const int size, unsigned int base, unsigned int threadSize) 
{
    int mid = threadIdx.x + threadSize * base;
    unsigned int one = 1;

    if(mid >= size)
        return;
    unsigned int val = 0;
    unsigned int nval = 0;
    if(mid > 0)
    {
        val = ((d_inputVals[mid-1] & (one<<pass))  == (one<<pass)) ? 1 : 0;
        nval = !val;
    }
    else
        val = 0;

    d_output[mid] = val;
    d_noutput[mid] = nval;
    
    __syncthreads();
    
    for(int s = 1; s <= threadSize; s *= 2) 
    {
        int spot = mid - s; 
         
        if(spot >= 0 && spot >=  threadSize*base)
        {
            val = d_output[spot];
            nval = d_noutput[spot];
        }
        __syncthreads();
        if(spot >= 0 && spot >= threadSize*base)
        {
            d_output[mid] += val;
            d_noutput[mid] += nval;
        }
        __syncthreads();
    }
    // This is the trick we use to update the current result with the last kernel
    // REMEMBER: The first element of the current scan (not count the first kernel call) will be at least the output 
    //           of the last element from the previous kernel
    if (base > 0)
    {       
        d_output[mid] += d_output[base*threadSize - 1];
        d_noutput[mid] += d_noutput[base*threadSize - 1];
    }
        
    
}

/*
// Calculate the predicate of each element on the input
__global__ void Gpu_Predicate (unsigned int *in, unsigned int *out, unsigned int *nout, const int mask, const int i, const size_t n) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n)
    {
      // Predicate format = Check if the ith bit of the element is set
      unsigned int bit = (in[tid] & mask) >> i;
    
      out[tid] = bit;
      nout[tid] = !bit;
    } 
}
*/

// Map the correct position of each element based on the scan arrays
__global__ void Gpu_Mapping (unsigned int *in, unsigned int *out,
                             unsigned int *inPos, unsigned int *outPos,   
                             unsigned int *hist, unsigned int *scan, unsigned int *nscan,
                             const int mask, const int i, const size_t n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = hist[0];
    
    // Avoid out of bounds
    if (tid < n)
    {
        // Check the predicate of my element
        unsigned int bit = (in[tid] & mask) >> i;
        // If bit 1 is set, then we need to jump the elements which have the bit 0 set
        // This number is the first element of the histogram 
        // (REMEMBER: We are sorting by moving all numbers with the 0 bit set to the beginning of the array)
        if (bit)
        {
            out[scan[tid] + offset] = in[tid];
            outPos[scan[tid] + offset] = inPos[tid];
        }
        else
        {
            out[nscan[tid]] = in[tid];
            outPos[nscan[tid]] = inPos[tid];
        }
    }
    
}

/*
// DEBUG function
void checkDeviceVector (const char *name, unsigned int *darr, const size_t n)
{
    unsigned int *arr = (unsigned int*)malloc(sizeof(unsigned int)*n);
    cudaMemcpy(arr,darr,sizeof(unsigned int)*n,cudaMemcpyDeviceToHost);

    printf("%s\n",name);
    for (size_t i = 0; i < 20; i++)
        printf("%u ",arr[i]);
    printf("\n\n\n");

    free(arr);
}
*/

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
    // Calculate the dimension of the kernel
    dim3 blockSize(512,1);
    dim3 gridSize(numElems/blockSize.x + 1,1);

    // Size of the bins of histogram
    const int numBits = 1;
    const int numBins = 2;

    // Shared memory sizes
    size_t sizeBins = sizeof(unsigned int)*numBins;
    size_t sizeN = sizeof(unsigned int)*numElems;

    // Allocate GPU memory for the histogram and result of the scan on the 0's bits (nscan) and
    // on the 1's bits (scan)
    unsigned int *binHistogram, *scan, *nscan;
    cudaMalloc(&binHistogram,sizeof(unsigned int)*numBins);
    cudaMalloc(&scan,sizeof(unsigned int)*numElems);
    cudaMalloc(&nscan,sizeof(unsigned int)*numElems);

    // Iterate over all 32 bits of an 'unsigned int'
    for (int i = 0; i < 32; i += numBits)
    {
        unsigned int mask = numBits << i;

        // Reset the bins
        cudaMemset(binHistogram,0,sizeBins);
        cudaMemset(scan,0,sizeN);
        cudaMemset(nscan,0,sizeN);

        // Call kernel to compute histogram
        Gpu_Histogram<<<gridSize,blockSize>>>(d_inputVals,binHistogram,mask,i,numElems);

        // Call kernel to compute the exclusive scan on the predicates
        for(int j = 0; j < numElems / blockSize.x + 1; j++) 
            Gpu_Exclusive_Scan<<<1,blockSize>>>(i,d_inputVals,scan,nscan,numElems,j,blockSize.x);
        //printf("Number of elements = %d\n",numElems);
        //checkDeviceVector("scan",scan,numElems);
        //checkDeviceVector("nscan",nscan,numElems);
        
        // Call kernel to do the final mapping using the scan arrays
        Gpu_Mapping<<<gridSize,blockSize>>>(d_inputVals,d_outputVals,d_inputPos,d_outputPos,binHistogram,scan,nscan,mask,i,numElems);

        // Move for the next iteration
        cudaMemcpy(d_inputVals,d_outputVals,sizeof(unsigned int)*numElems,cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_inputPos,d_outputPos,sizeof(unsigned int)*numElems,cudaMemcpyDeviceToDevice);
    }

    // Free allocate memory
    cudaFree(binHistogram);
    cudaFree(scan);
    cudaFree(nscan);
}
