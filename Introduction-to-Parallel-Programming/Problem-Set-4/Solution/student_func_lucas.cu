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

// Compute an exclusive scan on the input array and store the result on the output array
__global__ void Gpu_ExScan (unsigned int *in, unsigned int *out, unsigned int *gcache, const size_t n)    
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n)
      gcache[tid] = in[tid];
    __syncthreads();

    for (int d = 1; d < n; d <<= 1)
    {
        if (tid >= d)
            gcache[tid] += gcache[tid - d];
        __syncthreads();
    }

    // Write results to output
    // -- Inclusive
    //out[myId] = cache[myId];       

    // -- Exclusive
    if (tid == 0) out[0] = 0;
    else out[tid] = gcache[tid-1];
    
}

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

// Map the correct position of each element on input based on the scan array of the predicate
__global__ void Gpu_Mapping (unsigned int *in, unsigned int *out, \
                             unsigned int *bscan, unsigned int *scan, unsigned int *nscan, \
                             const int mask, const int i, const size_t n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n)
    {
      unsigned int bit = (in[tid] & mask) >> i;
      if (nscan[tid] >= n)
        printf("DANGER\n");
      if (scan[tid] + bscan[1] >= n)
        printf("DANGER 2\n");
      
      /*
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
      */
    }
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  // Calculate the dimension of the kernel
  dim3 blockSize(512,1);
  dim3 gridSize(numElems/blockSize.x + 1,1);

  const int numBits = 1;
  const int numBins = 2;
  // Shared memory sizes
  size_t sizeBins = sizeof(unsigned int)*numBins;
  size_t sizeN = sizeof(unsigned int)*numElems;

  // Allocate GPU memory
  unsigned int *binHistogram, *binScan, *pred, *npred, *scan, *nscan, *gcache;
  cudaMalloc(&binHistogram,sizeof(unsigned int)*numBins);
  cudaMalloc(&binScan,sizeof(unsigned int)*numBins);
  cudaMalloc(&pred,sizeof(unsigned int)*numElems);
  cudaMalloc(&npred,sizeof(unsigned int)*numElems);
  cudaMalloc(&scan,sizeof(unsigned int)*numElems);
  cudaMalloc(&nscan,sizeof(unsigned int)*numElems);
  cudaMalloc(&gcache,sizeof(unsigned int)*2*numElems);              // Global memory cache

  for (int i = 0; i < 8 * sizeof(unsigned int); i += numBits)
  {
      unsigned int mask = numBits << i;

      // Reset the bins
      cudaMemset(binHistogram,0,sizeBins);
      cudaMemset(binScan,0,sizeBins);
      cudaMemset(scan,0,sizeN);
      cudaMemset(nscan,0,sizeN);

      // Call kernel to compute histogram
      Gpu_Histogram<<<gridSize,blockSize>>>(d_inputVals,binHistogram,mask,i,numElems);

      // Call kernel to compute the exclusive scan on the histogram
      Gpu_ExScan<<<1,numBins>>>(binHistogram,binScan,gcache,numBins);

      // Call kernel that calculate the predicate
      Gpu_Predicate<<<gridSize,blockSize>>>(d_inputVals,pred,npred,mask,i,numElems);

      // Call kernel to compute the exclusive scan on the predicates
      cudaMemset(gcache,0,sizeN);
      Gpu_ExScan<<<gridSize,blockSize>>>(pred,scan,gcache,numElems);
      cudaMemset(gcache,0,sizeN);
      Gpu_ExScan<<<gridSize,blockSize>>>(npred,nscan,gcache,numElems);

      // Call kernel to do the final mapping using the scan array
      //Gpu_Mapping<<<gridSize,blockSize>>>(d_inputVals,d_outputVals,binScan,scan,nscan,mask,i,numElems);

      //Swap(d_outputVals,d_inputVals);
      //Swap(d_outputVals,d_inputVals);
  }

}
