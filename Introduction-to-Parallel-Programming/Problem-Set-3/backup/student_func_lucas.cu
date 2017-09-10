/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include <stdio.h>
#include <float.h>
#include "utils.h"

__global__ void gpu_reduce_min (const float* const in, float* out, const size_t numRows, const size_t numCols)
{
  extern __shared__ float cache[];
  int nElem = numRows * numCols;
  int nCache = blockDim.x * blockDim.y;
  // Index calculations
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int gid = row * numCols + col;
  int tid = threadIdx.y * blockDim.y + threadIdx.x;
  int bid = blockIdx.x * gridDim.y + blockIdx.y;

  // Copy input to shared memory
  if (gid < nElem)
    cache[tid] = in[gid];
  __syncthreads();

  for (int s = nCache/2; s > 0; s >>= 1)
  {
    if (tid < s)
      cache[tid] = min(cache[tid],cache[tid + s]);
    __syncthreads();
  }

  if (tid == 0)
  {
    out[bid] = cache[0];
  }
}

__global__ void gpu_reduce_max (const float* const in, float* out, const size_t numRows, const size_t numCols)
{
  extern __shared__ float cache[];
  int nElem = numRows * numCols;
  int nCache = blockDim.x * blockDim.y;
  // Index calculations
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int gid = row * numCols + col;
  int tid = threadIdx.y * blockDim.y + threadIdx.x;
  int bid = blockIdx.x * gridDim.y + blockIdx.y;

  // Copy input to shared memory
  if (gid < nElem)
    cache[tid] = in[gid];
  __syncthreads();

  for (int s = nCache/2; s > 0; s >>= 1)
  {
    if (tid < s)
      cache[tid] = max(cache[tid],cache[tid + s]);
    __syncthreads();
  }

  if (tid == 0)
  {
    out[bid] = cache[0];
  }
}

__global__ void gpu_histogram (const float* const d_logLuminance, unsigned int *hist, const size_t numRows, const size_t numCols,
                               const float logLumMin, const float logLumMax, const float logLumRange,
                               const size_t numBins)
{
  int nElem = numRows * numCols;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int gid = row * numCols + col;

  if (gid < nElem)
  {
    int bin = (d_logLuminance[gid] - logLumMin) / logLumRange * numBins;
    atomicAdd(&hist[bin],1);
  }
}

__global__ void gpu_exclusive_scan (unsigned int *in, unsigned int* const out, const size_t n)
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
}

float cpu_reduce_min (float *out, int n, int m)
{
  float minValue = out[0];
  for (int i = 0; i < n*m; i++)
    minValue = min(minValue,out[i]);
  return minValue;
}

float minReduction (const float* const d_logLuminance, const size_t numRows, const size_t numCols)
{
  float *h_out, min_logLum;
  float *d_out;
  // Dimensions of the kernel
  dim3 blockDim(32,32);
  dim3 gridDim(numCols/blockDim.x+1,numRows/blockDim.y+1);
  size_t sharedMem = sizeof(float)*blockDim.x*blockDim.y;
  // Output vector of the reduction
  size_t sizeOut = sizeof(float)*gridDim.x*gridDim.y;
  h_out = (float*)malloc(sizeOut);
  checkCudaErrors(cudaMalloc(&d_out,sizeOut)); 

  // Call the min reduction kernel
  gpu_reduce_min<<<gridDim,blockDim,sharedMem>>>(d_logLuminance,d_out,numRows,numCols);
  checkCudaErrors(cudaMemcpy(h_out,d_out,sizeOut,cudaMemcpyDeviceToHost));
  min_logLum = cpu_reduce_min(h_out,gridDim.x,gridDim.y);

  free(h_out);
  checkCudaErrors(cudaFree(d_out));
  
  return min_logLum;
}

float cpu_reduce_max (float *out, int n, int m)
{
  float maxValue = out[0];
  for (int i = 0; i < n*m; i++)
    maxValue = max(maxValue,out[i]);
  return maxValue;
}

float maxReduction (const float* const d_logLuminance, const size_t numRows, const size_t numCols)
{
  float *h_out, max_logLum;
  float *d_out;
  // Dimensions of the kernel
  dim3 blockDim(32,32);
  dim3 gridDim(numCols/blockDim.x+1,numRows/blockDim.y+1);
  size_t sharedMem = sizeof(float)*blockDim.x*blockDim.y;
  // Output vector of the reduction
  size_t sizeOut = sizeof(float)*gridDim.x*gridDim.y;
  h_out = (float*)malloc(sizeOut);
  checkCudaErrors(cudaMalloc(&d_out,sizeOut)); 

  // Call the max reduction kernel
  gpu_reduce_max<<<gridDim,blockDim,sharedMem>>>(d_logLuminance,d_out,numRows,numCols);
  checkCudaErrors(cudaMemcpy(h_out,d_out,sizeOut,cudaMemcpyDeviceToHost));
  max_logLum = cpu_reduce_max(h_out,gridDim.x,gridDim.y);

  free(h_out);
  checkCudaErrors(cudaFree(d_out));
  
  return max_logLum;
}

unsigned int* compHistogram (const float* const d_logLuminance, const size_t numRows, const size_t numCols,
                      const float min_logLum, const float max_logLum, const float logLumRange,
                      const size_t numBins)
{
  unsigned int *h_hist;
  unsigned int *d_hist;
  // Dimensions of the kernel
  dim3 blockDim(32,32);
  dim3 gridDim(numCols/blockDim.x+1,numRows/blockDim.y+1);
  // Allocate memory for host and device histogram
  h_hist = (unsigned int*)calloc(numBins,sizeof(unsigned int));
  checkCudaErrors(cudaMalloc(&d_hist,sizeof(unsigned int)*numBins));
  checkCudaErrors(cudaMemset(d_hist,0,sizeof(unsigned int)*numBins));

  // Call the histogram kernel
  gpu_histogram<<<gridDim,blockDim>>>(d_logLuminance,d_hist,numRows,numCols,min_logLum,max_logLum,logLumRange,numBins);
  checkCudaErrors(cudaMemcpy(h_hist,d_hist,sizeof(unsigned int)*numBins,cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(d_hist));
  return h_hist;
}

void exclusiveScan (unsigned int *hist, unsigned int* const d_cdf, const size_t numBins)
{
  // Copy the histogram to the GPU
  unsigned int *d_hist;
  size_t size = sizeof(unsigned int)*numBins;
  checkCudaErrors(cudaMalloc(&d_hist,size));
  checkCudaErrors(cudaMemcpy(d_hist,hist,size,cudaMemcpyHostToDevice));

  // Call exclusive scan kernel
  dim3 blockDim(32,32);
  dim3 gridDim(1,1);
  size_t sharedMem = sizeof(unsigned int)*numBins*2;
  gpu_exclusive_scan<<<gridDim,blockDim,sharedMem>>>(d_hist,d_cdf,numBins);

  // DEBUG: Copy result to CPU
  //int *h_cdf = (int*)malloc(sizeof(int)*numBins); 
  //checkCudaErrors(cudaMemcpy(h_cdf,d_cdf,sizeof(int)*numBins,cudaMemcpyDeviceToHost));

  //printf("Histogram || Scan\n");
  //for (int i = 0; i < numBins; i++)
  //  printf("Hist[%d] = %d || CDF[%d] = %d\n",i,hist[i],i,h_cdf[i]);

}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

    // Step 1) Min & Max Reduction
    min_logLum = minReduction(d_logLuminance,numRows,numCols);
    max_logLum = maxReduction(d_logLuminance,numRows,numCols);

    // Step 2) Calculate the range
    float logLumRange = max_logLum - min_logLum;
    printf("max_logLum: %f  min_logLum: %f  logLumRange: %f\n", max_logLum, min_logLum, logLumRange);

    // Step 3) Compute a histogram
    unsigned int *hist = compHistogram(d_logLuminance,numRows,numCols,min_logLum,max_logLum,logLumRange,numBins);

    // Step 4) Perform an exclusive scan on the histogram
    exclusiveScan(hist,d_cdf,numBins);

}
