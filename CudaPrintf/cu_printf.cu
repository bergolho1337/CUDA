/* File:     cu_printf_ex.cu
 * Purpose:  Show how to use the cuPrintf function from the CUDA SDK
 *
 * Compile:
 *    MacOS: nvcc -o cu_printf_ex cu_printf_ex.cu
 *              -I /Developer/GPU\ Computing/shared/inc
 *              -I /Developer/GPU\ Computing/C/common/inc
 *    Linux: nvcc -o cu_printf_ex cu_printf_ex.cu
 *              -I /usr/local/NVIDIA_GPU_Computing_SDK/shared/inc
 *              -I /usr/local/NVIDIA_GPU_Computing_SDK/C/common/inc
 *
 * Run:      ./cu_printf
 *
 * Input:    none
 * Output:   The value "10" printed by 6 threads
 */
#include <stdio.h>
#include "cuPrintf.cu"
#include <shrUtils.h>
#include "cutil_inline.h"


__global__ void testKernel(int val)
{
   cuPrintf("\tValue is:%d\n", val);
}

int main(int argc, char **argv)
{
   cudaPrintfInit();
   testKernel<<< 2, 3 >>>(10);
   cudaPrintfDisplay(stdout, true);
   cudaPrintfEnd();

   return 0;
}
