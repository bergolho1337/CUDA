/* ================================== GRADIENTE CONJUGADO versao CUDA =========================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "book.h"

const int THREAD_PER_BLOCK = 256;
const int BLOCKS_PER_GRID = 32;

double MAX_ITER;
double TOLER;


// Kernel that calcualtes the dot product
__global__ void Dot (double *da, double *db, double *dc, int n)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int cacheIndex = threadIdx.x;
  __shared__ double cache[THREAD_PER_BLOCK];
  double sum = 0.0;
  while (tid < n)
  {
    sum += da[tid]*db[tid];
    tid += gridDim.x * blockDim.x;
  }
  cache[cacheIndex] = sum;
  __syncthreads();
  // Make the reduction of the cache
  int i = blockDim.x / 2;
  while (i != 0)
  {
    if (cacheIndex < i)
      cache[cacheIndex] += cache[cacheIndex + i];
    __syncthreads();
    i /= 2;
  }
  //The result of each block will be at index 0 in the cache vector
  if (cacheIndex == 0)
    dc[blockIdx.x] = cache[0];
}

// Kernel that calculates the SAXPY (Single-Precision A·X Plus Y)
__global__ void SAXPY (double *dev_x, double *dev_y, double alpha, int n)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  while (tid < n)
  {
    dev_x[tid] += alpha * dev_y[tid];
    tid += gridDim.x * blockDim.x;
  }
}

// Kernel that calculates the SAXPY (Single-Precision A·X Plus Y)
__global__ void SAXPY2 (double *dev_x, double *dev_y, double alpha, int n)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  while (tid < n)
  {
    dev_x[tid] = dev_y[tid] + alpha * dev_x[tid];
    tid += gridDim.x * blockDim.x;
  }
}

// Kernel that calculates matrix-vector multiplication
__global__ void MatVec (double *dev_A, double *dev_b, double *dev_x, int n)
{
  int k;
  int row = threadIdx.x + blockDim.x * blockIdx.x;
  double sum;
  while (row < n)
  {
    sum = 0.0;
    for (k = 0; k < n; k++)
      sum += dev_A[row * n + k] * dev_b[k];
    dev_x[row] = sum;
    row += gridDim.x * blockDim.x;
  }
}

void readMatrix_A (double *A, int n)
{
  int i, j;
  for (i = 0; i < n; i++)
  {
    for (j = 0; j < n; j++)
      scanf("%lf",&A[i*n+j]);
  }
}

void readVector_b (double *b, int n)
{
  int i;
  for (i = 0; i < n; i++)
    scanf("%lf",&b[i]);
}

void testSolution (double *A, double *b, double *x, int n)
{
  int i, j;
  double solution = 0.0;
  double sum;
  for (i = 0; i < n; i++)
  {
    sum = 0.0;
    for (j = 0; j < n; j++)
      sum += A[i*n+j]*x[j];
    solution += b[i]-sum;
    solution *= solution;
  }
  printf("Norm of the solution = %e\n",sqrt(solution));
}

void printMatrix (char name[], double *A, int n)
{
  int i, j;
  printf("\n=== %s ===\n",name);
  for (i = 0; i < n; i++)
  {
    printf("\n");
    for (j = 0; j < n; j++)
      printf("%e ",A[i*n+j]);
  }
  printf("\n");
}

void printVector (char name[], double *b, int n)
{
  int i;
  printf("\n=== %s ===\n",name);
  for (i = 0; i < n; i++)
    printf("%e\n",b[i]);
  printf("\n");
}


double calcNorm (double *v, int n)
{
  int i;
  double sum = 0.0;
  for (i = 0; i < n; i++)
    sum += v[i]*v[i];
  return (sqrt(sum));
}

void copyVector (double *o, double *d, int n)
{
  int i;
  for (i = 0; i < n; i++)
    d[i] = o[i];
}

double dotProduct (double *da, double *db, double *dc, double *c, int n)
{
  int i;
  // Call the kernel that calculates the dot product
  Dot<<<BLOCKS_PER_GRID,THREAD_PER_BLOCK>>>(da,db,dc,n);
  HANDLE_ERROR(cudaThreadSynchronize());

  // Copy the result from the GPU to the CPU
  HANDLE_ERROR(cudaMemcpy(c,dc,BLOCKS_PER_GRID*sizeof(double),cudaMemcpyDeviceToHost));

  // Let CPU calculates the final result
  double result = 0.0;
  for (i = 0; i < BLOCKS_PER_GRID; i++)
    result += c[i];

  //printf("Alpha = %e\n",result);
  return result;
}

void saxpy (double *dev_x, double *dev_y, double alpha, int n)
{
  //double *x = (double*)malloc(sizeof(double)*n);

  // Call the kernel that calculates the SAXPY
  SAXPY<<<BLOCKS_PER_GRID,THREAD_PER_BLOCK>>>(dev_x,dev_y,alpha,n);
  HANDLE_ERROR(cudaThreadSynchronize());

  // Copy the result to the CPU
  //HANDLE_ERROR(cudaMemcpy(x,dev_x,n*sizeof(double),cudaMemcpyDeviceToHost));
  //printVector("Vector u",x,n);
  //free(x);
}

void saxpy2 (double *dev_x, double *dev_y, double alpha, int n)
{
  //double *x = (double*)malloc(sizeof(double)*n);

  // Call the kernel that calculates the SAXPY
  SAXPY2<<<BLOCKS_PER_GRID,THREAD_PER_BLOCK>>>(dev_x,dev_y,alpha,n);
  HANDLE_ERROR(cudaThreadSynchronize());

  // Copy the result to the CPU
  //HANDLE_ERROR(cudaMemcpy(x,dev_x,n*sizeof(double),cudaMemcpyDeviceToHost));
  //printVector("Vector p",x,n);
  //free(x);
}

void matvec (double *dev_A, double *dev_b, double *dev_x, int n)
{
    //double *x = (double*)malloc(sizeof(double)*n);

    // Call the kernel that calculates matrix-vector multiplication
    MatVec<<<BLOCKS_PER_GRID,THREAD_PER_BLOCK>>>(dev_A,dev_b,dev_x,n);
    HANDLE_ERROR(cudaThreadSynchronize());

    // Copy the result to the CPU
    //HANDLE_ERROR(cudaMemcpy(x,dev_x,n*sizeof(double),cudaMemcpyDeviceToHost));
    //printVector("Vector s",x,n);
    //free(x);
}

void conjGrad (double *dev_A, double *dev_b, double *dev_x, double *dev_r, double *dev_s, double *dev_p, double *dev_pc, double *pc, \
  double *x, int n)
{
  int k = 0;
  double *r, *dev_r_ant;
  double alpha, beta;

  // Allocate memory for temporary vectors
  r = (double*)malloc(sizeof(double)*n);
  HANDLE_ERROR(cudaMalloc((void**)&dev_r_ant,sizeof(double)*n));
  // Copy vector 'r0' from device to host
  HANDLE_ERROR(cudaMemcpy(r,dev_r,sizeof(double)*n,cudaMemcpyDeviceToHost));

  while (calcNorm(r,n) > TOLER && k < MAX_ITER)
  {
    k++;
    if (k == 1)
      HANDLE_ERROR(cudaMemcpy(dev_p,dev_r,sizeof(double)*n,cudaMemcpyDeviceToDevice));
    else
    {
      beta = dotProduct(dev_r,dev_r,dev_pc,pc,n) / dotProduct(dev_r_ant,dev_r_ant,dev_pc,pc,n);
      // Se der merda pode ser isso aqui
      saxpy2(dev_p,dev_r,beta,n);
    }
    // Copy the last vector 'r'
    HANDLE_ERROR(cudaMemcpy(dev_r_ant,dev_r,sizeof(double)*n,cudaMemcpyDeviceToDevice));

    matvec(dev_A,dev_p,dev_s,n);
    alpha = dotProduct(dev_r,dev_r,dev_pc,pc,n) / dotProduct(dev_p,dev_s,dev_pc,pc,n);
    saxpy(dev_x,dev_p,alpha,n);
    saxpy(dev_r,dev_s,-alpha,n);

    // Copy vector 'r' from device to host
    HANDLE_ERROR(cudaMemcpy(r,dev_r,sizeof(double)*n,cudaMemcpyDeviceToHost));
  }

  // Copy the solution vector 'x' to the CPU
  HANDLE_ERROR(cudaMemcpy(x,dev_x,sizeof(double)*n,cudaMemcpyDeviceToHost));

  free(r);
  cudaFree(dev_r_ant);

  printf("Number of iterations = %d\n",k);
}

int main (int argc, char *argv[])
{
  if (argc-1 < 3)
  {
    printf("============== CUDA CONJUGATE GRADIENT =================\n");
    printf("Usage:> %s <n> <TOLER> <MAX_ITER>\n",argv[0]);
    printf("<n> = Order of the system\n");
    printf("<TOLER> = Tolerance\n");
    printf("<MAX_ITER> = Maximum number of iterations\n");
    exit(1);
  }
  else
  {
    cudaEvent_t start, stop;
    int n;
    double *A, *b, *x;                                             // CPU pointers
    double *dev_A, *dev_b, *dev_x, *dev_p, *dev_r, *dev_s;         // GPU pointers
    double *pc, *dev_pc;

    // Read command line arguments
    n = atoi(argv[1]);
    TOLER = atof(argv[2]);
    MAX_ITER = atof(argv[3]);

    // Allocate memory on host
    A = (double*)malloc(sizeof(double)*n*n);
    b = (double*)malloc(sizeof(double)*n);
    x = (double*)calloc(n,sizeof(double));                      // Initial guess x0 = 0
    pc = (double*)malloc(sizeof(double)*BLOCKS_PER_GRID);

    // Allocate memory on the device
    HANDLE_ERROR(cudaMalloc((void**)&dev_A,sizeof(double)*n*n));
    HANDLE_ERROR(cudaMalloc((void**)&dev_b,sizeof(double)*n));
    HANDLE_ERROR(cudaMalloc((void**)&dev_x,sizeof(double)*n));
    HANDLE_ERROR(cudaMalloc((void**)&dev_p,sizeof(double)*n));
    HANDLE_ERROR(cudaMalloc((void**)&dev_r,sizeof(double)*n));
    HANDLE_ERROR(cudaMalloc((void**)&dev_s,sizeof(double)*n));
    HANDLE_ERROR(cudaMalloc((void**)&dev_pc,sizeof(double)*BLOCKS_PER_GRID));

    // Read the input
    readMatrix_A(A,n);
    readVector_b(b,n);

    printMatrix("Matrix A",A,n);
    printVector("Vector b",b,n);

    // Start the clock for timing
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start,0));

    // Copy vectors to device
    HANDLE_ERROR(cudaMemcpy(dev_A,A,sizeof(double)*n*n,cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b,b,sizeof(double)*n,cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_x,x,sizeof(double)*n,cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_r,b,sizeof(double)*n,cudaMemcpyHostToDevice));

    // Call the CG method
    conjGrad(dev_A,dev_b,dev_x,dev_r,dev_s,dev_p,dev_pc,pc,x,n);

    // Calculate the time elapsed
    HANDLE_ERROR(cudaEventRecord(stop,0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,start,stop));

    // Print time elapsed
    printf("[!] Time elapsed for the cudaCG = %e ms\n",elapsedTime);
    printVector("Solution vector",x,n);
    testSolution(A,b,x,n);

    // Free memory
    free(A);
    free(b);
    free(x);
    free(pc);
    cudaFree(dev_A);
    cudaFree(dev_b);
    cudaFree(dev_x);
    cudaFree(dev_p);
    cudaFree(dev_r);
    cudaFree(dev_s);
    cudaFree(dev_pc);
    return 0;

  }
}
