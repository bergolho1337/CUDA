// Multiplicacao matriz-matriz em CUDA sem utilizar memoria compartihada entre as threads
// Input:   A (m x n)
//          B (n x p)
// Output:  C (m x p)

#include <stdio.h>
#include <stdlib.h>
#include "book.h"

const int BLOCK_SIZE = 2;          // Numero de blocos

// Estrutura de uma Matriz
// M(row,col) = *(M.elements + row * M.width + col)
typedef struct Matrix
{
  int width;                  // Largura
  int height;                 // Altura
  double *elements;           // Elementos
}Matrix;

// Funcoa de kernel da GPU
__global__ void MatMultKernel (const Matrix A, const Matrix B, Matrix C)
{
  // Cada thread computa um elemento de C
  // acumulando o resultado em Cvalue
  double Cvalue = 0;
  // A linha da matriz eh dada pelo eixo y dos blocos e das threads
  // A coluna da matriz pelo eixo x dos blocos e das threads
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // Multiplica a linha de A pela coluna de B
  for (int e = 0; e < A.width; ++e)
    Cvalue += A.elements[row * A.width + e]*B.elements[e * B.width + col];
  // Salva o elemento
  C.elements[row * C.width + col] = Cvalue;
}

// Imprime uma matriz
void printMatrix (char *matrixName, Matrix M)
{
  int i, j;
  printf("\n========================================================================");
  printf("\n%s:\n\n",matrixName);
  for (i = 0; i < M.height; i++)
  {
    for (j = 0; j < M.width; j++)
      printf("%.2f ",M.elements[i * M.width + j]);
    printf("\n");
  }
  printf("\n========================================================================");
}

// Multiplicacao Matriz-Matriz - Codigo da CPU
void MatMult (const Matrix A, const Matrix B, Matrix C)
{
  // Carregar A e B para a memoria do device
  Matrix d_A;
  d_A.width = A.width; d_A.height = A.height;
  size_t size = A.width * A.height * sizeof(double);
  HANDLE_ERROR(cudaMalloc(&d_A.elements,size));
  HANDLE_ERROR(cudaMemcpy(d_A.elements,A.elements,size,cudaMemcpyHostToDevice));

  Matrix d_B;
  d_B.width = B.width; d_B.height = B.height;
  size = B.width * B.height * sizeof(double);
  HANDLE_ERROR(cudaMalloc(&d_B.elements,size));
  HANDLE_ERROR(cudaMemcpy(d_B.elements,B.elements,size,cudaMemcpyHostToDevice));

  // Alocar memoria para C no device
  Matrix d_C;
  d_C.width = C.width; d_C.height = C.height;
  size = C.width * C.height * sizeof(double);
  HANDLE_ERROR(cudaMalloc(&d_C.elements,size));

  // Chamar o kernel
  dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
  dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
  MatMultKernel<<<dimGrid,dimBlock>>>(d_A,d_B,d_C);

  // Ler matrix C da memoria do device
  HANDLE_ERROR(cudaMemcpy(C.elements,d_C.elements,size,cudaMemcpyDeviceToHost));

  // Imprime a matriz
  printMatrix("Matrix C",C);

  // Libera memoria
  cudaFree(d_A.elements);
  cudaFree(d_B.elements);
  cudaFree(d_C.elements);

}

int main ()
{
  int i, j;
  Matrix A, B, C;

  // Aloca memoria
  A.width = A.height = 4;
  B.width = B.height = 4;
  C.width = B.width; C.height = A.height;

  A.elements = (double*)calloc(A.width*A.height,sizeof(double));
  B.elements = (double*)calloc(B.width*B.height,sizeof(double));
  C.elements = (double*)calloc(C.width*C.height,sizeof(double));

  // Leitura das matrizes
  for (i = 0; i < A.height; i++)
    for (j = 0; j < A.width; j++)
      A.elements[i * A.width + j] = j+1;
  for (i = 0; i < B.height; i++)
    B.elements[i * B.width + i] = 1;

  // Imprime as matrizes
  printMatrix("Matrix A",A);
  printMatrix("Matrix B",B);

  // Chama a funcao que copia as matrizes para GPU e que chama o kernel
  MatMult(A,B,C);

  // Libera memoria
  free(A.elements);
  free(B.elements);
  free(C.elements);

  return 0;
}
