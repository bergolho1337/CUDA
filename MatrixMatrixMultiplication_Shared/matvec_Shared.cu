// Multiplicacao matriz-matriz em CUDA utilizando memoria compartihada entre as threads
// Input:   A (m x n)
//          B (n x p)
// Output:  C (m x p)

#include <stdio.h>
#include <stdlib.h>
#include "book.h"

const int BLOCK_SIZE = 2;          // Numero de blocos

// Estrutura de uma Matriz
// M(row,col) = *(M.elements + row * M.stride + col)
typedef struct Matrix
{
  int width;                  // Largura
  int height;                 // Altura
  int stride;                 // Espacamento das sub-matrizes
  float *elements;           // Elementos
}Matrix;

// Captura um elemento da matriz da memoria do Device
__device__ float GetElement (const Matrix A, int row, int col)
{
  return (A.elements[row * A.stride + col]);
}

// Atribui um elemento na matriz na memoria do Device
__device__ void SetElement (Matrix A, int row, int col, float value)
{
  A.elements[row * A.stride + col] = value;
}

// Captura a sub-matriz BLOCK_SIZExBLOCK_SIZE, Asub de A que
// esta localizada a 'col' sub-matrizes para a direita e
// 'row sub-matrizes' para baixo em relação a diagonal
// superior esquerda da matriz A
__device__ Matrix GetSubMatrix (Matrix A, int row, int col)
{
  Matrix Asub;
  Asub.width = BLOCK_SIZE;
  Asub.height = BLOCK_SIZE;
  Asub.stride = A.stride;
  Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
  return Asub;
}

// Funcoa de kernel da GPU
__global__ void MatMultKernel (const Matrix A, const Matrix B, Matrix C)
{
  int m, e;
  // Indice do bloco que estou
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;
  // Indice da thread que estou
  int row = threadIdx.y;
  int col = threadIdx.x;

  // Cada bloco computa uma sub-matriz Csub de C
  Matrix Csub = GetSubMatrix(C,blockRow,blockCol);

  // Cada thread computa um elemento de Csub acumulando o
  // resultado em Cvalue
  float Cvalue = 0;

  // Iterar ao longo de todas as sub-matrizes de A e B que sao
  // necessarias para computar Csub
  // Multiplicar cada par de sub-matrizes e acumular o resultado
  for (m = 0; m < (A.width / BLOCK_SIZE); ++m)
  {
    // Captura a sub-matriz Asub de A
    Matrix Asub = GetSubMatrix(A,blockRow,m);

    // Captura a sub-matriz Bsub de B
    Matrix Bsub = GetSubMatrix(B,m,blockCol);

    // Memoria compartilhada usada para guardar Asub e Bsub
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Carrega Asub e Bsub da memoria do Device para a
    // memoria compartilhada.
    // Cada thread carrega um elemento de cada sub-matriz
    As[row][col] = GetElement(Asub,row,col);
    Bs[row][col] = GetElement(Bsub,row,col);

    // Sincroniza as threads para garantir que todas as sub-matrizes
    // foram carregadas
    __syncthreads();

    // Multiplica Asub e Bsub
    for (e = 0; e < BLOCK_SIZE; ++e)
      Cvalue += As[row][e] * Bs[e][col];

    // Sincroniza as threads para garantir que a computacao
    // anterior foi feita antes de carregar outra sub-matriz
    // na proxima iteracao
    __syncthreads();
  }
  // Escreve Csub na memoria do Device
  // Cada thread escreve um elemento
  SetElement(Csub,row,col,Cvalue);
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
  d_A.width = A.width;
  d_A.height = A.height;
  d_A.stride = A.width;
  size_t size = A.width * A.height * sizeof(float);
  HANDLE_ERROR(cudaMalloc(&d_A.elements,size));
  HANDLE_ERROR(cudaMemcpy(d_A.elements,A.elements,size,cudaMemcpyHostToDevice));

  Matrix d_B;
  d_B.width = B.width;
  d_B.height = B.height;
  d_B.stride = B.width;
  size = B.width * B.height * sizeof(float);
  HANDLE_ERROR(cudaMalloc(&d_B.elements,size));
  HANDLE_ERROR(cudaMemcpy(d_B.elements,B.elements,size,cudaMemcpyHostToDevice));

  // Alocar memoria para C no device
  Matrix d_C;
  d_C.width = C.width;
  d_C.height = C.height;
  d_C.stride = C.width;
  size = C.width * C.height * sizeof(float);
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

  A.elements = (float*)calloc(A.width*A.height,sizeof(float));
  B.elements = (float*)calloc(B.width*B.height,sizeof(float));
  C.elements = (float*)calloc(C.width*C.height,sizeof(float));

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
