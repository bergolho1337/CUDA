// Programa que constroi um JuliaSet e exporta a imagem como .bmp.
// Exemplo: Cuda_By_Example Chapter 4 (A Fun Example)

#include <stdio.h>
#include <stdlib.h>
#include "cpu_bitmap.h"         // Biblioteca que controla a imagem de saída no formato .bmp usando o OpenGL
#include "book.h"               // Biblitoeca do livro que contém MACROS para ao CUDA

#define DIM 100                 // Dimensao da iamgem

// Classe de um numero complexo
struct cuComplex
{
  // Atributos
  float r;            // Parte real
  float i;            // Parte imaginaria

  // Construtor
  __device__ cuComplex (float a, float b) : r(a), i(b) {}

  // Metodos. __device__ -> Essas funcoes so podem ser executadas pelo device
  __device__ float magnitude2 ()
  {
    return (r*r + i*i);
  }

  // Sobrecarga de operadores
  __device__ cuComplex operator* (const cuComplex &a)
  {
    return (cuComplex(r*a.r - i*a.i, i*a.r + r*a.i));
  }

  __device__ cuComplex operator+ (const cuComplex &a)
  {
    return (cuComplex(r+a.r,i+a.i));
  }
}typedef cuComplex;

// Funcao da que calcula o JuliaSet na GPU
__device__ int julia (int x, int y)
{
  const float scale = 1.5;        // Escala para aumentar o zoom da imagem
  // Calcula a posicao do ponto nas coordenadas complexas
  float jx = scale*(float)(DIM/2 - x)/(DIM/2);
  float jy = scale*(float)(DIM/2 - y)/(DIM/2);

  cuComplex c(-0.8,0.156);
  cuComplex a(jx,jy);

  int i;
  for (i = 0; i < 200; i++)
  {
    // Z_n+1 = Z_n + C
    a = a*a + c;
    if (a.magnitude2() > 1000)
      return 0;
  }
  return 1;
}

// Funcao do kernel
__global__ void kernel (unsigned char *ptr)
{
  // Para cada pixel teremos um bloco rodando a operacao
  // Como chamamos o kernel usando um grid NxN as coordenadas
  // do pixel são iguais as coordenadas dos blocos
  int x = blockIdx.x;
  int y = blockIdx.y;
  int offset = x + y*gridDim.x;

  // Calcula o valor dessa posicao
  int juliaValue = julia(x,y);
  ptr[offset*4+0] = 255*juliaValue;
  ptr[offset*4+1] = 0;
  ptr[offset*4+2] = 0;
  ptr[offset*4+3] = 255;
}

// Funcao principal
int main ()
{
  CPUBitmap bitmap(DIM,DIM);
  unsigned char *dev_bitmap;    // Ponteiro para referencia da imagem na GPU

  // Aloca memoria para GPU
  HANDLE_ERROR(cudaMalloc( (void**)&dev_bitmap,bitmap.image_size()));

  // Define um grid bidimensional (DIMxDIM)
  dim3 grid(DIM,DIM);
  // Chama o kerne passando o grid e o ponteiro para armazenar
  // os calculos nos pixels da imagem
  kernel<<<grid,1>>>(dev_bitmap);

  // Copia o resultado da GPU para a CPU
  HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(),dev_bitmap,bitmap.image_size(),cudaMemcpyDeviceToHost));

  // Salva em arquivo .ppm a imagem
  bitmap.toTxt();
  //bitmap.display_and_exit();

  // Libera memoria
  HANDLE_ERROR(cudaFree(dev_bitmap));

  return 0;
}
