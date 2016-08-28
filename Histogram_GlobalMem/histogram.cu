// Versao paralela em GPU do problema do histograma usando memoria global
// !!! Problemas: a operacao atomica vai criar uma fila enorme para os indices que tiverem threads querendo escrever seu conteudo
// Tempo = 1.201668e+02 ms

#include <stdio.h>
#include <stdlib.h>
#include "book.h"

const int SIZE = (100*1024*1024);

// Kernel da GPU usando operacao atomica na memoria global
__global__ void histo_kernel (unsigned char *buffer, long size, unsigned int *histo)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;
  while (i < size)
  {
    // Incrementa em 1 atomicamente o valor contido no endereco especificado
    atomicAdd(&(histo[buffer[i]]),1);
    i += stride;
  }
}

int main ()
{
    int i;
    // Varaiveis de medicao de tempo
    cudaEvent_t start, stop;

    // Vetor randomico de 'char' de tamanho SIZE
    unsigned char *buffer = (unsigned char*)big_random_block(SIZE);

    // Inicializa as variaveis de medicao
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start,0));

    // Alocar memoria na GPU e copiar conteudo para lah
    unsigned char *dev_buffer;
    unsigned int *dev_histo;
    HANDLE_ERROR(cudaMalloc((void**)&dev_buffer,SIZE));
    HANDLE_ERROR(cudaMemcpy(dev_buffer,buffer,SIZE,cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaMalloc((void**)&dev_histo,256*sizeof(long)));
    HANDLE_ERROR(cudaMemset(dev_histo,0,256*sizeof(int)));

    // Descobre a quantidade de multiprocessadores do device
    // Manda a quantidade de blocos como sendo o dobro do
    // numero multiprocessadores (foi o que gerou melhor resultado)
    cudaDeviceProp prop;
    HANDLE_ERROR(cudaGetDeviceProperties(&prop,0));
    int blocks = prop.multiProcessorCount;
    // Chama o kernel do histograma
    histo_kernel<<<blocks*2,256>>>(dev_buffer,SIZE,dev_histo);

    // Copia o resultado para CPU
    unsigned int histo[256];
    HANDLE_ERROR(cudaMemcpy(histo,dev_histo,256*sizeof(int),cudaMemcpyDeviceToHost));

    // Para o cronometro e calcula o tempo
    HANDLE_ERROR(cudaEventRecord(stop,0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,start,stop));

    // Confere se a conta esta correta e imprime o tempo em milisegundos
    long histoCount = 0;
    for (i = 0; i < 256; i++)
      histoCount += histo[i];
    // Verifica se as contas batem com a CPU fazendo a operacao inversa
    for (i = 0; i < SIZE; i++)
      histo[buffer[i]]--;
    for (i = 0; i < 256; i++)
    {
      if (histo[i] != 0)
        printf("Failure at %d!\n",i);
    }
    printf("Histogram Sum: %ld\n",histoCount);
    printf("This result must the same as SIZE: %ld\n",SIZE);
    printf("Time to generate: %e ms\n",elapsedTime);

    // Liberar memoria e destruir estruturas de tempo
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    cudaFree(dev_histo);
    cudaFree(dev_buffer);
    free(buffer);
    return 0;
}
