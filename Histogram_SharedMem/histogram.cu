// Versao paralela em GPU do problema do histograma usando memoria compartilhada
// Tempo = 7.608275e+01 ms

#include <stdio.h>
#include <stdlib.h>
#include "book.h"

const int SIZE = (100*1024*1024);

// Kernel da GPU usando operacao atomica na memoria global
__global__ void histo_kernel (unsigned char *buffer, long size, unsigned int *histo)
{
  // Histograma local de cada bloco
  __shared__ unsigned int temp[256];
  // Cada thread zera sua parte
  temp[threadIdx.x] = 0;
  // Sincroniza essa mudanca na memoria compartilhada
  __syncthreads();
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int offset = blockDim.x * gridDim.x;
  while (i < size)
  {
    // Somar no histograma compartilhado os valores de cada thread
    atomicAdd(&temp[buffer[i]],1);
    i += offset;
  }
  __syncthreads();
  // Aqui cada bloco possui um histograma local de tamanho 256
  // Vamos fazer cada thread somar o valor do histograma na sua
  // posicao para o vetor do histograma global 'histo'
  atomicAdd(&(histo[threadIdx.x]),temp[threadIdx.x]);
  // Isso funcionou pq escolhemos um numero de threads igual
  // ao numero de 'bins' do histograma
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
    printf("Number of multiprocessor = %d\n",blocks);
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
