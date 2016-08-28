// Versao serial do problema do histograma na CPU
// Tempo = 5.811268e+02 ms

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "book.h"

const int SIZE = (100*1024*1024);

int main ()
{
    // Varaiveis de medicao de tempo
    cudaEvent_t start, stop;
    int i;

    // Vetor randomico de 'char' de tamanho SIZE
    unsigned char *buffer = (unsigned char*)big_random_block(SIZE);
    unsigned int  histo[256];

    // Inicializa o contador do histograma com 0's
    for (i = 0; i < 256; i++)
      histo[i] = 0;

    // Inicializa as variaveis de medicao
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start,0));

    // Percorre os dados do buffer contando quantas vezes
    // cada elemento apareceu
    for (i = 0; i < SIZE; i++)
      histo[buffer[i]]++;

    // Para o cronometro e calcula o tempo
    HANDLE_ERROR(cudaEventRecord(stop,0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,start,stop));

    // Confere se a conta esta correta e imprime o tempo em milisegundos
    long histoCount = 0;
    for (i = 0; i < 256; i++)
      histoCount += histo[i];
    printf("Histogram Sum: %ld\n",histoCount);
    printf("This result must the same as SIZE: %ld\n",SIZE);
    printf("Time to generate: %e ms\n",elapsedTime);
    return 0;
}
