#include <cstdlib>
#include <cstdio>
#include "particle.h"

// Kernel que move as particulas. (Cada thread move uma particula)
__global__
void advanceParticles (float dt, particle *pArray, int nParticles)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < nParticles) { pArray[idx].advance(dt); }
}

// Pode-se passar 2 parametros para o programa
// O primeiro eh o numero de particulas
// O segundo eh uma semente para gerar valores diferentes para as particulas e suas trajetorias.
int main (int argc, char *argv[])
{
  int n = 1000000;
  if (argc > 1) { n = atoi(argv[1]); }            // Numero de particulas
  if (argc > 2) { srand(atoi(argv[2])); }         // Semente random

  particle *pArray = new particle[n];
  particle *devPArray = NULL;
  cudaMalloc(&devPArray,n*sizeof(particle));
  cudaMemcpy(devPArray,pArray,n*sizeof(particle),cudaMemcpyHostToDevice);
  // Executar 100 movimentos para as particulas
  for (int i = 0; i < 100; i++)
  {
    // Distancia random a cada passo
    float dt = (float)rand() / (float) RAND_MAX;
    advanceParticles<<< 1 + n/256, 256>>>(dt,devPArray,n);
    // Garante que somente teremos um kernel executando por vez
    cudaDeviceSynchronize();
  }
  cudaMemcpy(pArray,devPArray,n*sizeof(particle),cudaMemcpyDeviceToHost);
  v3 totalDistance(0,0,0);
  v3 temp;
  for (int i = 0; i < n; i++)
  {
      temp = pArray[i].getTotalDistance();
      totalDistance.x += temp.x;
      totalDistance.y += temp.y;
      totalDistance.z += temp.z;
  }
  float avgX = totalDistance.x /(float)n;
  float avgY = totalDistance.y /(float)n;
  float avgZ = totalDistance.z /(float)n;
  float avgNorm = sqrt(avgX*avgX + avgY*avgY + avgZ*avgZ);
  printf("Moved %d particles 100 steps. Average distance traveled is |(%f, %f, %f)| = %f\n",
                                          n, avgX, avgY, avgZ, avgNorm);
  return 0;
}
