#include "particle.h"

particle::particle() : position(), velocity(), totalDistance(0,0,0) {}

__device__ __host__
void particle::advance(float d)
{
  velocity.normalize();
  // Movimento em x
  float dx = d * velocity.x;
  position.x += dx;
  totalDistance.x += dx;
  // Movimento em y
  float dy = d * velocity.y;
  position.y += dy;
  totalDistance.y += dy;
  // Movimento em z
  float dz = d * velocity.z;
  position.z += dz;
  totalDistance.z += dz;
  // Calcula a proxima direcao
  velocity.scramble();
}

const v3& particle::getTotalDistance() const
{
  return totalDistance;
}

/*
  -- Declarar uma funcao com __device__ e __host__ diz para o 'nvcc' que a rotina deve ser compilada tanto em codigo para CPU quanto
  para GPU.
*/
