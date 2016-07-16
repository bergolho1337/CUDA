#include <stdio.h>
#include <stdlib.h>
#include "cpu_bitmap.h"

int main ()
{
  FILE *file = fopen("julia.txt","r");
  CPUBitmap bitmap(file);
  fclose(file);

  // Mostra a imagem
  bitmap.display_and_exit();

  return 0;
}
