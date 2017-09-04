#include <iostream>

using namespace std;

const int NUMCOLS = 8;
const int BLOCKSIZE = 2;
const int GRIDSIZE = 4;

int main ()
{
    // In a grid loop for each block(x,y)
    for (int a = 0; a < GRIDSIZE; a++)
    {
        for (int b = 0; b < GRIDSIZE; b++)
        {
            cout << "Block - " << a << "," << b << endl;
            // In a block loop for each thread(x,y)
            for (int c = 0; c < BLOCKSIZE; c++)
            {
                for (int d = 0; d < BLOCKSIZE; d++)
                {
                    int row = a * BLOCKSIZE + c;
                    int col = b * BLOCKSIZE + d;
                    int tid = row * NUMCOLS + col;
                    cout << "\tThread - " << c << "," << d;
                    cout << " - Element " << tid << endl;
                }
            }
        }
    }
}