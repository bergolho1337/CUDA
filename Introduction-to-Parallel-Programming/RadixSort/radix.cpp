#include <iostream>
#include <cstdio>
#include <cstring>

const int N = 8;

const int MAXINT = 9999;
const int MININT = 0;

// ---------------------------------------------------------------------------------------------------------------------------
// -------------------------------------------------- CPU Functions ----------------------------------------------------------

void Print (unsigned int *v)
{
    for (int i = 0; i < N; i++)
        printf("%4u\n",v[i]);
    printf("\n");
}

void GetInput (unsigned int *v)
{
    for (int i = 0; i < N; i++)
        scanf("%u",&v[i]); 
} 

void RadixSort (unsigned int *in, unsigned int *out)
{
    const int numBits = 1;
    const int numBins = 2;          // Bin 0 = Number of 0's || Bin 1 = Number of 1's

    unsigned int *binHistogram = new unsigned int[numBins];
    unsigned int *binScan = new unsigned int[numBins];

    // An unsigned int has 32 bits, so we will make a mask for each bit
    for (int i = 0; i < 8 * sizeof(unsigned int); i += numBits)
    {
        // Create the mask
        unsigned int mask = numBits << i;
        
        // Reset the bins
        memset(binHistogram,0,sizeof(unsigned int)*numBins);
        memset(binScan,0,sizeof(unsigned int)*numBins);

        // For each element test the mask and compute the histogram of how many 0's and 1's we had
        for (int j = 0; j < N; j++)
        {
            // Check if the current bit is set on the current element
            // If so, shift to the right to get index of the bin
            unsigned int bin = (in[j] & mask) >> i;
            binHistogram[bin]++;
        }

        // Now do a scan operation on the histogram to get the start position of each bin
        for (int j = 1; j < numBins; j++)
        {
            binScan[j] = binScan[j-1] + binHistogram[j-1];
        }

        // Finally compute where each element will be using the scan array
        for (int j = 0; j < N; j++)
        {
            unsigned int bin = (in[j] & mask) >> i;
            out[binScan[bin]] = in[j];
            binScan[bin]++;                         // The element is on the correct position, so update the index for the next one
        }

        // Swap the pointers
        std::swap(out,in);
    }

    std::copy(in,in + N,out);

    delete [] binHistogram;
    delete [] binScan;
}

void Usage (char pName[])
{
    printf("============================================\n");
    printf("Usage:> %s \n",pName);
    printf("============================================\n");
}

// ---------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------ MAIN FUNCTION ------------------------------------------------------------

int main (int argc, char *argv[])
{
    if (argc-1 != 0)
    {
        Usage(argv[0]);
        exit(1);
    }

    // Declare and allocate memory for the host
    unsigned int *in, *out;
    in = new unsigned int[N]();
    out = new unsigned int[N]();

    // Get or Generate the array to sort
    GetInput(in);
    Print(in);

    // Sort the array using RadixSort
    RadixSort(in,out);

    // Print the result
    Print(out);

    return 0;
}