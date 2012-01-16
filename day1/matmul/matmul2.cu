//
// Perform square matrix multiplication using shared memory
//

__global__ void matmul2(float* a, float* b, int n, float* c)
{
    // Base indexes inside A and B, index in C
    // ...

    // Shared memory for the "tile" sub-matrix of A and B
    // ...

    // Go through "tiles" of size blockDim.x * blockDim.y
    // For each tile: load to shared memory, multiply two
    // tiles using shared memory
    // ... 

    // Write the resulting sum to global memory
    // ...
}


#define kernel matmul2
#include "main.h"

