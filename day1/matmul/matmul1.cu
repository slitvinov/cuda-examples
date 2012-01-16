//
// Perform "naive" square matrix multiplication
//

__global__ void matmul1 (float* a, float* b, int n, float* c)
{
  //  const int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Calculate start indexes for row and column
    int ia = (blockDim.y * blockIdx.y + threadIdx.y) * n;
    int ib = blockDim.x * blockIdx.x + threadIdx.x;
    int ic = ia + ib;

    float sum = 0;
    // Multiply two matrices
    for (int k = 0; k < n; k++) {
      
      sum += a [ia + k] * b [ib + k * n];
    }

    // Write the block sub-matrix to global memory;
    c[ic] = sum; 
}

#define kernel matmul1
#include "main.h"

