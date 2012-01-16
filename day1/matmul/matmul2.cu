//
// Perform square matrix multiplication using shared memory
//

__global__ void matmul2(const float* const a, const float* const b, const int n, float* const c)
{
    // Base indexes inside A and B
    const int ia = (blockDim.y * blockIdx.y) * n;
    const int ib = blockDim.x * blockIdx.x;
    
    // Subindex inside a "tile"
    const int tileidx = n * threadIdx.y + threadIdx.x;
    
    // Index in C
    const int ic = ia + ib + tileidx;

    float sum = 0.0f;
    int aoff = 0, boff = 0;

    // Shared memory for the "tile" sub-matrix of A and B
    __shared__ float as [BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float bs [BLOCK_SIZE][BLOCK_SIZE];

    // Go through "tiles" of size blockDim.x * blockDim.y
    for (; aoff < n; aoff += blockDim.x, boff += blockDim.y * n)
    {
       // Load the "tile" matrices from global memory to shared memory
        as [threadIdx.y][threadIdx.x] = a [ia + aoff + tileidx];
        bs [threadIdx.y][threadIdx.x] = b [ib + boff + tileidx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices
        for (int k = 0; k < BLOCK_SIZE; k++)
            sum += as [threadIdx.y][k] * bs [k][threadIdx.x];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to global memory
    // each thread writes one element
    c [ic] = sum;
}


#define kernel matmul2
#include "main.h"