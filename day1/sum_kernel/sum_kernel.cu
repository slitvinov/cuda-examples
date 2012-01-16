// Kernel is executed on numerous parallel threads.
__global__ void sum_kernel ( float * a, float * b, float * c )
{
    // Global thread index.
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Perform computations in entire thread.	
    c [idx] = a [idx] + b [idx];
}

#include <stdio.h>

int sum_host( float * a, float * b, float * c, int n )
{
    int nb = n * sizeof ( float );
    float * aDev = NULL;
    float * bDev = NULL;
    float * cDev = NULL;

    // Allocate global memory on GPU.
    cudaError_t cuerr = cudaMalloc ( (void**)&aDev, nb );
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot allocate GPU memory for aDev: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }
    cuerr = cudaMalloc ( (void**)&bDev, nb );
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot allocate GPU memory for bDev: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }
    cuerr = cudaMalloc ( (void**)&cDev, nb );
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot allocate GPU memory for cDev: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }

    // Setup GPU compute grid configuration.
    dim3 threads = dim3(BLOCK_SIZE, 1);
    dim3 blocks  = dim3(n / BLOCK_SIZE, 1);

    // Copy input data from host to GPU global memory.
    cuerr = cudaMemcpy ( aDev, a, nb, cudaMemcpyHostToDevice );
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy data from a to aDev: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }
    cuerr = cudaMemcpy ( bDev, b, nb, cudaMemcpyHostToDevice );
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy data from b to bDev: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }

    // Execute kernel with the specified config and args.
    sum_kernel<<<blocks, threads>>> (aDev, bDev, cDev);
    cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot launch CUDA kernel: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }

    // Wait for kernel to finish.
    cuerr = cudaDeviceSynchronize();
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }

    // Copy results back to the host memory.
    cuerr = cudaMemcpy ( c, cDev, nb, cudaMemcpyDeviceToHost );
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy data from cdev to c: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }

    // Release GPU memory.
    cudaFree ( aDev );
    cudaFree ( bDev );
    cudaFree ( cDev );

    return 0;
}

#include <malloc.h>
#include <stdlib.h>

int main ( int argc, char* argv[] )
{
    if (argc != 2)
    {
        printf("Usage: %s <n>\n", argv[0]);
        printf("Where n must be a multiplier of %d\n", BLOCK_SIZE);
        return 0;
    }

    int n = atoi(argv[1]), nb = n * sizeof(float);
    printf("n = %d\n", n);
    if (n <= 0)
    {
        fprintf(stderr, "Invalid n: %d, must be positive\n", n);
        return 1;
    }
    if (n % BLOCK_SIZE)
    {
        fprintf(stderr, "Invalid n: %d, must be a multiplier of %d\n",
            n, BLOCK_SIZE);
        return 1;
    }

    float* a = (float*)malloc(nb);
    float* b = (float*)malloc(nb);
    float* c = (float*)malloc(nb);
    double idrandmax = 1.0 / RAND_MAX;
    for (int i = 0; i < n; i++)
    {
        a[i] = rand() * idrandmax;
        b[i] = rand() * idrandmax;
    }

    int status = sum_host (a, b, c, n);
    if (status) return status;

    int imaxdiff = 0;
    float maxdiff = 0.0;
    for (int i = 0; i < n; i++)
    {
        float diff = c[i] / (a[i] + b[i]);
        if (diff != diff) diff = 0; else diff = 1.0 - diff;
        if (diff > maxdiff)
        {
            maxdiff = diff;
            imaxdiff = i;
        }
    }
    printf("Max diff = %f% @ i = %d: %f != %f\n",
        maxdiff * 100, imaxdiff, c[imaxdiff],
        a[imaxdiff] + b[imaxdiff]);
    return 0;
}

