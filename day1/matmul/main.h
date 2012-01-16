#include <cublas_v2.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>

void printmat(float* m, int n)
{
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < n; i++)
            printf("%f ", m[i + j * n]);
        printf("\n");
    }
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        printf("Usage: %s <n>\n", argv[0]);
        printf("Where n must be > %d and a multiplier of %d\n",
            BLOCK_SIZE, BLOCK_SIZE);
        return 0;
    }

    int n = atoi(argv[1]);
    if ((n < BLOCK_SIZE) || (n % BLOCK_SIZE))
    {
        fprintf(stderr, "Invalid n: %d\n", n);
        printf("n must be > %d and a multiplier of %d\n",
            BLOCK_SIZE, BLOCK_SIZE);
        return 1;
    }

    printf("n = %d\n", n);

    int n2 = n * n, n2b = n2 * sizeof(float);

    // Allocate host memory
    float* a = (float*)malloc(n2b);
    float* b = (float*)malloc(n2b);
    float* c1 = (float*)malloc(n2b);
    float* c2 = (float*)malloc(n2b);
    
    // create matrices
    double dinvrandmax = 1.0 / RAND_MAX;
    for (int i = 0; i < n2; i++)
    {
        a[i] = rand() * dinvrandmax;
        b[i] = rand() * dinvrandmax;
    }

    // Allocate device memory
    float* adev = NULL;
    cudaError_t cuerr = cudaMalloc((void**)&adev, n2b);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot allocate device array for a: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }
    float* bdev = NULL;
    cuerr = cudaMalloc((void**)&bdev, n2b);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot allocate device array for b: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }
    float * cdev = NULL;
    cuerr = cudaMalloc((void**)&cdev, n2b);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot allocate device array for c: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }

    // Create cuda event handles
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cuerr = cudaEventCreate(&start);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot create CUDA start event: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }
    cuerr = cudaEventCreate(&stop);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot create CUDA end event: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }

    cuerr = cudaMemcpy(adev, a, n2b, cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy a array from host to device: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }
    cuerr = cudaMemcpy(bdev, b, n2b, cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy b array from host to device: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }

    // Measure time of kernel execution
    cuerr = cudaEventRecord(start, 0);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot record CUDA event: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }

    // Set kernel launch configuration
    // note pre-given BLOCK_SIZE as a preprocessor macro
    // Set kernel launch configuration
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks( n/threads.x, n/threads.y);

    // Launch the kernel (suppose it is named "kernel")
    // ...
    kernel<<<blocks, threads>>>(adev, bdev, n, cdev);

    cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot launch CUDA kernel: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }

    cuerr = cudaDeviceSynchronize();
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }

    cuerr = cudaEventRecord(stop, 0);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy c array from device to host: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }
    cuerr = cudaMemcpy(c1, cdev, n2b, cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy c array from device to host: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }

    cuerr = cudaEventElapsedTime(&gpuTime, start, stop);
    printf("time spent executing %s: %.2f millseconds\n", "kernel", gpuTime);

    // Measure time of CUBLAS gemm execution
    cuerr = cudaEventRecord(start, 0);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot record CUDA event: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }

    cublasHandle_t handle;
    cublasStatus_t cberr = cublasCreate_v2(&handle);
    if (cberr != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "Cannot create cublas handle: %d\n", cberr);
        return 1;
    }

    float alpha = 1.0, beta = 0.0;
    cberr = cublasSgemm_v2(
        handle, CUBLAS_OP_T, CUBLAS_OP_T, n, n, n,
        &alpha, adev, n, bdev, n, &beta, cdev, n);
    if (cberr != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "Error launching cublasSgemm_v2: %d\n", cberr);
        return 1;
    }

    cuerr = cudaDeviceSynchronize();
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot synchronize kernel: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }

    cberr = cublasDestroy_v2(handle);
    if (cberr != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "Cannot destroy cublas handle: %d\n", cberr);
        return 1;
    }

    cuerr = cudaEventRecord(stop, 0);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy c array from device to host: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }
    cuerr = cudaMemcpy(c2, cdev, n2b, cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy c array from device to host: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }

    cuerr = cudaEventElapsedTime(&gpuTime, start, stop);
    printf("time spent executing %s: %.2f millseconds\n",
        "cublasSgemm_v2", gpuTime);

    if (n <= 8)
    {
        printf("A =\n"); printmat(a, n);
        printf("B =\n"); printmat(b, n);
        printf("C1 = \n"); printmat(c1, n);
        printf("C2 = \n"); printmat(c2, n);
    }

    // Compare results
    int imaxdiff = 0, jmaxdiff = 0;
    double maxdiff = 0;
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < n; i++)
        {
            double diff = (double)c1[i + j * n] / c2[j + i * n];
            if (diff != diff) diff = 0.0; else diff = 1.0 - diff;
            if (diff > maxdiff)
            {
                maxdiff = diff;
                imaxdiff = i;
                jmaxdiff = j;
            }
        }
    }
    printf("max diff = %.2f% @ [%d, %d]: %f != %f\n",
        maxdiff * 100, imaxdiff, jmaxdiff,
        c1[imaxdiff + jmaxdiff * n],
        c2[jmaxdiff + imaxdiff * n]);

    // Release resources
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(adev);
    cudaFree(bdev);
    cudaFree(cdev);
    free(a);
    free(b);
    free(c1);
    free(c2);

    return 0;
}
