#ifdef USE_DOUBLE
typedef double realtype;
#else
typedef float realtype;
#endif

__constant__ realtype PI_NUMBER;

// The kernel to be executed in many threads
__global__ void sine_kernel ( realtype Period, realtype * result )
{
  // Global thread index
  // ...
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
  // Do the calculations, corresponding to the thread. Use PI_NUMBER constant!
  // ...
  result[idx] = sin(2.0 * PI_NUMBER / Period * (realtype) idx);
}

#include <stdio.h>

int sine_device( realtype Period, size_t n, realtype *result )
{
  int nb = n * sizeof ( realtype );
  realtype * resultDev = NULL;

  // Allocate memory on GPU
  cudaError_t cuerr = cudaMalloc ( (void**)&resultDev, nb );
  if (cuerr) {
    fprintf(stderr, "%s:%d: Error in cudaMalloc\n", __FILE__, __LINE__);
    return 1;
  }

  // Send PI number to constant memory PI_NUMBER
  // cuerr = cudaMemcpyToSymbol (...
  realtype pi_number = (realtype)M_PI;
  cuerr = cudaMemcpyToSymbol ( PI_NUMBER, &pi_number, sizeof(realtype), 0,  cudaMemcpyHostToDevice );
  if (cuerr) {
    fprintf(stderr, "%s:%d: Error in cudaMemcpyToSymbol\n", __FILE__, __LINE__);
    return 1;
  }
    
  // Set up the kernel launch configuration for n threads 
  // (note BLOCK_SIZE is a pre-defined macro value!)
  dim3 threads = dim3(BLOCK_SIZE, 1);
  dim3 blocks  = dim3(n / BLOCK_SIZE, 1);

  // Launch the kernel using the configuration set up before
  // ...    
  // Copy input data from host to GPU global memory.
  sine_kernel<<<blocks, threads>>>(Period, resultDev);
  cuerr = cudaGetLastError();
  if (cuerr) {
    fprintf(stderr, "%s:%d: Error after sine_kernel\n", __FILE__, __LINE__);
    return 1;
  }


  // Wait the kernel to be finished (cudaDeviceSynchronize)
  // ...
  cuerr = cudaDeviceSynchronize();

  // Copy the results back to CPU memory
  // cuerr = cudaMemcpy (...
  cuerr = cudaMemcpy ( result, resultDev, nb, cudaMemcpyDeviceToHost );
  if (cuerr) {
    fprintf(stderr, "%s:%d: Error after cudaMemcpy\n", __FILE__, __LINE__);
    return 1;
  }

  // Free GPU memory
  // cudaFree (...
  cudaFree ( resultDev );

  return 0;
}

#include <malloc.h>
#include <stdlib.h>

realtype original_function(int i, realtype Period) {
  return sinf(2.0 * realtype(M_PI) / Period * realtype(i));
} 

int main ( int argc, char* argv[] )
{
  fprintf(stderr, "LOG: ===start sine_calc===\n");
  if (argc != 2)
    {
      printf("Usage: %s <n>\n", argv[0]);
      printf("Where n must be a multiplier of %d\n", BLOCK_SIZE);
      return 0;
    }

  int n = atoi(argv[1]), nb = n * sizeof(realtype);
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

  realtype Period = 256.0;

  realtype * result = (realtype*)malloc(nb);

  fprintf(stderr, "LOG: befor sine_device\n");
  int status = sine_device (Period, n, result);
  if (status) {
    fprintf(stderr, "sine_device returns error\n");
    return status;
  }
  fprintf(stderr, "LOG: after sine_device\n");

  int imaxdiff = 0;
  realtype maxdiff = 0.0;
  realtype maxdiff_good = 0.0;
  realtype maxdiff_bad = 0.0;
  for (int i = 0; i < n; i++)
    {
      const realtype gold = original_function(i, Period); 
      realtype diff = result[i] / gold;
      if (diff != diff) diff = 0; else diff = 1.0 - diff;
      if (diff > maxdiff)
        {
	  maxdiff = diff;
	  imaxdiff = i;
	  maxdiff_good = gold;
	  maxdiff_bad = result[i];
        }
    }
  printf("Max diff = %f% @ i = %d: %f != %f\n",
	 maxdiff * 100, imaxdiff, maxdiff_bad, maxdiff_good);
  fprintf(stderr, "LOG: ===finish sine_calc===\n");
  return 0;
}

