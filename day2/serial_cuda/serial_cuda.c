/*
 * MSU CUDA Course Examples and Exercises.
 *
 * Copyright (c) 2011 Dmitry Mikushin
 *
 * This software is provided 'as-is', without any express or implied warranty.
 * In no event will the authors be held liable for any damages arising 
 * from the use of this software.
 * Permission is granted to anyone to use this software for any purpose, 
 * including commercial applications, and to alter it and redistribute it freely,
 * without any restrictons.
 *
 * This sample demonstates how multiple CUDA devices can work
 * in parallel in one execution thread.
 *
 */

#include "pattern2d.h"
#include "timing.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <malloc.h>
#include <math.h>
#include <stdio.h>

// The thread configuration structure.
typedef struct
{
	float  *in_dev, *out_dev;
	struct time_t start;
}
config_t;

// The size of memory region.
int nx = 16384, ny = 16384;

int main(int argc, char* atgv[])
{
	int ndevices = 0;
	cudaError_t cuda_status = cudaGetDeviceCount(&ndevices);
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "Cannot get the cuda device count, status = %d: %s\n",
			cuda_status, cudaGetErrorString(cuda_status));
		return cuda_status;
	}
	
	// Return if no cuda devices present.
	printf("%d CUDA device(s) found\n", ndevices);
	if (!ndevices) return 0;
	
	// Create input data. Each device will have an equal
	// piece of data.
	size_t np = nx * ny, size = np * sizeof(float);
	float* data = (float*)malloc(size * 2);
	float *input = data, *output = data + np;
	float invdrandmax = 1.0 / RAND_MAX;
	for (size_t i = 0; i < np; i++)
		input[i] = rand() * invdrandmax;

	struct time_t start, finish;
	get_time(&start);

	// Get control result on CPU (to compare with results on devices).
	pattern2d_cpu(1, nx, 1, 1, ny, 1, input, output, ndevices);

	get_time(&finish);
	
	printf("CPU time = %f sec\n", get_time_diff(&start, &finish));
	
	// Create config structures to store device-specific
	// values.
	config_t* configs = (config_t*)malloc(
		sizeof(config_t) * ndevices);

	// Initialize CUDA devices.
	for (int idevice = 0; idevice < ndevices; idevice++)
	{
		config_t* config = configs + idevice;

		// TODO: Set curent CUDA device to idevice.
		cudaError_t cuda_status = cudaSetDevice(idevice);
		if (cuda_status != cudaSuccess) {
		  fprintf(stderr, "Cannot get the cuda device count, status = %d: %s\n",
			cuda_status, cudaGetErrorString(cuda_status));
		  return cuda_status;
		}

		// Create device arrays for input and output data.
		cuda_status = cudaMalloc((void**)&config->in_dev, size);
		if (cuda_status != cudaSuccess)
		{
			fprintf(stderr, "Cannot allocate CUDA input buffer on device %d, status = %d: %s\n",
				idevice, cuda_status, cudaGetErrorString(cuda_status));
			return cuda_status;
		}
		cuda_status = cudaMalloc((void**)&config->out_dev, size);
		if (cuda_status != cudaSuccess)
		{
			fprintf(stderr, "Cannot allocate CUDA output buffer on device %d, status = %d: %s\n",
				idevice, cuda_status, cudaGetErrorString(cuda_status));
			return cuda_status;
		}
	
		// Copy input data to device buffer.
		cuda_status = cudaMemcpy(config->in_dev, input, size,
			cudaMemcpyHostToDevice);
		if (cuda_status != cudaSuccess)
		{
			fprintf(stderr, "Cannot copy input data to CUDA buffer on device %d, status = %d: %s\n",
				idevice, cuda_status, cudaGetErrorString(cuda_status));
			return cuda_status;
		}
	       
		printf("Device %d initialized\n", idevice);
	}

	// Start execution of kernels. One kernel
	// is executed on each device in parallel.
	for (int idevice = 0; idevice < ndevices; idevice++)
	{
		config_t* config = configs + idevice;

		// TODO: Set curent CUDA device to idevice.
		cudaError_t cuda_status = cudaSetDevice(idevice);

		get_time(&config->start);
		
		// Run test kernel on the current device.
		int status = pattern2d_gpu(1, nx, 1, 1, ny, 1,
			config->in_dev, config->out_dev, idevice);
		if (status)
		{
			fprintf(stderr, "Cannot execute pattern 2d on device %d, status = %d: %s\n",
				idevice, status, cudaGetErrorString(status));
			return status;
		}
	}
	
	// Synchronize kernels execution.
	for (int idevice = 0; idevice < ndevices; idevice++)
	{
		config_t* config = configs + idevice;

		// TODO: Set curent CUDA device to idevice.
		cudaError_t cuda_status = cudaSetDevice(idevice);
	
		// Wait for current device to finish processing
		// the kernels.
		cuda_status = cudaThreadSynchronize();
		if (cuda_status != cudaSuccess)
		{
			fprintf(stderr, "Cannot synchronize thread by device %d, status = %d: %s\n",
				idevice, cuda_status, cudaGetErrorString(cuda_status));
			return cuda_status;
		}

		get_time(&finish);

		printf("GPU %d time = %f sec\n", idevice,
			get_time_diff(&config->start, &finish));
	}

	// Check results and dispose resources used by devices.
	for (int idevice = 0; idevice < ndevices; idevice++)
	{
		config_t* config = configs + idevice;

		// TODO: Set curent CUDA device to idevice.
	
		// Offload results back to host memory.
		cuda_status = cudaMemcpy(input, config->out_dev, size,
			cudaMemcpyDeviceToHost);
		if (cuda_status != cudaSuccess)
		{
			fprintf(stderr, "Cannot copy output data from CUDA buffer on device %d, status = %d: %s\n",
				idevice, cuda_status, cudaGetErrorString(cuda_status));
			return cuda_status;
		}

		// Free device arrays.	
		cuda_status = cudaFree(config->in_dev);
		if (cuda_status != cudaSuccess)
		{
			fprintf(stderr, "Cannot release input buffer on device %d, status = %d: %s\n",
				idevice, cuda_status, cudaGetErrorString(cuda_status));
			return cuda_status;
		}
		cuda_status = cudaFree(config->out_dev);
		if (cuda_status != cudaSuccess)
		{
			fprintf(stderr, "Cannot release output buffer on device %d, status = %d: %s\n",
				idevice, cuda_status, cudaGetErrorString(cuda_status));
			return cuda_status;
		}

		printf("Device %d deinitialized\n", idevice);

		// Compare each GPU result to CPU result.
		// Find the maximum abs difference.
		int maxi = 0, maxj = 0;
		float maxdiff = fabs(input[0] - output[0]);
		for (int j = 0; j < ny; j++)
		{
			for (int i = 0; i < nx; i++)
			{
				float diff = fabs(
					input[i + j * nx] - output[i + j * nx]);
				if (diff > maxdiff)
				{
					maxdiff = diff;
					maxi = i; maxj = j;
				}
			}
		}

		printf("Device %d result abs max diff = %f @ (%d,%d)\n",
			idevice, maxdiff, maxi, maxj);
	}
	
	// Measure time between first GPU launch and last GPU
	// finish. This will show how much time is spent on GPU
	// kernels in total.
	// XXX If this time is comparabe to the time of
	// individual GPU, then we likely reached our goal:
	// kernels are executed in parallel.
	printf("Total time of %d GPUs = %f\n", ndevices,
		get_time_diff(&configs[0].start, &finish));
	
	free(configs);
	free(data);

	return 0;
}

