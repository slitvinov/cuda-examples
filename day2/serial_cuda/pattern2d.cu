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
 */

#include "pattern2d.h"

#include <cuda_runtime.h>
#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Grid block size (see comment in pattern2d_gpu below).
#define BLOCK_LENGTH 32
#define BLOCK_HEIGHT 16

#define IN(i,j) in[i + (j) * nx]
#define OUT(i,j) out[i + (j) * nx]

// Perform some dummy 2D field processing on CPU.
int pattern2d_cpu(
	int bx, int nx, int ex, int by, int ny, int ey,
	float* in, float* out, int id)
{
	size_t size = nx * ny * sizeof(float);
	memset(out, 0, size);

	for (int j = by; j < ny - ey; j++)
		for (int i = bx; i < nx - ex; i++)
			OUT(i,j) = sqrtf(fabs(IN(i,j) + IN(i-1,j) + IN(i+1,j) -
				2.0f * IN(i,j-1) + 3.0f * IN(i,j+1))); 
	return 0;
}

// GPU device kernel.
__global__ void pattern2d_gpu_kernel(
	int bx, int nx, int ex, int by, int ny, int ey,
	int block_length, int block_height,
	float* in, float* out, int id)
{
	// Compute absolute (i,j) indexes for
	// the current GPU thread using grid mapping params.
	int i = blockIdx.x * block_length + threadIdx.x + bx;
	int j = blockIdx.y * block_height + threadIdx.y + by;
	
	// Compute one data point - a piece of
	// work for the current GPU thread.
	OUT(i,j) = sqrtf(fabs(IN(i,j) + IN(i-1,j) + IN(i+1,j) -
		2.0f * IN(i,j-1) + 3.0f * IN(i,j+1))); 
}

// Perform some dummy 2D field processing on GPU.
int pattern2d_gpu(
	int bx, int nx, int ex, int by, int ny, int ey,
	float* in, float* out, int id)
{
	// Configure GPU computational grid:
	// nx = nblocks_x * block_length
	// ny = nblocks_y * block_height
	//
	// NOTE: we have degree of freedom in
	// selecting how real problem grid maps onto
	// computational grid. Usually these params
	// are tuned to get optimal performance.
	//
	// NOTE: chose of grid/block config is
	// also limited by device properties:
	// - Maximum number of threads per block (512)
	// - Maximum sizes of each dimension of a block (512 x 512 x 64)
	// - Maximum sizes of each dimension of a grid (65535 x 65535 x 1)
	int nblocks_x = (nx - 2) / BLOCK_LENGTH;
	int nblocks_y = (ny - 2) / BLOCK_HEIGHT;
	
	// Fill output will zeros (actually, to zero the borders).
	size_t size = nx * ny * sizeof(float);
	cudaError_t status = cudaMemset(out, 0, size);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "Cannot erase output memory on GPU by device %d, status = %d\n",
			id, status);
		return status; 
	}
	
	// Invoke GPU kernel to process the 2D field.
	pattern2d_gpu_kernel<<<
		dim3(nblocks_x, nblocks_y, 1),
		dim3(BLOCK_LENGTH, BLOCK_HEIGHT, 1)>>>(
			1, nx, 1, 1, ny, 1,
			BLOCK_LENGTH, BLOCK_HEIGHT,
			in, out, id);
	status = cudaGetLastError();
	if (status != cudaSuccess)
	{
		fprintf(stderr, "Cannot execute CUDA kernel #1 by device %d, status = %d\n",
			id, status);
		return status;
	}

	// Invoke GPU kernel with reduced efficiency
	// to process the rows remainder in the bottom of 2d field.
	pattern2d_gpu_kernel<<<
		dim3(nx - 2, (ny - 2) % BLOCK_HEIGHT + 1, 1),
		dim3(1, 1, 1)>>>(
			1, nx, 1,
			ny - (ny - 2) % BLOCK_HEIGHT - 2, ny, 1,
			1, 1,
			in, out, id);
	status = cudaGetLastError();
	if (status != cudaSuccess)
	{
		fprintf(stderr, "Cannot execute CUDA kernel #2 by device %d, status = %d\n",
			id, status);
		return status;
	}

	// Invoke GPU kernel with reduced efficiency
	// to process the columns remainder on the left of 2d field.
	pattern2d_gpu_kernel<<<
		dim3((nx - 2) % BLOCK_LENGTH + 1, ny - 2, 1),
		dim3(1, 1, 1)>>>(
			nx - (nx - 2) % BLOCK_LENGTH - 2, nx, 1,
			1, ny, 1,
			1, 1,
			in, out, id);
	status = cudaGetLastError();
	if (status != cudaSuccess)
	{
		fprintf(stderr, "Cannot execute CUDA kernel #3 by device %d, status = %d\n",
			id, status);
		return status;
	}

	return 0;
}

