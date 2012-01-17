#include <cufft.h>
#include <stdio.h>

#define SIGMA      (cufftDoubleReal(0.1))
#define SIGMA2     (SIGMA  * SIGMA)
#define SIGMA4     (SIGMA2 * SIGMA2)

__global__ void set_rhs(int n, cufftDoubleReal* rhs) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < n && j < n) {
		cufftDoubleReal h = 1 / cufftDoubleReal(n);
		cufftDoubleReal x = i * h - cufftDoubleReal(0.5);
		cufftDoubleReal y = j * h - cufftDoubleReal(0.5);
		cufftDoubleReal s = x * x + y * y;

		rhs[j * n + i] = (s - 2 * SIGMA2) * exp(-s / (2 * SIGMA2)) / SIGMA4;
	}
}

__global__ void solve_transformed(int n,
	const cufftDoubleComplex* rhs, cufftDoubleComplex* u) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	int m = n / 2 + 1;
	if (i < m && j < n) {
		cufftDoubleComplex t = rhs[j * m + i];

		cufftDoubleReal w = cufftDoubleReal(M_PI) * (i < n/2 ? i : i - n);
		cufftDoubleReal v = cufftDoubleReal(M_PI) * (j < n/2 ? j : j - n);
		cufftDoubleReal s = (!i && !j) ? 1 : -4 * (w * w + v * v);

		t.x /= s;
		t.y /= s;

		u[j * m + i] = t;
	}
}

__global__ void scale_and_shift(int n, cufftDoubleReal* u, double shift) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if (i < n && j < n)
		u[j * n + i] = (u[j * n + i] - shift) / (n * n);
}

extern "C" int fft_gpu(int n, double* u) {
	dim3 blk (32, 2);
	dim3 rgrd((n + blk.x - 1) / blk.x, (n + blk.y - 1) / blk.y);
	dim3 cgrd((n/2 + blk.x) / blk.x, (n + blk.y - 1) / blk.y);
	double shift;

	cufftDoubleComplex* v = NULL;
	cudaError_t cuerr = cudaMalloc((void**)&v, n * (n / 2 + 1) * sizeof(cufftDoubleComplex));
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot create gpu memory buffer for v: %s\n",
			cudaGetErrorString(cuerr));
		return 1;
	}

	set_rhs<<<rgrd, blk>>>(n, u);

	cufftHandle forward, inverse;
	cufftPlan2d(&forward, n, n, CUFFT_D2Z);
	cufftPlan2d(&inverse, n, n, CUFFT_Z2D);
	cufftExecD2Z(forward, u, v);

	solve_transformed<<<cgrd, blk>>>(n, v, v);
	cufftExecZ2D(inverse, v, u);
	cudaMemcpy(&shift, u, sizeof(double), cudaMemcpyDeviceToHost);
	scale_and_shift<<<rgrd, blk>>>(n, u, shift);
	cudaFree(v);
	
	cufftDestroy(forward);
	cufftDestroy(inverse);

	return 0;
}