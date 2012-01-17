#include <cuda_runtime.h>
#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const double sigma = 0.1;
double sigma2, sigma4;

double s(double x, double y) {
	return (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5);
}

double rho(double x, double y) {
	const double ss = s(x, y);
	return (ss - 2 * sigma2) * exp(-ss / (2 * sigma2)) / sigma4;
}

double u0(double x, double y) {
	return exp(-s(x,y) / (2 * sigma2));
}

double wave_num2(int i, int j, int n) {
	if (!i && !j) return 1.0;

	double wn1 = i < n / 2 ? i : i - n;
	double wn2 = j < n / 2 ? j : j - n;

	return -4 * M_PI * M_PI * (wn1 * wn1 + wn2 * wn2);
}

int fft_cpu(int n, double* u) {
	const double h = 1.0 / n;

	fftw_complex* v = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (n / 2 + 1) * n);

	fftw_plan forward = fftw_plan_dft_r2c_2d(n, n, u, v, FFTW_ESTIMATE);
	fftw_plan inverse = fftw_plan_dft_c2r_2d(n, n, v, u, FFTW_ESTIMATE);

	fftw_execute(forward);

	for (int j = 0; j < n; j++)
		for (int i = 0; i < (n / 2 + 1); i++)
			v[j * (n / 2 + 1) + i][0] /= wave_num2(i, j, n);

	fftw_execute(inverse);

	double shift = u[0];
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			u[j * n + i] -= shift;
			u[j * n + i] /= n * n;
		}
	}

	fftw_destroy_plan(forward);
	fftw_destroy_plan(inverse);

	fftw_free(v);

	return 0;
}

int fft_gpu(int n, double* u);

int main(int argc, char *argv[])
{
  fprintf(stderr, "Running poisson_cpu\n");
	if (argc != 2)
	{
		printf("Usage: %s <n>\n", argv[0]);
		return 0;
	}

	int n = atoi(argv[1]);
	size_t size = sizeof(double) * n * n;

	fprintf(stderr, "sigma: %f\n", sigma);
	sigma2 = sigma  * sigma;
	sigma4 = sigma2 * sigma2;

	double* u_cpu = (double*)fftw_malloc(size);
	const double h = 1.0 / n;
	for(int i = 0; i < n; i++)
		for(int j = 0; j < n; j++)
			u_cpu[j * n + i] = rho(i * h, j * h);

	double* u_gpu_host = (double*)fftw_malloc(size);
	memcpy(u_gpu_host, u_cpu, size);

	double* u_gpu_dev = NULL;
	cudaError_t cuerr = cudaMalloc((void**)&u_gpu_dev, size);
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot allocate device buffer for u: %s\n",
			cudaGetErrorString(cuerr));
		return 1;
	}

	cuerr = cudaMemcpy(u_gpu_dev, u_gpu_host, size, cudaMemcpyHostToDevice);
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy u data from host to device: %s\n",
			cudaGetErrorString(cuerr));
		return 1;
	}

	int istat = fft_cpu(n, u_cpu);
	if (istat) return istat;

	istat = fft_gpu(n, u_gpu_dev);
	if (istat) return istat;

	cuerr = cudaMemcpy(u_gpu_host, u_gpu_dev, size, cudaMemcpyDeviceToHost);
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy u data from device to host: %s\n",
			cudaGetErrorString(cuerr));
		return 1;
	}
	
	int imaxdiff = 1;
	double maxdiff = 0.0;
	for (int i = 0; i < n * n; i++)
	{
		double diff = u_cpu[i] / u_gpu_host[i];
		if (diff != diff)
			diff = 1.0;
		else
		{
			if (diff > 1.0)
				diff = 1.0 - 1.0 / diff;
			else
				diff = 1.0 - diff;
			if (diff > maxdiff)
			{
				maxdiff = diff;
				imaxdiff = i;
			}
		}
		const float x = i/n;
		const float y = i%n;
		printf("%f %f %f %f\n", x, y, u_cpu[i], u_gpu_host[i]);
	}

	fftw_free(u_cpu);
	fftw_free(u_gpu_host);
	cudaFree(u_gpu_dev);

	return 0;
}

