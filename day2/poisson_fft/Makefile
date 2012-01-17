NVCC = nvcc -g -O3 -arch=sm_13

all: poisson_fft

poisson_fft: poisson_cpu.o poisson_gpu.o
	$(NVCC) poisson_cpu.o poisson_gpu.o -o $@ -lcufft -lfftw3

poisson_cpu.o: poisson_cpu.c
	$(NVCC) -D_GNU_SOURCE -Xcompiler -std=c99 -c $< -o $@

poisson_gpu.o: poisson_gpu.cu
	$(NVCC) -c $< -o $@

clean:
	rm -rf *.o poisson_fft

