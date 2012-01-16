BLOCK_SIZE = 512

NVCC = nvcc --compiler-options -Wall -g -O3 -DBLOCK_SIZE=$(BLOCK_SIZE)
PGF90 = pgf90 -g -O3 -DBLOCK_SIZE=$(BLOCK_SIZE)

all: sum_kernel

sum_kernel: sum_kernel.cu
	$(NVCC) $< -o $@

sum_kernel_f: sum_kernel.CUF
	$(PGF90) $< -o $@

clean:
	rm -rf sum_kernel sum_kernel_f

