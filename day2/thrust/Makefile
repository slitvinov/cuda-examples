# The size of shared memory block size
BLOCK_SIZE = 16

NVCC = nvcc 
CFLAGS = --ptxas-options -v --compiler-options -Wall -g -O3 -lcublas -DBLOCK_SIZE=$(BLOCK_SIZE)

all:  sort_by_key

sort_by_key: sort_by_key.cu
	$(NVCC) $(CFLAGS) $< -o $@

clean:
	rm -rf sort_by_key

