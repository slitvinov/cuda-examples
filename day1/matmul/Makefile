#
# Makefile to build CUDA examples for Linux
#
# Author: Alex V. Boreskoff, <steps3d@narod.ru>, <steps3d@gmail.com>
#

# The size of shared memory block size
BLOCK_SIZE = 16

NVCC = nvcc 
CFLAGS = --ptxas-options -v --compiler-options -Wall -g -O0 -lcublas -DBLOCK_SIZE=$(BLOCK_SIZE)

all:  matmul1 matmul2

matmul1: matmul1.cu main.h
	$(NVCC) $(CFLAGS) $< -o $@

matmul2: matmul2.cu main.h
	$(NVCC) $(CFLAGS) $< -o $@

clean:
	rm -rf matmul1 matmul2

