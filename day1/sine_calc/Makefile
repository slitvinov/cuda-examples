BLOCK_SIZE = 512

NVCC = nvcc
CFLAGS = --ptxas-options -v --compiler-options -Wall -g -O0 -DBLOCK_SIZE=$(BLOCK_SIZE)

all: sine_calc

sine_calc: sine_calc.cu
	$(NVCC) $(CFLAGS) $< -o $@

clean:
	rm -rf sine_calc *.o

