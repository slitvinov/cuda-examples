BLOCK_SIZE = 512

NVCC = nvcc
CFLAGS = -g -O3 -DBLOCK_SIZE=$(BLOCK_SIZE)

all: sine_calc

sine_calc: sine_calc.cu
	$(NVCC) $(CFLAGS) $< -o $@

clean:
	rm -rf sine_calc *.o

