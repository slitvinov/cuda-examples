BLOCK_SIZE = 512

NVCC = nvcc
CFLAGS = --ptxas-options -v --compiler-options -Wall -g -O3 -DBLOCK_SIZE=$(BLOCK_SIZE)

bin=sine_calc_comp10_f sine_calc_comp10_d \
	sine_calc_comp13_f sine_calc_comp13_d

all: ${bin}

sine_calc_comp10_f: sine_calc.cu
	$(NVCC) $(CFLAGS) $< -o $@

sine_calc_comp10_d: sine_calc.cu
	$(NVCC) $(CFLAGS) -DUSE_DOUBLE $< -o $@

sine_calc_comp13_f: sine_calc.cu
	$(NVCC) $(CFLAGS) -arch=compute_13 $< -o $@

sine_calc_comp13_d: sine_calc.cu
	$(NVCC) $(CFLAGS) -arch=compute_13  -DUSE_DOUBLE $< -o $@

clean:
	rm -rf ${bin} sine_calc_d *.o

