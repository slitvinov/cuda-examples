all:
	gcc -g -std=c99 cholessky.c -o cholessky -L../magma/lib -lmagma -lmagmablas -L../lapack -llapack -lblas -lm -lgfortran -L/opt/cuda/lib64 -L/usr/local/cuda/lib64 -lcublas -lstdc++

clean:
	rm -rf cholessky

