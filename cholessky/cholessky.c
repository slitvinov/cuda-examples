/*
 * MSU CUDA Course Examples and Exercises.
 *
 * Copyright (c) 2011 Dmitry Mikushin
 *
 * This software is provided 'as-is', without any express or implied warranty.
 * In no event will the authors be held liable for any damages arising 
 * from the use of this software.
 * Permission is granted to anyone to use this software for any purpose, 
 * including commercial applications, and to alter it and redistribute it freely,
 * without any restrictons.
 *
 * This sample uses UTK ICL MAGMA that comes with the following notice:
 *
 * Copyright (c) 2011 The University of Tennessee. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer
 *   listed in this license in the documentation and/or other materials
 *   provided with the distribution.
 * - Neither the name of the copyright holders nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * This software is provided by the copyright holders and contributors
 * "as is" and any express or implied warranties, including, but not
 * limited to, the implied warranties of merchantability and fitness
 * for a particular purpose are disclaimed. In no event shall the
 * copyright owner or contributors be liable for any direct, indirect,
 * incidental, special, exemplary, or consequential damages (including,
 * but not limited to, procurement of substitute goods or services;
 * loss of use, data, or profits; or business interruption) however
 * caused and on any theory of liability, whether in contract, strict
 * liability, or tort (including negligence or otherwise) arising
 * in any way out of the use of this software, even if advised of
 * the possibility of such damage.
 */

#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Reference for function implementing
// Cholessky decomposition on GPU (C interface).
int magma_spotrf(char uplo, int n, float* a, int lda, int* info);

// Reference for function implementing
// Cholessky decomposition on CPU (Fortran inteace).
void spotrf_(char* uplo, int* n, float* a, int* lda, int* info);

// Random matrix generator.
void slarnv_(int* dist, int* seed, int* n, float* A);

int usage(char* name)
{
	printf("Example of symmetric matrix Cholessky decomposition\n");
	printf("%s <n>\n", name);
	return 0;
}

// Interpret spotrf return codes.
void chkerr(int info)
{
	if (!info)
	{
		printf("OK\n");
		return;
	}
	
	if (info > 0)
		printf("%s %d %s %s\n",
			"The leading minor of order", info,
			"is not positive definite, and the factorization",
			"could not be completed");
	else
		printf("The %d-th argument had an illegal value\n", -info);
}

int main(int argc, char* argv[])
{
	if (argc != 2) return usage(argv[0]);
	
	int n = atoi(argv[1]), n2 = n * n;
	
	if (n <= 0) return usage(argv[0]);
	
	// Generate random matrix.
	size_t size = sizeof(float) * n2;
	float* A1 = (float*)malloc(size);
	int one = 1, seed[4] = { 0, 0, 0, 1 };
	slarnv_(&one, seed, &n2, A1);

	// Symmetrize and increase the diagonal.
	for (int i = 0; i < n; i++)
	{
		A1[i * n + i] += n;
		for (int j = 0; j < i; j++)
			A1[i * n + j] = A1[j * n + i];
	}

	// Clone generated matrix for GPU version
	// (we can't use one copy of A, because
	// spotrf rewrites the input matrix).
	float* A2 = (float*)malloc(size);
	memcpy(A2, A1, size);
	
	// Use upper part of input matrix and
	// rewrite it with Cholessky factor.
	char uplo = 'U';
	
	// The status info (routine must return 0 into info).
	int info = 0;
	
	// Perform decomposition on CPU.
	printf("Computing on CPU ... "); fflush(stdout);
	spotrf_(&uplo, &n, A1, &n, &info);
	chkerr(info);

	// Perform decomposition on GPU.
	printf("Computing on GPU ... "); fflush(stdout);
	magma_spotrf(uplo, n, A2, n, &info);
	chkerr(info);
	
	// Compare results.
	float maxdiff = fabs(A1[0] - A2[0]);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < i; j++)
		{
			maxdiff = fmax(maxdiff,
				fabs(A1[i * n + j] - A2[i * n + j]));
			maxdiff = fmax(maxdiff,
				fabs(A1[j * n + i] - A2[j * n + i]));
		}

	printf("Done! max diff = %f\n", maxdiff);
	
	free(A1); free(A2);
}

