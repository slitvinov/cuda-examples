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
 */

#ifndef PATTERN2D_H
#define PATTERN2D_H

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

// Perform some dummy 2D field processing on CPU.
int pattern2d_cpu(
	int bx, int nx, int ex, int by, int ny, int ey,
	float* in, float* out, int id);

// Perform some dummy 2D field processing on GPU.
int pattern2d_gpu(
	int bx, int nx, int ex, int by, int ny, int ey,
	float* in, float* out, int id);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // PATTERN2D_H

