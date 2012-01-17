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

#include <stdint.h>
#include <stdio.h>
#include <time.h>

#pragma pack(push, 1)

// The timer value type.
struct time_t
{
	int64_t seconds;
	int64_t nanoseconds;
};

#pragma pack(pop)

// Get the timer resolution.
void get_timer_resolution(struct time_t* val);

// Get the timer value.
void get_time(struct time_t* val);

// Get the timer measured values difference.
double get_time_diff(struct time_t* val1, struct time_t* val2);

// Print the timer measured values difference.
void print_time_diff(struct time_t* val1, struct time_t* val2);

