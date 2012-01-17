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

#include "timing.h"

#include <stdio.h>
#include <time.h>

#define CLOCKID CLOCK_REALTIME

// Get the timer resolution.
void get_timer_resolution(struct time_t* val)
{
	if ((sizeof(int64_t) == sizeof(time_t)) &&
		(sizeof(int64_t) == sizeof(long)))
		clock_getres(CLOCKID, (struct timespec *)val);
	else
	{
		struct timespec t;
		clock_getres(CLOCKID, &t);
		val->seconds = t.tv_sec;
		val->nanoseconds = t.tv_nsec;
	}
}

// Get the timer value.
void get_time(struct time_t* val)
{
	if ((sizeof(int64_t) == sizeof(time_t)) &&
		(sizeof(int64_t) == sizeof(long)))
		clock_gettime(CLOCKID, (struct timespec *)val);
	else
	{
		struct timespec t;
		clock_gettime(CLOCKID, &t);
		val->seconds = t.tv_sec;
		val->nanoseconds = t.tv_nsec;
	}
}

// Get the timer measured values difference.
double get_time_diff(
	struct time_t* val1, struct time_t* val2)
{
	int64_t seconds = val2->seconds - val1->seconds;
	int64_t nanoseconds = val2->nanoseconds - val1->nanoseconds;
	
	if (val2->nanoseconds < val1->nanoseconds)
	{
		seconds--;
		nanoseconds = (1000000000 - val1->nanoseconds) + val2->nanoseconds;
	}
	
	return (double)0.000000001 * nanoseconds + seconds;
}

// Print the timer measured values difference.
void print_time_diff(
	struct time_t* val1, struct time_t* val2)
{
	int64_t seconds = val2->seconds - val1->seconds;
	int64_t nanoseconds = val2->nanoseconds - val1->nanoseconds;
	
	if (val2->nanoseconds < val1->nanoseconds)
	{
		seconds--;
		nanoseconds = (1000000000 - val1->nanoseconds) + val2->nanoseconds;
	}
	if (sizeof(uint64_t) == sizeof(long))
		printf("%ld.%09ld", (long)seconds, (long)nanoseconds);
	else
		printf("%lld.%09lld", seconds, nanoseconds);

}

