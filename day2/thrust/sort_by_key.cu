#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdio>
#include <cstdlib>

void usage(const char* filename)
{
	printf("Sort the random key-value data set of the given length by key.\n");
	printf("Usage: %s <n>\n", filename);
}

using namespace thrust;

// TODO: Please refer to sorting examples:
// http://code.google.com/p/thrust/
// http://code.google.com/p/thrust/wiki/QuickStartGuide#Sorting

int main(int argc, char* argv[])
{
	const int printable_n = 128;

	if (argc != 2)
	{
		usage(argv[0]);
		return 0;
	}

	int n = atoi(argv[1]);
	if (n <= 0)
	{
		usage(argv[0]);
		return 0;
	}

	// TODO: Generate random keys and values on host
	// host_vector<int> ...
	// generate( ...
	thrust::host_vector<int> h_keys(n);
	thrust::host_vector<int> h_keys(n);

	thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
	for (size_t i=0; i<n; i++) {
	  
	}

	// Print out the input data if n is small.
	if (n <= printable_n)
	{
		printf("Input data:\n");
		for (int i = 0; i < n; i++)
			printf("(%d, %d)\n", h_keys[i], h_vals[i]);
		printf("\n");
	}

	// TODO: Transfer data to the device.
	// device_vector<int> ...

	// TODO: Use sort_by_key or stable_sort_by_key to sort
	// pairs by key.
	// sort_by_key( ...

	// TODO: Transfer data back to host.
	// copy( ...

	// Print out the output data if n is small.
	if (n <= printable_n)
	{
		printf("Output data:\n");
		for (int i = 0; i < n; i++)
			printf("(%d, %d)\n", h_keys[i], h_vals[i]);
		printf("\n");
	}

	return 0;
}

