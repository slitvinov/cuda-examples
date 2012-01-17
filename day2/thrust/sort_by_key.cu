#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <cstdio>
#include <cstdlib>
#include <thrust/random.h>

void usage(const char* filename)
{
	printf("Sort the random key-value data set of the given length by key.\n");
	printf("Usage: %s <n>\n", filename);
}

thrust::host_vector<int> random_vector(size_t N)
{
    thrust::host_vector<int> vec(N);
    static thrust::default_random_engine rng;
    static thrust::uniform_int_distribution<int> dist(0, 9);

    for (size_t i = 0; i < N; i++)
        vec[i] = dist(rng);

    return vec;
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
	thrust::host_vector<int> h_keys = random_vector(n);
	thrust::host_vector<int> h_vals = random_vector(n);

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
	device_vector<int> keys = h_keys;
	device_vector<int> values = h_vals;

	// TODO: Use sort_by_key or stable_sort_by_key to sort
	// pairs by key.
	// sort_by_key( ...
	thrust::sort_by_key(keys.begin(), keys.end(), 
			    values.begin(), thrust::greater<int>());


	// TODO: Transfer data back to host.
	thrust::copy(keys.begin(), keys.end(), h_keys.begin());
	thrust::copy(values.begin(), values.end(), h_vals.begin());
	

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

