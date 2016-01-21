#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define WSIZE 32
#define LOOPS 100
#define UPPER_BIT 10
#define LOWER_BIT 0

__device__ unsigned int ddata[WSIZE];
__device__ int ddata_s[WSIZE];

template <typename T, unsigned S>
inline unsigned arraysize(const T(&v)[S])
{
	return S;
}

template<typename T>
void printArray(T &arr)
{
	for (int i = 0; i < arraysize(arr); ++i)
	{
		cout << "Array[" << i << "]: " << *(arr + i) << endl;
	}
}

template<typename T>
void fillArray(T &arr)
{
	srand(time(NULL));
	for (int i = 0; i < arraysize(arr); ++i)
	{
		arr[i] = rand() % 1024;
	}
}

void print(int arr[], int n)
{
	for (int i = 0; i < n; i++)
	{
		cout << arr[i] << " ";
	}
	cout << endl;
}

template<typename T>
T findMax(T &arr)
{
	T max = 0;
	for (int i = 0; i < arraysize(arr); ++i)
	{
		if (arr[i] > max)
		{
			max = arr[i];
		}
	}
	return max;
}

__device__ int getMax(int arr[], int n)
{
	int mx = arr[0];
	for (int i = 1; i < n; i++)
		if (arr[i] > mx)
			mx = arr[i];
	return mx;
}

__device__ void countSort(int arr[], int n, int exp)
{
	int output[1024]; // Output array
	int i, count[10] = { 0 };

	// Store count of occurrences in count[]
	for (i = 0; i < n; i++)
		count[(arr[i] / exp) % 10]++;

	// Change count[i] so that count[i] now contains actual
	// position of this digit in output[]
	for (i = 1; i < 10; i++)
	{
		count[i] += count[i - 1];
	}

	// Build the output array
	for (i = n - 1; i >= 0; i--)
	{
		output[count[(arr[i] / exp) % 10] - 1] = arr[i];
		count[(arr[i] / exp) % 10]--;
	}

	// Copy the output array to arr[], so that arr[] now
	// contains sorted numbers according to current digit
	for (i = 0; i < n; i++)
		arr[i] = output[i];
}

__device__ void radixsort(int arr[], int n)
{
	// Find the maximum number to know number of digits
	int m = getMax(arr, n);

	// Do counting sort for every digit. Note that instead
	// of passing digit number, exp is passed. exp is 10^i
	// where i is current digit number
	for (int exp = 1; m / exp > 0; exp *= 10)
		countSort(arr, n, exp);
}

__global__ void serialRadix()
{
	radixsort(ddata_s, WSIZE);
	__syncthreads();
}

__global__ void parallelRadix()
{
	// This data in shared memory
	__shared__ volatile unsigned int sdata[WSIZE * 2];

	// Load from global into shared variable
	sdata[threadIdx.x] = ddata[threadIdx.x];

	unsigned int bitmask = 1 << LOWER_BIT;
	unsigned int offset = 0;
	// -1, -2, -4, -8, -16, -32, -64, -128, -256,...
	unsigned int thrmask = 0xFFFFFFFFU << threadIdx.x;
	unsigned int mypos;

	// For each LSB to MSB
	for (int i = LOWER_BIT; i <= UPPER_BIT; i++)
	{
		unsigned int mydata = sdata[((WSIZE - 1) - threadIdx.x) + offset];
		unsigned int mybit = mydata&bitmask;
		// Get population of ones and zeroes
		unsigned int ones = __ballot(mybit);
		unsigned int zeroes = ~ones;
		// Switch ping-pong buffers
		offset ^= WSIZE;

		// Do zeroes, then ones
		if (!mybit)
		{
			mypos = __popc(zeroes&thrmask);
		}
		else  {      // Threads with a one bit
			// Get my position in ping-pong buffer
			mypos = __popc(zeroes) + __popc(ones&thrmask);
		}

		// Move to buffer  (or use shfl for cc 3.0)
		sdata[mypos - 1 + offset] = mydata;
		// Repeat for next bit
		bitmask <<= 1;
	}
	// Put results to global
	ddata[threadIdx.x] = sdata[threadIdx.x + offset];
}

int main() {

	/* Parallel Radix Sort */

	unsigned int hdata[WSIZE];
	float totalTime = 0;

	for (int lcount = 0; lcount < LOOPS; lcount++)
	{
		srand(time(NULL));
		// Array elements have value in range of 1024
		unsigned int range = 1U << UPPER_BIT;

		// Fill array with random elements
		// Range = 1024
		for (int i = 0; i < WSIZE; i++)
		{
			hdata[i] = i;
		}

		// Copy data from host to device
		cudaMemcpyToSymbol(ddata, hdata, WSIZE * sizeof(unsigned int));

		// Execution time measurement, that point starts the clock
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		parallelRadix <<< 1, WSIZE >>>();
		// Make kernel function synchronous
		cudaDeviceSynchronize();
		// Execution time measurement, that point stops the clock
		high_resolution_clock::time_point t2 = high_resolution_clock::now();

		// Execution time measurement, that is the result
		auto duration = duration_cast<milliseconds>(t2 - t1).count();

		// Summination of each loops' execution time
		totalTime += (float)duration / 1000.00;

		// Copy data from device to host
		cudaMemcpyFromSymbol(hdata, ddata, WSIZE * sizeof(unsigned int));
	}

	printf("Parallel Radix Sort:\n");
	printf("Array size = %d\n", WSIZE * LOOPS);
	printf("Time elapsed = %fseconds\n", totalTime);

	/* Serial Radix Sort */

	unsigned int hdata_s[WSIZE];
	totalTime = 0;

	for (int lcount = 0; lcount < LOOPS; lcount++)
	{
		srand(time(NULL));
		// Array elements have value in range of 1024
		unsigned int range = 1U << UPPER_BIT;

		// Fill array with random elements
		// Range = 1024
		for (int i = 0; i < WSIZE; i++)
		{
			hdata_s[i] = i;
		}

		// Copy data from host to device
		cudaMemcpyToSymbol(ddata_s, hdata_s, WSIZE * sizeof(unsigned int));

		// Execution time measurement, that point starts the clock
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		serialRadix <<< 1, 1 >>>();
		// Make kernel function synchronous
		cudaDeviceSynchronize();
		// Execution time measurement, that point stops the clock
		high_resolution_clock::time_point t2 = high_resolution_clock::now();

		// Execution time measurement, that is the result
		auto duration = duration_cast<milliseconds>(t2 - t1).count();

		// Summination of each loops' execution time
		totalTime += (float)duration / 1000.00;

		// Copy data from device to host
		cudaMemcpyFromSymbol(hdata_s, ddata_s, WSIZE * sizeof(unsigned int));
	}

	printf("\nSerial Radix Sort:\n");
	printf("Array size = %d\n", WSIZE * LOOPS);
	printf("Time elapsed = %fseconds\n\n", totalTime);

	return 0;
}
