/*
 * nn.cu
 * Nearest Neighbor
 * Modified by André Du Bois: changed depracated api, creating data set in memory. clean up code not used
 */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <chrono>
#include "cuda.h"

__device__ static double atomic_cas(double *address, double oldv, double newv)
{
    unsigned long long int *address_as_i = (unsigned long long int *)address;
    return __longlong_as_double(atomicCAS(address_as_i, __double_as_longlong(oldv), __double_as_longlong(newv)));
}

__device__ double euclid(double *d_locations, float lat, float lng)
{
    return (sqrt((((lat - d_locations[0]) * (lat - d_locations[0])) + ((lng - d_locations[1]) * (lng - d_locations[1])))));
}

extern "C" __global__ void map_step_2para_1resp_kernel(double *d_array, double *d_result, int step, float par1, float par2, int size)
{
    int globalId = (threadIdx.x + (blockIdx.x * blockDim.x));
    int id = (step * globalId);
    if ((globalId < size))
    {
        d_result[globalId] = euclid((d_array + id), par1, par2);
    }
}

__device__ double menor(double x, double y)
{
    if ((x < y))
    {
        return (x);
    }
    else
    {
        return (y);
    }
}

extern "C" __global__ void reduce_kernel(double *a, double *ref4, int n)
{
    __shared__ double cache[256];
    int tid = (threadIdx.x + (blockIdx.x * blockDim.x));
    int cacheIndex = threadIdx.x;
    double temp = ref4[0];
    while ((tid < n))
    {
        temp = menor(a[tid], temp);
        tid = ((blockDim.x * gridDim.x) + tid);
    }
    cache[cacheIndex] = temp;
    __syncthreads();
    int i = (blockDim.x / 2);
    while ((i != 0))
    {
        if ((cacheIndex < i))
        {
            cache[cacheIndex] = menor(cache[(cacheIndex + i)], cache[cacheIndex]);
        }

        __syncthreads();
        i = (i / 2);
    }
    if ((cacheIndex == 0))
    {
        double current_value = ref4[0];
        while ((!(current_value == atomic_cas(ref4, current_value, menor(cache[0], current_value)))))
        {
            current_value = ref4[0];
        }
    }
}

void loadData(double *locations, int size);

/**
 * This program finds the k-nearest neighbors
 **/

int main(int argc, char *argv[])
{
    double *locations;

    int numRecords = atoi(argv[1]);

    locations = (double *)malloc(sizeof(double) * 2 * numRecords);
    // int numRecords = loadData(filename,records,locations);
    loadData(locations, numRecords);

    double *distances;
    // Pointers to device memory
    double *d_locations;
    double *d_distances;

    /**
     * Allocate memory on host and device
     */

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    distances = (double *)malloc(sizeof(double) * numRecords);
    cudaMalloc((void **)&d_locations, sizeof(double) * 2 * numRecords);
    cudaMalloc((void **)&d_distances, sizeof(double) * numRecords);

    /**
     * Transfer data from host to device
     */
    auto copy_1_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_locations, &locations[0], sizeof(double) * 2 * numRecords, cudaMemcpyHostToDevice);
    auto copy_1_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> copy_1_duration = copy_1_end - copy_1_start; // Calculate duration in ms

    /**
     * Execute kernel --
     */
    auto kernel_1_start = std::chrono::high_resolution_clock::now();
    map_step_2para_1resp_kernel<<<numRecords, 1>>>(d_locations, d_distances, 2, 0.0, 0.0, numRecords);

    cudaDeviceSynchronize();
    auto kernel_1_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> kernel_1_duration = kernel_1_end - kernel_1_start; // Calculate duration in ms

    int threadsPerBlock = 256;
    int blocksPerGrid = (numRecords + threadsPerBlock - 1) / threadsPerBlock;

    double *resp, *d_resp;
    resp = (double *)malloc(sizeof(double));
    resp[0] = 50000;
    cudaMalloc((void **)&d_resp, sizeof(double));

    auto copy_2_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_resp, resp, sizeof(double), cudaMemcpyHostToDevice);
    auto copy_2_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> copy_2_duration = copy_2_end - copy_2_start; // Calculate duration in ms

    auto kernel_2_start = std::chrono::high_resolution_clock::now();
    reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_distances, d_resp, numRecords);
    cudaDeviceSynchronize();
    auto kernel_2_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> kernel_2_duration = kernel_2_end - kernel_2_start; // Calculate duration in ms

    // Copy data from device memory to host memory
    auto copy_3_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(resp, d_resp, sizeof(double), cudaMemcpyDeviceToHost);
    auto copy_3_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> copy_3_duration = copy_3_end - copy_3_start; // Calculate duration in ms

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    // Free memory
    free(locations);
    free(distances);
    free(resp);
    cudaFree(d_locations);
    cudaFree(d_distances);
    cudaFree(d_resp);

    double total_time = copy_1_duration.count() + kernel_1_duration.count() + copy_2_duration.count() + kernel_2_duration.count() + copy_3_duration.count();

    printf("CUDA\t%d\t%3.1f\n", numRecords, time);
    printf("Total time (chrono): %3.1f ms\n", total_time);
    printf("Copy 1 time (H2D) [locations]: %3.1f ms\n", copy_1_duration.count());
    printf("Kernel 1 time: %3.1f ms\n", kernel_1_duration.count());
    printf("Copy 2 time (H2D) [resp]: %3.1f ms\n", copy_2_duration.count());
    printf("Kernel 2 time: %3.1f ms\n", kernel_2_duration.count());
    printf("Copy 3 time (D2H) [d_resp]: %3.1f ms\n", copy_3_duration.count());

    return 0;
}

void loadData(double *locations, int size)
{
    for (int i = 0; i < size; i++)
    {
        locations[0] = ((double)(7 + rand() % 63)) + ((double)rand() / (double)0x7fffffff);

        locations[1] = ((double)(rand() % 358)) + ((double)rand() / (double)0x7fffffff);

        locations = locations + 2;
    }
}
