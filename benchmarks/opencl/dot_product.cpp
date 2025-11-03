#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <chrono>

#include "ocl_benchs.hpp"

std::string opencl_kernel_code = R"CLC(
inline float cas_float(volatile __global float *address, float oldv, float newv)
{
  volatile __global uint *i_address = (volatile __global uint *)address;
  uint i_oldv = as_uint(oldv);
  uint i_newv = as_uint(newv);

  uint i_res = atomic_cmpxchg(i_address, i_oldv, i_newv);

  // Return the float representation of the result
  return as_float(i_res);
}

float anon_jn6kj70i4c(float a, float b)
{
  return ((a * b));
}

__kernel void map_2kernel(__global float *a1, __global float *a2, __global float *a3, int size)
{
  int id = ((get_group_id(0) * get_local_size(0)) + get_local_id(0));
  if ((id < size))
  {
    a3[id] = anon_jn6kj70i4c(a1[id], a2[id]);
  }
}

float anon_lilmlk478i(float a, float b)
{
  return ((a + b));
}

__kernel void reduce_kernel(__global float *a, __global float *ref4, float initial, int n)
{
  __local float cache[256];
  int tid = (get_local_id(0) + (get_group_id(0) * get_local_size(0)));
  int cacheIndex = get_local_id(0);
  float temp = initial;
  while ((tid < n))
  {
    temp = anon_lilmlk478i(a[tid], temp);
    tid = ((get_local_size(0) * get_num_groups(0)) + tid);
  }
  cache[cacheIndex] = temp;
  barrier(CLK_LOCAL_MEM_FENCE);
  int i = (get_local_size(0) / 2);
  while ((i != 0))
  {
    if ((cacheIndex < i))
    {
      cache[cacheIndex] = anon_lilmlk478i(cache[(cacheIndex + i)], cache[cacheIndex]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    i = (i / 2);
  }
  if ((cacheIndex == 0))
  {
    float current_value = ref4[0];
    while ((!(current_value == cas_float(ref4, current_value, anon_lilmlk478i(cache[0], current_value)))))
    {
      current_value = ref4[0];
    }
  }
}
)CLC";

int main(int argc, char *argv[])
{
  cl::Platform platform = getDefaultPlatform();
  cl::Device device = getDefaultDevice(platform);
  cl::Context context(device);

  // Create a queue with profiling enabled, this is needed to measure execution time of kernels
  cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

  cl::Program program(context, opencl_kernel_code, true);
  cl::Kernel map_2kernel(program, "map_2kernel");
  cl::Kernel reduce_kernel(program, "reduce_kernel");

  float *a, *b, *resp;

  int N = atoi(argv[1]);

  a = (float *)malloc(N * sizeof(float));
  b = (float *)malloc(N * sizeof(float));
  resp = (float *)malloc(N * sizeof(float));

  int tot = N / 2;
  for (int i = 0; i < tot; i++)
  {
    int v = rand() % 100 + 1;

    if (i % 2 == 0)
    {
      a[i] = (float)v;
      a[tot + i] = (float)-v;
    }
    else
    {
      a[i] = (float)-v;
      a[tot + i] = (float)v;
    }
    int n = rand() % 5 + 1;

    b[i] = (float)n;
    b[tot + i] = (float)n;
  }

  float *final = (float *)malloc(sizeof(float));
  final[0] = 0;

  int threadsPerBlock = 256;
  int numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

  // Creating equivalent NDRange objects
  cl::NDRange global_range(numberOfBlocks * threadsPerBlock);
  cl::NDRange local_range(threadsPerBlock);

  auto start = std::chrono::high_resolution_clock::now();

  cl::Buffer buffer_a(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(float), a);
  cl::Buffer buffer_b(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(float), b);
  cl::Buffer buffer_resp(context, CL_MEM_READ_WRITE, N * sizeof(float));
  cl::Buffer d_final(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float), final);

  map_2kernel.setArg(0, buffer_a);
  map_2kernel.setArg(1, buffer_b);
  map_2kernel.setArg(2, buffer_resp);
  map_2kernel.setArg(3, N);

  reduce_kernel.setArg(0, buffer_resp);
  reduce_kernel.setArg(1, d_final);
  reduce_kernel.setArg(2, 0.0f);
  reduce_kernel.setArg(3, N);

  // Measure execution time of kernels with event profiling
  cl::Event map_event, reduce_event, read_event;

  // Run kernels
  queue.enqueueNDRangeKernel(map_2kernel, cl::NullRange, global_range, local_range, nullptr, &map_event);
  queue.enqueueNDRangeKernel(reduce_kernel, cl::NullRange, global_range, local_range, nullptr, &reduce_event);

  // Read back the result
  queue.enqueueReadBuffer(d_final, CL_TRUE, 0, sizeof(float), final, nullptr, &read_event);

  // Wait for all operations to finish
  queue.finish();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;

  // Calculate total time using event profiling info
  cl_ulong map_start = map_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  cl_ulong map_end = map_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  cl_ulong reduce_start = reduce_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  cl_ulong reduce_end = reduce_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  cl_ulong read_start = read_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  cl_ulong read_end = read_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

  double time = ((map_end - map_start) + (reduce_end - reduce_start) + (read_end - read_start)) / 1e6; // Convert to milliseconds

  printf("OpenCL\t%d\t%3.1f\n", N, time);
  printf("Result: %f\n", final[0]);
  printf("-------------------------\n");
  printf("Threads per block: %d\n", threadsPerBlock);
  printf("Number of blocks: %d\n", numberOfBlocks);
  printf("-------------------------\n");
  printf("Global range: %lu\n", global_range[0]);
  printf("Local range: %lu\n", local_range[0]);
  printf("-------------------------\n");
  printf("Elapsed time (chrono): %3.5f ms\n", elapsed.count());
  printf("Elapsed time (profiling): %3.5f ms\n", time);

  free(a);
  free(b);
  free(resp);
  free(final);
}
