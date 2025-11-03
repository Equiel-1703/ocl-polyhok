#include "../../c_src/ocl_interface/OCLInterface.hpp"

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <chrono>

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
  OCLInterface ocl(true);
  ocl.selectDefaultPlatform();
  ocl.selectDefaultDevice(CL_DEVICE_TYPE_GPU);

  cl::Program program = ocl.createProgram(opencl_kernel_code);
  cl::Kernel map_2kernel = ocl.createKernel(program, "map_2kernel");
  cl::Kernel reduce_kernel = ocl.createKernel(program, "reduce_kernel");

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

  // Start measuring time
  auto start = std::chrono::high_resolution_clock::now();

  cl::Buffer buffer_a = ocl.createBuffer(N * sizeof(float), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a);
  cl::Buffer buffer_b = ocl.createBuffer(N * sizeof(float), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, b);
  cl::Buffer buffer_resp = ocl.createBuffer(N * sizeof(float), CL_MEM_READ_WRITE);
  cl::Buffer d_final = ocl.createBuffer(sizeof(float), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, final);

  map_2kernel.setArg(0, buffer_a);
  map_2kernel.setArg(1, buffer_b);
  map_2kernel.setArg(2, buffer_resp);
  map_2kernel.setArg(3, N);

  reduce_kernel.setArg(0, buffer_resp);
  reduce_kernel.setArg(1, d_final);
  reduce_kernel.setArg(2, 0.0f);
  reduce_kernel.setArg(3, N);

  // Run kernels
  ocl.executeKernel(map_2kernel, global_range, local_range);
  ocl.executeKernel(reduce_kernel, global_range, local_range);

  ocl.readBuffer(d_final, final, sizeof(float));

  auto end =  std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  double time = elapsed.count();

  printf("OpenCL\t%d\t%3.1f\n", N, time);
  printf("Result: %f\n", final[0]);
  printf("-------------------------\n");
  printf("Threads per block: %d\n", threadsPerBlock);
  printf("Number of blocks: %d\n", numberOfBlocks);
  printf("-------------------------\n");
  printf("Global range: %lu\n", global_range[0]);
  printf("Local range: %lu\n", local_range[0]);

  free(a);
  free(b);
  free(resp);
  free(final);
}
