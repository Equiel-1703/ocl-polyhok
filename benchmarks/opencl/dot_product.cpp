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

  // Get platform and device name
  std::string platform_name = platform.getInfo<CL_PLATFORM_NAME>();
  std::string device_name = device.getInfo<CL_DEVICE_NAME>();

  // Create a queue with profiling enabled, this is needed to measure execution time of kernels
  cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

  // Create Program object from kernel source code
  cl::Program program(context, opencl_kernel_code);

  // std::string build_options = "-w -cl-fast-relaxed-math -cl-mad-enable";
  std::string build_options = "-w";

  try
  {
    program.build(build_options.c_str());
  }
  catch (cl::BuildError &e)
  {
    std::cerr << "OpenCL build failed!" << std::endl;
    std::cerr << "Build Log:" << std::endl;
    for (const auto &err : e.getBuildLog())
    {
      std::cerr << err.second << std::endl;
    }
    return 1; // Exit if the kernel can't be built
  }

  cl::Kernel map_2kernel(program, "map_2kernel");
  cl::Kernel reduce_kernel(program, "reduce_kernel");

  float *a, *b, *resp, *final;

  int N = atoi(argv[1]);

  // Creating and populating host arrays
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

  final = (float *)malloc(sizeof(float));
  final[0] = 0;

  // Defining work group sizes
  int threadsPerBlock = 256;
  int numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

  // Creating equivalent NDRange objects
  cl::NDRange global_range(numberOfBlocks * threadsPerBlock);
  cl::NDRange local_range(threadsPerBlock);

  // Creating OpenCL events to measure time using profiling info
  cl::Event write_buffers_start_ev, write_buffers_end_ev, read_buffer_final_ev;

  auto chrono_start = std::chrono::high_resolution_clock::now();

  // Creating device buffers
  cl::Buffer buffer_a(context, CL_MEM_READ_ONLY, N * sizeof(float));
  cl::Buffer buffer_b(context, CL_MEM_READ_ONLY, N * sizeof(float));
  cl::Buffer buffer_resp(context, CL_MEM_READ_WRITE, N * sizeof(float));
  cl::Buffer d_final(context, CL_MEM_READ_WRITE, sizeof(float));

  // Copying data from host to device (H2D), non-blocking calls
  queue.enqueueWriteBuffer(buffer_a, CL_FALSE, 0, N * sizeof(float), a, nullptr, &write_buffers_start_ev);
  queue.enqueueWriteBuffer(buffer_b, CL_FALSE, 0, N * sizeof(float), b);
  queue.enqueueWriteBuffer(d_final, CL_FALSE, 0, sizeof(float), final, nullptr, &write_buffers_end_ev);

  // Set kernel arguments
  map_2kernel.setArg(0, buffer_a);
  map_2kernel.setArg(1, buffer_b);
  map_2kernel.setArg(2, buffer_resp);
  map_2kernel.setArg(3, N);

  reduce_kernel.setArg(0, buffer_resp);
  reduce_kernel.setArg(1, d_final);
  reduce_kernel.setArg(2, 0.0f);
  reduce_kernel.setArg(3, N);

  // Run kernels
  queue.enqueueNDRangeKernel(map_2kernel, cl::NullRange, global_range, local_range);
  queue.enqueueNDRangeKernel(reduce_kernel, cl::NullRange, global_range, local_range);

  // Read back the result
  queue.enqueueReadBuffer(d_final, CL_TRUE, 0, sizeof(float), final, nullptr, &read_buffer_final_ev);

  auto chrono_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> chrono_time = chrono_end - chrono_start;

  // Calculate total time using event profiling info
  cl_ulong write_buffers_start = write_buffers_start_ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  cl_ulong write_buffers_end = write_buffers_end_ev.getProfilingInfo<CL_PROFILING_COMMAND_END>();

  cl_ulong read_buffer_final_start = read_buffer_final_ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  cl_ulong read_buffer_final_end = read_buffer_final_ev.getProfilingInfo<CL_PROFILING_COMMAND_END>();

  double total_time_profiling = (read_buffer_final_end - write_buffers_start) / 1e6;
  double write_buffers_time_profiling = (write_buffers_end - write_buffers_start) / 1e6;
  double kernel_execution_time_profiling = (read_buffer_final_start - write_buffers_end) / 1e6;
  double read_buffer_time_profiling = (read_buffer_final_end - read_buffer_final_start) / 1e6;

  printf("OpenCL\t%d\t%3.1f\n", N, total_time_profiling);
  printf("Result: %f\n", final[0]);
  printf("-------------------------\n");
  printf("Platform: %s\n", platform_name.c_str());
  printf("Device: %s\n", device_name.c_str());
  printf("-------------------------\n");
  printf("Threads per block: %d\n", threadsPerBlock);
  printf("Number of blocks: %d\n", numberOfBlocks);
  printf("-------------------------\n");
  printf("Global range: %lu\n", global_range[0]);
  printf("Local range: %lu\n", local_range[0]);
  printf("-------------------------\n");
  printf("Total time (chrono): %3.5f ms\n", chrono_time.count());
  printf("Total time (profiling): %3.5f ms\n", total_time_profiling);
  printf("Write buffers time (profiling): %3.5f ms\n", write_buffers_time_profiling);
  printf("Kernel execution time (profiling): %3.5f ms\n", kernel_execution_time_profiling);
  printf("Read buffer time (profiling): %3.5f ms\n", read_buffer_time_profiling);

  free(a);
  free(b);
  free(resp);
  free(final);
}
