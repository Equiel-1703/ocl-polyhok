#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>

#include "ocl_benchs.hpp"

std::string opencl_kernel_code = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

inline double cas_double(volatile __global double *address, double oldv, double newv)
{
  volatile __global ulong *l_address = (volatile __global ulong *)address;
  ulong l_oldv = as_ulong(oldv);
  ulong l_newv = as_ulong(newv);

  // For 64-bit atomic compare-and-swap, the function is atom_cmpxchg (weird, but this is the specification)
  ulong l_res = atom_cmpxchg(l_address, l_oldv, l_newv);

  // Return the double representation of the result
  return as_double(l_res);
}

double euclid(__global double *d_locations, float lat, float lng)
{
  return (sqrt((((lat - d_locations[0]) * (lat - d_locations[0])) + ((lng - d_locations[1]) * (lng - d_locations[1])))));
}

__kernel void map_step_2para_1resp_kernel(__global double *d_array, __global double *d_result, int step, float par1, float par2, int size)
{
  int globalId = (get_local_id(0) + (get_group_id(0) * get_local_size(0)));
  int id = (step * globalId);
  if ((globalId < size))
  {
    d_result[globalId] = euclid((d_array + id), par1, par2);
  }
}

double menor(double x, double y)
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

__kernel void reduce_kernel(__global double *a, __global double *ref4, int n)
{
  __local double cache[256];
  int tid = (get_local_id(0) + (get_group_id(0) * get_local_size(0)));
  int cacheIndex = get_local_id(0);
  double temp = ref4[0];
  while ((tid < n))
  {
    temp = menor(a[tid], temp);
    tid = ((get_local_size(0) * get_num_groups(0)) + tid);
  }
  cache[cacheIndex] = temp;
  barrier(CLK_LOCAL_MEM_FENCE);
  int i = (get_local_size(0) / 2);
  while ((i != 0))
  {
    if ((cacheIndex < i))
    {
      cache[cacheIndex] = menor(cache[(cacheIndex + i)], cache[cacheIndex]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    i = (i / 2);
  }
  if ((cacheIndex == 0))
  {
    double current_value = ref4[0];
    while ((!(current_value == cas_double(ref4, current_value, menor(cache[0], current_value)))))
    {
      current_value = ref4[0];
    }
  }
}
)CLC";

void loadData(double *locations, int size);

/**
 * This program finds the k-nearest neighbors
 **/
int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    std::cerr << "Usage: " << argv[0] << " <num_records>" << std::endl;
    return 1;
  }

  cl::Platform platform = getDefaultPlatform();
  cl::Device device = getDefaultDevice(platform);
  cl::Context context(device);

  // Get platform and device name
  std::string platform_name = platform.getInfo<CL_PLATFORM_NAME>();
  std::string device_name = device.getInfo<CL_DEVICE_NAME>();

  // Create a command queue with profiling enabled so we can measure execution time of kernels
  cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

  // Create Program object from kernel source code
  cl::Program program(context, opencl_kernel_code);

  // std::string build_options = "-w -cl-fast-relaxed-math -cl-mad-enable";
  std::string build_options = "-w"; // Ignore warnings

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

  // Create the kernels
  cl::Kernel nn_kernel_1(program, "map_step_2para_1resp_kernel");
  cl::Kernel nn_kernel_2(program, "reduce_kernel");

  int numRecords = atoi(argv[1]);

  // Creating events to measure time using profiling info
  cl::Event write_1_ev, write_2_ev;
  cl::Event kernel_1_ev, kernel_2_ev;
  cl::Event read_ev;

  // Allocate host memory
  double *host_locations = (double *)malloc(sizeof(double) * 2 * numRecords);
  double *host_distances = (double *)malloc(sizeof(double) * numRecords);
  double *host_result = (double *)malloc(sizeof(double));
  host_result[0] = DBL_MAX;

  // Initialize host data
  loadData(host_locations, numRecords);

  // Create device buffers
  cl::Buffer buffer_locations(context, CL_MEM_READ_WRITE, sizeof(double) * 2 * numRecords);
  cl::Buffer buffer_distances(context, CL_MEM_READ_WRITE, sizeof(double) * numRecords);
  cl::Buffer buffer_result(context, CL_MEM_READ_WRITE, sizeof(double));

  // Copy data from host to device (H2D copy), non-blocking calls
  queue.enqueueWriteBuffer(buffer_locations, CL_FALSE, 0, sizeof(double) * 2 * numRecords, host_locations, nullptr, &write_1_ev);
  queue.enqueueWriteBuffer(buffer_result, CL_FALSE, 0, sizeof(double), host_result, nullptr, &write_2_ev);

  // ---- Execute kernel 1 ----

  // Preparing NDRange
  cl::NDRange global_work_size_k1(numRecords);
  cl::NDRange local_work_size_k1(1);

  // Set kernel arguments
  nn_kernel_1.setArg(0, buffer_locations);
  nn_kernel_1.setArg(1, buffer_distances);
  nn_kernel_1.setArg(2, 2);          // step
  nn_kernel_1.setArg(3, 0.0f);       // par1
  nn_kernel_1.setArg(4, 0.0f);       // par2
  nn_kernel_1.setArg(5, numRecords); // size

  // Launch kernel 1
  queue.enqueueNDRangeKernel(nn_kernel_1, cl::NullRange, global_work_size_k1, local_work_size_k1, nullptr, &kernel_1_ev);

  // ---- Execute kernel 2 ----

  // Preparing NDRange
  int threadsPerBlock = 256;
  int blocksPerGrid = (numRecords + threadsPerBlock - 1) / threadsPerBlock;

  cl::NDRange global_work_size_k2(blocksPerGrid * threadsPerBlock);
  cl::NDRange local_work_size_k2(threadsPerBlock);

  // Set kernel arguments
  nn_kernel_2.setArg(0, buffer_distances);
  nn_kernel_2.setArg(1, buffer_result);
  nn_kernel_2.setArg(2, numRecords);

  // Launch kernel 2
  queue.enqueueNDRangeKernel(nn_kernel_2, cl::NullRange, global_work_size_k2, local_work_size_k2, nullptr, &kernel_2_ev);

  // Copy data from device memory to host memory (D2H copy), blocking call
  queue.enqueueReadBuffer(buffer_result, CL_TRUE, 0, sizeof(double), host_result, nullptr, &read_ev);

  // Free host memory
  free(host_locations);
  free(host_distances);
  free(host_result);

  // Calculate kernel execution time using profiling info
  cl_ulong write_1_start = write_1_ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  cl_ulong write_1_end = write_1_ev.getProfilingInfo<CL_PROFILING_COMMAND_END>();

  cl_ulong write_2_start = write_2_ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  cl_ulong write_2_end = write_2_ev.getProfilingInfo<CL_PROFILING_COMMAND_END>();

  cl_ulong kernel_1_start = kernel_1_ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  cl_ulong kernel_1_end = kernel_1_ev.getProfilingInfo<CL_PROFILING_COMMAND_END>();

  cl_ulong kernel_2_start = kernel_2_ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  cl_ulong kernel_2_end = kernel_2_ev.getProfilingInfo<CL_PROFILING_COMMAND_END>();

  cl_ulong read_start = read_ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  cl_ulong read_end = read_ev.getProfilingInfo<CL_PROFILING_COMMAND_END>();

  double write_time = ((write_1_end - write_1_start) + (write_2_end - write_2_start)) / 1e6;
  double kernel_time = ((kernel_1_end - kernel_1_start) + (kernel_2_end - kernel_2_start)) / 1e6;
  double read_time = (read_end - read_start) / 1e6;
  double total_time = write_time + kernel_time + read_time;

  printf("OpenCL\t%d\t%3.1f\n", numRecords, total_time);
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
