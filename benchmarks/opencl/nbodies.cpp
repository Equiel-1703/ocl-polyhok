#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>

#include "ocl_benchs.hpp"

std::string opencl_kernel_code = R"CLC(
void gpu_nBodies(__global double *p, __global double *c, int n)
{
  float softening = 1.0e-9;
  float dt = 0.01;
  float fx = 0.0;
  float fy = 0.0;
  float fz = 0.0;
  for (int j = 0; j < n; j++)
  {
    double dx = (c[(6 * j)] - p[0]);
    double dy = (c[((6 * j) + 1)] - p[1]);
    double dz = (c[((6 * j) + 2)] - p[2]);
    double distSqr = ((((dx * dx) + (dy * dy)) + (dz * dz)) + softening);
    float invDist = (1.0 / sqrt(distSqr));
    float invDist3 = ((invDist * invDist) * invDist);
    fx = (fx + (dx * invDist3));
    fy = (fy + (dy * invDist3));
    fz = (fz + (dz * invDist3));
  }

  p[3] = (p[3] + (dt * fx));
  p[4] = (p[4] + (dt * fy));
  p[5] = (p[5] + (dt * fz));
}

__kernel void map1(__global double *d_array, int step, __global double *par1, int par2, int size)
{
  int globalId = ((get_local_size(0) * ((get_num_groups(0) * get_group_id(1)) + get_group_id(0))) + get_local_id(0));
  int id = (step * globalId);
  if ((globalId < size))
  {
    gpu_nBodies((d_array + id), par1, par2);
  }
}

void gpu_integrate(__global double *p, float dt, int n)
{
  p[0] = (p[0] + (p[3] * dt));
  p[1] = (p[1] + (p[4] * dt));
  p[2] = (p[2] + (p[5] * dt));
}

__kernel void map2(__global double *d_array, int step, float par1, int par2, int size)
{
  int globalId = ((get_local_size(0) * ((get_num_groups(0) * get_group_id(1)) + get_group_id(0))) + get_local_id(0));
  int id = (step * globalId);
  if ((globalId < size))
  {
    gpu_integrate((d_array + id), par1, par2);
  }
}
)CLC";

void randomizeBodies(double *data, int n)
{
  for (int i = 0; i < n; i++)
  {
    data[i] = (double)(rand() / (float)RAND_MAX);
  }
}

int main(const int argc, const char **argv)
{
  if (argc != 2)
  {
    std::cerr << "Usage: " << argv[0] << " <nbodies>" << std::endl;
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
  cl::Kernel nbodies_kernel_1(program, "map1");
  cl::Kernel nbodies_kernel_2(program, "map2");

  int nBodies = atoi(argv[1]);
  int bytes = nBodies * sizeof(double) * 6;
  
  int block_size = 128;
  int nBlocks = (nBodies + block_size - 1) / block_size;

  // Creating NDRange for kernel launch
  cl::NDRange global_range(nBlocks * block_size);
  cl::NDRange local_range(block_size);

  // Creating OpenCL events to measure time using profiling info
  cl::Event write_ev, kernel_1_ev, kernel_2_ev, read_ev;

  // Creating host arrays
  double *host_buf = (double *)malloc(bytes);
  double *host_resp = (double *)malloc(bytes);

  // Initializing data in host array
  randomizeBodies(host_buf, 6 * nBodies); // Init pos / vel data

  // Device buffer
  cl::Buffer device_buf(context, CL_MEM_READ_WRITE, bytes);

  // Copy data to device buffer
  queue.enqueueWriteBuffer(device_buf, CL_TRUE, 0, bytes, host_buf, nullptr, &write_ev);

  // Launch kernel 1
  nbodies_kernel_1.setArg(0, device_buf);
  nbodies_kernel_1.setArg(1, 6);
  nbodies_kernel_1.setArg(2, device_buf);
  nbodies_kernel_1.setArg(3, nBodies);
  nbodies_kernel_1.setArg(4, nBodies);

  queue.enqueueNDRangeKernel(nbodies_kernel_1, cl::NullRange, global_range, local_range, nullptr, &kernel_1_ev);

  // Launch kernel 2 (compute interbody forces)
  nbodies_kernel_2.setArg(0, device_buf);
  nbodies_kernel_2.setArg(1, 6);
  nbodies_kernel_2.setArg(2, 0.01f);
  nbodies_kernel_2.setArg(3, nBodies);
  nbodies_kernel_2.setArg(4, nBodies);

  queue.enqueueNDRangeKernel(nbodies_kernel_2, cl::NullRange, global_range, local_range, nullptr, &kernel_2_ev);

  // Read back the results
  queue.enqueueReadBuffer(device_buf, CL_TRUE, 0, bytes, host_resp, nullptr, &read_ev);

  // Calculate time using profiling info
  cl_ulong write_start_time = write_ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  cl_ulong write_end_time = write_ev.getProfilingInfo<CL_PROFILING_COMMAND_END>();

  cl_ulong kernel_1_start_time = kernel_1_ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  cl_ulong kernel_1_end_time = kernel_1_ev.getProfilingInfo<CL_PROFILING_COMMAND_END>();

  cl_ulong kernel_2_start_time = kernel_2_ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  cl_ulong kernel_2_end_time = kernel_2_ev.getProfilingInfo<CL_PROFILING_COMMAND_END>();

  cl_ulong read_start_time = read_ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  cl_ulong read_end_time = read_ev.getProfilingInfo<CL_PROFILING_COMMAND_END>();

  // Calculate total time in milliseconds using profiling info
  double write_time = (write_end_time - write_start_time) / 1e6;
  double kernel_1_time = (kernel_1_end_time - kernel_1_start_time) / 1e6;
  double kernel_2_time = (kernel_2_end_time - kernel_2_start_time) / 1e6;
  double read_time = (read_end_time - read_start_time) / 1e6;
  double total_time = write_time + kernel_1_time + kernel_2_time + read_time;

  printf("OpenCL\t%d\t%3.1f\n", nBodies, total_time);

  free(host_buf);
  free(host_resp);
}
