#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <chrono>

#include "ocl_benchs.hpp"

std::string opencl_kernel_code = R"CLC(
float anon_ajh07a72e0(__global float *mat1, __global float *mat2, int M, int x, int y)
{
    float sum = 0.0;
    for (int i = 0; i < M; i += 1)
    {
        sum = (sum + (mat1[((x * M) + i)] * mat2[((i * M) + y)]));
    }

    return (sum);
}

__kernel void map2xy2D_kernel(__global float *arr1, __global float *arr2, int par, __global float *resp, int size)
{
    int row = ((get_group_id(1) * get_local_size(1)) + get_local_id(1));
    int col = ((get_group_id(0) * get_local_size(0)) + get_local_id(0));
    if (((col < size) && (row < size)))
    {
        resp[((row * size) + col)] = anon_ajh07a72e0(arr1, arr2, par, row, col);
    }
}
)CLC";

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
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

    // Create the kernel
    cl::Kernel mm_kernel(program, "map2xy2D_kernel");

    int M = atoi(argv[1]);

    // Creating host arrays (only a and b)
    float *a = (float *)malloc(M * M * sizeof(float));
    float *b = (float *)malloc(M * M * sizeof(float));

    // Populating host arrays with random values
    srand(time(0));

    for (int i = 0; i < M * M; ++i)
    {
        a[i] = rand() % 1000;
    }

    for (int i = 0; i < M * M; ++i)
    {
        b[i] = rand() % 1000;
    }

    // Setup NDRange for kernel launch
    int block_size = 16;
    int grid_rows = (M + block_size - 1) / block_size;
    int grid_cols = (M + block_size - 1) / block_size;

    cl::NDRange global_range(grid_cols * block_size, grid_rows * block_size);
    cl::NDRange local_range(block_size, block_size);

    // Creating OpenCL events to measure time using profiling info
    cl::Event write_buffers_start_ev, write_buffers_end_ev;
    cl::Event kernel_execution_ev;
    cl::Event read_buffer_final_ev;

    auto chrono_start = std::chrono::high_resolution_clock::now();

    // Creating device buffers
    cl::Buffer buffer_a(context, CL_MEM_READ_WRITE, M * M * sizeof(float));
    cl::Buffer buffer_b(context, CL_MEM_READ_WRITE, M * M * sizeof(float));
    cl::Buffer buffer_c(context, CL_MEM_READ_WRITE, M * M * sizeof(float));

    // Copying data from host to device (H2D), non-blocking calls
    queue.enqueueWriteBuffer(buffer_a, CL_FALSE, 0, M * M * sizeof(float), a, nullptr, &write_buffers_start_ev);
    queue.enqueueWriteBuffer(buffer_b, CL_FALSE, 0, M * M * sizeof(float), b, nullptr, &write_buffers_end_ev);

    // Set kernel arguments
    mm_kernel.setArg(0, buffer_a);
    mm_kernel.setArg(1, buffer_b);
    mm_kernel.setArg(2, M);
    mm_kernel.setArg(3, buffer_c);
    mm_kernel.setArg(4, M);

    // Launching the kernel
    queue.enqueueNDRangeKernel(mm_kernel, cl::NullRange, global_range, local_range, nullptr, &kernel_execution_ev);

    // Copying result from device to host (D2H)
    // Here we are reusing one of the CPU buffers holding the input matrices, since we no longer need them
    queue.enqueueReadBuffer(buffer_c, CL_TRUE, 0, M * M * sizeof(float), a, nullptr, &read_buffer_final_ev);

    auto chrono_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_chrono_time = chrono_end - chrono_start;

    // Calculate time using profiling info
    cl_ulong write_start = write_buffers_start_ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong write_end = write_buffers_end_ev.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    cl_ulong kernel_start = kernel_execution_ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong kernel_end = kernel_execution_ev.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    cl_ulong read_start = read_buffer_final_ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong read_end = read_buffer_final_ev.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    double write_time = (write_end - write_start) / 1e6;
    double kernel_time = (kernel_end - kernel_start) / 1e6;
    double read_time = (read_end - read_start) / 1e6;
    double total_time_profiling = write_time + kernel_time + read_time;

    printf("OpenCL\t%d\t%3.1f\n", M, total_time_profiling);
    printf("-------------------------\n");
    printf("Platform: %s\n", platform_name.c_str());
    printf("Device: %s\n", device_name.c_str());
    printf("-------------------------\n");
    printf("Threads per block: %d x %d\n", block_size, block_size);
    printf("Blocks per grid: %d x %d\n", grid_cols, grid_rows);
    printf("-------------------------\n");
    printf("Global range: %lu x %lu\n", global_range[0], global_range[1]);
    printf("Local range: %lu x %lu\n", local_range[0], local_range[1]);
    printf("-------------------------\n");
    printf("Total time (chrono): %3.5f ms\n", total_chrono_time.count());
    printf("Total time (profiling): %3.5f ms\n", total_time_profiling);
    printf("Write buffers time (profiling): %3.5f ms\n", write_time);
    printf("Kernel execution time (profiling): %3.5f ms\n", kernel_time);
    printf("Read buffer time (profiling): %3.5f ms\n", read_time);

    free(a);
    free(b);
}
