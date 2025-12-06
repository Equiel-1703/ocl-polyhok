#include <iostream>
#include <cstdlib>
#include <cstdio>

#include "ocl_benchs.hpp"
#include "ocl_bmp.hpp"

std::string opencl_kernel_code = R"CLC(
int julia(int x, int y, int dim)
{
    float scale = 0.1;
    float jx = ((scale * (dim - x)) / dim);
    float jy = ((scale * (dim - y)) / dim);
    float cr = (-0.8);
    float ci = 0.156;
    float ar = jx;
    float ai = jy;
    for (int i = 0; i < 200; i++)
    {
        float nar = (((ar * ar) - (ai * ai)) + cr);
        float nai = (((ai * ar) + (ar * ai)) + ci);
        if ((((nar * nar) + (nai * nai)) > 1.0e3))
        {
            return (0);
        }

        ar = nar;
        ai = nai;
    }

    return (1);
}

void julia_function(__global int *ptr, int x, int y, int dim)
{
    int offset = (x + (y * dim));
    int juliaValue = julia(x, y, dim);
    ptr[((offset * 4) + 0)] = (255 * juliaValue);
    ptr[((offset * 4) + 1)] = 0;
    ptr[((offset * 4) + 2)] = 0;
    ptr[((offset * 4) + 3)] = 255;
}

__kernel void mapgen2D_xy_1para_noret_ker(__global int *resp, int arg1, int size)
{
    int x = ((get_group_id(0) * get_local_size(0)) + get_local_id(0));
    int y = ((get_group_id(1) * get_local_size(1)) + get_local_id(1));
    if (((x < size) && (y < size)))
    {
        julia_function(resp, x, y, arg1);
    }
}
)CLC";

int main(int argc, char const *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <image_size>" << std::endl;
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
    cl::Kernel julia_kernel(program, "mapgen2D_xy_1para_noret_ker");

    int usr_value = atoi(argv[1]);

    int height = usr_value;
    int width = usr_value;
    int DIM = usr_value;
    int size_array = height * width * 4 * sizeof(int);

    // Setup OpenCL events for timing
    cl::Event kernel_event, read_event;

    // Host pixel buffer
    int *h_pixelbuffer = (int *)malloc(size_array);
    // Create device pixel buffer
    cl::Buffer d_pixelbuffer(context, CL_MEM_READ_WRITE, size_array);

    // Setup NDRange for kernel launch
    cl::NDRange local_range(1, 1);
    cl::NDRange global_range(DIM, DIM);

    // Setup kernel arguments
    julia_kernel.setArg(0, d_pixelbuffer);
    julia_kernel.setArg(1, DIM);
    julia_kernel.setArg(2, DIM);

    // Launch kernel
    queue.enqueueNDRangeKernel(julia_kernel, cl::NullRange, global_range, local_range, nullptr, &kernel_event);

    // Read back the results
    queue.enqueueReadBuffer(d_pixelbuffer, CL_TRUE, 0, size_array, h_pixelbuffer, nullptr, &read_event);

    // Calculate time using profiling info
    cl_ulong kernel_start_time = kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong kernel_end_time = kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    cl_ulong read_start_time = read_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong read_end_time = read_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    double kernel_execution_time = (kernel_end_time - kernel_start_time) * 1e-6; // Convert to milliseconds
    double read_execution_time = (read_end_time - read_start_time) * 1e-6;       // Convert to milliseconds

    // The total execution time includes kernel execution and read time
    // The buffer creation time is ignored, since CUDA events do not account for it:
    /*
     * cudaMalloc(), however, is primarily a host-side (CPU) operation.
     * While it does interact with the GPU driver to allocate memory on the device,
     * the bulk of its execution time is spent on the CPU, managing memory and setting up the allocation.
     * Since CUDA events are tied to the GPU's execution stream, they cannot reliably capture the time spent
     * by the CPU during cudaMalloc().
     */
    double total_execution_time = kernel_execution_time + read_execution_time;

    printf("OpenCL\t%d\t%3.1f\n", usr_value, total_execution_time);

    // Debug stuff
    /* 
    printf("Platform: %s\n", platform_name.c_str());
    printf("Device: %s\n", device_name.c_str());
    printf("Kernel Execution Time (ms): %3.1f\n", kernel_execution_time);
    printf("Read Execution Time (ms): %3.1f\n", read_execution_time);
    printf("Total Execution Time (ms): %3.1f\n", total_execution_time);
    */

    // Generate BMP file
    // genBpm("julia.bmp", height, width, h_pixelbuffer);

    free(h_pixelbuffer);
}
