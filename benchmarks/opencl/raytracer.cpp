#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstdint>

#include "ocl_benchs.hpp"
#include "ocl_bmp.hpp"

#define rnd(x) (x * rand() / RAND_MAX)
#define INF 2e10f

#define SPHERES 20

struct Sphere
{
    float r, b, g;
    float radius;
    float x, y, z;
};

void loadSpheres(Sphere *vet, int size, float dim, float radius, float sum)
{
    for (int i = 0; i < size; i++)
    {
        Sphere sphere;
        sphere.r = rnd(1.0);
        sphere.b = rnd(1.0);
        sphere.g = rnd(1.0);
        sphere.radius = rnd(radius) + sum;
        sphere.x = rnd(dim) - trunc(dim / 2);
        sphere.y = rnd(dim) - trunc(dim / 2);
        sphere.z = rnd(256) - 128;

        vet[i] = sphere;
    }
}

std::string opencl_kernel_code = R"CLC(
void raytracing(__global int *image, int width, __global float *spheres, int x, int y)
{
    float ox = 0.0;
    float oy = 0.0;
    ox = (x - (width / 2));
    oy = (y - (width / 2));
    float r = 0.0;
    float g = 0.0;
    float b = 0.0;
    float maxz = (-99999.0);
    for (int i = 0; i < 20; i++)
    {
        float sphereRadius = spheres[((i * 7) + 3)];
        float dx = (ox - spheres[((i * 7) + 4)]);
        float dy = (oy - spheres[((i * 7) + 5)]);
        float n = 0.0;
        float t = (-99999.0);
        float dz = 0.0;
        if ((((dx * dx) + (dy * dy)) < (sphereRadius * sphereRadius)))
        {
            dz = sqrt((((sphereRadius * sphereRadius) - (dx * dx)) - (dy * dy)));
            n = (dz / sqrt((sphereRadius * sphereRadius)));
            t = (dz + spheres[((i * 7) + 6)]);
        }
        else
        {
            t = (-99999.0);
            n = 0.0;
        }

        if ((t > maxz))
        {
            float fscale = n;
            r = (spheres[((i * 7) + 0)] * fscale);
            g = (spheres[((i * 7) + 1)] * fscale);
            b = (spheres[((i * 7) + 2)] * fscale);
            maxz = t;
        }
    }

    image[0] = (r * 255);
    image[1] = (g * 255);
    image[2] = (b * 255);
    image[3] = 255;
}

__kernel void mapxy_2D_step_2_para_no_resp_kernel(__global int *d_array, int step, int par1, __global float *par2, int size)
{
    int x = (get_local_id(0) + (get_group_id(0) * get_local_size(0)));
    int y = (get_local_id(1) + (get_group_id(1) * get_local_size(1)));
    int offset = (x + ((y * get_local_size(0)) * get_num_groups(0)));
    int id = (step * offset);
    if ((offset < (size * size)))
    {
        raytracing((d_array + id), par1, par2, x, y);
    }
}
)CLC";

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <img_dimension>" << std::endl;
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
    cl::Kernel raytracer_kernel(program, "mapxy_2D_step_2_para_no_resp_kernel");

    int dim = atoi(argv[1]);

    // Create OpenCL events to measure time using profiling info
    cl::Event write_ev;
    cl::Event kernel_ev;
    cl::Event read_ev;

    // Allocate host memory
    int *final_image = (int *)malloc(dim * dim * sizeof(int) * 4);
    Sphere *temp_s = (Sphere *)malloc(sizeof(Sphere) * SPHERES);

    // Initialize spheres in host
    loadSpheres(temp_s, SPHERES, dim, 160, 20);

    // Create device buffers
    cl::Buffer buffer_image(context, CL_MEM_READ_WRITE, sizeof(int) * dim * dim * 4);
    cl::Buffer buffer_spheres(context, CL_MEM_READ_WRITE, sizeof(Sphere) * SPHERES);

    // Copy data from host to device (H2D copy), non-blocking calls
    queue.enqueueWriteBuffer(buffer_spheres, CL_FALSE, 0, sizeof(Sphere) * SPHERES, temp_s, nullptr, &write_ev);

    // Preparing NDRange
    cl::NDRange global_work_size(dim, dim);
    cl::NDRange local_work_size(16, 16);

    // Set kernel arguments
    raytracer_kernel.setArg(0, buffer_image);
    raytracer_kernel.setArg(1, 4);              // step
    raytracer_kernel.setArg(2, 0);              // par1
    raytracer_kernel.setArg(3, buffer_spheres); // par2
    raytracer_kernel.setArg(4, dim);            // size

    // Launch kernel
    queue.enqueueNDRangeKernel(raytracer_kernel, cl::NullRange, global_work_size, local_work_size, nullptr, &kernel_ev);

    // Copy data from device memory to host memory (D2H copy), blocking call
    queue.enqueueReadBuffer(buffer_image, CL_TRUE, 0, sizeof(int) * dim * dim * 4, final_image, nullptr, &read_ev);

    // Calculate execution time using profiling info
    cl_ulong write_start = write_ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong write_end = write_ev.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    cl_ulong kernel_start = kernel_ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong kernel_end = kernel_ev.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    cl_ulong read_start = read_ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong read_end = read_ev.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    // Convert to milliseconds
    double write_time = (write_end - write_start) / 1e6;
    double kernel_time = (kernel_end - kernel_start) / 1e6;
    double read_time = (read_end - read_start) / 1e6;
    double total_time = write_time + kernel_time + read_time;

    printf("OpenCL\t%d\t%3.1f\n", dim, total_time);

    // genBpm("raytracer.bmp", dim, dim, final_image);

    free(temp_s);
    free(final_image);

    return 0;
}