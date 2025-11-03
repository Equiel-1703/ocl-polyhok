/*
    This header defines the OpenCL version to use, include OpenCL headers and enables C++ exceptions. 
    This is used only for the OpenCL benchmarks.
    
    It is configured to use OpenCL 2.0 features.
*/

#pragma once

#define OPENCL_VERSION 200 // We are going to be using OpenCL 2.0

#define CL_TARGET_OPENCL_VERSION OPENCL_VERSION
#define CL_HPP_TARGET_OPENCL_VERSION OPENCL_VERSION
#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>
#include <vector>

// Useful functions
inline cl::Platform getDefaultPlatform()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty())
    {
        throw std::runtime_error("No OpenCL platforms found");
    }

    return platforms[0];
}

inline cl::Device getDefaultDevice(const cl::Platform &platform, cl_device_type device_type = CL_DEVICE_TYPE_GPU)
{
    std::vector<cl::Device> devices;
    platform.getDevices(device_type, &devices);

    if (devices.empty())
    {
        throw std::runtime_error("No OpenCL devices found for the selected platform");
    }

    return devices[0];
}