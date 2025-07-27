#pragma once

#include "../cldef.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

class OCLInterface
{
private:
    cl::Platform selected_platform;
    cl::Device selected_device;
    cl::Context context;

    cl::CommandQueue command_queue;

    void createContext();
    void createCommandQueue();
    
    std::string getKernelCode(const char *file_name);
public:
    OCLInterface();
    ~OCLInterface();

    std::vector<cl::Platform> getAvailablePlatforms();
    void selectPlatform(cl::Platform p);
    void selectDefaultPlatform();

    std::vector<cl::Device> getAvailableDevices(cl_device_type device_type = CL_DEVICE_TYPE_ALL);
    void selectDevice(cl::Device d);
    void selectDefaultDevice(cl_device_type device_type = CL_DEVICE_TYPE_ALL);

    cl::Program createProgram(const char *file_name);
    cl::Kernel createKernel(const cl::Program &program, const char *kernel_name);
    cl::Buffer createBuffer(size_t size, cl_mem_flags flags, void *host_ptr = nullptr);

    void executeKernel(cl::Kernel &kernel, const cl::NDRange &global_range, const cl::NDRange &local_range);
    void readBuffer(const cl::Buffer &buffer, void *host_ptr, size_t size, size_t offset = 0);
    void writeBuffer(const cl::Buffer &buffer, const void *host_ptr, size_t size, size_t offset = 0);

    cl::Context getContext() const { return context; }
    cl::Device getSelectedDevice() const { return selected_device; }
    cl::Platform getSelectedPlatform() const { return selected_platform; }
    cl::CommandQueue getCommandQueue() const { return command_queue; }
};
