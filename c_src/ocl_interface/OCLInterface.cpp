#include "OCLInterface.hpp"

OCLInterface::OCLInterface() : OCLInterface(false) {}

OCLInterface::OCLInterface(bool enable_debug_logs)
{
    // Initialize OpenCL interface with null pointers for
    // for good error handling and predictable behavior
    this->selected_platform = nullptr;
    this->selected_device = nullptr;
    this->context = nullptr;
    this->command_queue = nullptr;

    this->build_options = "";
    this->debug_logs = enable_debug_logs;
}

OCLInterface::~OCLInterface()
{
    // Clean up OpenCL resources if they were created
    if (this->command_queue() != nullptr)
    {
        this->command_queue.finish();
    }
}

void OCLInterface::setDebugLogs(bool enable)
{
    this->debug_logs = enable;
}

void OCLInterface::createContext()
{
    this->context = cl::Context(this->selected_device);
}

void OCLInterface::createCommandQueue()
{
    this->command_queue = cl::CommandQueue(this->context, this->selected_device);
}

std::string OCLInterface::getKernelCode(const char *file_name)
{
    std::ifstream kernel_file(file_name);
    std::string output, line;

    if (!kernel_file.is_open())
    {
        std::cerr << "[OCL C++ Interface] Unable to open kernel file '" << file_name << "'." << std::endl;
        throw std::runtime_error("Kernel file not found");
    }

    while (std::getline(kernel_file, line))
    {
        output += line += "\n";
    }

    kernel_file.close();

    return output;
}

std::vector<cl::Platform> OCLInterface::getAvailablePlatforms()
{
    std::vector<cl::Platform> platforms;

    cl::Platform::get(&platforms);

    return platforms;
}

void OCLInterface::selectPlatform(cl::Platform p)
{
    if (p() == nullptr)
    {
        throw std::runtime_error("Invalid OpenCL platform selected");
    }

    this->selected_platform = p;

    if (this->debug_logs)
    {
        std::cout << "[OCL C++ Interface] Selected OpenCL platform: " << this->selected_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    }
}

void OCLInterface::selectDefaultPlatform()
{
    std::vector<cl::Platform> platforms = this->getAvailablePlatforms();

    if (platforms.empty())
    {
        throw std::runtime_error("No OpenCL platforms found");
    }

    this->selectPlatform(platforms[0]);
}

std::vector<cl::Device> OCLInterface::getAvailableDevices(cl_device_type device_type)
{
    if (this->selected_platform() == nullptr)
    {
        throw std::runtime_error("No OpenCL platform selected");
    }

    std::vector<cl::Device> devices;
    this->selected_platform.getDevices(device_type, &devices);

    return devices;
}

void OCLInterface::selectDevice(cl::Device d)
{
    if (d() == nullptr)
    {
        throw std::runtime_error("Invalid OpenCL device selected");
    }

    this->selected_device = d;

    this->createContext();
    this->createCommandQueue();

    if (this->debug_logs)
    {
        std::cout << "[OCL C++ Interface] Selected OpenCL device: " << this->selected_device.getInfo<CL_DEVICE_NAME>() << std::endl;
    }
}

void OCLInterface::selectDefaultDevice(cl_device_type device_type)
{
    std::vector<cl::Device> devices = this->getAvailableDevices(device_type);

    if (devices.empty())
    {
        throw std::runtime_error("No OpenCL devices found");
    }

    this->selectDevice(devices[0]);
}

std::vector<std::pair<std::string, bool>> OCLInterface::checkDeviceExtensions(std::vector<std::string> &extensions)
{
    if (this->selected_device() == nullptr)
    {
        throw std::runtime_error("No OpenCL device selected");
    }

    std::string device_extensions = this->selected_device.getInfo<CL_DEVICE_EXTENSIONS>();
    std::vector<std::pair<std::string, bool>> results;

    for (const auto &ext : extensions)
    {
        bool supported = (device_extensions.find(ext) != std::string::npos);
        results.push_back(std::make_pair(ext, supported));
    }

    return results;
}

void OCLInterface::setBuildOptions(const std::string &options)
{
    this->build_options = options;

    if (this->debug_logs)
    {
        std::cout << "[OCL C++ Interface] Set OpenCL build options: " << this->build_options << std::endl;
    }
}

cl::Program OCLInterface::createProgram(std::string &program_code)
{
    cl::Program program(this->context, program_code);

    try
    {
        program.build(this->selected_device, this->build_options.c_str());
    }
    catch (const cl::BuildError &err)
    {
        std::cerr << "[OCL C++ Interface] Build Error!" << std::endl;

        std::string device_name = err.getBuildLog().front().first.getInfo<CL_DEVICE_NAME>();
        std::string build_log = err.getBuildLog().front().second;

        std::cerr << "> Device: " << device_name << std::endl;
        std::cerr << "> Build Log:\n" << std::endl;
        std::cerr << build_log << std::endl;
        
        throw std::runtime_error("Failed to build OpenCL program");
    }

    if (this->debug_logs)
    {
        std::cout << "[OCL C++ Interface] OpenCL program created and builded successfully." << std::endl;
    }

    return program;
}

cl::Program OCLInterface::createProgram(const char *program_code)
{
    std::string code_str(program_code);
    return this->createProgram(code_str);
}

cl::Program OCLInterface::createProgramFromFile(const char *file_name)
{
    std::string code_str = this->getKernelCode(file_name);
    return this->createProgram(code_str);
}

cl::Kernel OCLInterface::createKernel(const cl::Program &program, const char *kernel_name)
{
    cl_int err = CL_SUCCESS;
    cl::Kernel kernel(program, kernel_name, &err);

    if (err != CL_SUCCESS || kernel() == nullptr)
    {
        std::cerr << "[OCL C++ Interface] Failed to create OpenCL kernel '" << kernel_name << "'. Error code: " << err << std::endl;
        throw std::runtime_error("Failed to create OpenCL kernel");
    }

    if (this->debug_logs)
    {
        std::cout << "[OCL C++ Interface] OpenCL kernel '" << kernel_name << "' created successfully." << std::endl;
    }

    return kernel;
}

cl::Buffer OCLInterface::createBuffer(size_t size, cl_mem_flags flags, void *host_ptr)
{
    cl::Buffer buffer(this->context, flags, size, host_ptr);

    if (buffer() == nullptr)
    {
        std::cerr << "[OCL C++ Interface] Failed to create OpenCL buffer of size " << size << "." << std::endl;
        throw std::runtime_error("Failed to create OpenCL buffer");
    }

    if (this->debug_logs)
    {
        std::cout << "[OCL C++ Interface] OpenCL buffer of size " << size << " created successfully." << std::endl;
    }

    return buffer;
}

void OCLInterface::executeKernel(cl::Kernel &kernel, const cl::NDRange &global_range, const cl::NDRange &local_range)
{
    try
    {
        this->command_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_range, local_range);
        this->command_queue.finish();
    }
    catch (const cl::Error &err)
    {
        std::cerr << "[OCL C++ Interface] Failed to execute OpenCL kernel." << std::endl;
        std::cerr << "> Error code: " << err.err() << std::endl;
        std::cerr << "> Error message: " << err.what() << std::endl;
        throw std::runtime_error("Failed to execute OpenCL kernel");
    }

    if (this->debug_logs)
    {
        std::cout << "[OCL C++ Interface] OpenCL kernel executed successfully." << std::endl;
    }
}

void OCLInterface::readBuffer(const cl::Buffer &buffer, void *host_ptr, size_t size, size_t offset) const
{
    this->command_queue.enqueueReadBuffer(buffer, CL_TRUE, offset, size, host_ptr);
}

void OCLInterface::writeBuffer(const cl::Buffer &buffer, const void *host_ptr, size_t size, size_t offset) const
{
    this->command_queue.enqueueWriteBuffer(buffer, CL_TRUE, offset, size, host_ptr);
}

void *OCLInterface::mapHostPtrToPinnedMemory(const cl::Buffer &buffer, cl_map_flags flags, size_t size, size_t offset) const
{
    cl_int err;

    void *mapped_ptr = this->command_queue.enqueueMapBuffer(buffer, CL_TRUE, flags, offset, size, nullptr, nullptr, &err);

    if (err != CL_SUCCESS)
    {
        std::cerr << "[OCL C++ Interface] Failed to map OpenCL buffer to pinned memory. Error code: " << err << std::endl;
        throw std::runtime_error("Failed to map OpenCL buffer to pinned memory");
    }

    return mapped_ptr;
}

void OCLInterface::unMapHostPtr(const cl::Buffer &buffer, void *host_ptr) const
{
    this->command_queue.enqueueUnmapMemObject(buffer, host_ptr);
    this->command_queue.finish(); // Ensure the unmap operation is complete
}

void OCLInterface::synchronize() const
{
    this->command_queue.finish(); // Wait for all commands up to this point to complete in the command queue
}