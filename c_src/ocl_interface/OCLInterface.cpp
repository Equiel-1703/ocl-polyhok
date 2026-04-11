#include "OCLInterface.hpp"

OCLInterface::OCLInterface() : OCLInterface(false) {}

OCLInterface::OCLInterface(bool enable_debug_logs)
{
    // Initialize OpenCL interface with null pointers for
    // for good error handling and predictable behavior

    this->gpu_platform = nullptr;
    this->gpu = nullptr;
    this->gpu_context = nullptr;
    this->gpu_command_queue = nullptr;

    this->cpu_platform = nullptr;
    this->cpu = nullptr;
    this->cpu_context = nullptr;
    this->cpu_command_queue = nullptr;

    this->build_options_gpu = "";
    this->debug_logs = enable_debug_logs;
}

OCLInterface::~OCLInterface()
{
    // Clean up OpenCL resources if they were created
    if (this->gpu_command_queue() != nullptr)
    {
        this->gpu_command_queue.finish();
    }

    if (this->cpu_command_queue() != nullptr)
    {
        this->cpu_command_queue.finish();
    }
}

void OCLInterface::setDebugLogs(bool enable)
{
    this->debug_logs = enable;
}

// This function is deprecated.
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

void OCLInterface::selectPlatformsAndDevices()
{
    // Getting available OpenCL platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    // Now we can iterate over the platforms and look for GPU and CPU devices.
    // I'll be selecting the first GPU and CPU devices I find, but we could implement something more robust in the future.

    bool gpu_found = false, cpu_found = false;
    std::vector<cl::Device> devices;

    for (auto &p : platforms)
    {
        devices.clear();
        p.getDevices(CL_DEVICE_TYPE_ALL, &devices);

        // Iterating over devices of the current platform p
        for (auto &d : devices)
        {
            cl_device_type d_type = d.getInfo<CL_DEVICE_TYPE>();

            if (d_type == CL_DEVICE_TYPE_GPU && !gpu_found)
            {
                this->gpu_platform = p;
                this->gpu = d;
                gpu_found = true;
            }
            else if (d_type == CL_DEVICE_TYPE_CPU && !cpu_found)
            {
                this->cpu_platform = p;
                this->cpu = d;
                cpu_found = true;
            }
        }

        if (gpu_found && cpu_found)
        {
            break;
        }
    }

    if (!cpu_found || !gpu_found)
    {
        std::cerr << "[OCL C++ Interface] Error: Unable to find both a GPU and a CPU device on the available OpenCL platforms." << std::endl;
        std::cerr << "> GPU found: " << (gpu_found ? "Yes" : "No") << std::endl;
        std::cerr << "> CPU found: " << (cpu_found ? "Yes" : "No") << std::endl;

        throw std::runtime_error("Required OpenCL devices not found");
    }

    // ---- for debug only, remove this stuff later ----
    std::cout << "Selected GPU: " << this->gpu.getInfo<CL_DEVICE_NAME>() << " from platform " << this->gpu_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    std::cout << "Selected CPU: " << this->cpu.getInfo<CL_DEVICE_NAME>() << " from platform " << this->cpu_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

    // Creating contexts and command queues for the selected devices
    this->gpu_context = cl::Context(this->gpu);
    this->gpu_command_queue = cl::CommandQueue(this->gpu_context, this->gpu);

    this->cpu_context = cl::Context(this->cpu);
    this->cpu_command_queue = cl::CommandQueue(this->cpu_context, this->cpu);

    // Get CPU alignment requirements for efficient memory transfers
    this->cpu_alignment_bytes = this->cpu.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>();
    this->cpu_alignment_bytes /= 8; // Convert from bits to bytes
}

std::vector<std::pair<std::string, bool>> OCLInterface::checkDeviceExtensions(std::vector<std::string> &extensions, OCLInterface::DeviceType device_type)
{
    std::string device_extensions = (device_type == DeviceType::GPU) ? this->gpu.getInfo<CL_DEVICE_EXTENSIONS>() : this->cpu.getInfo<CL_DEVICE_EXTENSIONS>();
    std::vector<std::pair<std::string, bool>> results;

    for (const auto &ext : extensions)
    {
        bool supported = (device_extensions.find(ext) != std::string::npos);
        results.push_back(std::make_pair(ext, supported));
    }

    return results;
}

void OCLInterface::setBuildOptions(const std::string &options, OCLInterface::DeviceType device_type)
{
    if (device_type == DeviceType::GPU)
    {
        this->build_options_gpu = options;
    }
    else
    {
        this->build_options_cpu = options;
    }

    if (this->debug_logs)
    {
        std::cout << "[OCL C++ Interface] Set OpenCL build options to " << (device_type == DeviceType::GPU ? "GPU" : "CPU") << ": " << options << std::endl;
    }
}

cl::Program OCLInterface::createProgram(std::string &program_code, OCLInterface::DeviceType device_type)
{
    cl::Context &context = (device_type == DeviceType::GPU) ? this->gpu_context : this->cpu_context;
    cl::Device &device = (device_type == DeviceType::GPU) ? this->gpu : this->cpu;

    cl::Program program(context, program_code);

    try
    {
        program.build(device, this->build_options_gpu.c_str());
    }
    catch (const cl::BuildError &err)
    {
        std::cerr << "[OCL C++ Interface] Build Error!" << std::endl;

        std::string device_name = err.getBuildLog().front().first.getInfo<CL_DEVICE_NAME>();
        std::string build_log = err.getBuildLog().front().second;

        std::cerr << "> Device: " << device_name << std::endl;
        std::cerr << "> Build Log:\n"
                  << std::endl;
        std::cerr << build_log << std::endl;

        throw std::runtime_error("Failed to build OpenCL program");
    }

    if (this->debug_logs)
    {
        std::cout << "[OCL C++ Interface] OpenCL program created and builded successfully." << std::endl;
    }

    return program;
}

cl::Program OCLInterface::createProgram(const char *program_code, OCLInterface::DeviceType device_type)
{
    std::string code_str(program_code);
    return this->createProgram(code_str, device_type);
}

cl::Program OCLInterface::createProgramFromFile(const char *file_name, OCLInterface::DeviceType device_type)
{
    std::string code_str = this->getKernelCode(file_name);
    return this->createProgram(code_str, device_type);
}

cl::Kernel OCLInterface::createKernel(const cl::Program &program, const char *kernel_name)
{
    try
    {
        cl::Kernel kernel(program, kernel_name);

        if (this->debug_logs)
        {
            std::cout << "[OCL C++ Interface] OpenCL kernel '" << kernel_name << "' created successfully." << std::endl;
        }

        return kernel;
    }
    catch (const cl::Error &e)
    {
        std::cerr << "[OCL C++ Interface] Failed to create OpenCL kernel '" << kernel_name << "'." << std::endl;
        std::cerr << "> Error code: " << e.err() << std::endl;
        std::cerr << "> Error message: " << e.what() << std::endl;

        throw std::runtime_error("Failed to create OpenCL kernel");
    }
}

cl::Buffer OCLInterface::createBuffer(size_t size, cl_mem_flags flags, OCLInterface::DeviceType device_type, void *host_ptr)
{
    cl::Context &context = (device_type == DeviceType::GPU) ? this->gpu_context : this->cpu_context;
    cl::Device &device = (device_type == DeviceType::GPU) ? this->gpu : this->cpu;
    std::string device_type_str = (device_type == DeviceType::GPU) ? "GPU" : "CPU";

    try
    {
        cl::Buffer buffer(context, flags, size, host_ptr);

        if (this->debug_logs)
        {
            std::cout << "[OCL C++ Interface] OpenCL buffer of size " << size << " created successfully in the " << device_type_str << "." << std::endl;
        }

        return buffer;
    }
    catch (const cl::Error &e)
    {
        // e.what() provides a description of the error, for example "clCreateBuffer"
        // e.err() provides the OpenCL error code (e.g., CL_MEM_OBJECT_ALLOCATION_FAILURE)
        cl_int error_code = e.err();
        std::string error_msg;

        // Retrieve device memory info for better error messages
        cl_ulong global_mem = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();   // Total global memory size
        cl_ulong max_alloc = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>(); // Max allocation size allowed

        // Converting sizes to MB for easier readability
        cl_ulong global_mem_mb = global_mem / (1024 * 1024);
        cl_ulong max_alloc_mb = max_alloc / (1024 * 1024);
        cl_ulong buff_size_mb = size / (1024 * 1024);

        switch (error_code)
        {
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            std::cerr << "[OCL C++ Interface] Error: " << device_type_str << " out of memory for buffer allocation of size " << size << "." << std::endl;
            std::cerr << "> Device Global Memory Size: " << global_mem_mb << " MB" << std::endl;
            std::cerr << "> Requested Buffer Size: " << buff_size_mb << " MB" << std::endl;

            error_msg = device_type_str + " out of memory for buffer allocation";
            break;

        case CL_INVALID_BUFFER_SIZE:
            std::cerr << "[OCL C++ Interface] Error: Invalid buffer size requested: " << size << " bytes." << std::endl;
            std::cerr << ">  Device Global Memory Size: " << global_mem_mb << " MB" << std::endl;
            std::cerr << ">  Device Max Allocation Size: " << max_alloc_mb << " MB" << std::endl;
            std::cerr << ">  Requested Buffer Size: " << buff_size_mb << " MB" << std::endl;

            error_msg = "Invalid buffer size requested";
            break;

        default:
            std::cerr << "[OCL C++ Interface] Failed to create OpenCL buffer of size " << size << "." << std::endl;
            std::cerr << "> Error: " << e.what() << std::endl;
            std::cerr << "> Error code: " << std::to_string(error_code) << std::endl;

            std::cerr << "\n[Device Memory Info]" << std::endl;
            std::cerr << "> Device Global Memory Size: " << global_mem_mb << " MB" << std::endl;
            std::cerr << "> Device Max Allocation Size: " << max_alloc_mb << " MB" << std::endl;
            std::cerr << "> Requested Buffer Size: " << buff_size_mb << " MB" << std::endl;

            error_msg = std::string(e.what()) + " (Error code: " + std::to_string(error_code) + ")";
            break;
        }

        throw std::runtime_error(error_msg);
    }
}

void OCLInterface::executeKernel(cl::Kernel &kernel, const cl::NDRange &global_range, const cl::NDRange &local_range, OCLInterface::DeviceType device_type)
{
    cl::CommandQueue &command_queue = (device_type == DeviceType::GPU) ? this->gpu_command_queue : this->cpu_command_queue;

    try
    {
        command_queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_range, local_range);
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

void OCLInterface::readBuffer(const cl::Buffer &buffer, void *host_ptr, size_t size, DeviceType device_type, size_t offset) const
{
    const cl::CommandQueue &command_queue = (device_type == DeviceType::GPU) ? this->gpu_command_queue : this->cpu_command_queue;
    command_queue.enqueueReadBuffer(buffer, CL_TRUE, offset, size, host_ptr);
}

void OCLInterface::writeBuffer(const cl::Buffer &buffer, const void *host_ptr, size_t size, DeviceType device_type, size_t offset) const
{
    const cl::CommandQueue &command_queue = (device_type == DeviceType::GPU) ? this->gpu_command_queue : this->cpu_command_queue;

    // This function could be non-blocking in the future...
    command_queue.enqueueWriteBuffer(buffer, CL_TRUE, offset, size, host_ptr);
}

void *OCLInterface::createSVM(size_t size, OCLInterface::DeviceType device_type)
{
    cl::Context &context = (device_type == DeviceType::GPU) ? this->gpu_context : this->cpu_context;
    cl::CommandQueue &command_queue = (device_type == DeviceType::GPU) ? this->gpu_command_queue : this->cpu_command_queue;
    std::string device_type_str = (device_type == DeviceType::GPU) ? "GPU" : "CPU";

    try
    {
        // Since we are using OpenCL 2.0, we can use Shared Virtual Memory (SVM) for efficient host-device memory sharing.
        // This allows us to allocate memory that is directly accessible by both the host and the device via the same
        // pointer, without needing a cl::Buffer object. This is ideal for our use case, since our goal for aligned host
        // memory is for efficient CPU parallel execution, not the GPU.

        void *svm_shared_pointer = clSVMAlloc(
            context(), // Use the underlying cl_context from the C API
            CL_MEM_READ_WRITE,
            size,
            0);

        // Check if the allocation was successful. If svm_shared_pointer is nullptr, it means the allocation failed.
        if (svm_shared_pointer == nullptr)
        {
            std::cerr << "[OCL C++ Interface] Failed to allocate aligned SVM memory of size " << size << " bytes for the " << device_type_str << "." << std::endl;
            throw std::runtime_error("Failed to allocate aligned Sjared Virtual Memory");
        }

        // We are using SVM coarse-grained sharing. This is because coarse-grained is widerly supported in essentially all OpenCL 2.0
        // compatible devices. Unfortunately, coarse-grained SVM needs explicit synchronization.
        // Therefore, we need to map/unmap the SVM pointer to ensure proper synchronization.

        // Map the SVM pointer to ensure it's properly synchronized and can be safely accessed by the host.
        command_queue.enqueueMapSVM(
            svm_shared_pointer,
            CL_TRUE,                    // Blocking call to ensure the mapping is complete before we return the pointer
            CL_MAP_READ | CL_MAP_WRITE, // We want to read and write to this memory
            size);                      // Size of the memory to map

        if (this->debug_logs)
        {
            std::cout
                << "[OCL C++ Interface] Aligned SVM memory of size " << size << " bytes created successfully for the "
                << device_type_str << " at address "
                << svm_shared_pointer << "." << std::endl;
        }

        return svm_shared_pointer;
    }
    catch (const cl::Error &e)
    {
        std::cerr << "[OCL C++ Interface] Failed to create aligned SVM memory. Error code: " << e.what() << std::endl;
        throw std::runtime_error("Failed to create aligned host memory");
    }
}

void OCLInterface::destroySVM(void *svm_ptr, DeviceType device_type)
{
    cl::Context &context = (device_type == DeviceType::GPU) ? this->gpu_context : this->cpu_context;
    std::string device_type_str = (device_type == DeviceType::GPU) ? "GPU" : "CPU";

    try
    {
        clSVMFree(context(), svm_ptr);

        if (this->debug_logs)
        {
            std::cout << "[OCL C++ Interface] Aligned SVM memory freed successfully from the " << device_type_str << "." << std::endl;
        }
    }
    catch (const cl::Error &e)
    {
        std::cerr << "[OCL C++ Interface] Failed to free aligned SVM memory. Error code: " << e.what() << std::endl;
        throw std::runtime_error("Failed to free aligned host memory");
    }
}

void OCLInterface::mapSVM(void *host_ptr, size_t size, DeviceType device_type) const
{
    const cl::CommandQueue &command_queue = (device_type == DeviceType::GPU) ? this->gpu_command_queue : this->cpu_command_queue;
    std::string device_type_str = (device_type == DeviceType::GPU) ? "GPU" : "CPU";

    try
    {
        command_queue.enqueueMapSVM(
            host_ptr,
            CL_TRUE,                    // Blocking call to ensure the mapping is complete before returning
            CL_MAP_READ | CL_MAP_WRITE, // We want to read and write to this memory
            size);                      // Size of the memory to map

        if (this->debug_logs)
        {
            std::cout << "[OCL C++ Interface] Mapped SVM memory for " << device_type_str << " at address " << host_ptr << "." << std::endl;
        }
    }
    catch (const cl::Error &e)
    {
        std::cerr << "[OCL C++ Interface] Failed to map SVM memory for " << (device_type == DeviceType::GPU ? "GPU" : "CPU") << ". Error code: " << e.what() << std::endl;
        throw std::runtime_error("Failed to map SVM memory");
    }
}

void OCLInterface::unMapSVM(void *host_ptr, DeviceType device_type) const
{
    const cl::CommandQueue &command_queue = (device_type == DeviceType::GPU) ? this->gpu_command_queue : this->cpu_command_queue;
    std::string device_type_str = (device_type == DeviceType::GPU) ? "GPU" : "CPU";

    try
    {
        command_queue.enqueueUnmapSVM(host_ptr);

        if (this->debug_logs)
        {
            std::cout << "[OCL C++ Interface] Unmapped SVM memory for " << device_type_str << " at address " << host_ptr << "." << std::endl;
        }
    }
    catch (const cl::Error &e)
    {
        std::cerr << "[OCL C++ Interface] Failed to unmap SVM memory for " << device_type_str << ". Error code: " << e.what() << std::endl;
        throw std::runtime_error("Failed to unmap SVM memory");
    }
}

void OCLInterface::synchronize(DeviceType device_type) const
{
    const cl::CommandQueue &command_queue = (device_type == DeviceType::GPU) ? this->gpu_command_queue : this->cpu_command_queue;
    command_queue.finish();
}