#pragma once

#include "../cldef.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstdint>

/**
 * @class OCLInterface
 * @brief A class that provides an interface for OpenCL operations, including platform and device selection,
 * context and command queue creation, program and kernel management, and buffer operations.
 *
 * This class simplifies the process of working with OpenCL by encapsulating common tasks.
 *
 * @authors Henrique Gabriel Rodrigues, Prof. Dr. André Rauber Du Bois
 */
class OCLInterface
{
public:
    enum class DeviceType
    {
        CPU,
        GPU
    };

private:
    cl::Platform gpu_platform, cpu_platform;
    cl::Device gpu, cpu;
    cl::Context gpu_context, cpu_context;
    cl::CommandQueue gpu_command_queue, cpu_command_queue;

    /**
     * @deprecated This function is deprecated and is not recommended for use. I left it here just in case, but it may be removed in future versions.
     *
     * @brief Reads the kernel code from a file.
     *
     * @param file_name The name of the file containing the kernel code.
     * @return The kernel code as a string.
     */
    std::string getKernelCode(const char *file_name);

    /**
     * @brief OpenCL program build options for the GPU.
     */
    std::string build_options_gpu;
    /**
     * @brief OpenCL program build options for the CPU.
     */
    std::string build_options_cpu;

    /**
     * @brief Debug logs flag.
     */
    bool debug_logs;

    /**
     * @brief Alignment requirement in bytes of the CPU for efficient memory access.
     */
    cl_uint cpu_alignment_bytes = 0;

public:
    /**
     * @brief Default constructor for OCLInterface.
     *
     * Initializes an OCLInterface object with default settings.
     * Delegates construction to the parameterized constructor with 'false' as the argument for enable_debug_logs.
     */
    OCLInterface();
    /**
     * @brief Parameterized constructor for OCLInterface.
     *
     * Initializes an OCLInterface object with the specified debug log setting. The selected platform, device,
     * context, and command queue are initialized to null pointers for better error handling and predictable behavior.
     *
     * @param enable_debug_logs A boolean flag to enable or disable debug logs.
     */
    OCLInterface(bool enable_debug_logs);
    /**
     * @brief Destructor for OCLInterface.
     *
     * Guarantees that all operations in the command queue are completed before the object is destroyed,
     * ensuring proper cleanup of OpenCL resources.
     */
    ~OCLInterface();

    /**
     * @brief Sets the debug logs flag.
     *
     * @param enable A boolean flag to enable or disable debug logs.
     */
    void setDebugLogs(bool enable);

    /**
     * @brief Selects a CPU and GPU device from the available OpenCL platforms. When selected, initializes the corresponding OpenCL contexts and command queues for both devices.
     * If something goes wrong during platform/device selection, a detailed error message is printed to stderr and a runtime_error exception is thrown.
     */
    void selectPlatformsAndDevices();

    /**
     * @brief Checks if the selected device supports the specified OpenCL extensions.
     *
     * @param extensions A vector of strings containing extension names to check for support. Example: {"cl_khr_fp64"}
     * @param device_type The type of device for which to check the extensions (CPU or GPU).
     * @return A vector of pairs, where each pair contains the extension name and a boolean indicating support (true if
     * the extension is supported, false otherwise).
     */
    std::vector<std::pair<std::string, bool>> checkDeviceExtensions(std::vector<std::string> &extensions, DeviceType device_type);
    /**
     * @brief Sets the OpenCL program build options.
     *
     * @param options A string containing the build options. Example: "-D MY_DEFINE=1 -cl-fast-relaxed-math"
     * @param device_type The type of device for which to set the build options (CPU or GPU).
     */
    void setBuildOptions(const std::string &options, DeviceType device_type);

    /**
     * @brief Creates and builds an OpenCL program from the given program code string. The build options are used during the build process.
     * If something goes wrong during the build, a detailed error message is printed to stderr and a runtime_error exception is thrown.
     *
     * @param program_code A string containing the OpenCL program source code.
     * @param device_type The type of device where the program will be executed (CPU or GPU).
     * @return The created OpenCL program.
     */
    cl::Program createProgram(std::string &program_code, DeviceType device_type);
    /**
     * @brief Creates an OpenCL program from the given program code C-string. Same behavior as createProgram(std::string &).
     *
     * @param program_code A C-string containing the OpenCL program source code.
     * @param device_type The type of device where the program will be executed (CPU or GPU).
     * @return The created OpenCL program.
     */
    cl::Program createProgram(const char *program_code, DeviceType device_type);
    /**
     * @deprecated This function is deprecated and is not recommended for use. I left it here just in case, but it may be removed in future versions.
     *
     * @brief Creates an OpenCL program by reading the kernel code from the specified file. Same behavior as createProgram(std::string &).
     *
     * @param file_name The name of the file containing the OpenCL kernel code.
     * @param device_type The type of device where the program will be executed (CPU or GPU).
     * @return The created OpenCL program.
     */
    cl::Program createProgramFromFile(const char *file_name, DeviceType device_type);

    /**
     * @brief Creates an OpenCL kernel from the given program and kernel name.
     *
     * @param program The OpenCL program containing the kernel.
     * @param kernel_name The name of the kernel to create as a C-string.
     * @return The created OpenCL kernel.
     */
    cl::Kernel createKernel(const cl::Program &program, const char *kernel_name);

    /**
     * @brief Executes the given OpenCL kernel with the specified global and local work sizes. This call is non-blocking.
     *
     * @param kernel The OpenCL kernel to execute.
     * @param global_range The global work size as cl::NDRange.
     * @param local_range The local work size as cl::NDRange.
     * @param device_type The type of device where the kernel will be executed (CPU or GPU).
     */
    void executeKernel(cl::Kernel &kernel, const cl::NDRange &global_range, const cl::NDRange &local_range, DeviceType device_type);

    /**
     * @brief Creates an OpenCL buffer with the specified size and memory flags. If somehing goes wrong during buffer creation, a
     * detailed error message is printed to stderr and a runtime_error exception is thrown.
     *
     * @param size The size of the buffer to create in bytes.
     * @param flags The memory flags for the buffer (e.g., CL_MEM_READ_WRITE).
     * @param device_type The type of device where the buffer will be allocated (CPU or GPU).
     * @param host_ptr Optional pointer to host memory to initialize the buffer with (default is nullptr).
     * @return The created OpenCL buffer.
     */
    cl::Buffer createBuffer(size_t size, cl_mem_flags flags, DeviceType device_type, void *host_ptr = nullptr);
    /**
     * @brief Reads data from the given OpenCL buffer into the specified host memory. This call is blocking and
     * will wait until the read operation is complete.
     *
     * @param buffer The OpenCL buffer to read from.
     * @param host_ptr Pointer to the host memory where the data will be copied.
     * @param size The size of data to read in bytes.
     * @param offset The offset in the buffer from where to start reading (default is 0).
     */
    void readBuffer(const cl::Buffer &buffer, void *host_ptr, size_t size, DeviceType device_type, size_t offset = 0) const;
    /**
     * @brief Writes data from the specified host memory into the given OpenCL buffer. This call is blocking and
     * will wait until the write operation is complete.
     *
     * @param buffer The OpenCL buffer to write to.
     * @param host_ptr Pointer to the host memory containing the data to write.
     * @param size The size of data to write in bytes.
     * @param offset The offset in the buffer from where to start writing (default is 0).
     */
    void writeBuffer(const cl::Buffer &buffer, const void *host_ptr, size_t size, DeviceType device_type, size_t offset = 0) const;

    /**
     * @brief Creates an aligned shared virtual memory (SVM) region that can be accessed by both the host and the device.
     * This is used for efficient zero-copy data transfers.
     *
     * @param size The size of the SVM region to create in bytes.
     * @param device_type The type of device for which to create the SVM region (CPU or GPU).
     * @return A pointer to the allocated SVM memory that can be used on the host and the device.
     */
    void *createSVM(size_t size, DeviceType device_type);
    /**
     * @brief Frees a previously allocated aligned shared virtual memory (SVM) region.
     *
     * @param svm_ptr Pointer to the SVM memory to free.
     * @param device_type The type of device for which the SVM region was created (CPU or GPU).
     */
    void destroySVM(void *svm_ptr, DeviceType device_type);
    /**
     * TODO
     */
    void *mapSVM(const cl::Buffer &buffer, size_t size, DeviceType device_type) const;
    /**
     * TODO
     */
    void unMapSVM(void *host_ptr, DeviceType device_type) const;

    /**
     * @brief Synchronizes the OpenCL command queue, ensuring that all previously enqueued commands have completed.
     */
    void synchronize() const;

    // --- Getters ---

    std::string getBuildOptions(DeviceType device_type) const
    {
        if (device_type == DeviceType::GPU)
        {
            return build_options_gpu;
        }
        else
        {
            return build_options_cpu;
        }
    }

    cl_uint getCPUAlignmentBytes() const
    {
        return cpu_alignment_bytes;
    }
};
