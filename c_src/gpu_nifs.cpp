/*
    This file implements the Native Implemented Functions (NIFs) for GPU operations using OpenCL
    in Elixir.

    Ported to OpenCL/C++ by: Henrique Gabriel Rodrigues
    Oriented and supervised by: Prof. Dr. André Rauber Du Bois
    Original code by: Prof. Dr. André Rauber Du Bois

    Laboratory of Ubiquitous and Parallel Systems (LUPS) - Universidade Federal de Pelotas (UFPel)
*/

#include "ocl_interface/OCLInterface.hpp"

#include <erl_nif.h>

#include <iostream>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <chrono>

bool debug_logs = false;
bool fp64_supported = false;
bool int64_base_atomics_supported = false;

OCLInterface *open_cl = nullptr;

// Global resource type for GPU arrays (cl::Buffer objects)
ErlNifResourceType *ARRAY_TYPE;
// Global resource type for aligned host memory (pinned memory for efficient transfers)
ErlNifResourceType *CPU_SVM_TYPE;

// Destructor for device array resource (cl::Buffer)
void dev_array_destructor(ErlNifEnv * /* env */, void *res)
{
  cl::Buffer *dev_array = (cl::Buffer *)res;

  // Explicitly call the destructor for the cl::Buffer object without deallocating
  // the resource memory itself (the memory where the pointer to cl::Buffer is stored).
  // This is Erlang's garbage collector responsibility, and if we do this we'll get a
  // deallocation error.
  dev_array->~Buffer();

  if (debug_logs)
  {
    std::cout << "[C++ GPU NIF] Device array resource destroyed." << std::endl;
  }
}

// Destructor for aligned host memory resource
void cpu_svm_destructor(ErlNifEnv * /* env */, void *res)
{
  void *svm_ptr = *((void **)res);

  if (svm_ptr == nullptr)
  {
    std::cerr << "[C++ GPU NIF] Warning: Attempted to destroy a null SVM pointer. Something went very wrong..." << std::endl;
    return;
  }

  try
  {
    open_cl->destroySVM(svm_ptr, OCLInterface::DeviceType::CPU);

    if (debug_logs)
    {
      std::cout << "[C++ GPU NIF] CPU SVM array at address " << svm_ptr << " was successfully freed." << std::endl;
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << '\n';
  }
}

// This function initializes the OpenCL interface, selects the default platform, GPU device,
// and checks for required extension support.
void init_ocl(ErlNifEnv *env)
{
  if (open_cl != nullptr)
    return; // Already initialized

  open_cl = new OCLInterface(debug_logs);

  try
  {
    // This will scan the available platforms and devices and select the first GPU and CPU it finds.
    // If it doesn't find a CPU and GPU, it will throw an exception and print an error message in cerr.
    open_cl->selectPlatformsAndDevices();

    // Check for double data type support in GPU and int64 atomics support (required for compare and swap (CAS) operations)
    std::vector<std::string> desired_extensions = {"cl_khr_fp64", "cl_khr_int64_base_atomics"};
    std::vector<std::pair<std::string, bool>> extensions_supported = open_cl->checkDeviceExtensions(desired_extensions, OCLInterface::DeviceType::GPU);

    // Update global flags for extension support
    for (const auto &ext : extensions_supported)
    {
      if (ext.first == "cl_khr_fp64")
      {
        fp64_supported = ext.second;
      }
      else if (ext.first == "cl_khr_int64_base_atomics")
      {
        int64_base_atomics_supported = ext.second;
      }
    }

    // If both extensions are supported, then the selected GPU can handle the double type
    if (fp64_supported && int64_base_atomics_supported)
    {
      // Define flag for double support in OpenCL build options
      open_cl->setBuildOptions("-D DOUBLE_SUPPORTED=1", OCLInterface::DeviceType::GPU);
    }

    // Add ignore warnings build option (regardless of double support) in the GPU
    open_cl->setBuildOptions(open_cl->getBuildOptions(OCLInterface::DeviceType::GPU) + " -w", OCLInterface::DeviceType::GPU);
  }
  catch (const std::exception &e)
  {
    std::cerr << "[ERROR] Failed to initialize OpenCL interface: " << e.what() << std::endl;
    enif_raise_exception(env, enif_make_string(env, e.what(), ERL_NIF_LATIN1));

    delete open_cl;
  }
}

// This function is called when the NIF library is loaded
static int load(ErlNifEnv *env, void ** /* priv_data */, ERL_NIF_TERM /* load_info */)
{
  // Defines the resource type for GPU arrays (Buffer objects in our case)
  ARRAY_TYPE = enif_open_resource_type(
      env,
      NULL,
      "gpu_ref",
      dev_array_destructor,
      ERL_NIF_RT_CREATE,
      NULL);

  // Defines the resource type for Shared Virtual Memory (SVM) pointers (aligned host memory).
  CPU_SVM_TYPE = enif_open_resource_type(
      env,
      NULL,
      "cpu_svm_ref",
      cpu_svm_destructor,
      ERL_NIF_RT_CREATE,
      NULL);

  // Initialize OpenCL interface
  init_ocl(env);

  return 0;
}

// This function is called when the NIF library is unloaded
static void unload(ErlNifEnv * /* env */, void * /* priv_data */)
{
  if (open_cl != nullptr)
  {
    delete open_cl;
    open_cl = nullptr;
  }

  if (debug_logs)
  {
    std::cout << "[C++ GPU NIF] GPU NIFs unloaded successfully." << std::endl;
  }
}

// This function is used to retrieve the OpenCL device type (GPU or CPU) from the given Erlang term (an Elixir atom)
// and convert it to the corresponding OCLInterface::DeviceType enum value.
// If the atom is not a valid device type, it throws an exception.
// Since this is used frequently in the NIFs, I defined it as an inline function for better performance.
inline OCLInterface::DeviceType get_device_type(ERL_NIF_TERM e_device_type, ErlNifEnv *env)
{
  if (enif_is_identical(e_device_type, enif_make_atom(env, "gpu")))
  {
    return OCLInterface::DeviceType::GPU;
  }
  else if (enif_is_identical(e_device_type, enif_make_atom(env, "cpu")))
  {
    return OCLInterface::DeviceType::CPU;
  }
  else
  {
    std::cerr << "[ERROR] Invalid device type. Expected 'gpu' or 'cpu'." << std::endl;
    throw std::invalid_argument("Invalid device type");
  }
}

// This function compiles the given kernel code and launches it with the specified blocks and threads.
static ERL_NIF_TERM jit_compile_and_launch_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  // Check argc
  if (argc != 8)
  {
    std::cerr << "[ERROR] Invalid number of arguments for jit_compile_and_launch_nif." << std::endl;
    return enif_make_badarg(env);
  }

  // Get kernel name
  ERL_NIF_TERM e_name = argv[0];
  unsigned int size_name;
  if (!enif_get_list_length(env, e_name, &size_name))
  {
    return enif_make_badarg(env);
  }

  std::string kernel_name(size_name, '\0');
  enif_get_string(env, e_name, kernel_name.data(), size_name + 1, ERL_NIF_LATIN1);

  // Get kernel code to compile
  ERL_NIF_TERM e_code = argv[1];
  unsigned int size_code;
  if (!enif_get_list_length(env, e_code, &size_code))
  {
    return enif_make_badarg(env);
  }

  std::string code(size_code, '\0');
  enif_get_string(env, e_code, code.data(), size_code + 1, ERL_NIF_LATIN1);

  // Creating program and kernel objects
  cl::Program program;
  cl::Kernel kernel;

  // Getting device type (GPU or CPU). It is the last argument.
  ERL_NIF_TERM e_device_type = argv[7];
  OCLInterface::DeviceType device_type = get_device_type(e_device_type, env);

  try
  {
    program = open_cl->createProgram(code, device_type);
    kernel = open_cl->createKernel(program, kernel_name.c_str());
  }
  catch (const std::exception &e)
  {
    return enif_raise_exception(env, enif_make_string(env, e.what(), ERL_NIF_LATIN1));
  }

  // Getting blocks and threads tuples pointers
  const ERL_NIF_TERM *tuple_blocks, *tuple_threads;
  int arity;

  if (!enif_get_tuple(env, argv[2], &arity, &tuple_blocks))
  {
    std::cerr << "[ERROR] The given blocks argument is not a tuple." << std::endl;
    return enif_make_badarg(env);
  }

  if (arity != 3)
  {
    std::cerr << "[ERROR] The blocks and threads tuples must have exactly 3 elements (for x, y, z dimensions)." << std::endl;
    return enif_make_badarg(env);
  }

  if (!enif_get_tuple(env, argv[3], &arity, &tuple_threads))
  {
    std::cerr << "[ERROR] The given threads argument is not a tuple." << std::endl;
    return enif_make_badarg(env);
  }

  if (arity != 3)
  {
    std::cerr << "[ERROR] The blocks and threads tuples must have exactly 3 elements (for x, y, z dimensions)." << std::endl;
    return enif_make_badarg(env);
  }

  // Extracting the number of blocks and threads from the tuples
  int blocks_x, blocks_y, blocks_z, threads_x, threads_y, threads_z;

  enif_get_int(env, tuple_blocks[0], &blocks_x);
  enif_get_int(env, tuple_blocks[1], &blocks_y);
  enif_get_int(env, tuple_blocks[2], &blocks_z);
  enif_get_int(env, tuple_threads[0], &threads_x);
  enif_get_int(env, tuple_threads[1], &threads_y);
  enif_get_int(env, tuple_threads[2], &threads_z);

  // Creating NDRange objects for blocks and threads
  // The global range is the total number of threads (work-items) in each dimension
  // The local range is the size of each block (work-group) in every dimension
  // So we need to calculate the global range in each dimension
  cl::NDRange global_range(blocks_x * threads_x, blocks_y * threads_y, blocks_z * threads_z);
  cl::NDRange local_range(threads_x, threads_y, threads_z);

  if (debug_logs)
  {
    std::cout << "[C++ GPU NIF] Kernel '" << kernel_name << "' will be executed with a global range of "
              << global_range[0] << "x" << global_range[1] << "x" << global_range[2]
              << " and a local range of " << local_range[0] << "x" << local_range[1]
              << "x" << local_range[2] << "." << std::endl;
  }

  // Getting the number of arguments given to the kernel
  int size_args;

  if (!enif_get_int(env, argv[4], &size_args))
  {
    return enif_make_badarg(env);
  }

  // Collecting the arguments and their types
  ERL_NIF_TERM list_args_types;
  ERL_NIF_TERM head_args_types;
  ERL_NIF_TERM tail_args_types;

  ERL_NIF_TERM list_args;
  ERL_NIF_TERM head_args;
  ERL_NIF_TERM tail_args;

  list_args_types = argv[5];
  list_args = argv[6];

  for (int i = 0; i < size_args; i++)
  {
    ERL_NIF_TERM arg;
    char arg_type_name[1024];
    unsigned int arg_type_name_lenght;

    // Get first element of the list of types
    if (!enif_get_list_cell(env, list_args_types, &head_args_types, &tail_args_types))
    {
      std::cerr << "[ERROR] Error getting list cell for kernel argument types." << std::endl;
      return enif_make_badarg(env);
    }

    // Get length of the type name
    if (!enif_get_list_length(env, head_args_types, &arg_type_name_lenght))
    {
      std::cerr << "[ERROR] Error getting type name length for kernel argument types." << std::endl;
      return enif_make_badarg(env);
    }

    // Get the type name as a string
    enif_get_string(env, head_args_types, arg_type_name, arg_type_name_lenght + 1, ERL_NIF_LATIN1);

    // Get first element of the list of arguments
    // This is the actual argument that will be passed to the kernel
    if (!enif_get_list_cell(env, list_args, &head_args, &tail_args))
    {
      std::cerr << "[ERROR] Error getting list cell for kernel argument " << i << "." << std::endl;
      return enif_make_badarg(env);
    }
    arg = head_args;

    // Now that we have the argument and its type name
    // We can convert the argument to the appropriate type and set it in the kernel object
    if (strcmp(arg_type_name, "int") == 0)
    {
      int iarg;
      if (!enif_get_int(env, arg, &iarg))
      {
        std::cerr << "[ERROR] Error getting integer argument for kernel." << std::endl;
        return enif_make_badarg(env);
      }

      kernel.setArg(i, iarg);
    }
    else if (strcmp(arg_type_name, "float") == 0)
    {
      double darg;
      if (!enif_get_double(env, arg, &darg))
      {
        std::cerr << "[ERROR] Error getting float argument for kernel." << std::endl;
        return enif_make_badarg(env);
      }

      float farg = static_cast<float>(darg);
      kernel.setArg(i, farg);
    }
    else if (strcmp(arg_type_name, "double") == 0)
    {
      double darg;
      if (!enif_get_double(env, arg, &darg))
      {
        std::cerr << "[ERROR] Error getting double argument for kernel." << std::endl;
        return enif_make_badarg(env);
      }

      kernel.setArg(i, darg);
    }
    else if (
        strcmp(arg_type_name, "tint") == 0 ||
        strcmp(arg_type_name, "tfloat") == 0 ||
        strcmp(arg_type_name, "tdouble") == 0)
    {
      cl::Buffer *array_res;
      if (!enif_get_resource(env, arg, ARRAY_TYPE, (void **)&array_res))
      {
        std::cerr << "[ERROR] Error getting buffer (array) resource for kernel." << std::endl;
        return enif_make_badarg(env);
      }

      kernel.setArg(i, *array_res);
    }
    else
    {
      std::cerr << "[ERROR] Unknown argument type '" << arg_type_name << "' for kernel." << std::endl;
      return enif_make_badarg(env);
    }

    list_args_types = tail_args_types;
    list_args = tail_args;
  }

  // Now we can execute the kernel
  try
  {
    open_cl->executeKernel(kernel, global_range, local_range, device_type);

    if (debug_logs)
    {
      std::cout << "[C++ GPU NIF] Kernel '" << kernel_name << "' executed successfully." << std::endl;
    }
  }
  catch (const std::exception &e)
  {
    return enif_raise_exception(env, enif_make_string(env, e.what(), ERL_NIF_LATIN1));
  }

  return enif_make_int(env, 0);
}

// This function retrieves the OpenCL array from the device (GPU/CPU) and returns it to the host as an Erlang term.
static ERL_NIF_TERM get_device_array_nif(ErlNifEnv *env, int /* argc */, const ERL_NIF_TERM argv[])
{
  cl::Buffer *device_array = nullptr;

  // Get the Buffer resource to copy data from
  if (!enif_get_resource(env, argv[0], ARRAY_TYPE, (void **)&device_array))
  {
    return enif_make_badarg(env);
  }

  // Get number of rows and columns
  int nrow, ncol;

  if (!enif_get_int(env, argv[1], &nrow))
  {
    return enif_make_badarg(env);
  }

  if (!enif_get_int(env, argv[2], &ncol))
  {
    return enif_make_badarg(env);
  }

  // Get type name
  ERL_NIF_TERM e_type_name = argv[3];
  unsigned int size_type_name;
  char type_name[1024];

  if (!enif_get_list_length(env, e_type_name, &size_type_name))
  {
    return enif_make_badarg(env);
  }

  enif_get_string(env, e_type_name, type_name, size_type_name + 1, ERL_NIF_LATIN1);

  // Calculating the size of the result
  size_t data_size;

  if (strcmp(type_name, "float") == 0)
  {
    data_size = sizeof(float) * nrow * ncol;
  }
  else if (strcmp(type_name, "int") == 0)
  {
    data_size = sizeof(int) * nrow * ncol;
  }
  else if (strcmp(type_name, "double") == 0)
  {
    data_size = sizeof(double) * nrow * ncol;
  }
  else // Unknown type
  {
    char message[200];
    strcpy(message, "[ERROR] (get_device_array_nif) copying data from device to host: unknown type ");
    strcat(message, type_name);
    return enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }

  // Get device type (GPU or CPU)
  ERL_NIF_TERM e_device_type = argv[4];
  OCLInterface::DeviceType device_type = get_device_type(e_device_type, env);

  // Allocate memory in the host to store the result
  // According to Erlang's docs, for LARGE binaries, it is better to use enif_alloc_binary.
  ErlNifBinary host_bin;

  if (!enif_alloc_binary(data_size, &host_bin))
  {
    char message[200];
    strcpy(message, "[ERROR] (get_device_array_nif) failed to allocate binary of size ");
    strcat(message, std::to_string(data_size).c_str());
    return enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }

  // Getting pointer to the allocated binary data in the host
  void *result_data_pointer = (void *)host_bin.data;

  // Copying data from device to host
  try
  {
    open_cl->readBuffer(*device_array, result_data_pointer, data_size, device_type);

    if (debug_logs)
    {
      std::cout << "[C++ GPU NIF] Data copied from device to host successfully." << std::endl;
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << "[ERROR] (get_device_array_nif) copying data from device to host: " << e.what() << std::endl;
    return enif_raise_exception(env, enif_make_string(env, e.what(), ERL_NIF_LATIN1));
  }

  // Creating the Erlang binary term to return (passing ownership to the BEAM)
  ERL_NIF_TERM erl_term_result = enif_make_binary(env, &host_bin);
  return erl_term_result;
}

// This function creates a new GPU array with the specified number of rows, columns, and type.
// It allocates memory on the GPU and copies data to it from the host array provided.
static ERL_NIF_TERM new_array_from_nx_nif(ErlNifEnv *env, int /* argc */, const ERL_NIF_TERM argv[])
{
  // Get the host array binary
  ErlNifBinary host_array_el;

  if (!enif_inspect_binary(env, argv[0], &host_array_el))
  {
    return enif_make_badarg(env);
  }

  // Get number of rows and columns
  int nrow, ncol;

  if (!enif_get_int(env, argv[1], &nrow))
  {
    return enif_make_badarg(env);
  }

  if (!enif_get_int(env, argv[2], &ncol))
  {
    return enif_make_badarg(env);
  }

  // Get type name
  ERL_NIF_TERM e_type_name = argv[3];
  unsigned int size_type_name;
  if (!enif_get_list_length(env, e_type_name, &size_type_name))
  {
    return enif_make_badarg(env);
  }

  char type_name[1024];
  enif_get_string(env, e_type_name, type_name, size_type_name + 1, ERL_NIF_LATIN1);

  // Calculates the size of the data to be copied to the GPU/CPU
  size_t data_size;

  if (strcmp(type_name, "float") == 0)
  {
    data_size = sizeof(float) * ncol * nrow;
  }
  else if (strcmp(type_name, "int") == 0)
  {
    data_size = sizeof(int) * ncol * nrow;
  }
  else if (strcmp(type_name, "double") == 0)
  {
    data_size = sizeof(double) * ncol * nrow;
  }
  else // Unknown type
  {
    char message[200];
    strcpy(message, "[ERROR] (new_array_from_nx_nif): unknown type ");
    strcat(message, type_name);
    return enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }

  // Get device type (GPU or CPU)
  ERL_NIF_TERM e_device_type = argv[4];
  OCLInterface::DeviceType device_type = get_device_type(e_device_type, env);

  try
  {
    // Allocate an empty buffer on the device for the array
    cl::Buffer dev_array = open_cl->createBuffer(data_size, CL_MEM_READ_WRITE, device_type);
    // Copy data from host to device (H2D copy)
    open_cl->writeBuffer(dev_array, (void *)host_array_el.data, data_size, device_type);

    // Allocate an Erlang resource to hold the C++ buffer object
    cl::Buffer *device_res = (cl::Buffer *)enif_alloc_resource(ARRAY_TYPE, sizeof(cl::Buffer));

    // Using placement new to construct the cl::Buffer in the resource's memory
    new (device_res) cl::Buffer(dev_array);

    ERL_NIF_TERM return_term = enif_make_resource(env, device_res);

    // Release the C++ handle to the resource, letting the BEAM manage its lifetime
    enif_release_resource(device_res);

    if (debug_logs)
    {
      std::cout << "[C++ GPU NIF] New device array created with " << nrow << " rows, " << ncol << " columns, and type " << type_name << std::endl;
      std::cout << "[C++ GPU NIF] Data copied from host to device successfully." << std::endl;
    }

    return return_term;
  }
  catch (const std::exception &e)
  {
    return enif_raise_exception(env, enif_make_string(env, e.what(), ERL_NIF_LATIN1));
  }
}

// Creates a new empty GPU/CPU array with the specified number of rows, columns, and type
static ERL_NIF_TERM new_empty_array_nif(ErlNifEnv *env, int /* argc */, const ERL_NIF_TERM argv[])
{
  // Get number of rows and columns
  int nrow, ncol;

  if (!enif_get_int(env, argv[0], &nrow))
  {
    return enif_make_badarg(env);
  }

  if (!enif_get_int(env, argv[1], &ncol))
  {
    return enif_make_badarg(env);
  }

  // Get type name
  // The type name is a list of characters, so we need to get its length first
  ERL_NIF_TERM e_type_name = argv[2];
  unsigned int size_type_name;
  if (!enif_get_list_length(env, e_type_name, &size_type_name))
  {
    return enif_make_badarg(env);
  }

  // Create a buffer to hold the type name
  // We add 1 to the size to accommodate the null terminator
  // Note: ERL_NIF_LATIN1 is used for encoding the string
  char type_name[1024];
  enif_get_string(env, e_type_name, type_name, size_type_name + 1, ERL_NIF_LATIN1);

  // From here on, we will use the type name to determine the data size and allocate memory accordingly
  size_t data_size;

  if (strcmp(type_name, "float") == 0)
  {
    data_size = sizeof(float) * nrow * ncol;
  }
  else if (strcmp(type_name, "int") == 0)
  {
    data_size = sizeof(int) * nrow * ncol;
  }
  else if (strcmp(type_name, "double") == 0)
  {
    data_size = sizeof(double) * nrow * ncol;
  }
  else // Unknown type
  {
    char message[200];
    strcpy(message, "[ERROR] new_empty_array_nif: unknown type: ");
    strcat(message, type_name);
    return enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }

  // Get device type (GPU or CPU)
  ERL_NIF_TERM e_device_type = argv[3];
  OCLInterface::DeviceType device_type = get_device_type(e_device_type, env);

  try
  {
    // Allocate memory on the GPU/CPU
    cl::Buffer dev_array = open_cl->createBuffer(data_size, CL_MEM_READ_WRITE, device_type);

    // Allocate an Erlang resource to hold the C++ buffer object
    cl::Buffer *device_res = (cl::Buffer *)enif_alloc_resource(ARRAY_TYPE, sizeof(cl::Buffer));

    // Using placement new to construct the cl::Buffer in the resource's memory
    new (device_res) cl::Buffer(dev_array);

    ERL_NIF_TERM return_term = enif_make_resource(env, device_res);

    // Release the C++ handle to the resource, letting the BEAM manage its lifetime
    enif_release_resource(device_res);

    if (debug_logs)
    {
      std::cout << "[C++ GPU NIF] New device array created with " << nrow << " rows, " << ncol << " columns, and type " << type_name << std::endl;
    }

    return return_term;
  }
  catch (const std::exception &e)
  {
    return enif_raise_exception(env, enif_make_string(env, e.what(), ERL_NIF_LATIN1));
  }
}

// This function synchronizes the OpenCL command queue, ensuring that all previously enqueued commands have completed.
static ERL_NIF_TERM synchronize_nif(ErlNifEnv *env, int /* argc */, const ERL_NIF_TERM /* argv */[])
{
  open_cl->synchronize();

  if (debug_logs)
  {
    std::cout << "[C++ GPU NIF] OpenCL command queue synchronized successfully." << std::endl;
  }

  return enif_make_int(env, 0);
}

// This function sets the debug logs flag for the NIFs.
static ERL_NIF_TERM set_debug_logs_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  if (argc != 1)
  {
    std::cerr << "[ERROR] Invalid number of arguments for set_debug_logs_nif." << std::endl;
    return enif_make_badarg(env);
  }

  if (!enif_is_atom(env, argv[0]))
  {
    std::cerr << "[ERROR] Argument for set_debug_logs_nif must be either 'true' or 'false' atoms." << std::endl;
    return enif_make_badarg(env);
  }

  ERL_NIF_TERM true_atom = enif_make_atom(env, "true");

  debug_logs = (enif_compare(argv[0], true_atom) == 0);
  open_cl->setDebugLogs(debug_logs);

  return enif_make_int(env, 0);
}

// This function checks if the current GPU supports double precision floating points
// and int64 base atomics extensions for CAS
static ERL_NIF_TERM double_supported_nif(ErlNifEnv *env, int /* argc */, const ERL_NIF_TERM /* argv */[])
{
  if (fp64_supported && int64_base_atomics_supported)
  {
    return enif_make_atom(env, "true");
  }
  else
  {
    return enif_make_atom(env, "false");
  }
}

// This function creates a new aligned SVM region on the CPU and copies data to it from the Elixir list provided.
// It returns an Erlang Resource Binary that points directly to the aligned SVM allocated by OpenCL.
// Parameters:
// 1 - A flat list (without nested lists) containing the data to be copied to the SVM memory.
// 2 - The length of the list (number of elements).
// 3 - The type of the data (e.g., "float", "int", "double") as an Elixir charlist (a list of characters).
static ERL_NIF_TERM new_aligned_nx_from_list_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  if (argc != 3)
  {
    return enif_make_badarg(env);
  }

  ERL_NIF_TERM list = argv[0];
  ERL_NIF_TERM list_length_term = argv[1];
  ERL_NIF_TERM type_name_term = argv[2];

  // Check if the first argument is a list
  if (!enif_is_list(env, list))
  {
    return enif_make_badarg(env);
  }

  // Get the length of the list (number of elements)
  uint32_t list_length;
  if (!enif_get_uint(env, list_length_term, &list_length))
  {
    return enif_make_badarg(env);
  }

  // Get the type of the array to be created (e.g., "float", "int", "double")
  uint32_t size_type_name;
  if (!enif_get_list_length(env, type_name_term, &size_type_name))
  {
    return enif_make_badarg(env);
  }

  std::string type_name(size_type_name, '\0');
  if (!enif_get_string(env, type_name_term, type_name.data(), size_type_name + 1, ERL_NIF_LATIN1))
  {
    return enif_make_badarg(env);
  }

  // Calculate the size in bytes for the SVM array to hold the list data, based on the
  // data type and the number of elements
  size_t array_size_bytes;

  if (type_name == "float")
  {
    array_size_bytes = sizeof(float) * list_length;
  }
  else if (type_name == "int")
  {
    array_size_bytes = sizeof(int) * list_length;
  }
  else if (type_name == "double")
  {
    array_size_bytes = sizeof(double) * list_length;
  }
  else // Unknown type
  {
    std::string message = "[ERROR] new_aligned_nx_from_list_nif: unknown type: " + type_name;
    return enif_raise_exception(env, enif_make_string(env, message.c_str(), ERL_NIF_LATIN1));
  }

  if (debug_logs)
  {
    std::cout << "[C++ GPU NIF] Creating new aligned SVM array from list with " << list_length << " elements of type <" << type_name << ">." << std::endl;
    std::cout << "[C++ GPU NIF] Calculated array size in bytes: " << array_size_bytes << " bytes." << std::endl;
  }

  try
  {
    // Allocate Shared Virtual Memory (SVM) that is aligned and can be accessed by any CPU core.
    // By default, all OCL-PolyHok's SVMs will be allocated in the CPU for CPU parallel computations.
    // We do not intend to use SVM for GPU computations, because we are already using cl::Buffer for this.
    void *aligned_mem = open_cl->createSVM(array_size_bytes, OCLInterface::DeviceType::CPU);

    // Now we can populate the allocated SVM array with the list data

    ERL_NIF_TERM head, tail;
    size_t index = 0;

    while (enif_get_list_cell(env, list, &head, &tail))
    {
      if (type_name == "float")
      {
        double value;
        if (!enif_get_double(env, head, &value))
        {
          return enif_make_badarg(env);
        }
        static_cast<float *>(aligned_mem)[index] = static_cast<float>(value);
      }
      else if (type_name == "int")
      {
        int value;
        if (!enif_get_int(env, head, &value))
        {
          return enif_make_badarg(env);
        }
        static_cast<int *>(aligned_mem)[index] = value;
      }
      else if (type_name == "double")
      {
        double value;
        if (!enif_get_double(env, head, &value))
        {
          return enif_make_badarg(env);
        }
        static_cast<double *>(aligned_mem)[index] = value;
      }

      list = tail;
      index += 1;
    }

    // Allocate an Erlang resource to hold the pointer to the SVM memory.
    void **svm_res = (void **)enif_alloc_resource(CPU_SVM_TYPE, sizeof(void *));

    // Store the pointer to the SVM memory in the resource
    *svm_res = aligned_mem;

    // Creating an Erlang Resource Binary to point directly to the SVM memory
    // An Erlang Resource Binary will behave like a normal binary in Elixir, but its data pointer will point
    // to the aligned SVM memory OpenCL allocated for us. And when the BEAM garbage collects the Resource Binary,
    // it will call the cpu_svm_destructor we defined, which will free the SVM memory correctly using OpenCL's API.
    ERL_NIF_TERM resource_bin = enif_make_resource_binary(env, (void *)svm_res, aligned_mem, array_size_bytes);

    // Release the resource handle letting the BEAM manage its lifetime
    enif_release_resource(svm_res);

    // Returning our Resource Binary
    return resource_bin;
  }
  catch (const std::exception &e)
  {
    return enif_raise_exception(env, enif_make_string(env, e.what(), ERL_NIF_LATIN1));
  }
}

static ERL_NIF_TERM is_nx_aligned_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  if (argc != 1)
  {
    return enif_make_badarg(env);
  }

  // Get the binary
  ErlNifBinary bin;

  if (!enif_inspect_binary(env, argv[0], &bin))
  {
    return enif_make_badarg(env);
  }

  // Check if it is aligned
  cl_uint alignment_bytes = open_cl->getCPUAlignmentBytes();
  uintptr_t ptr_value = reinterpret_cast<uintptr_t>(bin.data);

  // Print the address and the alignment for debugging
  if (debug_logs)
  {
    std::cout << "[C++ GPU NIF] Checking alignment of pointer at address " << static_cast<void *>(bin.data) << std::endl;
    std::cout << "[C++ GPU NIF] - Alignment requirement: " << alignment_bytes << " bytes." << std::endl;
  }

  if (ptr_value % alignment_bytes == 0)
  {
    if (debug_logs)
    {
      std::cout << "[C++ GPU NIF] The pointer IS aligned." << std::endl;
    }
    return enif_make_atom(env, "true");
  }
  else
  {
    if (debug_logs)
    {
      std::cout << "[C++ GPU NIF] The pointer is NOT aligned." << std::endl;
    }
    return enif_make_atom(env, "false");
  }
}

// The ErlNifFunc struct in the Erlang headers expects the arguments in this exact order: name, arity, fptr, flags.
// I'm using this syntax because the designated initializer syntax was not adopted in C++ until C++20, and this project
// uses C++17. Therefore, I'm using the traditional aggregate initialization syntax, which requires the fields to be in
// the order they are declared in the struct.
static ErlNifFunc nif_funcs[] = {
    {"jit_compile_and_launch_nif", 8, jit_compile_and_launch_nif, 0},
    {"new_empty_array_nif", 4, new_empty_array_nif, 0},
    {"get_device_array_nif", 5, get_device_array_nif, 0},
    {"new_array_from_nx_nif", 5, new_array_from_nx_nif, 0},
    {"synchronize_nif", 0, synchronize_nif, 0},
    {"set_debug_logs_nif", 1, set_debug_logs_nif, 0},
    {"double_supported_nif", 0, double_supported_nif, 0},
    {"new_aligned_nx_from_list_nif", 3, new_aligned_nx_from_list_nif, 0},
    {"is_nx_aligned_nif", 1, is_nx_aligned_nif, 0}};

ERL_NIF_INIT(Elixir.OCLPolyHok, nif_funcs, &load, NULL, NULL, &unload)
