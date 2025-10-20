#include "ocl_interface/OCLInterface.hpp"

#include <erl_nif.h>

#include <iostream>

#include <cmath>
#include <cstdint>
#include <dlfcn.h>
#include <assert.h>

bool debug_logs = false;

void dev_array_destructor(ErlNifEnv * /* env */, void *res)
{
  cl::Buffer *dev_array = (cl::Buffer *)res;

  // Explicitly call the destructor for the cl::Buffer object
  // without deallocating the resource memory itself. This is
  // Erlang's garbage collector responsibility.
  dev_array->~Buffer();

  if (debug_logs)
  {
    std::cout << "[C++ GPU NIF] Device array resource destroyed." << std::endl;
  }
}

OCLInterface *open_cl = nullptr;

ErlNifResourceType *ARRAY_TYPE;

void init_ocl(ErlNifEnv *env)
{
  if (open_cl != nullptr)
    return; // Already initialized

  open_cl = new OCLInterface(debug_logs);

  try
  {
    // Selecting default platform
    open_cl->selectDefaultPlatform();

    // Selecting default GPU device
    open_cl->selectDefaultDevice(CL_DEVICE_TYPE_GPU);
  }
  catch (const std::exception &e)
  {
    std::cerr << "[ERROR] Failed to initialize OpenCL interface: " << e.what() << std::endl;
    enif_raise_exception(env, enif_make_string(env, e.what(), ERL_NIF_LATIN1));

    delete open_cl;
  }
}

static int
load(ErlNifEnv *env, void ** /* priv_data */, ERL_NIF_TERM /* load_info */)
{
  ARRAY_TYPE = enif_open_resource_type(
      env,
      NULL,
      "gpu_ref",
      dev_array_destructor,
      ERL_NIF_RT_CREATE,
      NULL);

  // Initialize OpenCL
  init_ocl(env);

  return 0;
}

static void
unload(ErlNifEnv * /* env */, void * /* priv_data */)
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

// This function compiles the given kernel code and launches it with the specified blocks and threads.
static ERL_NIF_TERM jit_compile_and_launch_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  // Check argc
  if (argc != 7)
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

  char kernel_name[size_name + 1];
  enif_get_string(env, e_name, kernel_name, size_name + 1, ERL_NIF_LATIN1);

  // Get kernel code to compile
  ERL_NIF_TERM e_code = argv[1];
  unsigned int size_code;
  if (!enif_get_list_length(env, e_code, &size_code))
  {
    return enif_make_badarg(env);
  }

  char code[size_code + 1];
  enif_get_string(env, e_code, code, size_code + 1, ERL_NIF_LATIN1);

  // Creating program and kernel objects
  cl::Program program = open_cl->createProgram(code);
  cl::Kernel kernel = open_cl->createKernel(program, kernel_name);

  // Getting blocks and threads tuples pointers
  const ERL_NIF_TERM *tuple_blocks, *tuple_threads;
  int arity;

  if (!enif_get_tuple(env, argv[2], &arity, &tuple_blocks))
  {
    std::cerr << "[ERROR] The given blocks argument is not a tuple." << std::endl;
    return enif_make_badarg(env);
  }

  if (!enif_get_tuple(env, argv[3], &arity, &tuple_threads))
  {
    std::cerr << "[ERROR] The given threads argument is not a tuple." << std::endl;
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
  cl::NDRange global_range(blocks_x * threads_x,
                           blocks_y * threads_y,
                           blocks_z * threads_z);
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
    open_cl->executeKernel(kernel, global_range, local_range);

    if (debug_logs)
    {
      std::cout << "[C++ GPU NIF] Kernel '" << kernel_name << "' executed successfully." << std::endl;
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << "[ERROR] Error executing kernel '" << kernel_name << "': " << e.what() << std::endl;
    return enif_raise_exception(env, enif_make_string(env, e.what(), ERL_NIF_LATIN1));
  }

  return enif_make_int(env, 0);
}

// This function retrieves the OpenCL array from the device (GPU) and returns it to the host as an Erlang term.
static ERL_NIF_TERM get_gpu_array_nif(ErlNifEnv *env, int /* argc */, const ERL_NIF_TERM argv[])
{
  int nrow, ncol;
  size_t data_size;
  char type_name[1024];
  ERL_NIF_TERM result;

  cl::Buffer dev_array;
  cl::Buffer *array_res = nullptr;

  // Get the Buffer resource to copy data from
  if (!enif_get_resource(env, argv[0], ARRAY_TYPE, (void **)&array_res))
  {
    return enif_make_badarg(env);
  }

  dev_array = *array_res;

  // Get number of rows
  if (!enif_get_int(env, argv[1], &nrow))
  {
    return enif_make_badarg(env);
  }

  // Get number of columns
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

  enif_get_string(env, e_type_name, type_name, size_type_name + 1, ERL_NIF_LATIN1);

  // Calculating the size of the result
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
    strcpy(message, "[ERROR] (get_gpu_array_nif) copying data from device to host: unknown type ");
    strcat(message, type_name);
    return enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }

  // Allocate memory in host for the result

  // -> FIXME: According to Erlang's docs, the function enif_make_new_binary is used to
  // create SMALL binaries in the BEAM heap. For LARGE binaries, it is recommended to use
  // enif_alloc_binary and enif_release_binary.
  void *host_result_data = (void *)enif_make_new_binary(env, data_size, &result);

  // Copying data from device to host
  try
  {
    open_cl->readBuffer(dev_array, host_result_data, data_size);

    if (debug_logs)
    {
      std::cout << "[C++ GPU NIF] Data copied from device to host successfully." << std::endl;
    }
  }
  catch (const std::exception &e)
  {
    std::cerr << "[ERROR] (get_gpu_array_nif) copying data from device to host: " << e.what() << std::endl;
    return enif_raise_exception(env, enif_make_string(env, e.what(), ERL_NIF_LATIN1));
  }

  return result;
}

// This function creates a new GPU array with the specified number of rows, columns, and type.
// It allocates memory on the GPU and copies the data from the host array passed to the device.
static ERL_NIF_TERM create_gpu_array_nx_nif(ErlNifEnv *env, int /* argc */, const ERL_NIF_TERM argv[])
{
  int nrow, ncol;
  size_t data_size;
  ErlNifBinary host_array_el;

  // Get the host array binary
  if (!enif_inspect_binary(env, argv[0], &host_array_el))
    return enif_make_badarg(env);

  // Get number of rows
  if (!enif_get_int(env, argv[1], &nrow))
  {
    return enif_make_badarg(env);
  }

  // Get number of columns
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

  // Calculates the size of the data to be copied to the GPU
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
    strcpy(message, "[ERROR] (create_gpu_array_nx_nif): unknown type ");
    strcat(message, type_name);
    return enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }

  try
  {
    // Allocate memory on the GPU and copy the data from the host array
    // Note: The host_array_el.data is a pointer to the data in the Erlang binary
    cl::Buffer dev_array = open_cl->createBuffer(data_size, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (void *)host_array_el.data);

    // Allocate an Erlang resource to hold the C++ buffer object
    cl::Buffer *gpu_res = (cl::Buffer *)enif_alloc_resource(ARRAY_TYPE, sizeof(cl::Buffer));

    // Using placement new to construct the cl::Buffer in the resource's memory
    new (gpu_res) cl::Buffer(dev_array);

    ERL_NIF_TERM return_term = enif_make_resource(env, gpu_res);

    // Release the C++ handle to the resource, letting the BEAM manage its lifetime
    enif_release_resource(gpu_res);

    if (debug_logs)
    {
      std::cout << "[C++ GPU NIF] New GPU array created with " << nrow << " rows, " << ncol << " columns, and type " << type_name << std::endl;
      std::cout << "[C++ GPU NIF] Data copied from host to device successfully." << std::endl;
    }

    return return_term;
  }
  catch (const std::exception &e)
  {
    std::cerr << "[ERROR] (create_gpu_array_nx_nif) creating GPU buffer: " << e.what() << std::endl;
    return enif_raise_exception(env, enif_make_string(env, e.what(), ERL_NIF_LATIN1));
  }
}

// Creates a new empty GPU array with the specified number of rows, columns, and type
static ERL_NIF_TERM new_gpu_array_nif(ErlNifEnv *env, int /* argc */, const ERL_NIF_TERM argv[])
{
  int nrow, ncol;
  size_t data_size;

  // Get number of rows
  if (!enif_get_int(env, argv[0], &nrow))
  {
    return enif_make_badarg(env);
  }

  // Get number of columns
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
  if (strcmp(type_name, "float") == 0)
  {
    data_size = nrow * ncol * sizeof(float);
  }
  else if (strcmp(type_name, "int") == 0)
  {
    data_size = nrow * ncol * sizeof(int);
  }
  else if (strcmp(type_name, "double") == 0)
  {
    data_size = nrow * ncol * sizeof(double);
  }
  else // Unknown type
  {
    char message[200];
    strcpy(message, "[ERROR] new_gpu_array_nif: unknown type: ");
    strcat(message, type_name);
    return enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }

  try
  {
    // Allocate memory on the GPU
    cl::Buffer dev_array = open_cl->createBuffer(data_size, CL_MEM_READ_WRITE);

    // Allocate an Erlang resource to hold the C++ buffer object
    cl::Buffer *gpu_res = (cl::Buffer *)enif_alloc_resource(ARRAY_TYPE, sizeof(cl::Buffer));

    // Using placement new to construct the cl::Buffer in the resource's memory
    new (gpu_res) cl::Buffer(dev_array);

    ERL_NIF_TERM return_term = enif_make_resource(env, gpu_res);

    // Release the C++ handle to the resource, letting the BEAM manage its lifetime
    enif_release_resource(gpu_res);

    if (debug_logs)
    {
      std::cout << "[C++ GPU NIF] New GPU array created with " << nrow << " rows, " << ncol << " columns, and type " << type_name << std::endl;
    }

    return return_term;
  }
  catch (const std::exception &e)
  {
    std::cerr << "[ERROR] Failed to create GPU buffer: " << e.what() << std::endl;
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
static ERL_NIF_TERM set_debug_logs(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{
  if (argc != 1)
  {
    std::cerr << "[ERROR] Invalid number of arguments for set_debug_logs." << std::endl;
    return enif_make_badarg(env);
  }

  if (!enif_is_atom(env, argv[0]))
  {
    std::cerr << "[ERROR] Argument for set_debug_logs must be either 'true' or 'false' atoms." << std::endl;
    return enif_make_badarg(env);
  }

  ERL_NIF_TERM true_atom = enif_make_atom(env, "true");

  debug_logs = (enif_compare(argv[0], true_atom) == 0);
  open_cl->setDebugLogs(debug_logs);

  return enif_make_int(env, 0);
}

static ErlNifFunc nif_funcs[] = {
    {.name = "jit_compile_and_launch_nif", .arity = 7, .fptr = jit_compile_and_launch_nif, .flags = 0},
    {.name = "new_gpu_array_nif", .arity = 3, .fptr = new_gpu_array_nif, .flags = 0},
    {.name = "get_gpu_array_nif", .arity = 4, .fptr = get_gpu_array_nif, .flags = 0},
    {.name = "create_gpu_array_nx_nif", .arity = 4, .fptr = create_gpu_array_nx_nif, .flags = 0},
    {.name = "synchronize_nif", .arity = 0, .fptr = synchronize_nif, .flags = 0},
    {.name = "set_debug_logs", .arity = 1, .fptr = set_debug_logs, .flags = 0}};

ERL_NIF_INIT(Elixir.OCLPolyHok, nif_funcs, &load, NULL, NULL, &unload)
