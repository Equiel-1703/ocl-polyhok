#include "ocl_interface/OCLInterface.hpp"

#include <erl_nif.h>

#include <iostream>

#include <stdio.h> // Refactor code to use iostream and remove this later

#include <cmath>
#include <cstdint>
#include <dlfcn.h>
#include <assert.h>

void dev_array_destructor(ErlNifEnv * /* env */, void *res)
{
  cl::Buffer *dev_array = (cl::Buffer *)res;
  delete dev_array; // cl::Buffer destructor will properly release OpenCL resources
}

void dev_pinned_array_destructor(ErlNifEnv * /* env */, void *res)
{
  cl::Buffer *dev_array = (cl::Buffer *)res;
  delete dev_array; // cl::Buffer destructor will properly release OpenCL resources
}

OCLInterface *open_cl = nullptr;

ErlNifResourceType *KERNEL_TYPE;
ErlNifResourceType *ARRAY_TYPE;
ErlNifResourceType *PINNED_ARRAY;

void init_ocl(ErlNifEnv *env)
{
  if (open_cl != nullptr)
    return; // Already initialized

  open_cl = new OCLInterface();

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
  KERNEL_TYPE =
      enif_open_resource_type(env, NULL, "kernel", NULL, ERL_NIF_RT_CREATE, NULL);
  ARRAY_TYPE =
      enif_open_resource_type(env, NULL, "gpu_ref", dev_array_destructor, ERL_NIF_RT_CREATE, NULL);
  PINNED_ARRAY =
      enif_open_resource_type(env, NULL, "pinned_array", dev_pinned_array_destructor, ERL_NIF_RT_CREATE, NULL);

  std::cout << "GPU NIFs loaded successfully." << std::endl;

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

  std::cout << "GPU NIFs unloaded successfully." << std::endl;
}

// The next 3 functions are used to compile and launch the CUDA kernels using NVRTC (NVIDIA Runtime Compilation).

// Temporary commenting out the NVRTC related code for testing
/*

void fail_nvrtc(ErlNifEnv *env, nvrtcResult result, const char *obs)
{

  char message[1000];

  strcpy(message, "Error  NVRTC ");
  strcat(message, obs);
  strcat(message, ": ");
  strcat(message, nvrtcGetErrorString(result));
  enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
}

char *compile_to_ptx(ErlNifEnv *env, char *program_source)
{
  nvrtcResult rv;

  // create nvrtc program
  nvrtcProgram prog;
  rv = nvrtcCreateProgram(
      &prog,
      program_source,
      nullptr,
      0,
      nullptr,
      nullptr);
  if (rv != NVRTC_SUCCESS)
    fail_nvrtc(env, rv, "nvrtcCreateProgram");

  int size_options = 10;
  const char *options[10] = {
      "--include-path=/lib/erlang/usr/include/",
      "--include-path=/usr/include/",
      "--include-path=/usr/lib/",
      "--include-path=/usr/include/x86_64-linux-gnu/",
      "--include-path=/usr/include/c++/11",
      "--include-path=/usr/include/x86_64-linux-gnu/c++/11",
      "--include-path=/usr/include/c++/11/backward",
      "--include-path=/usr/lib/gcc/x86_64-linux-gnu/11/include",
      "--include-path=/usr/include/i386-linux-gnu/",
      "--include-path=/usr/local/include"};

  rv = nvrtcCompileProgram(prog, size_options, options);
  if (rv != NVRTC_SUCCESS)
  {
    nvrtcResult erro_g = rv;
    size_t log_size;
    rv = nvrtcGetProgramLogSize(prog, &log_size);
    if (rv != NVRTC_SUCCESS)
      fail_nvrtc(env, rv, "nvrtcGetProgramLogSize");
    // auto log = std::make_unique<char[]>(log_size);
    char log[log_size];
    rv = nvrtcGetProgramLog(prog, log);
    if (rv != NVRTC_SUCCESS)
      fail_nvrtc(env, rv, "nvrtcGetProgramLog");
    assert(log[log_size - 1] == '\0');

    printf("Compilation error; log: %s\n", log);

    fail_nvrtc(env, erro_g, "nvrtcCompileProgram");
    // return enif_make_int(env, 0);
  }
  // get ptx code
  size_t ptx_size;
  rv = nvrtcGetPTXSize(prog, &ptx_size);
  if (rv != NVRTC_SUCCESS)
    fail_nvrtc(env, rv, "nvrtcGetPTXSize");
  char *ptx_source = new char[ptx_size];
  nvrtcGetPTX(prog, ptx_source);

  if (rv != NVRTC_SUCCESS)
    fail_nvrtc(env, rv, "nvrtcGetPTX");
  assert(ptx_source[ptx_size - 1] == '\0');

  nvrtcDestroyProgram(&prog);

  return ptx_source;
}

static ERL_NIF_TERM jit_compile_and_launch_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[])
{

  ERL_NIF_TERM list_types;
  ERL_NIF_TERM head_types;
  ERL_NIF_TERM tail_types;

  ERL_NIF_TERM list_args;
  ERL_NIF_TERM head_args;
  ERL_NIF_TERM tail_args;

  const ERL_NIF_TERM *tuple_blocks;
  const ERL_NIF_TERM *tuple_threads;
  int arity;

  CUmodule module;
  CUfunction function;
  CUresult err;

  /// START COLLECTING TIME

  // float time;
  // cudaEvent_t start, stop;
  //  cudaEventCreate(&start) ;
  // cudaEventCreate(&stop) ;
  // cudaEventRecord(start, 0) ;

  /////////// get name kernel

  ERL_NIF_TERM e_name = argv[0];
  unsigned int size_name;
  if (!enif_get_list_length(env, e_name, &size_name))
  {
    return enif_make_badarg(env);
  }

  char kernel_name[size_name + 1];

  enif_get_string(env, e_name, kernel_name, size_name + 1, ERL_NIF_LATIN1);

  ///////////// get code

  ERL_NIF_TERM e_code = argv[1];
  unsigned int size_code;
  if (!enif_get_list_length(env, e_code, &size_code))
  {
    return enif_make_badarg(env);
  }

  char code[size_code + 1];

  enif_get_string(env, e_code, code, size_code + 1, ERL_NIF_LATIN1);

  if (!enif_get_tuple(env, argv[2], &arity, &tuple_blocks))
  {
    printf("spawn: blocks argument is not a tuple");
  }

  if (!enif_get_tuple(env, argv[3], &arity, &tuple_threads))
  {
    printf("spawn:threads argument is not a tuple");
  }
  int b1, b2, b3, t1, t2, t3;

  enif_get_int(env, tuple_blocks[0], &b1);
  enif_get_int(env, tuple_blocks[1], &b2);
  enif_get_int(env, tuple_blocks[2], &b3);
  enif_get_int(env, tuple_threads[0], &t1);
  enif_get_int(env, tuple_threads[1], &t2);
  enif_get_int(env, tuple_threads[2], &t3);

  int size_args;

  if (!enif_get_int(env, argv[4], &size_args))
  {
    return enif_make_badarg(env);
  }

  CUdeviceptr arrays[size_args];
  float floats[size_args];
  int ints[size_args];
  double doubles[size_args];
  int arrays_ptr = 0;
  int floats_ptr = 0;
  int doubles_ptr = 0;
  int ints_ptr = 0;
  // printf("%s\n",code);
  // printf("Args: %d %d %d %d %d %d\n",b1,b2,b3,t1,t2,t3);

  char *ptx = compile_to_ptx(env, code);

  init_cuda(env);
  // int device =0;
  // CUcontext  context2 = NULL;
  // err = cuCtxCreate(&context2, 0, device);
  err = cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
  // printf("after module load\n");
  if (err != CUDA_SUCCESS)
    fail_cuda(env, err, "cuModuleLoadData jit compile");

  // And here is how you use your compiled PTX

  err = cuModuleGetFunction(&function, module, kernel_name);
  // printf("after get funcction\n");
  if (err != CUDA_SUCCESS)
    fail_cuda(env, err, "cuModuleGetFunction jit compile");

  void *args[size_args];

  list_types = argv[5];
  list_args = argv[6];

  for (int i = 0; i < size_args; i++)
  {
    char type_name[1024];
    unsigned int size_type;
    if (!enif_get_list_cell(env, list_types, &head_types, &tail_types))
    {
      printf("erro get list cell\n");
      return enif_make_badarg(env);
    }
    if (!enif_get_list_length(env, head_types, &size_type))
    {
      printf("erro get list length\n");
      return enif_make_badarg(env);
    }

    enif_get_string(env, head_types, type_name, size_type + 1, ERL_NIF_LATIN1);

    if (!enif_get_list_cell(env, list_args, &head_args, &tail_args))
    {
      printf("erro get list cell\n");
      return enif_make_badarg(env);
    }

    if (strcmp(type_name, "int") == 0)
    {
      int iarg;
      if (!enif_get_int(env, head_args, &iarg))
      {
        printf("error getting int arg\n");
        return enif_make_badarg(env);
      }
      ints[ints_ptr] = iarg;
      args[i] = (void *)&ints[ints_ptr];
      ints_ptr++;
    }
    else if (strcmp(type_name, "float") == 0)
    {

      double darg;
      if (!enif_get_double(env, head_args, &darg))
      {
        printf("error getting float arg\n");
        return enif_make_badarg(env);
      }

      floats[floats_ptr] = (float)darg;
      args[i] = (void *)&floats[floats_ptr];
      floats_ptr++;
    }
    else if (strcmp(type_name, "double") == 0)
    {

      double darg;
      if (!enif_get_double(env, head_args, &darg))
      {
        printf("error getting double arg\n");
        return enif_make_badarg(env);
      }

      doubles[doubles_ptr] = darg;
      args[i] = (void *)&doubles[doubles_ptr];
      doubles_ptr++;
    }
    else if (strcmp(type_name, "tint") == 0)
    {

      CUdeviceptr *array_res;
      enif_get_resource(env, head_args, ARRAY_TYPE, (void **)&array_res);
      arrays[arrays_ptr] = *array_res;
      args[i] = (void *)&arrays[arrays_ptr];
      arrays_ptr++;
    }
    else if (strcmp(type_name, "tfloat") == 0)
    {
      CUdeviceptr *array_res;
      enif_get_resource(env, head_args, ARRAY_TYPE, (void **)&array_res);
      arrays[arrays_ptr] = *array_res;
      args[i] = (void *)&arrays[arrays_ptr];
      arrays_ptr++;
    }
    else if (strcmp(type_name, "tdouble") == 0)
    {

      CUdeviceptr *array_res;
      enif_get_resource(env, head_args, ARRAY_TYPE, (void **)&array_res);
      arrays[arrays_ptr] = *array_res;
      // printf("pointer %p\n",arrays[arrays_ptr]);
      args[i] = (void *)&arrays[arrays_ptr];
      arrays_ptr++;
    }
    else
    {
      printf("Type %s not suported\n", type_name);
      return enif_make_badarg(env);
    }

    list_types = tail_types;
    list_args = tail_args;
  }

  // printf("after arguments and types\n");

  // LAUNCH KERNEL

  /// END COLLECTING TIME

  // cudaEventRecord(stop, 0) ;
  //  cudaEventSynchronize(stop) ;
  //  cudaEventElapsedTime(&time, start, stop) ;

  // printf("cuda%s\t%3.1f\n", kernel_name,time);

  init_cuda(env);

  err = cuLaunchKernel(function, b1, b2, b3, // Nx1x1 blocks
                       t1, t2, t3,           // 1x1x1 threads
                       0, 0, args, 0);
  // printf("after kernel launch\n");
  if (err != CUDA_SUCCESS)
    fail_cuda(env, err, "cuLaunchKernel jit compile");

  cuCtxSynchronize();

  // int ptr_matrix[1000];
  // CUdeviceptr *dev_array = (CUdeviceptr*) args[1];
  // err=  cuMemcpyDtoH(ptr_matrix, dev_array, 3*sizeof(int)) ;
  // printf("pointer %p\n",*dev_array);
  //  printf("blah %p\n",args[0]);
  if (err != CUDA_SUCCESS)
  {
    char message[200]; // printf("its ok\n");
    const char *error;
    cuGetErrorString(err, &error);
    strcpy(message, "Error at kernel launch: ");
    strcat(message, error);
    enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }

  return enif_make_int(env, 0);
}

*/

// This function retrieves the OpenCL array from the device and returns it to the host as an Erlang term.
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
    enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }

  // Allocate memory in host for the result
  void *host_result_data = (void *)enif_make_new_binary(env, data_size, &result);

  // Copying data from device to host
  try
  {
    open_cl->readBuffer(dev_array, host_result_data, data_size);
  }
  catch (const std::exception &e)
  {
    std::cerr << "[ERROR] (get_gpu_array_nif) copying data from device to host: " << e.what() << std::endl;
    enif_raise_exception(env, enif_make_string(env, e.what(), ERL_NIF_LATIN1));
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
  cl::Buffer dev_array;
  ERL_NIF_TERM result_term;

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
    enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }

  // Allocate memory on the GPU and copy the data from the host array
  // Note: The host_array_el.data is a pointer to the data in the Erlang binary
  try
  {
    dev_array = open_cl->createBuffer(data_size, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (void *)host_array_el.data);
  }
  catch (const std::exception &e)
  {
    std::cerr << "[ERROR] (create_gpu_array_nx_nif) creating GPU buffer: " << e.what() << std::endl;
    enif_raise_exception(env, enif_make_string(env, e.what(), ERL_NIF_LATIN1));
  }

  cl::Buffer *gpu_res = (cl::Buffer *)enif_alloc_resource(ARRAY_TYPE, sizeof(cl::Buffer));
  *gpu_res = dev_array;
  result_term = enif_make_resource(env, gpu_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
  enif_release_resource(gpu_res);

  return result_term;
}

// Creates a new empty GPU array with the specified number of rows, columns, and type
static ERL_NIF_TERM new_gpu_array_nif(ErlNifEnv *env, int /* argc */, const ERL_NIF_TERM argv[])
{
  int nrow, ncol;
  size_t data_size;
  ERL_NIF_TERM return_term;
  cl::Buffer dev_array;

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
    enif_raise_exception(env, enif_make_string(env, message, ERL_NIF_LATIN1));
  }

  // Allocate memory on the GPU
  try
  {
    dev_array = open_cl->createBuffer(data_size, CL_MEM_READ_WRITE);
  }
  catch (const std::exception &e)
  {
    std::cerr << "[ERROR] Failed to create GPU buffer: " << e.what() << std::endl;
    enif_raise_exception(env, enif_make_string(env, e.what(), ERL_NIF_LATIN1));
  }

  cl::Buffer *gpu_res = (cl::Buffer *)enif_alloc_resource(ARRAY_TYPE, sizeof(cl::Buffer));
  *gpu_res = dev_array;
  return_term = enif_make_resource(env, gpu_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
  enif_release_resource(gpu_res);

  return return_term;
}

// This function synchronizes the OpenCL command queue, ensuring that all previously enqueued commands have completed.
static ERL_NIF_TERM synchronize_nif(ErlNifEnv *env, int /* argc */, const ERL_NIF_TERM /* argv */[])
{
  open_cl->synchronize();

  return enif_make_int(env, 0);
}

static ErlNifFunc nif_funcs[] = {
    // { .name = "jit_compile_and_launch_nif", .arity = 7, .fptr = jit_compile_and_launch_nif, .flags = 0 },
    {.name = "new_gpu_array_nif", .arity = 3, .fptr = new_gpu_array_nif, .flags = 0},
    {.name = "get_gpu_array_nif", .arity = 4, .fptr = get_gpu_array_nif, .flags = 0},
    {.name = "create_gpu_array_nx_nif", .arity = 4, .fptr = create_gpu_array_nx_nif, .flags = 0},
    {.name = "synchronize_nif", .arity = 0, .fptr = synchronize_nif, .flags = 0}};

ERL_NIF_INIT(Elixir.OCLPolyHok, nif_funcs, &load, NULL, NULL, &unload)
