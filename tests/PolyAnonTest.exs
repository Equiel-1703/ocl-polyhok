require OCLPolyHok

# OCLPolyHok.set_debug_logs(true)
OCLPolyHok.TypeInference.set_debug_logs(true)

OCLPolyHok.defmodule SimpleTest do
  defk simple_kernel(array, size, f) do
    index = blockIdx.x * blockDim.x + threadIdx.x

    if (index < size) do
      array[index] = f(array[index])
    end
  end
end

array_size = 100

array_cpu_int = Nx.tensor(Enum.to_list(1..array_size), type: {:s, 32})
array_cpu_float = Nx.tensor(Enum.to_list(1..array_size), type: {:f, 32})

IO.inspect(array_cpu_int, label: "CPU Array [int]")
IO.inspect(array_cpu_float, label: "CPU Array [float]")

# Create a tensor on the GPU copying the data from the CPU tensor
array_gpu_int = array_cpu_int |> OCLPolyHok.new_gnx()
array_gpu_float = array_cpu_float |> OCLPolyHok.new_gnx()

# Spawn the kernel to run on the GPU
OCLPolyHok.spawn(
          &SimpleTest.simple_kernel/2,  # Kernel function
          {1, 1, 1},                    # Number of blocks
          {array_size, 1, 1},           # Threads per block
          [ # Kernel parameters
            array_gpu_int,
            array_size,
            OCLPolyHok.phok(fn x -> x * 2 end)
          ])

OCLPolyHok.spawn(
          &SimpleTest.simple_kernel/2,  # Kernel function
          {1, 1, 1},                    # Number of blocks
          {array_size, 1, 1},           # Threads per block
          [ # Kernel parameters
            array_gpu_float,
            array_size,
            OCLPolyHok.phok(fn x -> x * 2 end)
          ])

# Get result back to CPU
result_int = OCLPolyHok.get_gnx(array_gpu_int)
result_float = OCLPolyHok.get_gnx(array_gpu_float)

IO.inspect(result_int, label: "Result after kernel execution [int]")
IO.inspect(result_float, label: "Result after kernel execution [float]")
