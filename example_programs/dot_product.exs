require OCLPolyHok

OCLPolyHok.defmodule DP do
  include(CAS_Poly)

  defk map_2kernel(a1, a2, a3, size, f) do
    id = blockIdx.x * blockDim.x + threadIdx.x

    if(id < size) do
      a3[id] = f(a1[id], a2[id])
    end
  end

  def map2(t1_gnx, t2_gnx, func) do
    {l, c} = OCLPolyHok.get_shape_gnx(t1_gnx)
    size = l * c

    type = OCLPolyHok.get_type_gnx(t2_gnx)
    result_gnx = OCLPolyHok.new_gnx(l, c, type)

    threadsPerBlock = 128
    numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

    OCLPolyHok.spawn(&DP.map_2kernel/5, {numberOfBlocks, 1, 1}, {threadsPerBlock, 1, 1}, [
      t1_gnx,
      t2_gnx,
      result_gnx,
      size,
      func
    ])

    result_gnx
  end

  defk reduce_kernel(t1, result_arr, initial, f, size) do
    __shared__(cache[128])

    tid = threadIdx.x + blockIdx.x * blockDim.x
    cacheIndex = threadIdx.x

    temp = initial

    while tid < size do
      temp = f(t1[tid], temp)
      tid = blockDim.x * gridDim.x + tid
    end

    cache[cacheIndex] = temp
    __syncthreads()

    i = blockDim.x / 2

    while i != 0 do
      if cacheIndex < i do
        cache[cacheIndex] = f(cache[cacheIndex + i], cache[cacheIndex])
      end

      __syncthreads()
      i = i / 2
    end

    if cacheIndex == 0 do
      current_value = result_arr[0]

      while(
        !(current_value == cas_float(result_arr, current_value, f(cache[0], current_value)))
      ) do
        current_value = result_arr[0]
      end
    end
  end

  def reduce(t1_gnx, initial, f) do
    {l, c} = OCLPolyHok.get_shape_gnx(t1_gnx)
    size = l * c

    type = OCLPolyHok.get_type_gnx(t1_gnx)
    result_gnx = OCLPolyHok.new_gnx(Nx.tensor([[initial]], type: type))

    threadsPerBlock = 128
    numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

    OCLPolyHok.spawn(&DP.reduce_kernel/5, {numberOfBlocks, 1, 1}, {threadsPerBlock, 1, 1}, [
      t1_gnx,
      result_gnx,
      initial,
      f,
      size
    ])

    result_gnx
  end

  def generate_nx_vet1(n) do
    l = for i <- 1..n, do: i * 1.0

    Nx.tensor([l], type: {:f, 32})
  end

  def generate_nx_vet2(n) do
    l = for _ <- 1..n, do: 2.0

    Nx.tensor([l], type: {:f, 32})
  end
end

# -- Command line handling --
n =
  case System.argv() do
    [n_str] ->
      try do
        n = String.to_integer(n_str)

        if n > 0 do
          n
        else
          IO.puts("The size of the vectors must be greater than 0.")
          System.halt(1)
        end
      rescue
        ArgumentError ->
          IO.puts("The size of the vectors must be a valid integer greater than 0.")
          System.halt(1)
      end

    _ ->
      IO.puts("Usage: mix run example_programs/dot_product.exs <size_of_vectors>")
      System.halt(1)
  end

# -----------------------------

vet1 = DP.generate_nx_vet1(n)
vet2 = DP.generate_nx_vet2(n)

start = System.monotonic_time()

vet1_gnx = OCLPolyHok.new_gnx(vet1)
vet2_gnx = OCLPolyHok.new_gnx(vet2)

dp_result =
  vet1_gnx
  |> DP.map2(vet2_gnx, OCLPolyHok.phok(fn a, b -> a * b end))
  |> DP.reduce(0.0, OCLPolyHok.phok(fn a, b -> a + b end))
  |> OCLPolyHok.get_gnx()

finish = System.monotonic_time()

IO.puts("Dot Product Example")
IO.puts("===================")
IO.puts("Input size:      #{n}")
IO.puts("Vector 1 values: 1.0 to #{n}.0")
IO.puts("Vector 2 values: 2.0 repeated #{n} times")
IO.puts("Expected result: #{(2.0 + n * 2.0) * n / 2.0}") # Arithmetic Progression Sum formula
IO.puts("Computed result: #{Nx.to_number(dp_result[0][0])}")
IO.puts("Elapsed time:    #{System.convert_time_unit(finish - start, :native, :millisecond)} ms")
