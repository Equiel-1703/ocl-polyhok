require OCLPolyHok

# OCLPolyHok.set_debug_logs(true)

OCLPolyHok.defmodule DP do
  include(CAS_Poly)

  defk map_2kernel(a1, a2, a3, size, f) do
    id = get_global_id(0)

    if(id < size) do
      a3[id] = f(a1[id], a2[id])
    end
  end

  defk reduce_kernel(arr, result, initial, f, n) do
    __shared__(cache[8])

    tid = get_global_id(0)
    cacheIndex = get_local_id(0)

    temp = initial

    if tid < n do
      temp = f(arr[tid], temp)
    end

    cache[cacheIndex] = temp
    __syncthreads()

    i = get_local_size(0) / 2

    while i != 0 do
      if cacheIndex < i do
        cache[cacheIndex] = f(cache[cacheIndex + i], cache[cacheIndex])
      end

      __syncthreads()
      i = i / 2
    end

    if cacheIndex == 0 do
      current_value = result[0]
      new_value = f(cache[0], current_value)

      while(current_value != cas_float(result, current_value, new_value)) do
        current_value = result[0]
        new_value = f(cache[0], current_value)
      end
    end
  end

  def map2(t1, t2, func) do
    shape = OCLPolyHok.get_shape(t1)
    type = OCLPolyHok.get_type(t1)
    len = Nx.size(shape)

    # New empty tensor to hold the result
    result_tensor = OCLPolyHok.tensor(shape, type)

    # Small sizes are a good fit for CPU execution
    threadsPerBlock = 8
    numberOfBlocks = div(len + threadsPerBlock - 1, threadsPerBlock)

    OCLPolyHok.with OCLPolyHok.cpu() do
      OCLPolyHok.spawn(
        &DP.map_2kernel/5,
        {numberOfBlocks, 1, 1},
        {threadsPerBlock, 1, 1},
        [t1, t2, result_tensor, len, func]
      )
    end

    result_tensor
  end

  def reduce(tensor, initial, f) do
    shape = OCLPolyHok.get_shape(tensor)
    type = OCLPolyHok.get_type(tensor)
    len = Nx.size(shape)

    result_tensor = OCLPolyHok.tensor([initial], type)

    threadsPerBlock = 8
    numberOfBlocks = div(len + threadsPerBlock - 1, threadsPerBlock)

    OCLPolyHok.with OCLPolyHok.cpu() do
      OCLPolyHok.spawn(
        &DP.reduce_kernel/5,
        {numberOfBlocks, 1, 1},
        {threadsPerBlock, 1, 1},
        [tensor, result_tensor, initial, f, len]
      )
    end

    result_tensor
  end
end

[arg] = System.argv()

n = String.to_integer(arg)

vet1 = OCLPolyHok.tensor({n}, :f32, fn _i -> 1.0 end)
vet2 = OCLPolyHok.tensor({n}, :f32, fn _i -> 2.0 end)

prev = System.monotonic_time()

res = DP.map2(vet1, vet2, OCLPolyHok.phok(fn a, b -> a * b end)) |> DP.reduce(0.0, OCLPolyHok.phok(fn a, b -> a + b end))

next = System.monotonic_time()

res_value = res[0] |> Nx.to_number()
expected_value = n * 2

IO.inspect(vet1, label: "Input tensor 1")
IO.inspect(vet2, label: "Input tensor 2")
IO.inspect(res_value, label: "Dot product result")
IO.puts("Expected result: #{expected_value}")

IO.puts("OCLPolyHok (CPU)\t#{n}\t#{System.convert_time_unit(next - prev, :native, :millisecond)}")
