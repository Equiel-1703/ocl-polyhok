require OCLPolyHok

OCLPolyHok.set_debug_logs(true)

Nx.default_backend(Nx.BinaryBackend)

OCLPolyHok.defmodule DP do
  defk map_2kernel(a1, a2, a3, size, f) do
    id = get_global_id(0)

    if(id < size) do
      a3[id] = f(a1[id], a2[id])
    end
  end

  # This reduce kernel was rewritten to better perform in CPU
  defk reduce_kernel(arr, result_arr, initial_value, f, arr_size) do
    tid = get_global_id(0)
    num_cores = get_global_size(0)

    range = (arr_size + num_cores - 1) / num_cores;
    start = tid * range
    stop = start + range

    if stop > arr_size do
      stop = arr_size
    end

    local_sum = initial_value

    for i in range(start, stop) do
      local_sum = f(local_sum, arr[i])
    end

    if start < arr_size do
      result_arr[tid] = local_sum
    else
      result_arr[tid] = initial_value
    end
  end

  def map2(t1, t2, func) do
    shape = OCLPolyHok.get_shape(t1)
    type = OCLPolyHok.get_type(t1)
    len = Nx.size(shape)

    # New empty tensor to hold the result
    result_tensor = OCLPolyHok.tensor(shape, type)

    # Small sizes are a good fit for CPU execution
    # threadsPerBlock = 8
    # numberOfBlocks = div(len + threadsPerBlock - 1, threadsPerBlock)

    OCLPolyHok.with OCLPolyHok.cpu() do
      OCLPolyHok.spawn(
        &DP.map_2kernel/5,
        {len},
        {0},
        [t1, t2, result_tensor, len, func]
      )
    end

    result_tensor
  end

  def reduce(tensor, initial, f) do
    cores = 12

    shape = OCLPolyHok.get_shape(tensor)
    type = OCLPolyHok.get_type(tensor)
    len = Nx.size(shape)

    result_tensor = OCLPolyHok.tensor({cores}, type, fn _i -> initial end)

    OCLPolyHok.with OCLPolyHok.cpu() do
      OCLPolyHok.spawn(
        &DP.reduce_kernel/5,
        {cores},
        {1},
        [tensor, result_tensor, initial, f, len]
      )
    end

    Nx.sum(result_tensor)
  end
end

[arg] = System.argv()

n = String.to_integer(arg)

IO.puts("Using Nx backend: #{inspect(Nx.default_backend())}\n")

vet1 = OCLPolyHok.tensor({n}, :f32, fn _i -> 1.0 end)
vet2 = OCLPolyHok.tensor({n}, :f32, fn _i -> 2.0 end)

prev = System.monotonic_time()

res =
  DP.map2(vet1, vet2, OCLPolyHok.phok(fn a, b -> a * b end))
  |> DP.reduce(0.0, OCLPolyHok.phok(fn a, b -> a + b end))

next = System.monotonic_time()

res_value = res |> Nx.to_number()
expected_value = n * 2

IO.inspect(vet1, label: "Input tensor 1")
IO.inspect(vet2, label: "Input tensor 2")
IO.inspect(res, label: "result tensor")
IO.inspect(res_value, label: "Dot product result")
IO.puts("Expected result: #{expected_value}")

IO.puts("OCLPolyHok (CPU)\t#{n}\t#{System.convert_time_unit(next - prev, :native, :millisecond)}")
