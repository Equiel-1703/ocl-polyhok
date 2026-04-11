require OCLPolyHok

OCLPolyHok.set_debug_logs(true)

OCLPolyHok.defmodule ArraySum do
  defk sum_ker(a1, a2, result_array, size) do
    index = get_global_id(0)

    if (index < size) do
      result_array[index] = a1[index] + a2[index]
    end
  end
end

cpu_ctx = OCLPolyHok.cpu()

IO.puts("Created CPU context: #{inspect(cpu_ctx)}")

arr_1 = OCLPolyHok.tensor([1,2,3], type: {:s, 32})
arr_2 = OCLPolyHok.tensor([2,4,6], type: {:s, 32})
arr_res = OCLPolyHok.tensor([0,0,0], type: {:s, 32})

IO.puts("Input Array 1: #{inspect(arr_1)}")
IO.puts("Input Array 2: #{inspect(arr_2)}")
IO.puts("Result Array before computation: #{inspect(arr_res)}")

OCLPolyHok.with cpu_ctx do
  OCLPolyHok.spawn(
      &ArraySum.sum_ker/4,
      {1, 1, 1},
      {3, 1, 1},
      [arr_1, arr_2, arr_res, 3]
  )
end

IO.puts("Result Array after computation: #{inspect(arr_res)}")