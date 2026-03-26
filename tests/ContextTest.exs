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
IO.inspect(cpu_ctx, label: "Context created")

arr_1 = Nx.tensor([1,2,3], type: {:s, 32})
IO.inspect(arr_1, label: "Original Nx Tensor 1")

arr_2 = Nx.tensor([2,4,6], type: {:s, 32})
IO.inspect(arr_2, label: "Original Nx Tensor 2")

res_arr = OCLPolyHok.with cpu_ctx do
  dev_arr_1 = OCLPolyHok.new_gnx(arr_1)
  dev_arr_2 = OCLPolyHok.new_gnx(arr_2)
  dev_res_arr = OCLPolyHok.new_gnx({1,3}, {:s, 32})

  OCLPolyHok.spawn(
      &ArraySum.sum_ker/4,
      {1, 1, 1},
      {3, 1, 1},
      [dev_arr_1, dev_arr_2, dev_res_arr, 3]
  )

  OCLPolyHok.get_gnx(dev_res_arr)
end

IO.inspect(res_arr, label: "Result array after kernel execution")
