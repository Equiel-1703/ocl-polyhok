require OCLPolyHok

OCLPolyHok.set_debug_logs(true)

OCLPolyHok.defmodule ArraySum do
  defk sum_ker(a1, a2, result_array, size) do
    index = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(index, size, stride) do
      result_array[i] = a1[i] + a2[i]
    end
  end

  def sum(input_1, input_2) do
    # Getting shape and type of the input tensors
    shape = OCLPolyHok.get_shape(input_1)
    type = OCLPolyHok.get_type(input_1)

    # Creating a new GPU tensor to hold the result
    result_gpu = OCLPolyHok.new_gnx(shape, type)

    # Calculating the number of blocks based on the size of the input
    size = Tuple.product(shape)
    threadsPerBlock = 128
    numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

    OCLPolyHok.spawn(&ArraySum.sum_ker/4,
              {numberOfBlocks, 1, 1},
              {threadsPerBlock, 1, 1},
              [input_1, input_2, result_gpu, size]) # Kernel parameters

    result_gpu
  end

end

a = Nx.tensor(Enum.to_list(1..100), type: {:f, 32}) |> OCLPolyHok.new_gnx
b = Nx.tensor(Enum.to_list(1..100), type: {:f, 32}) |> OCLPolyHok.new_gnx

result = ArraySum.sum(a, b) |> OCLPolyHok.get_gnx

IO.inspect(result, label: "Result of ArraySum.sum: ")
