require OCLPolyHok

OCLPolyHok.defmodule ArraySum do
  defk sum_ker(a1, a2, size) do
    index = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    sum = 0.0
    for i in range(index, size, stride) do
      sum = sum + a1[i]
    end
  end

  def sum(input) do
    IO.puts "ArraySum.sum called"
    shape = OCLPolyHok.get_shape(input)
    type = OCLPolyHok.get_type(input)
    result_gpu = OCLPolyHok.new_gnx(shape, type)

    size = Tuple.product(shape)
    threadsPerBlock = 128
    numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

    OCLPolyHok.spawn(&ArraySum.sum_ker/3,
              {numberOfBlocks, 1, 1},
              {threadsPerBlock, 1, 1},
              [input, result_gpu, size])
    result_gpu
  end

end

a = Nx.tensor(Enum.to_list(1..100), type: {:f, 32})

result = a
    |> OCLPolyHok.new_gnx
    |> ArraySum.sum()
    |> OCLPolyHok.get_gnx

IO.inspect(result, limit: :infinity)
