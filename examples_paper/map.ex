require OCLPolyHok

OCLPolyHok.defmodule PMap do
  defk map_ker(a1,a2,size,f) do
    index = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(index,size,stride) do
          a2[i] = f(a1[i])
    end
  end
  defd inc(x) do
    x+1
  end
  def map(input, f) do
    shape = OCLPolyHok.get_shape(input)
    type = OCLPolyHok.get_type(input)
    result_gpu = OCLPolyHok.new_gnx(shape,type)
    size = Tuple.product(shape)
    threadsPerBlock = 128;
    numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

    OCLPolyHok.spawn(&PMap.map_ker/4,
              {numberOfBlocks,1,1},
              {threadsPerBlock,1,1},
              [input,result_gpu,size, f])
    result_gpu
  end
end

n = 10000000
arr1 = Nx.tensor([Enum.to_list(1..n)],type: {:s, 32})
arr2 = Nx.tensor([Enum.to_list(1..n)],type: {:f, 32})
arr3 = Nx.tensor([Enum.to_list(1..n)],type: {:f, 64})

host_res1 = arr1
    |> OCLPolyHok.new_gnx
    |> PMap.map(&PMap.inc/1)
    |> OCLPolyHok.get_gnx

host_res2 = arr2
    |> OCLPolyHok.new_gnx
    |> PMap.map(&PMap.inc/1)
    |> OCLPolyHok.get_gnx

host_res3 = arr3
    |> OCLPolyHok.new_gnx
    |> PMap.map(OCLPolyHok.phok fn (x) -> x + 1 end)
    |> OCLPolyHok.get_gnx

IO.inspect host_res1
IO.inspect host_res2
IO.inspect host_res3
