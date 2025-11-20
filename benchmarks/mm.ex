require OCLPolyHok

OCLPolyHok.set_debug_logs(true)

OCLPolyHok.defmodule MM do
  defk map2xy2D_kernel(arr1, arr2, par, resp, size, f) do
    row = blockIdx.y * blockDim.y + threadIdx.y
    col = blockIdx.x * blockDim.x + threadIdx.x

    if(col < size && row < size) do
      resp[row * size + col] = f(arr1, arr2, par, row, col)
    end
  end

  def map2xy2D1p(arr1, arr2, par, resp, size, f) do
    block_size = 16
    grid_rows = trunc((size + block_size - 1) / block_size)
    grid_cols = trunc((size + block_size - 1) / block_size)

    OCLPolyHok.spawn(
      &MM.map2xy2D_kernel/6,
      {grid_cols, grid_rows, 1},
      {block_size, block_size, 1},
      [arr1, arr2, par, resp, size, f]
    )
  end

  def comp2xy2D1p(arr1, arr2, par, size1, size2, f) do
    result_gpu = OCLPolyHok.new_gnx(size1, size2, OCLPolyHok.get_type(arr1))
    arr1_gpu = OCLPolyHok.new_gnx(arr1)
    arr2_gpu = OCLPolyHok.new_gnx(arr2)

    MM.map2xy2D1p(arr1_gpu, arr2_gpu, par, result_gpu, size1, f)

    r_gpu = OCLPolyHok.get_gnx(result_gpu)
    r_gpu
  end
end

[arg] = System.argv()

m = String.to_integer(arg)

# vet1 = Nx.iota({m,m}, type: :f32)
# vet2 = Nx.iota({m,m}, type: :f32)

# {mat1,_} = Nx.Random.uniform(Nx.Random.key(1), shape: {m, m}, type: :f32)
# {mat2,_} = Nx.Random.uniform(Nx.Random.key(1), shape: {m, m}, type: :f32)

# mat1 = Matrex.new(1, m*m, fn -> :rand.uniform(1000) end)
# mat2 = Matrex.new(1, m*m, fn -> :rand.uniform(1000) end)

prev = System.monotonic_time()

mat1 = OCLPolyHok.new_nx_from_function(m, m, {:f,32}, fn -> 1.0 end)
mat2 = OCLPolyHok.new_nx_from_function(m, m, {:f,32}, fn -> 1.0 end)

# mat1 = Nx.tensor(Enum.to_list(1..(m * m)), type: :f32)
# mat2 = Nx.tensor(Enum.to_list(1..(m * m)), type: :f32)

# tensors_finish = System.monotonic_time()

# mat1 = Nx.reshape(mat1, {m, m})
# mat2 = Nx.reshape(mat2, {m, m})

kernel_start = System.monotonic_time()

result =
  OCLPolyHok.gpufor x <- 0..m, y <- 0..m, mat1, mat2, m do
    sum = 0.0 # Fix: this must start with 0.0 to be identified as float, otherwise results are truncated

    for i in range(0, m, 1) do
      sum = sum + mat1[x * m + i] * mat2[i * m + y]
    end

    sum
  end

kernel_end = System.monotonic_time()

IO.puts("NX creation time: #{System.convert_time_unit(kernel_start - prev, :native, :millisecond)} ms")
IO.puts("Kernel time: #{System.convert_time_unit(kernel_end - kernel_start, :native, :millisecond)} ms")
# IO.puts("Reshape time: #{System.convert_time_unit(kernel_start - tensors_finish, :native, :millisecond)} ms")
IO.puts("Total time: #{System.convert_time_unit(kernel_end - prev, :native, :millisecond)} ms")

# Check result
IO.inspect result

# OCLPolyHok.null(mat1)
# OCLPolyHok.null(mat2)
# m1 = Matrex.reshape(mat1,m,m)
# m2 = Matrex.reshape(mat2,m,m)
# res_cpu = Matrex.dot(m1,m2)
# IO.inspect Matrex.sum(res_cpu)
# IO.inspect Matrex.sum(result)
