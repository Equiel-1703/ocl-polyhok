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

defmodule CheckMM do
  def check_spots(num_spots, m, mat1, mat2, result) do
    indexes = for _ <- 1..num_spots, do: {:rand.uniform(m) - 1, :rand.uniform(m) - 1}

    Enum.each(
      indexes,
      fn {x_idx, y_idx} ->
        # Get row x_idx from mat1
        row_mat1 = Enum.map(0..(m - 1), fn col -> Nx.to_number(mat1[x_idx][col]) end)
        # Get column y_idx from mat2
        col_mat2 = Enum.map(0..(m - 1), fn row -> Nx.to_number(mat2[row][y_idx]) end)

        # Multiply every element in row_mat1 with every element in col_mat2 and sum the results
        expected_val =
          Enum.zip(row_mat1, col_mat2) |> Enum.map(fn {a, b} -> a * b end) |> Enum.sum()

        # Get computed value from result
        computed_val = Nx.to_number(result[x_idx][y_idx])

        IO.puts("* Position (#{x_idx}, #{y_idx}):")
        IO.puts("  - Expected value: #{expected_val}")
        IO.puts("  - GPU computed value: #{computed_val}")
        IO.puts("  - Diff: #{abs(expected_val - computed_val)}\n")
      end
    )
  end
end

size =
  try do
    [size] = System.argv()
    size
  rescue
    _ ->
      IO.puts("Usage: mix run benchmarks/mm.ex [MATRIX_SIZE]")
      IO.puts("  - MATRIX_SIZE: Size of the square matrices to be multiplied (MxM)")
      System.halt(0)
  end

m = String.to_integer(size)

start = System.monotonic_time()

mat1 = OCLPolyHok.new_nx_from_function(m, m, {:f, 32}, fn -> :rand.uniform(1000) end)
mat2 = OCLPolyHok.new_nx_from_function(m, m, {:f, 32}, fn -> :rand.uniform(1000) end)

matrices_end = System.monotonic_time()

IO.puts("Matrices initialized.\n")
IO.inspect(mat1, label: "Matrix 1")
IO.inspect(mat2, label: "Matrix 2")
IO.puts("")

kernel_start = System.monotonic_time()

result =
  OCLPolyHok.gpufor x <- 0..m, y <- 0..m, mat1, mat2, m do
    # Fix: this must start with 0.0 to be identified as float, otherwise results are truncated
    sum = 0.0

    for i in range(0, m, 1) do
      sum = sum + mat1[x * m + i] * mat2[i * m + y]
    end

    sum
  end

kernel_end = System.monotonic_time()

# Calculate times in milliseconds
kernel_time = System.convert_time_unit(kernel_end - kernel_start, :native, :millisecond)
matrices_time = System.convert_time_unit(matrices_end - start, :native, :millisecond)
total_time = kernel_time + matrices_time

IO.puts(
  "\nNX matrices creation time: #{matrices_time} ms"
)
IO.puts(
  "Kernel time (from Elixir perspective): #{kernel_time} ms"
)
IO.puts(
  "Total time: #{total_time} ms\n"
)

IO.puts("--- Results Check ---")
IO.puts("Checking 10 random spots in the result matrix.\n")

CheckMM.check_spots(10, m, mat1, mat2, result)
