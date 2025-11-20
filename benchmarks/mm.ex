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

{size, mode} =
  try do
    [size, mode] = System.argv()
    {size, mode}
  rescue
    _ ->
      IO.puts("Usage: mix run benchmarks/mm.ex [MATRIX_SIZE] [0|1]")

      IO.puts("  MATRIX_SIZE: Size of the square matrices to be multiplied (MxM)")

      IO.puts(
        "  0: Initialize matrices with random values using OCLPolyHok.new_nx_from_function/4"
      )

      IO.puts(
        "  1: Initialize matrices with sequential values using Nx.tensor/2 and Enum.to_list/1"
      )

      System.halt(0)
  end

m = String.to_integer(size)
n = String.to_integer(mode)

start = System.monotonic_time()

mat1 =
  if(n == 0) do
    OCLPolyHok.new_nx_from_function(m, m, {:f, 32}, fn -> :rand.uniform(1000) end)
  else
    Nx.tensor(Enum.to_list(1..(m * m)), type: :f32) |> Nx.reshape({m, m})
  end

mat2 =
  if(n == 0) do
    OCLPolyHok.new_nx_from_function(m, m, {:f, 32}, fn -> :rand.uniform(1000) end)
  else
    Nx.tensor(Enum.to_list(1..(m * m)), type: :f32) |> Nx.reshape({m, m})
  end

matrices_end = System.monotonic_time()

IO.puts("\nMatrices initialized.\n")
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

kernel_time = System.convert_time_unit(kernel_end - kernel_start, :native, :millisecond)
matrices_time = System.convert_time_unit(matrices_end - start, :native, :millisecond)
total_time = kernel_time + matrices_time

IO.puts(
  "\nNX creation time: #{matrices_time} ms"
)

IO.puts(
  "Kernel time: #{kernel_time} ms"
)

IO.puts("Total time: #{total_time} ms\n")

IO.puts("Checking 10 random spots in the result matrix...\n")

CheckMM.check_spots(10, m, mat1, mat2, result)
