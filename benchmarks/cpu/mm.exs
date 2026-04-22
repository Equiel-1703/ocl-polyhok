require OCLPolyHok

# OCLPolyHok.set_debug_logs(true)

Nx.default_backend(Nx.BinaryBackend)

OCLPolyHok.defmodule MM do
  defk map2xy2D_kernel(arr1, arr2, par, resp, size, f) do
    row = get_global_id(1)
    col = get_global_id(0)

    if(col < size && row < size) do
      resp[row * size + col] = f(arr1, arr2, par, row, col)
    end
  end

  defd mult(arr1, arr2, size, row, col) do
    sum = 0.0

    for i in range(0, size, 1) do
      sum = sum + arr1[row * size + i] * arr2[i * size + col]
    end

    sum
  end

  def mm(arr1, arr2, size) do
    block_size = 4
    num_blocks = div(size + block_size - 1, block_size)

    type = OCLPolyHok.get_type(arr1)

    # Empty tensor to hold the result
    result_nx = OCLPolyHok.tensor({size, size}, type)

    OCLPolyHok.with OCLPolyHok.cpu() do
      OCLPolyHok.spawn(
        &MM.map2xy2D_kernel/6,
        {num_blocks, num_blocks},
        {block_size, block_size},
        [arr1, arr2, size, result_nx, size, &MM.mult/5]
      )
    end

    result_nx
  end
end

defmodule CheckMM do
  def check_spots(num_spots, size, mat1, mat2, result) do
    indexes = for _ <- 1..num_spots, do: {:rand.uniform(size) - 1, :rand.uniform(size) - 1}

    Enum.each(
      indexes,
      fn {x_idx, y_idx} ->
        # Get row x_idx from mat1
        row_mat1 = mat1[x_idx]
        # Get column y_idx from mat2
        col_mat2 = mat2[[.., y_idx]]

        # Multiply every element in row_mat1 with every element in col_mat2 and sum the results
        expected_val = Nx.dot(row_mat1, col_mat2) |> Nx.to_number()

        # Get computed value from result
        computed_val = Nx.to_number(result[x_idx][y_idx])

        IO.puts("* Position (#{x_idx}, #{y_idx}):")
        IO.puts("  - Expected value: #{expected_val}")
        IO.puts("  - Computed value: #{computed_val}")
        IO.puts("  - Diff: #{abs(expected_val - computed_val)}\n")
      end
    )
  end
end

arg =
  try do
    [arg] = System.argv()
    arg
  rescue
    _ ->
      IO.puts("Usage: mix run benchmarks/mm.ex [MATRIX_SIZE]")
      IO.puts("  - MATRIX_SIZE: Size of the square matrices to be multiplied (MxM)")
      System.halt(0)
  end

size = String.to_integer(arg)

IO.puts("Using Nx backend: #{inspect(Nx.default_backend())}\n")

# Generate random matrices in CPU memory
mat1 = OCLPolyHok.tensor({size, size}, {:f, 32}, fn _i -> :rand.uniform(100) * 1.0 end)
mat2 = OCLPolyHok.tensor({size, size}, {:f, 32}, fn _i -> :rand.uniform(100) * 1.0 end)

timing_start = System.monotonic_time()

result = MM.mm(mat1, mat2, size)

timing_end = System.monotonic_time()

# Calculate times in milliseconds
time = System.convert_time_unit(timing_end - timing_start, :native, :millisecond)

IO.puts("OCLPolyHok (CPU)\t#{size}\t#{time}")

IO.puts("\nChecking results for 5 random positions...\n")
CheckMM.check_spots(5, size, mat1, mat2, result)
