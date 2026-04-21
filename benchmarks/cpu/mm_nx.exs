Nx.default_backend({EXLA.Backend, client: :host})

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

# Print the current Nx backend being used
IO.puts("Using Nx backend: #{inspect(Nx.default_backend())}\n")

# Generate random matrices in CPU memory
mat1 = OCLPolyHok.tensor({size, size}, {:f, 32}, fn _i -> :rand.uniform(100) * 1.0 end)
mat2 = OCLPolyHok.tensor({size, size}, {:f, 32}, fn _i -> :rand.uniform(100) * 1.0 end)

timing_start = System.monotonic_time()

result = Nx.dot(mat1, mat2)

timing_end = System.monotonic_time()

# Calculate times in milliseconds
time = System.convert_time_unit(timing_end - timing_start, :native, :millisecond)

IO.puts("Nx.dot\t#{size}\t#{time}")

IO.puts("\nChecking results for 5 random positions...\n")
CheckMM.check_spots(5, size, mat1, mat2, result)
