# Force 4gb of min bin vheap size, this will make Erlng's VM not to try to reuse memory
# from previous allocations, but always allocate new memory, inducing the 'cold start' scenario we want
Process.flag(:min_bin_vheap_size, 4 * 1024 * 1024 * 1024)

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

mat1 = OCLPolyHok.new_nx_from_function(m, m, {:f, 32}, fn -> :rand.uniform(1000) end)
mat2 = OCLPolyHok.new_nx_from_function(m, m, {:f, 32}, fn -> :rand.uniform(1000) end)

# Allocate ~500MB of zeros. This consumes any "warm" memory sitting in the allocator.
# We use bit-syntax <<0 :: size(...)>> which is extremely fast to allocate.
spacer_size = 500 * 1024 * 1024
_warm_memory_eater = <<0 :: size(spacer_size)-unit(8)>>

kernel_start = System.monotonic_time()

_result =
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

# f_el_mat1 = Nx.to_number(mat1[0][0])
# f_el_mat2 = Nx.to_number(mat2[0][0])
# f_el_res = Nx.to_number(result[0][0])

IO.puts("OCLPolyHok\t#{m}\t#{kernel_time}")
# IO.puts("  - Sample elements: ")
# IO.puts("    * mat1[0][0]: #{f_el_mat1}")
# IO.puts("    * mat2[0][0]: #{f_el_mat2}")
# IO.puts("    * result[0][0]: #{f_el_res}\n")
