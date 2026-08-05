require OCLPolyHok

# BMP generation module
defmodule BMP do
  @on_load :load_nifs
  def load_nifs do
    :erlang.load_nif(~c"./priv/bmp_nifs", 0)
  end

  def gen_bmp_int_nif(_string, _dim, _mat) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def gen_bmp_float_nif(_string, _dim, _mat) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def gen_bmp_int(string, dim, %Nx.Tensor{data: data, type: _type, shape: _shape, names: _name}) do
    %Nx.BinaryBackend{state: array} = data
    gen_bmp_int_nif(string, dim, array)
  end

  def gen_bmp_float(string, dim, %Nx.Tensor{data: data, type: _type, shape: _shape, names: _name}) do
    %Nx.BinaryBackend{state: array} = data
    gen_bmp_float_nif(string, dim, array)
  end
end

# Julia module
OCLPolyHok.defmodule Julia do
  defd julia(x, y, dim) do
    scale = 0.1
    jx = scale * (dim - x) / dim
    jy = scale * (dim - y) / dim

    cr = -0.8
    ci = 0.156
    ar = jx
    ai = jy

    for i in range(0, 200) do
      nar = ar * ar - ai * ai + cr
      nai = ai * ar + ar * ai + ci

      if nar * nar + nai * nai > 1000.0 do
        return(0)
      end

      ar = nar
      ai = nai
    end

    return(1)
  end

  defd julia_function(ptr, x, y, dim) do
    offset = x + y * dim
    juliaValue = julia(x, y, dim)

    ptr[offset * 4 + 0] = 255 * juliaValue
    ptr[offset * 4 + 1] = 0
    ptr[offset * 4 + 2] = 0
    ptr[offset * 4 + 3] = 255
  end

  defk mapgen2D_xy_1para_noret_ker(resp, arg1, size, f) do
    x = blockIdx.x * blockDim.x + threadIdx.x
    y = blockIdx.y * blockDim.y + threadIdx.y

    if(x < size && y < size) do
      f(resp, x, y, arg1)
    end
  end

  def mapgen2D_step_xy_1para_noret(result_gnx, arg1, size, f) do
    OCLPolyHok.spawn(&Julia.mapgen2D_xy_1para_noret_ker/4, {size, size, 1}, {1, 1, 1}, [
      result_gnx,
      arg1,
      size,
      f
    ])

    result_gnx
  end
end

# -- Command line handling --
dim =
  case System.argv() do
    [dim_str] ->
      try do
        dim = String.to_integer(dim_str)

        if dim > 0 do
          dim
        else
          IO.puts("Image dimension must be greater than 0.")
          System.halt(1)
        end
      rescue
        ArgumentError ->
          IO.puts("The output image dimension must be a valid integer greater than 0.")
          System.halt(1)
      end

    _ ->
      IO.puts("Usage: mix run example_programs/julia.exs <img_dimension_pixels>")
      System.halt(1)
  end

# -----------------------------

start = System.monotonic_time()

result_gnx = OCLPolyHok.new_gnx(dim * dim, 4, {:s, 32})

image =
  result_gnx
  |> Julia.mapgen2D_step_xy_1para_noret(dim, dim, &Julia.julia_function/4)
  |> OCLPolyHok.get_gnx()

finish = System.monotonic_time()

BMP.gen_bmp_int(~c"julia.bmp", dim, image)

IO.puts("Output image size:    #{dim}px x #{dim}px")
IO.puts("Time took:            #{System.convert_time_unit(finish - start, :native, :millisecond)}ms")
