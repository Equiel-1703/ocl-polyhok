require OCLPolyHok

defmodule BMP do
  @on_load :load_nifs
  def load_nifs do
    :erlang.load_nif(~c"./priv/bmp_nifs", 0)
  end

  def gen_bmp_int_nif(_string, _dim, _mat) do
    raise "gen_bmp_nif not implemented"
  end

  def gen_bmp_float_nif(_string, _dim, _mat) do
    raise "gen_bmp_nif not implemented"
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
    x = get_global_id(0)
    y = get_global_id(1)

    if(x < size && y < size) do
      f(resp, x, y, arg1)
    end
  end

  def run_julia(size) do
    result_tensor = OCLPolyHok.tensor({size * size, 4}, {:s, 32})

    threadsPerDimension = 4
    numberOfBlocks = div(size + threadsPerDimension - 1, threadsPerDimension)

    f = &Julia.julia_function/4

    OCLPolyHok.with OCLPolyHok.cpu() do
      OCLPolyHok.spawn(
        &Julia.mapgen2D_xy_1para_noret_ker/4,
        {numberOfBlocks, numberOfBlocks, 1},
        {threadsPerDimension, threadsPerDimension, 1},
        [result_tensor, size, size, f]
      )
    end

    result_tensor
  end
end

[arg] = System.argv()
dim = String.to_integer(arg)

prev = System.monotonic_time()

image = Julia.run_julia(dim)

next = System.monotonic_time()

IO.puts(
  "OCLPolyHok (CPU)\t#{dim}\t#{System.convert_time_unit(next - prev, :native, :millisecond)}"
)

BMP.gen_bmp_int(~c"julia_set.bmp", dim, image)
