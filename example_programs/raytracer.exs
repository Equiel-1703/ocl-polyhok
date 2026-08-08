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

OCLPolyHok.defmodule RayTracer do
  defd raytracing(image_arr, img_size, spheres_arr, x, y) do
    diff_x = (x - img_size) * 1
    diff_y = (y - img_size) * 1
    ox = diff_x / 2.0
    oy = diff_y / 2.0

    r = 0.0
    g = 0.0
    b = 0.0

    maxz = -99999.0

    for i in range(0, 20) do
      sphereRadius = spheres_arr[i * 7 + 3]

      dx = ox - spheres_arr[i * 7 + 4]
      dy = oy - spheres_arr[i * 7 + 5]
      n = 0.0
      t = -99999.0
      dz = 0.0

      if dx * dx + dy * dy < sphereRadius * sphereRadius do
        dz = sqrtf(sphereRadius * sphereRadius - dx * dx - dy * dy)
        n = dz / sqrtf(sphereRadius * sphereRadius)
        t = dz + spheres_arr[i * 7 + 6]
      else
        t = -99999.0
        n = 0.0
      end

      if t > maxz do
        fscale = n
        r = spheres_arr[i * 7 + 0] * fscale
        g = spheres_arr[i * 7 + 1] * fscale
        b = spheres_arr[i * 7 + 2] * fscale
        maxz = t
      end
    end

    image_arr[0] = trunc(r * 255)
    image_arr[1] = trunc(g * 255)
    image_arr[2] = trunc(b * 255)
    image_arr[3] = 255
  end

  defk raytracing_kernel(image_arr, num_color_channes, img_size, spheres_arr) do
    x = threadIdx.x + blockIdx.x * blockDim.x
    y = threadIdx.y + blockIdx.y * blockDim.y
    offset = x + y * blockDim.x * gridDim.x

    id = num_color_channes * offset

    if offset < img_size * img_size do
      raytracing(image_arr + id, img_size, spheres_arr, x, y)
    end
  end

  def render(image_gnx, num_channels, img_size, spheres_gnx) do
    OCLPolyHok.spawn(
      &RayTracer.raytracing_kernel/4,
      {trunc(img_size / 16), trunc(img_size / 16), 1},
      {16, 16, 1},
      [image_gnx, num_channels, img_size, spheres_gnx]
    )

    image_gnx
  end
end

defmodule SphereMaker do
  defp rnd(x) do
    :rand.uniform() * x
  end

  def generate(0, _dim), do: []

  def generate(num_spheres, dim) do
    [
      rnd(1),
      rnd(1),
      rnd(1),
      rnd(trunc(dim / 10)) + dim / 50,
      rnd(dim) - trunc(dim / 2),
      rnd(dim) - trunc(dim / 2),
      rnd(dim) - trunc(dim / 2)
      | generate(num_spheres - 1, dim)
    ]
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
      IO.puts("Usage: mix run example_programs/raytracer.exs <img_dimension_pixels>")
      System.halt(1)
  end

# -----------------------------

spheres_num = 20
spheres_tensor = Nx.tensor([SphereMaker.generate(spheres_num, dim)], type: {:f, 32})

start = System.monotonic_time()

spheres_gnx = OCLPolyHok.new_gnx(spheres_tensor)
image_gnx = OCLPolyHok.new_gnx(1, dim * dim * 4, {:s, 32})

RayTracer.render(
  image_gnx,
  4,
  dim,
  spheres_gnx
)

image = OCLPolyHok.get_gnx(image_gnx)

finish = System.monotonic_time()

IO.puts("Raytracer Example")
IO.puts("=================")
IO.puts("Output image size:    #{dim}px x #{dim}px")
IO.puts(
  "Time took:            #{System.convert_time_unit(finish - start, :native, :millisecond)}ms"
)

BMP.gen_bmp_int(~c"raytracer.bmp", dim, image)
