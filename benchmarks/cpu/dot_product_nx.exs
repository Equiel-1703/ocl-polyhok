Nx.default_backend({EXLA.Backend, client: :host})

[arg] = System.argv()

n = String.to_integer(arg)

IO.puts("Using Nx backend: #{inspect(Nx.default_backend())}\n")

vet1 = OCLPolyHok.tensor({n}, :f32, fn _i -> 1.0 end)
vet2 = OCLPolyHok.tensor({n}, :f32, fn _i -> 2.0 end)

prev = System.monotonic_time()

res = Nx.dot(vet1, vet2)

next = System.monotonic_time()

res_value = res |> Nx.to_number()
res_type = OCLPolyHok.get_type(res)
expected_value = n * 2

IO.inspect(vet1, label: "Input tensor 1")
IO.inspect(vet2, label: "Input tensor 2")
IO.inspect(res_type, label: "Result type")
IO.inspect(res_value, label: "Dot product result")
IO.puts("Expected value\t#{expected_value}")

IO.puts("Nx.dot\t#{n}\t#{System.convert_time_unit(next - prev, :native, :millisecond)}")
