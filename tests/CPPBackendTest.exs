require OCLPolyHok

IO.puts "Running CPPBackendTest"

# Creating NX array
nx_tensor = Nx.tensor([1, 2, 3], type: :s32)

buf = OCLPolyHok.new_gnx(nx_tensor)

# Retrieving data from GPU to host
result = OCLPolyHok.get_gnx(buf)

# Verifying the result
IO.inspect(result, label: "Result from GPU")
