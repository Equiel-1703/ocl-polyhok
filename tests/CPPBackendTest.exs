require OCLPolyHok

IO.puts "Running CPPBackendTest"

# Creating code in a scope block to ensure resources will be released
result = fn ->
  # Creating NX array
  nx_tensor = Nx.tensor([1, 2, 3], type: :s32)

  # Creating array on GPU from NX tensor in CPU memory
  buf = OCLPolyHok.new_gnx(nx_tensor)

  # Retrieving data from GPU to host
  result = OCLPolyHok.get_gnx(buf)

  # Verifying the result
  IO.inspect(result, label: "Result from GPU")

  # Explicitly remove reference
  buf = nil
  # Force garbage collection
  :erlang.garbage_collect()
end

# Execute the result function
result.()

# Give some time for cleanup messages to appear
Process.sleep(1000)
