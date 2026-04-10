require OCLPolyHok

OCLPolyHok.set_debug_logs(true)

IO.puts("\n---Testando OCLPolyHok.tensor/2---\n")

nx_f_1 = OCLPolyHok.tensor([1.0, 2.0, 3.0], type: {:f, 32})

OCLPolyHok.is_nx_aligned?(nx_f_1)

IO.inspect(nx_f_1, label: "Nx com 1 linha e 3 colunas, tipo float 32")
