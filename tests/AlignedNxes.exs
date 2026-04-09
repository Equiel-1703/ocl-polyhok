require OCLPolyHok

IO.puts("\n---Testando new_nx---\n")

nx_f_1 = OCLPolyHok.new_nx(1, 3, {:f, 32})

IO.inspect(nx_f_1, label: "Nx com 1 linha e 3 colunas, tipo float 32")

IO.puts("Inserindo valores no Nx...")

indices = Nx.tensor([[0, 0], [0, 1], [0, 2]])
values = Nx.tensor([1.5, 2.5, 3.5], type: {:f, 32})

new_nx = Nx.indexed_put(nx_f_1, indices, values)

IO.inspect(new_nx, label: "Nx atualizado com os valores inseridos")
