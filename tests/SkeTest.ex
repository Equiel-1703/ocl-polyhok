require OCLPolyHok
use Ske

n = 1000

arr1 = Nx.tensor([Enum.to_list(1..n)],type: {:s, 32})


arr1
    |> OCLPolyHok.new_gnx
    |> Ske.map(OCLPolyHok.phok fn (x) -> x + 1 end)
    |> OCLPolyHok.get_gnx
    |> IO.inspect
