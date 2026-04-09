defmodule ListFlattening do
  def calculate_list_dimensions(list), do: flatten_list(list)

  defp flatten_list(list) do
    flatten_list(list, []) |> Enum.reverse() |> List.to_tuple()
  end

  # == Lista vazia. Coloca 0 nas dimensões e retorna
  defp flatten_list([], dimensions) do
    [0 | dimensions]
  end

  # == Lista aninhada (nesting). Aqui é um pouquinho mais elaborado
  defp flatten_list([head | rest], _dimensions) when is_list(head) do
    # Primeiro, calculamos as dimensões da lista aninhada
    child_dimensions = flatten_list(head, [])

    # Agora, precisamos garantir que todas as outras listas aninhadas tenham as mesmas dimensões
    # Usamos Enum.reduce para iterar sobre as outras listas (rest) e verificar se elas têm 
    # as mesmas dimensões que a primeira (child_dimensions)
    n =
      Enum.reduce(rest, 1, fn list, count ->
        case flatten_list(list, []) do
          # Se as dimensões da lista aninhada for igual ao esperado, incrementamos o contador e continuamos
          ^child_dimensions ->
            count + 1

          # Se as dimensões forem diferentes, levantamos um erro informando que as listas têm formas diferentes
          other_dimensions ->
            raise ArgumentError,
                  "cannot build tensor because lists have different shapes, got " <>
                    inspect(List.to_tuple(child_dimensions)) <>
                    " at position 0 and " <>
                    inspect(List.to_tuple(other_dimensions)) <> " at position #{count + 1}"
        end
      end)

    # Depois de verificar tudo, concatenamos as dimensões da lista aninhada e o número de listas
    # aninhadas (n) para formar as dimensões finais
    child_dimensions ++ [n]
  end

  # == Lista simples (sem nesting). Coloca o tamanho da lista nas dimensões e retorna
  defp flatten_list(list, dimensions) do
    [length(list) | dimensions]
  end
end

# Lista vazia
IO.inspect(ListFlattening.calculate_list_dimensions([]))

# 1 Dimensão
IO.inspect(ListFlattening.calculate_list_dimensions([1, 2, 3, 4]))

# 2 Dimensões
IO.inspect(
  ListFlattening.calculate_list_dimensions([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
  ])
)

# 3 Dimensões
IO.inspect(
  ListFlattening.calculate_list_dimensions([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
  ])
)

# 4 Dimensões
IO.inspect(
  ListFlattening.calculate_list_dimensions([
    [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
    [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]
  ])
)

# 5 Dimensões
IO.inspect(
  ListFlattening.calculate_list_dimensions([
    [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]],
    [[[[17, 18], [19, 20]], [[21, 22], [23, 24]]], [[[25, 26], [27, 28]], [[29, 30], [31, 32]]]]
  ])
)

# 5 Dimensões com erro no formato - tem que levantar um erro!
IO.inspect(
  ListFlattening.calculate_list_dimensions([
    [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]],
    [[[[17, 18], [19, 20]], [[21, 22], [23, 24]]], [[[25, 26], [27, 28]], [[29, 30], [31, 32, 33]]]]
  ])
)