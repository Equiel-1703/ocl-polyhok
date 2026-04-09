defmodule TensorTools do
  # O algortimo que calcula a dimensão das listas aninhadas foi portado diretamente do código da biblioteca
  # Numerical Elixir (Nx): https://github.com/elixir-nx/nx/blob/main/nx/lib/nx.ex#L879
  def calculate_list_dimensions(list) when is_list(list) do
    calculate_list_dimensions(list, []) |> Enum.reverse() |> List.to_tuple()
  end

  # == Lista vazia. Coloca 0 nas dimensões e retorna
  defp calculate_list_dimensions([], dimensions) do
    [0 | dimensions]
  end

  # == Lista aninhada (nesting). Aqui é um pouquinho mais elaborado
  defp calculate_list_dimensions([head | rest], _dimensions) when is_list(head) do
    # Primeiro, calculamos as dimensões da lista aninhada
    child_dimensions = calculate_list_dimensions(head, [])

    # Agora, precisamos garantir que todas as outras listas aninhadas tenham as mesmas dimensões
    # Usamos Enum.reduce para iterar sobre as outras listas (rest) e verificar se elas têm 
    # as mesmas dimensões que a primeira (child_dimensions)
    n =
      Enum.reduce(rest, 1, fn list, count ->
        case calculate_list_dimensions(list, []) do
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
  defp calculate_list_dimensions(list, dimensions) do
    [length(list) | dimensions]
  end
end