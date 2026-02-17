defmodule OCLPolyHok.CallGraphSorter do
  @moduledoc """
  This module provides a function to sort a list of functions based on their call dependencies.
  It uses a topological sort approach to ensure that if function A calls function B, then B will appear before A in the sorted list.
  """

  @doc """
  Sorts a list of {function_name, ast, dependencies} tuples topologically.
  Dependencies (called functions) will appear BEFORE the functions that call them.

  If there is a circular dependency, an error will be raised.

  # Returns
    - A list of {function_name, ast} tuples sorted by their call dependencies.
  """
  def sort(functions) do
    # Convert the list of tuples into a map for easy and O(1) lookup: %{function_name => {dependencies, ast}}
    graph = Map.new(functions, fn {name, ast, deps} -> {name, {deps, ast}} end)

    # Perform a Topological Sort using Depth-First Search (DFS)
    # Acc state: {visited_set, sorted_list}
    {_visited, sorted_names} =
      Enum.reduce(functions, {MapSet.new(), []}, fn {func, _ast, _deps}, acc ->
        visit(func, graph, MapSet.new(), acc)
      end)

    # The search will produce a list in reverse order (dependencies come after their callers),
    # so we reverse it to get the correct order
    final_order = Enum.reverse(sorted_names)

    # Now we add the ASTs to the sorted function names
    # We don't need the call graph anymore
    Enum.map(final_order, fn func_name ->
      {_deps, ast} = Map.get(graph, func_name, {[], nil})
      {func_name, ast}
    end)
  end

  # -- Private Helpers --

  defp visit(node, graph, path, {visited, _list} = acc) do
    if MapSet.member?(visited, node) do
      # If already visited globally, skip
      acc
    else
      do_visit(node, graph, path, acc)
    end
  end

  defp do_visit(node, graph, path, {visited, sorted_list}) do
    if MapSet.member?(path, node) do
      # Cycle detection: If we see a node that is currently in our recursion stack
      raise "Circular dependency detected involving function: :#{node}"
    end

    # Get dependencies. If node isn't in graph, return empty list
    {deps, _ast} = Map.get(graph, node, [])

    # Add current node to the recursion path (so we can detect cycles)
    new_path = MapSet.put(path, node)

    # Do recursion on all children first
    {new_visited, new_sorted} =
      Enum.reduce(deps, {visited, sorted_list}, fn dep, acc_inner ->
        if Map.has_key?(graph, dep) do
          # Only visit dependencies that are actually part of our project (keys in graph)
          visit(dep, graph, new_path, acc_inner)
        else
          acc_inner
        end
      end)

    # After visiting children, we mark the current node as visited and prepend to list
    {MapSet.put(new_visited, node), [node | new_sorted]}
  end
end
