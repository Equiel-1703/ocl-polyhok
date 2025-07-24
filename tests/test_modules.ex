defmodule TestModules do
  defmacro defmodule_t(header, do: block) do
    IO.inspect header, label: "Header"
    IO.inspect block, label: "Block"

    quote do
      defmodule unquote(header) do
        unquote(block)
      end
    end
  end
end
