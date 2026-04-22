defmodule OCLPolyHok do
  @on_load :load_nifs

  # This function is a @on_load callback that is called when the module is loaded.
  # It attempts to load the NIF (Native Implemented Function) library from the specified path.
  # It prints a success message if the library is loaded successfully, and an error otherwise.
  # The BEAM VM is shut down if the NIF fails to load.
  def load_nifs() do
    ret = :erlang.load_nif(to_charlist("./priv/gpu_nifs"), 0)

    case ret do
      :ok ->
        # The Erlang VM sets the SIGCHLD signal to be ignored by default to avoid zombies, but some OpenCL implementations
        # (like PoCL) require it to be set to the default handler to work properly. So I've set it to the default handler 
        # when the NIF library is loaded. As far as I understand, this should not cause any issues in the BEAM VM, in fact,
        # even José Valim had a similar issue working in TensorFlow: 
        # + SOURCE: https://erlang.org/pipermail/erlang-questions/2020-November/100109.html
        # An Erlang developer said that the VM doesn't really care about this signal - it just ignores it. The problem 
        # is more about zombie processes that may be created by other stuff running in the same process, a.k.a BEAM, like
        # other NIFs or Erlang Ports. Therefore, we need to be careful.
        # - Henrique
        :os.set_signal(:sigchld, :default)

        :ok

      {:error, reason} ->
        IO.puts("Failed to load NIF: #{inspect(reason)}")
        :erlang.nif_error(reason)
    end
  end

  # The phok macro is used to create an anonymous function that can be passed to GPU kernels.
  # It takes a function definition as input, adds a return statement to the body,
  # generates a unique name for the function, and returns a tuple containing the function type (:anon),
  # name, and the function itself.
  defmacro phok({:fn, aa, [{:->, bb, [para, body]}]}) do
    body = OCLPolyHok.OpenCLBackend.add_return(body)
    name = "anon_" <> OCLPolyHok.OpenCLBackend.gen_lambda_name()
    function = {:fn, aa, [{:->, bb, [para, body]}]}
    resp = quote(do: {:anon, unquote(name), unquote(Macro.escape(function))})
    resp
  end

  defmacro gpu_for({:<-, _, [var, tensor]}, do: b) do
    quote do:
            OCLPolyHok.new_gnx(unquote(tensor))
            |> PMap.map(OCLPolyHok.phok(fn unquote(var) -> unquote(b) end))
            |> OCLPolyHok.get_gnx()
  end

  defmacro gpu_for({:<-, _, [var1, {:.., _, [_b1, e1]}]}, arr1, arr2, do: body) do
    r =
      quote do:
              PMap.comp_func(
                unquote(arr1),
                unquote(arr2),
                unquote(e1),
                OCLPolyHok.phok(fn unquote(arr1), unquote(arr2), unquote(var1) ->
                  unquote(body)
                end)
              )

    r
  end

  defmacro gpufor({:<-, _, [var, tensor]}, do: b) do
    quote do: Comp.comp(unquote(tensor), OCLPolyHok.phok(fn unquote(var) -> unquote(b) end))
  end

  defmacro gpufor({:<-, _, [var1, {:.., _, [_b1, e1]}]}, arr1, arr2, do: body) do
    r =
      quote do:
              Comp.comp_xy_2arrays(
                unquote(arr1),
                unquote(arr2),
                unquote(e1),
                OCLPolyHok.phok(fn unquote(arr1), unquote(arr2), unquote(var1) ->
                  unquote(body)
                end)
              )

    r
  end

  defmacro gpufor(
             {:<-, _, [var1, {:.., _, [_b1, e1]}]},
             {:<-, _, [var2, {:.., _, [_b2, e2]}]},
             arr1,
             arr2,
             par3,
             do: body
           ) do
    r =
      quote do:
              MM.comp2xy2D1p(
                unquote(arr1),
                unquote(arr2),
                unquote(par3),
                unquote(e1),
                unquote(e2),
                OCLPolyHok.phok(fn unquote(arr1),
                                   unquote(arr2),
                                   unquote(par3),
                                   unquote(var1),
                                   unquote(var2) ->
                  unquote(body)
                end)
              )

    r
  end

  # This is the defmodule macro that defines a new OCLPolyHok module.
  # This macro basicallly processes the module header and body internally, and generates a new module
  # wich replaces the kernels and device functions with exceptions (you can only execute kernels with 'spawn').
  defmacro defmodule(header, do: body) do
    {:__aliases__, _, [module_name]} = header

    # JIT.process_module will capture the functions ASTs, their type and call graph, storing them
    # in a map.
    JIT.process_module(module_name, body)

    # The new module that will be genearated here will throw exceptions when a kernel or device
    # function is called directly without using the 'spawn' function.
    ast_new_module = OCLPolyHok.OpenCLBackend.gen_new_module(header, body)
    ast_new_module
  end

  # ----------------- With Macro ------------------
  defmacro with(ctx, do: body) do
    new_body = process_with_body(body, ctx)

    quote do
      unquote(new_body)
    end
  end

  # == Helper functions of 'with' macro ==
  defp process_with_body({:__block__, _, commands}, ctx) do
    new_commands = Enum.map(commands, fn command -> process_with_command(command, ctx) end)
    {:__block__, [], new_commands}
  end

  defp process_with_body(command, ctx) do
    process_with_command(command, ctx)
  end

  defp process_with_command(c, ctx) do
    # IO.inspect(c, label: "Processing command")

    new_c =
      case c do
        {:=, _, [left, right]} -> {:=, [], [left, process_with_exp(right, ctx)]}
        _ -> process_with_exp(c, ctx)
      end

    new_c
  end

  defp process_with_exp(exp, ctx) do
    new_exp =
      case exp do
        {{:., _, [{:__aliases__, _, [:OCLPolyHok]}, fun_name]}, _, args} ->
          # IO.inspect({fun_name, args}, label: "Processing function")
          {{:., [], [{:__aliases__, [], [:OCLPolyHok]}, fun_name]}, [], [ctx | args]}

        _ ->
          exp
      end

    new_exp
  end

  # ----------------- Synchronize function -----------------

  def synchronize(%OCLPolyHok.Context{device: d}) do
    synchronize_nif(d)
  end

  # ----------------- Set debug logs function -----------------

  def set_debug_logs(enable) do
    Agent.update(:debug_logs_agent, fn _old -> enable end)
    set_debug_logs_nif(enable)
  end

  # ----------------- GPU NX miscellaneous functions -----------------

  def get_type_gnx(_ctx, {{:nx, type, _shape, _name, _ref}, _gnx_ctx}), do: type

  def get_type_gnx({{:nx, type, _shape, _name, _ref}, _gnx_ctx}), do: type

  def get_type(%Nx.Tensor{type: type}), do: type

  def get_shape_gnx(_ctx, {{:nx, _type, shape, _name, _ref}, _gnx_ctx}), do: shape

  def get_shape_gnx({{:nx, _type, shape, _name, _ref}, _gnx_ctx}), do: shape

  def get_shape(%Nx.Tensor{shape: shape}), do: shape

  # ===== Context Initializers -- based on MONAD pattern =====
  def cpu() do
    %OCLPolyHok.Context{device: :cpu}
  end

  def gpu() do
    %OCLPolyHok.Context{device: :gpu}
  end

  # ------- Helper functions for Nx and GNx creation -------
  defp get_type_charlist(type) do
    case type do
      t when t in [{:f, 32}, :f32] -> Kernel.to_charlist("float")
      t when t in [{:f, 64}, :f64] -> Kernel.to_charlist("double")
      t when t in [{:s, 32}, :s32] -> Kernel.to_charlist("int")
      x -> raise "OCLPolyHok: type #{inspect(x)} is not suported"
    end
  end

  # ------- New GPU NX Functions -------

  # == Helper functions for creating a new GNx
  defp new_gnx_from_tensor(array, type, shape, name, device) do
    {l, c} =
      case shape do
        {c} -> {1, c}
        {l, c} -> {l, c}
        {l1, l2, c} -> {l1 * l2, c}
      end

    t_charlist = get_type_charlist(type)
    ref = new_array_from_nx_nif(array, l, c, t_charlist, device)

    {:nx, type, shape, name, ref}
  end

  defp new_gnx_empty(shape, type, device) do
    {l, c} =
      case shape do
        {c} -> {1, c}
        {l, c} -> {l, c}
        {l1, l2, c} -> {l1 * l2, c}
      end

    t_charlist = get_type_charlist(type)
    ref = new_empty_array_nif(l, c, t_charlist, device)

    {:nx, type, shape, nil, ref}
  end

  # == New from nx tensor
  def new_gnx(
        %OCLPolyHok.Context{} = ctx,
        %Nx.Tensor{
          data: data,
          type: type,
          shape: shape,
          names: name
        }
      ) do
    %Nx.BinaryBackend{state: array} = data

    gnx = new_gnx_from_tensor(array, type, shape, name, ctx.device)

    {gnx, ctx}
  end

  # == New empty gnx
  def new_gnx(%OCLPolyHok.Context{} = ctx, shape, type) do
    gnx = new_gnx_empty(shape, type, ctx.device)

    {gnx, ctx}
  end

  # ------- Function to retrieve device arrays (gnx) back to Elixir -------
  def get_gnx(
        %OCLPolyHok.Context{} = ctx,
        {{:nx, type, shape, name, gnx_ref}, %OCLPolyHok.Context{} = gnx_ctx}
      ) do
    cond do
      gnx_ctx.device == ctx.device ->
        :ok

      true ->
        raise "Device mismatch: the current context is from device '#{ctx.device}', but the provided GNx argument is in a context with device '#{gnx_ctx.device}'. GNx = #{inspect({:nx, type, shape, name, gnx_ref})}"
    end

    {l, c} =
      case shape do
        {c} -> {1, c}
        {l, c} -> {l, c}
        {d1, d2, d3} -> {d1 * d2, d3}
      end

    t_charlist = get_type_charlist(type)
    ref = get_device_array_nif(gnx_ref, l, c, t_charlist, ctx.device)

    Nx.from_binary(ref, type) |> Nx.reshape(shape, names: name)
  end

  # ------- New NX Tensor functions (they allocate aligned memory) -------

  # == tensor/2 clauses
  def tensor(list, type: t) when is_list(list), do: tensor(list, t)

  # Tensor from Elixir list
  def tensor(list, type) when is_list(list) do
    shape = TensorTools.calculate_list_dimensions(list)

    cond do
      tuple_size(shape) > 3 ->
        raise "OCLPolyHok.tensor/2: OCL-PolyHok only supports tensors with up to 3 dimensions, but got a tensor with shape #{inspect(shape)}"

      true ->
        :ok
    end

    array_len =
      case shape do
        {c} -> c
        {l, c} -> l * c
        {l, c, d} -> l * c * d
      end

    flat_list = List.flatten(list)

    t_charlist = get_type_charlist(type)
    binary = new_aligned_nx_from_list_nif(flat_list, array_len, t_charlist)

    Nx.from_binary(binary, type) |> Nx.reshape(shape)
  end

  # Empty tensor
  def tensor(shape, type: t) when is_tuple(shape), do: tensor(shape, t)

  def tensor(shape, type) when is_tuple(shape) do
    array_len =
      case shape do
        {c} ->
          c

        {l, c} ->
          l * c

        {l, c, d} ->
          l * c * d

        _ ->
          raise "OCLPolyHok.tensor/2: shape must be a tuple of 1, 2 or 3 dimensions, but got #{inspect(shape)}"
      end

    t_charlist = get_type_charlist(type)
    binary = new_empty_aligned_nx_nif(array_len, t_charlist)

    Nx.from_binary(binary, type) |> Nx.reshape(shape)
  end

  def tensor(shape, type: t, fun: f) when is_tuple(shape) and is_function(f),
    do: tensor(shape, t, f)

  # == tensor/3 clauses
  def tensor(%OCLPolyHok.Context{} = ctx, list, type: t) when is_list(list),
    do: tensor(ctx, list, t)

  def tensor(%OCLPolyHok.Context{} = ctx, list, type) when is_list(list) do
    cond do
      ctx.device == :cpu ->
        tensor(list, type)

      true ->
        raise "Creating Nx tensors is only allowed in CPU contexts, but the current context is from device '#{ctx.device}'"
    end
  end

  # Tensor from Elixir function
  def tensor(shape, type, fun) when is_tuple(shape) and is_function(fun) do
    t_charlist = get_type_charlist(type)

    validate_function(fun, t_charlist)

    array_len =
      case shape do
        {c} ->
          c

        {l, c} ->
          l * c

        {l, c, d} ->
          l * c * d

        _ ->
          raise "OCLPolyHok.tensor/3: shape must be a tuple of 1, 2 or 3 dimensions, but got #{inspect(shape)}"
      end

    list = gen_list_from_function([], array_len, fun)

    binary = new_aligned_nx_from_list_nif(list, array_len, t_charlist)

    Nx.from_binary(binary, type) |> Nx.reshape(shape)
  end

  def tensor(%OCLPolyHok.Context{} = ctx, shape, type: t, fun: f)
      when is_tuple(shape) and is_function(f),
      do: tensor(ctx, shape, t, f)

  # == tensor/4 clauses
  def tensor(%OCLPolyHok.Context{} = ctx, shape, type, fun)
      when is_tuple(shape) and is_function(fun) do
    cond do
      ctx.device == :cpu ->
        tensor(shape, type, fun)

      true ->
        raise "Creating Nx tensors is only allowed in CPU contexts, but the current context is from device '#{ctx.device}'"
    end
  end

  # ------- Check if a Nx is aligned -------

  def is_nx_aligned?(nx) do
    %Nx.Tensor{data: %Nx.BinaryBackend{state: ref}} = nx
    is_nx_aligned_nif(ref)
  end

  # -- Helpers for generating tensor from function --

  # == Check function arity and return type for OCLPolyHok.tensor/3
  defp validate_function(fun, type_charlist) when is_function(fun) do
    if not is_function(fun, 1) do
      raise "OCLPolyHok.tensor/3: the provided function must receive exactly 1 argument (the current element index), but the provided function has arity #{:erlang.fun_info(fun)[:arity]}"
    end

    fun_type =
      case fun.(1) do
        x when is_float(x) ->
          :float

        x when is_integer(x) ->
          :integer

        x ->
          raise "OCLPolyHok.tensor/3: the provided function must return either float or integer values, but it returned a value of type '#{inspect(x)}'"
      end

    cond do
      (type_charlist == ~c"float" or type_charlist == ~c"double") and fun_type == :float ->
        :ok

      type_charlist == ~c"int" and fun_type == :integer ->
        :ok

      true ->
        raise "OCLPolyHok.tensor/3: the return type of the provided function is '#{fun_type}' and the type of the tensor to be created is '#{type_charlist}'"
    end
  end

  # == Generates an Elixir list of a given size with the elements generated from a 
  # function that receives the element index as argument
  defp gen_list_from_function(list_acc, 0, _fun), do: list_acc

  defp gen_list_from_function(list_acc, size, fun) do
    el = fun.(size)

    new_list_acc = [el | list_acc]

    gen_list_from_function(new_list_acc, size - 1, fun)
  end

  # ====== The next functions are deprecated because we now need to allocate the Nx tensors with aligned memory ======
  #
  # == Creates a new Nx tensor from a function that generates its elements
  # def new_nx_from_function(l, c, type, fun) do
  #   size = l * c

  #   ref =
  #     case type do
  #       {:f, 32} -> new_matrix_from_function_f(size - 1, fun, <<fun.()::float-little-32>>)
  #       {:f, 64} -> new_matrix_from_function_d(size - 1, fun, <<fun.()::float-little-64>>)
  #       {:s, 32} -> new_matrix_from_function_i(size - 1, fun, <<fun.()::integer-little-32>>)
  #     end

  #   %Nx.Tensor{data: %Nx.BinaryBackend{state: ref}, type: type, shape: {l, c}, names: [nil, nil]}
  # end

  # # ----------------- Helper functions for new_nx_from_function -----------------
  # defp new_matrix_from_function_d(0, _, accumulator), do: accumulator

  # defp new_matrix_from_function_d(size, function, accumulator),
  #   do:
  #     new_matrix_from_function_d(
  #       size - 1,
  #       function,
  #       <<accumulator::binary, function.()::float-little-64>>
  #     )

  # defp new_matrix_from_function_i(0, _, accumulator), do: accumulator

  # defp new_matrix_from_function_i(size, function, accumulator),
  #   do:
  #     new_matrix_from_function_i(
  #       size - 1,
  #       function,
  #       <<accumulator::binary, function.()::integer-little-32>>
  #     )

  # defp new_matrix_from_function_f(0, _, accumulator), do: accumulator

  # defp new_matrix_from_function_f(size, function, accumulator),
  #   do:
  #     new_matrix_from_function_f(
  #       size - 1,
  #       function,
  #       <<accumulator::binary, function.()::float-little-32>>
  #     )

  # # == Creates a new Nx tensor from a function that generates its elements receiving the size as argument
  # def new_nx_from_function_arg(l, c, type, fun) do
  #   size = l * c

  #   ref =
  #     case type do
  #       {:f, 32} ->
  #         new_matrix_from_function_f_arg(size - 1, fun, <<fun.(size)::float-little-32>>)

  #       {:f, 64} ->
  #         new_matrix_from_function_d_arg(size - 1, fun, <<fun.(size)::float-little-64>>)

  #       {:s, 32} ->
  #         new_matrix_from_function_i_arg(size - 1, fun, <<fun.(size)::integer-little-32>>)
  #     end

  #   %Nx.Tensor{data: %Nx.BinaryBackend{state: ref}, type: type, shape: {l, c}, names: [nil, nil]}
  # end

  # # ----------------- Helper functions for new_nx_from_function_arg -----------------

  # defp new_matrix_from_function_d_arg(0, _, accumulator), do: accumulator

  # defp new_matrix_from_function_d_arg(size, function, accumulator),
  #   do:
  #     new_matrix_from_function_d_arg(
  #       size - 1,
  #       function,
  #       <<accumulator::binary, function.(size)::float-little-64>>
  #     )

  # defp new_matrix_from_function_i_arg(0, _, accumulator), do: accumulator

  # defp new_matrix_from_function_i_arg(size, function, accumulator),
  #   do:
  #     new_matrix_from_function_i_arg(
  #       size - 1,
  #       function,
  #       <<accumulator::binary, function.(size)::integer-little-32>>
  #     )

  # defp new_matrix_from_function_f_arg(0, _, accumulator), do: accumulator

  # defp new_matrix_from_function_f_arg(size, function, accumulator),
  #   do:
  #     new_matrix_from_function_f_arg(
  #       size - 1,
  #       function,
  #       <<accumulator::binary, function.(size)::float-little-32>>
  #     )

  @doc """
  Loads the Abstract Syntax Tree (AST) for a given kernel or function used inside a kernel.

  This function tries to extract the module and function name from the provided kernel function reference (assuming to be a kernel).
  If it is a kernel, then the name is extracted this way. If it is a function name, the name is already provided (is the atom itself).

  With the name, a message is sent to the `:module_server` process to request the AST for the specified function.
  The function then waits for a response from the `:module_server` process and returns the AST. If it fails, an error is raised.

  ## Parameters

    - `kernel`: A function reference (e.g., `&Module.function/arity`) representing the kernel function whose AST is to be loaded. Or
    a function name atom (e.g., `:function_name`) representing a function used inside a kernel.

  ## Returns

    - The AST of the specified kernel function.

  ## Raises

    - Raises an error if an unknown message is received from the `:module_server`.
  """
  def load_ast(kernel) do
    # The function may receives a kernel function reference (like `&Module.function/arity`), so we need to extract
    # the module and function name from it.
    # The Macro.escape is used to convert the function reference into a form that can be pattern matched.
    # The pattern matching extracts the module and function name from the function reference.
    {_module, f_name} =
      case Macro.escape(kernel) do
        {:&, [], [{:/, [], [{{:., [], [module, f_name]}, [no_parens: true], []}, _nargs]}]} ->
          {module, f_name}

        # This fallback is used in case we receive a function name directly (for functions used inside kernels).
        f ->
          {:ok, f}
      end

    # - This old code reads the ASTs from a file, but it is commented out, so I'll not touch it.
    # bytes = File.read!("c_src/#{module}.asts")
    # map_asts = :erlang.binary_to_term(bytes)
    # IO.inspect map_size(map_asts)
    # {ast,_typed?,_types} = Map.get(map_asts,String.to_atom("#{f_name}"))
    # ast

    # Asks the `:module_server` process to get the AST for the specified function name.
    send(:module_server, {:get_ast, f_name, self()})

    # Waits for a response from the `:module_server` process and returns the AST.
    # If an unknown message is received, we raise an error.
    receive do
      {:ast, ast} -> ast
      h -> raise "unknown message for function type server #{inspect(h)}"
    end
  end

  # -------------------- Helper functions for spawn --------------------
  defp unmap_nx_tensor(%Nx.Tensor{data: %Nx.BinaryBackend{state: svm_ref}}) do
    unmap_nx_svm_nif(svm_ref)
  end

  defp map_nx_tensor(%Nx.Tensor{data: %Nx.BinaryBackend{state: svm_ref}, type: type, shape: shape}) do
    svm_len =
      case shape do
        {c} -> c
        {l, c} -> l * c
        {l, c, d} -> l * c * d
      end

    t_charlist = get_type_charlist(type)

    map_nx_svm_nif(svm_ref, svm_len, t_charlist)
  end

  defp map_all_nx_tensors(args) do
    Enum.each(args, fn arg ->
      case arg do
        %Nx.Tensor{} = nx ->
          map_nx_tensor(nx)

        _ ->
          :ok
      end
    end)
  end

  defp process_cpu_kernel_args(args) do
    Enum.map(args, fn arg ->
      case arg do
        # Check if it's a GNx tensor
        {{:nx, _type, _shape, _name, _ref}, _gnx_ctx} ->
          raise "In a CPU context, GNx tensors cannot be used as kernel arguments. Found argument: #{inspect(arg)}"

        # If it's an Nx tensor, it must be aligned
        %Nx.Tensor{} = nx ->
          if not is_nx_aligned?(nx) do
            raise "In a CPU context, all Nx tensors used as kernel arguments must be aligned. Found unaligned tensor: #{inspect(nx)}"
          end

          # We also need to unmap the Nx tensor. This tells OpenCL that Elixir is done accessing the tensor data, 
          # and the device can now have it.
          unmap_nx_tensor(nx)

          # Return the Nx tensor
          nx

        # Anything else we keep as is
        e ->
          e
      end
    end)
  end

  defp process_gpu_kernel_args(args, ctx) do
    Enum.map(args, fn arg ->
      case arg do
        # Check if it's a GNx tensor
        {{:nx, _type, _shape, _name, _ref} = gnx, gnx_ctx} ->
          # For now, we are only checking if the devices match. In the future, we may need to check other things.
          if gnx_ctx.device != ctx.device do
            raise "Device mismatch: the current context is from device '#{ctx.device}', but the provided GNx argument is in a context with device '#{gnx_ctx.device}'. GNx = #{inspect(arg)}"
          end

          # If everything is fine, we return only the gnx part. The context is no longer needed
          gnx

        # If it's an Nx tensor, we can't accept! They are only valid in CPU contexts.
        %Nx.Tensor{} = nx ->
          raise "In a GPU context, Nx tensors cannot be used as kernel arguments. Found argument: #{inspect(nx)}"

        # Anything else we keep as is
        e ->
          e
      end
    end)
  end

  # Validates the tuple size and check for zeros in the tuples. Returns the processed tuples with fixed 3 dimensions.
  # If the user wants OpenCL to decide the number of threads automatically, they can set the block size tuple to {0}.
  # In this case, we return {0, 0, 0} as the processed threads tuple.
  defp process_tuples(grid_tuple, threads_tuple) do
    threads_tuple_len = tuple_size(threads_tuple)
    grid_tuple_len = tuple_size(grid_tuple)

    cond do
      threads_tuple_len < 1 or threads_tuple_len > 3 ->
        raise "Invalid block size tuple: #{inspect(threads_tuple)}. The block size must be a tuple of 1, 2 or 3 dimensions."

      grid_tuple_len < 1 or grid_tuple_len > 3 ->
        raise "Invalid grid size tuple: #{inspect(grid_tuple)}. The grid size must be a tuple of 1, 2 or 3 dimensions."

      true ->
        :ok
    end

    # Check if thread tuple with 2 or 3 elements contains zero; Check if grid tuple contains zero
    cond do
      threads_tuple_len > 1 and Enum.any?(Tuple.to_list(threads_tuple), fn x -> x == 0 end) ->
        raise "If you wish that OpenCL decides the number of threads automatically please set the block size tuple to {0}. Otherwise, zero is not allowed for a block dimension."

      Enum.any?(Tuple.to_list(grid_tuple), fn x -> x == 0 end) ->
        raise "The grid size tuple cannot contain zero."

      true ->
        :ok
    end

    processed_threads_tuple =
      case threads_tuple do
        {x, y, z} -> {x, y, z}
        {x, y} -> {x, y, 1}
        {0} -> {0, 0, 0}
        {x} -> {x, 1, 1}
      end

    processed_grid_tuple =
      case grid_tuple do
        {x, y, z} -> {x, y, z}
        {x, y} -> {x, y, 1}
        {x} -> {x, 1, 1}
      end

    {processed_grid_tuple, processed_threads_tuple}
  end

  # ----------------------- Spawn function -----------------------
  @doc """
  Spwans a kernel with JIT compilation.

  Generates the OpenCL kernel code for the given kernel, compiles it, and queues it for execution.

  ## Parameters

    - `ctx`: The OCLPolyHok context containing the device information.
    - `k`: The kernel function to be compiled and executed.
    - `b`: A tuple containing the number of blocks on each dimension (x, y, z), a.k.a grid size.
    - `t`: A tuple containing the blocks size on each dimension (x, y, z), a.k.a thread group size.
    - `l`: A list of arguments to be passed to the kernel.
  """
  def spawn(%OCLPolyHok.Context{} = ctx, k, b, t, l) do
    # Process the kernel arguments based on the device context.
    l =
      cond do
        ctx.device == :cpu ->
          process_cpu_kernel_args(l)

        ctx.device == :gpu ->
          process_gpu_kernel_args(l, ctx)
      end

    {b, t} = process_tuples(b, t)

    # Get kernel name from the kernel function reference.
    kernel_name = JIT.get_kernel_name(k)

    # Load, from the module_server, the AST and function graph for the kernel.
    {kast, fun_graph} =
      case load_ast(k) do
        {a, g} -> {a, g}
        nil -> raise "Unknown kernel #{inspect(kernel_name)}"
      end

    # Generates a map called 'delta' that maps the formal parameters of the kernel to the inferred types
    # of the actual parameters provided to the kernel (contained in the list `l`).
    delta = JIT.gen_types_delta(kast, l)

    # FIRST, we need to infer the signature types of all functions used in the kernel (return type and args types)
    # This is needed to correctly infer the types of the kernel's internal variables and parameters, since they may depend on the return
    # types of the functions used within the kernel.

    # To start, let's get the ASTs of all functions used in the kernel (contained in the `fun_graph`). The 'fun_graph' doesn't include
    # the functions passed as arguments to the kernel, but only those used within the kernel that are not parameters.
    # This is good, because parameters functions may not exist yet at compile time (e.g. anonymous functions), an their types are
    # highly dependent on the context of the kernel execution, so they are better inferred later during the kernel inference.
    funs_graph_asts =
      JIT.get_non_parameters_func_asts(fun_graph)
      # Now we need to sort these functions in the correct order of inference
      |> JIT.sort_functions_by_call_graph()

    # We now infer the types of each function and get a new delta map that contains the function type signatures of each device function
    new_delta = JIT.infer_device_functions_types(funs_graph_asts)

    # Now we merge this new_dalta containing the type signatures of the device functions with the previous delta containing the types
    # of the kernel parameters, so when we infer the types of the kernel, it can use both the types of the kernel parameters and the types
    # of the device functions used within the kernel.
    delta = Map.merge(delta, new_delta)

    # Infers the types of the kernel's variables and functions based on the AST and the new delta map
    inf_types =
      case JIT.infer_types(kast, delta, kernel_name) do
        {:ok, types} -> types
        {:error, _types, reason} -> raise "Type inference failed: #{reason}"
      end

    # Check if the inferred types contain 'double' or 'tdouble' types
    contains_double =
      Map.values(inf_types) |> Enum.any?(fn x -> x == :double or x == :tdouble end)

    # If double precision is used, check if the device supports it.
    if contains_double and not double_supported_nif(ctx.device) do
      raise "[OCL-PolyHok] Your OpenCL device does not support double precision floating point operations (fp64). The 'double' data type cannot be used in kernels."
    end

    # Returns a map of formal parameters that are functions and their actual names in OpenCL code.
    # This is needed so JIT.compile_kernel can replace the function parameters with their actual names in
    # the generated OpenCL code.
    subs = JIT.get_function_parameters(kast, l)

    # Compiles the kernel AST into a string representation of the OpenCL code. The inferred types are used
    # to generate the correct OpenCL types for all the kernel internal variables and parameters.
    # The `subs` map is used to replace function parameters with their actual device function names in the generated code.
    kernel = JIT.compile_kernel(kast, inf_types, subs)

    # Here we are getting a list of tuples {actual_function_param, type} for all formal parameters that are functions.
    # This is needed because we will compile these functions and their type signatures will be used as their initial delta type map.
    funs = JIT.get_function_parameters_and_their_types(kast, l, inf_types)

    # Takes the function graph and the kernel final inferred types and creates a list of tuples where each tuple contains
    # a function name and its inferred type signature. This is used to compile the functions that are not directly
    # passed as arguments to the kernel, but are used within the kernel.
    # The kernel final inferred types contains the inferred types of these functions because during the kernel type inference
    # their type is updated. So if the type was incomplete before (e.g. just the return type was inferred), by the end of the kernel
    # inference their type should be complete (return type and args types) =D
    # I'm using the fun_graph_asts because its ordered according to dependencies
    other_funs =
      funs_graph_asts
      |> Enum.map(fn {x, _ast} -> {x, inf_types[x]} end)
      # Remove functions that could not be inferred
      |> Enum.filter(fn {_, i} -> i != nil end)

    # Compiles all functions (both those passed as arguments and those used within the kernel) with the latest inferred types
    all_funs = other_funs ++ funs

    # The JIT.compile_function/2 function compiles the provided function AND it's dependencies (other functions called within
    # a function). To avoid recompiling functions that were already compiled, we provide a MapSet of already compiled functions,
    # so the JIT.compile_function/2 can check and skip a function if necessary.
    {comp, _compiled_funs} =
      Enum.reduce(all_funs, {[], MapSet.new()}, fn fun, {code_acc, compiled_funs_acc} ->
        {new_code, compiled_funs_acc} = JIT.compile_function(fun, compiled_funs_acc)
        {code_acc ++ new_code, compiled_funs_acc}
      end)

    # The `JIT.get_includes/0` function returns a list of OpenCL code that
    # will be prepended to the generated kernel code.
    includes = JIT.get_includes()
    prog = [includes | comp] ++ [kernel]

    # Here we are concatenating the generated OpenCL code into a single string.
    prog = Enum.reduce(prog, "", fn x, y -> y <> x end)

    # Print the generated OpenCL code for debugging purposes if debug logs is enabled.
    debug_logs = Agent.get(:debug_logs_agent, fn state -> state end)

    if debug_logs do
      IO.puts("===== Generated OpenCL code for kernel '#{kernel_name}' =====")

      IO.puts(prog)

      IO.puts("==============================================================")
    end

    # 'args' is a list of the actual arguments passed to the kernel, processed to remove any function references
    args = process_args_no_fun(l)

    # 'types_args' is a list of the inferred types of the actual arguments passed to the kernel (excluding functions).
    types_args = JIT.get_types_para(kast, inf_types)

    jit_compile_and_launch_nif(
      Kernel.to_charlist(kernel_name),
      Kernel.to_charlist(prog),
      b,
      t,
      length(args),
      types_args,
      args,
      ctx.device
    )

    case ctx.device do
      # We need to map the Nx tensors before returning so Elixir can access their data again
      :cpu -> map_all_nx_tensors(l)
      # In a GPU context, we don't need to do anything after launching the kernel
      :gpu -> :ok
    end
  end

  defp process_args_no_fun([]), do: []

  defp process_args_no_fun([{:anon, _name, _type} | t1]) do
    process_args_no_fun(t1)
  end

  defp process_args_no_fun([{:func, _func, _type} | t1]) do
    process_args_no_fun(t1)
  end

  defp process_args_no_fun([arg | t1]) when is_function(arg) do
    process_args_no_fun(t1)
  end

  # Aligned Nx tensors
  defp process_args_no_fun([%Nx.Tensor{data: %Nx.BinaryBackend{state: svm_ref}} | t1]) do
    [svm_ref | process_args_no_fun(t1)]
  end

  # GNx
  defp process_args_no_fun([{:nx, _type, _shape, _name, ref} | t1]) do
    [ref | process_args_no_fun(t1)]
  end

  defp process_args_no_fun([arg | t1]) do
    [arg | process_args_no_fun(t1)]
  end

  # ----------------- NIF function definitions -----------------
  def set_debug_logs_nif(_enable) do
    raise "NIF set_debug_logs_nif/1 not implemented"
  end

  def double_supported_nif(_d) do
    raise "NIF double_supported_nif/1 not implemented"
  end

  def new_empty_array_nif(_l, _c, _type, _d) do
    raise "NIF new_empty_array_nif/4 not implemented"
  end

  def get_device_array_nif(_gnx, _l, _c, _type, _d) do
    raise "NIF get_device_array_nif/5 not implemented"
  end

  def new_array_from_nx_nif(_gnx, _l, _c, _type, _d) do
    raise "NIF new_array_from_nx_nif/5 not implemented"
  end

  def new_aligned_nx_from_list_nif(_flat_list, _list_len, _type) do
    raise "NIF new_aligned_nx_from_list_nif/3 not implemented"
  end

  def new_empty_aligned_nx_nif(_len, _type) do
    raise "NIF new_empty_aligned_nx_nif/2 not implemented"
  end

  def map_nx_svm_nif(_svm_ref, _arr_len, _type) do
    raise "NIF map_nx_svm_nif/3 not implemented"
  end

  def unmap_nx_svm_nif(_svm_ref) do
    raise "NIF unmap_nx_svm_nif/1 not implemented"
  end

  def is_nx_aligned_nif(_res) do
    raise "NIF is_nx_aligned_nif/1 not implemented"
  end

  def synchronize_nif(_d) do
    raise "NIF syncronize_nif/1 not implemented"
  end

  def jit_compile_and_launch_nif(_n, _k, _grid, _threads, _size, _types, _l, _d) do
    raise "NIF jit_compile_and_launch_nif/8 not implemented"
  end
end
