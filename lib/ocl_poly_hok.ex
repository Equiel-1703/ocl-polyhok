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

  # ----------------- Synchronize function -----------------

  def synchronize() do
    synchronize_nif()
  end

  # ----------------- Set debug logs function -----------------

  def set_debug_logs(enable) do
    Agent.update(:debug_logs_agent, fn _old -> enable end)
    set_debug_logs_nif(enable)
  end

  # ----------------- GPU NX Array functions -----------------

  def get_type_gnx({:nx, type, _shape, _name, _ref}) do
    type
  end

  def get_type({:nx, type, _shape, _name, _ref}) do
    type
  end

  def get_type(%Nx.Tensor{type: type}) do
    type
  end

  def get_shape_gnx({:nx, _type, shape, _name, _ref}) do
    shape
  end

  def get_shape({:nx, _type, shape, _name, _ref}) do
    shape
  end

  def new_gnx(%Nx.Tensor{data: data, type: type, shape: shape, names: name}) do
    %Nx.BinaryBackend{state: array} = data
    # IO.inspect name
    # raise "hell"
    {l, c} =
      case shape do
        {c} -> {1, c}
        {l, c} -> {l, c}
        {l1, l2, c} -> {l1 * l2, c}
      end

    ref =
      case type do
        {:f, 32} -> create_gpu_array_nx_nif(array, l, c, Kernel.to_charlist("float"))
        {:f, 64} -> create_gpu_array_nx_nif(array, l, c, Kernel.to_charlist("double"))
        {:s, 32} -> create_gpu_array_nx_nif(array, l, c, Kernel.to_charlist("int"))
        x -> raise "new_gnx: type #{x} not suported"
      end

    {:nx, type, shape, name, ref}
  end

  def new_gnx(l, c, type) do
    # IO.puts "aque"
    ref =
      case type do
        {:f, 32} -> new_gpu_array_nif(l, c, Kernel.to_charlist("float"))
        {:f, 64} -> new_gpu_array_nif(l, c, Kernel.to_charlist("double"))
        {:s, 32} -> new_gpu_array_nif(l, c, Kernel.to_charlist("int"))
        x -> raise "new_gnx: type #{x} not suported"
      end

    {:nx, type, {l, c}, [nil, nil], ref}
  end

  def new_gnx({c}, type) do
    l = 1
    # IO.puts "aque"
    ref =
      case type do
        {:f, 32} -> new_gpu_array_nif(l, c, Kernel.to_charlist("float"))
        {:f, 64} -> new_gpu_array_nif(l, c, Kernel.to_charlist("double"))
        {:s, 32} -> new_gpu_array_nif(l, c, Kernel.to_charlist("int"))
        x -> raise "new_gnx: type #{x} not suported"
      end

    {:nx, type, {c}, [nil], ref}
  end

  def new_gnx({l, c}, type) do
    # IO.puts "aque"
    ref =
      case type do
        {:f, 32} -> new_gpu_array_nif(l, c, Kernel.to_charlist("float"))
        {:f, 64} -> new_gpu_array_nif(l, c, Kernel.to_charlist("double"))
        {:s, 32} -> new_gpu_array_nif(l, c, Kernel.to_charlist("int"))
        x -> raise "new_gnx: type #{x} not suported"
      end

    {:nx, type, {l, c}, [nil, nil], ref}
  end

  def new_gnx({d1, d2, d3}, type) do
    {l, c} = {d1 * d2, d3}

    ref =
      case type do
        {:f, 32} -> new_gpu_array_nif(l, c, Kernel.to_charlist("float"))
        {:f, 64} -> new_gpu_array_nif(l, c, Kernel.to_charlist("double"))
        {:s, 32} -> new_gpu_array_nif(l, c, Kernel.to_charlist("int"))
        x -> raise "new_gnx: type #{x} not suported"
      end

    {:nx, type, {d1, d2, d3}, [nil, nil, nil], ref}
  end

  def get_gnx({:nx, type, shape, name, ref}) do
    # IO.puts "aqui..."
    {l, c} =
      case shape do
        {c} -> {1, c}
        {l, c} -> {l, c}
        {d1, d2, d3} -> {d1 * d2, d3}
      end

    ref =
      case type do
        {:f, 32} -> get_gpu_array_nif(ref, l, c, Kernel.to_charlist("float"))
        {:f, 64} -> get_gpu_array_nif(ref, l, c, Kernel.to_charlist("double"))
        {:s, 32} -> get_gpu_array_nif(ref, l, c, Kernel.to_charlist("int"))
        x -> raise "new_gnx: type #{x} not suported"
      end

    %Nx.Tensor{data: %Nx.BinaryBackend{state: ref}, type: type, shape: shape, names: name}
  end

  ## ----------------- Creates a new Nx tensor from a function that generates its elements -----------------
  def new_nx_from_function(l, c, type, fun) do
    size = l * c

    ref =
      case type do
        {:f, 32} -> new_matrix_from_function_f(size - 1, fun, <<fun.()::float-little-32>>)
        {:f, 64} -> new_matrix_from_function_d(size - 1, fun, <<fun.()::float-little-64>>)
        {:s, 32} -> new_matrix_from_function_i(size - 1, fun, <<fun.()::integer-little-32>>)
      end

    %Nx.Tensor{data: %Nx.BinaryBackend{state: ref}, type: type, shape: {l, c}, names: [nil, nil]}
  end

  # ----------------- Helper functions for new_nx_from_function -----------------

  defp new_matrix_from_function_d(0, _, accumulator), do: accumulator

  defp new_matrix_from_function_d(size, function, accumulator),
    do:
      new_matrix_from_function_d(
        size - 1,
        function,
        <<accumulator::binary, function.()::float-little-64>>
      )

  defp new_matrix_from_function_i(0, _, accumulator), do: accumulator

  defp new_matrix_from_function_i(size, function, accumulator),
    do:
      new_matrix_from_function_i(
        size - 1,
        function,
        <<accumulator::binary, function.()::integer-little-32>>
      )

  defp new_matrix_from_function_f(0, _, accumulator), do: accumulator

  defp new_matrix_from_function_f(size, function, accumulator),
    do:
      new_matrix_from_function_f(
        size - 1,
        function,
        <<accumulator::binary, function.()::float-little-32>>
      )

  ## ----------------- Creates a new Nx tensor from a function that generates its elements receiving the size as argument -----------------
  def new_nx_from_function_arg(l, c, type, fun) do
    size = l * c

    ref =
      case type do
        {:f, 32} ->
          new_matrix_from_function_f_arg(size - 1, fun, <<fun.(size)::float-little-32>>)

        {:f, 64} ->
          new_matrix_from_function_d_arg(size - 1, fun, <<fun.(size)::float-little-64>>)

        {:s, 32} ->
          new_matrix_from_function_i_arg(size - 1, fun, <<fun.(size)::integer-little-32>>)
      end

    %Nx.Tensor{data: %Nx.BinaryBackend{state: ref}, type: type, shape: {l, c}, names: [nil, nil]}
  end

  # ----------------- Helper functions for new_nx_from_function_arg -----------------

  defp new_matrix_from_function_d_arg(0, _, accumulator), do: accumulator

  defp new_matrix_from_function_d_arg(size, function, accumulator),
    do:
      new_matrix_from_function_d_arg(
        size - 1,
        function,
        <<accumulator::binary, function.(size)::float-little-64>>
      )

  defp new_matrix_from_function_i_arg(0, _, accumulator), do: accumulator

  defp new_matrix_from_function_i_arg(size, function, accumulator),
    do:
      new_matrix_from_function_i_arg(
        size - 1,
        function,
        <<accumulator::binary, function.(size)::integer-little-32>>
      )

  defp new_matrix_from_function_f_arg(0, _, accumulator), do: accumulator

  defp new_matrix_from_function_f_arg(size, function, accumulator),
    do:
      new_matrix_from_function_f_arg(
        size - 1,
        function,
        <<accumulator::binary, function.(size)::float-little-32>>
      )

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

  # ----------------- JIT compilation and kernel spawning -----------------

  @doc """
  Spwans a kernel with JIT compilation.

  Generates the OpenCL kernel code for the given kernel, compiles it, and queues it for execution.

  ## Parameters

    - `k`: The kernel function to be compiled and executed.
    - `t`: The work group size in each dimension (a.k.a number of blocks).
    - `b`: A list containing the number of work items in each dimension (a.k.a threads per block).
    - `l`: A list of arguments to be passed to the kernel.
  """
  def spawn(k, t, b, l) do
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
      # Remove call graph from the sorted list, since we don't need it anymore
      |> Enum.map(fn {fun, ast, _call_graph} -> {fun, ast} end)

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
    if contains_double and not double_supported_nif() do
      raise "[OCL-PolyHok] Your OpenCL device does not support double precision floating point operations (fp64). The 'double' data type cannot be used in kernels."
    end

    # Returns a map of formal parameters that are functions and their actual names in OpenCL code.
    # This is needed so JIT.compile_kernel can replace the function parameters with their actual names in
    # the generated OpenCL code.
    subs = JIT.get_function_parameters(kast, l)

    # Compiles the kernel AST into a string representation of the OpenCL code. The inferred types are used
    # to generate the correct OpenCL types for the kernel parameters. The `subs` map is used to replace
    # function parameters with their actual names in the generated code.
    kernel = JIT.compile_kernel(kast, inf_types, subs)

    # Here we are getting a list of tuples {function_name, type} for all formal parameters that are functions.
    # This is needed so we can compile correctly the functions that are passed as arguments to the kernel.
    funs = JIT.get_function_parameters_and_their_types(kast, l, inf_types)

    # Takes the function graph and the kernel final inferred types and creates a list of tuples where each tuple contains
    # a function name and its inferred type signature. This is used to compile the functions that are not directly
    # passed as arguments to the kernel, but are used within the kernel.
    # The kernel final inferred types contains the inferred types of these functions because during the kernel type inference
    # their type is updated. So if the type was incomplete before (e.g. just the return type was inferred), by the end of the kernel
    # inference their type should be complete (return type and args types) =D
    other_funs =
      fun_graph
      |> Enum.map(fn x -> {x, inf_types[x]} end)
      # Remove functions that could not be inferred
      |> Enum.filter(fn {_, i} -> i != nil end)

    # Compiles all functions (both those passed as arguments and those used within the kernel).
    comp =
      Enum.map(
        funs ++ other_funs,
        fn f ->
          JIT.compile_function(f)
        end
      )

    comp = Enum.reduce(comp, [], fn x, y -> y ++ x end)

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

      # We don't print the includes to reduce clutter
      case comp do
        [] -> IO.puts(kernel)
        l -> IO.puts(Enum.reduce(l, "", fn x, y -> y <> x end) <> kernel)
      end

      IO.puts("==============================================================")
    end

    # 'args' is a list of the actual arguments passed to the kernel, processed to remove any function references
    args = process_args_no_fun(l)

    # 'types_args' is a list of the inferred types of the actual arguments passed to the kernel (excluding functions).
    types_args = JIT.get_types_para(kast, inf_types)

    jit_compile_and_launch_nif(
      Kernel.to_charlist(kernel_name),
      Kernel.to_charlist(prog),
      t,
      b,
      length(args),
      types_args,
      args
    )
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

  def double_supported_nif() do
    raise "NIF double_supported_nif/0 not implemented"
  end

  def new_gpu_array_nif(_l, _c, _type) do
    raise "NIF new_gpu_array_nif/4 not implemented"
  end

  def get_gpu_array_nif(_matrex, _l, _c, _type) do
    raise "NIF get_gpu_array_nif/4 not implemented"
  end

  def create_gpu_array_nx_nif(_matrex, _l, _c, _type) do
    raise "NIF create_gpu_array_nx_nif/4 not implemented"
  end

  def synchronize_nif() do
    raise "NIF syncronize_nif/0 not implemented"
  end

  def jit_compile_and_launch_nif(_n, _k, _t, _b, _size, _types, _l) do
    raise "NIF jit_compile_and_launch_nif/7 not implemented"
  end
end
