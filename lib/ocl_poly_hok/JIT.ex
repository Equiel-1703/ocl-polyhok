require OCLPolyHok.OpenCLBackend

defmodule JIT do
  @doc """
  Compiles a function or anonymous function into OpenCL code.
  ## Parameters
    - `func`: A tuple representing the function to compile. It can be either:
      - `{:anon, fname, code, type}` for anonymous functions.
      - `{name, type}` for named functions.
  ## Returns
    - A list containing a string with the generated OpenCL code.
  """
  def compile_function({:anon, fname, code, type}) do
    delta = gen_delta_from_type(code, type)

    inf_types =
      case infer_types(code, delta, fname) do
        {:ok, types} ->
          types

        {:error, _types, reason} ->
          raise "Type inference failed for anonymous function #{fname}: #{reason}"
      end

    {:fn, _, [{:->, _, [para, body]}]} = code

    # Here I had to use the same principle as in compile_kernel to generate the parameter list
    # If the parameter is a pointer, we need to add the global address space
    param_list =
      para
      |> Enum.map(fn {p, _, _} -> OCLPolyHok.OpenCLBackend.gen_para(p, Map.get(inf_types, p)) end)
      |> Enum.filter(fn p -> p != nil end)
      |> Enum.map(fn x ->
        case String.contains?(x, "*") do
          true ->
            # If it is a pointer, we add the global address space
            "__global #{x}"

          false ->
            # If it is not a pointer, we leave it as is
            x
        end
      end)
      |> Enum.join(", ")

    param_vars =
      para
      |> Enum.map(fn {p, _, _} -> p end)

    fun_type = Map.get(inf_types, :return)

    opencl_body =
      OCLPolyHok.OpenCLBackend.gen_ocl_jit(body, inf_types, param_vars, "module", MapSet.new())

    k = OCLPolyHok.OpenCLBackend.gen_function_jit(fname, param_list, opencl_body, fun_type)

    function = "\n" <> k <> "\n\n"

    [function]
  end

  def compile_function({name, type}) do
    nast = OCLPolyHok.load_ast(name)

    case nast do
      nil ->
        [""]

      {fast, fun_graph} ->
        delta = gen_delta_from_type(fast, type)

        inf_types =
          case infer_types(fast, delta, name) do
            {:ok, types} ->
              types

            {:error, _types, reason} ->
              raise "Type inference failed for device function #{name}: #{reason}"
          end

        {:defd, _iinfo, [header, [body]]} = fast
        {fname, _, para} = header

        param_list =
          para
          |> Enum.map(fn {p, _, _} ->
            OCLPolyHok.OpenCLBackend.gen_para(p, Map.get(inf_types, p))
          end)
          |> Enum.filter(fn p -> p != nil end)
          |> Enum.map(fn x ->
            case String.contains?(x, "*") do
              true ->
                # If it is a pointer, we add the global address space
                "__global #{x}"

              false ->
                # If it is not a pointer, we leave it as is
                x
            end
          end)
          |> Enum.join(", ")

        param_vars =
          para
          |> Enum.map(fn {p, _, _} -> p end)

        fun_type = Map.get(inf_types, :return)

        fun_type =
          if fun_type == :unit do
            :void
          else
            fun_type
          end

        # This will generate the function body in OpenCL code
        opencl_body =
          OCLPolyHok.OpenCLBackend.gen_ocl_jit(
            body,
            inf_types,
            param_vars,
            "module",
            MapSet.new()
          )

        # This will generate the function declaration in OpenCL code
        k = OCLPolyHok.OpenCLBackend.gen_function_jit(fname, param_list, opencl_body, fun_type)

        function = "\n" <> k <> "\n\n"

        other_funs =
          fun_graph
          |> Enum.map(fn x -> {x, inf_types[x]} end)
          |> Enum.filter(fn {_, i} -> i != nil end)

        comp = Enum.map(other_funs, &JIT.compile_function/1)
        comp = Enum.reduce(comp, [], fn x, y -> y ++ x end)
        comp ++ [function]
    end
  end

  @doc """
  Generates a mapping of formal parameters to their respective types, including the return type.
  ## Parameters

    - `code`: The abstract syntax tree (AST) of the function.
    - `type`: A tuple containing the return type and a list of types for the parameters.
  ## Returns
    - A map where keys are formal parameter names and values are their corresponding types. The return type is stored with the key `:return`.
  """
  def gen_delta_from_type({:defd, _, [header, [_body]]}, {return_type, types}) do
    {_, _, formal_para} = header

    delta =
      formal_para
      |> Enum.map(fn {p, _, _} -> p end)
      |> Enum.zip(types)
      |> Map.new()

    Map.put(delta, :return, return_type)
  end

  def gen_delta_from_type({:fn, _, [{:->, _, [para, _body]}]}, {return_type, types}) do
    delta =
      para
      |> Enum.map(fn {p, _, _} -> p end)
      |> Enum.zip(types)
      |> Map.new()

    Map.put(delta, :return, return_type)
  end

  @doc """
  Infers the types of variables in the function's AST based on the provided delta mapping.

  ## Parameters

    - `code`: The abstract syntax tree (AST) of the function.
    - `delta`: A map where keys are variable names and values are their corresponding types.

  ## Returns
    - A tuple containing:
      - `:ok` if the type inference was successful without any type errors, or `:error` if there were type errors.
      - A map where keys are variable names and values are their inferred types.
      - An optional reason for the error if the inference failed.
  """
  def infer_types({:defk, _, [_header, [body]]}, delta, kernel_name) do
    OCLPolyHok.TypeInference.type_check(delta, body, kernel_name)
  end

  def infer_types({:defd, _, [_header, [body]]}, delta, fun_name) do
    OCLPolyHok.TypeInference.type_check(delta, body, fun_name)
  end

  def infer_types({:fn, _, [{:->, _, [_para, body]}]}, delta, fun_name) do
    OCLPolyHok.TypeInference.type_check(delta, body, fun_name)
  end

  @doc """
  Compiles a kernel definition into OpenCL code.

  ## Parameters

    - `kernel_ast`: The abstract syntax tree (AST) of the kernel definition.
    - `inf_types`: A map where keys are variable names and values are their inferred types.
    - `subs`: A map of formal parameters that are functions mapped to their actual names in OpenCL code.
  ## Returns
    - A string containing the generated OpenCL code for the kernel.
  """
  def compile_kernel({:defk, _, [header, [body]]}, inf_types, subs) do
    {fname, _, para} = header

    # From OpenCL Programming Guide:

    # "Arguments to a kernel function that are declared to be a pointer of a type
    # must point to one of the following address spaces only: global, local, or
    # constant. Not specifying an address space name for such arguments will
    # result in a compilation error."

    # This means that, for our purposes, we can use the global address space for pointers and leave
    # scalar values as they are (passed by value).
    param_list =
      para
      |> Enum.map(fn {p, _, _} -> OCLPolyHok.OpenCLBackend.gen_para(p, Map.get(inf_types, p)) end)
      |> Enum.filter(fn p -> p != nil end)
      |> Enum.map(fn x ->
        case String.contains?(x, "*") do
          true ->
            # If it is a pointer, we add the global address space prefix
            "__global #{x}"

          false ->
            # If it is not a pointer, we leave it as is
            x
        end
      end)
      |> Enum.join(", ")

    param_vars =
      para
      |> Enum.map(fn {p, _, _} -> p end)

    opencl_body =
      OCLPolyHok.OpenCLBackend.gen_ocl_jit(body, inf_types, param_vars, "module", subs)

    k = OCLPolyHok.OpenCLBackend.gen_kernel_jit(fname, param_list, opencl_body)

    "\n" <> k <> "\n\n"
  end

  @doc """
  Returns a list of types of all formal parameters of a kernel, excluding function types.

  ## Parameters

    - `kernel_ast`: The abstract syntax tree (AST) of the kernel definition.
    - `delta`: A map where keys are variable names and values are their corresponding types.

  ## Returns
    - A list of charlists representing the types of the formal parameters.
  """
  def get_types_para({:defk, _, [header, [_body]]}, delta) do
    {_, _, formal_para} = header

    formal_para
    |> Enum.map(fn {p, _, _} -> delta[p] end)
    |> Enum.filter(fn p ->
      case p do
        # Ignoring function types
        {_, _} -> false
        _ -> true
      end
    end)
    |> Enum.map(fn x -> Kernel.to_charlist(to_string(x)) end)
  end

  @doc """
  Returns a list of tuples {function_name, type} of all formal parameters that are functions.

  If the actual parameter is an anonymous function, it returns {:anon, name, code, type}.
  """
  def get_function_parameters_and_their_types({:defk, _, [header, [_body]]}, actual_para, delta) do
    {_, _, formal_para} = header

    formal_para
    |> Enum.map(fn {p, _, _} -> p end)
    |> Enum.zip(actual_para)
    |> Enum.filter(fn {_n, p} -> is_function_para(p) end)
    |> Enum.map(fn {n, p} ->
      case p do
        {:anon, name, code} -> {:anon, name, code, delta[n]}
        _ -> {get_function_name(p), delta[n]}
      end
    end)
  end

  @doc """
  Returns a map of formal parameters that are functions mapped to their actual names in OpenCL code.
  """
  def get_function_parameters({:defk, _, [header, [_body]]}, actual_para) do
    {_, _, formal_para} = header

    formal_para
    |> Enum.map(fn {p, _, _} -> p end)
    |> Enum.zip(actual_para)
    |> Enum.filter(fn {_n, p} -> is_function_para(p) end)
    |> Enum.reduce(Map.new(), fn {n, p}, map -> Map.put(map, n, get_function_name(p)) end)
  end

  def is_anon(func) do
    case func do
      {:anon, _name, _code} -> true
      _ -> false
    end
  end

  def is_function_para(func) do
    case func do
      {:anon, _name, _code} -> true
      func when is_function(func) -> true
      _h -> false
    end
  end

  @doc """
  Extracts the function name from an anonymous function or a function reference.

  ## Parameters

    - `fun`: The function, which can be either an anonymous function or a function reference.

  ## Returns
    - The atom representing the function name.
  """
  def get_function_name({:anon, name, _code}) do
    name
  end

  def get_function_name(fun) do
    {_module, f_name} =
      case Macro.escape(fun) do
        {:&, [], [{:/, [], [{{:., [], [module, f_name]}, [no_parens: true], []}, _nargs]}]} ->
          {module, f_name}

        _ ->
          raise "Argument to spawn should be a function: #{inspect(Macro.escape(fun))}"
      end

    f_name
  end

  @doc """
  Finds the types of the actual parameters and creates a mapping of formal parameters to these inferred types.
  """
  def gen_types_delta({:defk, _, [header, [_body]]}, actual_param) do
    {_, _, formal_para} = header
    inferred_types = infer_types_actual_parameters(actual_param)

    formal_para
    |> Enum.map(fn {p, _, _} -> p end)
    |> Enum.zip(inferred_types)
    |> Map.new()
  end

  @doc """
  The gen_types_delta for device functions is just a trick =D
  Since the function signature of a device function doesn't include any hint about its parameters types,
  we set all of them to :none, so the TypeInference module will try to figure out the types based on the body of the
  device function only.
  """
  def gen_types_delta_device({:defd, _, [header, [_body]]}) do
    {_, _, formal_para} = header

    delta =
      formal_para
      |> Enum.map(fn {p, _, _} -> {p, :none} end)
      |> Map.new()

    # The return type is initially set to :none (unknown)
    Map.put(delta, :return, :none)
  end

  # Returns a list of atoms representing the names of the formal parameters of a device function.
  defp get_parameter_names_device({:defd, _, [header, [_body]]}) do
    {_, _, formal_para} = header

    formal_para
    |> Enum.map(fn {p, _, _} -> p end)
  end

  @doc """
  Retrieves the AST of all non-parameter functions called within a kernel or device function.

  ## Parameters
    - `fun_graph`: A list of function names (atoms) that are called within the kernel or device function.

  ## Returns
    - A list of tuples {function_name, ast} where `function_name` is the name of the function and `ast` is its abstract syntax tree.
  """
  def get_non_parameters_func_asts(fun_graph) do
    fun_graph
    # discard special functions
    |> Enum.filter(fn f -> not OCLPolyHok.TypeInference.is_special_function?(f) end)
    |> Enum.map(fn f ->
      # Load function ast from module server
      {ast, _funs} = OCLPolyHok.load_ast(f)
      # Return function name and its ast as a tuple
      {f, ast}
    end)
  end

  @doc """
  Infers and returns the name and type signature of all device functions used in a kernel or device function based on their ASTs.

  ## Parameters
    - `funs_graph_asts`: A list of tuples {function_name, ast}
  ## Returns
    - A list of tuples {function_name, {return_type, [param_types]}} representing the inferred type signatures of the device functions.
  """
  def infer_device_functions_types(funs_graph_asts) do
    funs_graph_asts
    # Remove functions that were not found (ast == nil)
    |> Enum.filter(fn {_f, ast} -> ast != nil end)
    |> Enum.map(fn {f, ast} ->
      delta = gen_types_delta_device(ast)

      inf_types =
        case infer_types(ast, delta, f) do
          {:ok, types} -> types
          {:error, types, _reason} -> types
        end

      # Formatting the return type and parameter types of the function in a tuple {return_type, [param_types]}
      parameters_type_list =
        get_parameter_names_device(ast)
        |> Enum.map(fn p_name -> Map.get(inf_types, p_name) end)

      return_type = Map.get(inf_types, :return)

      ft = {f, {return_type, parameters_type_list}}

      ft
    end)
  end

  @doc """
  Infers the types of actual parameters passed to a kernel or function.

  ## Parameters

    - `actual_param`: A list of actual parameters passed to the kernel or function.

  ## Returns
    - A list of inferred types corresponding to the actual parameters. These types are represented as atoms.

    The types allowed are:
      - `:tfloat` for 32-bit floating-point numbers.
      - `:tdouble` for 64-bit floating-point numbers.
      - `:tint` for 32-bit integers.
      - `:float` for literal floating-point numbers.
      - `:int` for literal integers.
      - `:none` for function types (like anonymous functions or function references).
  """
  def infer_types_actual_parameters([]) do
    []
  end

  def infer_types_actual_parameters([h | t]) do
    case h do
      {:nx, type, _shape, _name, _ref} ->
        case type do
          {:f, 32} -> [:tfloat | infer_types_actual_parameters(t)]
          {:f, 64} -> [:tdouble | infer_types_actual_parameters(t)]
          {:s, 32} -> [:tint | infer_types_actual_parameters(t)]
        end

      {:matrex, _kref, _size} ->
        [:tfloat | infer_types_actual_parameters(t)]

      {:anon, _name, _code} ->
        [:none | infer_types_actual_parameters(t)]

      float when is_float(float) ->
        [:float | infer_types_actual_parameters(t)]

      int when is_integer(int) ->
        [:int | infer_types_actual_parameters(t)]

      func when is_function(func) ->
        [:none | infer_types_actual_parameters(t)]
    end
  end

  @doc """
  Retrieves the code for the includes stored in the module server.

  # Returns
    - A string containing the concatenated include code.
  """
  def get_includes() do
    send(:module_server, {:get_include, self()})

    inc =
      receive do
        {:include, inc} -> inc
        h -> raise "unknown message for function type server #{inspect(h)}"
      end

    case inc do
      nil -> ""
      list -> Enum.reduce(list, "", fn x, y -> y <> x end)
    end
  end

  @doc """
  Extracts the kernel function name from a given kernel reference.

  ## Parameters

    - `kernel`: A function reference, like `&Module.function/arity`.

  ## Returns

    - The atom representing the kernel function name.

  ## Raises

    - Raises an exception if the provided kernel is not a valid function reference.

  ## Examples

      iex> get_kernel_name(&MyModule.my_kernel/2)
      :my_kernel
  """
  def get_kernel_name(kernel) do
    case Macro.escape(kernel) do
      {:&, [], [{:/, [], [{{:., [], [_module, kernelname]}, [no_parens: true], []}, _nargs]}]} ->
        kernelname

      _ ->
        IO.inspect(kernel, label: "Invalid kernel")
        raise "OCLPolyHok.build: invalid kernel"
    end
  end

  @doc """
  Processes a module and populates the module_server with information about functions (their ast, and call graph).
  It spawns a module server if it is not already running, which contains two maps:

  - A map of function names to their types.
  - A map of function names to their ASTs.

  ## Parameters

    - `module_name`: The name of the module to process.
    - `body`: The body or content associated with the module.
  """
  def process_module(module_name, body) do
    # initiate server that collects types and asts
    if Process.whereis(:module_server) == nil do
      pid = spawn_link(fn -> module_server(%{}, %{}) end)
      Process.register(pid, :module_server)
    end

    # If the module body is a block, process its definitions, otherwise process the body directly.
    # Usually, if the body is not a block, it will be a single definition of function or kernel.
    _defs =
      case body do
        {:__block__, [], definitions} -> process_definitions(module_name, definitions, [])
        _ -> process_definitions(module_name, [body], [])
      end
  end

  @doc """
  This server constructs two maps:
  - Function names to types.
  - Function names to {AST, functions_called}.

  Types are used to type check at runtime a kernel call, while ASTs are used to recompile a kernel at runtime,
  substituting the names of the formal parameters of a function for the actual parameters.
  """
  def module_server(types_map, ast_map) do
    receive do
      {:add_ast, fun, ast, funs} ->
        module_server(types_map, Map.put(ast_map, fun, {ast, funs}))

      {:get_ast, f_name, pid} ->
        send(pid, {:ast, ast_map[f_name]})
        module_server(types_map, ast_map)

      {:add_type, fun, type} ->
        module_server(Map.put(types_map, fun, type), ast_map)

      {:get_map, pid} ->
        send(pid, {:map, {types_map, ast_map}})
        module_server(types_map, ast_map)

      {:get_include, pid} ->
        send(pid, {:include, ast_map[:include]})
        module_server(types_map, ast_map)

      {:add_include, inc} ->
        case ast_map[:include] do
          nil -> module_server(types_map, Map.put(ast_map, :include, [inc]))
          l -> module_server(types_map, Map.put(ast_map, :include, [inc | l]))
        end

      {:kill} ->
        :ok
    end
  end

  # Processes a list of definitions (kernels, device functions, include directives) and registers them
  # in the module server.
  #
  ## Parameters
  #
  #  - `module_name`: The name of the module being processed.
  #  - `definitions`: The ast of the module to process its definitions.
  #  - `l`: An accumulator list (not used in this implementation).
  #
  ## Returns
  #  - `:ok` when all definitions have been processed.
  defp process_definitions(_module_name, [], _l), do: :ok

  defp process_definitions(module_name, [h | t], l) do
    case h do
      {:defk, _, [header, [_body]]} ->
        # Get function name from header
        {fname, _, _para} = header
        # Get list of functions called inside the kernel
        funs = find_functions(h)
        # Register the function in the module server
        register_function(module_name, fname, h, funs)
        # Go to next definition
        process_definitions(module_name, t, [
          {module_name, fname, h, funs} | t
        ])

      {:defd, ii, [header, [body]]} ->
        # Get function name from header
        {fname, _, _para} = header

        # Travels the function body and adds a return statement if the function returns an expression
        body = OCLPolyHok.TypeInference.add_return(Map.put(%{}, :return, :none), body)

        # Get list of functions called inside the device function
        funs = find_functions({:defd, ii, [header, [body]]})
        # Register the function in the module server
        register_function(module_name, fname, {:defd, ii, [header, [body]]}, funs)
        # Go to next definition
        process_definitions(module_name, t, [
          {module_name, fname, {:defd, ii, [header, [body]]}, funs} | l
        ])

      {:include, _, [{_, _, [name]}]} ->
        # The include directive will read an OpenCL file with the name given in the include directive
        # and add it to the module server so that it can be added in the kernel or device function later.
        code = File.read!("c_src/Elixir.#{name}.cl")
        send(:module_server, {:add_include, code})
        process_definitions(module_name, t, l)

      _ ->
        # If it is not a device function/kernel definition nor an include directive,
        # ignore and continue processing the rest of the definitions.
        process_definitions(module_name, t, l)
    end
  end

  @doc """
  Registers a function in the module server.

  ## Parameters

    - `_module_name`: The name of the module (not used in this function).
    - `fun_name`: The name of the function to register.
    - `ast`: The abstract syntax tree (AST) of the function.
    - `funs`: A list of functions called within the function being registered.
  """
  def register_function(_module_name, fun_name, ast, funs) do
    send(:module_server, {:add_ast, fun_name, ast, funs})
  end

  @doc """
  Finds all function calls within a kernel or device function definition.

  ## Parameters
    - `ast`: The abstract syntax tree (AST) of the kernel or device function definition.

  ## Returns
    - A list of function names (atoms) that are called within the kernel or device function.
  """
  def find_functions({:defk, _i1, [header, [body]]}) do
    {_fname, _, para} = header

    param_vars =
      para
      |> Enum.map(fn {p, _, _} -> p end)
      |> MapSet.new()

    {_args, funs} = find_function_calls_body({param_vars, MapSet.new()}, body)

    MapSet.to_list(funs)
  end

  def find_functions({:defd, _i1, [header, [body]]}) do
    # IO.inspect "aqui inicio"
    {_fname, _, para} = header

    param_vars =
      para
      |> Enum.map(fn {p, _, _} -> p end)
      |> MapSet.new()

    # IO.inspect "body #{inspect body}"
    {_args, funs} = find_function_calls_body({param_vars, MapSet.new()}, body)

    MapSet.to_list(funs)
  end

  @doc """
  Traverses the body of a kernel or device function to find function calls.
  ## Parameters
    - `map`: A tuple containing a set of parameter names and a set of function names found.
    - `body`: The body of the kernel or device function.

  ## Returns
    - An updated tuple with the set of parameter names and the set of function names found.
  """
  def find_function_calls_body(map, body) do
    case body do
      {:__block__, _, _code} ->
        find_function_calls_block(map, body)

      {:do, {:__block__, pos, code}} ->
        find_function_calls_block(map, {:__block__, pos, code})

      {:do, exp} ->
        # IO.inspect "here #{inspect exp}"
        find_function_calls_command(map, exp)

      {_, _, _} ->
        find_function_calls_command(map, body)
    end
  end

  # Uses reduce to iterate through each command in the block
  # and applies find_function_calls_command to each one.
  defp find_function_calls_block(map, {:__block__, _info, code}) do
    Enum.reduce(code, map, fn x, acc -> find_function_calls_command(acc, x) end)
  end

  # Pattern matches every possible command structure to find function calls.
  # It handles various constructs like loops, conditionals, assignments, and function calls.
  # It also checks for function calls in conditional expressions and assignments.
  # When it finds a function call, it adds it to the set of functions in the map only if
  # the function is not in the parameters list of the kernel or device function.
  defp find_function_calls_command(map, code) do
    case code do
      {:for, _i, [_param, [body]]} ->
        find_function_calls_body(map, body)

      {:do_while, _i, [[doblock]]} ->
        find_function_calls_body(map, doblock)

      {:do_while_test, _i, [exp]} ->
        find_function_calls_exp(map, exp)

      {:while, _i, [bexp, [body]]} ->
        map = find_function_calls_exp(map, bexp)
        find_function_calls_body(map, body)

      # CRIAÇÃO DE NOVOS VETORES
      {{:., _i1, [Access, :get]}, _i2, [arg1, arg2]} ->
        map = find_function_calls_exp(map, arg1)
        find_function_calls_exp(map, arg2)

      {:__shared__, _i1, [{{:., _i2, [Access, :get]}, _i3, [arg1, arg2]}]} ->
        map = find_function_calls_exp(map, arg1)
        find_function_calls_exp(map, arg2)

      # assignment
      {:=, _i1, [{{:., _i2, [Access, :get]}, _i3, [{_array, _a1, _a2}, acc_exp]}, exp]} ->
        map = find_function_calls_exp(map, acc_exp)
        find_function_calls_exp(map, exp)

      {:=, _i, [_var, exp]} ->
        find_function_calls_exp(map, exp)

      {:if, _i, if_com} ->
        find_function_calls_if(map, if_com)

      {:var, _i1, [{_var, _i2, [{:=, _i3, [{_type, _ii, nil}, exp]}]}]} ->
        find_function_calls_exp(map, exp)

      {:var, _i1, [{_var, _i2, [{:=, _i3, [_type, exp]}]}]} ->
        find_function_calls_exp(map, exp)

      {:var, _i1, [{_var, _i2, [{_type, _i3, _t}]}]} ->
        map

      {:var, _i1, [{_var, _i2, [_type]}]} ->
        map

      {:type, _i1, [{_var, _i2, [{_type, _i3, _t}]}]} ->
        map

      {:type, _i1, [{_var, _i2, [_type]}]} ->
        map

      {:return, _i, [arg]} ->
        #     IO.inspect "Aqui3"
        find_function_calls_exp(map, arg)

      {fun, _info, args} when is_list(args) ->
        #    IO.inspect "Aqui3 #{length args} #{inspect fun}"
        {args, funs} = map

        if MapSet.member?(args, fun) do
          map
        else
          {args, MapSet.put(funs, fun)}
        end

      number when is_integer(number) or is_float(number) ->
        raise "Error: #{inspect(number)} is a command"

      {str, i1, a} ->
        {str, i1, a}
    end
  end

  defp find_function_calls_if(map, [bexp, [do: then]]) do
    map = find_function_calls_exp(map, bexp)
    find_function_calls_body(map, then)
  end

  defp find_function_calls_if(map, [bexp, [do: thenbranch, else: elsebranch]]) do
    map = find_function_calls_exp(map, bexp)
    map = find_function_calls_body(map, thenbranch)
    find_function_calls_body(map, elsebranch)
  end

  # This function recursively traverses the expression tree to find function calls.
  defp find_function_calls_exp(map, exp) do
    case exp do
      {{:., _i1, [Access, :get]}, _i2, [_arg1, arg2]} ->
        find_function_calls_exp(map, arg2)

      {{:., _i1, [{_struct, _i2, nil}, _field]}, _i3, []} ->
        map

      {{:., _i1, [{:__aliases__, _i2, [_struct]}, _field]}, _i3, []} ->
        map

      {op, _info, args} when op in [:+, :-, :/, :*] ->
        # IO.inspect "Aqui"
        Enum.reduce(args, map, fn x, acc -> find_function_calls_exp(acc, x) end)

      {op, _info, args} when op in [:<=, :<, :>, :>=, :&&, :||, :!, :!=, :==] ->
        Enum.reduce(args, map, fn x, acc -> find_function_calls_exp(acc, x) end)

      {var, _info, nil} when is_atom(var) ->
        map

      {fun, _info, _args} ->
        # IO.inspect "Aqui2"
        {args, funs} = map

        if MapSet.member?(args, fun) do
          map
        else
          {args, MapSet.put(funs, fun)}
        end

      float when is_float(float) ->
        map

      int when is_integer(int) ->
        map

      string when is_binary(string) ->
        map
    end
  end
end
