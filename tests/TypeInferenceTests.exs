require OCLPolyHok

OCLPolyHok.set_debug_logs(true)
OCLPolyHok.TypeInference.set_debug_logs(true)

OCLPolyHok.defmodule TypeInferenceTests do
  defd comp_num(num) do
    return(sqrt(num * num))
  end

  defk type_inference_ker(result_gpu_array, len) do
    index = get_global_id(0)

    if(index < len) do
      foo = comp_num(index)

      if foo > 0.0 do
        result_gpu_array[index] = 1.0
      else
        result_gpu_array[index] = 0.0
      end
    end
  end

  def run_kernel(gpu_array, len) do
    threadsPerBlock = 32
    numberOfBlocks = div(len + threadsPerBlock - 1, threadsPerBlock)

    OCLPolyHok.spawn(&TypeInferenceTests.type_inference_ker/2,
              {numberOfBlocks, 1, 1},
              {threadsPerBlock, 1, 1},
              [gpu_array, len])

    gpu_array
  end
end

# Tamanho 100
len = 100
# Cria novo array na GPU com 'len' colunas e tipo de dado float com 32 bits
gpu_array = OCLPolyHok.new_gnx({len}, {:f, 32})

# Roda o kernel na GPU e copia o resultado de para a RAM
ram_array = TypeInferenceTests.run_kernel(gpu_array, len) |> OCLPolyHok.get_gnx()

IO.inspect(ram_array, label: "Array após execução do kernel")
