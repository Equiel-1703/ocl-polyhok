# OCL-PolyHok: An OpenCL backend for PolyHok

The OCL-PolyHok project is an extension of the [PolyHok DSL](https://github.com/ardubois/poly_hok) that uses OpenCL as the backend technology for parallel computing instead of CUDA.

This implementation expands the compatible hardware range of PolyHok by leveraging OpenCL's ability to run accross various GPUs from different vendors.

## OCL-PolyHok Features

- Support for a wide range of GPUs without losing PolyHok's high-level abstractions
- Seamless integration with existing PolyHok codebases

## Prerequisites

To get started with OCL-PolyHok, first ensure you have the following prerequisites on your system:

- **Elixir 1.17 with Erlang/OTP 27 or higher**. We recommend using [asdf](https://asdf-vm.com/) for managing Elixir and Erlang versions.

- **Erlang development libraries**. In Debian/Ubuntu systems, you can install them via:

  ```bash
  sudo apt install erlang-dev
  ```

- **OpenCL 2.0 compatible hardware**. Check your GPU specifications to ensure it supports OpenCL version 2.0 or higher.

- **OpenCL generic ICD loader and C/C++ headers**. You can install the necessary OpenCL packages using your package manager. For Debian/Ubuntu systems, you can use:

  ```bash
  sudo apt install ocl-icd-opencl-dev opencl-c-headers opencl-clhpp-headers
  ```

  **Note**: The packages listed above are for general OpenCL development in C/C++. You will need to ensure that your system has the latest drivers for your GPU. This ensures that the OpenCL ICD loader can find and route commands to the appropriate OpenCL driver provided by your GPU vendor. You can verify that OpenCL is properly set up on your system by using the `clinfo` tool, which provides detailed information about the OpenCL platforms and devices available on your system. This tool may not be installed by default in some Linux distributions, but you can easily install it using your package manager. For Debian/Ubuntu systems, you can run:

  ```bash
  sudo apt install clinfo
  ```

  **Pro Tip**: OCL-PolyHok was developed and tested on a Lenovo IdeaPad 3 running Linux Mint 22.3 (Zena), equipped with an AMD Ryzen 5 5500U CPU and its integrated GPU, the AMD Lucienne. If you have a similar setup, we strongly recommend using the Mesa OpenCL driver, which works very well with AMD's iGPUs, and enabling the `rusticl` implementation (a modern OpenCL implementation written in Rust). You can install the Mesa OpenCL driver by running the following command:

  ```bash
  sudo apt install mesa-opencl-icd
  ```

  And to enable the `rusticl` implementation, set the `RUSTICL_ENABLE` environment variable to `radeonsi` in your shell configuration file (e.g., `~/.bashrc` or `~/.zshrc`):

  ```bash
  export RUSTICL_ENABLE='radeonsi'
  ```

## Getting Started

Once you have the prerequisites, follow these steps to set up the project:

1. Clone the repository and navigate to the project directory:

   ```bash
   git clone https://github.com/Equiel-1703/ocl-polyhok.git
   cd ocl-polyhok
   ```

2. Get Elixir dependencies:

    ```bash
    mix deps.get
    ```

3. Compile C++ NIFS for OpenCL and BMP generation:

    ```bash
    make all
    make bmp
    ```

4. Compile the Elixir project:

    ```bash
    mix compile
    ```

5. All done! You can now run the provided benchmarks or start developing your own PolyHok applications. As an example, this snippet runs the Julia set benchmark generating a 1,024 x 1,024 BMP image:

    ```bash
    mix run benchmarks/julia.ex 1024
    ```

<img width="1024" height="1024" alt="juliaske.bmp" src="https://github.com/user-attachments/assets/0bd736da-aeae-4702-9c84-15b5c5674ed1" />

## Licensing

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
