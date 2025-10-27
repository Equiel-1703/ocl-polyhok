# OCL-PolyHok: An OpenCL backend for PolyHok

The OCL-PolyHok project is an implementation of the [PolyHok DSL](https://github.com/ardubois/poly_hok) using OpenCL as the backend technology for parallel computing, instead of PolyHok's original CUDA backend.

This implementation expands the compatible hardware range of PolyHok by leveraging OpenCL's ability to run accross various GPUs from different vendors.

## OCL-PolyHok Features

- OpenCL backend for PolyHok DSL
- Support for a wide range of GPUs
- Seamless integration with existing PolyHok codebases

## Prerequisites

To get started with OCL-PolyHok, first ensure you have the following prerequisites on your system:

- **Elixir 1.17 with Erlang/OTP 27 or higher**. We recommend using [asdf](https://asdf-vm.com/) for managing Elixir and Erlang versions.

- **Erlang development libraries**. In Debian/Ubuntu systems, you can install them via:

  ```bash
  sudo apt install erlang-dev
  ```

- **OpenCL 2.0 compatible hardware**

- **OpenCL C and C++ headers and development libraries**. On Debian/Ubuntu systems, you can install them via:

  ```bash
  sudo apt install ocl-icd-opencl-dev opencl-c-headers opencl-clhpp-headers
  ```

  It is important to note that the above packages provide only the OpenCL headers and ICD loader. You will also need to install the appropriate OpenCL driver for your specific GPU hardware. For example, if you are using an NVIDIA GPU, you will need to install the NVIDIA OpenCL driver:
  
  ```bash
  sudo apt-get install nvidia-opencl-dev
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

## Licensing

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
