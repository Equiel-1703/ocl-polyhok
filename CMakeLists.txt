cmake_minimum_required(VERSION 3.10)
project(OCL_PolyHok_Runtime VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCL - it is REQUIRED, so CMake will stop if it is not found
find_package(OpenCL REQUIRED)

# Set sources

set(GPU_NIFS_SRC
    c_src/gpu_nifs.cpp
)

set(BMP_NIFS_SRC
    c_src/bmp_nifs.cpp
)

# Creating libraries used by the NIFs and making their headers available

add_library(OCL_INTERFACE_LIB STATIC
    c_src/ocl_interface/OCLInterface.cpp
)
# For including cldef.hpp and finding the library headers
target_include_directories(OCL_INTERFACE_LIB PUBLIC c_src)
# Link OpenCL to the OCL_INTERFACE_LIB, making it public so that gpu_nifs can use it without needing to link OpenCL directly
target_link_libraries(OCL_INTERFACE_LIB PUBLIC OpenCL::OpenCL)

add_library(BMP_LIB STATIC
    c_src/bmp/BMP.cpp
)
# For including cldef.hpp and finding the library headers
target_include_directories(BMP_LIB PUBLIC c_src)

# Marking the libraries as POSITION_INDEPENDENT_CODE for shared library compatibility
set_target_properties(OCL_INTERFACE_LIB PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(BMP_LIB PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Adding output directory as 'priv/' for the NIF libraries

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/priv")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/priv")

# Handling multi-config generators (this is needed to compile on Windows with Visual Studio)
if(CMAKE_CONFIGURATION_TYPES)
    foreach(config ${CMAKE_CONFIGURATION_TYPES})
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config} "${CMAKE_SOURCE_DIR}/priv")
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config} "${CMAKE_SOURCE_DIR}/priv")
    endforeach()
endif()

# Creating the NIF libraries and linking them with the appropriate libraries

add_library(gpu_nifs SHARED ${GPU_NIFS_SRC})
target_link_libraries(gpu_nifs PRIVATE OCL_INTERFACE_LIB)

add_library(bmp_nifs SHARED ${BMP_NIFS_SRC})
target_link_libraries(bmp_nifs PRIVATE BMP_LIB)

# Removing the 'lib' prefix from the generated shared libraries
set_target_properties(gpu_nifs PROPERTIES PREFIX "")
set_target_properties(bmp_nifs PROPERTIES PREFIX "")