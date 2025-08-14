/*
    Defines the OpenCL version to use and include the OpenCL headers
*/

#pragma once

#define OPENCL_VERSION 200 // We are going to be using OpenCL 2.0

#define CL_TARGET_OPENCL_VERSION OPENCL_VERSION
#define CL_HPP_TARGET_OPENCL_VERSION OPENCL_VERSION
#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>