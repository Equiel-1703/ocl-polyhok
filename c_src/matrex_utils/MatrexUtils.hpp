#pragma once

#include "../ocl_interface/OCLInterface.hpp"
#include <cstdint>

class MatrexUtils
{
private:
    MatrexUtils() = default;
    ~MatrexUtils() = default;

public:
    static uint32_t getRows(const void *matrix);
    static uint32_t getRows(const cl::Buffer &buffer, OCLInterface &ocl);

    static uint32_t getCols(const void *matrix);
    static uint32_t getCols(const cl::Buffer &buffer, OCLInterface &ocl);

    static void setRows(void *matrix, uint32_t rows);
    static void setRows(cl::Buffer &buffer, uint32_t rows, OCLInterface &ocl);
    
    static void setCols(void *matrix, uint32_t cols);
    static void setCols(cl::Buffer &buffer, uint32_t cols, OCLInterface &ocl);
    
    static uint32_t getLength(const void *matrix);
    static uint32_t getLength(const cl::Buffer &buffer, OCLInterface &ocl);
};
