#include "MatrexUtils.hpp"

uint32_t MatrexUtils::getRows(const void *matrix)
{
    return ((uint32_t *)matrix)[0];
}

uint32_t MatrexUtils::getRows(const cl::Buffer &buffer, const OCLInterface &ocl)
{
    uint32_t rows;
    ocl.readBuffer(buffer, &rows, sizeof(uint32_t));

    return rows;
}

uint32_t MatrexUtils::getCols(const void *matrix)
{
    return ((uint32_t *)matrix)[1];
}

uint32_t MatrexUtils::getCols(const cl::Buffer &buffer, const OCLInterface &ocl)
{
    uint32_t cols;
    ocl.readBuffer(buffer, &cols, sizeof(uint32_t), sizeof(uint32_t));

    return cols;
}

void MatrexUtils::setRows(void *matrix, uint32_t rows)
{
    ((uint32_t *)matrix)[0] = rows;
}

void MatrexUtils::setRows(cl::Buffer &buffer, uint32_t rows, const OCLInterface &ocl)
{
    ocl.writeBuffer(buffer, (void *)&rows, sizeof(uint32_t));
}

void MatrexUtils::setCols(void *matrix, uint32_t cols)
{
    ((uint32_t *)matrix)[1] = cols;
}

void MatrexUtils::setCols(cl::Buffer &buffer, uint32_t cols, const OCLInterface &ocl)
{
    ocl.writeBuffer(buffer, (void *)&cols, sizeof(uint32_t), sizeof(uint32_t));
}

uint32_t MatrexUtils::getLength(const void *matrix)
{
    uint32_t rows = MatrexUtils::getRows(matrix);
    uint32_t cols = MatrexUtils::getCols(matrix);

    return rows * cols + 2; // +2 for rows and cols themselves
}

uint32_t MatrexUtils::getLength(const cl::Buffer &buffer, const OCLInterface &ocl)
{
    uint32_t rows = MatrexUtils::getRows(buffer, ocl);
    uint32_t cols = MatrexUtils::getCols(buffer, ocl);

    return rows * cols + 2; // +2 for rows and cols themselves
}