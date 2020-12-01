

#include "Matrix.h"

using namespace mathlib;

Matrix::Matrix(int rows, int cols) :
    rows(rows), cols(cols), mat(rows * cols)
{
}




void Matrix::setDimensions(int rows, int cols)
{
    this->rows = rows;
    this->cols = cols;

    mat.resize(rows * cols);
    std::fill(mat.begin(), mat.end(), 0);
}


double& Matrix::operator()(int row, int col)
{
    return mat[row * cols + col];
}


double Matrix::operator()(int row, int col) const
{
    return mat[row * cols + col];
}


// make >> operator overload
void Matrix::print() const
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            std::cout << mat[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }
}


Matrix applyFunc(std::function<int(int)> func)
{
    return Matrix(1,2);
}

