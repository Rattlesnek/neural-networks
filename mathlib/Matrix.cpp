

#include "Matrix.h"

namespace mathlib
{

Matrix::Matrix(int rows, int cols) :
    rows(rows), cols(cols), mat(rows * cols)
{
}

Matrix::Matrix(const Matrix& m) :
    rows(m.rows),cols(m.cols),mat(m.mat)
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

Matrix Matrix::operator+(Matrix m2)
{
    const Matrix& m1 = *this;
    Matrix m = Matrix(*this);
    if (m1.cols != m2.cols || m1.rows != m2.rows){
        throw MatrixWrongDimensionsException("Wrong dimensions, can't add");
    }
    for (int r = 0; r < m2.rows; r++){
        for (int c = 0; c < m2.cols; c++){
            m(r,c) = m1(r,c) + m2(r,c);
        }
    }
}

Matrix Matrix::operator*(Matrix m2)
{
    const Matrix& m1 = *this;
    Matrix m = Matrix(m1.rows, m2.cols);
    if (m1.cols != m2.rows){
        throw MatrixWrongDimensionsException("Wrong dimensions,can't multiply");
    }
    for (int r = 0; r < m.rows; r++)
    {
        for (int c = 0; c < m.cols; c++)
        {
            for (int i = 0; i < m1.cols; i++)
            {
                m(r,c) += m1(r,i) * m2(i,c);
            }
        }
    }
    return m;
}


Matrix Matrix::applyFunc(std::function<double(double)> func)
{
    return Matrix(1,2);
}

std::ostream& operator<<(std::ostream &stream, const Matrix &m)
{
    for (int c = 0; c < m.Matrix::getCols();c++){
            stream << "  - ";
        }
    stream << std::endl;
    for (int r = 0; r < m.Matrix::getRows(); r++){
        for (int c = 0; c < m.Matrix::getCols(); c++){
            stream << "| " << m(r,c) << " ";
        }
        stream << "|" << std::endl;
        for (int c = 0; c < m.Matrix::getCols();c++){
            stream << "  - ";
        }
        stream << std::endl;
    }
    return stream;
}
}
