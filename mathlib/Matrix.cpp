#include "Matrix.h"
#include "MatrixException.h"

using namespace mathlib;

Matrix::Matrix(int rows, int cols) :
    rows(rows), cols(cols), mat(rows * cols)
{
}

Matrix::Matrix(int rows, int cols, std::vector<double> vec) :
    rows(rows), cols(cols), mat(vec)
{
    if (vec.size() != rows * cols)
    {
        throw MatrixException("Could not construct Matrix. Incorect vector size!"); 
    }
}

int Matrix::getRows() const noexcept
{
    return rows;
}

int Matrix::getCols() const noexcept 
{
    return cols;
}

void Matrix::setDimensions(int rows, int cols)
{
    this->rows = rows;
    this->cols = cols;

    mat.resize(rows * cols);
    std::fill(mat.begin(), mat.end(), 0);
}

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

Matrix Matrix::applyFunc(std::function<double(double)> func) const
{
    const Matrix& m = *this;
    Matrix mo(m.rows, m.cols);

    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            mo(i,j) = func(m(i,j));
        }
    } 
    return mo;
}

Matrix Matrix::T() const
{
    const Matrix& m = *this;
    Matrix tm = Matrix(m.cols, m.rows);

    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            tm(j,i) = m(i,j);
        }
    }
    return tm;
}

double& Matrix::operator()(int row, int col)
{
    return mat[row * cols + col];
}

double Matrix::operator()(int row, int col) const
{
    return mat[row * cols + col];
}

double& Matrix::operator[](int i)
{
    return mat[i];
}

Matrix Matrix::operator+(Matrix m2) const
{
    const Matrix& m1 = *this;
    if (m1.cols != m2.cols || m1.rows != m2.rows)
    {
        throw MatrixException("Wrong dimensions, can't add!");
    }
    
    Matrix m(m1.rows, m1.cols);
    for (int r = 0; r < m2.rows; r++)
    {
        for (int c = 0; c < m2.cols; c++)
        {
            m(r,c) = m1(r,c) + m2(r,c);
        }
    }
    return m;
}

Matrix Matrix::operator*(Matrix m2) const
{
    const Matrix& m1 = *this;
    if (m1.cols != m2.rows)
    {
        throw MatrixException("Wrong dimensions, can't multiply!");
    }
    
    Matrix m = Matrix(m1.rows, m2.cols);
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

std::ostream& mathlib::operator<<(std::ostream& stream, const Matrix& m)
{
    for (int c = 0; c < m.Matrix::getCols();c++)
    {
        stream << "  - ";
    }
    stream << std::endl;
    for (int r = 0; r < m.Matrix::getRows(); r++)
    {
        for (int c = 0; c < m.Matrix::getCols(); c++)
        {
            stream << "| " << m(r,c) << " ";
        }
        stream << "|" << std::endl;
        for (int c = 0; c < m.Matrix::getCols();c++)
        {
            stream << "  - ";
        }
        stream << std::endl;
    }
    return stream;
}
