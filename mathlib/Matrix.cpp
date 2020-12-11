#include "Matrix.h"
#include "MatrixException.h"

#include <memory>
#include <numeric>

using namespace mathlib;

Matrix::Matrix(int rows, int cols) :
    rows(rows), cols(cols), mat(rows * cols)
{
}

Matrix::Matrix(int rows, int cols, std::vector<float> vec) :
    rows(rows), cols(cols), mat(std::move(vec))
{
    if (mat.size() != rows * cols)
    {
        throw MatrixException("Could not construct Matrix. Incorect vector size!"); 
    }
}

const int Matrix::getRows() const noexcept
{
    return rows;
}

const int Matrix::getCols() const noexcept 
{
    return cols;
}

const std::vector<float>& Matrix::getVector() const noexcept
{
    return mat;
}

void Matrix::setDimensions(int rows, int cols)
{
    this->rows = rows;
    this->cols = cols;

    mat.resize(rows * cols);
    std::fill(mat.begin(), mat.end(), 0.f);
}

void Matrix::print() const
{
    std::cout << *this;
}

void Matrix::applyFunc(std::function<float(float)> func)
{
    Matrix& m = *this;
    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            m(i,j) = func(m(i,j));
        }
    } 
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

float Matrix::sum() const
{
    return std::accumulate(mat.begin(), mat.end(), 0.0);
}

std::vector<float>::iterator Matrix::begin()
{
    return mat.begin();
}

std::vector<float>::iterator Matrix::end()
{
    return mat.end();
}

float& Matrix::operator()(int row, int col)
{
    return mat.at(row * cols + col); // change to [] in future
}

float Matrix::operator()(int row, int col) const
{
    return mat.at(row * cols + col); // change to [] in future
}

float& Matrix::operator[](int i)
{
    return mat.at(i); // change to [] in future
}

float Matrix::operator[](int i) const
{
    return mat.at(i); // change to [] in future
}

Matrix Matrix::operator+(const Matrix& m2) const
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

Matrix Matrix::operator*(const Matrix& m2) const
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
    stream << "[" << std::endl;
    for (int r = 0; r < m.getRows(); r++)
    {
        for (int c = 0; c < m.getCols(); c++)
        {
            stream << "\t" << m(r,c) << ((c < m.getCols()-1) ? "," : ";\n");
        }
    }
    stream << "]" << std::endl;
    return stream;
}
