#pragma once

#include <vector>
#include <functional>
#include <iostream>

namespace mathlib
{

class Matrix
{
    // Fields
private:    
    int rows;
    int cols;

    std::vector<double> mat;
    friend std::ostream& operator<<(std::ostream &stream, const Matrix &m);
    
    
    // Constructors / destructor
public:
    Matrix (){}
    Matrix(int rows, int cols);

    // Methods
public:
    void setDimensions(int rows, int cols);

    double& operator()(int row, int col);

    double operator()(int row, int col) const;

    void print() const;
    
    

    
    // Matrix operator+(Matrix m);
    // Matrix operator*(Matrix m); exception (wrong dimensions)
    // Matrix product(Matrix m);
    Matrix applyFunc(std::function<double(double)> func);
    // Matrix transpose();
    // Matrix times(double x);
};

std::ostream& operator<<(std::ostream &stream, const Matrix &m)
{
    for (int c = 0; c < m.cols;c++){
            stream << "  - ";
        }
    stream << std::endl;
    for (int r = 0; r < m.rows; r++){
        for (int c = 0; c < m.cols; c++){
            stream << "| " << m(r,c) << " ";
        }
        stream << "|" << std::endl;
        for (int c = 0; c < m.cols;c++){
            stream << "  - ";
        }
        stream << std::endl;
    }
    return stream;
}
}