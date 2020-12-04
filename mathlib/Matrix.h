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
    
    // Constructors / destructor
public:
    Matrix() = default;

    Matrix(int rows, int cols);

    Matrix(int rows, int cols, std::vector<double> vec);

    // Methods 
public:
    int getRows() const noexcept;

    int getCols() const noexcept;

    void setDimensions(int rows, int cols);

    void print() const;

    Matrix applyFunc(std::function<double(double)> func) const;

    Matrix T() const;

    // Operators
public:
    double& operator()(int row, int col);

    double operator()(int row, int col) const;

    double& operator[](int i); // added this so we can more easily generate matrices for testing
    
    Matrix operator+(Matrix m) const;

    Matrix operator*(Matrix m) const; 

    // Matrix times(double x);
    friend std::ostream& operator<<(std::ostream& stream, const Matrix& m);
};

std::ostream& operator<<(std::ostream& stream, const Matrix& m);

}