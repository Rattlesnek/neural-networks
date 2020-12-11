#pragma once

#include <vector>
#include <functional>
#include <iostream>
#include <memory>

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
    const int getRows() const noexcept;

    const int getCols() const noexcept;

    void setDimensions(int rows, int cols);

    void print() const;

    Matrix applyFunc(std::function<double(double)> func) const;

    Matrix T() const;

    double sum() const;


    // Operators
public:
    double& operator()(int row, int col);

    double operator()(int row, int col) const;

    double& operator[](int i); // added this so we can more easily generate matrices for testing
    
    double operator[](int i) const;

    Matrix operator+(const Matrix& m2) const;

    Matrix operator*(const Matrix& m2) const;

    friend std::ostream& operator<<(std::ostream& stream, const Matrix& m);
};

std::ostream& operator<<(std::ostream& stream, const Matrix& m);

}