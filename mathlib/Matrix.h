#pragma once

#include <vector>
#include <functional>
#include <iostream>
#include "MatrixException.h"
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
    Matrix (){}
    Matrix(int rows, int cols);
    Matrix(const Matrix& m);

    // Methods
    const int getRows() const noexcept {return rows;};
    const int getCols() const noexcept {return cols;};
public:
    void setDimensions(int rows, int cols);

    double& operator()(int row, int col);

    double operator()(int row, int col) const;
    double& operator[](int i); // added this so we can more easily generate matrices for testing
    void print() const;
    
    Matrix operator+(Matrix m);

    Matrix operator*(Matrix m); 

    // Matrix product(Matrix m);

    Matrix applyFunc(std::function<double(double)> func);

    Matrix T();

    // Matrix times(double x);

    
};

double addThing(double n);
std::ostream& mathlib::operator<<(std::ostream &stream, const Matrix &m);

}