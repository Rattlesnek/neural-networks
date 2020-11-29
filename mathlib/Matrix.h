#pragma once

#include <vector>

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
    Matrix(int rows, int cols);

    // Methods
public:
    void setDimensions(int rows, int cols);

    double& operator()(int row, int col);

    double operator()(int row, int col) const;

    void print() const;

};

}